"""3D state estimator: photo + camera -> the car's state summary (plus generator parameters and body style).

A pretrained ResNet reads the photo; its features are joined with the camera vector (the planner is given the exact
camera too) and an MLP regresses the normalised targets. Runs on CUDA, Apple MPS or CPU.

Training data comes from carworld.state_target.make_examples (stacked arrays). Used by
scripts/modal_state_estimator.py (cloud GPU) and scripts/train_state_estimator.py (local).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import state_target as stt

MEAN_RGB = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD_RGB = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
IMG_HW = (240, 320)
# which camera features flip sign under a horizontal image flip (sin of azimuth)
CAM_FLIP = torch.tensor([-1.0, 1, 1, 1, 1, 1])


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class StateEstimator(nn.Module):
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True):
        super().__init__()
        import torchvision
        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models, backbone)(weights=weights)
        feat = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.cam = nn.Sequential(nn.Linear(stt.CAM_DIM, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(feat + 64, 512), nn.ReLU(), nn.Dropout(0.1),
                                   nn.Linear(512, 512), nn.ReLU())
        self.state_head = nn.Linear(512, stt.STATE_DIM)
        self.param_head = nn.Linear(512, len(stt.PARAM_KEYS))
        self.style_head = nn.Linear(512, len(stt.STYLES))

    def forward(self, img, cam):
        h = self.trunk(torch.cat([self.backbone(img), self.cam(cam)], 1))
        return self.state_head(h), self.param_head(h), self.style_head(h)


def to_input(imgs_uint8: torch.Tensor) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> normalised (N, 3, H, W) float."""
    x = imgs_uint8.permute(0, 3, 1, 2).float() / 255.0
    if tuple(x.shape[-2:]) != IMG_HW:
        x = F.interpolate(x, size=IMG_HW, mode="bilinear", align_corners=False)
    return (x - MEAN_RGB.to(x.device)) / STD_RGB.to(x.device)


def augment(x: torch.Tensor, cam: torch.Tensor):
    """Horizontal flip (the car is mirror-symmetric, so the targets are unchanged; the azimuth changes sign) and
    mild brightness / contrast jitter."""
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
    cam = torch.where(flip.view(-1, 1), cam * CAM_FLIP.to(cam.device), cam)
    b = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * 0.4
    c = 1 + (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * 0.3
    return x * c + b, cam


class Normaliser:
    def __init__(self, mean, std):
        self.mean, self.std = np.asarray(mean, np.float32), np.maximum(np.asarray(std, np.float32), 1e-3)

    @classmethod
    def fit(cls, a):
        return cls(a.mean(0), a.std(0))

    def fwd(self, a):
        return (a - self.mean) / self.std

    def inv(self, a):
        return a * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def train(data: dict, val: dict, epochs: int = 15, batch: int = 96, lr: float = 5e-4, backbone: str = "resnet34",
          pretrained: bool = True, log=print) -> tuple[dict, dict]:
    """Train on stacked arrays (see state_target.make_examples). Returns (checkpoint, val report)."""
    dev = pick_device()
    torch.manual_seed(0)
    s_norm, p_norm = Normaliser.fit(data["states"]), Normaliser.fit(data["params"])
    model = StateEstimator(backbone, pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(data["imgs"])
    steps = epochs * math.ceil(n / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    imgs = torch.from_numpy(data["imgs"])                       # stays uint8 on the CPU; batches go to the device
    cams = torch.from_numpy(data["cams"])
    st = torch.from_numpy(s_norm.fwd(data["states"]))
    pr = torch.from_numpy(p_norm.fwd(data["params"]))
    sy = torch.from_numpy(data["styles"])
    best, best_state = None, None
    for ep in range(epochs):
        model.train()
        perm, t0, tot = torch.randperm(n), time.time(), 0.0
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            if len(j) < 2:                                       # BatchNorm needs at least two samples
                continue
            x = to_input(imgs[j].to(dev, non_blocking=True))
            c = cams[j].to(dev)
            x, c = augment(x, c)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                ps, pp, py = model(x, c)
                loss = (F.smooth_l1_loss(ps.float(), st[j].to(dev), beta=0.5)
                        + 0.5 * F.smooth_l1_loss(pp.float(), pr[j].to(dev), beta=0.5)
                        + 0.2 * F.cross_entropy(py.float(), sy[j].to(dev)))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.detach().item() * len(j)
        rep = evaluate(model, val, s_norm, p_norm, dev)
        log(f"epoch {ep + 1}/{epochs}  train loss {tot / n:.4f}  val: body size err {rep['body_size_mae_mm']:.0f} mm, "
            f"wheels size err {rep['wheels_size_mae_mm']:.0f} mm, style acc {rep['style_acc']:.0%}  "
            f"({time.time() - t0:.0f}s)")
        if best is None or rep["score"] < best["score"]:
            best = rep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ckpt = {"model": best_state, "backbone": backbone, "state_norm": s_norm.to_dict(), "param_norm": p_norm.to_dict(),
            "state_names": stt.STATE_NAMES, "param_keys": stt.PARAM_KEYS, "styles": stt.STYLES, "img_hw": IMG_HW}
    return ckpt, best


@torch.no_grad()
def predict_arrays(model, imgs_uint8: np.ndarray, cams: np.ndarray, s_norm, p_norm, dev, batch: int = 128,
                   tta: bool = True):
    """Returns (states, params, style_probs) in real units. tta: average with the mirrored image."""
    model.eval()
    out_s, out_p, out_y = [], [], []
    for i in range(0, len(imgs_uint8), batch):
        x = to_input(torch.from_numpy(imgs_uint8[i:i + batch]).to(dev))
        c = torch.from_numpy(cams[i:i + batch]).to(dev)
        ps, pp, py = (t.float() for t in model(x, c))
        if tta:
            fs, fp, fy = (t.float() for t in model(x.flip(-1), c * CAM_FLIP.to(dev)))
            ps, pp, py = (ps + fs) / 2, (pp + fp) / 2, (py + fy) / 2
        out_s.append(ps.cpu().numpy()); out_p.append(pp.cpu().numpy()); out_y.append(py.softmax(-1).cpu().numpy())
    return s_norm.inv(np.concatenate(out_s)), p_norm.inv(np.concatenate(out_p)), np.concatenate(out_y)


def evaluate(model, val: dict, s_norm, p_norm, dev) -> dict:
    s, p, y = predict_arrays(model, val["imgs"], val["cams"], s_norm, p_norm, dev, tta=False)
    err = np.abs(s - val["states"])
    names = stt.STATE_NAMES
    col = {k: names.index(k) for k in names}
    body = [col[f"body.{k}"] for k in ("length", "width", "height")]
    wheels = [col[f"wheels.{k}"] for k in ("length", "width", "height")]
    rep = {"n": int(len(s)),
           "body_size_mae_mm": float(err[:, body].mean()), "wheels_size_mae_mm": float(err[:, wheels].mean()),
           "per_target_mae": {k: round(float(err[:, i].mean()), 3) for k, i in col.items()},
           "param_mae": {k: round(float(v), 4) for k, v in zip(stt.PARAM_KEYS, np.abs(p - val["params"]).mean(0))},
           "style_acc": float((y.argmax(1) == val["styles"]).mean())}
    # one number for model selection: mean normalised error over the state targets
    rep["score"] = float((err / s_norm.std).mean())
    return rep


def load(ckpt: dict, device=None):
    dev = device or pick_device()
    model = StateEstimator(ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()
    return (model, Normaliser(**ckpt["state_norm"]), Normaliser(**ckpt["param_norm"]), dev)
