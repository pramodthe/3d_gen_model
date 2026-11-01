"""Train the 3D state estimator on Modal (cloud GPU) and run it on our test photos.

One-time setup on your Mac (in the project's .venv):
    pip install modal
    modal setup                      # opens a browser to log in; stores a token in ~/.modal.toml

Run (from the project folder):
    modal run scripts/modal_state_estimator.py --smoke           # ~3 min, cents: checks every step end to end
    modal run scripts/modal_state_estimator.py                   # full: generate, train, predict (~30-40 min)
    modal run scripts/modal_state_estimator.py --step predict    # only re-run predictions with the saved model

Steps
    gen      cars are generated and rendered on many cloud CPUs in parallel (state_target.make_examples) and stored
             in the Modal volume "carworld-state"; finished chunks are skipped on a re-run
    train    one GPU trains the estimator (carworld/estimator.py); checkpoint + report saved to the volume
    predict  the estimator reads every photo in --data (default data_v21/test) on your Mac; for each car it writes
             est_state_0.json (estimated state summary) and est_params_0.json (estimated parameters + style),
             and prints the error against the true state
Then use the estimate in the planner:
    python scripts/run_baseline.py --data data_v21/test --limit 25 --predictor est_params
    python scripts/run_baseline.py --data data_v21/test --limit 25 --no-program-example --track-3d \
        --out results_v21_3d --predictor vlm_program_est3d --resume
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "carworld-state-estimator"
VOL_PATH = "/vol"
DATA_ROOT = f"{VOL_PATH}/data_v21"         # generator v2.1; one sub-folder per data tag ("main", "smoke")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("carworld-state", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "pillow", "torch", "torchvision")
         # bake the ImageNet weights into the image so training never waits on a download
         .run_commands("python -c \"import torchvision; torchvision.models.resnet34(weights='DEFAULT')\"")
         .add_local_dir(ROOT / "carworld", remote_path="/root/carworld"))


@app.function(image=image, cpu=1.0, memory=3072, timeout=3600, volumes={VOL_PATH: volume}, retries=1)
def gen_chunk(tag: str, split: str, start: int, n: int, views: int) -> str:
    import numpy as np
    from carworld import state_target as stt
    out = Path(DATA_ROOT) / tag / split / f"chunk_{start:06d}_{n}.npz"
    if out.exists():
        return f"{out.name} (exists)"
    out.parent.mkdir(parents=True, exist_ok=True)
    d = stt.make_examples(split, start, n, views=views)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **d)
    tmp.rename(out)
    volume.commit()
    return out.name


def _load_split(tag: str, split: str, limit_cars: int | None = None) -> dict:
    import numpy as np
    files = sorted((Path(DATA_ROOT) / tag / split).glob("chunk_*.npz"))
    if not files:
        raise FileNotFoundError(f"no data in {DATA_ROOT}/{tag}/{split}: run the gen step first")
    parts = [dict(np.load(f)) for f in files]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if limit_cars is not None:
        keep = d["ids"] < limit_cars
        d = {k: v[keep] for k, v in d.items()}
    return d


@app.function(image=image, gpu="A10G", cpu=8.0, memory=49152, timeout=4 * 3600, volumes={VOL_PATH: volume})
def train(tag: str, run: str, n_train: int, epochs: int, batch: int, lr: float, backbone: str) -> dict:
    import torch
    from carworld import estimator as est
    volume.reload()
    data, val = _load_split(tag, "est_train", n_train), _load_split(tag, "est_val")
    print(f"training on {len(data['imgs'])} images ({len(set(data['ids'].tolist()))} cars), "
          f"validating on {len(val['imgs'])}; device {est.pick_device()}", flush=True)
    ckpt, rep = est.train(data, val, epochs=epochs, batch=batch, lr=lr, backbone=backbone,
                          log=lambda s: print(s, flush=True))
    ck_dir = Path(VOL_PATH) / "ckpt"
    ck_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, ck_dir / f"{run}.pt")
    (ck_dir / f"{run}_val.json").write_text(json.dumps(rep, indent=1))
    volume.commit()
    return rep


@app.function(image=image, cpu=4.0, memory=8192, timeout=1800, volumes={VOL_PATH: volume})
def predict(run: str, items: list[dict]) -> list[dict]:
    """items: {id, img (raw uint8 bytes), shape [H, W, 3], camera (dict), program (list of ops or None)}."""
    import numpy as np
    import torch
    from carworld import estimator as est
    from carworld import state_target as stt
    volume.reload()
    ckpt = torch.load(Path(VOL_PATH) / "ckpt" / f"{run}.pt", map_location="cpu", weights_only=False)
    model, s_norm, p_norm, dev = est.load(ckpt, torch.device("cpu"))
    imgs = np.stack([np.frombuffer(it["img"], np.uint8).reshape(it["shape"]) for it in items])
    cams = np.stack([stt.camera_vec(it["camera"]) for it in items])
    S, P, Y = est.predict_arrays(model, imgs, cams, s_norm, p_norm, dev)
    out = []
    for it, s, p, y in zip(items, S, P, Y):
        row = {"id": it["id"], "est_state": stt.decode_state(s), "est_params": {
            **stt.params_from_vec(p, int(y.argmax())).to_dict(),
            "style_probs": {k: round(float(v), 3) for k, v in zip(stt.STYLES, y)}}}
        if it.get("program"):
            true = stt.encode_state(stt.state_of(it["program"]))
            row["abs_err"] = {k: round(float(abs(a - b)), 2) for k, a, b in zip(stt.STATE_NAMES, s, true)}
        out.append(row)
    return out


@app.local_entrypoint()
def main(step: str = "all", run: str = "", n_train: int = 10000, n_val: int = 400, views: int = 2,
         chunk: int = 40, epochs: int = 20, batch: int = 96, lr: float = 5e-4, backbone: str = "resnet34",
         data: str = "data_v21/test", view: int = 0, smoke: bool = False):
    tag = "main"
    if smoke:
        n_train, n_val, chunk, epochs, run, tag = 40, 20, 20, 1, run or "smoke", "smoke"
    run = run or f"{backbone}_{n_train}x{views}"
    steps = ["gen", "train", "predict"] if step == "all" else [step]

    if "gen" in steps:
        jobs = [(tag, "est_train", s, min(chunk, n_train - s), views) for s in range(0, n_train, chunk)]
        jobs += [(tag, "est_val", s, min(chunk, n_val - s), 1) for s in range(0, n_val, chunk)]
        print(f"generating {n_train} training cars x {views} views and {n_val} validation cars "
              f"in {len(jobs)} parallel chunks ...")
        for i, name in enumerate(gen_chunk.starmap(jobs, order_outputs=False), 1):
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} chunks done ({name})", flush=True)

    if "train" in steps:
        rep = train.remote(tag, run, n_train, epochs, batch, lr, backbone)
        print("\nvalidation (best epoch): body size error "
              f"{rep['body_size_mae_mm']:.0f} mm, wheels size error {rep['wheels_size_mae_mm']:.0f} mm, "
              f"style accuracy {rep['style_acc']:.0%}")

    if "predict" in steps:
        import numpy as np
        from PIL import Image
        root = Path(data)
        cars = sorted(p for p in root.iterdir() if (p / f"view_{view}.png").exists())
        items = []
        for d in cars:
            img = np.asarray(Image.open(d / f"view_{view}.png").convert("RGB"), np.uint8)
            items.append({"id": d.name, "img": img.tobytes(), "shape": list(img.shape),
                          "camera": json.loads((d / f"camera_{view}.json").read_text()),
                          "program": json.loads((d / "program.json").read_text())
                          if (d / "program.json").exists() else None})
        print(f"\npredicting {len(items)} photos from {root} with model '{run}' ...")
        rows = predict.remote(run, items)
        for r in rows:
            (root / r["id"] / f"est_state_{view}.json").write_text(json.dumps(r["est_state"], indent=1))
            (root / r["id"] / f"est_params_{view}.json").write_text(json.dumps(r["est_params"], indent=1))
        errs = [r["abs_err"] for r in rows if "abs_err" in r]
        if errs:
            keys = ["body.length", "body.width", "body.height", "body.x0", "body.z0",
                    "wheels.length", "wheels.width", "wheels.height", "wheels.x0"]
            print("mean absolute error on these photos (mm):")
            for k in keys:
                print(f"  {k:15s} {np.mean([e[k] for e in errs]):6.0f}")
            print(f"  body volume     {np.mean([e['body.volume_l'] for e in errs]):6.0f} L")
        print(f"wrote est_state_{view}.json and est_params_{view}.json into {len(rows)} car folders")
