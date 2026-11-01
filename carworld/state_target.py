"""Targets for the 3D state estimator: photo (+ known camera) -> the car's state summary.

The state summary is the same one the CAD kernel reports and the world model predicts (per part: size, position,
volume, surface materials). The estimator outputs it as a fixed-length vector; decode_state() turns that vector back
into the dict format, so a planner can use an estimated state exactly where the upper-bound experiment used the true
one. Generator parameters and body style are predicted as auxiliary targets.

Everything here is numpy only, so data can be generated on any CPU (laptop or cloud workers).
"""
from __future__ import annotations

import json
import math

import numpy as np

from . import params as cp
from . import program as pg
from . import render as rd
from . import sdf_kernel as sk

STATE_RES = 40.0
PARTS = ["body", "wheels"]
PART_MATERIALS = {"body": ["paint", "glass", "arch", "trim", "grille", "light", "taillight"],
                  "wheels": ["tire", "rim"]}
PARAM_KEYS = ["length", "width", "height", "wheelbase_ratio", "front_overhang_ratio", "wheel_radius", "tire_width",
              "ground_clearance", "belt_ratio", "nose_ratio", "cowl_ratio", "windshield_rake", "roof_ratio",
              "rear_rake", "tumblehome", "nose_taper", "tail_taper", "arch_gap", "spoiler_height"]
STYLES = list(cp.STYLES)

# target vector layout: per part [L, W, H, x0, z0, volume] + its material shares
# (y0 is not stored: every part is mirrored, so y0 = -W/2)
STATE_NAMES = []
for _p in PARTS:
    STATE_NAMES += [f"{_p}.length", f"{_p}.width", f"{_p}.height", f"{_p}.x0", f"{_p}.z0", f"{_p}.volume_l"]
    STATE_NAMES += [f"{_p}.mat.{m}" for m in PART_MATERIALS[_p]]
STATE_DIM = len(STATE_NAMES)
CAM_DIM = 6
SEED_BASE = {"est_train": 500_000_000, "est_val": 600_000_000}   # disjoint from generate_dataset's splits


def state_of(ops: list[dict], res: float = STATE_RES) -> dict:
    """Kernel state summary after the whole program, on a grid centred on the car's midline."""
    P, _, res = sk.make_grid(res, centered=True)
    ops = json.loads(json.dumps([sk.public_op(o) for o in ops]))
    *_, (i, part_d, part_m) = sk.Model(ops).iter_eval(P)
    return sk._summary(ops[i], part_d, part_m, P, res)


def encode_state(state: dict) -> np.ndarray:
    v = []
    for p in PARTS:
        info = state["parts"].get(p) or {}
        if info.get("bbox_mm"):
            (L, W, H), (x0, _, z0) = info["bbox_mm"], info["min_mm"]
        else:
            L = W = H = x0 = z0 = 0.0
        v += [L, W, H, x0, z0, info.get("volume_l", 0.0)]
        sm = info.get("surface_materials") or {}
        v += [sm.get(m, 0.0) for m in PART_MATERIALS[p]]
    return np.asarray(v, np.float32)


def decode_state(vec) -> dict:
    """Vector -> state summary dict (same format as the kernel's), with sane clipping."""
    vec = [float(x) for x in vec]
    out, k = {"valid": True, "error": None, "parts": {}}, 0
    for p in PARTS:
        L, W, H, x0, z0, vol = vec[k:k + 6]
        k += 6
        mats = np.clip(vec[k:k + len(PART_MATERIALS[p])], 0, None)
        k += len(PART_MATERIALS[p])
        mats = mats / max(mats.sum(), 1e-6)
        L, W, H = (max(0.0, round(x)) for x in (L, W, H))
        out["parts"][p] = {"volume_l": round(max(vol, 0.0), 1), "bbox_mm": [int(L), int(W), int(H)],
                           "min_mm": [int(round(x0)), int(round(-W / 2)), int(round(z0))],
                           "surface_materials": {m: round(float(s), 3) for m, s in zip(PART_MATERIALS[p], mats)
                                                 if s >= 0.005},
                           "left_right_symmetric": True}
    return out


def camera_vec(cam) -> np.ndarray:
    """Camera features the planner is also given: view direction, elevation, distance, field of view, target height."""
    if isinstance(cam, dict):
        cam = rd.Camera(**cam)
    a = math.radians(cam.azimuth)
    return np.asarray([math.sin(a), math.cos(a), cam.elevation / 20.0, cam.distance / 8000.0, cam.fov / 32.0,
                       cam.target_z / 650.0], np.float32)


def params_vec(p: cp.CarParams) -> np.ndarray:
    return np.asarray([getattr(p, k) for k in PARAM_KEYS], np.float32)


def params_from_vec(vec, style_idx: int) -> cp.CarParams:
    d = {k: float(v) for k, v in zip(PARAM_KEYS, vec)}
    d["style"] = STYLES[int(style_idx)]
    p = cp.CarParams.from_dict(d)
    return p


def make_examples(split: str, start: int, n: int, views: int = 2, width: int = 320, height: int = 240) -> dict:
    """Generate cars [start, start + n) of a split: same sampler, camera and appearance randomisation as the
    test photos, the true state at STATE_RES, the parameters and the style. Returns stacked arrays."""
    imgs, cams, states, prms, styles, ids = [], [], [], [], [], []
    for idx in range(start, start + n):
        rng = np.random.default_rng(SEED_BASE[split] + idx)
        p = cp.sample(rng)
        ops = pg.build_program(p)
        model = sk.Model(ops)
        st = encode_state(state_of(ops))
        for _ in range(views):
            cam = rd.random_camera(rng, p, width, height)
            look = rd.random_look(rng)
            imgs.append(rd.render(model, cam, look)["rgb"])
            cams.append(camera_vec(cam))
            states.append(st)
            prms.append(params_vec(p))
            styles.append(STYLES.index(p.style))
            ids.append(idx)
    return {"imgs": np.stack(imgs).astype(np.uint8), "cams": np.stack(cams), "states": np.stack(states),
            "params": np.stack(prms), "styles": np.asarray(styles, np.int64), "ids": np.asarray(ids, np.int64)}
