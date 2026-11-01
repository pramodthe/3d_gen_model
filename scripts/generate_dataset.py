"""Step 1: generate a dataset of procedural cars.

For every car it writes
    params.json         the ~19 design parameters (ground truth for parameter prediction)
    program.json        the CAD program as JSON operations (ground truth for the planner)
    program_cq.py       the same program as a CadQuery script
    trajectory.jsonl    (state, action, next_state) for every operation (world-model training data)
    view_k.png          rendered photo-like images (random camera, paint, sky, ground, sun)
    mask_k.png, depth_k.npz (key 'depth', mm), camera_k.json, look_k.json
and two training files at the split root:
    planner_sft.jsonl   image -> program pairs in chat format (Stage 1 SFT)
    world_model.jsonl   (state, action) -> next_state pairs in chat format, with a loss weight per turn
                        (information-weighted loss: turns that barely change the state count less)

Usage:
    python scripts/generate_dataset.py --out data --split train --n 200 --views 3 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carworld import params as cp  # noqa: E402
from carworld import program as pg  # noqa: E402
from carworld import render as rd  # noqa: E402
from carworld import sdf_kernel as sk  # noqa: E402
from carworld.prompts import WORLD_MODEL_SYSTEM, planner_messages, world_model_user  # noqa: E402

SPLIT_SEED = {"train": 1000, "val": 2000, "test": 3000}


def informative_weight(rec: dict) -> float:
    """Loss weight for one world-model turn: down-weight turns that barely change the state.

    The world model should spend its capacity on turns that carry real information about the kernel.
    """
    s, n = rec["state"], rec["next_state"]
    if not n["valid"] or n.get("error"):
        return 1.0                                      # failures are the most informative turns
    if set(s.get("parts", {})) != set(n.get("parts", {})):
        return 1.0                                      # a new part appeared
    for name, info in n["parts"].items():
        before = s["parts"].get(name, {})
        v0, v1 = before.get("volume_l", 0.0), info.get("volume_l", 0.0)
        if abs(v1 - v0) > 0.002 * max(v0, 1.0):
            return 1.0
        if before.get("bbox_mm") != info.get("bbox_mm"):
            return 1.0
        if before.get("surface_materials") != info.get("surface_materials"):
            return 0.5
    return 0.1                                          # near no-op: mostly copying the state


def make_car(args):
    split, idx, out_root, views, width, height, res = args
    rng = np.random.default_rng(SPLIT_SEED[split] * 100000 + idx)
    p = cp.sample(rng)
    ops = pg.build_program(p)
    model = sk.Model(ops)
    cid = f"{split}_{idx:06d}"
    d = Path(out_root) / split / cid
    d.mkdir(parents=True, exist_ok=True)
    (d / "params.json").write_text(p.to_json())
    pub = [sk.public_op(o) for o in ops]
    (d / "program.json").write_text(json.dumps(pub, indent=1))
    (d / "program_cq.py").write_text(pg.to_cadquery(ops, p))
    traj = sk.trajectory(ops, res=res)
    with open(d / "trajectory.jsonl", "w") as f:
        for r in traj:
            r["weight"] = informative_weight(r)
            f.write(json.dumps(r) + "\n")
    for k in range(views):
        cam = rd.random_camera(rng, p, width, height)
        look = rd.random_look(rng)
        r = rd.render(model, cam, look)
        Image.fromarray(r["rgb"]).save(d / f"view_{k}.png")
        Image.fromarray((r["mask"] * 255).astype(np.uint8)).save(d / f"mask_{k}.png")
        np.savez_compressed(d / f"depth_{k}.npz", depth=r["depth"].astype(np.float16))
        (d / f"camera_{k}.json").write_text(json.dumps(cam.to_dict()))
        (d / f"look_{k}.json").write_text(json.dumps(look.to_dict()))
    return {"id": cid, "style": p.style, "views": views, "n_ops": len(ops),
            "length": round(p.length), "weights": [r["weight"] for r in traj]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--split", default="train", choices=list(SPLIT_SEED))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--grid", type=float, default=40.0, help="voxel size (mm) for state summaries")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    root = Path(a.out)
    (root / a.split).mkdir(parents=True, exist_ok=True)
    jobs = [(a.split, i, str(root), a.views, a.width, a.height, a.grid) for i in range(a.start, a.start + a.n)]
    t0 = time.time()
    rows = []
    with Pool(a.workers) as pool:
        for i, row in enumerate(pool.imap(make_car, jobs), 1):
            rows.append(row)
            if i % 10 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} cars  ({time.time() - t0:.0f}s)", flush=True)

    # index + training files
    with open(root / a.split / "index.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "style", "views", "n_ops", "length")}) + "\n")
    with open(root / a.split / "planner_sft.jsonl", "a") as f_pl, open(root / a.split / "world_model.jsonl", "a") as f_wm:
        for r in rows:
            d = root / a.split / r["id"]
            program = json.loads((d / "program.json").read_text())
            for k in range(r["views"]):
                f_pl.write(json.dumps({"id": r["id"], "image": f"{r['id']}/view_{k}.png",
                                       "messages": planner_messages(program)}) + "\n")
            for line in open(d / "trajectory.jsonl"):
                rec = json.loads(line)
                f_wm.write(json.dumps({"id": r["id"], "weight": rec["weight"], "messages": [
                    {"role": "system", "content": WORLD_MODEL_SYSTEM},
                    {"role": "user", "content": world_model_user(rec["state"], rec["action"])},
                    {"role": "assistant", "content": json.dumps(rec["next_state"])}]}) + "\n")
    w = np.concatenate([r["weights"] for r in rows])
    print(f"done: {len(rows)} cars, {len(w)} world-model turns "
          f"(full weight {np.mean(w == 1.0):.0%}, partial {np.mean(w == 0.5):.0%}, near no-op {np.mean(w == 0.1):.0%})")


if __name__ == "__main__":
    main()
