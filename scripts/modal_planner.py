"""Run a planner (run_baseline predictor) on Modal: every test car in its own cloud container, in parallel.

Why: a program-writing run spends most of its time waiting for the language model's replies, so a GPU does not help,
but running cars side by side does (25 cars: ~10 min instead of ~70). It also no longer depends on your Mac staying
awake and online.

One-time setup: store the Meta API key as a Modal secret (it stays in your Modal account; .env stays on your Mac):
    modal secret create meta-api --from-dotenv .env
(or: modal secret create meta-api META_API_KEY=<your key>)

Run (from the project folder):
    modal run scripts/modal_planner.py --predictor vlm_program_est3d --out results_v21_3d \
        --extra "--no-program-example --track-3d"
Options: --data (default data_v21/test), --limit, --start, --parallel (default 10 cars at once),
--no-resume (re-run cars that already have a result). Results land in --out exactly as run_baseline writes them:
<predictor>.jsonl, the round panels, compare/ images and summary.md.
"""
from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
PARALLEL = int(os.environ.get("PLANNER_PARALLEL", "10"))

app = modal.App("carworld-planner")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "pillow")
         .add_local_dir(ROOT / "carworld", remote_path="/root/carworld")
         .add_local_dir(ROOT / "scripts", remote_path="/root/scripts"))
CAR_FILES = ("params.json", "program.json")
VIEW_FILES = ("view_{v}.png", "mask_{v}.png", "camera_{v}.json", "look_{v}.json", "est_state_{v}.json",
              "est_params_{v}.json")


@app.function(image=image, cpu=2.0, memory=4096, timeout=3600, max_containers=PARALLEL,
              secrets=[modal.Secret.from_name(os.environ.get("PLANNER_SECRET", "meta-api"))])
def run_car(name: str, car_id: str, files: dict, argv: list) -> dict:
    sys.path[:0] = ["/root", "/root/scripts"]
    import numpy as np
    import run_baseline as rb
    data, out = Path("/tmp/data"), Path("/tmp/out")
    d = data / car_id
    d.mkdir(parents=True, exist_ok=True)
    for fname, blob in files.items():
        (d / fname).write_bytes(blob)
    a = rb.finalize_args(rb.build_parser().parse_args(
        argv + ["--predictor", name, "--data", str(data), "--out", str(out), "--limit", "1"]))
    s = next(rb.load_samples(data, 0, 1, a.view))
    row = rb.run_one(name, s, np.random.default_rng(a.seed), a, out)
    produced = {str(f.relative_to(out)): f.read_bytes() for f in out.rglob("*.png")}
    return {"row": json.loads(json.dumps(row, default=float)), "files": produced}


@app.local_entrypoint()
def main(predictor: str = "vlm_program_est3d", data: str = "data_v21/test", out: str = "results_v21_3d",
         start: int = 0, limit: int = 25, view: int = 0, extra: str = "", no_resume: bool = False):
    sys.path[:0] = [str(ROOT / "scripts")]
    import run_baseline as rb
    rb.load_dotenv(ROOT / ".env")                 # only the non-secret settings are sent; the key comes from the secret
    argv = shlex.split(extra) + ["--view", str(view), "--api-key-env", "META_API_KEY",
                                 "--base-url", os.environ.get("META_API_BASE", "https://api.meta.ai/v1"),
                                 "--model", os.environ.get("MODEL_API_MODEL", "muse-spark-1.3")]
    if os.environ.get("REASONING_EFFORT") or "api.meta.ai" in os.environ.get("META_API_BASE", "api.meta.ai"):
        argv += ["--reasoning-effort", os.environ.get("REASONING_EFFORT") or "low"]

    out_dir, res_file = Path(out), Path(out) / f"{predictor}.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if res_file.exists() and not no_resume:
        done = {r["id"] for r in map(json.loads, open(res_file)) if r.get("executed")}
    cars = sorted(p for p in Path(data).iterdir() if p.is_dir())[start:start + limit]
    jobs = []
    for d in cars:
        if d.name in done:
            continue
        names = list(CAR_FILES) + [f.format(v=view) for f in VIEW_FILES]
        files = {n: (d / n).read_bytes() for n in names if (d / n).exists()}
        jobs.append((predictor, d.name, files, argv))
    print(f"{predictor}: {len(jobs)} cars to run ({len(done)} already done), up to {PARALLEL} at a time; "
          f"model {os.environ.get('MODEL_API_MODEL')}")
    for i, res in enumerate(run_car.starmap(jobs, order_outputs=False), 1):
        row = res["row"]
        with open(res_file, "a") as f:
            f.write(json.dumps(row) + "\n")
        for rel, blob in res["files"].items():
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / rel).write_bytes(blob)
        msg = (f"IoU {row['silhouette_iou']:.3f}  voxel {row['voxel_iou']:.3f}  chamfer {row['chamfer_mm']:.0f} mm"
               if row.get("executed") else row.get("error", "")[:120])
        print(f"  [{i}/{len(jobs)}] {row['id']}  {msg}  ({row['seconds']:.0f}s)", flush=True)
    print("\n" + rb.summarize(out_dir))
