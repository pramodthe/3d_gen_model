"""Build the CAD world-model benchmark: (state, operation) -> next state, with ground truth from the kernel.

Cars are freshly sampled from the generator with their own seeds (never used for training data). Every sample is one modelling step:

  normal        a real step of the car's build program
  scaled        that step with its tool resized (radius / extent / outline x0.6-1.5): same kind of operation,
                different numbers, so the answer cannot be recalled from the template
  no_effect     a cut placed entirely outside the car: nothing changes
  wipe_body     a cut that swallows the whole body: the body vanishes (invalid if nothing else is left)
  missing_part  an edit of the "wheels" part before it exists: rejected, state unchanged, valid=false
  bad_tool      a tool with a reversed extent: rejected, state unchanged, valid=false

Usage:
    python scripts/build_wm_bench.py --cars 20 --normal-per-car 6 --out data_v21/wm_bench/bench.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carworld import params as cp  # noqa: E402
from carworld import program as pg  # noqa: E402
from carworld import sdf_kernel as sk  # noqa: E402

BENCH_SEED = 777_000


def scale_tool(tool: dict, f: float) -> dict:
    t = copy.deepcopy(tool)
    t.pop("_np", None)          # drop the kernel's cached outline, or the old size would be evaluated
    if t["type"] == "intersect":
        t["children"] = [scale_tool(c, f) for c in t["children"]]
        return t
    if t["type"] == "cylinder":
        t["radius"] = round(t["radius"] * f, 2)
    else:  # scale the outline about its centroid
        pts = np.array(t["points"], float)
        c = pts.mean(0)
        t["points"] = [[round(float(x), 2), round(float(y), 2)] for x, y in (c + (pts - c) * f)]
    return t


def make_samples(car_idx: int, rng, normal_per_car: int, res: float):
    p = cp.sample(rng)
    ops = pg.build_program(p)
    traj = sk.trajectory(ops, res=res)
    cid = f"wm_{car_idx:04d}"
    out = []

    def rec(kind, i, action, state, nxt):
        out.append({"id": f"{cid}_{kind}_{i}", "car": cid, "style": p.style, "kind": kind, "step_index": i,
                    "state": state, "action": sk.public_op(action), "next_state": nxt})

    # normal steps (skip step 0: from-nothing is trivial), spread over the program
    idx = sorted(rng.choice(np.arange(1, len(ops)), size=min(normal_per_car, len(ops) - 1), replace=False))
    for i in idx:
        rec("normal", int(i), traj[i]["action"], traj[i]["state"], traj[i]["next_state"])

    first_wheel = next(i for i, o in enumerate(ops) if o["part"] == "wheels")
    body_steps = list(range(3, first_wheel))

    # scaled: resize the tool of a real step that changes the volume by more than 1 % (arches, glass, outlines...)
    def vol(st):
        return sum((p.get("volume_l") or 0.0) for p in st["parts"].values())
    big = [j for j in range(1, len(ops)) if abs(vol(traj[j]["next_state"]) - vol(traj[j]["state"]))
           > 0.01 * max(vol(traj[j]["state"]), 1.0)]
    i = int(rng.choice(big))
    op = copy.deepcopy(ops[i])
    op["tool"] = scale_tool(op["tool"], float(rng.choice([0.6, 0.75, 1.3, 1.5])))
    rec("scaled", i, op, traj[i]["state"], sk.apply_step(ops[:i], op, traj[i]["state"], res))

    # no_effect: a cut box beyond the car's front bumper
    i = int(rng.choice(body_steps))
    xf = p.length / 2
    op = dict(step=i + 1, name="test_cut", part="body", mode="cut", mirror=False, material=None, comment="Cut a pocket",
              tool=dict(type="prism", plane="XZ", points=[[xf + 150, 300], [xf + 400, 300], [xf + 400, 700],
                                                          [xf + 150, 700]], extent=[-300.0, 300.0]))
    rec("no_effect", i, op, traj[i]["state"], sk.apply_step(ops[:i], op, traj[i]["state"], res))

    # wipe_body: a cut that contains the whole body (sometimes after the wheels exist, so the result stays valid)
    i = int(rng.choice([*body_steps, len(ops)]))      # len(ops) = after the whole program, wheels included
    state_i = traj[i]["state"] if i < len(ops) else traj[-1]["next_state"]
    op = dict(step=i + 1, name="clear_body", part="body", mode="cut", mirror=False, material=None,
              comment="Cut away the body",
              tool=dict(type="prism", plane="XY", points=[[-2700, -1100], [2700, -1100], [2700, 1100], [-2700, 1100]],
                        extent=[-50.0, 2050.0]))
    rec("wipe_body", i, op, state_i, sk.apply_step(ops[:i], op, state_i, res))

    # missing_part: cut the wheels before they exist
    i = int(rng.choice(body_steps))
    op = copy.deepcopy(ops[first_wheel])
    op.update(step=i + 1, mode="cut", name="trim_wheels", comment="Trim the wheels")
    rec("missing_part", i, op, traj[i]["state"], sk.apply_step(ops[:i], op, traj[i]["state"], res))

    # bad_tool: reversed extent on a real step
    i = int(rng.choice(body_steps))
    op = copy.deepcopy(ops[i])

    def reverse(t):
        if t["type"] == "intersect":
            reverse(t["children"][0])
        else:
            t["extent"] = [t["extent"][1], t["extent"][0]]
    reverse(op["tool"])
    rec("bad_tool", i, op, traj[i]["state"], sk.apply_step(ops[:i], op, traj[i]["state"], res))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", type=int, default=20)
    ap.add_argument("--normal-per-car", type=int, default=6)
    ap.add_argument("--grid", type=float, default=40.0, help="voxel size, same as the training data")
    ap.add_argument("--out", default="data_v21/wm_bench/bench.jsonl")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    kinds = Counter()
    with open(out, "w") as f:
        for c in range(a.cars):
            rng = np.random.default_rng(BENCH_SEED + c)
            for s in make_samples(c, rng, a.normal_per_car, a.grid):
                f.write(json.dumps(s) + "\n")
                kinds[s["kind"]] += 1
            print(f"  car {c + 1}/{a.cars}", flush=True)
    print(f"wrote {sum(kinds.values())} samples to {out}: {dict(kinds)}")


if __name__ == "__main__":
    main()
