"""Check that the CadQuery version of each generated program builds and matches the numpy kernel.

Needs CadQuery:  pip install cadquery   (or: conda install -c conda-forge cadquery)

Usage:
    python scripts/verify_cadquery.py --data data/test --limit 10 --export
For every car it runs program_cq.py with OpenCascade, compares part volumes with the numpy kernel,
prints faces/edges/validity, and (with --export) writes car.step / car.stl next to the program.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carworld import cq_kernel  # noqa: E402
from carworld import sdf_kernel as sk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--grid", type=float, default=15.0, help="voxel size (mm) for the numpy volume")
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    dirs = sorted(p for p in Path(a.data).iterdir() if p.is_dir())[: a.limit]
    worst = 0.0
    for d in dirs:
        res = cq_kernel.run_program((d / "program_cq.py").read_text())
        st = res["state"]
        if not st["valid"] or st["error"]:
            print(f"{d.name}: FAILED  {st['error'] or 'invalid B-rep'}")
            continue
        ops = json.loads((d / "program.json").read_text())
        P, _, _ = sk.make_grid(a.grid)
        model = sk.Model(ops)
        *_, (_, part_d, _) = list(model.iter_eval(P))
        line = [d.name]
        for name, info in st["parts"].items():
            v_sdf = float((part_d[name] < 0).sum() * a.grid ** 3 / 1e6)
            rel = abs(info["volume_l"] - v_sdf) / max(v_sdf, 1e-9)
            worst = max(worst, rel)
            line.append(f"{name}: OCC {info['volume_l']:.0f} L vs numpy {v_sdf:.0f} L ({rel:.1%}), "
                        f"{info['faces']} faces, {info['edges']} edges, valid={info['brep_valid']}")
        print("  ".join(line))
        if a.export:
            for p in cq_kernel.export(res["parts"], d):
                print("   wrote", p)
    print(f"largest volume disagreement: {worst:.1%} (a few % is expected from the voxel grid)")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
