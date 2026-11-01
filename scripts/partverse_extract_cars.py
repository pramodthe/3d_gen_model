"""Pull the cars out of PartVerse's anno_infos archive and measure them: part boxes, captions, real proportions.

Needs: data/partverse/car_uids.txt (from partverse_find_cars.py), anno_infos.tar.gz (2.8 GB), and trimesh:
    pip install trimesh
    curl -L -o data/partverse/anno_infos.tar.gz \
      https://huggingface.co/datasets/dscdyc/partverse/resolve/main/anno_infos.tar.gz

    python scripts/partverse_extract_cars.py extract   # stream the archive, keep only the car folders (~a few hundred MB)
    python scripts/partverse_extract_cars.py measure   # part boxes + proportions -> data/partverse/cars_parts.jsonl

Each object in anno_infos has <uid>_segmented.glb (the whole mesh) and <uid>_face2label.json (face index -> part id).
The mesh is loaded exactly as PartVerse's own code does (trimesh.load(force='mesh', process=False)) so face order
matches the labels. glTF is Y-up: height = Y; the car's length is the longer of X and Z.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

WHEEL = re.compile(r"wheel|tire|tyre|rim|hubcap", re.I)
NOT_WHEEL = re.compile(r"steering|spare|wheel arch|fender|mudguard|wheel well", re.I)
TOY = re.compile(r"\btoy\b|cartoon|lego|keychain|miniature", re.I)
STYLES = ["sedan", "suv", "hatchback", "coupe", "convertible", "wagon", "pickup", "minivan", "van", "jeep",
          "race car", "sports car", "taxi", "police", "limousine", "vintage", "classic", "off-road", "futuristic",
          "low-poly", "formula", "supercar", "muscle car"]


def extract(args):
    uids = {u.strip() for u in open(args.uids) if u.strip()}
    out = Path(args.cars_dir)
    out.mkdir(parents=True, exist_ok=True)
    found, n = set(), 0
    with tarfile.open(args.archive, mode="r|gz") as tar:  # streaming: never holds the whole archive
        for m in tar:
            n += 1
            if not m.isfile():
                continue
            hit = next((p for p in Path(m.name).parts if p in uids), None)
            if hit is None:
                continue
            dest = out / hit / Path(m.name).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            found.add(hit)
            if len(found) % 50 == 0 and dest.name.endswith(".glb"):
                print(f"  {len(found)} cars extracted ({n} archive entries scanned)", flush=True)
    print(f"done: {len(found)} of {len(uids)} cars found in {n} archive entries -> {out}")
    missing = sorted(uids - found)
    if missing:
        (out / "missing_uids.txt").write_text("\n".join(missing) + "\n")
        print(f"{len(missing)} not in the archive (listed in missing_uids.txt)")


def caption_of(c):
    if isinstance(c, dict):
        return str(c.get("brief_caption", "")), str(c.get("detailed_caption", ""))
    if isinstance(c, (list, tuple)):
        return (str(c[0]) if c else ""), (str(c[1]) if len(c) > 1 else "")
    return str(c), ""


def measure_one(uid, d: Path, caps: dict, trimesh):
    glb = next(d.glob("*_segmented.glb"), None)
    f2l = next(d.glob("*_face2label.json"), None)
    if glb is None or f2l is None:
        return None, "missing segmented.glb or face2label.json"
    mesh = trimesh.load(str(glb), force="mesh", process=False)
    V, F = np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces)
    labels = json.loads(f2l.read_text())
    lab = np.full(len(F), -1)
    for k, v in labels.items():
        if int(k) < len(F):
            lab[int(k)] = int(v)
    lo, hi = V.min(0), V.max(0)
    ext = hi - lo
    length_axis = 0 if ext[0] >= ext[2] else 2     # Y is up in glTF
    width_axis = 2 if length_axis == 0 else 0
    L, W, H = ext[length_axis], ext[width_axis], ext[1]
    car_caps = caps.get(uid, {})
    parts, wheels = [], []
    for pid in sorted(set(lab.tolist()) - {-1}):
        fi = np.where(lab == pid)[0]
        pv = V[np.unique(F[fi].ravel())]
        plo, phi = pv.min(0), pv.max(0)
        b, dcap = caption_of(car_caps.get(str(pid), ""))
        is_wheel = bool(WHEEL.search(b) and not NOT_WHEEL.search(b))
        part = {"id": int(pid), "caption": b, "n_faces": int(len(fi)),
                # box in car-relative units: x along length, y along width, z up; origin at the car's min corner
                "box_min": [round(float((plo[length_axis] - lo[length_axis]) / L), 4),
                            round(float((plo[width_axis] - lo[width_axis]) / L), 4),
                            round(float((plo[1] - lo[1]) / L), 4)],
                "box_max": [round(float((phi[length_axis] - lo[length_axis]) / L), 4),
                            round(float((phi[width_axis] - lo[width_axis]) / L), 4),
                            round(float((phi[1] - lo[1]) / L), 4)],
                "is_wheel": is_wheel, "captioned": bool(b)}
        parts.append(part)
        if is_wheel:
            wheels.append(part)
    rec = {"uid": uid, "n_parts": len(parts), "captioned_parts": sum(p["captioned"] for p in parts),
           "dims_rel": {"length": 1.0, "width": round(float(W / L), 4), "height": round(float(H / L), 4)},
           "toy": bool(TOY.search(" ".join(" ".join(caption_of(c)) for c in car_caps.values()))),
           "style_words": sorted({s for s in STYLES if re.search(r"(?<![a-z])" + re.escape(s),
                                                                  " ".join(" ".join(caption_of(c)) for c in car_caps.values()).lower())}),
           "parts": parts}
    # wheel geometry: radius from each wheel part's height (robust when one part holds a pair of wheels)
    if wheels:
        radii = [(w["box_max"][2] - w["box_min"][2]) / 2 for w in wheels]
        xs = [(w["box_min"][0] + w["box_max"][0]) / 2 for w in wheels]
        rec["wheel_radius_rel"] = round(float(np.median(radii)), 4)
        if len(wheels) >= 2 and max(xs) - min(xs) > 0.2:
            front, rear = max(xs), min(xs)      # which end is the front is unknown; overhangs are reported sorted
            rec["wheelbase_rel"] = round(float(front - rear), 4)
            oh = sorted([round(float(1 - front), 4), round(float(rear), 4)])
            rec["overhangs_rel_sorted"] = oh
        rec["wheels_touch_ground"] = bool(min(w["box_min"][2] for w in wheels) < 0.02)
    return rec, None


def measure(args):
    try:
        import trimesh  # noqa: F401
    except ImportError:
        sys.exit("measure needs trimesh:  pip install trimesh")
    import trimesh
    caps = json.loads(Path(args.captions).read_text())
    cars = sorted(p for p in Path(args.cars_dir).iterdir() if p.is_dir())
    out = Path(args.out)
    ok, errs = 0, defaultdict(int)
    with open(out, "w") as f:
        for i, d in enumerate(cars, 1):
            try:
                rec, err = measure_one(d.name, d, caps, trimesh)
            except Exception as e:  # a broken mesh should not stop the run
                rec, err = None, f"{type(e).__name__}: {e}"
            if rec is None:
                errs[err.split(':')[0]] += 1
                continue
            f.write(json.dumps(rec) + "\n")
            ok += 1
            if i % 50 == 0:
                print(f"  measured {i}/{len(cars)}", flush=True)
    print(f"done: {ok} cars measured -> {out}; skipped: {dict(errs)}")
    summarize(out)


def summarize(path):
    recs = [json.loads(l) for l in open(path)]
    real = [r for r in recs if not r["toy"] and r.get("wheels_touch_ground")]
    print(f"\n{len(recs)} cars, {len(real)} non-toy with wheels on the ground (used for proportions)")

    def pct(vals):
        v = np.array(vals)
        return f"median {np.median(v):.3f}  (10th-90th pct {np.percentile(v, 10):.3f}-{np.percentile(v, 90):.3f}, n={len(v)})"
    for name, get in [("height / length", lambda r: r["dims_rel"]["height"]),
                      ("width / length", lambda r: r["dims_rel"]["width"]),
                      ("wheel radius / length", lambda r: r.get("wheel_radius_rel")),
                      ("wheelbase / length", lambda r: r.get("wheelbase_rel"))]:
        vals = [get(r) for r in real if get(r) is not None]
        if vals:
            print(f"  {name:22s} {pct(vals)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "measure", "summary"])
    ap.add_argument("--archive", default="data/partverse/anno_infos.tar.gz")
    ap.add_argument("--uids", default="data/partverse/car_uids.txt")
    ap.add_argument("--captions", default="data/partverse/text_captions.json")
    ap.add_argument("--cars-dir", default="data/partverse/cars")
    ap.add_argument("--out", default="data/partverse/cars_parts.jsonl")
    a = ap.parse_args()
    {"extract": extract, "measure": measure, "summary": lambda a: summarize(a.out)}[a.step](a)


if __name__ == "__main__":
    main()
