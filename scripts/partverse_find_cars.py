"""Find the cars in PartVerse (https://huggingface.co/datasets/dscdyc/partverse) from its part captions.

PartVerse: 12k Objaverse objects split into 91k human-verified parts, each with a VLM caption
({"brief_caption", "detailed_caption"}). Captions name the object they belong to ("...the front wheel of the car"),
so cars can be found without downloading any meshes.

Step 1 - download only the captions (46 MB):
    mkdir -p data/partverse
    curl -L -o data/partverse/text_captions.json \
      https://huggingface.co/datasets/dscdyc/partverse/resolve/main/text_captions.json
Step 2 - run:
    python scripts/partverse_find_cars.py --captions data/partverse/text_captions.json --out data/partverse
Writes cars.json (one entry per likely car, best first) and prints counts and examples.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# object words -> coarse type; checked against every caption of an object
OBJECT_WORDS = {
    "car": "car", "cars": "car", "automobile": "car", "sedan": "car", "saloon": "car", "hatchback": "car",
    "coupe": "car", "coupé": "car", "convertible": "car", "suv": "car", "crossover": "car", "jeep": "car",
    "wagon": "car", "estate car": "car", "station wagon": "car", "sports car": "car", "race car": "car",
    "racecar": "car", "taxi": "car", "police car": "car", "minivan": "car", "limousine": "car",
    "truck": "truck", "pickup": "truck", "lorry": "truck", "van": "truck", "bus": "bus", "tractor": "other_vehicle",
    "forklift": "other_vehicle", "tank": "other_vehicle", "vehicle": "vehicle",
}
# parts that cars have; an object with several of them is very likely a road vehicle
CAR_PARTS = ["wheel", "tire", "tyre", "rim", "hubcap", "windshield", "windscreen", "headlight", "headlamp",
             "taillight", "tail light", "bumper", "hood", "bonnet", "trunk", "boot lid", "door", "side mirror",
             "rear-view mirror", "rearview mirror", "grille", "fender", "license plate", "roof", "chassis",
             "body shell", "car body", "exhaust", "spoiler", "steering wheel", "seat"]
TOY_WORDS = ["toy", "lego", "cartoon", "model car", "miniature", "plastic toy", "keychain"]


def words_in(text: str, vocab) -> list[str]:
    t = text.lower()
    return [w for w in vocab if re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", t)]


def caption_text(c) -> str:
    """Captions come as [brief, detailed] lists in the released file (dicts in the authors' caption code)."""
    if isinstance(c, dict):
        return " ".join(str(c.get(k, "")) for k in ("brief_caption", "detailed_caption"))
    if isinstance(c, (list, tuple)):
        return " ".join(map(str, c))
    return str(c)


def brief(c) -> str:
    if isinstance(c, dict):
        return str(c.get("brief_caption", ""))
    if isinstance(c, (list, tuple)):
        return str(c[0]) if c else ""
    return str(c)


def classify(parts: dict) -> dict:
    texts = [caption_text(c) for c in parts.values()]
    obj_hits = Counter()
    part_hits = set()
    toy = 0
    for t in texts:
        for w in words_in(t, OBJECT_WORDS):
            obj_hits[OBJECT_WORDS[w]] += 1
        part_hits.update(words_in(t, CAR_PARTS))
        toy += bool(words_in(t, TOY_WORDS))
    kind, votes = (obj_hits.most_common(1)[0] if obj_hits else (None, 0))
    if kind == "vehicle" and obj_hits.get("car", 0) >= max(1, votes // 3):
        kind = "car"
    n = max(len(texts), 1)
    score = votes / n + 0.15 * len(part_hits & {"wheel", "tire", "tyre", "windshield", "windscreen", "headlight",
                                                 "headlamp", "bumper", "door"})
    return {"type": kind, "score": round(score, 3), "n_parts": len(texts), "object_word_votes": dict(obj_hits),
            "car_parts_named": sorted(part_hits), "toy_mentions": toy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="data/partverse/text_captions.json")
    ap.add_argument("--out", default="data/partverse")
    ap.add_argument("--min-score", type=float, default=0.5)
    a = ap.parse_args()

    caps = json.loads(Path(a.captions).read_text())
    print(f"{len(caps)} objects, {sum(len(v) for v in caps.values())} parts in {a.captions}")
    rows = []
    for uid, parts in caps.items():
        if not isinstance(parts, dict) or not parts:
            continue
        info = classify(parts)
        if info["type"] in ("car", "truck", "bus", "vehicle", "other_vehicle") and info["score"] >= a.min_score:
            info["uid"] = uid
            info["captions"] = {pid: brief(c) for pid, c in parts.items()}
            rows.append(info)
    rows.sort(key=lambda r: (r["type"] != "car", -r["score"]))
    types = Counter(r["type"] for r in rows)
    print("vehicle-like objects by type:", dict(types))
    cars = [r for r in rows if r["type"] == "car"]
    if cars:
        np_ = sorted(r["n_parts"] for r in cars)
        print(f"cars: {len(cars)} (of which mention toys: {sum(r['toy_mentions'] > 0 for r in cars)}), "
              f"parts per car: median {np_[len(np_) // 2]}, range {np_[0]}-{np_[-1]}")
        named = Counter(p for r in cars for p in r["car_parts_named"])
        print("most often named car parts:", ", ".join(f"{k} ({v})" for k, v in named.most_common(15)))
        for r in cars[:3]:
            print(f"\n  example {r['uid']}  ({r['n_parts']} parts, score {r['score']}):")
            for pid, c in list(r["captions"].items())[:6]:
                print(f"    part {pid}: {c}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vehicles.json").write_text(json.dumps(rows, indent=1))
    (out / "car_uids.txt").write_text("\n".join(r["uid"] for r in cars) + "\n")
    print(f"\nwrote {out / 'vehicles.json'} ({len(rows)} vehicles) and {out / 'car_uids.txt'} ({len(cars)} cars)")


if __name__ == "__main__":
    main()
