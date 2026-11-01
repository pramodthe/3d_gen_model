"""Convert our training files into Fireworks fine-tuning datasets (OpenAI chat JSONL).

    planner  -> images embedded as base64 data URLs (Fireworks rejects plain file paths or http URLs for VLM
                fine-tuning). Use with a Fireworks vision base model that supports LoRA SFT (Qwen2.5-VL at the time
                of writing).
    world    -> text-only (state, operation) -> next state. Our per-turn `weight` is applied by keeping each turn
                with probability = weight (near no-op turns are mostly dropped), so the file works with any
                SFT service, not only ones that read loss weights.

Usage:
    python scripts/export_fireworks.py --data data/train --kind planner --out fireworks_planner.jsonl
    python scripts/export_fireworks.py --data data/train --kind world   --out fireworks_world.jsonl
Then upload and start a job with Fireworks' firectl or web UI (see their fine-tuning docs).
"""
from __future__ import annotations

import argparse
import base64
import json
import random
from pathlib import Path


def image_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def export_planner(split: Path, out, limit: int | None) -> int:
    n = 0
    for line in open(split / "planner_sft.jsonl"):
        rec = json.loads(line)
        msgs = []
        for m in rec["messages"]:
            if m["role"] == "user" and isinstance(m["content"], list):
                parts = []
                for c in m["content"]:
                    if c.get("type") == "image":
                        parts.append({"type": "image_url", "image_url": {"url": image_url(split / rec["image"])}})
                    else:
                        parts.append(c)
                msgs.append({"role": "user", "content": parts})
            else:
                msgs.append(m)
        out.write(json.dumps({"messages": msgs}) + "\n")
        n += 1
        if limit and n >= limit:
            break
    return n


def export_world(split: Path, out, limit: int | None, seed: int) -> int:
    rng = random.Random(seed)
    n = 0
    for line in open(split / "world_model.jsonl"):
        rec = json.loads(line)
        if rng.random() > rec.get("weight", 1.0):
            continue
        out.write(json.dumps({"messages": rec["messages"]}) + "\n")
        n += 1
        if limit and n >= limit:
            break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train", help="a split folder made by generate_dataset.py")
    ap.add_argument("--kind", choices=["planner", "world"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    split = Path(a.data)
    with open(a.out, "w") as out:
        n = export_planner(split, out, a.limit) if a.kind == "planner" else export_world(split, out, a.limit, a.seed)
    size = Path(a.out).stat().st_size / 1e6
    print(f"wrote {n} examples to {a.out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
