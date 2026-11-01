"""Evaluate a CAD world model on the benchmark: given (state, operation), predict the next state.

Predictors
    oracle  the kernel's answer (sanity check: perfect scores)
    copy    "nothing happens": the previous state with the new step number (the floor every model must beat;
            strong on small cuts, useless on failures and real geometry changes)
    llm     any OpenAI-compatible model, prompted with the world-model system prompt (zero-shot); settings from .env
            (META_API_BASE / MODEL_API_MODEL / META_API_KEY), or --base-url / --model / --api-key-env

Usage:
    python scripts/run_world_model.py --bench data_v21/wm_bench/bench.jsonl --predictor oracle copy
    python scripts/run_world_model.py --bench data_v21/wm_bench/bench.jsonl --predictor llm --limit 30
    python scripts/run_world_model.py --summarize-only
Results: results_wm21/<predictor>.jsonl (one line per sample, appended as it goes; --resume skips finished ones)
and results_wm21/summary.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carworld.llm import ChatError, chat, last_json_object, load_dotenv  # noqa: E402
from carworld.prompts import WORLD_MODEL_SYSTEM, world_model_user  # noqa: E402

MAT_MIN = 0.01   # a material "is present" on a part if it covers at least 1 % of its surface
CHANGE_TOL = 0.002


# ------------------------------------------------------------------ predictors
def predict_oracle(s, args):
    return s["next_state"], {}


def predict_copy(s, args):
    return {**s["state"], "step": s["action"].get("step", s["state"].get("step", 0) + 1)}, {}


def predict_llm(s, args):
    messages = [{"role": "system", "content": WORLD_MODEL_SYSTEM},
                {"role": "user", "content": world_model_user(s["state"], s["action"])}]
    budget, attempts = args.max_tokens, []
    for attempt in range(1 + args.retries):
        text, finish = chat(messages, base_url=args.base_url, model=args.model,
                            api_key=os.environ.get(args.api_key_env, "none"), max_tokens=budget,
                            reasoning_effort=args.reasoning_effort, timeout=args.timeout)
        attempts.append({"finish_reason": finish, "chars": len(text)})
        try:
            return last_json_object(text, must_have=("parts",)), {"attempts": len(attempts), "raw_tail": text[-600:]}
        except ValueError:
            budget *= 2
            messages = messages + ([{"role": "assistant", "content": text}] if text else []) + [
                {"role": "user", "content": "Reply with the next state as one JSON object only."}]
    raise ChatError(f"no usable JSON after {len(attempts)} attempts", {"attempts": attempts})


PREDICTORS = {"oracle": predict_oracle, "copy": predict_copy, "llm": predict_llm}


# ------------------------------------------------------------------ scoring
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _mats(part: dict) -> set:
    sm = part.get("surface_materials") or {}
    return {m for m, v in sm.items() if (_num(v) or 0) >= MAT_MIN}


def _changed(a: dict, b: dict) -> bool:
    """Did the state change between a and b (geometry, parts or validity)?"""
    if bool(a.get("valid", True)) != bool(b.get("valid", True)):
        return True
    pa, pb = a.get("parts") or {}, b.get("parts") or {}
    if set(pa) != set(pb):
        return True
    for name, x in pb.items():
        v0, v1 = _num(pa[name].get("volume_l")) or 0.0, _num(x.get("volume_l")) or 0.0
        if abs(v1 - v0) > CHANGE_TOL * max(v0, 1.0) or _mats(pa[name]) != _mats(x):
            return True
    return False


def score(pred: dict, s: dict) -> dict:
    gt, st = s["next_state"], s["state"]
    gv, pv = bool(gt.get("valid", True)), bool(pred.get("valid", True))
    m = {"valid_correct": gv == pv, "gt_valid": gv, "pred_valid": pv,
         "change_correct": _changed(st, gt) == _changed(st, pred)}
    gp, pp = gt.get("parts") or {}, pred.get("parts") if isinstance(pred.get("parts"), dict) else {}
    m["parts_exact"] = set(gp) == set(pp)
    sp = st.get("parts") or {}
    vol_errs, delta_errs, bbox_errs, mat_j, sym = [], [], [], [], []
    for name, g in gp.items():
        p = pp.get(name)
        if not isinstance(p, dict):
            vol_errs.append(1.0)
            delta_errs.append(1.0)
            continue
        gvol, pvol = _num(g.get("volume_l")) or 0.0, _num(p.get("volume_l"))
        svol = _num((sp.get(name) or {}).get("volume_l")) or 0.0
        vol_errs.append(1.0 if pvol is None else min(1.0, abs(pvol - gvol) / max(gvol, 1.0)))
        # error in the predicted CHANGE of volume, relative to the true change (a no-change step tolerates 0.2 %)
        gd, pd = gvol - svol, (None if pvol is None else pvol - svol)
        tol = CHANGE_TOL * max(svol, gvol, 1.0)
        delta_errs.append(1.0 if pd is None else min(1.0, abs(pd - gd) / max(abs(gd), tol)))
        gb, pb = g.get("bbox_mm"), p.get("bbox_mm")
        if gb and isinstance(pb, list) and len(pb) == 3 and all(_num(v) is not None for v in pb):
            bbox_errs.append(float(np.mean([abs(_num(a) - b) for a, b in zip(pb, gb)])))
        gm, pm = _mats(g), _mats(p)
        if gm or pm:
            mat_j.append(len(gm & pm) / len(gm | pm))
        if "left_right_symmetric" in g:
            sym.append(bool(p.get("left_right_symmetric")) == bool(g["left_right_symmetric"]))
    m["volume_change_err"] = float(np.mean(delta_errs)) if delta_errs else 0.0
    m["volume_rel_err"] = float(np.mean(vol_errs)) if vol_errs else 0.0
    m["bbox_err_mm"] = float(np.mean(bbox_errs)) if bbox_errs else None
    m["materials_jaccard"] = float(np.mean(mat_j)) if mat_j else None
    m["symmetry_correct"] = float(np.mean(sym)) if sym else None
    # one number per sample, in [0, 1]: validity, did-it-change, parts, size of the volume change, materials
    terms = [m["valid_correct"], m["change_correct"], m["parts_exact"], 1 - m["volume_change_err"]]
    if m["materials_jaccard"] is not None:
        terms.append(m["materials_jaccard"])
    m["score"] = float(np.mean([float(t) for t in terms]))
    return m


# ------------------------------------------------------------------ summary
def summarize(out: Path) -> str:
    lines = []
    kinds_order = ["normal", "scaled", "no_effect", "wipe_body", "missing_part", "bad_tool"]
    header = ("| predictor | n | parsed | score | valid acc | failure F1 | change acc | parts exact | "
              "volume-change err | volume err | bbox err (mm) | materials J |")
    lines += [header, "|" + "---|" * 12]
    per_kind = {}
    for f in sorted(Path(out).glob("*.jsonl")):
        rows = list({r["id"]: r for r in map(json.loads, open(f))}.values())
        ok = [r for r in rows if r.get("parsed")]

        def mean(k):
            v = [r[k] for r in ok if r.get(k) is not None and not (isinstance(r[k], float) and math.isnan(r[k]))]
            return float(np.mean(v)) if v else float("nan")
        tp = sum((not r["gt_valid"]) and (not r["pred_valid"]) for r in ok)
        fp = sum(r["gt_valid"] and (not r["pred_valid"]) for r in ok)
        fn = sum((not r["gt_valid"]) and r["pred_valid"] for r in ok)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        # unparsed replies count as score 0
        sc = float(np.mean([r.get("score", 0.0) if r.get("parsed") else 0.0 for r in rows])) if rows else float("nan")
        lines.append(f"| {f.stem} | {len(rows)} | {len(ok) / max(len(rows), 1):.0%} | {sc:.3f} | "
                     f"{mean('valid_correct'):.0%} | {f1:.2f} | {mean('change_correct'):.0%} | "
                     f"{mean('parts_exact'):.0%} | {mean('volume_change_err'):.3f} | {mean('volume_rel_err'):.3f} | "
                     f"{mean('bbox_err_mm'):.0f} | "
                     f"{mean('materials_jaccard'):.2f} |")
        by = defaultdict(list)
        for r in rows:
            by[r["kind"]].append(r.get("score", 0.0) if r.get("parsed") else 0.0)
        per_kind[f.stem] = {k: (float(np.mean(v)), len(v)) for k, v in by.items()}
    lines += ["", "Score by step kind:", "| predictor | " + " | ".join(kinds_order) + " |",
              "|" + "---|" * (len(kinds_order) + 1)]
    for name, d in per_kind.items():
        lines.append(f"| {name} | " + " | ".join(f"{d[k][0]:.3f} (n={d[k][1]})" if k in d else "-"
                                                 for k in kinds_order) + " |")
    text = "\n".join(lines)
    (Path(out) / "summary.md").write_text(text + "\n")
    return text


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data_v21/wm_bench/bench.jsonl")
    ap.add_argument("--predictor", nargs="+", default=["oracle", "copy"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results_wm21")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--base-url", default=os.environ.get("META_API_BASE", "http://localhost:1234/v1"))
    ap.add_argument("--model", default=os.environ.get("MODEL_API_MODEL", "muse-spark-1.3"))
    ap.add_argument("--api-key-env", default="META_API_KEY" if os.environ.get("META_API_KEY") else "OPENAI_API_KEY")
    ap.add_argument("--reasoning-effort",
                    default=os.environ.get("REASONING_EFFORT") or ("low" if "api.meta.ai" in os.environ.get("META_API_BASE", "") else None))
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.summarize_only:
        print(summarize(out))
        return
    samples = [json.loads(l) for l in open(a.bench)]
    if a.limit:  # keep the kind mix: take samples evenly through the file
        idx = np.linspace(0, len(samples) - 1, a.limit).round().astype(int)
        samples = [samples[i] for i in sorted(set(idx.tolist()))]
    for name in a.predictor:
        if name == "llm" and not os.environ.get(a.api_key_env) and "localhost" not in a.base_url:
            sys.exit(f"No API key: put {a.api_key_env}=... in .env")
        res = out / f"{name}.jsonl"
        if not (a.resume) and res.exists():
            res.unlink()
        done = {r["id"] for r in map(json.loads, open(res)) if r.get("parsed")} if res.exists() else set()
        for s in samples:
            if s["id"] in done:
                continue
            row = {"id": s["id"], "kind": s["kind"], "style": s["style"]}
            t0 = time.time()
            try:
                pred, info = PREDICTORS[name](s, a)
                row.update(score(pred, s), parsed=True, prediction=pred, info=info)
            except Exception as e:  # unparsable reply / API error: counted as score 0
                row.update(parsed=False, error=f"{type(e).__name__}: {e}", info=getattr(e, "info", {}))
            row["seconds"] = round(time.time() - t0, 2)
            with open(res, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"  [{name}] {s['id']:24s} " + (f"score {row['score']:.3f}" if row.get("parsed") else row["error"][:80]),
                  flush=True)
    print("\n" + summarize(out))


if __name__ == "__main__":
    main()
