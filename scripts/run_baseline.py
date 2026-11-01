"""Step 2: Stage-0 baseline. Predict each test car from ONE image, rebuild it, and score it.

Predictors
    oracle   ground-truth parameters (sanity check: should score ~perfect)
    mean     the average car (ignores the image) - the floor any real method must beat
    random   a random car of a random style
    search   render-and-compare: no learning, just search over parameters to match the photo's silhouette
             (assumes the camera is known; the learned planner will have to estimate it)
    vlm      a vision-language model through any OpenAI-compatible endpoint (LM Studio, Ollama, vLLM, a cloud API)
             e.g. --predictor vlm --base-url http://localhost:1234/v1 --model qwen2.5-vl-7b-instruct
             Meta (Muse Spark): put META_API_KEY, META_API_BASE=https://api.meta.ai/v1 and
                        MODEL_API_MODEL=muse-spark-1.3 in .env; then no flags are needed.
             Fireworks: --base-url https://api.fireworks.ai/inference/v1 --api-key-env FIREWORKS_API_KEY
                        --model accounts/fireworks/models/<a vision model>
             The API key, if one is needed, is read from the environment variable named by --api-key-env.
    vlm_program          the model writes the CAD program; feedback = kernel errors or render + outline overlay
    vlm_program_3d       + 3D feedback about its OWN program: kernel state summary and front/side/top renders
    vlm_program_truth3d  + the TRUE car's state summary (upper bound for 3D feedback, not a method)

Usage:
    python scripts/run_baseline.py --data data/test --predictor mean search --limit 20
Writes results/<predictor>.jsonl (one line per car, appended as it goes), results/summary.md and side-by-side
images in results/compare/. Long runs can be split across machines with --start/--limit/--append, then merged
with --summarize-only.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from carworld import metrics  # noqa: E402
from carworld import params as cp  # noqa: E402
from carworld import program as pg  # noqa: E402
from carworld import sdf_kernel as sk  # noqa: E402
from carworld.prompts import FRAME, OP_LANGUAGE, PARAMS_PROMPT  # noqa: E402
from carworld.render import Camera, Look, render  # noqa: E402


# ------------------------------------------------------------------ predictors
def predict_oracle(sample, rng, args):
    return sample["params"], {}


def predict_mean(sample, rng, args):
    return cp.mean_params("sedan"), {}


def predict_random(sample, rng, args):
    return cp.sample(rng), {}


def predict_search(sample, rng, args):
    """Analysis by synthesis: render candidate cars at low resolution and keep the best silhouette match."""
    cam = sample["camera"]
    small = Camera(**{**cam.to_dict(), "width": cam.width // 3, "height": cam.height // 3})
    target = np.array(Image.fromarray(sample["mask"]).resize((small.width, small.height), Image.NEAREST)) > 0
    renders = 0

    def score(p):
        nonlocal renders
        renders += 1
        m = sk.Model(pg.build_program(p))
        return metrics.mask_iou(render(m, small, want_rgb=False, max_steps=90)["mask"], target)

    # 1) coarse: a few random cars per style
    cands = []
    for style in cp.STYLES:
        for _ in range(args.search_init):
            p = cp.sample(rng, style)
            cands.append((score(p), p))
    cands.sort(key=lambda t: -t[0])
    best_s, best = cands[0]
    # 2) local refinement: perturb a few parameters at a time, shrinking the step size
    for it in range(args.search_iters):
        scale = 0.25 * (1 - it / args.search_iters) + 0.03
        d = best.to_dict()
        r = cp.ranges_for(best.style)
        for k in rng.choice(cp.CONTINUOUS, size=3, replace=False):
            lo, hi = r[k]
            d[k] = float(np.clip(d[k] + rng.normal(0, scale * (hi - lo)), lo, hi))
        p = cp.CarParams.from_dict(d)
        if cp.validate(p):
            continue
        s = score(p)
        if s > best_s:
            best_s, best = s, p
    return best, {"renders": renders, "search_iou_lowres": round(best_s, 4)}


class PredictionError(ValueError):
    """A model reply we could not use; carries the reply so it can be inspected in results/<predictor>.jsonl."""

    def __init__(self, msg, info):
        super().__init__(msg)
        self.info = info


def _chat(args, messages, max_tokens, effort):
    body = {"model": args.model, "temperature": 0.2, "max_tokens": max_tokens, "messages": messages,
            **({"reasoning_effort": effort} if effort else {})}
    req = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ.get(args.api_key_env, 'none')}"})
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            choice = json.loads(resp.read())["choices"][0]
    except urllib.error.HTTPError as e:
        raise PredictionError(f"HTTP {e.code}", {"http_body": e.read()[:500].decode(errors="replace")})
    msg = choice.get("message") or {}
    return msg.get("content") or "", choice.get("finish_reason")


def _png_url(arr_or_path) -> str:
    if isinstance(arr_or_path, (str, Path)):
        data = Path(arr_or_path).read_bytes()
    else:
        import io
        buf = io.BytesIO()
        Image.fromarray(arr_or_path).save(buf, format="PNG")
        data = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _ask_params(args, content, parser=None):
    """Send one user turn (text + images) and parse car parameters from the reply.
    Reasoning models can spend the whole token budget thinking and return nothing, so an empty or unparsable reply
    is retried with twice the budget and a stricter instruction."""
    messages = [{"role": "user", "content": content}]
    attempts = []
    budget, effort = args.max_tokens, args.reasoning_effort
    for attempt in range(1 + args.retries):
        text, finish = _chat(args, messages, budget, effort)
        attempts.append({"finish_reason": finish, "chars": len(text), "tail": text[-300:],
                         "head": text[:1500] if parser is parse_program else ""})
        try:
            return (parser or parse_params)(text), {"raw": text[-2000:], "attempts": len(attempts), "finish_reason": finish}
        except ValueError:
            budget *= 2
            if attempt == 0:
                messages = messages + ([{"role": "assistant", "content": text}] if text else []) + [
                    {"role": "user", "content": "Reply with the JSON only: no explanation, no code fence."}]
    why = "empty reply" if not attempts[-1]["chars"] else "no usable JSON in reply"
    raise PredictionError(f"{why} after {len(attempts)} attempts (finish_reason={attempts[-1]['finish_reason']})",
                          {"attempts": attempts})


def predict_vlm(sample, rng, args):
    """Zero-shot: photo only."""
    return _ask_params(args, [{"type": "text", "text": PARAMS_PROMPT},
                              {"type": "image_url", "image_url": {"url": _png_url(sample["image_path"])}}])


def predict_vlm_cam(sample, rng, args):
    """Control: photo + the camera, so the model can work out absolute size (search gets the camera too)."""
    return _ask_params(args, [{"type": "text", "text": PARAMS_PROMPT + "\n\n" + camera_text(sample["camera"])},
                              {"type": "image_url", "image_url": {"url": _png_url(sample["image_path"])}}])


def predict_vlm_loop(sample, rng, args):
    """Render-and-revise: the model sees the photo next to a render of its current guess plus a difference overlay,
    and revises. Uses only what a real photo would give: the image, its car mask (SAM on real photos) and the camera.
    The best round by silhouette IoU against the photo's mask is returned."""
    cam, target = sample["camera"], sample["mask"]
    content = [{"type": "text", "text": PARAMS_PROMPT + "\n\n" + camera_text(cam)},
               {"type": "image_url", "image_url": {"url": _png_url(sample["image_path"])}}]
    p, info0 = _ask_params(args, content)
    history, best = [], None
    for rnd in range(args.loop_rounds + 1):
        r = render(sk.Model(pg.build_program(p)), cam, sample["look"])
        iou = metrics.mask_iou(r["mask"], target)
        history.append({"round": rnd, "iou": round(iou, 4), "params": p.to_dict()})
        if best is None or iou > best[0]:
            best = (iou, p)
        panel = feedback_panel(sample["image"], r["rgb"], target, r["mask"])
        if args.out:
            d = Path(args.out) / "loop"
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(panel).save(d / f"{sample['id']}_round{rnd}.png")
        if rnd == args.loop_rounds or iou >= args.loop_stop_iou:
            break
        tries = "; ".join(f"round {h['round']}: IoU {h['iou']:.3f}" for h in history)
        text = (PARAMS_PROMPT + "\n\n" + camera_text(cam) + "\n\n"
                "You already proposed parameters; they were built and rendered from the same camera. "
                "The second image shows, left to right: the photo, the render of your parameters, and a difference "
                "overlay. In the overlay GREEN = car in both, RED = car in the photo but missing from your model "
                "(make the model bigger or higher there), BLUE = your model where the photo has no car (make it "
                "smaller or lower there).\n"
                f"Outline overlap so far (1.0 = perfect): {tries}. Your best so far is shown below.\n"
                f"Best parameters so far: {json.dumps(best[1].to_dict())}\n"
                f"Parameters just rendered: {json.dumps(p.to_dict())}\n"
                "Work out which parameters cause the red and blue regions, change them, and output the complete "
                "revised JSON object only.")
        p, _ = _ask_params(args, [{"type": "text", "text": text},
                                  {"type": "image_url", "image_url": {"url": _png_url(sample["image_path"])}},
                                  {"type": "image_url", "image_url": {"url": _png_url(panel)}}])
    return best[1], {"rounds": history, "first_reply": info0.get("raw", "")[-500:]}


TWO_STYLES = ("\n\nGive your TWO most likely interpretations, with DIFFERENT body styles, most likely first, as a JSON "
              "list of two complete parameter objects: [{...}, {...}]. Output only that list.")
CLIP_NOTE = ("Each body style only allows the ranges listed above; values outside are clipped. If a value you need is "
             "clipped (for example the car is taller or higher off the ground than this style allows), switch "
             "'style' instead of fighting the limit.")


def _parse_two(text):
    groups = json_objects(text)
    if not groups:
        raise ValueError("no JSON object in reply")
    cands = groups[-1] if len(groups[-1]) >= 2 else [x for g in groups for x in g]
    out = [params_and_clips(d) for d in cands[:2]]
    return out


def _parse_one(text):
    groups = json_objects(text)
    if not groups:
        raise ValueError("no JSON object in reply")
    return params_and_clips(groups[-1][-1])


def predict_vlm_loop2(sample, rng, args):
    """Render-and-revise, v2: starts from the better of two candidate body styles, may switch style while revising,
    and is told which of its values were clipped by the style's ranges."""
    cam, target = sample["camera"], sample["mask"]
    base = PARAMS_PROMPT + "\n\n" + camera_text(cam)
    photo = {"type": "image_url", "image_url": {"url": _png_url(sample["image_path"])}}
    cands, _ = _ask_params(args, [{"type": "text", "text": base + TWO_STYLES}, photo], parser=_parse_two)

    def evaluate(p):
        r = render(sk.Model(pg.build_program(p)), cam, sample["look"])
        return metrics.mask_iou(r["mask"], target), r

    scored = []
    for p, clips in cands:
        iou, r = evaluate(p)
        scored.append((iou, p, clips, r))
    scored.sort(key=lambda t: -t[0])
    alt = scored[1] if len(scored) > 1 else None
    iou, p, clips, r = scored[0]
    history = [{"round": 0, "iou": round(iou, 4), "style": p.style, "params": p.to_dict(), "clipped": clips,
                "candidates": [{"style": c[1].style, "iou": round(c[0], 4)} for c in scored]}]
    best = (iou, p)
    for rnd in range(1, args.loop_rounds + 1):
        panel = feedback_panel(sample["image"], r["rgb"], target, r["mask"])
        if args.out:
            d = Path(args.out) / "loop2"
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(panel).save(d / f"{sample['id']}_round{rnd - 1}.png")
        if best[0] >= args.loop_stop_iou:
            break
        tries = "; ".join(f"round {h['round']} ({h['style']}): IoU {h['iou']:.3f}" for h in history)
        clip_txt = ("Values clipped in the parameters just rendered: " +
                    ", ".join(f"{k} asked {v['asked']} -> allowed {v['allowed']}" for k, v in clips.items())
                    if clips else "No values were clipped in the parameters just rendered.")
        alt_txt = (f"Your other initial interpretation was a {alt[1].style} (IoU {alt[0]:.3f})." if alt else "")
        text = (base + "\n\n" + CLIP_NOTE + "\n\n"
                "You already proposed parameters; they were built and rendered from the same camera. The second image "
                "shows, left to right: the photo, the render of your parameters, and a difference overlay. In the "
                "overlay GREEN = car in both, RED = car in the photo but missing from your model (make the model bigger "
                "or higher there), BLUE = your model where the photo has no car (make it smaller or lower there).\n"
                f"Outline overlap so far (1.0 = perfect): {tries}. {alt_txt}\n{clip_txt}\n"
                f"Best parameters so far: {json.dumps(best[1].to_dict())}\n"
                f"Parameters just rendered: {json.dumps(p.to_dict())}\n"
                "Decide first whether the body style is right (you may change it), then which parameters cause the red "
                "and blue regions. Output the complete revised JSON object only.")
        (p, clips), _ = _ask_params(args, [{"type": "text", "text": text}, photo,
                                           {"type": "image_url", "image_url": {"url": _png_url(panel)}}],
                                    parser=_parse_one)
        iou, r = evaluate(p)
        history.append({"round": rnd, "iou": round(iou, 4), "style": p.style, "params": p.to_dict(), "clipped": clips})
        if iou > best[0]:
            best = (iou, p)
    if args.out:  # the last render too
        Image.fromarray(feedback_panel(sample["image"], r["rgb"], target, r["mask"])).save(
            Path(args.out) / "loop2" / f"{sample['id']}_round{history[-1]['round']}.png")
    return best[1], {"rounds": history}


def _lock_style(p: cp.CarParams, clips: dict, style: str):
    """Force a branch's body style: keep the model's numbers, re-clip them to that style's ranges."""
    if p.style == style:
        return p, clips
    d = {**p.to_dict(), "style": style}
    q = cp.clamp_to_ranges(cp.CarParams.from_dict(d))
    clips = {**clips, "style": {"asked": p.style, "allowed": style}}
    for k in cp.CONTINUOUS:
        if abs(getattr(q, k) - getattr(p, k)) > 1e-6:
            clips[k] = {"asked": round(getattr(p, k), 3), "allowed": round(getattr(q, k), 3)}
    return q, clips


def predict_vlm_loop3(sample, rng, args):
    """Render-and-revise, v3: refine BOTH candidate body styles in separate branches (style locked per branch) and
    keep the better final result. Fixes v2's main error: picking the wrong style at the start and rarely switching."""
    cam, target = sample["camera"], sample["mask"]
    base = PARAMS_PROMPT + "\n\n" + camera_text(cam)
    photo = {"type": "image_url", "image_url": {"url": _png_url(sample["image_path"])}}
    cands, _ = _ask_params(args, [{"type": "text", "text": base + TWO_STYLES}, photo], parser=_parse_two)
    seen, branches_in = set(), []
    for p, clips in cands:                       # two different styles; if the model repeated one, keep one branch
        if p.style not in seen:
            seen.add(p.style)
            branches_in.append((p, clips))

    def evaluate(p):
        r = render(sk.Model(pg.build_program(p)), cam, sample["look"])
        return metrics.mask_iou(r["mask"], target), r

    branches = []
    for p, clips in branches_in:
        style = p.style
        iou, r = evaluate(p)
        history = [{"round": 0, "iou": round(iou, 4), "style": style, "params": p.to_dict(), "clipped": clips}]
        best = (iou, p)
        for rnd in range(1, args.loop_rounds + 1):
            panel = feedback_panel(sample["image"], r["rgb"], target, r["mask"])
            if args.out:
                d = Path(args.out) / "loop3"
                d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(panel).save(d / f"{sample['id']}_{style}_round{rnd - 1}.png")
            if best[0] >= args.loop_stop_iou:
                break
            tries = "; ".join(f"round {h['round']}: IoU {h['iou']:.3f}" for h in history)
            clip_txt = ("Values clipped in the parameters just rendered: " +
                        ", ".join(f"{k} asked {v['asked']} -> allowed {v['allowed']}" for k, v in clips.items())
                        if clips else "No values were clipped in the parameters just rendered.")
            text = (base + "\n\n"
                    f"In this attempt the body style is FIXED to '{style}': keep \"style\": \"{style}\" and adjust only the "
                    "numbers, within this style's ranges (values outside are clipped). Another attempt explores a "
                    "different style, so make this one the best possible " + style + ".\n\n"
                    "Your parameters were built and rendered from the same camera. The second image shows, left to "
                    "right: the photo, the render of your parameters, and a difference overlay. In the overlay GREEN = "
                    "car in both, RED = car in the photo but missing from your model (make the model bigger or higher "
                    "there), BLUE = your model where the photo has no car (make it smaller or lower there).\n"
                    f"Outline overlap so far (1.0 = perfect): {tries}.\n{clip_txt}\n"
                    f"Best parameters so far: {json.dumps(best[1].to_dict())}\n"
                    f"Parameters just rendered: {json.dumps(p.to_dict())}\n"
                    "Work out which parameters cause the red and blue regions, change them, and output the complete "
                    "revised JSON object only.")
            (p, clips), _ = _ask_params(args, [{"type": "text", "text": text}, photo,
                                               {"type": "image_url", "image_url": {"url": _png_url(panel)}}],
                                        parser=_parse_one)
            p, clips = _lock_style(p, clips, style)
            iou, r = evaluate(p)
            history.append({"round": rnd, "iou": round(iou, 4), "style": style, "params": p.to_dict(),
                            "clipped": clips})
            if iou > best[0]:
                best = (iou, p)
        if args.out:
            d = Path(args.out) / "loop3"
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(feedback_panel(sample["image"], r["rgb"], target, r["mask"])).save(
                d / f"{sample['id']}_{style}_round{history[-1]['round']}.png")
        branches.append({"style": style, "best_iou": round(best[0], 4), "best": best[1], "rounds": history})
    win = max(branches, key=lambda b: b["best_iou"])
    return win["best"], {"chosen_style": win["style"],
                         "branches": [{k: v for k, v in b.items() if k != "best"} for b in branches]}


PROGRAM_TASK = (
    "Write the complete CAD program that rebuilds the car in this photo, as a JSON list of operations in the "
    "language below. You decide which components to build, their shapes and their positions.\n" + OP_LANGUAGE +
    "\nRules: the first operation must use mode \"new\" on part \"body\"; \"cut\" and \"intersect\" need an existing "
    "part; points are in millimetres in the plane's coordinates; extent is [w0, w1] with w1 > w0 along the remaining "
    "axis. Wheels are a separate part (\"wheels\", mode \"union\"). Use mirror=true for anything that exists on both "
    "sides. Materials tag the faces a cut creates (glass, light, grille...) or the colour of an added piece.\n"
)
PROGRAM_HINT = ("A good order: body from a side profile (XZ prism across the width), round it in top view (XY) and "
                "narrow the cabin (YZ), cut wheel arches, inset the glass, cut lights and grille, add bumpers and "
                "mirrors, then the tyres and rims.\n")


def _example_program() -> str:
    ops = [sk.public_op(o) for o in pg.build_program(cp.mean_params("sedan"))]
    for o in ops:
        o.pop("step", None)
    return json.dumps(ops, separators=(",", ":"))


def _lenient_json(text: str) -> str:
    """Remove the usual JSON slips of language models: // and /* */ comments, trailing commas, code fences."""
    text = re.sub(r"```(?:json)?", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r'(?m)(^|[^:"\\])//[^\n]*', r"\1", text)
    text = re.sub(r",\s*([\]}])", r"\1", text)
    # a slip seen from Muse Spark: the "{" opening each next operation is dropped ('},"name":' for '},{"name":')
    return re.sub(r'\}\s*,\s*"name"\s*:', '},{"name":', text)


def parse_program(text: str) -> list[dict]:
    """The last JSON list of operations in a reply (or an object with an 'operations'/'program' list).
    Lenient about comments and trailing commas; entries without a 'tool' (notes, headers) are dropped."""
    if not text or not text.strip():
        raise ValueError("empty reply")
    dec, found = json.JSONDecoder(), None
    for candidate in (text, _lenient_json(text)):
        for m in re.finditer(r"[\[{]", candidate):
            try:
                obj, _ = dec.raw_decode(candidate[m.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                obj = obj.get("operations") or obj.get("program") or obj.get("ops")
            if isinstance(obj, list):
                tools = [o for o in obj if isinstance(o, dict) and "tool" in o]
                if tools and len(tools) >= 0.8 * len(obj):
                    found = tools
        if found is not None:
            break
    if found is None:
        raise ValueError("no JSON list of operations in reply")
    if len(found) > 80:
        raise ValueError(f"program too long ({len(found)} operations, max 80)")
    ops = []
    for i, o in enumerate(found):  # fill harmless defaults; the kernel judges everything else
        o = dict(o)
        o["step"] = i + 1
        o.setdefault("name", f"op{i + 1}")
        o.setdefault("mirror", False)
        o.setdefault("material", None)
        o.setdefault("comment", "")
        ops.append(o)
    return ops


# ------------------------------------------------------------------ 3D state feedback for the program writer
STATE_RES = 40.0          # mm grid for state summaries (same as the world-model benchmark)
SELF_VIEWS = [("front", 0.0, 4.0, 6500.0), ("left side", 90.0, 4.0, 9500.0), ("top", 90.0, 89.0, 9500.0)]  # name, az, el, distance


def program_state(ops: list[dict]) -> dict:
    """The kernel's state summary after the whole program: per part volume, bounding box, materials, symmetry.
    This is exactly what the world model predicts step by step, so a trained world model can stand in for it."""
    P, _, res = sk.make_grid(STATE_RES, centered=True)
    ops = [sk.public_op(o) for o in ops]          # drop the kernel's cached arrays (e.g. on the true program)
    *_, (i, part_d, part_m) = sk.Model(json.loads(json.dumps(ops))).iter_eval(P)
    return sk._summary(ops[i], part_d, part_m, P, res)


def _overall(state: dict):
    boxes = [(p["min_mm"], p["bbox_mm"]) for p in state["parts"].values() if p.get("bbox_mm")]
    if not boxes:
        return None
    lo = np.min([b[0] for b in boxes], 0)
    hi = np.max([np.add(b[0], b[1]) for b in boxes], 0)
    return lo, hi


def state_text(state: dict, title: str) -> str:
    """Readable table of a state summary: L along X (front-back), W along Y (across), H along Z (up), in mm."""
    lines = [title]
    ov = _overall(state)
    if ov:
        lo, hi = ov
        lines.append(f"  whole car: length {hi[0] - lo[0]:.0f}, width {hi[1] - lo[1]:.0f}, height {hi[2] - lo[2]:.0f} "
                     f"(x {lo[0]:.0f}..{hi[0]:.0f}, y {lo[1]:.0f}..{hi[1]:.0f}, z {lo[2]:.0f}..{hi[2]:.0f})")
    for name, p in state["parts"].items():
        if not p.get("bbox_mm"):
            lines.append(f"  {name}: empty")
            continue
        (x0, y0, z0), (L, W, H) = p["min_mm"], p["bbox_mm"]
        mats = ", ".join(f"{m} {v:.0%}" for m, v in sorted(p.get("surface_materials", {}).items(), key=lambda kv: -kv[1]))
        lines.append(f"  {name}: length {L}, width {W}, height {H} (x {x0}..{x0 + L}, y {y0}..{y0 + W}, "
                     f"z {z0}..{z0 + H}); volume {p['volume_l']:.0f} L; surface {mats}; "
                     f"{'left-right symmetric' if p.get('left_right_symmetric') else 'NOT left-right symmetric'}")
    return "\n".join(lines)


def self_views(ops: list[dict], look) -> np.ndarray:
    """The model's own car seen from the front, the left side and above (no information about the true car)."""
    model, tiles = sk.Model(json.loads(json.dumps(ops))), []
    for _, az, el, dist in SELF_VIEWS:
        cam = Camera(azimuth=az, elevation=el, distance=dist, fov=32.0, target_z=650.0, width=256, height=192)
        tiles += [render(model, cam, look)["rgb"], np.full((192, 6, 3), 255, np.uint8)]
    return np.concatenate(tiles[:-1], 1)


def predict_vlm_program_3d(sample, rng, args):
    """vlm_program plus 3D state feedback about the model's OWN program: the kernel's state summary (per-part sizes,
    positions, volumes, materials) and renders of its car from the front, side and top. Nothing about the true car
    is shown, so this is a fair method; the kernel's summary is what a learned world model would supply."""
    return predict_vlm_program(sample, rng, args, state_feedback="own", tag="program_3d")


def predict_vlm_program_truth3d(sample, rng, args):
    """UPPER BOUND, not a method: as vlm_program_3d, plus the TRUE car's state summary. It measures whether a model
    can act on 3D numbers at all, i.e. the ceiling for any 3D feedback."""
    return predict_vlm_program(sample, rng, args, state_feedback="truth", tag="program_truth3d")


def predict_vlm_program(sample, rng, args, state_feedback: str = "none", tag: str = "program"):
    """The model WRITES the build program (components, shapes, positions). The kernel executes it; the model sees
    kernel errors or its rendered result next to the photo with a difference overlay, and rewrites the program.
    state_feedback: "none" (silhouette only), "own" (+ its own 3D state and views), "truth" (+ the true state)."""
    cam, target = sample["camera"], sample["mask"]
    photo = {"type": "image_url", "image_url": {"url": _png_url(sample["image_path"])}}
    example = ("\nExample of a complete, valid program for a DIFFERENT car (a generic sedan), to show the format and "
               "the scale of the numbers. Do not copy it: measure the car in the photo.\n" + _example_program() + "\n"
               if args.program_example else "")
    base = PROGRAM_TASK + PROGRAM_HINT + example + "\n" + camera_text(cam) + "\n" + FRAME_NOTE
    history, best, ops = [], None, None
    content = [{"type": "text", "text": base + "\nOutput ONLY the JSON list of operations."}, photo]
    for rnd in range(args.loop_rounds + 1):
        ops, info = _ask_params(args, content, parser=parse_program)
        try:
            model = sk.Model(json.loads(json.dumps(ops)))   # the kernel caches arrays in its copy; keep ours JSON-clean
            r = render(model, cam, sample["look"])
            iou, err = metrics.mask_iou(r["mask"], target), None
        except Exception as e:  # the kernel rejected the program: that message is the feedback
            r, iou, err = None, 0.0, f"{type(e).__name__}: {e}"
        state = None
        if err is None and state_feedback != "none":
            try:
                state = program_state(ops)
            except Exception as e:
                state = None
        hist = {"round": rnd, "iou": round(iou, 4), "n_ops": len(ops), "error": err}
        if args.track_3d and err is None:   # diagnostics only, never shown to the model
            hist["voxel_iou"] = round(float(evaluate_ops(json.loads(json.dumps(ops)), sample)[1]["voxel_iou"]), 4)
        history.append(hist)
        if err is None and (best is None or iou > best[0]):
            best = (iou, ops)
        if args.out and r is not None:
            d = Path(args.out) / tag
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(feedback_panel(sample["image"], r["rgb"], target, r["mask"])).save(
                d / f"{sample['id']}_round{rnd}.png")
        if rnd == args.loop_rounds or (best and best[0] >= args.loop_stop_iou):
            break
        tries = "; ".join(f"round {h['round']}: " + (f"IoU {h['iou']:.3f}" if not h["error"] else "rejected")
                          for h in history)
        if err:
            fb = (f"Your program was REJECTED by the CAD kernel: {err}\nFix it and output the complete corrected "
                  "program.")
            imgs = [photo]
        else:
            fb = ("Your program ran. The second image shows, left to right: the photo, the render of your program "
                  "from the same camera, and a difference overlay: GREEN = car in both, RED = car in the photo but "
                  "missing from your model, BLUE = your model where the photo has no car. Change the operations that "
                  "cause the red and blue regions (and add or remove components if needed). Output the complete "
                  "revised program.")
            imgs = [photo, {"type": "image_url", "image_url": {"url": _png_url(
                feedback_panel(sample["image"], r["rgb"], target, r["mask"]))}}]
            if state_feedback != "none" and state is not None:
                views = self_views(ops, sample["look"])
                imgs.append({"type": "image_url", "image_url": {"url": _png_url(views)}})
                if args.out:
                    Image.fromarray(views).save(Path(args.out) / tag / f"{sample['id']}_round{rnd}_views.png")
                fb += ("\n\n3D CHECK. The outline overlay only shows one direction; width, cross-section and the far "
                       "side are invisible in it. The third image shows YOUR car from the front, the left side and "
                       "the top (left to right; in the side and top views the car's front is on the left). The CAD kernel "
                       "measured your car as follows (part 'body' includes everything you built into it, mirrors "
                       "too).\n"
                       + state_text(state, "Your car (measured by the CAD kernel, mm, litres):"))
                if state_feedback == "truth":
                    fb += ("\n" + state_text(sample.setdefault("gt_state", program_state(sample["ops"])),
                                              "The TRUE car in the photo (same format):")
                           + "\nMake each part's sizes and positions match the true car.")
                else:
                    fb += ("\nCompare every number with what the photo shows and what a real car of this kind is "
                           "like (overall width, wheel track and tyre width, roof and glass height, the cross-section "
                           "seen from the front, part positions). Fix the numbers and shapes that are wrong, "
                           "including those the outline overlay cannot show.")
        text = (base + f"\nOutline overlap so far (1.0 = perfect): {tries}.\n"
                f"Your last program:\n{json.dumps([{k: v for k, v in o.items() if k != 'step'} for o in ops], separators=(',', ':'))}\n\n"
                + fb + "\nOutput ONLY the JSON list of operations.")
        content = [{"type": "text", "text": text}, *imgs]
    if best is None:
        raise PredictionError(f"no valid program in {len(history)} rounds (last error: {history[-1]['error']})",
                              {"rounds": history})
    return {"ops": best[1]}, {"rounds": history}


FRAME_NOTE = FRAME


def camera_text(cam: Camera) -> str:
    return (f"Camera (exact): the image is {cam.width} x {cam.height} px with a vertical field of view of "
            f"{cam.fov:.1f} degrees. The camera looks at a point {cam.target_z:.0f} mm above the ground at the car's "
            f"centre, from {cam.distance / 1000:.2f} m away, {cam.elevation:.1f} degrees above horizontal, at azimuth "
            f"{cam.azimuth:.1f} degrees (0 = facing the car's front, +90 = its left side, -90 = its right side, "
            f"+/-180 = its rear). Use this to judge real sizes in millimetres.")


def feedback_panel(photo, render_rgb, target_mask, pred_mask):
    """[photo | render | overlay]: green both, red photo-only, blue render-only."""
    ov = (0.35 * photo.mean(-1, keepdims=True).repeat(3, -1) + 0.65 * 255).astype(np.uint8)
    ov[target_mask & pred_mask] = (60, 170, 80)
    ov[target_mask & ~pred_mask] = (220, 40, 40)
    ov[~target_mask & pred_mask] = (40, 90, 230)
    gap = np.full((photo.shape[0], 6, 3), 255, np.uint8)
    return np.concatenate([photo, gap, render_rgb, gap, ov], 1)


def parse_params(text: str) -> cp.CarParams:
    """Pull the last JSON object that looks like car parameters out of a model reply (code fences, prose and
    nested objects are fine) and repair it into a valid parameter set."""
    if not text or not text.strip():
        raise ValueError("empty reply")
    dec = json.JSONDecoder()
    found = None
    for i in [m.start() for m in re.finditer(r"\{", text)]:
        try:
            obj, _ = dec.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("style" in obj or any(k in obj for k in cp.CONTINUOUS)):
            found = obj  # keep the last one: models often restate the final answer at the end
    if found is None:
        raise ValueError("no JSON object in reply")
    d = found
    style = str(d.get("style", "sedan")).lower()
    style = style if style in cp.STYLES else "sedan"
    base = cp.mean_params(style).to_dict()
    for k in cp.CONTINUOUS:
        if k in d:
            try:
                base[k] = float(d[k])
            except (TypeError, ValueError):
                pass
    base["style"] = style
    return cp.clamp_to_ranges(cp.CarParams.from_dict(base))


def params_and_clips(d: dict):
    """Like parse_params on an already-parsed dict, but also report values clipped to the style's ranges,
    so the model can be told (a value it cannot reach usually means the body style is wrong)."""
    p = parse_params(json.dumps(d))
    clips = {}
    for k in cp.CONTINUOUS:
        if k in d:
            try:
                asked = float(d[k])
            except (TypeError, ValueError):
                continue
            got = getattr(p, k)
            if abs(asked - got) > 1e-6 * max(1.0, abs(asked)):
                clips[k] = {"asked": round(asked, 3), "allowed": round(got, 3)}
    return p, clips


def json_objects(text: str) -> list:
    """Every top-level JSON value in a reply that is (or contains) car-parameter objects, in order."""
    dec, out, i = json.JSONDecoder(), [], 0
    while i < len(text):
        j = min([k for k in (text.find("{", i), text.find("[", i)) if k >= 0], default=-1)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text[j:])
        except json.JSONDecodeError:
            i = j + 1
            continue
        items = obj if isinstance(obj, list) else [obj]
        items = [x for x in items if isinstance(x, dict) and ("style" in x or any(k in x for k in cp.CONTINUOUS))]
        if items:
            out.append(items)
        i = j + end
    return out


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from a .env file into the environment (never overrides variables already set)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


PREDICTORS = {"oracle": predict_oracle, "mean": predict_mean, "random": predict_random,
              "search": predict_search, "vlm": predict_vlm, "vlm_cam": predict_vlm_cam, "vlm_loop": predict_vlm_loop,
              "vlm_loop2": predict_vlm_loop2, "vlm_loop3": predict_vlm_loop3,
              "vlm_program": predict_vlm_program, "vlm_program_3d": predict_vlm_program_3d,
              "vlm_program_truth3d": predict_vlm_program_truth3d}


# ------------------------------------------------------------------ evaluation loop
def load_samples(data: Path, start: int, limit: int, view: int):
    ids = sorted(p.name for p in data.iterdir() if p.is_dir())[start:start + limit]
    for cid in ids:
        d = data / cid
        yield {"id": cid, "params": cp.CarParams.from_dict(json.loads((d / "params.json").read_text())),
               "ops": json.loads((d / "program.json").read_text()),
               "camera": Camera(**json.loads((d / f"camera_{view}.json").read_text())),
               "look": Look(**json.loads((d / f"look_{view}.json").read_text())),
               "image_path": d / f"view_{view}.png",
               "image": np.array(Image.open(d / f"view_{view}.png").convert("RGB")),
               "mask": np.array(Image.open(d / f"mask_{view}.png")) > 0}


def evaluate_ops(ops, sample):
    """Metrics for a program written directly by a model (no parameters, so no style / parameter error)."""
    if "gt_cache" not in sample:
        sample["gt_cache"] = metrics.GeometryCache(sample["ops"])
        sample["gt_render"] = render(sample["gt_cache"].model, sample["camera"], want_rgb=False)
    m = metrics.compare(ops, sample["gt_cache"], sample["camera"], sample["gt_render"],
                        gt_length=sample["params"].length)
    m.update({"style_correct": None, "param_err": None, "n_ops": len(ops),
              "has_wheels": any(o.get("part") == "wheels" for o in ops)})
    return ops, m


def evaluate(pred_params, sample):
    ops = pg.build_program(pred_params)
    if "gt_cache" not in sample:  # computed once per test car, shared by all predictors
        sample["gt_cache"] = metrics.GeometryCache(sample["ops"])
        sample["gt_render"] = render(sample["gt_cache"].model, sample["camera"], want_rgb=False)
    m = metrics.compare(ops, sample["gt_cache"], sample["camera"], sample["gt_render"],
                        gt_length=sample["params"].length)
    pe = cp.param_error(pred_params, sample["params"])
    m.update({"style_correct": pe["style_correct"], "param_err": pe["mean_norm_abs_err"]})
    return ops, m


def main():
    # .env first, so its values can serve as defaults (META_API_BASE, MODEL_API_MODEL, META_API_KEY)
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test")
    ap.add_argument("--predictor", nargs="+", default=["oracle", "mean", "random", "search"])
    ap.add_argument("--start", type=int, default=0, help="index of the first test car (to split long runs)")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--append", action="store_true", help="append to existing results instead of overwriting")
    ap.add_argument("--summarize-only", action="store_true", help="just rebuild summary.md from results/*.jsonl")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--search-init", type=int, default=6)
    ap.add_argument("--search-iters", type=int, default=60)
    ap.add_argument("--base-url", default=os.environ.get("META_API_BASE", "http://localhost:1234/v1"),
                    help="OpenAI-compatible endpoint; defaults to META_API_BASE from .env")
    ap.add_argument("--model", default=os.environ.get("MODEL_API_MODEL", "qwen2.5-vl-7b-instruct"),
                    help="model id; defaults to MODEL_API_MODEL from .env (e.g. muse-spark-1.3)")
    ap.add_argument("--reasoning-effort",
                    default=os.environ.get("REASONING_EFFORT") or ("low" if "api.meta.ai" in os.environ.get("META_API_BASE", "") else None),
                    help="low | medium | high (Meta Muse Spark, OpenAI o-series); omitted if unset")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=16384, help="room for reasoning models to think before the JSON")
    ap.add_argument("--loop-rounds", type=int, default=None,
                    help="revisions after the first guess (default: 3 for vlm_loop, 5 for vlm_loop2, "
                         "4 per style branch for vlm_loop3)")
    ap.add_argument("--loop-stop-iou", type=float, default=0.985, help="vlm_loop: stop early above this IoU")
    ap.add_argument("--no-program-example", dest="program_example", action="store_false",
                    help="vlm_program: do not show an example program (pure zero-shot program writing)")
    ap.add_argument("--track-3d", action="store_true",
                    help="vlm_program*: also record each round's voxel IoU (diagnostic only; never shown to the model)")
    ap.add_argument("--retries", type=int, default=1, help="extra attempts after an empty or unparsable reply")
    ap.add_argument("--resume", action="store_true", help="keep finished cars in results/, redo missing and failed ones")
    ap.add_argument("--api-key-env", default="META_API_KEY" if os.environ.get("META_API_KEY") else "OPENAI_API_KEY",
                    help="name of the environment variable holding the API key (e.g. FIREWORKS_API_KEY)")
    a = ap.parse_args()
    if any(p.startswith("vlm") for p in a.predictor) and not os.environ.get(a.api_key_env) and "localhost" not in a.base_url:
        sys.exit(f"No API key: put {a.api_key_env}=... in the .env file (or export it in your shell).")
    if a.loop_rounds is None:
        a.loop_rounds = (5 if "vlm_loop2" in a.predictor else 4 if ("vlm_loop3" in a.predictor or any(p.startswith("vlm_program") for p in a.predictor))
                         else 3)
    if a.summarize_only:
        print(summarize(Path(a.out)))
        return

    out = Path(a.out)
    (out / "compare").mkdir(parents=True, exist_ok=True)
    samples = list(load_samples(Path(a.data), a.start, a.limit, a.view))
    print(f"{len(samples)} test cars from {a.data}")
    for name in a.predictor:
        fn = PREDICTORS[name]
        rng = np.random.default_rng(a.seed + a.start)
        res_file = out / f"{name}.jsonl"
        if not (a.append or a.resume) and res_file.exists():
            res_file.unlink()
        done = set()
        if a.resume and res_file.exists():
            done = {r["id"] for r in map(json.loads, open(res_file)) if r.get("executed")}
            print(f"  [{name}] resuming: {len(done)} cars already done, re-running the rest")
        for s in samples:
            if s["id"] in done:
                continue
            row = {"id": s["id"], "gt_style": s["params"].style}
            t0 = time.time()
            try:
                pred, info = fn(s, rng, a)
                if isinstance(pred, dict) and "ops" in pred:      # the model wrote the program itself
                    prog = json.loads(json.dumps(pred["ops"]))     # clean copy to save; the kernel gets its own
                    ops, m = evaluate_ops(json.loads(json.dumps(prog)), s)
                    row.update(m, executed=True, pred_style=None, info=info, pred_program=prog)
                else:
                    ops, m = evaluate(pred, s)
                    row.update(m, executed=True, pred_style=pred.style, info=info, pred_params=pred.to_dict())
                if name != "oracle":
                    pr = render(sk.Model(ops), s["camera"], s["look"])
                    Image.fromarray(np.concatenate([s["image"], pr["rgb"]], 1)).save(out / "compare" / f"{s['id']}_{name}.png")
            except Exception as e:  # a prediction that cannot be parsed or built counts as a failure
                row.update(executed=False, error=f"{type(e).__name__}: {e}", info=getattr(e, "info", {}))
            row["seconds"] = round(time.time() - t0, 2)
            with open(res_file, "a") as f:  # written per car, so long runs can be split and merged
                f.write(json.dumps(row, default=float) + "\n")
            print(f"  [{name}] {s['id']}  " + (f"IoU {row['silhouette_iou']:.3f}  chamfer {row['chamfer_mm']:.0f} mm"
                                               if row.get("executed") else row.get("error", "")), flush=True)
    print("\n" + summarize(out))


def summarize(out: Path) -> str:
    """Aggregate every results/<predictor>.jsonl in `out` into summary.md / summary.json."""
    summary = {}
    for res_file in sorted(Path(out).glob("*.jsonl")):
        rows = list({r["id"]: r for r in map(json.loads, open(res_file))}.values())  # last result per car wins
        ok = [r for r in rows if r.get("executed")]

        def mean(k):
            v = [r[k] for r in ok if r.get(k) is not None and not math.isnan(r[k])]
            return float(np.mean(v)) if v else float("nan")
        summary[res_file.stem] = {
            "n": len(rows), "exec_rate": len(ok) / max(len(rows), 1),
            "silhouette_iou": mean("silhouette_iou"), "voxel_iou": mean("voxel_iou"),
            "chamfer_mm": mean("chamfer_mm"), "chamfer_pct_length": mean("chamfer_pct_length"),
            "depth_mae_mm": mean("depth_mae_mm"),
            "style_acc": (float(np.mean([r["style_correct"] for r in ok if r.get("style_correct") is not None]))
                          if any(r.get("style_correct") is not None for r in ok) else float("nan")),
            "param_err": mean("param_err"), "sec_per_car": float(np.mean([r.get("seconds", np.nan) for r in rows]))}
    order = ["oracle", "search", "vlm_program_truth3d", "vlm_program_3d", "vlm_program", "vlm_loop3", "vlm_loop2", "vlm_loop", "vlm_cam", "vlm", "mean", "random"]
    names = sorted(summary, key=lambda k: order.index(k) if k in order else 99)
    lines = ["| predictor | n | runs | silhouette IoU | voxel IoU | Chamfer (mm) | Chamfer (% length) | depth MAE (mm) "
             "| style acc | param err | s/car |", "|" + "---|" * 11]
    for k in names:
        v = summary[k]
        lines.append(f"| {k} | {v['n']} | {v['exec_rate']:.0%} | {v['silhouette_iou']:.3f} | {v['voxel_iou']:.3f} | "
                     f"{v['chamfer_mm']:.0f} | {v['chamfer_pct_length']:.2f} | {v['depth_mae_mm']:.0f} | "
                     f"{v['style_acc']:.0%} | {v['param_err']:.3f} | "
                     + ("-" if math.isnan(v['sec_per_car']) else f"{v['sec_per_car']:.1f}") + " |")
    (Path(out) / "summary.md").write_text("\n".join(lines) + "\n")
    (Path(out) / "summary.json").write_text(json.dumps(summary, indent=2))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
