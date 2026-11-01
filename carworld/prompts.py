"""Prompts shared by the training-data files and the baseline (so training and testing see the same text)."""
from __future__ import annotations

import json

from .params import CONTINUOUS, DEFAULT_RANGES, STYLE_RANGES, STYLES

FRAME = ("Units are millimetres. +X points to the front of the car, +Y to its left side, +Z up; "
         "the ground is z = 0 and the car is centred at x = y = 0.")

OP_LANGUAGE = """Each operation applies ONE tool solid to ONE part ("body" or "wheels") with ONE mode:
  {"name": str, "part": "body"|"wheels", "mode": "new"|"union"|"intersect"|"cut", "mirror": bool,
   "material": null|"glass"|"tire"|"rim"|"arch"|"light"|"grille"|"taillight"|"trim", "tool": TOOL, "comment": str}
TOOL is one of
  {"type": "prism", "plane": "XY"|"XZ"|"YZ", "points": [[u, v], ...], "extent": [w0, w1]}
      a closed polygon in the plane (XY -> (x, y), XZ -> (x, z), YZ -> (y, z)) extruded between w0 and w1
      along the remaining axis
  {"type": "cylinder", "axis": "X"|"Y"|"Z", "center": [u, v], "radius": r, "extent": [w0, w1]}
  {"type": "intersect", "children": [TOOL, TOOL, ...]}
mirror=true also applies the tool mirrored across the XZ plane (left side -> right side)."""

WORLD_MODEL_SYSTEM = (
    "You are a CAD World Model: a precise simulator of a solid-modelling kernel. Given the current model state "
    "and one modelling operation, predict the exact state after the operation runs. " + FRAME + "\n" + OP_LANGUAGE +
    "\nThe state is JSON: per part its volume in litres, bounding-box size and minimum corner in mm, the share of "
    "its surface covered by each material, and whether it is left-right symmetric; plus 'valid' and 'error'. "
    "If the kernel would reject the operation (malformed tool, editing a part that does not exist yet), the model is "
    "unchanged: return the previous state with the new step number, valid=false and a short error message. If every "
    "part ends up empty, valid=false with error 'empty solid'. Answer with the next state as JSON only."
)


def world_model_user(state: dict, action: dict) -> str:
    return "Current state:\n" + json.dumps(state) + "\n\nOperation:\n" + json.dumps(action)


PLANNER_SYSTEM = (
    "You are a CAD reverse-engineering assistant. Look at the photo of a car and write the CAD program that "
    "builds it. " + FRAME + "\n" + OP_LANGUAGE +
    "\nBuild the body from a side profile (XZ), a top-view outline (XY) and a cross-section (YZ), then cut wheel "
    "arches, glass, lights and grille, add bumpers, mirrors, exhaust and (sports cars) a rear wing, then add tyres and rims. Answer with the JSON list of operations only."
)


def planner_messages(program: list[dict]) -> list[dict]:
    """Chat-format SFT sample. The image goes in the user turn (training code attaches the pixels)."""
    return [{"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Write the CAD program for this car."}]},
            {"role": "assistant", "content": json.dumps(program)}]


def param_schema_text() -> str:
    lines = [f'  "style": one of {STYLES}']
    for k in CONTINUOUS:
        lo, hi = DEFAULT_RANGES[k]
        lines.append(f'  "{k}": number, typically {lo}-{hi}')
    notes = "; ".join(f"{s}: " + ", ".join(f"{k} {v[0]}-{v[1]}" for k, v in r.items()) for s, r in STYLE_RANGES.items() if r)
    return "{\n" + ",\n".join(lines) + "\n}\nStyle-specific ranges: " + notes


PARAMS_PROMPT = (
    "This image shows a car. Estimate the design parameters of a simple parametric car model that would "
    "reproduce it. " + FRAME + "\nParameters (JSON object):\n" + param_schema_text() +
    "\nMeaning: wheelbase_ratio = wheelbase / length; front_overhang_ratio = front overhang / (length - wheelbase); "
    "belt_ratio = beltline height above ground clearance as a fraction of (height - clearance); "
    "nose_ratio = bonnet front height as a fraction of (beltline - clearance); cowl_ratio = how far behind the front "
    "axle the windscreen starts, as a fraction of the wheelbase; rakes are degrees from vertical; roof_ratio = flat "
    "roof length / wheelbase; tumblehome = cabin top width / body width; nose_taper and tail_taper = width at the "
    "ends / body width (top view); arch_gap = wheel-arch radius minus tyre radius (mm); spoiler_height = rear wing "
    "height above the boot deck in mm (0 = no wing; only sports cars have one).\n"
    "Think briefly about the body style, proportions and wheel size, then output ONLY the JSON object."
)
