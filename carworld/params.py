"""Car design parameters: sampling, validation and (de)serialisation.

Coordinate frame used everywhere in this project (units: millimetres):
    +X  points to the FRONT of the car
    +Y  points to the car's LEFT side
    +Z  points UP, ground plane is z = 0
The car is centred at x = 0, y = 0.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields

import numpy as np

STYLES = ["sedan", "hatchback", "wagon", "suv", "coupe", "sports"]

# v1: hand-set ranges.  v2: spreads widened toward PartVerse's 10th-90th percentiles of height/length, wheelbase/length
# and wheel radius/length (336 real 3D car models, see paper section 7.4); "sports" style added; width unchanged
# (PartVerse's wider cars may be stylised assets); bumpers, mirrors, exhaust and an optional spoiler added.
GENERATOR_VERSION = "v2.1"   # v2.1: windscreen / rear-glass insets pointed into the body (420 mm pocket); fixed

# (low, high) sampling ranges per style. Anything not listed per style falls back to DEFAULT_RANGES.
DEFAULT_RANGES = {
    "length": (4300, 5000),
    "width": (1720, 1900),
    "height": (1380, 1530),
    "wheelbase_ratio": (0.55, 0.64),      # wheelbase / length
    "front_overhang_ratio": (0.40, 0.54),  # front overhang / (length - wheelbase)
    "wheel_radius": (305, 375),
    "tire_width": (195, 255),
    "ground_clearance": (115, 165),
    "belt_ratio": (0.50, 0.58),           # beltline height above clearance, as a fraction of (height - clearance)
    "nose_ratio": (0.70, 0.90),           # nose height above clearance, as a fraction of (belt - clearance)
    "cowl_ratio": (0.08, 0.24),           # windscreen base sits this fraction of the wheelbase behind the front axle
    "windshield_rake": (55, 65),          # degrees from vertical
    "roof_ratio": (0.28, 0.44),           # flat roof length / wheelbase
    "rear_rake": (55, 67),                # rear glass, degrees from vertical
    "tumblehome": (0.76, 0.86),           # cabin top half-width / body half-width
    "nose_taper": (0.80, 0.90),           # half-width at the nose / body half-width (top view)
    "tail_taper": (0.86, 0.95),
    "arch_gap": (30, 60),                 # wheel-arch radius minus tyre radius
    "spoiler_height": (0, 0),             # rear wing height above the boot deck (mm); 0 = none (sports cars only)
}

STYLE_RANGES = {
    "sedan": {},
    "coupe": {
        "length": (4200, 4800), "height": (1280, 1400), "ground_clearance": (105, 140),
        "windshield_rake": (60, 67), "roof_ratio": (0.22, 0.32), "rear_rake": (64, 72),
        "belt_ratio": (0.55, 0.62), "nose_ratio": (0.78, 0.92), "wheelbase_ratio": (0.55, 0.62),
    },
    "sports": {
        "length": (4200, 4700), "height": (1150, 1320), "ground_clearance": (95, 130),
        "windshield_rake": (63, 70), "roof_ratio": (0.18, 0.28), "rear_rake": (68, 78),
        "belt_ratio": (0.62, 0.72), "nose_ratio": (0.80, 0.95), "cowl_ratio": (0.15, 0.30),
        "wheel_radius": (305, 345), "tire_width": (235, 305), "wheelbase_ratio": (0.56, 0.63),
        "tumblehome": (0.72, 0.82), "spoiler_height": (0, 140),
    },
    "hatchback": {
        "length": (3500, 4400), "height": (1420, 1660), "wheelbase_ratio": (0.60, 0.72),
        "wheel_radius": (290, 360), "roof_ratio": (0.38, 0.52),
    },
    "wagon": {
        "length": (4400, 5100), "height": (1420, 1560), "roof_ratio": (0.55, 0.68),
    },
    "suv": {
        "length": (4000, 4950), "height": (1620, 1950), "ground_clearance": (180, 250),
        "wheel_radius": (350, 440), "tire_width": (225, 285), "belt_ratio": (0.52, 0.60),
        "windshield_rake": (50, 60), "roof_ratio": (0.55, 0.70), "width": (1820, 1960),
        "tumblehome": (0.82, 0.90), "wheelbase_ratio": (0.55, 0.68),
    },
}

CONTINUOUS = list(DEFAULT_RANGES.keys())


@dataclass
class CarParams:
    style: str
    length: float
    width: float
    height: float
    wheelbase_ratio: float
    front_overhang_ratio: float
    wheel_radius: float
    tire_width: float
    ground_clearance: float
    belt_ratio: float
    nose_ratio: float
    cowl_ratio: float
    windshield_rake: float
    roof_ratio: float
    rear_rake: float
    tumblehome: float
    nose_taper: float
    tail_taper: float
    arch_gap: float
    spoiler_height: float = 0.0

    # ---- derived quantities -------------------------------------------------
    @property
    def wheelbase(self) -> float:
        return self.length * self.wheelbase_ratio

    @property
    def front_overhang(self) -> float:
        return (self.length - self.wheelbase) * self.front_overhang_ratio

    # ---- helpers -------------------------------------------------------------
    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "CarParams":
        names = {f.name for f in fields(cls)}
        clean = {k: v for k, v in d.items() if k in names}
        clean.setdefault("spoiler_height", 0.0)   # v1 files have no spoiler
        missing = names - clean.keys()
        if missing:
            raise ValueError(f"missing parameters: {sorted(missing)}")
        if clean["style"] not in STYLES:
            raise ValueError(f"unknown style {clean['style']!r}; expected one of {STYLES}")
        for k in CONTINUOUS:
            clean[k] = float(clean[k])
        return cls(**clean)

    def to_json(self) -> str:
        return json.dumps({**self.to_dict(), "generator_version": GENERATOR_VERSION}, indent=2)


def ranges_for(style: str) -> dict:
    r = dict(DEFAULT_RANGES)
    r.update(STYLE_RANGES[style])
    return r


def sample(rng: np.random.Generator, style: str | None = None) -> CarParams:
    """Sample a random, valid car. Resamples until the geometry constraints hold."""
    s = style or STYLES[int(rng.integers(len(STYLES)))]   # pick the style once, so rejections don't skew style frequencies
    for _ in range(500):
        r = ranges_for(s)
        vals = {k: float(rng.uniform(*r[k])) for k in CONTINUOUS}
        if s == "sports" and rng.random() < 0.4:
            vals["spoiler_height"] = 0.0          # many sports cars have no rear wing
        p = CarParams(style=s, **vals)
        if not validate(p):
            return p
    raise RuntimeError("could not sample a valid car")


def mean_params(style: str) -> CarParams:
    r = ranges_for(style)
    return CarParams(style=style, **{k: (r[k][0] + r[k][1]) / 2 for k in CONTINUOUS})


def clamp_to_ranges(p: CarParams) -> CarParams:
    """Clip every continuous value into its style's range (used to repair model predictions)."""
    r = ranges_for(p.style)
    d = p.to_dict()
    for k in CONTINUOUS:
        lo, hi = r[k]
        d[k] = float(min(max(d[k], lo), hi))
    return CarParams.from_dict(d)


def validate(p: CarParams) -> list[str]:
    """Return a list of problems (empty list = valid). Checks the derived side profile is sane."""
    from .program import side_profile_keypoints  # local import to avoid a cycle

    errs = []
    if 2 * p.wheel_radius >= p.height * 0.62:
        errs.append("wheels too large for body height")
    k = side_profile_keypoints(p)
    xr = -p.length / 2
    if k["x_roof_rear"] <= xr + 150:
        errs.append("roof runs past the rear of the car")
    if k["x_roof_rear"] >= k["x_ws_top"] - 150:
        errs.append("roof too short")
    if k["x_cowl"] >= p.length / 2 - 300:
        errs.append("hood too short")
    if p.style in ("sedan", "coupe") and k["x_deck"] <= xr + 150:
        errs.append("no room for a boot deck")
    if k["x_rear_axle"] - p.wheel_radius - p.arch_gap <= xr + 120:
        errs.append("rear wheel arch hits the rear bumper")
    if k["x_front_axle"] + p.wheel_radius + p.arch_gap >= p.length / 2 - 120:
        errs.append("front wheel arch hits the front bumper")
    if 2 * p.wheel_radius + p.arch_gap >= k["z_belt"] + 0.5 * (p.height - k["z_belt"]):
        errs.append("wheel arches reach into the windows")
    # the arch must stay under the body surface at each axle, or the tyres poke through the bonnet / boot
    from .program import side_profile, _profile_top_z
    top = side_profile(p)
    for axle in ("x_front_axle", "x_rear_axle"):
        if 2 * p.wheel_radius + p.arch_gap + 25 >= _profile_top_z(top, k[axle]):
            errs.append("wheels poke through the bonnet or boot")
            break
    return errs


def param_error(pred: CarParams, gt: CarParams) -> dict:
    """Mean absolute error of each continuous parameter, normalised by its sampling range."""
    r = ranges_for(gt.style)
    per = {}
    for k in CONTINUOUS:
        lo, hi = DEFAULT_RANGES[k] if k not in r else r[k]
        span = (hi - lo) if hi > lo else 100.0   # zero-width range (e.g. no spoiler): judge against 100 mm
        per[k] = abs(getattr(pred, k) - getattr(gt, k)) / span
    return {"style_correct": pred.style == gt.style, "mean_norm_abs_err": float(np.mean(list(per.values()))), "per_param": per}
