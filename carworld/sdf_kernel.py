"""A tiny, dependency-free (numpy only) solid-modelling kernel based on signed distance functions (SDFs).

It executes the JSON operation language from program.py exactly: extruded polygons, cylinders, intersections,
unions, cuts and XZ-mirroring. It is used to
  * produce the ground-truth next state after every operation (world-model training data),
  * render images, masks and depth maps (render.py),
  * score predictions (metrics.py).

The real CAD kernel (OpenCascade through CadQuery, see cq_kernel.py) builds the same solids as B-reps;
this one lets the whole pipeline run on any machine with only numpy installed.

An SDF returns, for every 3D point, the signed distance to the surface (negative inside).
Booleans:  union = min(a, b),  intersect = max(a, b),  cut = max(a, -b).
"""
from __future__ import annotations

import numpy as np

MATERIALS = ["paint", "glass", "tire", "rim", "arch", "light", "grille", "taillight", "trim"]
MAT_ID = {m: i for i, m in enumerate(MATERIALS)}
PART_DEFAULT_MAT = {"body": "paint", "wheels": "tire"}
PLANE_AXES = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}   # (u, v, normal) column indices
AXIS_PLANE = {"X": "YZ", "Y": "XZ", "Z": "XY"}


class ProgramError(ValueError):
    """Raised when a program is malformed or cannot be executed (e.g. cutting a part that does not exist)."""


# ---------------------------------------------------------------- 2D / 3D primitives
def sd_polygon(u: np.ndarray, v: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Exact signed distance to a closed polygon (any orientation, may be non-convex). Negative inside."""
    n = len(pts)
    d = (u - pts[0, 0]) ** 2 + (v - pts[0, 1]) ** 2
    s = np.ones_like(u)
    j = n - 1
    for i in range(n):
        ex, ey = pts[j, 0] - pts[i, 0], pts[j, 1] - pts[i, 1]
        if ex * ex + ey * ey == 0.0:   # repeated point: a zero-length edge adds nothing (and would divide by 0)
            j = i
            continue
        wx, wy = u - pts[i, 0], v - pts[i, 1]
        h = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
        bx, by = wx - ex * h, wy - ey * h
        d = np.minimum(d, bx * bx + by * by)
        c1 = v >= pts[i, 1]
        c2 = v < pts[j, 1]
        c3 = ex * wy > ey * wx
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        s = np.where(flip, -s, s)
        j = i
    return s * np.sqrt(d)


def _extrude(d2: np.ndarray, h: np.ndarray, w0: float, w1: float) -> np.ndarray:
    wz = np.abs(h - 0.5 * (w0 + w1)) - 0.5 * (w1 - w0)
    outside = np.sqrt(np.maximum(d2, 0) ** 2 + np.maximum(wz, 0) ** 2)
    return outside + np.minimum(np.maximum(d2, wz), 0)


def validate_tool(tool: dict, path: str = "tool") -> None:
    if not isinstance(tool, dict) or "type" not in tool:
        raise ProgramError(f"{path}: must be an object with a 'type'")
    t = tool["type"]
    if t == "intersect":
        ch = tool.get("children")
        if not isinstance(ch, list) or len(ch) < 2:
            raise ProgramError(f"{path}: intersect needs >= 2 children")
        for i, c in enumerate(ch):
            validate_tool(c, f"{path}.children[{i}]")
        return
    if t not in ("prism", "cylinder"):
        raise ProgramError(f"{path}: unknown tool type {t!r}")
    ext = tool.get("extent")
    if not (isinstance(ext, (list, tuple)) and len(ext) == 2 and float(ext[1]) > float(ext[0])):
        raise ProgramError(f"{path}: extent must be [w0, w1] with w1 > w0")
    if t == "prism":
        if tool.get("plane") not in PLANE_AXES:
            raise ProgramError(f"{path}: plane must be XY, XZ or YZ")
        pts = tool.get("points")
        if not isinstance(pts, list) or len(pts) < 3:
            raise ProgramError(f"{path}: prism needs >= 3 points")
    else:
        if tool.get("axis") not in AXIS_PLANE:
            raise ProgramError(f"{path}: axis must be X, Y or Z")
        if not float(tool.get("radius", 0)) > 0:
            raise ProgramError(f"{path}: radius must be > 0")
        if len(tool.get("center", [])) != 2:
            raise ProgramError(f"{path}: center must be [u, v]")


def eval_tool(tool: dict, P: np.ndarray) -> np.ndarray:
    """Signed distance from points P (N, 3) to a tool solid."""
    t = tool["type"]
    if t == "intersect":
        d = eval_tool(tool["children"][0], P)
        for c in tool["children"][1:]:
            d = np.maximum(d, eval_tool(c, P))
        return d
    plane = tool["plane"] if t == "prism" else AXIS_PLANE[tool["axis"]]
    iu, iv, ih = PLANE_AXES[plane]
    u, v, h = P[:, iu], P[:, iv], P[:, ih]
    if t == "prism":
        pts = tool.get("_np")
        if pts is None or pts.shape != (len(tool["points"]), 2) or not np.array_equal(pts, tool["points"]):
            pts = tool["_np"] = np.asarray(tool["points"], dtype=np.float64)   # (re)build if the outline changed
        d2 = sd_polygon(u, v, pts)
    else:
        cx, cz = tool["center"]
        d2 = np.sqrt((u - cx) ** 2 + (v - cz) ** 2) - tool["radius"]
    return _extrude(d2, h, *tool["extent"])


# ---------------------------------------------------------------- program execution
class Model:
    """An executed program. Parts are combined by union; each op edits one part."""

    def __init__(self, ops: list[dict]):
        self.ops = ops
        for i, op in enumerate(ops):
            self.check_op(op, i, [o["part"] for o in ops[:i] if o["mode"] in ("new", "union")])

    @staticmethod
    def check_op(op: dict, i: int, existing_parts: list[str]) -> None:
        for key in ("part", "mode", "tool"):
            if key not in op:
                raise ProgramError(f"op {i}: missing '{key}'")
        if op["mode"] not in ("new", "union", "intersect", "cut"):
            raise ProgramError(f"op {i}: unknown mode {op['mode']!r}")
        if op["mode"] in ("intersect", "cut") and op["part"] not in existing_parts:
            raise ProgramError(f"op {i}: cannot {op['mode']} part {op['part']!r} before it exists")
        if op.get("material") not in (None, *MATERIALS):
            raise ProgramError(f"op {i}: unknown material {op.get('material')!r}")
        validate_tool(op["tool"], f"op {i}.tool")

    def iter_eval(self, P: np.ndarray):
        """Yield (step_index, part_d, part_mat) after every op. part_d/part_mat are dicts of arrays."""
        part_d: dict[str, np.ndarray] = {}
        part_m: dict[str, np.ndarray] = {}
        Pm = None
        for i, op in enumerate(self.ops):
            Q = P
            if op.get("mirror"):
                if Pm is None:
                    Pm = P.copy()
                    Pm[:, 1] = np.abs(Pm[:, 1])
                Q = Pm
            f = eval_tool(op["tool"], Q)
            part, mode = op["part"], op["mode"]
            mat = MAT_ID[op.get("material") or PART_DEFAULT_MAT.get(part, "paint")]
            if mode == "new" or (mode == "union" and part not in part_d):
                part_d[part] = f
                part_m[part] = np.full(len(P), mat, dtype=np.int8)
            elif mode == "union":
                m = f < part_d[part]
                part_d[part] = np.where(m, f, part_d[part])
                part_m[part] = np.where(m, mat, part_m[part]).astype(np.int8)
            elif mode == "intersect":
                m = f > part_d[part]
                part_d[part] = np.where(m, f, part_d[part])
                part_m[part] = np.where(m, mat, part_m[part]).astype(np.int8)
            else:  # cut
                g = -f
                m = g > part_d[part]
                part_d[part] = np.where(m, g, part_d[part])
                part_m[part] = np.where(m, mat, part_m[part]).astype(np.int8)
            yield i, part_d, part_m

    def eval(self, P: np.ndarray, upto: int | None = None):
        """Scene SDF (union of parts) and material id per point, after op index `upto` (default: all)."""
        last = (len(self.ops) - 1) if upto is None else upto
        for i, part_d, part_m in self.iter_eval(P):
            if i == last:
                break
        names = list(part_d)
        D = np.stack([part_d[n] for n in names])
        k = np.argmin(D, axis=0)
        d = D[k, np.arange(len(P))]
        M = np.stack([part_m[n] for n in names])
        return d, M[k, np.arange(len(P))]

    def sdf(self, P: np.ndarray) -> np.ndarray:
        return self.eval(P)[0]


# ---------------------------------------------------------------- state summary (what the world model predicts)
GRID_BOUNDS = np.array([[-2800.0, 2800.0], [-1150.0, 1150.0], [-40.0, 2000.0]])


def make_grid(res: float = 30.0):
    axes = [np.arange(lo + res / 2, hi, res) for lo, hi in GRID_BOUNDS]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1), [len(a) for a in axes], res


def _summary(op, part_d, part_m, P, res) -> dict:
    vox = (res ** 3) / 1e6  # litres per voxel
    s = {"step": op["step"] if op else 0, "valid": True, "parts": {}, "error": None}
    all_occ = np.zeros(len(P), dtype=bool)
    for name, d in part_d.items():
        occ = d < 0
        all_occ |= occ
        info = {"volume_l": round(float(occ.sum() * vox), 1)}
        if occ.any():
            lo, hi = P[occ].min(0) - res / 2, P[occ].max(0) + res / 2
            info["bbox_mm"] = [int(round(v)) for v in (hi - lo)]
            info["min_mm"] = [int(round(v)) for v in lo]
            surf = np.abs(d) < res * 0.87
            counts = np.bincount(part_m[name][surf], minlength=len(MATERIALS))
            tot = max(counts.sum(), 1)
            info["surface_materials"] = {MATERIALS[i]: round(float(c / tot), 3) for i, c in enumerate(counts) if c / tot >= 0.005}
            left = occ & (P[:, 1] > 0)
            right = occ & (P[:, 1] < 0)
            info["left_right_symmetric"] = bool(abs(int(left.sum()) - int(right.sum())) <= max(3, 0.01 * occ.sum()))
        else:
            info["bbox_mm"] = None
        s["parts"][name] = info
    if not all_occ.any():
        s["valid"] = False
        s["error"] = "empty solid"
    return s


def trajectory(ops: list[dict], res: float = 30.0) -> list[dict]:
    """Ground-truth (state, action, next_state) records for every op, for world-model training."""
    P, _, res = make_grid(res)
    model = Model(ops)
    prev = {"step": 0, "valid": True, "parts": {}, "error": None}
    recs = []
    for i, part_d, part_m in model.iter_eval(P):
        op = ops[i]
        nxt = _summary(op, part_d, part_m, P, res)
        recs.append({"state": prev, "action": public_op(op), "next_state": nxt})
        prev = nxt
    return recs


def apply_step(prefix_ops: list[dict], op: dict, state: dict, res: float = 40.0) -> dict:
    """Ground-truth next state for applying `op` after `prefix_ops` (whose summary is `state`).
    An operation the kernel rejects leaves the model unchanged: the previous state, with valid=false and the error."""
    existing = [o["part"] for o in prefix_ops if o["mode"] in ("new", "union")]
    try:
        Model.check_op(op, len(prefix_ops), existing)
    except ProgramError as e:
        return {**state, "step": op["step"], "valid": False, "error": str(e).split(": ", 1)[-1]}
    P, _, res = make_grid(res)
    model = Model(prefix_ops + [op])
    *_, (i, part_d, part_m) = model.iter_eval(P)
    return _summary(op, part_d, part_m, P, res)


def public_op(op: dict) -> dict:
    """Op without private cached arrays."""
    def clean(t):
        t = {k: v for k, v in t.items() if not k.startswith("_")}
        if t.get("type") == "intersect":
            t["children"] = [clean(c) for c in t["children"]]
        return t
    o = dict(op)
    o["tool"] = clean(op["tool"])
    return o


def occupancy(ops: list[dict], res: float = 30.0) -> np.ndarray:
    P, shape, _ = make_grid(res)
    return (Model(ops).sdf(P) < 0).reshape(shape)


def surface_points(ops: list[dict], res: float = 20.0, max_pts: int = 4000, seed: int = 0) -> np.ndarray:
    """Points lying (within res/2) on the surface, for Chamfer distance."""
    P, _, _ = make_grid(res)
    d = Model(ops).sdf(P)
    pts = P[np.abs(d) < res * 0.5]
    if len(pts) > max_pts:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_pts, replace=False)]
    return pts
