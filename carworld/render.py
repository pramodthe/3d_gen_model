"""Sphere-tracing renderer for sdf_kernel models: RGB image, car mask, depth map. numpy only.

Domain randomisation (random paint, sky, ground, sun, camera) makes renders vary the way photos do,
so a model trained on them has a better chance on real photographs.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .sdf_kernel import MAT_ID, Model

PAINTS = [(0.70, 0.08, 0.08), (0.08, 0.20, 0.55), (0.85, 0.85, 0.86), (0.10, 0.10, 0.11), (0.55, 0.57, 0.60),
          (0.12, 0.35, 0.20), (0.90, 0.60, 0.10), (0.35, 0.20, 0.45), (0.80, 0.80, 0.70), (0.20, 0.45, 0.70)]


@dataclass
class Camera:
    azimuth: float = 35.0      # degrees; 0 = looking at the front of the car, 90 = left side
    elevation: float = 12.0    # degrees above horizontal
    distance: float = 9000.0   # mm from target
    fov: float = 32.0          # vertical field of view, degrees
    target_z: float = 650.0    # mm, height of the look-at point
    width: int = 256
    height: int = 192

    def to_dict(self):
        return asdict(self)

    def basis(self):
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        target = np.array([0.0, 0.0, self.target_z])
        eye = target + self.distance * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        fwd = (target - eye) / np.linalg.norm(target - eye)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return eye, fwd, right, up

    def rays(self):
        eye, fwd, right, up = self.basis()
        th = math.tan(math.radians(self.fov) / 2)
        aspect = self.width / self.height
        ys, xs = np.mgrid[0:self.height, 0:self.width]
        u = ((xs + 0.5) / self.width * 2 - 1) * th * aspect
        v = (1 - (ys + 0.5) / self.height * 2) * th
        D = fwd[None, :] + u.reshape(-1, 1) * right[None, :] + v.reshape(-1, 1) * up[None, :]
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        return eye, D


@dataclass
class Look:
    paint: tuple = (0.70, 0.08, 0.08)
    sky_top: tuple = (0.55, 0.70, 0.90)
    sky_bottom: tuple = (0.88, 0.90, 0.92)
    ground: tuple = (0.55, 0.55, 0.53)
    sun_az: float = 60.0
    sun_el: float = 45.0

    def to_dict(self):
        return asdict(self)


def random_camera(rng, params, width=256, height=192) -> Camera:
    side = rng.choice([-1, 1])
    az = float(rng.choice([rng.uniform(20, 70), rng.uniform(110, 160), rng.uniform(80, 100)], p=[0.45, 0.35, 0.20])) * side
    el = float(rng.uniform(4, 22))
    fov = float(rng.uniform(26, 38))
    # distance so that the car fills most of the frame width
    a = math.radians(az)
    span = abs(params.length * math.sin(a)) + abs(params.width * math.cos(a))
    half_h = math.atan(math.tan(math.radians(fov) / 2) * width / height)
    dist = 0.5 * span / math.tan(half_h) / float(rng.uniform(0.72, 0.88)) + 0.5 * params.length * abs(math.cos(a))
    return Camera(az, el, dist, fov, params.height * 0.42, width, height)


def random_look(rng) -> Look:
    def jitter(c, s=0.06):
        return tuple(float(np.clip(x + rng.uniform(-s, s), 0, 1)) for x in c)
    sky_top = jitter((rng.uniform(0.35, 0.75), rng.uniform(0.5, 0.8), rng.uniform(0.7, 0.98)), 0.1)
    return Look(paint=jitter(PAINTS[int(rng.integers(len(PAINTS)))], 0.05), sky_top=sky_top,
                sky_bottom=jitter((0.85, 0.87, 0.88), 0.08), ground=jitter((rng.uniform(0.3, 0.7),) * 3, 0.06),
                sun_az=float(rng.uniform(0, 360)), sun_el=float(rng.uniform(25, 70)))


def _aabb(model_ops, pad=150.0):
    """Loose bounding box of the scene from the first op's extents (enough for culling rays)."""
    return np.array([[-2800.0, -1150.0, -10.0], [2800.0, 1150.0, 2100.0]])


def render(model: Model, cam: Camera, look: Look | None = None, max_steps: int = 160, want_rgb: bool = True,
           bounds: np.ndarray | None = None):
    """Returns dict with rgb (H,W,3 uint8 or None), mask (H,W bool), depth (H,W float32 mm, 0 = background),
    material (H,W int8, -1 = background)."""
    look = look or Look()
    eye, D = cam.rays()
    n = len(D)
    lo, hi = bounds if bounds is not None else _aabb(model.ops)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / D
        t0 = (lo[None] - eye[None]) * inv
        t1 = (hi[None] - eye[None]) * inv
    tmin = np.nanmax(np.minimum(t0, t1), axis=1)
    tmax = np.nanmin(np.maximum(t0, t1), axis=1)
    tmin = np.maximum(tmin, 0)
    # ground plane z = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(D[:, 2] < -1e-6, -eye[2] / D[:, 2], np.inf)
    tmax = np.minimum(tmax, tg)

    t = tmin.copy()
    hit = np.zeros(n, dtype=bool)
    active = np.where(tmin < tmax)[0]
    for _ in range(max_steps):
        if len(active) == 0:
            break
        P = eye[None] + D[active] * t[active, None]
        d = model.sdf(P)
        eps = 0.8 + 0.00025 * t[active]
        h = d < eps
        hit[active[h]] = True
        t[active] += np.maximum(d, 0.5) * 0.92
        keep = (~h) & (t[active] < tmax[active])
        active = active[keep]

    depth = np.where(hit, t, 0.0).astype(np.float32)
    mat = np.full(n, -1, dtype=np.int8)
    rgb = None
    hp = np.where(hit)[0]
    if len(hp):
        Ph = eye[None] + D[hp] * t[hp, None]
        _, mh = model.eval(Ph)
        mat[hp] = mh
    if want_rgb:
        rgb = np.zeros((n, 3))
        # sky
        sky_t = np.clip(D[:, 2] * 2.5 + 0.35, 0, 1)[:, None]
        rgb[:] = np.array(look.sky_bottom) * (1 - sky_t) + np.array(look.sky_top) * sky_t
        sun = np.array([math.cos(math.radians(look.sun_el)) * math.cos(math.radians(look.sun_az)),
                        math.cos(math.radians(look.sun_el)) * math.sin(math.radians(look.sun_az)),
                        math.sin(math.radians(look.sun_el))])
        # ground with contact shadow / ambient occlusion from distance to the car
        gp = np.where(np.isfinite(tg) & ~hit & (tg < 1e6))[0]
        if len(gp):
            G = eye[None] + D[gp] * tg[gp, None]
            dg = model.sdf(G + np.array([0, 0, 40.0]))
            ao = 0.35 + 0.65 * np.clip(dg / 900.0, 0, 1)
            fog = np.clip((tg[gp] - cam.distance) / (cam.distance * 3), 0, 1)[:, None]
            col = np.array(look.ground)[None] * ao[:, None]
            rgb[gp] = col * (1 - fog) + np.array(look.sky_bottom)[None] * fog
        if len(hp):
            e = 2.0
            N = np.stack([model.sdf(Ph + np.array([e, 0, 0])) - model.sdf(Ph - np.array([e, 0, 0])),
                          model.sdf(Ph + np.array([0, e, 0])) - model.sdf(Ph - np.array([0, e, 0])),
                          model.sdf(Ph + np.array([0, 0, e])) - model.sdf(Ph - np.array([0, 0, e]))], 1)
            N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
            base = np.zeros((len(hp), 3))
            table = {"paint": look.paint, "glass": (0.10, 0.12, 0.15), "tire": (0.05, 0.05, 0.05),
                     "rim": (0.62, 0.63, 0.65), "arch": (0.02, 0.02, 0.02), "light": (0.88, 0.89, 0.86),
                     "grille": (0.06, 0.06, 0.07), "taillight": (0.65, 0.05, 0.05),
                     "trim": (0.11, 0.11, 0.12)}
            for name, c in table.items():
                base[mh == MAT_ID[name]] = c
            diff = np.clip(N @ sun, 0, 1)[:, None]
            amb = (0.35 + 0.25 * np.clip(N[:, 2], -1, 1))[:, None]
            col = base * (amb + 0.75 * diff)
            refl = D[hp] - 2 * (np.sum(D[hp] * N, 1, keepdims=True)) * N
            spec = np.clip(refl @ sun, 0, 1) ** 40
            shiny = np.isin(mh, [MAT_ID["paint"], MAT_ID["glass"], MAT_ID["rim"]])[:, None]
            sky_ref = np.clip(refl[:, 2] * 2 + 0.3, 0, 1)[:, None]
            env = np.array(look.sky_bottom) * (1 - sky_ref) + np.array(look.sky_top) * sky_ref
            gl = (mh == MAT_ID["glass"])[:, None]
            col = np.where(gl, col + 0.45 * env * sky_ref, col + 0.12 * env * shiny)
            col += 0.6 * spec[:, None] * shiny
            rgb[hp] = col
        rgb = (np.clip(rgb, 0, 1) ** (1 / 1.1) * 255).astype(np.uint8).reshape(cam.height, cam.width, 3)
    car = np.isin(mat, [MAT_ID[m] for m in MAT_ID]) & (mat >= 0)
    return {"rgb": rgb, "mask": car.reshape(cam.height, cam.width),
            "depth": depth.reshape(cam.height, cam.width), "material": mat.reshape(cam.height, cam.width)}
