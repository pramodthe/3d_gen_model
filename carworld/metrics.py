"""Scores comparing a predicted car program with the ground truth."""
from __future__ import annotations

import numpy as np

from . import sdf_kernel as sk
from .render import Camera, render


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def depth_error(da: np.ndarray, db: np.ndarray) -> float:
    """Mean absolute depth difference (mm) over pixels where both depth maps hit the car."""
    both = (da > 0) & (db > 0)
    return float(np.abs(da[both] - db[both]).mean()) if both.any() else float("nan")


class GeometryCache:
    """Everything about one solid that the metrics need: an occupancy grid, surface samples, its SDF."""

    def __init__(self, ops: list[dict], res: float = 30.0, max_pts: int = 6000, seed: int = 0):
        self.model = sk.Model(ops)
        P, shape, _ = sk.make_grid(res)
        d = self.model.sdf(P)
        self.occ = d < 0
        surf = P[np.abs(d) < res * 0.5]
        if len(surf) > max_pts:
            surf = surf[np.random.default_rng(seed).choice(len(surf), max_pts, replace=False)]
        # project the samples onto the surface along the SDF gradient (removes grid quantisation)
        self.surface = self._project(surf)

    def _project(self, X, e=2.0):
        for _ in range(2):
            d = self.model.sdf(X)
            g = np.stack([self.model.sdf(X + np.array([e, 0, 0])) - self.model.sdf(X - np.array([e, 0, 0])),
                          self.model.sdf(X + np.array([0, e, 0])) - self.model.sdf(X - np.array([0, e, 0])),
                          self.model.sdf(X + np.array([0, 0, e])) - self.model.sdf(X - np.array([0, 0, e]))], 1)
            g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
            X = X - d[:, None] * g
        return X


def chamfer(a: GeometryCache, b: GeometryCache) -> float:
    """Symmetric mean surface distance (mm). Uses each solid's exact SDF instead of nearest-neighbour search:
    the distance from a point on surface A to surface B is |sdf_B(point)|."""
    if len(a.surface) == 0 or len(b.surface) == 0:
        return float("nan")
    return float(0.5 * (np.abs(b.model.sdf(a.surface)).mean() + np.abs(a.model.sdf(b.surface)).mean()))


def compare(pred_ops, gt: GeometryCache, cam: Camera, gt_render: dict, gt_length: float | None = None) -> dict:
    """All geometry metrics for one prediction against a cached ground truth."""
    pg = GeometryCache(pred_ops, max_pts=len(gt.surface) or 6000)
    pr = render(pg.model, cam, want_rgb=False)
    cd = chamfer(pg, gt)
    out = {"silhouette_iou": mask_iou(pr["mask"], gt_render["mask"]),
           "depth_mae_mm": depth_error(pr["depth"], gt_render["depth"]),
           "voxel_iou": mask_iou(pg.occ, gt.occ),
           "chamfer_mm": cd}
    if gt_length:
        out["chamfer_pct_length"] = 100.0 * cd / gt_length
    return out
