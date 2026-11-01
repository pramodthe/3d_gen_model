"""Run a generated CadQuery program with the real OpenCascade kernel (needs `pip install cadquery`).

Returns the B-rep state the world model should eventually predict (faces, edges, volume, validity) and can
export STEP / STL files. Not used by the numpy pipeline; see scripts/verify_cadquery.py.
"""
from __future__ import annotations

from pathlib import Path


def run_program(source: str) -> dict:
    """Execute CadQuery source that defines `parts`. Returns {"parts": {name: Workplane}, "state": {...}}."""
    import cadquery as cq  # noqa: F401  (imported so the error is clear if it is missing)

    ns: dict = {}
    try:
        exec(compile(source, "<cad program>", "exec"), ns)
    except Exception as e:  # the program itself failed: a valuable world-model training signal
        return {"parts": {}, "state": {"valid": False, "error": f"{type(e).__name__}: {e}", "parts": {}}}
    parts = ns.get("parts", {})
    state = {"valid": True, "error": None, "parts": {}}
    for name, wp in parts.items():
        shape = _shape(wp)
        bb = shape.BoundingBox()
        state["parts"][name] = {
            "volume_l": round(shape.Volume() / 1e6, 1),
            "bbox_mm": [round(bb.xlen), round(bb.ylen), round(bb.zlen)],
            "min_mm": [round(bb.xmin), round(bb.ymin), round(bb.zmin)],
            "solids": len(shape.Solids()),
            "faces": len(shape.Faces()),
            "edges": len(shape.Edges()),
            "brep_valid": bool(shape.isValid()),
        }
        if not shape.isValid():
            state["valid"] = False
    return {"parts": parts, "state": state}


def _shape(wp):
    import cadquery as cq

    vals = [v for v in wp.vals() if isinstance(v, cq.Shape)]
    return vals[0] if len(vals) == 1 else cq.Compound.makeCompound(vals)


def export(parts: dict, out_dir: str | Path, stem: str = "car") -> list[Path]:
    """Write <stem>.step (all parts) and <stem>.stl."""
    import cadquery as cq

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compound = cq.Compound.makeCompound([_shape(wp) for wp in parts.values()])
    paths = [out_dir / f"{stem}.step", out_dir / f"{stem}.stl"]
    cq.exporters.export(compound, str(paths[0]))
    cq.exporters.export(compound, str(paths[1]), tolerance=1.0, angularTolerance=0.2)
    return paths
