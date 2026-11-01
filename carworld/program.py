"""Turn car parameters into a CAD *program*: an ordered list of modelling operations.

The program is the thing the planner model must learn to write. It exists in two equivalent forms:

1. A small JSON operation language (the "DSL") executed by our numpy kernel (sdf_kernel.py).
2. A CadQuery Python script (to_cadquery) executed by the real OpenCascade kernel (cq_kernel.py).

Every operation applies ONE tool solid to ONE part with ONE boolean mode:

    {"name": "arch_front", "part": "body", "mode": "cut", "mirror": true, "material": "arch",
     "tool": {"type": "cylinder", "axis": "Y", "center": [x, z], "radius": r, "extent": [y0, y1]},
     "comment": "..."}

Tool types:
    prism     - a closed 2D polygon on plane XY / XZ / YZ, extruded along the remaining axis
                between extent[0] and extent[1].   Plane coords: XY -> (x, y), XZ -> (x, z), YZ -> (y, z)
    cylinder  - a circle on the plane normal to `axis`, extruded along it (same plane coords as prism)
    intersect - the intersection of its `children` tools
Modes: new | union | intersect | cut.   mirror=True also applies the tool mirrored across the XZ plane (y -> -y).
"""
from __future__ import annotations

import math

from .params import CarParams

PLANE_OF_AXIS = {"X": "YZ", "Y": "XZ", "Z": "XY"}


# --------------------------------------------------------------------------------------------
# Geometry keypoints of the side profile (shared by validation, the program and documentation)
# --------------------------------------------------------------------------------------------
def side_profile_keypoints(p: CarParams) -> dict:
    L, H, clr = p.length, p.height, p.ground_clearance
    xf, xr = L / 2, -L / 2
    x_fa = xf - p.front_overhang
    x_ra = x_fa - p.wheelbase
    z_belt = clr + p.belt_ratio * (H - clr)
    z_nose = clr + p.nose_ratio * (z_belt - clr)
    z_cowl = z_belt + 0.06 * (H - z_belt)
    x_cowl = x_fa - p.cowl_ratio * p.wheelbase
    x_ws_top = x_cowl - (H - z_cowl) * math.tan(math.radians(p.windshield_rake))
    x_roof_rear = x_ws_top - p.roof_ratio * p.wheelbase
    z_deck = z_belt + 0.10 * (H - z_belt)
    x_deck = x_roof_rear - (H - z_deck) * math.tan(math.radians(p.rear_rake))
    return dict(xf=xf, xr=xr, x_front_axle=x_fa, x_rear_axle=x_ra, z_belt=z_belt, z_nose=z_nose,
                z_cowl=z_cowl, x_cowl=x_cowl, x_ws_top=x_ws_top, x_roof_rear=x_roof_rear,
                z_deck=z_deck, x_deck=x_deck)


def side_profile(p: CarParams) -> list[list[float]]:
    """Closed side silhouette in the XZ plane, counter-clockwise, front at +x."""
    k = side_profile_keypoints(p)
    H, clr = p.height, p.ground_clearance
    xf, xr = k["xf"], k["xr"]
    zb, zn = k["z_belt"], k["z_nose"]
    pts = [
        (xr + 90, clr),                              # rear underside
        (xf - 160, clr),                             # front underside / chin
        (xf - 12, clr + 0.45 * (zn - clr)),          # front bumper
        (xf, zn - 85),                               # bumper top corner
        (xf - 90, zn),                               # nose top / hood leading edge
        (k["x_cowl"], k["z_cowl"]),                  # base of windscreen
        (k["x_ws_top"], H),                          # top of windscreen
        (k["x_roof_rear"], H),                       # end of roof
    ]
    if p.style in ("sedan", "coupe", "sports"):
        zd = k["z_deck"]
        pts += [(k["x_deck"], zd),                   # base of rear glass
                (xr + 45, zd - 25),                  # end of boot lid
                (xr, zd - 130)]
    elif p.style == "hatchback":
        pts += [(xr + 70, zb + 0.22 * (H - zb)),      # bottom of hatch glass
                (xr, zb - 70)]
    else:  # wagon / suv: near-vertical tailgate
        pts += [(xr + 55, zb + 0.70 * (H - zb)),
                (xr, zb - 60)]
    pts += [(xr, clr + 130)]                         # rear bumper bottom
    return [[round(x, 2), round(z, 2)] for x, z in pts]


def _profile_top_z(profile: list[list[float]], x: float) -> float:
    """Height of the side profile's upper surface at position x (lowest upper crossing, conservative)."""
    zs = []
    n = len(profile)
    for i in range(n):
        (x0, z0), (x1, z1) = profile[i], profile[(i + 1) % n]
        if (x0 - x) * (x1 - x) <= 0 and x0 != x1:
            zs.append(z0 + (z1 - z0) * (x - x0) / (x1 - x0))
    return max(zs) if zs else 0.0


def body_section(p: CarParams) -> list[list[float]]:
    """Cross-section in the YZ plane: full width below the beltline, tumblehome above it."""
    k = side_profile_keypoints(p)
    hw, H, zb = p.width / 2, p.height, k["z_belt"]
    t = p.tumblehome
    y_sh = hw * (t + (1 - t) * 0.82)                 # shoulder, for a slightly rounded greenhouse
    z_sh = zb + 0.18 * (H - zb)
    pts = [(-hw * 0.97, 0), (hw * 0.97, 0), (hw, 0.35 * zb), (hw, zb), (y_sh, z_sh), (hw * t, H + 40),
           (-hw * t, H + 40), (-y_sh, z_sh), (-hw, zb), (-hw, 0.35 * zb)]
    return [[round(y, 2), round(z, 2)] for y, z in pts]


def plan_outline(p: CarParams) -> list[list[float]]:
    """Top-view outline in the XY plane (rounded nose and tail)."""
    hw, xf, xr = p.width / 2, p.length / 2, -p.length / 2
    n, t = p.nose_taper, p.tail_taper
    pts = [(xr, -hw * t), (xr + 260, -hw), (xf - 380, -hw), (xf - 130, -hw * (n + (1 - n) * 0.65)), (xf, -hw * n),
           (xf, hw * n), (xf - 130, hw * (n + (1 - n) * 0.65)), (xf - 380, hw), (xr + 260, hw), (xr, hw * t)]
    return [[round(x, 2), round(y, 2)] for x, y in pts]


def _glass_band(a, b, pad_a, pad_b, inset, outward, reach=420.0):
    """Quadrilateral covering the outside of the segment a->b, inset slightly below its surface.

    a, b: segment end points in plane coords; `outward` = +1/-1 chooses which normal points out of the body.
    """
    dx, dz = b[0] - a[0], b[1] - a[1]
    ln = math.hypot(dx, dz)
    ux, uz = dx / ln, dz / ln
    nx, nz = outward * uz, -outward * ux             # perpendicular
    a2 = (a[0] + ux * pad_a, a[1] + uz * pad_a)
    b2 = (b[0] - ux * pad_b, b[1] - uz * pad_b)
    q = [(a2[0] - nx * inset, a2[1] - nz * inset), (b2[0] - nx * inset, b2[1] - nz * inset),
         (b2[0] + nx * reach, b2[1] + nz * reach), (a2[0] + nx * reach, a2[1] + nz * reach)]
    return [[round(x, 2), round(z, 2)] for x, z in q]


def _lerp_x(p0, p1, z):
    (x0, z0), (x1, z1) = p0, p1
    return x0 + (x1 - x0) * (z - z0) / (z1 - z0)


def build_program(p: CarParams) -> list[dict]:
    """The ordered list of operations that builds the car."""
    k = side_profile_keypoints(p)
    L, W, H = p.length, p.width, p.height
    hw, xf, xr = W / 2, L / 2, -L / 2
    zb, zn, t = k["z_belt"], k["z_nose"], p.tumblehome
    r, tw = p.wheel_radius, p.tire_width
    ops: list[dict] = []

    def rnd(x):
        if isinstance(x, float):
            return round(x, 2)
        if isinstance(x, (list, tuple)):
            return [rnd(v) for v in x]
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        return x

    def add(name, part, mode, tool, comment, mirror=False, material=None):
        ops.append(dict(step=len(ops) + 1, name=name, part=part, mode=mode, mirror=mirror,
                        material=material, tool=rnd(tool), comment=comment))

    # 1-3: main body = side profile  ∩  top-view outline  ∩  cross-section
    add("side_profile", "body", "new",
        dict(type="prism", plane="XZ", points=side_profile(p), extent=[-hw, hw]),
        "Extrude the side silhouette across the full width")
    add("plan_outline", "body", "intersect",
        dict(type="prism", plane="XY", points=plan_outline(p), extent=[0, H + 100]),
        "Round the nose and tail in top view")
    add("cross_section", "body", "intersect",
        dict(type="prism", plane="YZ", points=body_section(p), extent=[xr - 50, xf + 50]),
        "Narrow the greenhouse above the beltline (tumblehome)")

    # 4-5: wheel arches (cut on the left side, mirrored to the right)
    arch_r = r + p.arch_gap
    arch_y = [hw - (tw + 120), hw + 60]
    for nm, xa in (("arch_front", k["x_front_axle"]), ("arch_rear", k["x_rear_axle"])):
        add(nm, "body", "cut",
            dict(type="cylinder", axis="Y", center=[round(xa, 2), round(r, 2)], radius=round(arch_r, 2),
                 extent=[round(arch_y[0], 2), round(arch_y[1], 2)]),
            f"Cut the {nm.split('_')[1]} wheel arch", mirror=True, material="arch")

    # 6: windscreen glass, 7: rear glass (thin inset bands following the glass lines)
    cowl, ws_top = (k["x_cowl"], k["z_cowl"]), (k["x_ws_top"], H)
    glass_hw = hw * t - 75
    add("windscreen", "body", "cut",
        dict(type="prism", plane="XZ", points=_glass_band(cowl, ws_top, 25, 35, 10, +1),
             extent=[round(-glass_hw, 2), round(glass_hw, 2)]),
        "Inset the windscreen glass", material="glass")
    roof_end = (k["x_roof_rear"], H)
    if p.style in ("sedan", "coupe", "sports"):
        rear_low = (k["x_deck"], k["z_deck"])
    elif p.style == "hatchback":
        rear_low = (xr + 70, zb + 0.22 * (H - zb))
    else:
        rear_low = (xr + 55, zb + 0.70 * (H - zb))
    add("rear_glass", "body", "cut",
        dict(type="prism", plane="XZ", points=_glass_band(roof_end, rear_low, 35, 25, 10, +1),
             extent=[round(-glass_hw, 2), round(glass_hw, 2)]),
        "Inset the rear glass", material="glass")

    # 8: side windows = window outline (XZ)  ∩  region just outside the sloped cabin side (YZ)
    zw0, zw1 = zb + 35, H - 75
    xa0 = _lerp_x(cowl, ws_top, zw0) - 95             # A-pillar
    xa1 = _lerp_x(cowl, ws_top, zw1) - 95
    if p.style in ("sedan", "coupe", "sports", "hatchback"):
        xc1 = _lerp_x(roof_end, rear_low, zw1) + (150 if p.style != "hatchback" else 190)  # C-pillar
        xc0 = _lerp_x(roof_end, rear_low, min(zw0, rear_low[1] + 1)) + (230 if p.style != "hatchback" else 260)
        xc0 = max(xc0, xc1 - 120)
    else:  # D-pillar near the tailgate
        xc1 = xr + 230
        xc0 = xr + 260
    window = [[xc0, zw0], [xa0, zw0], [xa1, zw1], [xc1, zw1]]
    sec = body_section(p)
    # sloped side of the cabin on the +y side: shoulder points offset 12 mm inwards, region outside it
    (y_b, z_b), (y_s, z_s), (y_t, z_t) = (sec[3][0], sec[3][1]), (sec[4][0], sec[4][1]), (sec[5][0], sec[5][1])
    outside = [[y_b - 12, z_b - 60], [hw + 200, z_b - 60], [hw + 200, H + 250], [y_t - 12, H + 250],
               [y_t - 12, z_t], [y_s - 12, z_s], [y_b - 12, z_b]]
    outside = [[round(a, 2), round(b, 2)] for a, b in outside]
    if p.style in ("coupe", "sports"):
        windows = [("side_windows", window)]
    else:  # split by a B-pillar into front and rear door glass
        xm = 0.5 * (xa0 + xc0) + 0.08 * (xa0 - xc0)
        front = [[xm + 45, zw0], [xa0, zw0], [xa1, zw1], [xm + 45 - 0.05 * (zw1 - zw0), zw1]]
        rear = [[xc0, zw0], [xm - 45, zw0], [xm - 45 - 0.05 * (zw1 - zw0), zw1], [xc1, zw1]]
        windows = [("front_door_glass", front), ("rear_door_glass", rear)]
    for nm, poly in windows:
        poly = [[round(a, 2), round(b, 2)] for a, b in poly]
        add(nm, "body", "cut",
            dict(type="intersect", children=[
                dict(type="prism", plane="XZ", points=poly, extent=[0, hw + 200]),
                dict(type="prism", plane="YZ", points=outside, extent=[xr, xf])]),
            f"Inset the {nm.replace('_', ' ')}", mirror=True, material="glass")

    # lights and grille: shallow cuts whose faces get a material
    yn = hw * p.nose_taper
    add("headlights", "body", "cut",
        dict(type="prism", plane="XZ",
             points=[[xf - 70, zn - 130], [xf + 120, zn - 130], [xf + 120, zn - 30], [xf - 70, zn - 30]],
             extent=[round(yn - 360, 2), round(yn - 60, 2)]),
        "Cut the headlight recesses", mirror=True, material="light")
    add("grille", "body", "cut",
        dict(type="prism", plane="XZ",
             points=[[xf - 45, p.ground_clearance + 0.30 * (zn - p.ground_clearance)], [xf + 120, p.ground_clearance + 0.30 * (zn - p.ground_clearance)],
                     [xf + 120, zn - 150], [xf - 45, zn - 150]],
             extent=[round(-(yn - 390), 2), round(yn - 390, 2)]),
        "Cut the front grille", material="grille")
    yt = hw * p.tail_taper
    z_tl = (k["z_deck"] - 150) if p.style in ("sedan", "coupe", "sports") else zb - 40
    add("taillights", "body", "cut",
        dict(type="prism", plane="XZ",
             points=[[xr - 120, z_tl - 70], [xr + 55, z_tl - 70], [xr + 55, z_tl + 45], [xr - 120, z_tl + 45]],
             extent=[round(yt - 400, 2), round(yt - 60, 2)]),
        "Cut the tail-light recesses", mirror=True, material="taillight")

    # v2 parts (PartVerse: bumpers, mirrors, exhausts and spoilers are common separate parts)
    clr = p.ground_clearance
    add("front_bumper", "body", "union",
        dict(type="prism", plane="XZ",
             points=[[xf - 190, clr + 15], [xf - 3, clr + 15], [xf - 3, clr + 0.38 * (zn - clr)],
                     [xf - 190, clr + 0.38 * (zn - clr)]],
             extent=[round(-(yn - 70), 2), round(yn - 70, 2)]),
        "Add the front bumper lip", material="trim")
    add("rear_bumper", "body", "union",
        dict(type="prism", plane="XZ",
             points=[[xr + 3, clr + 15], [xr + 200, clr + 15], [xr + 200, clr + 0.30 * (zb - clr)],
                     [xr + 3, clr + 0.30 * (zb - clr)]],
             extent=[round(-(yt - 70), 2), round(yt - 70, 2)]),
        "Add the rear bumper", material="trim")
    x_m = _lerp_x(cowl, ws_top, zb + 60) - 60          # just behind the A-pillar base
    add("mirrors", "body", "union",
        dict(type="prism", plane="XZ",
             points=[[x_m - 150, zb + 25], [x_m, zb + 25], [x_m + 15, zb + 125], [x_m - 140, zb + 125]],
             extent=[round(hw - 80, 2), round(hw + 110, 2)]),
        "Add the side mirrors", mirror=True)
    add("exhaust", "body", "cut",
        dict(type="cylinder", axis="X", center=[round(-0.45 * hw, 2), round(clr + 75, 2)], radius=34.0,
             extent=[round(xr - 60, 2), round(xr + 90, 2)]),
        "Cut the exhaust outlet", material="arch")
    if p.spoiler_height >= 20 and p.style in ("sedan", "coupe", "sports"):
        zt = k["z_deck"] + p.spoiler_height
        ys = round(yt - 90, 2)
        add("spoiler_wing", "body", "union",
            dict(type="prism", plane="XZ",
                 points=[[xr + 35, zt - 28], [xr + 270, zt - 8], [xr + 255, zt + 6], [xr + 45, zt + 4]],
                 extent=[-ys, ys]),
            "Add the rear wing", material="trim")
        add("spoiler_posts", "body", "union",
            dict(type="prism", plane="XZ",
                 points=[[xr + 120, k["z_deck"] - 40], [xr + 175, k["z_deck"] - 40], [xr + 180, zt - 15],
                         [xr + 125, zt - 15]],
                 extent=[round(0.55 * yt - 20, 2), round(0.55 * yt + 20, 2)]),
            "Add the wing supports", mirror=True, material="trim")

    # wheels: tyre + rim, separate part, mirrored
    y_out = hw - 18
    for nm, xa in (("front", k["x_front_axle"]), ("rear", k["x_rear_axle"])):
        add(f"tyres_{nm}", "wheels", "union",
            dict(type="cylinder", axis="Y", center=[round(xa, 2), round(r, 2)], radius=round(r, 2),
                 extent=[round(y_out - tw, 2), round(y_out, 2)]),
            f"Add the {nm} tyres", mirror=True, material="tire")
        add(f"rims_{nm}", "wheels", "union",
            dict(type="cylinder", axis="Y", center=[round(xa, 2), round(r, 2)], radius=round(0.64 * r, 2),
                 extent=[round(y_out - 0.5 * tw, 2), round(y_out + 6, 2)]),
            f"Add the {nm} rims", mirror=True, material="rim")
    return ops


# --------------------------------------------------------------------------------------------
# CadQuery export
# --------------------------------------------------------------------------------------------
def _cq_tool_expr(tool: dict) -> str:
    """Python expression (CadQuery) that builds `tool` as a Workplane solid."""
    t = tool["type"]
    if t == "intersect":
        exprs = [_cq_tool_expr(c) for c in tool["children"]]
        out = exprs[0]
        for e in exprs[1:]:
            out = f"({out}).intersect({e})"
        return out
    if t == "prism":
        plane, pts = tool["plane"], tool["points"]
    elif t == "cylinder":
        plane = PLANE_OF_AXIS[tool["axis"]]
    else:
        raise ValueError(t)
    w0, w1 = tool["extent"]
    depth = round(w1 - w0, 2)
    # CadQuery named planes: XY normal +Z, YZ normal +X, XZ normal -Y. Start the sketch on the face
    # the extrusion leaves from so the solid spans exactly [w0, w1].
    origin = {"XY": f"(0, 0, {w0})", "YZ": f"({w0}, 0, 0)", "XZ": f"(0, {w1}, 0)"}[plane]
    if t == "prism":
        pts_s = ", ".join(f"({a}, {b})" for a, b in pts)
        return f'cq.Workplane("{plane}", origin={origin}).polyline([{pts_s}]).close().extrude({depth})'
    cx, cz = tool["center"]
    return (f'cq.Workplane("{plane}", origin={origin}).pushPoints([({cx}, {cz})])'
            f'.circle({tool["radius"]}).extrude({depth})')


def to_cadquery(ops: list[dict], params: CarParams | None = None) -> str:
    """A standalone CadQuery script. Running it defines `parts` = {"body": Workplane, "wheels": Workplane}."""
    lines = ['"""Auto-generated CAD program (CadQuery). Units: mm. +X front, +Y left, +Z up."""',
             "import cadquery as cq", ""]
    if params is not None:
        lines.append(f"# car style: {params.style}, length {params.length:.0f}, width {params.width:.0f}, "
                     f"height {params.height:.0f}")
    lines.append("parts = {}")
    for op in ops:
        lines.append("")
        lines.append(f"# step {op['step']}: {op['comment']}" + (f"  [material: {op['material']}]" if op["material"] else ""))
        lines.append(f"tool = {_cq_tool_expr(op['tool'])}")
        if op["mirror"]:
            lines.append('tool = tool.union(tool.mirror("XZ"))')
        part, mode = op["part"], op["mode"]
        if mode == "new":
            lines.append(f'parts["{part}"] = tool')
        elif mode == "union":
            lines.append(f'parts["{part}"] = parts["{part}"].union(tool) if "{part}" in parts else tool')
        else:
            lines.append(f'parts["{part}"] = parts["{part}"].{mode}(tool)')
    lines += ["", "car = cq.Assembly()",
              "for name, wp in parts.items():",
              "    car.add(wp, name=name)", ""]
    return "\n".join(lines)
