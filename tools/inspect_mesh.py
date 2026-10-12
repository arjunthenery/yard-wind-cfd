"""
Author: Arjun Thenery Manikantan
Made for the Mathematics for Industry 2026 study week (BE-MATHS-IN, PSA
Antwerp problem) — https://be-maths-in.be/mfi26/problems/
Uses OpenFOAM (openfoam.com) for CFD meshing/solving.

inspect_mesh.py — visualizes an already-built mesh from mesh/<name>/: the
snapped stack surfaces (the actual meshed geometry, which can differ
slightly from the idealized boxes view_geometry.py shows, since
snappyHexMesh snaps the background grid onto them), so you can sanity-check
the container layout BEFORE running an expensive simulation against it.

Run via:  ./view_mesh <mesh-name>   (from the code_final_github/ root)
"""

import pyvista as pv
import numpy as np
import json

with open("case_meta.json") as f:
    META = json.load(f)
MESH_NAME = META.get("mesh_name", "?")
# Solver-frame canal line (rotated to match wind_angle_deg, same frame the
# mesh itself is in) — NOT canal_line_fixed, which is only for view_geometry.py's
# true/unrotated display.
CANAL_LINE = META.get("canal_line")


def _clip_polygon_halfplane(poly_pts, line_p0, line_p1, keep_sign):
    """Sutherland-Hodgman clip of a convex polygon (list of (x,y)) to one side
    of the infinite line through line_p0/line_p1 — keeps the side matching
    keep_sign (the sign of the 2D cross product d x (pt - line_p0))."""
    d = np.array(line_p1) - np.array(line_p0)

    def side(pt):
        v = np.array(pt) - np.array(line_p0)
        return d[0] * v[1] - d[1] * v[0]

    out = []
    n = len(poly_pts)
    for i in range(n):
        curr, nxt = np.array(poly_pts[i]), np.array(poly_pts[(i + 1) % n])
        s_curr, s_next = side(curr), side(nxt)
        curr_in = s_curr * keep_sign >= 0
        next_in = s_next * keep_sign >= 0
        if curr_in:
            out.append(curr)
        if curr_in != next_in:
            t = s_curr / (s_curr - s_next)
            out.append(curr + t * (nxt - curr))
    return out


def add_canal(plotter, rect_bounds, yard_ref_xy, z=0.0, label_z_offset=0.8, inset=0.5):
    """Fills the WATER region of the domain footprint: the part of the ground
    rectangle (rect_bounds = xmin,xmax,ymin,ymax) that lies on the far side of
    the canal line from the yard (yard_ref_xy, e.g. the stacks' centroid) —
    i.e. the actual canal/dock area, not just an abstract line drawn on top.

    `inset` shrinks the rectangle slightly before filling — rect_bounds is
    usually the bounding box of some real (coarse-celled, jagged-edged) mesh
    patch, not a perfect rectangle, so filling exactly to those bounds can
    poke a sliver past the real boundary into empty background. A small
    inward margin keeps the fill safely inside instead.
    """
    if not CANAL_LINE:
        return
    (cx0, cy0), (cx1, cy1) = CANAL_LINE
    xmin, xmax, ymin, ymax = rect_bounds
    xmin, xmax, ymin, ymax = xmin + inset, xmax - inset, ymin + inset, ymax - inset
    rect = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

    d = np.array([cx1 - cx0, cy1 - cy0])
    v_yard = np.array(yard_ref_xy) - np.array([cx0, cy0])
    yard_sign = np.sign(d[0] * v_yard[1] - d[1] * v_yard[0])
    water_poly = _clip_polygon_halfplane(rect, (cx0, cy0), (cx1, cy1), -yard_sign)
    if len(water_poly) < 3:
        return  # canal line doesn't actually cross this rectangle — nothing to fill

    pts = np.array([[p[0], p[1], z] for p in water_poly])
    n = len(pts)
    face = [n] + list(range(n))
    water = pv.PolyData(pts, faces=face)
    plotter.add_mesh(water, color="#5B9BD5", opacity=0.6, show_edges=False, specular=0.0)

    centroid = pts[:, :2].mean(axis=0)
    plotter.add_point_labels([(centroid[0], centroid[1], z + label_z_offset)], ["CANAL"],
                              font_size=14, text_color="#1B4F72", bold=True,
                              shape=None, always_visible=True, show_points=False)


def add_wind_arrow(plotter, x_min, x_max, y_mid, z, label_z_offset=0.8):
    # This view shows the SOLVER frame, where the whole yard is pre-rotated
    # so wind always blows along +X — unlike view_geometry.py's true/unrotated
    # display, no wind_angle_deg rotation is needed here, the arrow is
    # always +X by construction.
    span = x_max - x_min
    arrow_len = max(0.25 * span, 5.0)
    start = (x_min - 0.15 * span - arrow_len, y_mid, z)
    arrow = pv.Arrow(start=start, direction=(1, 0, 0), scale=arrow_len,
                      tip_length=0.3, tip_radius=0.12, shaft_radius=0.04)
    plotter.add_mesh(arrow, color="green")
    plotter.add_point_labels([(start[0], start[1], z + label_z_offset)], ["WIND"],
                              font_size=14, text_color="green", bold=True,
                              shape=None, always_visible=True, show_points=False)

open("yard_case.foam", "a").close()
reader = pv.OpenFOAMReader("yard_case.foam")
if reader.time_values:
    reader.set_active_time_value(reader.time_values[-1])
mesh = reader.read()
internal = mesh["internalMesh"]
patches = mesh["boundary"]

non_stack_patches = {"inlet", "outlet", "ground", "top", "sides"}
stack_names = sorted(n for n in patches.keys() if n not in non_stack_patches)
if not stack_names:
    raise SystemExit("No stack patches found — did meshing complete correctly?")

print(f"Mesh '{MESH_NAME}': {internal.n_cells:,} internal cells, {len(stack_names)} stack patch(es)")

_all_bounds = np.array([patches[name].bounds for name in stack_names])
YARD_CENTROID_XY = (
    (_all_bounds[:, 0].min() + _all_bounds[:, 1].max()) / 2,
    (_all_bounds[:, 2].min() + _all_bounds[:, 3].max()) / 2,
)

plotter = pv.Plotter(window_size=(900, 700))

# Snapped stack surfaces, colored by patch so individual stacks are
# distinguishable. No edges here — the surface triangulation is fine enough
# after snapping that per-triangle edges just render as solid black noise;
# a ground patch CROPPED to the yard (not the whole domain, which would
# dwarf the stacks to tiny specks) gives scale context instead.
palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
for i, name in enumerate(stack_names):
    plotter.add_mesh(patches[name], color=palette[i % len(palette)], show_edges=False)
_yard_bounds_1 = np.array([patches[name].bounds for name in stack_names])
_margin_1 = 10.0
_ground_clip = (
    _yard_bounds_1[:, 0].min() - _margin_1, _yard_bounds_1[:, 1].max() + _margin_1,
    _yard_bounds_1[:, 2].min() - _margin_1, _yard_bounds_1[:, 3].max() + _margin_1,
    -1, 1,
)
# A clean SYNTHETIC flat rectangle, not the real "ground" patch — the actual
# OpenFOAM ground patch has coarse (~2m) cells, so clip_box leaves a jagged,
# notched edge rather than a clean straight line. Building our own rectangle
# here guarantees it shares EXACT bounds with the water-fill polygon below,
# so the two tile together with no mismatched gaps.
_gb = (_ground_clip[0], _ground_clip[1], _ground_clip[2], _ground_clip[3])
_ground_rect = pv.PolyData(
    np.array([[_gb[0], _gb[2], 0], [_gb[1], _gb[2], 0], [_gb[1], _gb[3], 0], [_gb[0], _gb[3], 0]]),
    faces=[4, 0, 1, 2, 3])
plotter.add_mesh(_ground_rect, color="lightgray", opacity=0.4, show_edges=False)
add_canal(plotter, rect_bounds=(_gb[0], _gb[1], _gb[2], _gb[3]), yard_ref_xy=YARD_CENTROID_XY)
add_wind_arrow(plotter, _yard_bounds_1[:, 0].min(), _yard_bounds_1[:, 1].max(),
               (_yard_bounds_1[:, 2].min() + _yard_bounds_1[:, 3].max()) / 2,
               _yard_bounds_1[:, 5].max() + 2)
plotter.add_text(f"Snapped mesh surfaces — {len(stack_names)} stacks, "
                  f"{internal.n_cells:,} cells total", font_size=9)
plotter.add_axes(xlabel="X (wind is +X)", ylabel="Y", zlabel="Z")
plotter.view_isometric()

plotter.show()
