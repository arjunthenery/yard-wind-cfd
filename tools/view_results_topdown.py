"""
Author: Arjun Thenery Manikantan
Made for the Mathematics for Industry 2026 study week (BE-MATHS-IN, PSA
Antwerp problem) — https://be-maths-in.be/mfi26/problems/
Uses OpenFOAM (openfoam.com) for CFD meshing/solving.

view_results_topdown.py — flat, top-down "floor plan" view of the flow field
close to the containers: pressure (left) and velocity magnitude (right),
both sliced at a chosen height, with container footprints outlined.

Why this instead of 3D streamlines/arrows: this is exactly the kind of plot
that most clearly shows suction (low pressure, blue) and channeling
(high velocity, bright) between containers — no 3D perspective, no
occlusion, no arrow-size tuning. Just a clean heat map you look straight
down on, like a map.
"""

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import json

try:
    with open("case_meta.json") as f:
        CONTAINER_H = json.load(f)["container_h"]
except FileNotFoundError:
    CONTAINER_H = 3.048
    print("Warning: case_meta.json not found, using fallback CONTAINER_H.")

RHO = 1.225  # air density, for converting kinematic p to real pressure (Pa)

# --- SLICE HEIGHT: edit this to look at a different height ---
# Default: half a container height (near-ground gap flow, usually where
# suction/channeling is strongest). Try CONTAINER_H * 1.5 for mid-stack,
# or CONTAINER_H * 2.5 for higher up, depending on what you want to see.
SLICE_HEIGHT_MULTIPLIER = 0.5

open("yard_case.foam", "a").close()
reader = pv.OpenFOAMReader("yard_case.foam")
reader.set_active_time_value(reader.time_values[-1])
mesh = reader.read()

internal = mesh["internalMesh"]
patches = mesh["boundary"]

non_stack_patches = {"inlet", "outlet", "ground", "top", "sides"}
stack_names = sorted(n for n in patches.keys() if n not in non_stack_patches)
if not stack_names:
    raise SystemExit("No stack patches found in results — did the solve run correctly?")

# --- yard footprint + zoom region ---
yard_bounds = np.array([patches[name].bounds for name in stack_names])
yard_xmin, yard_xmax = yard_bounds[:, 0].min(), yard_bounds[:, 1].max()
yard_ymin, yard_ymax = yard_bounds[:, 2].min(), yard_bounds[:, 3].max()
margin = 8.0

slice_z = SLICE_HEIGHT_MULTIPLIER * CONTAINER_H
print(f"Slicing at z={slice_z:.2f}m ({SLICE_HEIGHT_MULTIPLIER}x container height)")

# --- clip to near the yard, then take a horizontal slice ---
clip_bounds = (yard_xmin - margin, yard_xmax + margin,
               yard_ymin - margin, yard_ymax + margin,
               0, slice_z + 5)
near_field = internal.clip_box(clip_bounds, invert=False)
flow_slice = near_field.slice(normal="z", origin=(0, 0, slice_z))

if flow_slice.n_points == 0:
    raise SystemExit(f"Slice at z={slice_z:.2f}m had no points — try a different "
                      f"SLICE_HEIGHT_MULTIPLIER, or check the mesh extends that high.")

# OpenFOAM stores p/U as CELL data, not POINT data — their length doesn't
# match the point coordinates directly, which is what caused the
# "z array must have same length" error. Converting to point data
# interpolates cell values onto points, so everything lines up.
flow_slice = flow_slice.cell_data_to_point_data()

pts = flow_slice.points
x, y = pts[:, 0], pts[:, 1]
p_actual = flow_slice["p"] * RHO  # kinematic -> real pressure (Pa)
u_mag = np.linalg.norm(flow_slice["U"], axis=1)

# --- container footprints at this height, for outlining on the plots ---
footprints = []
for name in stack_names:
    patch = patches[name]
    b = patch.bounds
    if b[4] <= slice_z <= b[5]:  # only outline stacks tall enough to reach this height
        fpts = patch.points
        bottom_xy = fpts[np.isclose(fpts[:, 2], b[4], atol=1e-3)][:, :2]
        corners = np.unique(np.round(bottom_xy, 3), axis=0)
        if len(corners) >= 3:
            c = corners.mean(axis=0)
            order = np.argsort(np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0]))
            loop = corners[order]
            footprints.append(np.vstack([loop, loop[0]]))

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

triang = mtri.Triangulation(x, y)

for ax, field, title, cmap in [
    (axes[0], p_actual, "Pressure (Pa) — blue = suction", "coolwarm"),
    (axes[1], u_mag, "Velocity magnitude (m/s) — bright = channeling", "viridis"),
]:
    cf = ax.tricontourf(triang, field, levels=30, cmap=cmap)
    fig.colorbar(cf, ax=ax, shrink=0.8)
    for fp in footprints:
        ax.fill(fp[:, 0], fp[:, 1], color="white", edgecolor="black", linewidth=1.2, zorder=5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(yard_xmin - margin, yard_xmax + margin)
    ax.set_ylim(yard_ymin - margin, yard_ymax + margin)

fig.suptitle(f"Top-down view at z={slice_z:.2f}m — solver frame (wind blows along +X here)")
plt.tight_layout()
plt.show()
