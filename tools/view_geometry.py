# Author: Arjun Thenery Manikantan
# Made for the Mathematics for Industry 2026 study week (BE-MATHS-IN, PSA
# Antwerp problem) — https://be-maths-in.be/mfi26/problems/
# Uses OpenFOAM (openfoam.com) for CFD meshing/solving.

import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from stl import mesh
import numpy as np

import json
try:
    with open("case_meta.json") as f:
        _meta = json.load(f)
    CONTAINER_H = _meta["container_h"]
    CANAL_LINE = _meta.get("canal_line_fixed")   # always the TRUE, unrotated canal position
    WIND_ANGLE_DEG = _meta.get("wind_angle_deg", 0)
except FileNotFoundError:
    CONTAINER_H = 3.048  # fallback if build_case.py hasn't been run yet with the metadata fix
    CANAL_LINE = None
    WIND_ANGLE_DEG = 0
    print("Warning: case_meta.json not found, using fallback values — rerun build_case.py to fix.")

# Read from view_geometry/, NOT constant/triSurface/ — this folder holds the
# yard in its TRUE, FIXED orientation (angle=0 always), so the canal and
# layout never appear to rotate here. The actual solver mesh (constant/
# triSurface) is separately rotated by wind_angle_deg internally — that
# rotation is a computational necessity for OpenFOAM, not something this
# visualization should show. Instead, only the wind arrow below rotates.
stl_files = sorted(glob.glob("view_geometry/*.stl"))
if not stl_files:
    raise SystemExit("No STL files found in view_geometry/ — run build_case.py first "
                      "(you may have an older build_case.py without this folder — update it).")

colors = plt.cm.tab10.colors

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

all_pts = []
for i, path in enumerate(stl_files):
    m = mesh.Mesh.from_file(path)
    color = colors[i % len(colors)]
    ax.add_collection3d(Poly3DCollection(m.vectors, alpha=0.6, facecolor=color,
                                          edgecolor='k', linewidths=0.3))
    pts = m.points.reshape(-1, 3)
    all_pts.append(pts)

    # --- container separator lines ---
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
    n_containers = max(round((z_max - z_min) / CONTAINER_H), 1)
    bottom_xy = pts[np.isclose(pts[:, 2], z_min)][:, :2]
    corners_xy = np.unique(bottom_xy, axis=0)
    if len(corners_xy) == 4:
        c = corners_xy.mean(axis=0)
        order = np.argsort(np.arctan2(corners_xy[:, 1] - c[1], corners_xy[:, 0] - c[0]))
        loop_xy = corners_xy[order]
        for k in range(1, n_containers):
            z = z_min + k * CONTAINER_H
            loop = [(x, y, z) for x, y in loop_xy] + [(loop_xy[0][0], loop_xy[0][1], z)]
            ax.add_collection3d(Line3DCollection([loop], colors='black', linewidths=1.0))

        cx, cy = c
        ax.text(cx, cy, z_max + 0.6, f"{n_containers}x\n{z_max - z_min:.1f}m",
                ha='center', va='bottom', fontsize=6)

        if i == 0:
            for k in range(2):
                p0, p1 = loop_xy[k], loop_xy[k + 1]
                edge_len = np.linalg.norm(p1 - p0)
                mid = (p0 + p1) / 2
                outward = mid - c
                outward = outward / (np.linalg.norm(outward) + 1e-9) * 1.0
                label_pos = mid + outward
                ft = edge_len / 0.3048
                ax.text(label_pos[0], label_pos[1], z_min - 0.3,
                        f"{edge_len:.2f}m\n({ft:.0f}ft)",
                        ha='center', va='top', fontsize=7, color='darkred', weight='bold')
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [z_min, z_min],
                        color='darkred', linewidth=1.5, linestyle='--')

all_pts = np.concatenate(all_pts)

# --- canal: always drawn at its true, fixed position — never rotates ---
if CANAL_LINE:
    (cx0, cy0), (cx1, cy1) = CANAL_LINE
    ax.plot([cx0, cx1], [cy0, cy1], [0, 0], color='royalblue', linewidth=4, alpha=0.8)
    mid_x, mid_y = (cx0 + cx1) / 2, (cy0 + cy1) / 2
    ax.text(mid_x, mid_y, -0.8, "CANAL", ha='center', va='top',
            fontsize=10, color='royalblue', weight='bold')
    all_pts = np.vstack([all_pts, [cx0, cy0, 0], [cx1, cy1, 0]])

# Axis limits are computed from geometry + canal ONLY, with equal padding
# on every side — this keeps the plot bounds symmetric regardless of which
# way the wind arrow happens to point (a 180° arrow pointing -X shouldn't
# stretch the -X bound out further than the +X bound; the actual geometry
# is centered on X=0 and the plot should visually reflect that).
geom_x_min, geom_x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
geom_y_min, geom_y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
yard_span = max(geom_x_max - geom_x_min, geom_y_max - geom_y_min)
pad = max(0.3 * yard_span, 2.0)

# --- wind arrow: THIS is what rotates as wind_angle_deg changes ---
# Direction convention matches build_case.py: local wind vector = (cos(angle), sin(angle)).
# angle=0 -> blows parallel to canal (+X); angle=90 -> blows straight in off
# the canal (+Y), hitting bay 0 first. Tail sits a fixed distance upwind of
# the yard's center (same distance regardless of angle), pointing inward —
# this keeps the arrow visible and consistently placed within the padded
# viewport no matter which direction wind_angle_deg points it.
half_x, half_y = (geom_x_max - geom_x_min) / 2, (geom_y_max - geom_y_min) / 2
theta = np.radians(WIND_ANGLE_DEG)
# offset scales with the footprint's actual extent IN the wind direction —
# using a single max(half_x, half_y) overshoots badly for elongated yards
# (e.g. narrow in X, long in Y), since that's only correct for one direction.
offset_dist = abs(np.cos(theta)) * half_x + abs(np.sin(theta)) * half_y + 0.3 * pad
arrow_len = 0.6 * pad
arrow_z = all_pts[:, 2].max() + 3.0   # float above the tallest stack, clear of all geometry
center_x, center_y = (geom_x_min + geom_x_max) / 2, (geom_y_min + geom_y_max) / 2
start_x = center_x - np.cos(theta) * offset_dist
start_y = center_y - np.sin(theta) * offset_dist
wind_dx, wind_dy = np.cos(theta) * arrow_len, np.sin(theta) * arrow_len
ax.quiver(start_x, start_y, arrow_z, wind_dx, wind_dy, 0,
          color='green', linewidth=3, arrow_length_ratio=0.25)
ax.text(start_x - wind_dx * 0.3, start_y - wind_dy * 0.3, arrow_z + 0.8,
        f"WIND\n({WIND_ANGLE_DEG}°)", color='green', fontsize=9, weight='bold', ha='center')

ax.set_xlim(geom_x_min - pad, geom_x_max + pad)
ax.set_ylim(geom_y_min - pad, geom_y_max + pad)
ax.set_zlim(0, arrow_z + 2.0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f"{len(stl_files)} stack(s) — yard shown in fixed orientation, "
             f"wind arrow shows wind_angle_deg — close to continue")
plt.show()
