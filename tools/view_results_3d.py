# Author: Arjun Thenery Manikantan
# Made for the Mathematics for Industry 2026 study week (BE-MATHS-IN, PSA
# Antwerp problem) — https://be-maths-in.be/mfi26/problems/
# Uses OpenFOAM (openfoam.com) for CFD meshing/solving.

import pyvista as pv
import numpy as np

import json
try:
    with open("case_meta.json") as f:
        CONTAINER_H = json.load(f)["container_h"]
except FileNotFoundError:
    CONTAINER_H = 3.048  # fallback if build_case.py hasn't been run yet with the metadata fix
    print("Warning: case_meta.json not found, using fallback CONTAINER_H — rerun build_case.py to fix.")

RHO = 1.225   # air density — converts OpenFOAM's kinematic pressure (m^2/s^2) to real Pa,
              # same convention as view_results_topdown.py

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

print(f"Found {len(stack_names)} stack patch(es)")


def separator_lines_for_patch(patch):
    """Build a list of pv.PolyData line loops at each container-height boundary."""
    bounds = patch.bounds  # (xmin,xmax, ymin,ymax, zmin,zmax)
    z_min, z_max = bounds[4], bounds[5]
    n_containers = max(round((z_max - z_min) / CONTAINER_H), 1)

    pts = patch.points
    bottom_xy = pts[np.isclose(pts[:, 2], z_min, atol=1e-3)][:, :2]
    if len(bottom_xy) == 0:
        return []
    corners_xy = np.unique(np.round(bottom_xy, 3), axis=0)
    if len(corners_xy) < 3:
        return []
    c = corners_xy.mean(axis=0)
    order = np.argsort(np.arctan2(corners_xy[:, 1] - c[1], corners_xy[:, 0] - c[0]))
    loop_xy = corners_xy[order]

    lines = []
    for k in range(1, n_containers):
        z = z_min + k * CONTAINER_H
        loop_pts = np.array([[x, y, z] for x, y in loop_xy] + [[loop_xy[0][0], loop_xy[0][1], z]])
        lines.append(pv.lines_from_points(loop_pts))
    return lines


plotter = pv.Plotter(shape=(1, 2), window_size=(1400, 700))

# Prefer time-averaged fields (pMean/UMean, written by fieldAverage in
# transient mode) over instantaneous ones — a single instant of an
# oscillating flow isn't representative on its own. Falls back to p/U for
# steady-mode runs, which never produce Mean fields in the first place.
any_patch = patches[stack_names[0]]
P_FIELD = "pMean" if "pMean" in any_patch.array_names else "p"
U_FIELD = "UMean" if "UMean" in internal.array_names else "U"
print(f"Using {P_FIELD} for pressure, {U_FIELD} for velocity "
      f"({'time-averaged' if P_FIELD == 'pMean' else 'instantaneous — steady run or no averaging available'})")

# --- Panel 1: pressure on the stack surfaces, with container separators ---
plotter.subplot(0, 0)
for name in stack_names:
    patch = patches[name].copy()
    # OpenFOAM's "p" is kinematic pressure (m^2/s^2) — convert to real Pa here,
    # so this matches view_results_topdown.py rather than silently showing the
    # wrong units.
    patch["p_Pa"] = patch[P_FIELD] * RHO
    plotter.add_mesh(patch, scalars="p_Pa", cmap="coolwarm", show_edges=False)
    for line in separator_lines_for_patch(patch):
        plotter.add_mesh(line, color="black", line_width=2)
plotter.add_text(f"Surface pressure, Pa ({P_FIELD}) — black lines = container boundaries", font_size=9)
plotter.add_axes(xlabel="X (wind is +X)", ylabel="Y", zlabel="Z")
plotter.view_isometric()

# --- Panel 2: streamlines through the domain ---
plotter.subplot(0, 1)
bounds = internal.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
streamlines = internal.streamlines(
    vectors=U_FIELD,
    # Seed a quarter of the way into the domain from the inlet (X), centered
    # in Y, partway up in Z — gives the flow room to develop before reaching
    # the stacks.
    source_center=(bounds[0] + (bounds[1] - bounds[0]) / 4, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 4),
    source_radius=max((bounds[3] - bounds[2]) / 3, 5),
    n_points=40,
    max_time=300,
)
plotter.add_mesh(streamlines.tube(radius=0.05), scalars=U_FIELD, cmap="viridis")
for name in stack_names:
    plotter.add_mesh(patches[name], color="lightgray")
plotter.add_text("Streamlines through the stacks", font_size=10)
plotter.add_axes(xlabel="X (wind is +X)", ylabel="Y", zlabel="Z")
plotter.view_isometric()

plotter.show()
