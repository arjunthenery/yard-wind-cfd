"""
Author: Arjun Thenery Manikantan
Made for the Mathematics for Industry 2026 study week (BE-MATHS-IN, PSA
Antwerp problem) — https://be-maths-in.be/mfi26/problems/
Uses OpenFOAM (openfoam.com) for CFD meshing/solving.

build_mesh.py — generates the geometry + mesh-defining OpenFOAM files
(STL surfaces, blockMeshDict, surfaceFeaturesDict, snappyHexMeshDict) for a
container-stack yard from a mesh-params YAML file (see mesh_params/).

This script only WRITES the dictionary/geometry files — it does not invoke
OpenFOAM itself. The `make_mesh` wrapper script runs blockMesh/surfaceFeatures/
snappyHexMesh/checkMesh afterwards and saves the result into mesh/<name>/.

HOW TO USE: you normally don't call this directly — run `./make_mesh <name>`
from the code_final_github/ root instead, which handles staging (OpenFOAM
can't run from a path containing spaces) and saving the finished mesh into
mesh/<name>/.

Mesh-params vs. simulation-params: this script only cares about the fields
that affect the MESH ITSELF — container/gap dimensions, the stack layout,
wind_angle_deg (the mesh geometry is pre-rotated so the solver always sees
wind along +X — so wind *direction* is a mesh property, even though wind
*speed* is not), domain sizing, and background cell size. Wind speed,
turbulence, solver settings, and processor count never affect the mesh and
belong in the simulation's own params file (params/) instead.
"""

import os
import argparse
import glob as _glob
import json as _json
import numpy as np
import yaml
from stl import mesh

# ============================================================================
# Load mesh params
# ============================================================================

_arg_parser = argparse.ArgumentParser(
    description="Generate mesh-defining OpenFOAM files from a mesh-params YAML file.")
_arg_parser.add_argument("--params", required=True,
                          help="Path to the mesh-params YAML file (see mesh_params/).")
_arg_parser.add_argument("--out", default=".",
                          help="Directory to write the generated mesh-input files into.")
_args = _arg_parser.parse_args()

with open(_args.params) as _f:
    _P = yaml.safe_load(_f)

CASE_DIR = _args.out
os.makedirs(CASE_DIR, exist_ok=True)

MESH_NAME = os.path.splitext(os.path.basename(_args.params))[0]

CONTAINER = dict(L=_P["container"]["length"], W=_P["container"]["width"], H=_P["container"]["height"])
ROW_GAP = _P["row_gap"]
BAY_GAP = _P["bay_gap"]
MODULE_GAP = _P["module_gap"]
CANAL_OFFSET = 5.0  # visual-only distance (m) from bay 0's edge to the drawn canal line
                     # in view_geometry.py. Purely cosmetic — no effect on mesh/domain.
GRID_HEIGHTS = _P["grid_heights"]

WIND_ANGLE_DEG = _P["wind_angle_deg"]   # affects mesh geometry (rotation) -> belongs here, not in sim params

DOMAIN_UPSTREAM_MULT = _P["domain_upstream_mult"]
DOMAIN_DOWNSTREAM_MULT = _P["domain_downstream_mult"]
DOMAIN_TOP_MULT = _P["domain_top_mult"]
DOMAIN_SIDE_MARGIN = _P["domain_side_margin"]

BACKGROUND_CELL_SIZE = _P["background_cell_size"]

# ============================================================================
# Layout generation
# ============================================================================

def generate_grid_layout():
    """
    Builds a layout from GRID_HEIGHTS: a list of modules, each module a 2D
    matrix GRID_HEIGHTS[m][r][b] = container count at that module's row r,
    bay b. 0 = empty bay (skipped).

    Axis convention (see mesh_params/ for the diagram):
      X = parallel to the canal/quay — modules sit side by side along X
          (separated by MODULE_GAP), and within a module, rows also step
          along X (separated by ROW_GAP). X = 0 is centered on the middle
          of the whole module block (not the edge of module 0).
      Y = perpendicular to the canal — bay 0 is closest to the canal,
          higher bay indices move further inland (separated by BAY_GAP).
    """
    row_pitch = CONTAINER["W"] + ROW_GAP
    bay_pitch = CONTAINER["L"] + BAY_GAP

    layout = []
    n_empty, n_total = 0, 0
    x_module_offset = 0.0

    for m, module in enumerate(GRID_HEIGHTS):
        n_rows_in_module = len(module)
        for r, row in enumerate(module):
            x0 = x_module_offset + r * row_pitch
            for b, n_containers in enumerate(row):
                n_total += 1
                y0 = b * bay_pitch
                if n_containers > 0:
                    layout.append((f"m{m}_r{r}_b{b}", x0, y0, int(n_containers)))
                else:
                    n_empty += 1
        x_module_offset += n_rows_in_module * row_pitch + MODULE_GAP

    # Center X = 0 on the middle of the whole module block, instead of
    # leaving it at module 0's leading edge.
    if layout:
        xs = [x0 for _, x0, y0, n in layout] + [x0 + CONTAINER["W"] for _, x0, y0, n in layout]
        x_center = (min(xs) + max(xs)) / 2
        layout = [(name, x0 - x_center, y0, n) for name, x0, y0, n in layout]

    total_containers = sum(n for module in GRID_HEIGHTS for row in module for n in row)
    print(f"Grid layout: {len(GRID_HEIGHTS)} module(s), {len(layout)} occupied stacks, "
          f"{n_empty} empty bays (out of {n_total} total), {total_containers} total containers")
    return layout


def print_layout_ascii(grid_heights):
    """
    Prints GRID_HEIGHTS in the SAME visual orientation as the diagram in
    mesh_params/ (bay 0 nearest the canal at top, modules left-to-right) —
    this is the direct sanity check for whether your numbers actually
    match what you intended, since the YAML's row-major nesting reads
    differently than the diagram's bay-major layout.
    """
    n_modules = len(grid_heights)
    n_rows = len(grid_heights[0])
    n_bays = len(grid_heights[0][0])
    col_width = n_rows * 3 + 2

    print("\nLayout preview (match this against the diagram in mesh_params/):")
    print("  canal / quay")
    print("  " + "=" * (n_modules * (col_width + 3)))
    for b in range(n_bays):
        line = "  "
        for m in range(n_modules):
            vals = [grid_heights[m][r][b] for r in range(n_rows)]
            line += "[" + " ".join(f"{v:>2}" for v in vals) + "]   "
        tag = " <- bay 0 (nearest canal)" if b == 0 else (
              f" <- bay {b} (furthest inland)" if b == n_bays - 1 else f" <- bay {b}")
        print(line + tag)
    label_line = "  "
    for m in range(n_modules):
        label_line += f"module {m}".center(col_width) + "   "
    print(label_line + "\n")


print_layout_ascii(GRID_HEIGHTS)
STACK_LAYOUT = generate_grid_layout()

if not STACK_LAYOUT:
    raise SystemExit("Generated layout has zero occupied stacks — check GRID_HEIGHTS in the mesh-params file.")

STACKS = [
    (name, x0, y0, 0.0, CONTAINER["W"], CONTAINER["L"], n * CONTAINER["H"])
    for name, x0, y0, n in STACK_LAYOUT
]
MAX_STACK_H = max(s[6] for s in STACKS)   # tallest stack — used for domain sizing


def rotate_xy(x, y, angle_rad, pivot):
    dx, dy = x - pivot[0], y - pivot[1]
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return pivot[0] + dx * c - dy * s, pivot[1] + dx * s + dy * c


def compute_stack_geometry(stacks, angle_deg):
    """
    Rotates every stack's footprint by angle_deg around the yard's own
    center (X,Y only — height is unaffected). Returns a list of dicts with
    the actual 8 corner points (world frame, possibly non-axis-aligned)
    plus the centroid (used for CofR) for each stack.
    angle_deg = 0 reproduces the original axis-aligned layout exactly.

    Also returns the canal reference line: in LOCAL (unrotated) yard
    coordinates the canal sits just before bay 0 (Y = -bay_gap/2, per the
    axis convention), spanning the full row/module extent along X. This
    gets rotated by the SAME angle/pivot as the stacks, so it stays
    physically fixed relative to the yard regardless of what wind angle
    you're testing — the canal doesn't move just because the simulated
    wind direction changes.
    """
    local_xs = [s[1] for s in stacks] + [s[1] + s[4] for s in stacks]
    local_ys = [s[2] for s in stacks] + [s[2] + s[5] for s in stacks]
    pivot = ((min(local_xs) + max(local_xs)) / 2, (min(local_ys) + max(local_ys)) / 2)
    angle_rad = np.radians(-angle_deg)  # sign flipped so +angle matches the documented
                                          # convention (90 = bay 0/canal side upstream)

    geom = []
    for name, x0, y0, z0, dx, dy, dz in stacks:
        x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz
        corners_local = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        corners_rot = [rotate_xy(px, py, angle_rad, pivot) for px, py in corners_local]
        corners = np.array([
            [corners_rot[0][0], corners_rot[0][1], z0],
            [corners_rot[1][0], corners_rot[1][1], z0],
            [corners_rot[2][0], corners_rot[2][1], z0],
            [corners_rot[3][0], corners_rot[3][1], z0],
            [corners_rot[0][0], corners_rot[0][1], z1],
            [corners_rot[1][0], corners_rot[1][1], z1],
            [corners_rot[2][0], corners_rot[2][1], z1],
            [corners_rot[3][0], corners_rot[3][1], z1],
        ])
        centroid = corners.mean(axis=0)
        geom.append(dict(name=name, corners=corners, centroid=centroid, height=dz))

    canal_y_local = min(local_ys) - CANAL_OFFSET
    margin = 0.15 * (max(local_xs) - min(local_xs))
    canal_x0_local, canal_x1_local = min(local_xs) - margin, max(local_xs) + margin
    canal_p0 = rotate_xy(canal_x0_local, canal_y_local, angle_rad, pivot)
    canal_p1 = rotate_xy(canal_x1_local, canal_y_local, angle_rad, pivot)

    return geom, (canal_p0, canal_p1)


STACK_GEOM, CANAL_LINE = compute_stack_geometry(STACKS, WIND_ANGLE_DEG)

# Fixed (angle=0) geometry, used ONLY by view_geometry.py so the canal and
# yard always render in their true, unrotated orientation — only a wind
# arrow rotates to show the current wind_angle_deg. The solver-facing STLs
# below (constant/triSurface) still use the rotated STACK_GEOM, since
# OpenFOAM always blows wind along a fixed axis — that rotation is a
# computational necessity, not something the visualization needs to show.
VIEW_GEOM, VIEW_CANAL_LINE = compute_stack_geometry(STACKS, 0)

all_corners = np.concatenate([g["corners"] for g in STACK_GEOM])
STACK_X_MIN, STACK_X_MAX = all_corners[:, 0].min(), all_corners[:, 0].max()
STACK_Y_MIN, STACK_Y_MAX = all_corners[:, 1].min(), all_corners[:, 1].max()
STACK_Z_MAX = all_corners[:, 2].max()

DOMAIN_X_MIN = STACK_X_MIN - DOMAIN_UPSTREAM_MULT * MAX_STACK_H
DOMAIN_X_MAX = STACK_X_MAX + DOMAIN_DOWNSTREAM_MULT * MAX_STACK_H
DOMAIN_Y_MIN = STACK_Y_MIN - DOMAIN_SIDE_MARGIN
DOMAIN_Y_MAX = STACK_Y_MAX + DOMAIN_SIDE_MARGIN
DOMAIN_Z_MIN = 0.0
DOMAIN_Z_MAX = DOMAIN_TOP_MULT * MAX_STACK_H

N_CELLS_X = max(int((DOMAIN_X_MAX - DOMAIN_X_MIN) / BACKGROUND_CELL_SIZE), 10)
N_CELLS_Y = max(int((DOMAIN_Y_MAX - DOMAIN_Y_MIN) / BACKGROUND_CELL_SIZE), 5)
N_CELLS_Z = max(int((DOMAIN_Z_MAX - DOMAIN_Z_MIN) / BACKGROUND_CELL_SIZE), 5)

# A point guaranteed inside the fluid domain and outside every stack:
# far downstream (past all stacks), mid-height, mid-span.
LOCATION_IN_MESH = (
    DOMAIN_X_MAX - 0.25 * (DOMAIN_X_MAX - STACK_X_MAX),
    (DOMAIN_Y_MIN + DOMAIN_Y_MAX) / 2,
    DOMAIN_Z_MAX / 2,
)

os.makedirs(f"{CASE_DIR}/0", exist_ok=True)  # harmless placeholder dir; the sim stage owns 0/
os.makedirs(f"{CASE_DIR}/constant/triSurface", exist_ok=True)

for _stale in _glob.glob(f"{CASE_DIR}/constant/triSurface/*.stl") + \
              _glob.glob(f"{CASE_DIR}/constant/triSurface/*.eMesh"):
    os.remove(_stale)
for _stale in _glob.glob(f"{CASE_DIR}/constant/extendedFeatureEdgeMesh/*"):
    os.remove(_stale)
for _stale in _glob.glob(f"{CASE_DIR}/view_geometry/*.stl"):
    os.remove(_stale)
os.makedirs(f"{CASE_DIR}/system", exist_ok=True)

with open(f"{CASE_DIR}/case_meta.json", "w") as _f:
    _json.dump({
        "mesh_name": MESH_NAME,
        "container": CONTAINER,
        "wind_angle_deg": WIND_ANGLE_DEG,
        "canal_line": [[float(v) for v in CANAL_LINE[0]], [float(v) for v in CANAL_LINE[1]]],
        "canal_line_fixed": [[float(v) for v in VIEW_CANAL_LINE[0]], [float(v) for v in VIEW_CANAL_LINE[1]]],
        "stacks": [{"name": g["name"], "centroid": [float(v) for v in g["centroid"]]} for g in STACK_GEOM],
        "max_stack_h": float(MAX_STACK_H),
        "domain": {
            "x_min": float(DOMAIN_X_MIN), "x_max": float(DOMAIN_X_MAX),
            "y_min": float(DOMAIN_Y_MIN), "y_max": float(DOMAIN_Y_MAX),
            "z_min": float(DOMAIN_Z_MIN), "z_max": float(DOMAIN_Z_MAX),
            "n_cells_x": N_CELLS_X, "n_cells_y": N_CELLS_Y, "n_cells_z": N_CELLS_Z,
            "background_cell_size": BACKGROUND_CELL_SIZE,
        },
        "stack_extent": {
            "x_min": float(STACK_X_MIN), "x_max": float(STACK_X_MAX),
            "y_min": float(STACK_Y_MIN), "y_max": float(STACK_Y_MAX), "z_max": float(STACK_Z_MAX),
        },
    }, _f, indent=2)


def write(path, content):
    with open(path, "w") as f:
        f.write(content)
    print(f"  wrote {path}")


# ============================================================================
# 1. STL geometry
# ============================================================================

def box_stl_from_corners(corners, outfile):
    """corners: 8x3 array, bottom face 0-3 then top face 4-7, same winding as before."""
    faces = np.array([
        [0, 3, 1], [1, 3, 2], [4, 5, 7], [5, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
    ])
    m = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            m.vectors[i][j] = corners[f[j]]
    m.save(outfile)


print("Writing STL geometry...")
os.makedirs(f"{CASE_DIR}/view_geometry", exist_ok=True)
for g, vg in zip(STACK_GEOM, VIEW_GEOM):
    box_stl_from_corners(g["corners"], f"{CASE_DIR}/constant/triSurface/{g['name']}.stl")
    box_stl_from_corners(vg["corners"], f"{CASE_DIR}/view_geometry/{vg['name']}.stl")
    c = g["corners"]
    print(f"  {g['name']}: bbox ({c[:,0].min():.2f},{c[:,1].min():.2f},{c[:,2].min():.2f}) "
          f"to ({c[:,0].max():.2f},{c[:,1].max():.2f},{c[:,2].max():.2f})  "
          f"centroid ({g['centroid'][0]:.2f},{g['centroid'][1]:.2f},{g['centroid'][2]:.2f})")


# ============================================================================
# 1b. system/controlDict (minimal stub)
# ============================================================================
# blockMesh/snappyHexMesh/checkMesh all require A controlDict to exist (they
# construct a Foam::Time object that reads it), even though they never
# actually advance time. The REAL controlDict — application, end time,
# forces function objects, etc. — is a solve-stage concern and gets
# overwritten by build_case.py once this mesh is used in a simulation; this
# stub only needs to be valid enough for the meshing utilities to run.

write(f"{CASE_DIR}/system/controlDict", """FoamFile
{
    version 2.0;
    format  ascii;
    class   dictionary;
    object  controlDict;
}

application     foamRun;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
""")


# ============================================================================
# 2. system/blockMeshDict
# ============================================================================

blockMeshDict = f"""FoamFile
{{
    version 2.0;
    format  ascii;
    class   dictionary;
    object  blockMeshDict;
}}

convertToMeters 1;

vertices
(
    ({DOMAIN_X_MIN} {DOMAIN_Y_MIN} {DOMAIN_Z_MIN})
    ({DOMAIN_X_MAX} {DOMAIN_Y_MIN} {DOMAIN_Z_MIN})
    ({DOMAIN_X_MAX} {DOMAIN_Y_MAX} {DOMAIN_Z_MIN})
    ({DOMAIN_X_MIN} {DOMAIN_Y_MAX} {DOMAIN_Z_MIN})
    ({DOMAIN_X_MIN} {DOMAIN_Y_MIN} {DOMAIN_Z_MAX})
    ({DOMAIN_X_MAX} {DOMAIN_Y_MIN} {DOMAIN_Z_MAX})
    ({DOMAIN_X_MAX} {DOMAIN_Y_MAX} {DOMAIN_Z_MAX})
    ({DOMAIN_X_MIN} {DOMAIN_Y_MAX} {DOMAIN_Z_MAX})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({N_CELLS_X} {N_CELLS_Y} {N_CELLS_Z}) simpleGrading (1 1 1)
);

edges ();

boundary
(
    inlet   {{ type patch;    faces ((0 4 7 3)); }}
    outlet  {{ type patch;    faces ((1 2 6 5)); }}
    ground  {{ type wall;     faces ((0 1 2 3)); }}
    top     {{ type patch;    faces ((4 5 6 7)); }}
    sides   {{ type symmetry; faces ((0 1 5 4) (3 2 6 7)); }}
);

mergePatchPairs ();
"""
write(f"{CASE_DIR}/system/blockMeshDict", blockMeshDict)


# ============================================================================
# 3. system/surfaceFeaturesDict
# ============================================================================

surf_list = " ".join(f'"{s[0]}.stl"' for s in STACKS)
surfaceFeaturesDict = f"""FoamFile
{{
    version 2.0;
    format  ascii;
    class   dictionary;
    object  surfaceFeaturesDict;
}}

surfaces ({surf_list});
includedAngle 150;
"""
write(f"{CASE_DIR}/system/surfaceFeaturesDict", surfaceFeaturesDict)


# ============================================================================
# 4. system/snappyHexMeshDict
# ============================================================================

geometry_entries = "\n".join(
    f'    {s[0]}.stl {{ type triSurfaceMesh; file "{s[0]}.stl"; name {s[0]}; }}' for s in STACKS
)
feature_entries = "\n".join(
    f'        {{ file "{s[0]}.eMesh"; level 3; }}' for s in STACKS
)
refsurf_entries = "\n".join(
    f"        {s[0]} {{ level (2 3); }}" for s in STACKS
)

gap_min = (STACK_X_MIN - 1, STACK_Y_MIN - 1, 0)
gap_max = (STACK_X_MAX + 1, STACK_Y_MAX + 1, MAX_STACK_H + 1)

snappyHexMeshDict = f"""FoamFile
{{
    version 2.0;
    format  ascii;
    class   dictionary;
    object  snappyHexMeshDict;
}}

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
{geometry_entries}

    gapRefineBox
    {{
        type searchableBox;
        min ({gap_min[0]} {gap_min[1]} {gap_min[2]});
        max ({gap_max[0]} {gap_max[1]} {gap_max[2]});
    }}
}}

castellatedMeshControls
{{
    maxLocalCells 2000000;
    maxGlobalCells 8000000;
    minRefinementCells 10;
    nCellsBetweenLevels 3;
    resolveFeatureAngle 30;

    features
    (
{feature_entries}
    );

    refinementSurfaces
    {{
{refsurf_entries}
    }}

    refinementRegions
    {{
        gapRefineBox {{ mode inside; levels ((1 3)); }}
    }}

    locationInMesh ({LOCATION_IN_MESH[0]:.3f} {LOCATION_IN_MESH[1]:.3f} {LOCATION_IN_MESH[2]:.3f});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
}}

addLayersControls
{{
    layers {{}}
    relativeSizes true;
    expansionRatio 1.2;
    finalLayerThickness 0.5;
    minThickness 0.1;
}}

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-9;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

writeFlags (scalarLevels layerSets layerFields);
mergeTolerance 1e-6;
"""
write(f"{CASE_DIR}/system/snappyHexMeshDict", snappyHexMeshDict)

print(f"\nStack/yard extent: X[{STACK_X_MIN:.2f},{STACK_X_MAX:.2f}] "
      f"Y[{STACK_Y_MIN:.2f},{STACK_Y_MAX:.2f}] Z[0,{STACK_Z_MAX:.2f}]")
print(f"Domain: X[{DOMAIN_X_MIN:.1f},{DOMAIN_X_MAX:.1f}] "
      f"Y[{DOMAIN_Y_MIN:.1f},{DOMAIN_Y_MAX:.1f}] Z[0,{DOMAIN_Z_MAX:.1f}]  "
      f"cells: {N_CELLS_X}x{N_CELLS_Y}x{N_CELLS_Z} = {N_CELLS_X*N_CELLS_Y*N_CELLS_Z}")
print(f"locationInMesh: {LOCATION_IN_MESH}  (verify this is outside all stacks!)")
print(f"\nMesh-input files written to {CASE_DIR}")
