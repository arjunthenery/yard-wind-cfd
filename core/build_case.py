"""
Author: Arjun Thenery Manikantan
Made for the Mathematics for Industry 2026 study week (BE-MATHS-IN, PSA
Antwerp problem) — https://be-maths-in.be/mfi26/problems/
Uses OpenFOAM (openfoam.com) for CFD meshing/solving.

build_case.py — generates a complete OpenFOAM SIMULATION case (boundary/
initial conditions, solver settings, run_case.sh) from a simulation-params
YAML file (see params/), using a MESH that was already built separately
by build_mesh.py / the `make_mesh` wrapper script.

This script deliberately knows nothing about layout/geometry/meshing — it
only consumes mesh/<mesh-name>/ (the mesh itself plus its case_meta.json
metadata: container size, stack names + centroids, wind angle, domain
extent). That split exists because wind speed, turbulence, and solver
settings never affect the mesh, so re-running a simulation with different
wind conditions should never require re-meshing — see build_mesh.py's
docstring for the mesh side of this.

HOW TO USE (pipeline, recommended):
     ./make_mesh single_module        # once per mesh scenario (see mesh_params/)
     ./run_simulation single_module   # as many times as you like, any wind/solver settings

HOW TO USE (direct — for editing/debugging a single case in place):
  1. Make sure the mesh it references (the `mesh:` field inside your params
     file) has already been built: ./make_mesh <mesh-name>.
  2. From the code_final_github/ root, run:  python3 core/build_case.py --params params/mycase.yaml
  3. Run:  bash run_case.sh   (also auto-generated — if you want to inspect
     or iterate on the generated system/*, copy it elsewhere first: the
     script deletes system/ and the non-mesh constant/* dicts, and finally
     itself, once the solve finishes — this is results data, not meant to
     be reproducible/re-run in place; re-generate with this script again
     if you need those dicts back)

See `python3 core/build_case.py --help` for all flags (--params/--out/--procs).
"""

import os
import shutil
import argparse
import datetime as _dt
import json as _json
import yaml

# ============================================================================
# Load simulation params — edit params.yaml (or whichever file you pass), not this
# ============================================================================

_arg_parser = argparse.ArgumentParser(
    description="Generate an OpenFOAM simulation case from a simulation-params YAML file.")
_arg_parser.add_argument("--params", default="params.yaml",
                          help="Path to the simulation-params YAML file (default: params.yaml in the current directory).")
_arg_parser.add_argument("--out", default=".",
                          help="Directory to write the generated case into (default: current directory). "
                               "Created if it doesn't exist yet.")
_arg_parser.add_argument("--procs", type=int, default=None,
                          help="Override num_processors from the params file (how many cores "
                               "decomposePar/mpirun use for the solve).")
_args, _ = _arg_parser.parse_known_args()

with open(_args.params) as _f:
    _P = yaml.safe_load(_f)

CASE_DIR = _args.out   # "." for the classic in-place workflow, or e.g. "out/myrun_20260101_120000"
                        # for the run_simulation workflow.
os.makedirs(CASE_DIR, exist_ok=True)

if "mesh" not in _P:
    raise SystemExit(
        f"'{_args.params}' has no 'mesh:' field. Simulation params files must reference a "
        f"pre-built mesh by name — see params/template.yaml for the expected format, and "
        f"mesh_params/ + the `make_mesh` script for building one.")
MESH_NAME = _P["mesh"]

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(_project_root, "mesh", MESH_NAME)
if not os.path.isdir(MESH_DIR):
    raise SystemExit(
        f"Mesh '{MESH_NAME}' not found at {MESH_DIR}\n"
        f"Build it first with:  ./make_mesh {MESH_NAME}\n"
        f"(see mesh_params/ for the mesh scenarios available, or add a new one there)")

with open(os.path.join(MESH_DIR, "case_meta.json")) as _f:
    _MESH_META = _json.load(_f)

CONTAINER = _MESH_META["container"]
WIND_ANGLE_DEG = _MESH_META["wind_angle_deg"]
MAX_STACK_H = _MESH_META["max_stack_h"]
STACK_NAMES = [s["name"] for s in _MESH_META["stacks"]]
_D = _MESH_META["domain"]
DOMAIN_X_MIN, DOMAIN_X_MAX = _D["x_min"], _D["x_max"]
DOMAIN_Y_MIN, DOMAIN_Y_MAX = _D["y_min"], _D["y_max"]
DOMAIN_Z_MAX = _D["z_max"]
N_CELLS_X, N_CELLS_Y, N_CELLS_Z = _D["n_cells_x"], _D["n_cells_y"], _D["n_cells_z"]
BACKGROUND_CELL_SIZE = _D["background_cell_size"]
_SE = _MESH_META["stack_extent"]
STACK_X_MIN, STACK_X_MAX = _SE["x_min"], _SE["x_max"]
STACK_Y_MIN, STACK_Y_MAX = _SE["y_min"], _SE["y_max"]
STACK_Z_MAX = _SE["z_max"]

# Domain-sizing multipliers are only needed for the printed physics summary
# below (display only) — re-read from the mesh's own saved copy of the mesh
# params it was built from, rather than duplicating these into case_meta.json.
with open(os.path.join(MESH_DIR, "mesh_params.yaml")) as _f:
    _MESH_P = yaml.safe_load(_f)
DOMAIN_UPSTREAM_MULT = _MESH_P["domain_upstream_mult"]
DOMAIN_DOWNSTREAM_MULT = _MESH_P["domain_downstream_mult"]
DOMAIN_TOP_MULT = _MESH_P["domain_top_mult"]
DOMAIN_SIDE_MARGIN = _MESH_P["domain_side_margin"]

RUN_NAME = _P.get("run_name", "") or MESH_NAME

WIND_SPEED = _P["wind_speed"]
TURBULENCE_INTENSITY = _P["turbulence_intensity"]
RHO_AIR = _P["rho_air"]
NU_AIR = _P["nu_air"]
END_TIME = _P["end_time"]
WRITE_INTERVAL = _P["write_interval"]

NUM_PROCESSORS = _args.procs if _args.procs is not None else _P["num_processors"]

SIMULATION_MODE = _P["simulation_mode"]
TURBULENCE_MODEL = _P["turbulence_model"]
MAX_COURANT = _P["max_courant_number"]
SPINUP_TIME = _P["spinup_time"]
TRANSIENT_END_TIME = _P["transient_end_time"]
TRANSIENT_WRITE_INTERVAL = _P["transient_write_interval"]

if SIMULATION_MODE not in ("steady", "transient"):
    raise SystemExit(f"simulation_mode must be 'steady' or 'transient', got {SIMULATION_MODE!r}")
if TURBULENCE_MODEL not in ("kEpsilon", "kOmegaSST"):
    raise SystemExit(f"turbulence_model must be 'kEpsilon' or 'kOmegaSST', got {TURBULENCE_MODEL!r}")

K_INLET = 1.5 * (TURBULENCE_INTENSITY * WIND_SPEED) ** 2
EPS_INLET = (0.09 ** 0.75) * (K_INLET ** 1.5) / 0.1   # mixing length ~0.1m assumed
OMEGA_INLET = EPS_INLET / (0.09 * K_INLET)   # standard exact relation: epsilon = Cmu*k*omega


def write(path, content):
    with open(path, "w") as f:
        f.write(content)
    print(f"  wrote {path}")


# ============================================================================
# 0. Bring in the pre-built mesh + geometry views
# ============================================================================

os.makedirs(f"{CASE_DIR}/0", exist_ok=True)
os.makedirs(f"{CASE_DIR}/constant", exist_ok=True)
os.makedirs(f"{CASE_DIR}/system", exist_ok=True)

print(f"Using mesh '{MESH_NAME}' from {MESH_DIR}")


def _symlink_mesh_dir(src, dst):
    """Point dst at src (an absolute path — MESH_DIR always is) instead of
    copying it. The mesh is already saved once in mesh/<name>/; copying its
    ~100s-of-MB polyMesh into every single run's output folder was pure
    duplication. A symlink works for OpenFOAM's own read access during the
    actual solve (decomposePar/foamRun/reconstructPar never write back into
    constant/polyMesh) just as well as it does for the viewer scripts
    afterward, and — being an absolute path — survives CASE_DIR later being
    staged in /tmp and moved into out/<name>_<timestamp>/."""
    if os.path.islink(dst):
        os.remove(dst)
    elif os.path.isdir(dst):
        shutil.rmtree(dst)
    elif os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


_symlink_mesh_dir(os.path.join(MESH_DIR, "constant", "polyMesh"), f"{CASE_DIR}/constant/polyMesh")
_symlink_mesh_dir(os.path.join(MESH_DIR, "view_geometry"), f"{CASE_DIR}/view_geometry")

# Case metadata for the viewer scripts (view_geometry.py, view_results_3d.py,
# view_results_topdown.py) to read automatically — same schema those scripts
# have always expected, just sourced from the mesh's own metadata now
# instead of being computed inline here.
with open(f"{CASE_DIR}/case_meta.json", "w") as _f:
    _json.dump({
        "container_h": CONTAINER["H"],
        "run_name": RUN_NAME,
        "canal_line": _MESH_META["canal_line"],
        "canal_line_fixed": _MESH_META["canal_line_fixed"],
        "wind_angle_deg": WIND_ANGLE_DEG,
        "num_processors": NUM_PROCESSORS,
    }, _f)


# ============================================================================
# 1. 0/ boundary condition files
# ============================================================================

def stack_bc_block(bc_line_template):
    return "\n".join(f"    {name:<9}{bc_line_template}" for name in STACK_NAMES)


U_file = f"""FoamFile
{{
    version 2.0;
    format  ascii;
    class   volVectorField;
    object  U;
}}

dimensions [0 1 -1 0 0 0 0];
internalField uniform ({WIND_SPEED} 0 0);

boundaryField
{{
    inlet    {{ type fixedValue; value uniform ({WIND_SPEED} 0 0); }}
    outlet   {{ type zeroGradient; }}
    ground   {{ type noSlip; }}
    top      {{ type slip; }}
    sides    {{ type symmetry; }}
{stack_bc_block("{ type noSlip; }")}
}}
"""
write(f"{CASE_DIR}/0/U", U_file)

p_file = f"""FoamFile
{{
    version 2.0;
    format  ascii;
    class   volScalarField;
    object  p;
}}

dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;

boundaryField
{{
    inlet    {{ type zeroGradient; }}
    outlet   {{ type fixedValue; value uniform 0; }}
    ground   {{ type zeroGradient; }}
    top      {{ type zeroGradient; }}
    sides    {{ type symmetry; }}
{stack_bc_block("{ type zeroGradient; }")}
}}
"""
write(f"{CASE_DIR}/0/p", p_file)

k_file = f"""FoamFile
{{
    version 2.0;
    format  ascii;
    class   volScalarField;
    object  k;
}}

dimensions [0 2 -2 0 0 0 0];
internalField uniform {K_INLET:.6g};

boundaryField
{{
    inlet    {{ type fixedValue; value uniform {K_INLET:.6g}; }}
    outlet   {{ type zeroGradient; }}
    ground   {{ type kqRWallFunction; value uniform {K_INLET:.6g}; }}
    top      {{ type slip; }}
    sides    {{ type symmetry; }}
{stack_bc_block(f"{{ type kqRWallFunction; value uniform {K_INLET:.6g}; }}")}
}}
"""
write(f"{CASE_DIR}/0/k", k_file)

for _stale_field in ("epsilon", "omega"):
    _stale_path = f"{CASE_DIR}/0/{_stale_field}"
    if os.path.exists(_stale_path):
        os.remove(_stale_path)

if TURBULENCE_MODEL == "kEpsilon":
    epsilon_file = f"""FoamFile
{{
    version 2.0;
    format  ascii;
    class   volScalarField;
    object  epsilon;
}}

dimensions [0 2 -3 0 0 0 0];
internalField uniform {EPS_INLET:.6g};

boundaryField
{{
    inlet    {{ type fixedValue; value uniform {EPS_INLET:.6g}; }}
    outlet   {{ type zeroGradient; }}
    ground   {{ type epsilonWallFunction; value uniform {EPS_INLET:.6g}; }}
    top      {{ type slip; }}
    sides    {{ type symmetry; }}
{stack_bc_block(f"{{ type epsilonWallFunction; value uniform {EPS_INLET:.6g}; }}")}
}}
"""
    write(f"{CASE_DIR}/0/epsilon", epsilon_file)
else:  # kOmegaSST
    omega_file = f"""FoamFile
{{
    version 2.0;
    format  ascii;
    class   volScalarField;
    object  omega;
}}

dimensions [0 0 -1 0 0 0 0];
internalField uniform {OMEGA_INLET:.6g};

boundaryField
{{
    inlet    {{ type fixedValue; value uniform {OMEGA_INLET:.6g}; }}
    outlet   {{ type zeroGradient; }}
    ground   {{ type omegaWallFunction; value uniform {OMEGA_INLET:.6g}; }}
    top      {{ type slip; }}
    sides    {{ type symmetry; }}
{stack_bc_block(f"{{ type omegaWallFunction; value uniform {OMEGA_INLET:.6g}; }}")}
}}
"""
    write(f"{CASE_DIR}/0/omega", omega_file)

nut_file = f"""FoamFile
{{
    version 2.0;
    format  ascii;
    class   volScalarField;
    object  nut;
}}

dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;

boundaryField
{{
    inlet    {{ type calculated; value uniform 0; }}
    outlet   {{ type calculated; value uniform 0; }}
    ground   {{ type nutkWallFunction; value uniform 0; }}
    top      {{ type slip; }}
    sides    {{ type symmetry; }}
{stack_bc_block("{ type nutkWallFunction; value uniform 0; }")}
}}
"""
write(f"{CASE_DIR}/0/nut", nut_file)


# ============================================================================
# 2. constant/ files
# ============================================================================

write(f"{CASE_DIR}/constant/physicsProperties", """FoamFile
{
    version 2.0;
    format  ascii;
    class   dictionary;
    object  physicsProperties;
}

type incompressibleFluid;
""")

write(f"{CASE_DIR}/constant/momentumTransport", f"""FoamFile
{{
    version 2.0;
    format  ascii;
    class   dictionary;
    object  momentumTransport;
}}

simulationType RAS;

RAS
{{
    model {TURBULENCE_MODEL};
    turbulence on;
    printCoeffs on;
}}
""")

write(f"{CASE_DIR}/constant/transportProperties", f"""FoamFile
{{
    version 2.0;
    format  ascii;
    class   dictionary;
    object  transportProperties;
}}

viscosityModel constant;
nu [0 2 -1 0 0 0 0] {NU_AIR};
""")


# ============================================================================
# 3. system/controlDict  (field-averaging functionObject, transient mode only —
#    no per-stack forces functionObjects: nothing in this pipeline currently
#    post-processes postProcessing/ force data, so it isn't generated)
# ============================================================================

avg_start = SPINUP_TIME
avg_fields = f"""    fieldAverage1
    {{
        type            fieldAverage;
        libs            ("libfieldFunctionObjects.so");
        timeStart       {avg_start};
        fields
        (
            U {{ mean on; prime2Mean off; base time; }}
            p {{ mean on; prime2Mean off; base time; }}
        );
        writeControl    writeTime;
    }}"""

if SIMULATION_MODE == "steady":
    _time_control_lines = f"""startTime       0;
stopAt          endTime;
endTime         {END_TIME};
deltaT          1;
writeControl    timeStep;
writeInterval   {WRITE_INTERVAL};"""
    _functions_section = ""
else:
    _time_control_lines = f"""startTime       0;
stopAt          endTime;
endTime         {TRANSIENT_END_TIME};
deltaT          0.001;
adjustTimeStep  yes;
maxCo           {MAX_COURANT};
maxDeltaT       0.5;
writeControl    adjustableRunTime;
writeInterval   {TRANSIENT_WRITE_INTERVAL};"""
    _functions_section = avg_fields

controlDict = f"""FoamFile
{{
    version 2.0;
    format  ascii;
    class   dictionary;
    object  controlDict;
}}

application     foamRun;
startFrom       startTime;
{_time_control_lines}
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
{_functions_section}
}}
"""
write(f"{CASE_DIR}/system/controlDict", controlDict)


# ============================================================================
# 4. system/fvSchemes, fvSolution  (static — rarely need changes)
# ============================================================================

_TURB_FIELD = "epsilon" if TURBULENCE_MODEL == "kEpsilon" else "omega"
_DDT_SCHEME = "steadyState" if SIMULATION_MODE == "steady" else "Euler"

write(f"{CASE_DIR}/system/fvSchemes", f"""FoamFile
{{
    version 2.0;
    format  ascii;
    class   dictionary;
    object  fvSchemes;
}}

ddtSchemes {{ default {_DDT_SCHEME}; }}
gradSchemes {{ default Gauss linear; }}

divSchemes
{{
    default             none;
    div(phi,U)          bounded Gauss linearUpwindV grad(U);
    div(phi,k)          bounded Gauss upwind;
    div(phi,{_TURB_FIELD})    bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}

laplacianSchemes {{ default Gauss linear corrected; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default corrected; }}
wallDist {{ method meshWave; }}
""")

if SIMULATION_MODE == "steady":
    write(f"{CASE_DIR}/system/fvSolution", f"""FoamFile
{{
    version 2.0;
    format  ascii;
    class   dictionary;
    object  fvSolution;
}}

solvers
{{
    p {{ solver GAMG; tolerance 1e-06; relTol 0.1; smoother GaussSeidel; }}
    "(U|k|{_TURB_FIELD})" {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-08; relTol 0.1; }}
}}

SIMPLE
{{
    nNonOrthogonalCorrectors 2;
    consistent yes;
    residualControl {{ p 1e-4; U 1e-4; "(k|{_TURB_FIELD})" 1e-4; }}
}}

relaxationFactors
{{
    fields {{ p 0.3; }}
    equations {{ U 0.7; k 0.7; {_TURB_FIELD} 0.7; }}
}}
""")
else:  # transient
    write(f"{CASE_DIR}/system/fvSolution", f"""FoamFile
{{
    version 2.0;
    format  ascii;
    class   dictionary;
    object  fvSolution;
}}

solvers
{{
    p
    {{
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.05;
        smoother        GaussSeidel;
    }}
    pFinal
    {{
        $p;
        relTol          0;
    }}
    U
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-07;
        relTol          0.1;
    }}
    UFinal {{ $U; relTol 0; }}
    k
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-07;
        relTol          0.1;
    }}
    kFinal {{ $k; relTol 0; }}
    {_TURB_FIELD}
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-07;
        relTol          0.1;
    }}
    {_TURB_FIELD}Final {{ ${_TURB_FIELD}; relTol 0; }}
}}

PIMPLE
{{
    nOuterCorrectors  1;
    nCorrectors       2;
    nNonOrthogonalCorrectors 1;
}}
""")

write(f"{CASE_DIR}/system/decomposeParDict", f"""FoamFile
{{
    version 2.0;
    format  ascii;
    class   dictionary;
    object  decomposeParDict;
}}

numberOfSubdomains {NUM_PROCESSORS};
method          scotch;
""")


# ============================================================================
# 5. run_case.sh — solve only (the mesh is already in place, copied in above)
# ============================================================================

run_script = f"""#!/bin/bash
# Resolve to an absolute path BEFORE cd-ing — $0 stays whatever string this
# script was invoked with (e.g. "out/name_ts/run_case.sh", relative to
# wherever the caller's cwd was), and cd-ing then trying to rm -f -- "$0"
# at the end would resolve that now-stale relative string against the NEW
# cwd instead, silently failing (rm -f swallows the "no such file" error)
# — this script just never deletes itself. Absolute, resolved once up
# front, stays correct regardless of what cd happens afterward.
_self="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"   # self-locating: works wherever this folder ends up (out/<name>/, an
                        # archived copy, or a copy sent elsewhere) — no hardcoded path needed.

echo "=== Loading OpenFOAM ==="
# set -e is turned on AFTER this block, not before — OpenFOAM's own bashrc
# internally runs `unalias wmRefresh` on first load, which exits 1 (nothing
# to unalias yet) and would otherwise silently kill this whole script right
# here if set -e were already active.
if [ -f /opt/openfoam11/etc/bashrc ]; then
    source /opt/openfoam11/etc/bashrc
else
    echo "Could not find OpenFOAM at /opt/openfoam11/etc/bashrc — adjust the" >&2
    echo "'Loading OpenFOAM' step in run_case.sh for how it's installed on" >&2
    echo "this machine (e.g. a different install path, or 'module load' on a cluster)." >&2
    exit 1
fi

set -e

echo "=== decomposePar ==="
decomposePar -force

echo "=== foamRun (solve, parallel, {NUM_PROCESSORS} cores) ==="
mpirun -np {NUM_PROCESSORS} foamRun -solver incompressibleFluid -parallel

echo "=== reconstructPar ==="
reconstructPar

echo "=== Cleaning up per-processor decomposition (already merged by reconstructPar) ==="
rm -rf processor*

echo "=== Cleaning up solver-config files (not needed to view results) ==="
rm -rf system constant/momentumTransport constant/physicsProperties constant/transportProperties

echo "=== DONE ==="
echo "Results are in: $(pwd)"
echo "View with, e.g.: ./view_results <this-folder-name>  (from the repo root)"
rm -f -- "$_self"
"""
write(f"{CASE_DIR}/run_case.sh", run_script)
os.chmod(f"{CASE_DIR}/run_case.sh", 0o755)

# No viewer scripts copied in here — they don't change per-run, so (same as
# mesh/<name>/ never getting a copy of inspect_mesh.py either) ./view_results
# just runs tools/view_results_3d.py directly via a relative path.

# --- Physics/numerics summary — printed every run AND saved to its own
# file (named after run_name), so the actual solver setup is always
# visible and stays as a permanent record alongside each run's results ---
_reynolds = WIND_SPEED * MAX_STACK_H / NU_AIR
_ddt_name = "steadyState (no real time — pseudo-iterations only)" if SIMULATION_MODE == "steady" else "Euler (1st-order implicit, real time)"
_algorithm = "SIMPLE" if SIMULATION_MODE == "steady" else "PIMPLE"
_turb_field_name = "epsilon" if TURBULENCE_MODEL == "kEpsilon" else "omega"

_summary_lines = []
_summary_lines.append("=" * 60)
_summary_lines.append("PHYSICS / NUMERICS SUMMARY")
_summary_lines.append("=" * 60)
_summary_lines.append(f"Run name                : {RUN_NAME or '(unnamed)'}")
_summary_lines.append(f"Mesh                     : {MESH_NAME}")
_summary_lines.append(f"Discretization method    : Finite Volume Method (FVM)")
_summary_lines.append(f"Flow regime              : incompressible (Mach << 0.3), Re ~ {_reynolds:.2e}")
_summary_lines.append(f"                            (U={WIND_SPEED} m/s, L=tallest stack={MAX_STACK_H:.2f}m, nu={NU_AIR})")
_summary_lines.append(f"Simulation mode          : {SIMULATION_MODE}  ->  {_algorithm} algorithm")
_summary_lines.append(f"Time discretization      : {_ddt_name}")
_summary_lines.append(f"Turbulence model         : {TURBULENCE_MODEL}  (turbulence field: {_turb_field_name})")
_summary_lines.append(f"Momentum convection      : bounded Gauss linearUpwindV (2nd-order upwind-biased)")
_summary_lines.append(f"Turbulence convection    : bounded Gauss upwind (1st-order, robust)")
_summary_lines.append(f"Diffusion (Laplacian)    : Gauss linear corrected (2nd-order, non-orthogonal correction)")
_summary_lines.append(f"Gradients                : Gauss linear")
if SIMULATION_MODE == "transient":
    _summary_lines.append(f"Max Courant number       : {MAX_COURANT}  (deltaT auto-adjusts to respect this)")
    _summary_lines.append(f"Spin-up / total time     : {SPINUP_TIME}s / {TRANSIENT_END_TIME}s "
                           f"({TRANSIENT_END_TIME - SPINUP_TIME}s used for time-averaging)")
_summary_lines.append(f"Air density / viscosity  : rho={RHO_AIR} kg/m^3, nu={NU_AIR} m^2/s")
_summary_lines.append(f"Wind angle / speed       : {WIND_ANGLE_DEG} deg / {WIND_SPEED} m/s")
_summary_lines.append(f"Mesh cells               : background {N_CELLS_X}x{N_CELLS_Y}x{N_CELLS_Z} = "
                       f"{N_CELLS_X*N_CELLS_Y*N_CELLS_Z} cells (before snappyHexMesh refinement)")
_est_min_cell = BACKGROUND_CELL_SIZE / 8   # max refinement level 3 -> /2^3
_summary_lines.append(f"Cell size (estimated)    : max {BACKGROUND_CELL_SIZE:.3f}m (background, unrefined) "
                       f"to min ~{_est_min_cell:.3f}m (at refinement level 3, near stacks/gaps)")
_summary_lines.append(f"Computational domain      : X[{DOMAIN_X_MIN:.1f}, {DOMAIN_X_MAX:.1f}]m  "
                       f"Y[{DOMAIN_Y_MIN:.1f}, {DOMAIN_Y_MAX:.1f}]m  Z[0, {DOMAIN_Z_MAX:.1f}]m")
_summary_lines.append(f"                            upstream={DOMAIN_UPSTREAM_MULT}x, "
                       f"downstream={DOMAIN_DOWNSTREAM_MULT}x, top={DOMAIN_TOP_MULT}x tallest stack height; "
                       f"side margin={DOMAIN_SIDE_MARGIN}m")
_summary_lines.append(f"Stack/yard extent         : X[{STACK_X_MIN:.2f}, {STACK_X_MAX:.2f}]m  "
                       f"Y[{STACK_Y_MIN:.2f}, {STACK_Y_MAX:.2f}]m  Z[0, {STACK_Z_MAX:.2f}]m")
_summary_lines.append("=" * 60)
_summary_text = "\n".join(_summary_lines)

print("\n" + _summary_text)

_summary_filename = f"physics_summary_{RUN_NAME}.txt" if RUN_NAME else \
    f"physics_summary_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(f"{CASE_DIR}/{_summary_filename}", "w") as _f:
    _f.write(_summary_text + "\n")
print(f"\nPhysics/numerics summary saved to: {_summary_filename}")

print(f"\nAll files written to {CASE_DIR}")
print(f"Now run:  bash {CASE_DIR}/run_case.sh")
