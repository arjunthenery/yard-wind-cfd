# yard-wind-cfd

OpenFOAM pipeline that computes 3D CFD wind loads (surface pressure,
velocity field) on a stacked-shipping-container yard layout — the input a
sliding/toppling risk assessment would need, not the risk assessment
itself. This repo covers mesh generation through flow solving and
visualization only; it doesn't compute risk ratios, friction thresholds,
or overturning moments.

The example config shipped here (`single_module`) is a **synthetic/toy
layout** for demonstrating the pipeline, not real terminal data.

## Background

Written for the **PSA Antwerp** problem at the
[**Mathematics for Industry 2026**](https://be-maths-in.be/mfi26/problems/)
study week (BE-MATHS-IN): predicting wind-induced toppling/sliding risk for
stacked containers, and identifying safe stacking/repositioning moves ahead
of severe weather. The study week's stated motivation is that existing
empirical models (wind forces, shielding matrices, friction coefficients)
underrepresent secondary aerodynamic effects — particularly suction/
underpressure between container stacks — which this CFD pipeline resolves
directly rather than approximating.

The broader project's sliding/toppling risk model (comparing these wind
loads against friction coefficients, container weight, and overturning
moments to flag at-risk containers) is a separate piece of work, not part
of this repo.

## Workflow

Meshing is independent of wind speed/turbulence/solver settings, so it's
split from running a simulation — build a mesh once per **layout**, then run
as many simulations against it as you like with different wind conditions.

```
./make_mesh <mesh-name>          # layout -> mesh/<mesh-name>/
./run_simulation <scenario-name> # mesh + wind/solver settings -> out/<scenario-name>_<timestamp>/
./view_results <output-name>     # 3D view of that run's results
```

1. **`./make_mesh <mesh-name>`** — builds geometry (container-stack STLs) and
   runs `blockMesh` -> `surfaceFeatures` -> `snappyHexMesh` -> `checkMesh`.
   Reads `mesh_params/<mesh-name>.yaml`, writes `mesh/<mesh-name>/` (standard
   OpenFOAM case layout: `constant/polyMesh/`, `system/controlDict`,
   `case_meta.json`, `view_geometry/*.stl`, a copy of the params file used).
   Re-running overwrites that mesh. `<mesh-name>` is whatever you choose —
   it's just the lookup key into `mesh_params/` and the output folder name
   under `mesh/`.
2. **`./run_simulation <scenario-name>`** — builds boundary conditions/solver
   dicts against an **already-built** mesh and runs it. Reads
   `params/<scenario-name>.yaml` (which points at a mesh via its `mesh:`
   field), writes `out/<scenario-name>_<timestamp>/` — just the results
   (time directories, `case_meta.json`, physics summary) and the mesh it
   used (a symlink, not a copy — see "Mesh reuse in `out/`" below). Not
   meant to be reproducible/re-run in place: the solver-config dicts and
   the generating scripts are cleaned up once the solve finishes, since
   nothing here reads them back and this is a results folder, not an
   archived case. No viewer scripts are copied in either — see below.
3. **`./view_results <output-name>`** — once a run finishes, opens the 3D
   results viewer (surface pressure + streamlines, preferring time-averaged
   `pMean`/`UMean` when available, else instantaneous), by running
   `tools/view_results_3d.py` directly against the output folder.
   Accepts either the exact `out/` folder name or a bare scenario name
   (picks the most recent matching run). For the top-down pressure/velocity
   map instead, run `python3 ../../tools/view_results_topdown.py` from
   inside the output folder.

`./make_mesh`/`./run_simulation` accept either a bare name (looked up in
`mesh_params/`/`params/`) or a literal path to a `.yaml` file.
`./run_simulation` also accepts a processor-count override as an extra
numeric arg, in either position: `./run_simulation single_module 36`.
`./view_results` looks things up under `out/` instead (see above).

**Mesh = geometry/layout/wind ANGLE** (the geometry is pre-rotated so the
solver frame always sees wind along +X — `wind_angle_deg` lives in
`mesh_params/`). **Simulation = wind SPEED, turbulence model, solver
settings, processor count** (lives in `params/`, which only references the
mesh by name). Don't put wind speed in `mesh_params/` or wind angle in
`params/` — the split is intentional and both scripts assume it.

### Quickstart

```bash
./setup                          # one-time: creates .venv/ and installs Python deps
                                  # (OpenFOAM itself is a separate, manual install — see Requirements below)
./make_mesh single_module
./run_simulation single_module
./view_results single_module     # opens the 3D results viewer
```

### Mesh reuse in `out/`

`out/<scenario>_<timestamp>/constant/polyMesh` and `view_geometry/` are
**symlinks** back to `mesh/<name>/`, not copies — the mesh is already saved
there once, so duplicating its STL/polyMesh data (often hundreds of MB)
into every single run would just be waste. OpenFOAM reads through the
symlink fine during the actual solve, and so does `view_results_3d.py`
afterward — nothing in the output folder needs to change for this to work.
The viewer scripts themselves aren't copied into `out/` either, for the
same reason they're never copied into `mesh/<name>/`: they don't change
per-run, so `./view_results` just runs `tools/view_results_3d.py` directly
against the output folder instead.

The tradeoff: if you later re-run `./make_mesh <name>` (which **overwrites**
that mesh), every existing `out/` folder that used it now points at the
new mesh too, not the one it was actually solved against. If you need an
`out/` folder to stay valid independently of what happens to `mesh/`
afterward, copy `mesh/<name>/` aside before rebuilding it, or copy the
`constant/polyMesh`/`view_geometry` symlink targets into the run folder
for real before deleting/overwriting the mesh.

### Naming your own mesh/scenario

Copy a template and name it whatever you like — the name is just the
filename (minus `.yaml`), and becomes the mesh's folder name under `mesh/`:

```bash
cp mesh_params/template.yaml mesh_params/my_layout.yaml
# edit mesh_params/my_layout.yaml: container grid, gaps, wind_angle_deg, cell size...
./make_mesh my_layout

cp params/template.yaml params/my_layout_25ms.yaml
# edit params/my_layout_25ms.yaml: set mesh: my_layout, wind_speed, turbulence model...
./run_simulation my_layout_25ms
```

Re-running a sweep over wind speeds against the same mesh is just multiple
`params/*.yaml` files with the same `mesh:` field and different `wind_speed`:

```bash
for p in params/my_layout_15ms.yaml params/my_layout_25ms.yaml params/my_layout_35ms.yaml; do
    ./run_simulation "$p"
done
```

## Folder map

- `setup` / `requirements.txt` — one-time Python environment setup (creates
  `.venv/`). Does **not** install OpenFOAM — see Requirements below.
- `mesh_params/{single_module,template}.yaml` —
  layout configs: container dims, row/bay/module gaps, `grid_heights`
  (per-module 2D row x bay matrix of stack heights, 0 = empty),
  `wind_angle_deg`, domain sizing multiples, `background_cell_size`. Heavily
  commented — read `template.yaml`'s header before writing a new one,
  especially the axis-convention diagram (rows/modules run parallel to the
  canal along +X, bays extend inland along +Y, bay 0 is closest to the
  water).
- `params/{single_module,template}.yaml` —
  simulation configs: `mesh:` (which built mesh to use), `wind_speed`,
  `turbulence_intensity`, `rho_air`/`nu_air`, `num_processors`,
  `simulation_mode` (`"steady"` = SIMPLE, mean-flow only, cheap;
  `"transient"` = PIMPLE, resolves vortex shedding/peak loads, 10-100x more
  expensive — **prefer transient**, steady residuals plateau rather than
  converge for this bluff-body flow), `turbulence_model` (`"kOmegaSST"`
  recommended for transient; `"kEpsilon"` was the old steady default), and
  `transient_*`/`spinup_time` settings (discard an initial spin-up window,
  then average over the rest).
- `core/build_mesh.py` — owns all geometry/STL generation, `blockMeshDict`,
  `snappyHexMeshDict`, and a minimal `system/controlDict` stub (`blockMesh`
  needs one to exist even for pure meshing — it constructs a `Foam::Time`
  object that reads it).
- `core/build_case.py` — consumes a pre-built mesh via
  `mesh/<name>/case_meta.json`; requires a `mesh:` field in its params file.
  Writes the BC/solver dicts and a `run_case.sh` that does the actual
  meshing-independent run (`decomposePar`/`foamRun`/`reconstructPar` etc.).
- `mesh/<name>/` — built mesh caches, standard OpenFOAM case layout
  (`constant/polyMesh/`, not flat `polyMesh/` — needed for
  `vtkOpenFOAMReader`/`view_mesh` to work directly against this folder).
  **Not checked into git** (regenerate with `./make_mesh`) — see
  `.gitignore`. `out/` folders reference this by symlink rather than
  copying it — see "Mesh reuse in `out/`" above.
- `out/<scenario>_<timestamp>/` — simulation results. **Not checked into
  git** — see `.gitignore`.
- `tools/inspect_mesh.py` (run via `./view_mesh <mesh-name>`) — sanity-check a
  built mesh before running an expensive simulation: the snapped stack
  surfaces (colored per patch), so you can confirm the container layout
  looks right before spending compute on a solve. Draws a water-region fill
  via Sutherland-Hodgman half-plane polygon clipping against the rotated
  `canal_line` from `case_meta.json`, and a wind arrow (always +X — this view
  is in the solver frame, already rotated for wind angle).
- `tools/view_geometry.py` — idealized container-stack layout preview (wind
  arrow + canal line), reading straight from `view_geometry/*.stl`. Largely
  redundant with `./view_mesh` (same purpose, different visual style) — no
  wrapper script for this one; run `python3 ../../tools/view_geometry.py`
  from inside a `mesh/<name>/` or `out/<scenario>_<timestamp>/` folder if
  you want it.
- `tools/view_results_3d.py` (run via `./view_results <output-name>`) — 3D
  surface pressure + streamlines, once a simulation finishes (time-averaged
  `pMean`/`UMean` when available, else instantaneous).
- `tools/view_results_topdown.py` — top-down pressure/velocity flow-field
  map; no wrapper script for this one, run
  `python3 ../../tools/view_results_topdown.py` from inside the output
  folder.

## Requirements

- **[OpenFOAM](https://www.openfoam.com/)** (tested with OpenFOAM-11) — **you
  need to install this yourself**; it's a separate system install, not
  something `./setup` or pip can install for you. `make_mesh`/`run_simulation`
  look for it at `/opt/openfoam11/etc/bashrc` — if yours lives elsewhere (a
  different install path, a cluster `module load`, etc.), edit the "Loading
  OpenFOAM" step in `make_mesh`, `run_simulation`, and `core/build_case.py`
  to match.
- Python 3.8+ with `pyyaml`, `numpy-stl`, `pyvista`, `vtk`, `matplotlib` — run
  **`./setup`** once to create a local `.venv/` with all of these installed.
  `make_mesh`/`run_simulation`/`view_mesh`/`view_results` auto-activate
  `.venv/` if present, else assume `python3` already has the packages.

## Known gotchas

- **OpenFOAM's `fileName` class fatally rejects paths containing spaces.**
  Both `make_mesh` and `run_simulation` detect a space in `pwd` and stage the
  actual OpenFOAM work (meshing / mesh+solve) in a space-free `/tmp`
  directory, then copy/move the finished result back into
  `mesh/<name>/`/`out/<name>_<timestamp>/`. Case *generation* (pure
  Python/YAML, no OpenFOAM calls) is unaffected and always writes straight
  to the final location. If your clone's path has no space in it, this
  staging step is unnecessary but harmless.
- **`set -e` is inherited into `( ... )` subshells** (unlike a separately
  invoked `bash script.sh`). OpenFOAM's own `bashrc` runs `unalias
  wmRefresh` on first load, which exits 1 — with inherited `-e` that silently
  kills the subshell before meshing/solving even starts. Fix pattern used
  throughout: `set +e` right before `source .../bashrc`, `set -e` right
  after. If you add a new OpenFOAM-sourcing step in a subshell, keep this
  pattern.
- **Mesh cache must be standard layout** (`constant/polyMesh/` +
  `system/controlDict`), not flat `polyMesh/` — `vtkOpenFOAMReader` (used by
  `inspect_mesh.py` and friends) requires the standard layout and fails with
  "Cannot open directory: ./constant/" otherwise.

## Credits

Author: **Arjun Thenery Manikantan**, on behalf of **Team PSA** at the
[Mathematics for Industry 2026](https://be-maths-in.be/mfi26/problems/)
study week (BE-MATHS-IN), tackling the PSA Antwerp problem.

Portions of this pipeline (refactoring, documentation, tooling) were
developed with assistance from Claude (Anthropic).

This pipeline uses [OpenFOAM](https://www.openfoam.com/) for all CFD meshing
and solving (`blockMesh`, `snappyHexMesh`, `foamRun`/PIMPLE) — install it
separately (see Requirements). This offering is not approved or endorsed by
OpenCFD Limited, producer and distributor of the OpenFOAM software via
www.openfoam.com, and owner of the OPENFOAM® and OpenCFD® trade marks.

## License

MIT — see `LICENSE`. Covers this repository's own code only, not OpenFOAM
itself (a separate, independently-licensed/installed program).
