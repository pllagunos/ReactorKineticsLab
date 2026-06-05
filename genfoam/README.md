# Prerequisites
- activate the `openmc` Conda environment before running the Python tooling
- previously ran the `openmc/` workflow to get XML, statepoint, and MGXS files
- have an installation of foamForNuclear and OpenFOAM available in the shell

# GeN-Foam Concentric Reactor Case

This directory now contains a generated GeN-Foam diffusion case based on the OpenMC concentric reactor model.

## Current Scope

The current implementation uses the structure of the GeN-Foam `2D_externalSourceDiffusion` feature case, but replaces its neutronics data and mesh with OpenMC-derived inputs from the concentric reactor workflow. The generator now:

- uses a manual geometry source derived from the OpenMC reference model dimensions and region names
- uses the Gmsh Python API to build an axisymmetric wedge mesh as the active neutronics mesh
- represents the centerline as a very small inner cylindrical symmetry patch to avoid collapsed-axis import defects in `gmshToFoam`
- loads the exported MGXS constants from `openmc/build/concentric/mgxs_export`
- converts geometry from cm to m and MGXS data to SI-style units
- collapses higher-order scattering exports to the zeroth Legendre moment for diffusion use
- sanitizes inactive-group diffusion coefficients
- sanitizes zero-removal and zero-inverse-velocity groups that break the GeN-Foam solve
- assigns valid precursor decay constants in non-fuel zones so diagonal precursor solves remain well-posed
- writes the corresponding GeN-Foam case files for the imported wedge mesh

`openmc/build/concentric/reactor_run/model.xml` is still required as a reference artifact in the upstream export, but the current `genfoam` workflow does not parse it automatically.

## Usage

From `ReactorKineticsLab/`:

```bash
conda activate openmc
python genfoam/prepare_concentric_case.py
```

Optional arguments:

```bash
python genfoam/prepare_concentric_case.py \
  --mgxs-export-dir openmc/build/concentric/mgxs_export \
  --output-dir genfoam/constant/generated
```

## Generated Files

The script writes the following metadata files under `genfoam/constant/generated/`:

- `concentric_case_manifest.json` — combined source, geometry, mesh, and material payload
- `concentric_reactor_wedge.msh` — axisymmetric Gmsh wedge mesh for the concentric reactor
- `concentric_reactor_mesh_manifest.json` — expected physical groups, zone names, geometry, and mesh sizing used for the Gmsh mesh
- `concentric_mesh_regions.csv` — canonical region-to-cellZone mapping shared by the mesh and MGXS-derived nuclear data
- `concentric_materials.json` — per-domain multigroup constants and delayed-neutron data in converted units

It also writes the active GeN-Foam case files in place:

- `system/controlDict`, `system/regionsDict`
- `system/neutroRegion/fvSchemes`, `fvSolution`
- `constant/neutroRegion/neutronicsProperties`, `nuclearData`
- `0/neutroRegion/defaultFlux` through `defaultFlux16`
- `Allclean`, `Allmesh`, `Allrun`

The old `blockMeshDict` path is retired from the active workflow. The earlier full-3D Gmsh path remains experimental only and is not used by `Allmesh` or `Allrun`.

## Mesh Generation

The standalone mesh entry point is:

```bash
python genfoam/gmsh_reactor_mesh.py generate
```

Useful options:

```bash
python genfoam/gmsh_reactor_mesh.py generate \
  --mesh-kind axisymmetric-wedge \
  --wedge-angle-deg 1.0 \
  --fuel-size-m 0.02 \
  --core-size-m 0.08 \
  --moderator-size-m 0.2 \
  --reflector-size-m 0.35
```

This writes a `.msh` file plus a JSON manifest and is the same path invoked by `prepare_concentric_case.py` and `./Allmesh`.

To regenerate the earlier full cylinder mesh for comparison only:

```bash
python genfoam/gmsh_reactor_mesh.py generate \
  --mesh-kind full-3d-experimental \
  --mesh-file genfoam/constant/generated/concentric_reactor_3d.msh
```

`gmsh_reactor_mesh.py` now requires a direct `import gmsh` from the active Python environment. The temporary vendored `genfoam/_vendor/gmsh_py` fallback has been removed.

## Plotting Results

`plot_results.py` supports the legacy r-z block manifest and the new axisymmetric wedge manifest:

```bash
python genfoam/plot_results.py
```

## Current Validation

The generated case has passed these checks:

- `conda run -n openmc python genfoam/prepare_concentric_case.py`
- `./Allmesh`
- `gmshToFoam` on the generated `concentric_reactor_wedge.msh`
- `conda run -n openmc python genfoam/gmsh_reactor_mesh.py configure-import ...`
- `checkMesh -region neutroRegion`

Current `checkMesh` status for the imported wedge mesh:

- mesh topology and region import are valid
- all expected reactor regions are present as OpenFOAM cell zones
- wedge patches are rewritten to OpenFOAM `wedge` patch types after `gmshToFoam`
- the centerline patch is rewritten to `symmetryPlane`

## openmcToFoam Reference Path

`openmcToFoam` is integrated as a separate comparison workflow, not as the default nuclear-data path. It reruns OpenMC with the `openmcToFoam` tally set, writes a reference `nuclearData` file, and compares the resulting zone vectors against the local hand-written generator:

```bash
conda run -n openmc python genfoam/openmc_to_foam_reference.py
```

The helper now expects `MultiGroupXS` to be importable in the active Python environment. If it is not installed into that environment, pass `--tool-root /path/to/openmcToFoam` or set `OPENMCTOFOAM_ROOT`.

Outputs are written under `genfoam/constant/generated/openmc_to_foam_reference/`.

## Next Steps
- Compare wedge `k_eff` and flux smoothness against the experimental full-3D path.
- Review the `openmcToFoam` comparison report and decide whether to cut over the default `nuclearData` generator.
- Make the generator target a selected group sweep so you can switch between 2, 8, 16, 25, and 40-group cases without editing files by hand.

## Visualization with ParaView
- either call `paraFoam -region {neutroRegion}` should work.
- or create a temporary `neutroRegion.foam` file that then you open with paraview
