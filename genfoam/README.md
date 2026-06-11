# GeN-Foam Concentric Reactor Case

This directory contains an axisymmetric GeN-Foam diffusion model of the
concentric OpenMC reactor.

## Prerequisites

- OpenFOAM and foamForNuclear available in the shell
- the `openmc` Conda environment with OpenMC, Gmsh, and `MultiGroupXS`
- an OpenMC model export under `openmc/build/concentric/mgxs_export/`

Run Python commands from the repository root with the `openmc` environment
active, or use `conda run -n openmc` as shown below.

## Workflow

The workflow has three distinct stages:

1. `prepare_concentric_case.py` reads or regenerates MGXS data, writes
   GeN-Foam `nuclearData` and initial flux files, and generates the Gmsh
   `.msh` file and mesh manifest.
2. `Allmesh` imports that prepared `.msh` file with `gmshToFoam`, configures
   patches and cell zones, runs `checkMesh`, and validates the import. It does
   not regenerate the Gmsh geometry.
3. `Allrun` calls `Allmesh` and then runs GeN-Foam.

For the first run, or after changing MGXS settings:

```bash
conda run -n openmc python genfoam/prepare_concentric_case.py --rerun-mgxs
cd genfoam
./Allclean
./Allrun
```

Without particle-count overrides, `--rerun-mgxs` reuses the run settings
recorded in the source export. The current 16-group source records 20,000
particles, 100 batches, and 10 inactive batches.

Once a compatible MGXS export exists, preparation can reuse it without an
OpenMC run:

```bash
conda run -n openmc python genfoam/prepare_concentric_case.py
```

`Allclean` removes time directories, imported `polyMesh`, and logs. It keeps
the prepared files under `constant/generated/`, so `Allrun` can re-import the
same mesh without rerunning OpenMC or Gmsh.

## Geometry And Mesh

The active geometry is a thin axisymmetric wedge representing the reactor
`r-z` plane. Dimensions and canonical region names are maintained manually
from the OpenMC reference model; `model.xml` is not parsed to construct the
Gmsh geometry.

The mesh preserves material boundaries and uses independent radial and axial
subdivision counts. Default controls are defined by `MeshSizingDefinition` in
`gmsh_reactor_mesh.py`. Geometry dimensions are not changed when the mesh is
refined.

To generate a standalone mesh with explicit subdivisions:

```bash
conda run -n openmc python genfoam/gmsh_reactor_mesh.py generate \
  --mesh-kind axisymmetric-wedge \
  --wedge-angle-deg 1.0 \
  --fuel-radial-divisions 3 \
  --fuel-axial-divisions 12 \
  --core-radial-divisions 2 \
  --core-axial-divisions 2
```

This writes:

- `constant/generated/concentric_reactor_wedge.msh`
- `constant/generated/concentric_reactor_mesh_manifest.json`

The centerline is represented by a very small inner cylindrical symmetry
patch to avoid collapsed-axis defects during `gmshToFoam` import. `Allmesh`
rewrites the wedge and centerline patch types and validates all expected cell
zones.

The earlier full-cylinder generator remains experimental and is not used by
`Allmesh` or `Allrun`:

```bash
conda run -n openmc python genfoam/gmsh_reactor_mesh.py generate \
  --mesh-kind full-3d-experimental \
  --mesh-file genfoam/constant/generated/concentric_reactor_3d.msh
```

## MGXS Contract

`openmc/mgxs_export.py` is the MGXS source of truth. The local
`openmc_to_genfoam_xs.py` writer converts its results to the GeN-Foam
`states ( reference { zones (...) } )` syntax.

Compatible exports contain:

- `consistent nu-scatter matrix`
- `chi-prompt`
- `scatter_correction = null`
- one-group `Beta` and decay-rate kinetics data

The writer uses the P0 moment of the consistent nu-scatter matrix and computes:

```text
sigmaRemoval[g] = total[g] - scatteringMatrixP0[g,g]
```

`legendre_order=0` and `scatter_correction=None` are separate settings. P0
selects the isotropic Legendre moment; disabling the correction prevents
OpenMC from modifying its diagonal with a transport correction.

Exports created before this contract was added are rejected. Regenerate them
with `--rerun-mgxs` or rerun `openmc/exportMGXS.ipynb` after reloading the
current `mgxs_export` module. New notebook exports use the compatible defaults.

Optional preparation overrides:

```bash
conda run -n openmc python genfoam/prepare_concentric_case.py \
  --rerun-mgxs \
  --openmc-particles 2000 \
  --openmc-batches 12 \
  --openmc-inactive 4 \
  --openmc-threads 8 \
  --legendre-order 0
```

The adapter sanitizes only solver-invalid values such as non-positive
diffusion/removal coefficients in inactive or statistically unresolved groups.
Every intervention is listed in
`constant/generated/mgxs_to_genfoam_xs/summary.json`.

## Reference Comparison

`openmc_to_foam_reference.py` runs the installed openmcToFoam
`MultiGroupXS` implementation and compares it with the local export/writer.
Use identical statistics and scattering order on both sides:

```bash
conda run -n openmc python genfoam/openmc_to_foam_reference.py \
  --rerun-local-mgxs \
  --local-particles 2000 \
  --local-batches 12 \
  --local-inactive 4 \
  --openmc-to-foam-particles 2000 \
  --openmc-to-foam-batches 12 \
  --openmc-to-foam-inactive 4 \
  --local-legendre-order 0 \
  --openmc-to-foam-legendre-order 0
```

With the corrected contract, the two paths agree on
`scatteringMatrixP0`, `chiPrompt`, `Beta`, and `lambda`; `sigmaRemoval`
differs only by floating-point roundoff.

## Generated And Static Files

`prepare_concentric_case.py` writes:

- `constant/neutroRegion/nuclearData`
- `0/neutroRegion/defaultFlux` through `defaultFlux16`
- `0/neutroRegion/defaultExternalSourceFlux`
- `constant/generated/concentric_case_manifest.json`
- `constant/generated/concentric_reactor_wedge.msh`
- `constant/generated/concentric_reactor_mesh_manifest.json`
- `constant/generated/concentric_mesh_regions.csv`
- `constant/generated/concentric_materials.json`
- `constant/generated/mgxs_to_genfoam_xs/`

The following case scaffolding remains static and checked in:

- `system/controlDict` and `system/regionsDict`
- `system/neutroRegion/fvSchemes` and `fvSolution`
- `constant/neutroRegion/neutronicsProperties`
- `Allclean`, `Allmesh`, and `Allrun`

## Validation And Visualization

`Allmesh` performs:

- `gmshToFoam`
- imported boundary and cell-zone configuration
- `checkMesh -region neutroRegion`
- manifest-based patch and cell-zone validation

Plot extracted results with:

```bash
conda run -n openmc python genfoam/plot_results.py
```

For ParaView:

```bash
cd genfoam
paraFoam -region neutroRegion
```

Alternatively, create `genfoam/neutroRegion.foam` and open it directly in
ParaView.
