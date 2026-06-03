# GeN-Foam Concentric Reactor Case

This directory now contains a generated GeN-Foam diffusion case based on the OpenMC concentric reactor model.

## Current Scope

The current implementation uses the structure of the GeN-Foam `2D_externalSourceDiffusion` feature case, but replaces its neutronics data and mesh with OpenMC-derived inputs from the concentric reactor workflow. The generator now:

- parses the built OpenMC `model.xml`
- loads the exported MGXS constants from `openmc/build/concentric/mgxs_export`
- converts geometry from cm to m and MGXS data to SI-style units
- collapses higher-order scattering exports to the zeroth Legendre moment for diffusion use
- sanitizes inactive-group diffusion coefficients
- sanitizes zero-removal and zero-inverse-velocity groups that break the GeN-Foam solve
- assigns valid precursor decay constants in non-fuel zones so diagonal precursor solves remain well-posed
- writes a non-overlapping r-z block layout and the corresponding GeN-Foam case files

## Usage

From `ReactorKineticsLab/`:

```bash
python3 genfoam/prepare_concentric_case.py
```

Optional arguments:

```bash
python3 genfoam/prepare_concentric_case.py \
  --mgxs-export-dir openmc/build/concentric/mgxs_export \
  --output-dir genfoam/constant/generated
```

## Generated Files

The script writes the following metadata files under `genfoam/constant/generated/`:

- `concentric_case_manifest.json` — combined source, geometry, zone, block, and material payload
- `concentric_materials.json` — per-domain multigroup constants and delayed-neutron data in converted units
- `concentric_overlay_zones.csv` — ordered overlay zones matching the resolved OpenMC geometry logic
- `concentric_mesh_blocks.csv` — disjoint r-z blocks used to build the current GeN-Foam wedge mesh and material zones

It also writes the active GeN-Foam case files in place:

- `system/controlDict`, `system/regionsDict`
- `system/neutroRegion/blockMeshDict`, `fvSchemes`, `fvSolution`
- `constant/neutroRegion/neutronicsProperties`, `nuclearData`
- `0/neutroRegion/defaultFlux` through `defaultFlux16`
- `Allclean`, `Allmesh`, `Allrun`

## Plotting Results

Use the small post-processing helper to plot flux-relevant outputs from the latest GeN-Foam time directory:

```bash
python3 genfoam/plot_results.py
```

By default it writes `genfoam/plots/results_<time>.png` and includes:

- `oneGroupFlux`
- `powerDensity` when present
- the first available multigroup flux field
- the last available multigroup flux field
- volume-weighted radial and axial profiles of `oneGroupFlux`

Useful options:

```bash
python3 genfoam/plot_results.py --time 1
python3 genfoam/plot_results.py --fields oneGroupFlux,flux0,flux15,powerDensity
python3 genfoam/plot_results.py --output genfoam/plots/custom_flux.png
```

## Current Validation

The generated case has passed these checks:

- `python3 genfoam/prepare_concentric_case.py`
- `./Allmesh`
- `GeN-Foam -case .`

The first successful 16-group run from the current `mgxs_export` input advances through the neutronics solve and reports `k_eff` near `1.03267` for the current generated case.

## Next Implementation Step

The next useful step is to tighten the generated nuclear-data translation against GeN-Foam semantics rather than broadening the case structure. In practice that means validating the interpretation of `Beta`, `chiDelayed`, and `sigmaPow` against the solver documentation and then comparing the resulting `k_eff` against the OpenMC multigroup validation run for the same export directory.