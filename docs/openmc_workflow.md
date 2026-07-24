# OpenMC Reactor Workflow

## Purpose

OpenMC defines the current reactor and produces the reference data consumed by
the Python backend. OpenMC is an offline tool in this project: normal webapp
requests load tracked JSON, XML, and CSV artifacts and do not launch a Monte
Carlo calculation.

## Model Development

The reactor was developed in three stages:

1. `docs/oneGroupHomogeneous.ipynb` explored a homogeneous buckling estimate
   and a one-group annular diffusion model.
2. `openmc/involute_reactor.ipynb` explored an FRM-II-inspired involute
   natural-uranium fuel concept.
3. `openmc/concentric_reactor.ipynb` established the concentric annular reactor
   used by the current reference-data and diffusion workflows.

The concentric geometry was selected because it retains resolved fuel and
moderator regions while remaining compatible with axisymmetric MGXS tallies
and a practical two-dimensional diffusion solve. It is inspired by
research-reactor geometry rather than intended as a replica of a licensed
facility.

## Model-Building Modules

The reusable OpenMC implementation is split by responsibility:

| Module | Responsibility |
| --- | --- |
| `concentric_fuel.py` | Builds concentric fuel-ring, coolant, moderator, and parked-rod cells. |
| `fuel_element.py` and `involutes.py` | Build the earlier involute fuel concept. |
| `reactor_geometry.py` | Places a selected fuel element inside moderator and reflector tanks. |
| `build_simulation.py` | Builds an eigenvalue model, settings, meshes, and standard tallies. |
| `ploting.py` | Provides geometry and tally plotting helpers. |
| `mgxs_export.py` | Builds resolved MGXS tallies, validates exports, and publishes reference sweeps. |
| `concentric_reactor_rodworth.py` | Runs the continuous-energy rod-insertion scan. |
| `concentric_reactor_reactivity.py` | Runs fuel-temperature, moderator-temperature, and density perturbations. |

The tracked four-group publication is a frozen result. Its `model.xml` is the
geometry source for the runtime diffusion adapter even if defaults in a model
builder later change.

## Offline Data Pipeline

### 1. Continuous-Energy Reactor

`openmc/concentric_reactor.ipynb` constructs and runs the concentric
continuous-energy model. It is used to inspect geometry, criticality, flux,
fission, entropy, and power shapes.

### 2. Rod Worth and Feedback Coefficients

The continuous-energy perturbation scripts publish:

```text
openmc/reference_data/concentric/
  rod_scan/results/rod_worth.csv
  reactivity_coefficients/results/reactivity_coefficients.csv
```

The point-kinetics backend loads these files directly. Rod worth is linearly
interpolated by insertion fraction. Thermal-feedback slopes are applied only
when the Modelica FMU supplies effective fuel temperature, moderator
temperature, and moderator density.

### 3. MGXS Generation and Validation

`openmc/exportMGXS.ipynb` orchestrates `mgxs_export.py`. For each requested
group structure, the workflow:

1. loads or builds the resolved continuous-energy model;
2. attaches cell-wise MGXS and delayed-neutron tallies;
3. runs the continuous-energy calculation;
4. writes canonical diffusion data;
5. optionally rebuilds a supplemental OpenMC multigroup library for transport
   validation;
6. publishes the complete group sweep atomically.

The current webapp uses:

```text
openmc/reference_data/concentric/group_sweep/group_4/
  reactor_run/model.xml
  outputs/mgxs_constants.json
```

The JSON contains resolved multigroup constants, prompt generation time,
delayed-neutron data, CE region/group flux tallies, and a cylindrical
`kappa-fission` power reference. The XML supplies exact material boundaries
and cell names.

### 4. Diffusion Verification

`openmc/diffusion_concentric_reactor.ipynb` consumes the published four-group
artifact through the same backend adapter and cache used by the webapp. It
checks:

- the XML-to-MGXS region mapping;
- transport-derived diffusion coefficients;
- the raw four-group eigenvalue and flux field;
- OpenMC multigroup and continuous-energy comparisons;
- the clean CE power-shape correction;
- equivalent-absorber rod-shape behavior.

The detailed scattering, finite-volume, cache, SPH, and online contracts are in
[`diffusion_mgxs_workflow.md`](diffusion_mgxs_workflow.md).

## Runtime Consumers

| Reference artifact | Backend consumer | Runtime role |
| --- | --- | --- |
| Four-group `mgxs_constants.json` | `kinetics_reference.py` | Six delayed groups and prompt generation time |
| Four-group JSON and `model.xml` | `openmc_mgxs_adapter.py` | Resolved Core-page diffusion input |
| CE power mesh in the JSON | `power_shape.py` | Clean power-shape correction |
| `rod_worth.csv` | `rod_worth.py` | Point-kinetics rod reactivity |
| `reactivity_coefficients.csv` | `reactivity_coefficients.py` | FMU thermal feedback |

The independent rod-scan CE result and the CE run associated with the
four-group MGXS are separate references. Their clean `k_eff` values are not
expected to be identical and are used for different comparisons.

## Generated and Tracked Data

Raw OpenMC statepoints, logs, temporary XML, and run directories belong under
the gitignored `openmc/build/` tree. Only compact artifacts promoted under
`openmc/reference_data/` should be treated as versioned application inputs.

The GeN-Foam export is an adjacent deterministic-code experiment. It is not
part of the current webapp runtime path.
