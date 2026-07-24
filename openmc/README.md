# OpenMC Reactor Modeling

This directory contains the continuous-energy reactor models and offline
reference-data workflows used by Reactor Kinetics Lab.

The project moved from the exploratory one-group model in
`docs/oneGroupHomogeneous.ipynb` to OpenMC so geometry, energy dependence, rod
worth, and feedback coefficients could be derived from a transport model.

## Reactor Concepts

- `involute_reactor.ipynb` explores an FRM-II-inspired involute natural-uranium
  fuel element.
- `concentric_reactor.ipynb` develops the concentric annular reactor used by
  the current reference-data workflow.
- `diffusion_concentric_reactor.ipynb` validates the resolved four-group
  diffusion model consumed by the backend Core page.
- `exportMGXS.ipynb` generates and publishes multigroup cross sections and
  delayed-neutron data.

## Main Modules

- `concentric_fuel.py`: concentric fuel-ring geometry and materials
- `fuel_element.py`, `involutes.py`: involute fuel-element geometry
- `reactor_geometry.py`: moderator and reflector tank geometry
- `build_simulation.py`: complete OpenMC model, settings, and standard tallies
- `mgxs_export.py`: MGXS generation, validation, and publication
- `concentric_reactor_rodworth.py`: continuous-energy rod-worth scan
- `concentric_reactor_reactivity.py`: thermal and density coefficient scans
- `ploting.py`: plotting helpers retained under the existing filename

Raw runs are generated under `openmc/build/`. Compact application inputs are
published under `openmc/reference_data/`.

See [`docs/openmc_workflow.md`](../docs/openmc_workflow.md) for the complete
model history, data pipeline, and runtime consumers.
