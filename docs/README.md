# Documentation

The top-level [README](../README.md) is the project overview and local
quick-start. This directory contains the detailed physics, numerical methods,
runtime coupling, and historical material.

## Current Model

- [OpenMC-Informed Reactor Model](reactor_model/reactor_model_theory.pdf)
  ([LaTeX source](reactor_model/reactor_model_theory.tex)) describes the
  continuous-energy reference model, published OpenMC data, point kinetics,
  resolved four-group diffusion, power-shape correction, and webapp coupling.
- [Thermal Hydraulics](TH_latex/ThermalHydraulics.pdf)
  ([LaTeX source](TH_latex/ThermalHydraulics.tex)) describes the Modelica
  primary loop, fuel heat structure, effective feedback state, and model
  limitations.
- [System Architecture](system_architecture.md) shows the offline OpenMC data
  path and the online React, FastAPI, diffusion, and FMU components.
- [FMU Coupling](fmu_coupling.md) documents FMU validation, initialization,
  time stepping, feedback, fallback behavior, and troubleshooting.
- [OpenMC Workflow](openmc_workflow.md) explains how the current concentric
  reactor and its reference artifacts are produced.
- [MGXS to Diffusion Workflow](diffusion_mgxs_workflow.md) defines the
  scattering, adapter, finite-volume, cache, Core-page, and offline SPH
  contracts.

## Historical Material

- [Legacy Homogeneous Reactor Study](reactor_model/legacy_homogeneous_reactor.pdf)
  ([LaTeX source](reactor_model/legacy_homogeneous_reactor.tex)) documents the
  first buckling and one-group annular models plus the retired Core and direct
  transient-diffusion backends.
- [One-Group Homogeneous Notebook](oneGroupHomogeneous.ipynb) is the executable
  analysis behind that legacy document.
- [Seminar Presentation](doc.pdf) is the presentation that motivated the
  current documentation narrative. It is useful context, but current code and
  tracked reference artifacts are authoritative for numerical values.

## Sources of Truth

Normal webapp operation does not run OpenMC. Published data under
`openmc/reference_data/concentric/` are immutable runtime inputs:

- `group_sweep/group_4` supplies the Core-page MGXS, delayed-neutron constants,
  prompt generation time, CE comparison data, and clean power reference.
- `rod_scan/results/rod_worth.csv` supplies point-kinetics rod worth.
- `reactivity_coefficients/results/reactivity_coefficients.csv` supplies fuel,
  moderator-temperature, and moderator-density feedback slopes.

Runtime settings are defined in `backend/reactor_backend/config.py`. The
Modelica source is `modelica/ResearchReactorThermalHydraulics.mo`; a generated
FMU is a platform-specific build artifact.
