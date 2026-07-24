# OpenMC Reference Data

This directory contains compact, versioned inputs promoted from OpenMC runs.
Raw statepoints, HDF5 files, logs, and run directories remain under the
gitignored `openmc/build/` tree.

The multigroup sweep is published to:

```text
concentric/group_sweep/
  manifest.json
  group_<N>/
    reactor_run/model.xml
    outputs/mgxs_constants.json
```

Use `publish_group_sweep` from `openmc/mgxs_export.py`, or enable
`PUBLISH_REFERENCE_SWEEP` in `openmc/exportMGXS.ipynb`. The publisher validates
the group structure and replaces the complete published sweep atomically.

The four-group publication is the default input for:

- `openmc/diffusion_concentric_reactor.ipynb`
- `backend/reactor_backend/multigroup_service.py`
- `backend/reactor_backend/kinetics_reference.py`
- `genfoam/prepare_concentric_case.py`

The publication describes the clean resolved core. Its MGXS, prompt generation
time, and delayed-neutron data feed the current backend. Point-kinetics rod
worth and thermal-feedback coefficients are published separately under
`concentric/rod_scan/` and `concentric/reactivity_coefficients/`.
