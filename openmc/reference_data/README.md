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

- `theory/concentricModel.ipynb`
- `backend/reactor_backend/multigroup_service.py`
- `genfoam/prepare_concentric_case.py`

These exports describe the clean resolved core. They do not replace the
backend point-kinetics rod-worth calibration or its transient model constants.
