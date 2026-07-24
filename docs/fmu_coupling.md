# Modelica FMU Coupling

This document describes how the Python backend couples the point-kinetics
simulation to the Modelica thermal-hydraulics model.

The coupling is implemented in `backend/reactor_backend/thermal_adapter.py`
and is owned by `backend/reactor_backend/service.py`.

For the full webapp data-flow diagram, see `docs/system_architecture.md`.

## Runtime Overview

The frontend does not run Modelica directly. It polls the FastAPI backend for
the latest simulation state. The backend advances three coupled pieces:

1. `ReactorEngine` advances point kinetics and computes reactor power.
2. `ThermalAdapter` advances the Modelica FMU or fallback thermal model.
3. `reactivity.py` converts rod worth plus FMI thermal feedback into total
   reactivity.

The thermal adapter returns a `ThermalSnapshot` through the API. The Overview
page uses that snapshot for the HMI values and thermal-hydraulic trend chart.

## Modelica Interface

The backend expects an FMU exported from:

```text
modelica/ResearchReactorThermalHydraulics.mo
```

The default FMU path is:

```text
modelica/build/ResearchReactorThermalHydraulics.fmu
```

The FMU is expected to expose a co-simulation interface with these inputs:

| FMU variable | Direction | Unit | Backend source |
| --- | --- | --- | --- |
| `totalPower` | input | W | point-kinetics thermal power |
| `axialPowerFractions[i]` | input | 1 | rod-aware 8-node axial power shape |

The FMU is expected to expose these outputs:

| FMU variable | Unit | API field |
| --- | --- | --- |
| `T_inlet` | K | `thermal.inletTemperatureK` |
| `T_outlet` | K | `thermal.outletTemperatureK` |
| `T_fuelCenterlineMax` | K | `thermal.fuelMaximumTemperatureK` |
| `T_fuelEff` | K | `thermal.fuelTemperatureK` |
| `T_moderatorEff` | K | `thermal.moderatorTemperatureK` |
| `rho_m_eff_SI` | kg/m3 | `thermal.moderatorDensityKgPerM3` |
| `rho_m_eff` | g/cm3 | `thermal.moderatorDensityGPerCm3` |
| `massFlow` | kg/s | `thermal.massFlowKgPerSecond` |
| `dp_core` | Pa | `thermal.corePressureDropPa` |

## FMU Build And Compatibility Check

`ThermalAdapter._ensure_fmu()` decides whether to reuse an FMU or export a new
one with OpenModelica.

An existing FMU is reused only when all of these are true:

- the FMU file exists
- the FMU is newer than the Modelica source file
- the FMU was built with the expected FMI flags
- the FMU exposes a co-simulation interface
- all required input/output variables are present
- at least one `axialPowerFractions[i]` input is present

These checks validate the FMI contract and build flags, but they do not
currently verify that the FMU contains a binary for the host platform. The
tracked FMU contains `binaries/linux64`; it cannot be instantiated directly on
macOS. A non-Linux host must export a local FMU with OpenModelica after removing
the Linux artifact, or the adapter will use its fallback model.

If any check fails, the adapter deletes the stale FMU and runs an OpenModelica
script equivalent to:

```modelica
setCommandLineOptions("--fmiFlags=s:cvode");
system("mkdir -p build");
cd("build");
loadModel(Modelica);
loadFile("../ResearchReactorThermalHydraulics.mo");
buildModelFMU(ResearchReactorThermalHydraulics, version="2.0", fmuType="cs");
```

The stale FMU is removed before export so a failed export cannot silently leave
an old, incompatible FMU in service.

## FMU Initialization

`ThermalAdapter._ensure_instance()` performs the runtime initialization:

1. Reads the FMU model description with FMPy.
2. Finds value references for all required variables.
3. Extracts the FMU zip into a temporary directory.
4. Creates an `FMU2Slave` co-simulation instance.
5. Instantiates and enters initialization mode.
6. Writes the initial `totalPower` and axial power fractions.
7. Exits initialization mode.
8. Reads the initial thermal outputs into a `ThermalSnapshot`.

`ThermalAdapter.close()` terminates the FMU instance, frees the slave, and
removes the temporary extraction directory.

## Time Stepping

`SimulationService` controls when the thermal model is advanced.

The point-kinetics engine steps with `SIMULATION_TUNING.integrator_step_seconds`.
Thermal updates are batched and flushed every:

```python
max(SIMULATION_TUNING.thermal_update_seconds,
    SIMULATION_TUNING.integrator_step_seconds)
```

For each thermal batch, the service:

1. Computes an 8-node axial power profile from the current rod insertion with
   `compute_axial_power_fractions_8()`.
2. Calls `ThermalAdapter.set_axial_fractions(...)`.
3. Calls `ThermalAdapter.step(fmu_power_mw, dt_seconds, reported_power_mw=...)`.
4. Sends the returned thermal snapshot back to `ReactorEngine` for feedback.

`fmu_power_mw` is the time-averaged power over the thermal batch. The reported
power is the latest point-kinetics power used for display.

The axial fractions currently come from
`backend/reactor_backend/core_service.py`, which runs the retired one-group
diffusion shape helper. They do not yet come from the four-group CE-corrected
Core-page power map. Replacing this remaining legacy dependency is a current
coupling limitation.

## Reactivity Feedback

The point-kinetics model only applies thermal feedback when the thermal
snapshot source is `fmu`.

`reactivity.py` requires all of these to be available before feedback is active:

- effective fuel temperature
- effective moderator temperature
- effective moderator density

The feedback terms are:

```text
fuel temperature feedback
moderator temperature feedback
moderator density feedback
```

The coefficients are loaded from the OpenMC-generated reactivity-coefficient
reference data in `backend/reactor_backend/reactivity_coefficients.py`.

On reset, the initial active FMU thermal state is captured as the zero-feedback
reference. This makes the reset state internally consistent even if the FMU
initial temperatures differ from the OpenMC coefficient baseline.

If the thermal snapshot source is `fallback` or `unavailable`, the backend
reports zero thermal-feedback reactivity.

## Fallback Thermal Model

If FMU export, initialization, or stepping fails, the adapter switches to a
fallback model instead of breaking the reactor controls.

The fallback model provides:

- first-order inlet/outlet temperature response
- fixed mass flow
- fixed core pressure drop
- approximate effective moderator temperature
- approximate D2O density from the same linear `rhoD2O_pT` form used by the
  Modelica model
- approximate fuel effective and maximum temperatures

Fallback values are useful for keeping the HMI populated during development,
but they are not used for point-kinetics thermal feedback. Thermal feedback is
FMI-only by design.

## Troubleshooting

If the HMI shows `n/a` or the API reports `thermal.source` as `fallback`, check
`thermal.message` in `/api/simulation/state`.

Common causes:

- the FMU is older than `ResearchReactorThermalHydraulics.mo`
- the FMU does not contain a binary for the current operating system
- the FMU is missing required variables such as `T_fuelEff` or `rho_m_eff`
- the FMU was built without the expected `s:cvode` FMI flag
- OpenModelica failed to export a fresh FMU
- FMPy could not instantiate or step the co-simulation slave

To test the same export path used by the backend:

```bash
cd modelica
omc
```

Then run:

```modelica
setCommandLineOptions("--fmiFlags=s:cvode");
system("mkdir -p build");
cd("build");
loadModel(Modelica);
loadFile("../ResearchReactorThermalHydraulics.mo");
buildModelFMU(ResearchReactorThermalHydraulics, version="2.0", fmuType="cs");
```

To test the Python adapter directly:

```bash
backend/.venv/bin/python -c 'import sys; sys.path.insert(0, "backend"); from reactor_backend.thermal_adapter import ThermalAdapter; a = ThermalAdapter(); print(a.reset(20.0).model_dump()); a.close()'
```

The result should report `source: "fmu"` for true Modelica coupling. If it
reports `source: "fallback"`, the `message` field is the first place to look.
