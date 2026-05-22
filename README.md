# Reactor Web App

A localhost reactor simulator with a **React + Bun frontend** and a **Python backend** for the reactor physics.

The current version models a simplified **heavy-water-moderated annular core** with one operator control: **control rod insertion**. The browser dashboard shows how **reactivity**, **total neutron flux**, and **thermal power** evolve over time, while the transient is solved in Python.

## What the app does

- lets you move a **single control rod bank** from fully withdrawn to fully inserted
- computes the resulting **reactivity**
- evolves the reactor with **point kinetics and delayed neutron groups**
- displays:
  - reactivity
  - total flux
  - thermal power
  - simulated time
  - neutron period estimate
- supports:
  - pause / resume
  - reset to critical
  - manual SCRAM
  - automatic SCRAM on overpower

## Running locally

### Prerequisites

- [Bun](https://bun.sh/) 1.3+
- [uv](https://docs.astral.sh/uv/) for Python environment and dependency management

### Install frontend dependencies

```bash
bun run frontend:install
```

### Create the backend environment with UV

```bash
bun run backend:install
```

This runs `uv sync` inside `backend/`, creates the backend virtual environment, and installs the Python dependencies from `backend/pyproject.toml`.

### Start the hybrid dev stack

```bash
bun run dev
```

That launches:

- the UV-managed Python backend on `http://127.0.0.1:8000`
- the Vite frontend on `http://127.0.0.1:5173`

Open `http://127.0.0.1:5173`.

### Quick backend check

```bash
curl http://127.0.0.1:8000/api/health
```

## Hybrid architecture

## Frontend

The frontend is a **Bun-managed React + TypeScript** app.

Its job is to:

- render the dashboard
- send user commands to the backend
- poll the latest state from the backend
- display the rolling history charts and core schematic

Key frontend files:

- `frontend/src/App.tsx`
  - dashboard composition and control wiring
- `frontend/src/hooks/useReactorSimulation.ts`
  - API polling, command calls, and React state integration
- `frontend/src/simulation/api.ts`
  - frontend API client for the Python service
- `frontend/src/components/`
  - schematic, charts, and metric cards

Vite proxies `/api` requests to the Python backend during development.

## Backend

The backend is a small **FastAPI** service in `backend/reactor_backend/`.

Its job is to:

- own the reactor engine state
- advance the transient over elapsed wall time when the simulation is running
- keep rolling history on the server side
- expose the current snapshot and control endpoints to the frontend

Key backend files:

- `backend/reactor_backend/config.py`
  - reactor constants and simulation tuning
- `backend/reactor_backend/reactivity.py`
  - rod-worth to reactivity mapping
- `backend/reactor_backend/engine.py`
  - point-kinetics engine
- `backend/reactor_backend/service.py`
  - stateful simulation service, history, and timing
- `backend/reactor_backend/app.py`
  - FastAPI routes

## API shape

The current backend exposes:

- `GET /api/health`
- `GET /api/simulation/state`
- `POST /api/simulation/reset`
- `POST /api/simulation/rod-insertion`
- `POST /api/simulation/running`
- `POST /api/simulation/scram`

The frontend treats the backend as the **source of truth** for:

- current reactor snapshot
- rolling history
- running / paused state
- reactor constants used for display

## How the simulation works

## 1. Core model

The reactor is treated as a simplified **annular core** (Estimate 2 geometry from `theory/reactorModel.ipynb`):

- inner radius: **0.8 m**
- outer radius: **3.456 m** (R_fuel = 345.6 cm from 2D critical search)
- active height: **6.912 m** (aspect ratio ≈ 1 for the full annular zone)

Heavy water moderation is a **fixed assumption** in this version.

## 2. Reactivity model

The only operator input is **rod insertion percent**.

Rod position is converted into reactivity using a **2D-calibrated rod-worth table** derived from `theory/reactorModel.ipynb`:

- The clean, unrodded core is slightly supercritical in the 2D solve: `k_eff ≈ 1.000395` (`ρ ≈ +39.4 pcm`).
- The table stores **rod-only** `Δρ(x)` at 11 points (x = 0.0 to 1.0 in steps of 0.1) from a 2D r-z finite-difference one-group diffusion eigenvalue scan.
- Backend interpolation is linear between table points.
- Total reactivity is computed as `ρ_total(x) = ρ_base(clean) + Δρ_rod(x) + ρ_scram`.
- The rod is modeled as a **combined control/shutdown bank** with effective parameters:
  - equivalent radius `r_rod = 50 cm`
  - effective absorber increment `ΔΣ_a,max = 0.25 cm⁻¹` (one-group homogenized value)

Current calibration values (frozen from the notebook):

| x (insertion fraction) | Δρ (pcm) |
|---|---|
| 0.0 | 0.0 (reference) |
| 0.1 | −4.7 |
| 0.2 | −16.3 |
| 0.5 | −84.7 |
| 1.0 | −159.6 |

Derived operating points:

- critical insertion: **~32%**
- full insertion total reactivity: **~−120 pcm** (without extra SCRAM penalty)
- extra SCRAM shutdown margin (latched trip): **450 pcm**

The dashboard shows reactivity in **pcm**, while the kinetics solver uses **delta-k/k** internally:

- `1 pcm = 1e-5 delta-k/k`
- `beta_eff = 0.00651`
- `beta_eff = 651 pcm`

## 3. Point-kinetics solver

The transient uses:

- **6 delayed neutron groups**
- `beta_eff = 0.00651`
- neutron generation time: **5e-4 s**

The engine evolves:

- neutron population
- delayed neutron precursor concentrations

From that, it derives:

- thermal power
- total flux

The integration step is **implicit**, which keeps the browser-driven local workflow stable for this stiff kinetics system.

### Nominal scaling

Displayed values are scaled from nominal operating conditions:

- nominal thermal power: **20 MWth**
- nominal flux: **1.5 × 10¹² n/cm²/s** (Estimate 2 annular-core peak, 2D solution)

## 4. Time stepping and history

The backend advances the model using:

- integrator step: **0.02 s**
- history sampling: **0.25 s**
- chart history length: **240 points**
- simulation speed: **8x wall clock**
- frontend poll interval: **100 ms**

The backend, not the browser, is responsible for:

- simulated elapsed time
- history accumulation
- pause / resume correctness
- capping long wall-time gaps

## 5. Safety behavior

There are two shutdown paths:

1. **Manual SCRAM** from the dashboard
2. **Automatic SCRAM** if power exceeds **26 MWth** (1.3 × nominal)

When SCRAM is latched:

- the rod bank is forced to **100% inserted**
- an extra shutdown reactivity penalty is applied
- the slider remains disabled until reset

## What is simplified

This is still an educational first slice, not a full reactor code.

Intentional simplifications:

- one effective rod bank instead of a full control system
- no thermal-hydraulic feedback
- no xenon, temperature coefficients, void coefficients, or burnup
- fixed heavy-water moderation assumptions
- flux and power are scaled from nominal values
- point kinetics only, with no spatial flux solution

## Likely next steps

Now that the physics layer is in Python, natural next steps are:

- see `theory/ThermalHydraulics.tex` for a first-pass thermal-hydraulic sizing
  and Modelica architecture note tied to the current 20 MW annular-core geometry
- add subpage where for time dependent diffusion with flux visualizations
  - could also use standard point kinetics but rod insertion reactivity be calculated via diffusion?
- integrate Modelica FMU for simple systems thermal hydraulics
- The **prompt generation time** $\Lambda$ could benefite from an adjoint solve i think (or better, an OpenMC calculation so you get: multigroup coefficients, beta_eff and lambda for your 1 or 2 group diffusion simulation model)

- FMU build handlded outside of python and frozen (not gitignored)? or handled by a script, like with bun run backend:install?

- architecture documentation:

i.e for python thermal_adapter.py you mentioned (older arch perhaps)

FMU build/export management
In thermal_adapter.py:275, _ensure_fmu() decides whether to reuse an existing FMU or run omc to export a fresh one.
In thermal_adapter.py:309, _build_export_script() generates the .mos script for OpenModelica.
In thermal_adapter.py:323, _fmu_has_expected_flags() checks that the FMU was built with the expected solver flags.

FMU runtime lifecycle
In thermal_adapter.py:149, _ensure_instance() loads the FMU metadata, finds variable references, extracts the FMU zip, instantiates the co-simulation slave, enters initialization mode, pushes initial inputs, exits initialization, and reads initial outputs.
In thermal_adapter.py:342, _set_inputs() writes totalPower and axialPowerFractions.
In thermal_adapter.py:349, _read_outputs() reads T_inlet, T_outlet, massFlow, and dp_core.
In thermal_adapter.py:141, close() terminates the FMU instance, frees it, and deletes the extracted working directory.

Failure containment / fallback model
In thermal_adapter.py:218, _activate_fallback() switches to a simple surrogate thermal model if FMU init or stepping fails.
In thermal_adapter.py:246, _advance_fallback() advances that surrogate with a first-order temperature response.
In thermal_adapter.py:382, _unavailable_snapshot() packages an error state for the API.