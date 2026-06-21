# Reactor Kinetics Lab

A web-based reactor simulator which runs a point kinetics and diffusion solver python backend coupled to a modelica based thermal-hydraulics module.

The reactor is modeled with montecarlo OpenMC software.

The current version models a simplified **heavy-water-moderated natural uranium fueled core** with one operator control: **control rod insertion**. The browser dashboard shows how **reactivity**, **total neutron flux**, **thermal power** and **coolant temperatures** evolve over time.

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
- `backend/reactor_backend/multigroup_service.py`
  - cached four-group clean-core diffusion service
- `backend/reactor_backend/multigroup_sph.py`
  - CE-referenced SPH fitting, application, and qualification
- `backend/reactor_backend/app.py`
  - FastAPI routes

## API shape (is this needed -> ne, maybe a webpage architecture MD file)

The current backend exposes:

- `GET /api/health`
- `GET /api/simulation/state`
- `POST /api/simulation/reset`
- `POST /api/simulation/rod-insertion`
- `POST /api/simulation/running`
- `POST /api/simulation/scram`
- `GET /api/multigroup-diffusion/state`
- `POST /api/multigroup-diffusion/recompute`

The frontend Core page is the validated OpenMC-informed multigroup view. It
uses the existing multigroup backend API, follows the
`openmc/diffusion_concentric_reactor.ipynb` four-group diffusion setup
(`groupwise_fvm`, 0.1/1/5/20 cm radial targets, 10 cm axial target), mirrors
the cylindrical field into an x-z image, and applies the clean OpenMC CE
power-shape correction only to the displayed fission power map. Its export and
cache locations can be overridden with `MULTIGROUP_MGXS_EXPORT_DIR` and
`MULTIGROUP_DIFFUSION_CACHE_DIR`.

The frontend treats the backend as the **source of truth** for:

- current reactor snapshot
- rolling history
- running / paused state
- reactor constants used for display

## How the simulation works

## 1. Core model

See `theory` for more details.

- Narrative: initial homogeneous core - the simplified **annular core** (Estimate 2 geometry from `theory/reactorModel.ipynb`); then an openmc `openmc` workflow where we first realize issues with the diffusion model and then generate better fidelity cores starting with a frm2 style involute core and ending with a concentric core.
- initial problems where the uneconomical size of the core, unrealistic homogeneous core.

## 2. Reactivity model

The only operator input is **rod insertion percent**. --> add flow rate in primary and secondary later!.

Rod position is converted into reactivity using a **2D-calibrated rod-worth table** derived from `theory/reactorModel.ipynb`: --> need new rod-wroth table for new concentric core.

- The clean, unrodded core is slightly supercritical in the 2D solve: `k_eff ≈ 1.000395` (`ρ ≈ +39.4 pcm`).
- The table stores **rod-only** `Δρ(x)` at 11 points (x = 0.0 to 1.0 in steps of 0.1) from a 2D r-z finite-difference one-group diffusion eigenvalue scan.
- Backend interpolation is linear between table points.
- Total reactivity is computed as `ρ_total(x) = ρ_base(clean) + Δρ_rod(x) + ρ_scram`.
- The rod is modeled as a **combined control/shutdown bank** with effective parameters:
  - equivalent radius `r_rod = 50 cm`
  - effective absorber increment `ΔΣ_a,max = 0.25 cm⁻¹` (one-group homogenized value)

Derived operating points:

- critical insertion: **~32%**
- full insertion total reactivity: **~−120 pcm** (without extra SCRAM penalty)
- extra SCRAM shutdown margin (latched trip): **450 pcm**

The dashboard shows reactivity in **pcm**, while the kinetics solver uses **delta-k/k** internally:

- `beta_eff = 0.00678910882978`
- `beta_eff = 678.91 pcm`

## 3. Point-kinetics solver

The transient uses:

- **6 delayed neutron groups** aggregated from
  `openmc/reference_data/concentric/group_sweep/group_4/outputs/mgxs_constants.json`
- neutron generation time: **0.00417212833712 s**, from the export's prompt
  generation-time tally
- fuel, moderator-temperature, and moderator-density feedback from the OpenMC
  reactivity-coefficient CSV when the Modelica FMU supplies effective feedback
  values

The engine evolves:

- neutron population
- delayed neutron precursor concentrations

From that, it derives:

- thermal power
- total flux

Thermal feedback is FMI-only. If the FMU is unavailable and the fallback
thermal model is active, the backend reports zero thermal-feedback reactivity.
On reset, the initial FMU effective state is captured as the zero-feedback
reference so the point model still starts at the critical rod insertion.

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

This is still an educational first slice. Intentional simplifications:

- one effective rod bank instead of a full control system
- no thermal-hydraulic feedback
- no xenon, temperature coefficients, void coefficients, or burnup
- flux and power are scaled from nominal values

## Next steps

Now that the physics layer is in Python, natural next steps are:

- once PK model of core is ready, replace backend core with openmc concentric core
- regenerate the four-group CE reference at <=15 pcm uncertainty and qualify
  the implemented SPH workflow
- see `theory/ThermalHydraulics.tex` adjust to new core size
- adjust modelica model
- FMU build handlded outside of python and frozen (not gitignored)? or handled by a script, like with bun run backend:install? this way simulator can run on macos (if doable)
- future: krylov instead of gauss-seidel

### frontend
- view should be based on laptop screen 16:10
- no scrolling should be needed at 100% zoom to visualize controls and graphs
- less cards explaining stuff that's in the README.
- does transient diffusion solver make sense?

# architecture documentation:
- Decide documentation structure (latex file for reactor modeling, TH and MD files for software arch?)

###
for openmc scripts, explaining architecture and functionality of fuel_element.py, involutes.py, ploting.py and reactor_geometry.py. Specially how its object oriented nature and dataclasses interact between each other. But also how they implement what they do (with special emphasis in how the involute plates are generated as polygons from sin,cos curves and transformations). Include also how the frm2_nat.ipynb notebook uses them.

###
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
