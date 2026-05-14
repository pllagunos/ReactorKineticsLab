# Reactor Web App

A localhost reactor simulator with a **React + Bun frontend** and a **Python backend** for the reactor physics.

The current version models a simplified **heavy-water-moderated annular core** with one operator control: **control rod insertion**. The browser dashboard shows how **reactivity**, **total neutron flux**, and **thermal power** evolve over time, while the transient is solved in Python.

## Why the project is hybrid

The project started as a Bun-managed React app with an in-browser solver. It now uses a hybrid architecture so the **simulation core can move into the Python scientific ecosystem** without losing the existing frontend.

That gives the project:

- a rich local browser UI
- a Python code path for future numerical work
- a cleaner separation between **physics** and **presentation**

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
bun install
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

### Other useful commands

```bash
bun run build
bun run lint
```

### Quick backend check

```bash
curl http://127.0.0.1:8000/api/health
```

## Hybrid architecture

## Frontend

The frontend remains a **Bun-managed React + TypeScript** app.

Its job is to:

- render the dashboard
- send user commands to the backend
- poll the latest state from the backend
- display the rolling history charts and core schematic

Key frontend files:

- `src/App.tsx`
  - dashboard composition and control wiring
- `src/hooks/useReactorSimulation.ts`
  - API polling, command calls, and React state integration
- `src/simulation/api.ts`
  - frontend API client for the Python service
- `src/components/`
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

- The clean, unrodded core is the **critical reference** (0 % insertion → ρ = 0 pcm).
- The table stores Δρ(x) at 11 points (x = 0.0 to 1.0 in steps of 0.1) from a 2D r-z finite-difference one-group diffusion eigenvalue scan.
- Backend interpolation is linear between table points.
- Rod insertion adds only **negative reactivity**; the full-bank worth is ≈ **−29 pcm**.

Current calibration values (frozen from the notebook):

| x (insertion fraction) | Δρ (pcm) |
|---|---|
| 0.0 | 0.0 (reference) |
| 0.1 | +0.6 (mesh artefact) |
| 0.2 | −0.9 |
| 0.5 | −13.4 |
| 1.0 | −29.0 |

Extra SCRAM shutdown margin: **450 pcm**

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

- include one group diffusion results from analytical python
- maybe solve time dependent diffusion equation instead of point kinetics?
- visualization of axial and radial flux distribution (inspiration form inspo folder)
- integrate Modelica FMU for simple systems thermal hydraulics


FIXES NOTEBOOK:
- how were the dimensions for the estimate 2 baseline model decided
- remove the radial only approach and instead find geometry using 2D r-z diffusion solver
- For the 2D approach we should get dimensions that make the reactor slightly overcritical. Perhaps iteration like in a gridsearch (idea from ML) or what method would you recommend to be able to get a geometry that gives k = 1.0000
- same question but for control rod, to get proper negative pcm?
- restructure folders. src (which is just frontend) to be renamed to frontend and bundled together with the bun, eslint, package, vite and tsconfig files. backend, which has the core python backend and physics solvers to be renamed approprietly (reactor-app?)
- python general solver for diffusion that can be used by both notebook and backend
- in reactorModel.ipynb use these diffusion solvers for the numerical calculations, instead of defining the solvers inside the notebook

