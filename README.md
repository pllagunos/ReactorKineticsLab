# Reactor Web App

A Bun-managed React app that simulates a simplified heavy-water-moderated nuclear reactor core.

This first version focuses on a **single hollow cylindrical core**, **one control rod bank**, and a **point-kinetics transient** that shows how **reactivity**, **total neutron flux**, and **thermal power** change over time.

## What the app does

- Lets you move the **control rod bank** from fully withdrawn to fully inserted.
- Computes the resulting **reactivity** from a simple rod-worth curve.
- Evolves the reactor state forward with a browser-side **point-kinetics solver**.
- Displays:
  - current reactivity
  - total flux
  - thermal power
  - simulated time
  - neutron period estimate
- Includes:
  - a simple **core schematic**
  - rolling trend charts
  - **pause**, **reset**, and **SCRAM** controls

## Running locally

### Prerequisites

- [Bun](https://bun.sh/) 1.3+

### Install

```bash
bun install
```

### Start the dev server

```bash
bun run dev --host 127.0.0.1
```

Open `http://127.0.0.1:5173`.

### Other commands

```bash
bun run build
bun run lint
```

## How the simulation works

## 1. Core model

The reactor is treated as a simplified **annular core**:

- inner radius: **0.8 m**
- outer radius: **2.2 m**
- active height: **5.8 m**

Heavy water moderation is a **fixed assumption** in this version. It affects the story the model is telling, but it is **not** an operator-controlled variable yet.

## 2. Reactivity model

The only operator input is **rod insertion percent**.

Internally, the app converts rod position into reactivity using a smooth cumulative rod-worth shape:

`worth(z) = z - sin(2*pi*z) / (2*pi)`

where `z` is insertion fraction from `0` to `1`.

That gives the rod more effect near the middle of the core than at the extreme ends, which is a better first approximation than a purely linear rod-worth curve.

Current tuning values:

- total rod bank worth: **700 pcm**
- critical rod insertion target: **50%**
- extra SCRAM shutdown margin: **450 pcm**

The dashboard shows reactivity in **pcm**, while the solver uses **delta-k/k** internally:

- `1 pcm = 1e-5 delta-k/k`
- `beta_eff = 0.00651`
- `beta_eff = 651 pcm`

## 3. Point-kinetics solver

The core transient is modeled with point kinetics using:

- **6 delayed neutron groups**
- `beta_eff = 0.00651`
- neutron generation time: **5e-4 s**

The engine evolves:

- neutron population
- delayed neutron precursor concentrations

From that, the app derives:

- **thermal power**
- **total flux**

The numerical update is implemented with an **implicit step**, which is much more stable than a naive explicit Euler step for this kind of stiff reactor kinetics system.

### Nominal scaling

The app scales the normalized neutron population to approximate engineering values:

- nominal thermal power: **250 MWth**
- nominal flux: **2.4e13 n/cm^2/s**

So if neutron population doubles, both displayed power and displayed flux double.

## 4. Time stepping

The UI runs the engine on `requestAnimationFrame`, but the physics is stepped using a fixed simulated step:

- integrator step: **0.02 s**
- history sampling: **0.25 s**
- chart history length: **240 points**
- simulation speed: **8x wall clock**

This means the charts show a rolling transient window instead of storing an unbounded number of points.

## 5. Safety behavior

There are two shutdown paths:

1. **Manual SCRAM** from the button in the dashboard
2. **Automatic SCRAM** if power exceeds **325 MWth**

When SCRAM is latched:

- the rod bank is forced to **100% inserted**
- an extra shutdown reactivity penalty is applied
- the rod slider is disabled until **Reset to critical**

## App architecture

The code is split into a few clear layers:

- `src/simulation/model.ts`
  - constants for geometry, kinetics parameters, and simulation tuning
- `src/simulation/reactivity.ts`
  - rod-position-to-reactivity mapping
- `src/simulation/engine.ts`
  - the reactor engine and time-step update
- `src/hooks/useReactorSimulation.ts`
  - React integration, animation loop, and rolling history management
- `src/components/`
  - schematic, metric cards, and chart rendering
- `src/App.tsx`
  - dashboard composition and control wiring

One important design choice is that the **reactor engine lives outside the React render state**. React only reads snapshots from it. That avoids stale-state problems that are common when simulation loops are written directly with component state.

## What is simplified

This is an educational first slice, not a high-fidelity reactor code.

Some intentional simplifications:

- one effective rod bank instead of a full control system
- no thermal-hydraulic feedback
- no xenon, temperature coefficients, void coefficients, or fuel burnup
- fixed heavy-water moderation assumptions
- flux and power are scaled from nominal values rather than derived from a full core neutronics model
- point kinetics only, with no spatial flux solution

## Future extensions

Natural next steps would be:

- moderator and fuel temperature feedback
- coolant and heat-balance modeling
- multiple control rods or rod banks
- more explicit startup and shutdown procedures
- alarms, trip logic, and operator scenarios
- configurable reactor constants from the UI

## Notes

This project is meant to be **interpretable and interactive first**. The goal of the current model is to make the relationship between rod insertion, reactivity, flux, and power easy to inspect before adding more physics.
