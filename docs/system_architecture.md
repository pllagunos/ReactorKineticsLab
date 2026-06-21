# Webapp System Architecture

This diagram shows the webapp runtime architecture and the offline reference
data paths that feed it.

```mermaid
flowchart LR
  subgraph OpenMC["OpenMC reference workflows"]
    CE["Continuous-energy OpenMC runs"]
    MGXSExport["openmc/mgxs_export.py"]
    RodScan["concentric_reactor_rodworth.py"]
    ReactivityScan["concentric_reactor_reactivity.py"]
    RefData["openmc/reference_data/concentric"]
  end

  subgraph ReferenceData["Published reference artifacts"]
    MGXSJson["group_sweep/group_4/.../mgxs_constants.json"]
    RodWorthData["rod worth table"]
    ReactivityCoeffData["reactivity coefficient CSV"]
    PowerShapeData["CE power-shape correction"]
  end

  subgraph Frontend["Browser frontend - React + Vite"]
    Overview["Overview page<br/>rod slider, HMI, trends"]
    CorePage["Core page<br/>validated multigroup view"]
    ApiClient["frontend/src/simulation/api.ts"]
  end

  subgraph Backend["FastAPI backend - backend/reactor_backend"]
    App["app.py<br/>API routes"]
    Service["SimulationService<br/>state, timing, history"]
    Engine["ReactorEngine<br/>point kinetics"]
    Reactivity["reactivity.py<br/>rod + thermal feedback"]
    ThermalAdapter["ThermalAdapter<br/>FMPy FMU lifecycle"]
    MultigroupService["MultigroupDiffusionService<br/>Core page solve/cache"]
    MGXSAdapter["openmc_mgxs_adapter.py<br/>MGXS -> diffusion input"]
  end

  subgraph Modelica["Modelica thermal hydraulics"]
    ModelicaSource["ResearchReactorThermalHydraulics.mo"]
    FMU["build/ResearchReactorThermalHydraulics.fmu"]
    Fallback["fallback thermal model"]
  end

  subgraph API["HTTP API"]
    SimulationAPI["/api/simulation/*"]
    CoreAPI["/api/multigroup-diffusion/*"]
  end

  CE --> MGXSExport
  CE --> RodScan
  CE --> ReactivityScan
  MGXSExport --> RefData
  RodScan --> RefData
  ReactivityScan --> RefData
  RefData --> MGXSJson
  RefData --> RodWorthData
  RefData --> ReactivityCoeffData
  RefData --> PowerShapeData

  MGXSJson --> Engine
  MGXSJson --> MGXSAdapter
  RodWorthData --> Reactivity
  ReactivityCoeffData --> Reactivity
  PowerShapeData --> MultigroupService

  Overview --> ApiClient
  CorePage --> ApiClient
  ApiClient --> SimulationAPI
  ApiClient --> CoreAPI
  SimulationAPI --> App
  CoreAPI --> App

  App --> Service
  App --> MultigroupService
  Service --> Engine
  Service --> ThermalAdapter
  Engine --> Reactivity
  ThermalAdapter --> Engine

  ModelicaSource --> FMU
  ThermalAdapter --> FMU
  FMU --> ThermalAdapter
  ThermalAdapter --> Fallback

  MultigroupService --> MGXSAdapter
  MGXSAdapter --> MultigroupService
```

## Runtime Data Flow

The Overview page uses `/api/simulation/*`.

1. The frontend sends rod, pause, reset, and SCRAM commands to FastAPI.
2. `SimulationService` owns simulation time, server-side history, and state.
3. `ReactorEngine` advances the point-kinetics equations.
4. `reactivity.py` combines OpenMC rod worth, SCRAM penalty, and thermal
   feedback.
5. `ThermalAdapter` advances the Modelica FMU and returns a `ThermalSnapshot`.
6. The frontend polls `/api/simulation/state` and renders HMI values and trend
   charts from the returned snapshot and history.

The Core page uses `/api/multigroup-diffusion/*`.

1. `MultigroupDiffusionService` loads the published OpenMC MGXS export.
2. `openmc_mgxs_adapter.py` validates domain metadata and converts it into
   resolved diffusion input.
3. The backend solves the validated four-group diffusion problem.
4. The returned state includes mirrored x-z flux and power maps plus axial and
   radial profiles for the frontend Core page.

## OpenMC Reference Data Flow

OpenMC is not run by the webapp at request time. It is an offline reference-data
producer.

The currently used reference artifacts include:

- MGXS constants and prompt generation time from
  `openmc/reference_data/concentric/group_sweep/group_4/outputs/mgxs_constants.json`
- OpenMC-derived delayed neutron data loaded by `kinetics_reference.py`
- OpenMC CE rod worth loaded by `rod_worth.py`
- OpenMC reactivity coefficients loaded by `reactivity_coefficients.py`
- CE power-shape correction data used by the multigroup Core visualization

These artifacts are treated as immutable inputs during normal webapp operation.

## FMU Coupling Summary

The Modelica coupling is owned by `ThermalAdapter`.

The adapter:

- checks whether the existing FMU is fresh and exposes the required variables
- exports a new FMU with OpenModelica when needed
- instantiates the FMU with FMPy
- writes `totalPower` and `axialPowerFractions`
- steps the co-simulation slave
- reads inlet/outlet temperatures, effective fuel/moderator temperatures,
  moderator density, mass flow, and core pressure drop

If FMU export, initialization, or stepping fails, the adapter switches to a
fallback thermal model so the webapp remains controllable. Thermal reactivity
feedback is still FMI-only: fallback thermal values can populate the HMI, but
they do not change point-kinetics reactivity.

See `docs/fmu_coupling.md` for the detailed FMU lifecycle.

## API Boundary

The frontend treats the backend as the source of truth. The browser does not
advance physics locally.

Primary API groups:

- `/api/simulation/*` for the Overview page and point-kinetics transient
- `/api/multigroup-diffusion/*` for the Core page diffusion solve

Legacy backend routes for old one-group core/transient visualizations may still
exist internally, but they are not the frontend's canonical spatial neutronics
view.
