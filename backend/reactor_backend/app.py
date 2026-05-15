from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    RodInsertionRequest,
    RunningRequest,
    SimulationState,
    CoreFluxResponse,
    TransientDiffusionState,
)
from .service import simulation_service
from .core_service import get_core_flux
from .transient_service import transient_diffusion_service


app = FastAPI(title="Reactor simulator backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def get_health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/simulation/state", response_model=SimulationState)
def get_simulation_state() -> SimulationState:
    return simulation_service.get_state()


@app.post("/api/simulation/reset", response_model=SimulationState)
def reset_simulation() -> SimulationState:
    return simulation_service.reset()


@app.post("/api/simulation/scram", response_model=SimulationState)
def scram_simulation() -> SimulationState:
    return simulation_service.scram()


@app.post("/api/simulation/rod-insertion", response_model=SimulationState)
def set_rod_insertion(payload: RodInsertionRequest) -> SimulationState:
    return simulation_service.set_rod_insertion(payload.insertionPercent)


@app.post("/api/simulation/running", response_model=SimulationState)
def set_running(payload: RunningRequest) -> SimulationState:
    return simulation_service.set_running(payload.running)


@app.get("/api/core/flux", response_model=CoreFluxResponse)
def get_core_flux_endpoint() -> CoreFluxResponse:
    """Return 2D flux distribution for the current rod insertion state.

    Results are cached per insertion fraction so repeated calls do not
    re-run the eigenvalue solve.
    """
    rod_pct = simulation_service.get_state().snapshot.rodInsertionPercent
    return get_core_flux(rod_pct)


# ---------------------------------------------------------------------------
# Transient diffusion page
# ---------------------------------------------------------------------------


@app.get("/api/transient-diffusion/state", response_model=TransientDiffusionState)
def get_transient_state() -> TransientDiffusionState:
    """Return current transient diffusion state, advancing if running."""
    return transient_diffusion_service.get_state()


@app.post("/api/transient-diffusion/reset", response_model=TransientDiffusionState)
def reset_transient() -> TransientDiffusionState:
    """Reset transient to the critical steady state."""
    return transient_diffusion_service.reset()


@app.post("/api/transient-diffusion/running", response_model=TransientDiffusionState)
def set_transient_running(payload: RunningRequest) -> TransientDiffusionState:
    return transient_diffusion_service.set_running(payload.running)


@app.post("/api/transient-diffusion/rod-insertion", response_model=TransientDiffusionState)
def set_transient_rod(payload: RodInsertionRequest) -> TransientDiffusionState:
    return transient_diffusion_service.set_rod_insertion(payload.insertionPercent)


@app.post("/api/transient-diffusion/step", response_model=TransientDiffusionState)
def manual_transient_step() -> TransientDiffusionState:
    """Advance the transient by exactly one time step."""
    return transient_diffusion_service.manual_step()
