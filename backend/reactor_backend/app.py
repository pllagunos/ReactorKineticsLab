from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import RodInsertionRequest, RunningRequest, SimulationState
from .service import simulation_service


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
