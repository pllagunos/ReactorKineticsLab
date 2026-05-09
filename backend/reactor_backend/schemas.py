from typing import Literal, Optional

from pydantic import BaseModel, Field


ReactorRegime = Literal["subcritical", "near-critical", "supercritical", "scrammed"]


class CoreGeometry(BaseModel):
    activeHeightMeters: float
    innerRadiusMeters: float
    outerRadiusMeters: float


class ReactorModel(BaseModel):
    autoScramPowerMw: float
    betaEffective: float
    betaEffectivePcm: float
    coreGeometry: CoreGeometry
    criticalRodInsertionPercent: float
    nominalFluxNeutronsPerSquareCentimeterSecond: float
    nominalThermalPowerMw: float
    neutronGenerationTimeSeconds: float
    scramShutdownPcm: float
    totalControlRodWorthPcm: float


class SimulationTuning(BaseModel):
    historyPointLimit: int
    historySampleSeconds: float
    integratorStepSeconds: float
    maxWallStepSeconds: float
    pollIntervalMs: int
    timeScale: float


class ReactivitySnapshot(BaseModel):
    baseExcessPcm: float
    dollars: float
    rodContributionPcm: float
    rodInsertionPercent: float
    scramPenaltyPcm: float
    totalDeltaK: float
    totalPcm: float


class ReactorSnapshot(BaseModel):
    lastPeriodSeconds: Optional[float]
    neutronPopulation: float
    reactivity: ReactivitySnapshot
    rodInsertionPercent: float
    scramLatched: bool
    status: ReactorRegime
    thermalPowerMw: float
    timeSeconds: float
    totalFlux: float


class HistoryPoint(BaseModel):
    reactivityPcm: float
    thermalPowerMw: float
    timeSeconds: float
    totalFlux: float


class SimulationState(BaseModel):
    history: list[HistoryPoint]
    model: ReactorModel
    running: bool
    snapshot: ReactorSnapshot
    tuning: SimulationTuning


class RodInsertionRequest(BaseModel):
    insertionPercent: float = Field(ge=0, le=100)


class RunningRequest(BaseModel):
    running: bool
