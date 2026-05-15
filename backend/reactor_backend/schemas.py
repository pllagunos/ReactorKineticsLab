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


# ---------------------------------------------------------------------------
# Core page — diffusion flux response
# ---------------------------------------------------------------------------


class CoreFluxGeometry(BaseModel):
    rInnerCm: float
    rFuelCm: float
    rReflCm: float
    hActiveCm: float
    hReflCm: float
    rodRadiusCm: float


class CoreFluxProfile(BaseModel):
    axisCm: list[float]
    phiNorm: list[float]


class CoreFluxMetadata(BaseModel):
    rodInsertionPercent: float
    kEff: float
    iterations: int
    cached: bool
    meshDrCm: float
    meshDzCm: float
    displayNr: int
    displayNz: int


class CoreFluxResponse(BaseModel):
    """2D flux heatmap + midplane profiles from the diffusion solver.

    heatmapRCm / heatmapZCm are the display-grid coordinates (cm).
    heatmapPhi[i][j] is the normalised flux at (r_i, z_j), in [0, 1].
    """

    heatmapRCm: list[float]
    heatmapZCm: list[float]
    heatmapPhi: list[list[float]]
    radial: CoreFluxProfile
    axial: CoreFluxProfile
    geometry: CoreFluxGeometry
    metadata: CoreFluxMetadata


# ---------------------------------------------------------------------------
# Transient diffusion page
# ---------------------------------------------------------------------------


class TransientHistoryPoint(BaseModel):
    timeSeconds: float
    reactivityPcm: float
    powerNorm: float


class TransientDiffusionState(BaseModel):
    """Full state payload for the transient diffusion page.

    heatmapPhi[i][j] is the peak-normalised flux at (r_i, z_j) in [0, 1].
    powerNorm = P/P_0 (ratio of current to initial fission rate, volume-weighted).
    """

    timeSeconds: float
    rodInsertionPercent: float
    running: bool
    reactivityPcm: float
    powerNorm: float
    heatmapRCm: list[float]
    heatmapZCm: list[float]
    heatmapPhi: list[list[float]]
    radial: CoreFluxProfile
    axial: CoreFluxProfile
    history: list[TransientHistoryPoint]
    geometry: CoreFluxGeometry
    dt: float
    meshDrCm: float
    meshDzCm: float
    stepCount: int
