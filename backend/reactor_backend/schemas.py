from typing import Any, Literal, Optional

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
    thermalUpdateSeconds: float
    maxWallStepSeconds: float
    pollIntervalMs: int
    timeScale: float


class ReactivitySnapshot(BaseModel):
    baseExcessPcm: float
    dollars: float
    fuelTemperatureFeedbackPcm: float = 0.0
    moderatorDensityFeedbackPcm: float = 0.0
    moderatorTemperatureFeedbackPcm: float = 0.0
    rodContributionPcm: float
    rodInsertionPercent: float
    scramPenaltyPcm: float
    thermalFeedbackApplied: bool = False
    thermalFeedbackPcm: float = 0.0
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


class ThermalSnapshot(BaseModel):
    available: bool
    source: Literal["fmu", "fallback", "unavailable"]
    timeSeconds: float
    powerMw: float
    inletTemperatureK: Optional[float] = None
    outletTemperatureK: Optional[float] = None
    fuelMaximumTemperatureK: Optional[float] = None
    fuelTemperatureK: Optional[float] = None
    moderatorTemperatureK: Optional[float] = None
    moderatorDensityKgPerM3: Optional[float] = None
    moderatorDensityGPerCm3: Optional[float] = None
    massFlowKgPerSecond: Optional[float] = None
    corePressureDropPa: Optional[float] = None
    message: Optional[str] = None
    axialPowerFractions: list[float] = Field(default_factory=lambda: [0.125] * 8)


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
    thermal: ThermalSnapshot
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
# Clean-core resolved multigroup diffusion page
# ---------------------------------------------------------------------------


class MultigroupDiffusionProfile(BaseModel):
    axisCm: list[float]
    values: list[float]


class MultigroupDiffusionGeometry(BaseModel):
    coreRadiusCm: float
    moderatorRadiusCm: float
    reflectorRadiusCm: float
    coreHeightCm: float
    outerHeightCm: float
    resolvedRegionCount: int


class MultigroupDiffusionMetadata(BaseModel):
    cleanCore: bool
    groupCount: int
    kEff: float
    reactivityPcm: float
    openmcCeReactivityPcm: float
    openmcReferenceKEff: float
    openmcReferenceStdDevPcm: float
    differencePcm: float
    iterations: int
    cached: bool
    cellCount: int
    meshSpacingCm: dict[str, float]
    timingsSeconds: dict[str, float]
    sphApplied: bool
    sphIterations: int | None
    provisional: bool
    qualified: bool
    qualification: dict[str, Any]
    rodInsertionPercent: float
    rodDeltaAbsorptionCmInv: float
    cleanCorrectionApplied: bool
    roddedSolveCached: bool
    powerShapeCorrectionApplied: bool
    powerShapeCorrectionActiveBins: int
    powerShapeCorrectionReferenceTotal: float | None = None
    powerShapeCorrectionDiffusionTotal: float | None = None


class MultigroupDiffusionResponse(BaseModel):
    """Mirrored x-z heatmaps and integrated profiles for the Core page.

    heatmapFlux and heatmapPower are indexed as [z][x].
    """

    heatmapXCm: list[float]
    heatmapZCm: list[float]
    heatmapFlux: list[list[float]]
    heatmapPower: list[list[float]]
    radialFlux: MultigroupDiffusionProfile
    axialFlux: MultigroupDiffusionProfile
    radialPower: MultigroupDiffusionProfile
    axialPower: MultigroupDiffusionProfile
    energyGroupEdgesEv: list[float]
    geometry: MultigroupDiffusionGeometry
    metadata: MultigroupDiffusionMetadata


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
