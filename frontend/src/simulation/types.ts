export type ReactorRegime =
  | 'subcritical'
  | 'near-critical'
  | 'supercritical'
  | 'scrammed'

export type CoreGeometry = {
  activeHeightMeters: number
  innerRadiusMeters: number
  outerRadiusMeters: number
}

export type ReactorModel = {
  autoScramPowerMw: number
  betaEffective: number
  betaEffectivePcm: number
  coreGeometry: CoreGeometry
  criticalRodInsertionPercent: number
  nominalFluxNeutronsPerSquareCentimeterSecond: number
  nominalThermalPowerMw: number
  neutronGenerationTimeSeconds: number
  scramShutdownPcm: number
  totalControlRodWorthPcm: number
}

export type SimulationTuning = {
  historyPointLimit: number
  historySampleSeconds: number
  integratorStepSeconds: number
  thermalUpdateSeconds: number
  maxWallStepSeconds: number
  pollIntervalMs: number
  timeScale: number
}

export type ReactivitySnapshot = {
  baseExcessPcm: number
  dollars: number
  rodContributionPcm: number
  rodInsertionPercent: number
  scramPenaltyPcm: number
  totalDeltaK: number
  totalPcm: number
}

export type ReactorSnapshot = {
  lastPeriodSeconds: number | null
  neutronPopulation: number
  reactivity: ReactivitySnapshot
  rodInsertionPercent: number
  scramLatched: boolean
  status: ReactorRegime
  thermalPowerMw: number
  timeSeconds: number
  totalFlux: number
}

export type ThermalSnapshot = {
  available: boolean
  source: 'fmu' | 'fallback' | 'unavailable'
  timeSeconds: number
  powerMw: number
  inletTemperatureK: number | null
  outletTemperatureK: number | null
  massFlowKgPerSecond: number | null
  corePressureDropPa: number | null
  message: string | null
  axialPowerFractions: number[]
}

export type HistoryPoint = {
  reactivityPcm: number
  thermalPowerMw: number
  timeSeconds: number
  totalFlux: number
}

export type SimulationState = {
  history: HistoryPoint[]
  model: ReactorModel
  running: boolean
  snapshot: ReactorSnapshot
  thermal: ThermalSnapshot
  tuning: SimulationTuning
}

// ---------------------------------------------------------------------------
// Core page — resolved multigroup diffusion response
// ---------------------------------------------------------------------------

export type MultigroupDiffusionProfile = {
  axisCm: number[]
  values: number[]
}

export type MultigroupDiffusionGeometry = {
  coreRadiusCm: number
  moderatorRadiusCm: number
  reflectorRadiusCm: number
  coreHeightCm: number
  outerHeightCm: number
  resolvedRegionCount: number
}

export type MultigroupDiffusionMetadata = {
  cleanCore: boolean
  groupCount: number
  kEff: number
  openmcReferenceKEff: number
  openmcReferenceStdDevPcm: number
  differencePcm: number
  iterations: number
  cached: boolean
  cellCount: number
  meshSpacingCm: Record<string, number>
  timingsSeconds: Record<string, number>
  sphApplied: boolean
  sphIterations: number | null
  provisional: boolean
  qualified: boolean
  qualification: Record<string, unknown>
  rodInsertionPercent: number
  rodDeltaAbsorptionCmInv: number
  cleanCorrectionApplied: boolean
  roddedSolveCached: boolean
  powerShapeCorrectionApplied: boolean
  powerShapeCorrectionActiveBins: number
  powerShapeCorrectionReferenceTotal: number | null
  powerShapeCorrectionDiffusionTotal: number | null
}

export type MultigroupDiffusionResponse = {
  heatmapRCm: number[]
  heatmapZCm: number[]
  heatmapFlux: number[][]
  heatmapPower: number[][]
  radialFlux: MultigroupDiffusionProfile
  axialFlux: MultigroupDiffusionProfile
  radialPower: MultigroupDiffusionProfile
  axialPower: MultigroupDiffusionProfile
  energyGroupEdgesEv: number[]
  geometry: MultigroupDiffusionGeometry
  metadata: MultigroupDiffusionMetadata
}
