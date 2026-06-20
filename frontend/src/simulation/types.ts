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
// Core page — diffusion flux response
// ---------------------------------------------------------------------------

export type CoreFluxGeometry = {
  rInnerCm: number
  rFuelCm: number
  rReflCm: number
  hActiveCm: number
  hReflCm: number
  rodRadiusCm: number
}

export type CoreFluxProfile = {
  axisCm: number[]
  phiNorm: number[]
}

export type CoreFluxMetadata = {
  rodInsertionPercent: number
  kEff: number
  iterations: number
  cached: boolean
  meshDrCm: number
  meshDzCm: number
  displayNr: number
  displayNz: number
}

export type CoreFluxResponse = {
  heatmapRCm: number[]
  heatmapZCm: number[]
  /** heatmapPhi[i][j] = normalised flux at (r_i, z_j) in [0, 1] */
  heatmapPhi: number[][]
  radial: CoreFluxProfile
  axial: CoreFluxProfile
  geometry: CoreFluxGeometry
  metadata: CoreFluxMetadata
}

// ---------------------------------------------------------------------------
// Clean-core resolved multigroup diffusion page
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

// ---------------------------------------------------------------------------
// Transient diffusion page
// ---------------------------------------------------------------------------

export type TransientHistoryPoint = {
  timeSeconds: number
  reactivityPcm: number
  powerNorm: number
}

export type TransientDiffusionState = {
  timeSeconds: number
  rodInsertionPercent: number
  running: boolean
  reactivityPcm: number
  powerNorm: number
  heatmapRCm: number[]
  heatmapZCm: number[]
  /** heatmapPhi[i][j] = peak-normalised flux at (r_i, z_j) in [0, 1] */
  heatmapPhi: number[][]
  radial: CoreFluxProfile
  axial: CoreFluxProfile
  history: TransientHistoryPoint[]
  geometry: CoreFluxGeometry
  dt: number
  meshDrCm: number
  meshDzCm: number
  stepCount: number
}
