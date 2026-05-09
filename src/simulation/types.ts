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
  tuning: SimulationTuning
}
