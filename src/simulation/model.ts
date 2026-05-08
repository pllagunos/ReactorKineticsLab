export const delayedNeutronGroups = [
  { beta: 0.00025, lambda: 0.0124 },
  { beta: 0.00138, lambda: 0.0305 },
  { beta: 0.00122, lambda: 0.111 },
  { beta: 0.00264, lambda: 0.301 },
  { beta: 0.00075, lambda: 1.14 },
  { beta: 0.00027, lambda: 3.01 },
] as const

export const reactorModel = {
  autoScramPowerMw: 1000,
  betaEffective: 0.00651,
  betaEffectivePcm: 651,
  coreGeometry: {
    activeHeightMeters: 5.8,
    innerRadiusMeters: 0.8,
    outerRadiusMeters: 2.2,
  },
  criticalRodInsertionPercent: 50,
  nominalFluxNeutronsPerSquareCentimeterSecond: 2.4e13,
  nominalThermalPowerMw: 250,
  neutronGenerationTimeSeconds: 5e-4,
  scramShutdownPcm: 450,
  totalControlRodWorthPcm: 700,
} as const

export const simulationTuning = {
  historyPointLimit: 240,
  historySampleSeconds: 0.25,
  integratorStepSeconds: 0.02,
  maxWallStepSeconds: 0.2,
  timeScale: 8,
} as const
