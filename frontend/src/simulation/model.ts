export const delayedNeutronGroups = [
  { beta: 0.000227917094163, lambda: 0.0133443203744 },
  { beta: 0.00119534670498, lambda: 0.0326776283533 },
  { beta: 0.00115198689522, lambda: 0.120914791504 },
  { beta: 0.00262514724228, lambda: 0.304202879275 },
  { beta: 0.00112075519129, lambda: 0.8554154214 },
  { beta: 0.000467955701846, lambda: 2.8729591688 },
] as const

export const reactorModel = {
  autoScramPowerMw: 1000,
  betaEffective: 0.00678910882978,
  betaEffectivePcm: 678.910883,
  coreGeometry: {
    activeHeightMeters: 5.8,
    innerRadiusMeters: 0.8,
    outerRadiusMeters: 2.2,
  },
  criticalRodInsertionPercent: 50,
  nominalFluxNeutronsPerSquareCentimeterSecond: 2.4e13,
  nominalThermalPowerMw: 250,
  neutronGenerationTimeSeconds: 0.00417212833712,
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
