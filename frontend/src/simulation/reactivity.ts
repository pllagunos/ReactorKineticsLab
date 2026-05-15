import { reactorModel } from './model'
import type { ReactivitySnapshot } from './types'

function cumulativeWorthShape(insertionFraction: number) {
  return (
    insertionFraction -
    Math.sin(2 * Math.PI * insertionFraction) / (2 * Math.PI)
  )
}

function clampInsertionPercent(insertionPercent: number) {
  return Math.min(Math.max(insertionPercent, 0), 100)
}

export function computeReactivity(
  rodInsertionPercent: number,
  scramLatched: boolean,
): ReactivitySnapshot {
  const clampedInsertionPercent = clampInsertionPercent(rodInsertionPercent)
  const insertionFraction = clampedInsertionPercent / 100
  const baseExcessPcm =
    reactorModel.totalControlRodWorthPcm *
    cumulativeWorthShape(reactorModel.criticalRodInsertionPercent / 100)
  const rodContributionPcm =
    -reactorModel.totalControlRodWorthPcm * cumulativeWorthShape(insertionFraction)
  const scramPenaltyPcm = scramLatched ? -reactorModel.scramShutdownPcm : 0
  const totalPcm = baseExcessPcm + rodContributionPcm + scramPenaltyPcm

  return {
    baseExcessPcm,
    dollars: totalPcm / reactorModel.betaEffectivePcm,
    rodContributionPcm,
    rodInsertionPercent: clampedInsertionPercent,
    scramPenaltyPcm,
    totalDeltaK: totalPcm * 1e-5,
    totalPcm,
  }
}
