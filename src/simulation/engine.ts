import { delayedNeutronGroups, reactorModel } from './model'
import { computeReactivity } from './reactivity'
import type { ReactorRegime, ReactorSnapshot } from './types'

function equilibriumPrecursors(neutronPopulation: number) {
  return delayedNeutronGroups.map(
    (group) =>
      (group.beta /
        (reactorModel.neutronGenerationTimeSeconds * group.lambda)) *
      neutronPopulation,
  )
}

function classifyRegime(totalPcm: number, scramLatched: boolean): ReactorRegime {
  if (scramLatched) {
    return 'scrammed'
  }

  if (Math.abs(totalPcm) < 8) {
    return 'near-critical'
  }

  return totalPcm > 0 ? 'supercritical' : 'subcritical'
}

function calculatePeriodSeconds(
  previousNeutronPopulation: number,
  nextNeutronPopulation: number,
  stepSeconds: number,
) {
  if (previousNeutronPopulation <= 0 || nextNeutronPopulation <= 0) {
    return null
  }

  const logarithmicChange = Math.log(nextNeutronPopulation / previousNeutronPopulation)

  if (Math.abs(logarithmicChange) < 1e-4) {
    return null
  }

  return stepSeconds / logarithmicChange
}

export class ReactorEngine {
  private rodInsertionPercent: number = reactorModel.criticalRodInsertionPercent
  private scramLatched = false
  private timeSeconds = 0
  private neutronPopulation = 1
  private precursorConcentrations: number[] = equilibriumPrecursors(1)
  private lastPeriodSeconds: number | null = null

  reset() {
    this.rodInsertionPercent = reactorModel.criticalRodInsertionPercent
    this.scramLatched = false
    this.timeSeconds = 0
    this.neutronPopulation = 1
    this.precursorConcentrations = equilibriumPrecursors(1)
    this.lastPeriodSeconds = null
  }

  setRodInsertion(insertionPercent: number) {
    if (this.scramLatched) {
      return
    }

    this.rodInsertionPercent = Math.min(Math.max(insertionPercent, 0), 100)
  }

  scram() {
    this.scramLatched = true
    this.rodInsertionPercent = 100
  }

  step(stepSeconds: number) {
    if (stepSeconds <= 0) {
      return
    }

    const reactivity = computeReactivity(
      this.rodInsertionPercent,
      this.scramLatched,
    )
    const promptTerm =
      (reactivity.totalDeltaK - reactorModel.betaEffective) /
      reactorModel.neutronGenerationTimeSeconds

    let delayedSource = 0
    let delayedCoupling = 0

    for (const [index, group] of delayedNeutronGroups.entries()) {
      const denominator = 1 + stepSeconds * group.lambda
      delayedSource +=
        (group.lambda * this.precursorConcentrations[index]) / denominator
      delayedCoupling +=
        (group.lambda *
          stepSeconds *
          group.beta /
          reactorModel.neutronGenerationTimeSeconds) /
        denominator
    }

    const numerator = this.neutronPopulation + stepSeconds * delayedSource
    const denominator = 1 - stepSeconds * promptTerm - stepSeconds * delayedCoupling
    const nextNeutronPopulation = Math.max(numerator / denominator, 1e-9)

    this.precursorConcentrations = delayedNeutronGroups.map((group, index) => {
      const denominator = 1 + stepSeconds * group.lambda
      return (
        this.precursorConcentrations[index] +
        stepSeconds *
          (group.beta / reactorModel.neutronGenerationTimeSeconds) *
          nextNeutronPopulation
      ) / denominator
    })

    this.lastPeriodSeconds = calculatePeriodSeconds(
      this.neutronPopulation,
      nextNeutronPopulation,
      stepSeconds,
    )

    this.neutronPopulation = nextNeutronPopulation
    this.timeSeconds += stepSeconds

    if (
      !this.scramLatched &&
      this.neutronPopulation * reactorModel.nominalThermalPowerMw >=
        reactorModel.autoScramPowerMw
    ) {
      this.scram()
    }
  }

  getSnapshot(): ReactorSnapshot {
    const reactivity = computeReactivity(
      this.rodInsertionPercent,
      this.scramLatched,
    )
    const thermalPowerMw =
      reactorModel.nominalThermalPowerMw * this.neutronPopulation
    const totalFlux =
      reactorModel.nominalFluxNeutronsPerSquareCentimeterSecond *
      this.neutronPopulation

    return {
      lastPeriodSeconds: this.lastPeriodSeconds,
      neutronPopulation: this.neutronPopulation,
      reactivity,
      rodInsertionPercent: this.rodInsertionPercent,
      scramLatched: this.scramLatched,
      status: classifyRegime(reactivity.totalPcm, this.scramLatched),
      thermalPowerMw,
      timeSeconds: this.timeSeconds,
      totalFlux,
    }
  }
}
