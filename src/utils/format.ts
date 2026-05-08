const percentFormatter = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 1,
})

const powerFormatter = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 1,
})

const pcmFormatter = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 0,
  signDisplay: 'always',
})

export function formatRodInsertion(insertionPercent: number) {
  return `${percentFormatter.format(insertionPercent)}%`
}

export function formatPercent(percentValue: number) {
  return `${percentFormatter.format(percentValue)}%`
}

export function formatPowerMw(powerMw: number) {
  return `${powerFormatter.format(powerMw)} MWth`
}

export function formatFlux(totalFlux: number) {
  return `${totalFlux.toExponential(2)} n/cm^2/s`
}

export function formatReactivityPcm(reactivityPcm: number) {
  return `${pcmFormatter.format(reactivityPcm)} pcm`
}

export function formatDollars(dollars: number) {
  return `${dollars >= 0 ? '+' : ''}${dollars.toFixed(2)} $`
}

export function formatSimTime(timeSeconds: number) {
  if (timeSeconds < 60) {
    return `${timeSeconds.toFixed(1)} s`
  }

  const minutes = Math.floor(timeSeconds / 60)
  const seconds = timeSeconds % 60
  return `${minutes}m ${seconds.toFixed(0)}s`
}

export function formatPeriodSeconds(periodSeconds: number | null) {
  if (periodSeconds === null) {
    return 'steady'
  }

  const absolutePeriod = Math.abs(periodSeconds)
  const formatted =
    absolutePeriod >= 60
      ? `${(absolutePeriod / 60).toFixed(1)} min`
      : `${absolutePeriod.toFixed(1)} s`

  return `${periodSeconds > 0 ? '+' : '-'}${formatted}`
}
