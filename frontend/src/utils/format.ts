const percentFormatter = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 1,
})

const powerFormatter = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 1,
})

const temperatureFormatter = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 1,
})

const massFlowFormatter = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 1,
})

const pressureFormatter = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 0,
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

export function formatTemperatureK(temperatureK: number | null) {
  if (temperatureK === null) {
    return 'n/a'
  }

  return `${temperatureFormatter.format(temperatureK - 273.15)} °C`
}

export function formatMassFlow(massFlowKgPerSecond: number | null) {
  if (massFlowKgPerSecond === null) {
    return 'n/a'
  }

  return `${massFlowFormatter.format(massFlowKgPerSecond)} kg/s`
}

export function formatDensityGPerCm3(densityGPerCm3: number | null) {
  if (densityGPerCm3 === null) {
    return 'n/a'
  }

  return `${densityGPerCm3.toFixed(4)} g/cm^3`
}

export function formatPressureDropPa(pressureDropPa: number | null) {
  if (pressureDropPa === null) {
    return 'n/a'
  }

  if (Math.abs(pressureDropPa) >= 1_000) {
    return `${(pressureDropPa / 1_000).toFixed(2)} kPa`
  }

  return `${pressureFormatter.format(pressureDropPa)} Pa`
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
