import { CoolingLoopDiagram } from '../components/CoolingLoopDiagram'
import { TrendChart } from '../components/TrendChart'
import type { TrendSeries } from '../components/TrendChart'
import { useReactorSimulation } from '../hooks/useReactorSimulation'
import type { HistoryPoint, ReactorRegime, ThermalSnapshot } from '../simulation/types'
import {
  formatFlux,
  formatMassFlow,
  formatPowerMw,
  formatReactivityPcm,
  formatSimTime,
  formatTemperatureK,
} from '../utils/format'

const statusTone: Record<ReactorRegime, string> = {
  supercritical: 'warning',
  subcritical: 'cool',
  'near-critical': 'neutral',
  scrammed: 'danger',
}

export function OverviewPage() {
  const {
    error,
    history,
    loading,
    model,
    reset,
    running,
    scram,
    setRodInsertionPercent,
    setRunning,
    snapshot,
    thermal,
    thermalHistory,
  } = useReactorSimulation()

  if (!snapshot || !model) {
    return (
      <div className="app-shell">
        <section className="panel loading-panel">
          <h2>{loading ? 'Connecting to Python backend' : 'Backend unavailable'}</h2>
          <p className="loading-copy">
            {error ??
              'Run `bun run backend:install` once, then start the hybrid stack with `bun run dev`.'}
          </p>
        </section>
      </div>
    )
  }

  const powerShare = (snapshot.thermalPowerMw / model.nominalThermalPowerMw) * 100
  const powerTone =
    snapshot.thermalPowerMw >= model.autoScramPowerMw
      ? 'danger'
      : powerShare >= 105
        ? 'warning'
        : 'neutral'

  const neutronicsSeries: TrendSeries<HistoryPoint>[] = [
    {
      id: 'reactivity',
      label: 'Reactivity',
      unit: 'pcm',
      color: '#f59e0b',
      data: history,
      valueAccessor: (p) => p.reactivityPcm,
      valueFormatter: formatReactivityPcm,
      tickValueFormatter: (v) => v.toFixed(0),
    },
    {
      id: 'flux',
      label: 'Flux',
      unit: 'n/cm²/s',
      color: '#38bdf8',
      data: history,
      valueAccessor: (p) => p.totalFlux,
      valueFormatter: formatFlux,
      tickValueFormatter: (v) => v.toExponential(1),
    },
    {
      id: 'power',
      label: 'Power',
      unit: 'MWth',
      color: '#f97316',
      data: history,
      valueAccessor: (p) => p.thermalPowerMw,
      valueFormatter: formatPowerMw,
      tickValueFormatter: (v) => v.toFixed(1),
    },
  ]

  const thSeries: TrendSeries<ThermalSnapshot>[] = [
    {
      id: 'tin',
      label: 'T inlet',
      unit: '°C',
      color: '#38bdf8',
      data: thermalHistory,
      valueAccessor: (t) => t.inletTemperatureK ?? 0,
      valueFormatter: (v) => formatTemperatureK(v),
      tickValueFormatter: (v) => (v - 273.15).toFixed(1),
    },
    {
      id: 'tout',
      label: 'T outlet',
      unit: '°C',
      color: '#f59e0b',
      data: thermalHistory,
      valueAccessor: (t) => t.outletTemperatureK ?? 0,
      valueFormatter: (v) => formatTemperatureK(v),
      tickValueFormatter: (v) => (v - 273.15).toFixed(1),
    },
    {
      id: 'flow',
      label: 'Flow',
      unit: 'kg/s',
      color: '#4ade80',
      data: thermalHistory,
      valueAccessor: (t) => t.massFlowKgPerSecond ?? 0,
      valueFormatter: (v) => formatMassFlow(v),
      tickValueFormatter: (v) => v.toFixed(1),
    },
  ]

  return (
    <div className="app-shell overview-shell">
      {/* ── Top stat chips ── */}
      <div className="overview-stat-row">
        <div className={`stat-chip stat-chip--${statusTone[snapshot.status]}`}>
          <span className="stat-chip__label">Reactivity</span>
          <span className="stat-chip__value">
            {formatReactivityPcm(snapshot.reactivity.totalPcm)}
          </span>
        </div>
        <div className="stat-chip stat-chip--cool">
          <span className="stat-chip__label">Total Flux</span>
          <span className="stat-chip__value">{formatFlux(snapshot.totalFlux)}</span>
        </div>
        <div className={`stat-chip stat-chip--${powerTone}`}>
          <span className="stat-chip__label">Thermal Power</span>
          <span className="stat-chip__value">
            {formatPowerMw(snapshot.thermalPowerMw)}
          </span>
        </div>
        <div className="stat-chip stat-chip--neutral">
          <span className="stat-chip__label">Sim Clock</span>
          <span className="stat-chip__value">
            {formatSimTime(snapshot.timeSeconds)}
          </span>
        </div>
      </div>

      {/* ── HMI + controls ── */}
      <div className="overview-hmi-row">
        <div className="overview-controls">
          <div className="vertical-slider-group">
            <label htmlFor="rod-insertion" className="vertical-slider-label">
              Rod
            </label>
            <input
              id="rod-insertion"
              type="range"
              className="slider-vertical"
              min="0"
              max="100"
              step="0.1"
              value={snapshot.rodInsertionPercent}
              onChange={(e) =>
                setRodInsertionPercent(Number(e.currentTarget.value))
              }
              disabled={snapshot.scramLatched}
            />
            <span className="vertical-slider-value">
              {snapshot.rodInsertionPercent.toFixed(1)}%
            </span>
          </div>

          <div className="overview-controls__buttons">
            <button type="button" className="button button--danger" onClick={scram}>
              SCRAM
            </button>
            <button
              type="button"
              className="button"
              onClick={() => setRunning((c) => !c)}
            >
              {running ? 'Pause' : 'Resume'}
            </button>
            <button
              type="button"
              className="button button--secondary"
              onClick={reset}
            >
              Reset
            </button>
          </div>

          {snapshot.scramLatched && (
            <p className="alert-banner">
              SCRAM latched — reset to clear.
            </p>
          )}
        </div>

        <CoolingLoopDiagram thermal={thermal} snapshot={snapshot} />

        <div className="overview-trends-stack">
          <TrendChart title="Neutronics" series={neutronicsSeries} />
          <TrendChart title="Thermal Hydraulics" series={thSeries} />
        </div>
      </div>

      {/* ── Info footer ── */}
      <div className="panel overview-info">
        <p>
          Heavy-water moderated research reactor point-kinetics demonstrator with
          Modelica-coupled thermal hydraulics. The control rod bank adjusts
          reactivity; the primary D₂O cooling loop transports heat from the annular
          core through a heat exchanger. Six delayed neutron groups are solved
          implicitly in the Python backend.
        </p>
      </div>
    </div>
  )
}
