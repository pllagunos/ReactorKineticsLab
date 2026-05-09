import './App.css'
import { CoreSchematic } from './components/CoreSchematic'
import { LineChart } from './components/LineChart'
import { MetricCard } from './components/MetricCard'
import { useReactorSimulation } from './hooks/useReactorSimulation'
import type { ReactorRegime } from './simulation/types'
import {
  formatDollars,
  formatFlux,
  formatPeriodSeconds,
  formatPowerMw,
  formatReactivityPcm,
  formatRodInsertion,
  formatSimTime,
  formatPercent,
} from './utils/format'

const regimeCopy: Record<ReactorRegime, { title: string; body: string }> = {
  scrammed: {
    title: 'SCRAMMED',
    body: 'Full insertion plus shutdown margin is latched until reset.',
  },
  'near-critical': {
    title: 'Near critical',
    body: 'Reactivity is close to zero, so flux and power stay near nominal.',
  },
  supercritical: {
    title: 'Supercritical',
    body: 'Positive reactivity is increasing flux and thermal power.',
  },
  subcritical: {
    title: 'Subcritical',
    body: 'Negative reactivity is driving the chain reaction down.',
  },
}

function App() {
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
  } =
    useReactorSimulation()

  if (!snapshot || !model) {
    return (
      <div className="app-shell">
        <header className="hero-header panel">
          <div>
            <p className="eyebrow">Bun + React + Python reactor demonstrator</p>
            <h1>Heavy-water reactor point-kinetics sandbox</h1>
            <p className="hero-copy">
              The dashboard now reads its reactor state from a local Python backend
              so the simulation core can move into the scientific Python ecosystem
              while keeping the current browser UI.
            </p>
          </div>
        </header>

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

  const status = regimeCopy[snapshot.status]
  const powerShare = (snapshot.thermalPowerMw / model.nominalThermalPowerMw) * 100
  const fluxShare =
    (snapshot.totalFlux / model.nominalFluxNeutronsPerSquareCentimeterSecond) * 100

  return (
    <div className="app-shell">
      <header className="hero-header panel">
        <div>
          <p className="eyebrow">Bun + React + Python reactor demonstrator</p>
          <h1>Heavy-water reactor point-kinetics sandbox</h1>
          <p className="hero-copy">
            This first slice models one hollow cylindrical core with fixed D2O
            moderation assumptions and one operator input: control rod insertion.
            The UI stays in React, while the transient simulation now runs in a
            local Python backend.
          </p>
        </div>
        <div className={`status-badge status-badge--${snapshot.status}`}>
          <span className="status-badge__label">{status.title}</span>
          <span>{status.body}</span>
        </div>
      </header>

      <section className="metrics-grid">
        <MetricCard
          label="Reactivity"
          value={formatReactivityPcm(snapshot.reactivity.totalPcm)}
          detail={`${formatDollars(snapshot.reactivity.dollars)} relative to beta_eff`}
          tone={
            snapshot.status === 'supercritical'
              ? 'warning'
              : snapshot.status === 'scrammed'
                ? 'danger'
                : 'neutral'
          }
        />
        <MetricCard
          label="Total flux"
          value={formatFlux(snapshot.totalFlux)}
          detail={`${formatPercent(fluxShare)} of nominal annular-core flux`}
          tone="cool"
        />
        <MetricCard
          label="Thermal power"
          value={formatPowerMw(snapshot.thermalPowerMw)}
          detail={`${formatPercent(powerShare)} of ${model.nominalThermalPowerMw} MWth rating`}
          tone={
            snapshot.thermalPowerMw >= model.autoScramPowerMw
              ? 'danger'
              : powerShare >= 105
                ? 'warning'
                : 'neutral'
          }
        />
        <MetricCard
          label="Simulated clock"
          value={formatSimTime(snapshot.timeSeconds)}
          detail={`Neutron period ${formatPeriodSeconds(snapshot.lastPeriodSeconds)}`}
          tone="neutral"
        />
      </section>

      <section className="dashboard-grid">
        <article className="panel controls-panel">
          <div className="panel-heading">
            <div>
              <p className="section-label">Core state</p>
              <h2>Control rod insertion</h2>
            </div>
            <button
              type="button"
              className="button button--danger"
              onClick={scram}
            >
              SCRAM
            </button>
          </div>

          <CoreSchematic
            insertionPercent={snapshot.rodInsertionPercent}
            scramLatched={snapshot.scramLatched}
          />

          <div className="slider-block">
            <div className="slider-header">
              <label htmlFor="rod-insertion">Rod bank position</label>
              <span className="slider-value">
                {formatRodInsertion(snapshot.rodInsertionPercent)}
              </span>
            </div>
            <input
              id="rod-insertion"
              type="range"
              min="0"
              max="100"
              step="0.5"
              value={snapshot.rodInsertionPercent}
              onChange={(event) =>
                setRodInsertionPercent(Number(event.currentTarget.value))
              }
              disabled={snapshot.scramLatched}
            />
            <div className="slider-scale">
              <span>0% withdrawn</span>
              <span>100% inserted</span>
            </div>
          </div>

          <div className="button-row">
            <button
              type="button"
              className="button"
              onClick={() => setRunning((current) => !current)}
            >
              {running ? 'Pause transient' : 'Resume transient'}
            </button>
            <button type="button" className="button button--secondary" onClick={reset}>
              Reset to critical
            </button>
          </div>

          {snapshot.scramLatched ? (
            <p className="alert-banner">
              SCRAM is latched. The modeled operator rod is fully inserted and an
              extra shutdown margin is applied until reset.
            </p>
          ) : null}

          <dl className="facts-grid">
            <div>
              <dt>Core annulus</dt>
              <dd>
                {model.coreGeometry.innerRadiusMeters.toFixed(1)} m inner radius
                to {model.coreGeometry.outerRadiusMeters.toFixed(1)} m
              </dd>
            </div>
            <div>
              <dt>Active height</dt>
              <dd>{model.coreGeometry.activeHeightMeters.toFixed(1)} m</dd>
            </div>
            <div>
              <dt>Critical setpoint</dt>
              <dd>{formatRodInsertion(model.criticalRodInsertionPercent)}</dd>
            </div>
            <div>
              <dt>Rod worth</dt>
              <dd>{model.totalControlRodWorthPcm} pcm total bank worth</dd>
            </div>
          </dl>
        </article>

        <section className="trends-column">
          <LineChart
            title="Reactivity trend"
            subtitle="Rod motion changes reactivity in pcm. Zero crossing is near 50% insertion."
            color="#f59e0b"
            data={history}
            valueAccessor={(point) => point.reactivityPcm}
            valueFormatter={formatReactivityPcm}
          />
          <LineChart
            title="Total flux trend"
            subtitle="Flux follows the point-kinetics neutron population and is shown in n/cm^2/s."
            color="#38bdf8"
            data={history}
            valueAccessor={(point) => point.totalFlux}
            valueFormatter={formatFlux}
          />
          <LineChart
            title="Thermal power trend"
            subtitle={`Power is scaled from nominal ${model.nominalThermalPowerMw} MWth. Automatic SCRAM trips at ${model.autoScramPowerMw} MWth.`}
            color="#f97316"
            data={history}
            valueAccessor={(point) => point.thermalPowerMw}
            valueFormatter={formatPowerMw}
          />
        </section>
      </section>

      <section className="notes-grid">
        <article className="panel note-panel">
          <p className="section-label">Model assumptions</p>
          <h2>What this version includes</h2>
          <ul className="note-list">
            <li>One annular core with a fixed heavy-water moderation assumption.</li>
            <li>One operator-controlled rod bank with sinusoidal worth shaping.</li>
            <li>
              Six delayed neutron groups solved in the Python backend with a
              stable implicit integration step.
            </li>
          </ul>
        </article>

        <article className="panel note-panel">
          <p className="section-label">Observables</p>
          <h2>How to read the dashboard</h2>
          <ul className="note-list">
            <li>Positive reactivity drives the core supercritical.</li>
            <li>Flux and power scale from nominal operating conditions.</li>
            <li>
              The simulated clock can run faster than wall time so short transients
              are easier to inspect.
            </li>
          </ul>
        </article>
      </section>
    </div>
  )
}

export default App
