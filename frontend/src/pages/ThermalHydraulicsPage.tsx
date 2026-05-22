import { CoolingLoopDiagram } from '../components/CoolingLoopDiagram'
import { LineChart } from '../components/LineChart'
import { MetricCard } from '../components/MetricCard'
import { useReactorSimulation } from '../hooks/useReactorSimulation'
import type { ThermalSnapshot } from '../simulation/types'
import {
  formatMassFlow,
  formatPowerMw,
  formatPressureDropPa,
  formatSimTime,
  formatTemperatureK,
} from '../utils/format'

const thermalSourceCopy = {
  fmu: {
    label: 'FMU coupled',
    tone: 'rgba(20,83,45,0.3)',
    border: 'rgba(74,222,128,0.35)',
    body: 'Live inlet and outlet temperatures are coming from the Modelica thermal-hydraulics FMU.',
  },
  fallback: {
    label: 'Fallback thermal model',
    tone: 'rgba(120,53,15,0.35)',
    border: 'rgba(245,158,11,0.4)',
    body: 'Backend is serving the degraded thermal model because the FMU path is temporarily unavailable.',
  },
  unavailable: {
    label: 'Thermal data unavailable',
    tone: 'rgba(127,29,29,0.35)',
    border: 'rgba(248,113,113,0.4)',
    body: 'No thermal data is currently available from the backend.',
  },
} as const

export function ThermalHydraulicsPage() {
  const { error, loading, model, snapshot, thermal, thermalHistory } = useReactorSimulation()

  if (!snapshot || !model || !thermal) {
    return (
      <div className="app-shell">
        <header className="hero-header panel">
          <div>
            <p className="eyebrow">Thermal hydraulics</p>
            <h1>Coolant &amp; heat transfer</h1>
            <p className="hero-copy">
              The thermal-hydraulics page now reads from the backend coupling path.
              If data is missing here, the Python backend or FMU runtime is not ready yet.
            </p>
          </div>
        </header>

        <section className="panel loading-panel">
          <h2>{loading ? 'Connecting to thermal backend' : 'Thermal backend unavailable'}</h2>
          <p className="loading-copy">{error ?? 'Thermal data has not been received yet.'}</p>
        </section>
      </div>
    )
  }

  const source = thermalSourceCopy[thermal.source]
  const inletMarginC = thermal.inletTemperatureK === null ? null : thermal.inletTemperatureK - 298.15
  const outletMarginC = thermal.outletTemperatureK === null ? null : thermal.outletTemperatureK - 318.15
  const powerShare = (snapshot.thermalPowerMw / model.nominalThermalPowerMw) * 100

  return (
    <div className="app-shell">
      <header className="hero-header panel">
        <div>
          <p className="eyebrow">Thermal hydraulics</p>
          <h1>Coolant &amp; heat transfer</h1>
          <p className="hero-copy">
            The selected design is a simplified forced-circulation D₂O primary loop
            with a target 25 °C inlet, 45 °C nominal outlet, and 237 kg/s nominal
            flow. This page now surfaces the live backend coupling state from the
            thermal-hydraulics model instead of a placeholder architecture note.
          </p>
        </div>
        <div
          className="status-badge"
          style={{ borderColor: source.border, background: source.tone }}
        >
          <span className="status-badge__label">{source.label}</span>
          <span>{source.body}</span>
          <span>
            Thermal time {formatSimTime(thermal.timeSeconds)} at {formatPowerMw(thermal.powerMw)}.
          </span>
        </div>
      </header>

      <section className="metrics-grid">
        <MetricCard
          label="Core inlet"
          value={formatTemperatureK(thermal.inletTemperatureK)}
          detail={
            inletMarginC === null
              ? 'No thermal inlet value from backend'
              : `${inletMarginC >= 0 ? '+' : ''}${inletMarginC.toFixed(1)} °C versus 25 °C target`
          }
          tone={inletMarginC !== null && inletMarginC > 5 ? 'warning' : 'cool'}
        />
        <MetricCard
          label="Core outlet"
          value={formatTemperatureK(thermal.outletTemperatureK)}
          detail={
            outletMarginC === null
              ? 'No thermal outlet value from backend'
              : `${outletMarginC >= 0 ? '+' : ''}${outletMarginC.toFixed(1)} °C versus 45 °C target`
          }
          tone={
            thermal.outletTemperatureK !== null && thermal.outletTemperatureK > 328.15
              ? 'danger'
              : thermal.outletTemperatureK !== null && thermal.outletTemperatureK > 323.15
                ? 'warning'
                : 'cool'
          }
        />
        <MetricCard
          label="Primary flow"
          value={formatMassFlow(thermal.massFlowKgPerSecond)}
          detail="Forced-circulation primary-side mass flow from the TH model"
          tone="neutral"
        />
        <MetricCard
          label="Core ΔP"
          value={formatPressureDropPa(thermal.corePressureDropPa)}
          detail="Inlet minus outlet pressure across the effective core channel"
          tone="neutral"
        />
      </section>

      {thermal.message ? <p className="alert-banner">{thermal.message}</p> : null}

      <CoolingLoopDiagram thermal={thermal} />

      <section className="th-charts-grid">
        <LineChart<ThermalSnapshot>
          title="Core inlet temperature"
          subtitle="Primary coolant temperature entering the core — target 25 °C"
          color="#38bdf8"
          data={thermalHistory}
          valueAccessor={t => t.inletTemperatureK ?? 0}
          valueFormatter={k => formatTemperatureK(k)}
        />
        <LineChart<ThermalSnapshot>
          title="Core outlet temperature"
          subtitle="Primary coolant temperature exiting the core — nominal 45 °C"
          color="#f59e0b"
          data={thermalHistory}
          valueAccessor={t => t.outletTemperatureK ?? 0}
          valueFormatter={k => formatTemperatureK(k)}
        />
        <LineChart<ThermalSnapshot>
          title="Primary mass flow"
          subtitle="Forced-circulation D₂O flow rate through the core channel"
          color="#4ade80"
          data={thermalHistory}
          valueAccessor={t => t.massFlowKgPerSecond ?? 0}
          valueFormatter={v => formatMassFlow(v)}
        />
        <LineChart<ThermalSnapshot>
          title="Core pressure drop"
          subtitle="Differential pressure across the effective core channel (inlet − outlet)"
          color="#a78bfa"
          data={thermalHistory}
          valueAccessor={t => t.corePressureDropPa ?? 0}
          valueFormatter={v => formatPressureDropPa(v)}
        />
      </section>

      <section className="notes-grid">
        <article className="panel note-panel">
          <p className="section-label">Live coupling state</p>
          <h2>Backend-driven thermal response</h2>
          <ul className="note-list">
            <li><strong style={{ color: 'var(--text-strong)' }}>Thermal source:</strong> {source.label}.</li>
            <li><strong style={{ color: 'var(--text-strong)' }}>Reactor power:</strong> {formatPowerMw(snapshot.thermalPowerMw)} ({powerShare.toFixed(1)}% of nominal).</li>
            <li><strong style={{ color: 'var(--text-strong)' }}>Rod insertion:</strong> {snapshot.rodInsertionPercent.toFixed(1)}%.</li>
            <li><strong style={{ color: 'var(--text-strong)' }}>Thermal clock:</strong> {formatSimTime(thermal.timeSeconds)}.</li>
            <li>Current backend path is one-way: neutronics power drives thermal hydraulics, but temperature feedback is not yet wired back into reactivity.</li>
          </ul>
        </article>

        <article className="panel note-panel">
          <p className="section-label">Primary loop — selected design</p>
          <h2>D₂O forced circulation</h2>
          <ul className="note-list">
            <li><strong style={{ color: 'var(--text-strong)' }}>Flow direction:</strong> top-to-bottom through the effective core channel.</li>
            <li><strong style={{ color: 'var(--text-strong)' }}>Design inlet:</strong> 25 °C.</li>
            <li><strong style={{ color: 'var(--text-strong)' }}>Design outlet:</strong> 45 °C nominal, 55 °C upper limit.</li>
            <li><strong style={{ color: 'var(--text-strong)' }}>Nominal mass flow:</strong> 237 kg/s.</li>
            <li><strong style={{ color: 'var(--text-strong)' }}>Pressure strategy:</strong> modestly pressurized loop with an external pressure reference in the current TH model.</li>
          </ul>
        </article>

        <article className="panel note-panel">
          <p className="section-label">Current TH model</p>
          <h2>What the backend is reading</h2>
          <ul className="note-list">
            <li>One effective 8-node core channel plus inlet and outlet plena.</li>
            <li>Single primary pump and distributed HX on the return side.</li>
            <li>Open pool represented as a thermal reservoir, not part of the forced-flow path.</li>
            <li>Backend reads T_in, T_out, ṁ, and ΔP from the FMU surface.</li>
          </ul>
        </article>

        <article className="panel note-panel">
          <p className="section-label">Still out of scope</p>
          <h2>What this page does not yet show</h2>
          <ul className="note-list">
            <li>No thermal-history chart yet; only live thermal state is exposed in the frontend today.</li>
            <li>No moderator or Doppler feedback back into the point-kinetics model yet.</li>
            <li>No head-curve pump, coastdown, or secondary-loop transient modeling yet.</li>
          </ul>
        </article>
      </section>
    </div>
  )
}
