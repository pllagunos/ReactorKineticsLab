import { useCallback, useRef } from 'react'
import { useTransientDiffusion } from '../hooks/useTransientDiffusion'
import type { TransientDiffusionState, TransientHistoryPoint, CoreFluxGeometry, CoreFluxProfile } from '../simulation/types'

// ---------------------------------------------------------------------------
// Colour map (identical to CorePage)
// ---------------------------------------------------------------------------

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

function fluxColor(t: number): string {
  const stops: [number, number, number][] = [
    [2, 10, 35],
    [12, 74, 110],
    [8, 145, 178],
    [245, 158, 11],
    [254, 243, 199],
  ]
  const positions = [0, 0.25, 0.55, 0.8, 1.0]
  let i = 0
  while (i < positions.length - 2 && t > positions[i + 1]) i++
  const span = positions[i + 1] - positions[i]
  const localT = span === 0 ? 0 : (t - positions[i]) / span
  const r = Math.round(lerp(stops[i][0], stops[i + 1][0], localT))
  const g = Math.round(lerp(stops[i][1], stops[i + 1][1], localT))
  const b = Math.round(lerp(stops[i][2], stops[i + 1][2], localT))
  return `rgb(${r},${g},${b})`
}

// ---------------------------------------------------------------------------
// Shared visual dimensions
// ---------------------------------------------------------------------------

const SVG_W = 420
const SVG_H = 300
const PROFILE_W = 480
const PROFILE_H = 120
const PROFILE_PAD = 14

// ---------------------------------------------------------------------------
// Heatmap
// ---------------------------------------------------------------------------

interface HeatmapData {
  heatmapRCm: number[]
  heatmapZCm: number[]
  heatmapPhi: number[][]
  geometry: CoreFluxGeometry
}

function Heatmap2D({ data }: { data: HeatmapData }) {
  const { heatmapRCm, heatmapZCm, heatmapPhi, geometry } = data
  const nr = heatmapRCm.length
  const nz = heatmapZCm.length
  const cellW = SVG_W / nr
  const cellH = SVG_H / nz

  let maxPhi = 0
  for (let ir = 0; ir < nr; ir++) {
    for (let iz = 0; iz < nz; iz++) {
      if (heatmapPhi[ir][iz] > maxPhi) maxPhi = heatmapPhi[ir][iz]
    }
  }
  const norm = maxPhi || 1

  const cells: React.ReactNode[] = []
  for (let ir = 0; ir < nr; ir++) {
    for (let iz = 0; iz < nz; iz++) {
      const t = heatmapPhi[ir][iz] / norm
      cells.push(
        <rect
          key={`${ir}-${iz}`}
          x={ir * cellW}
          y={(nz - 1 - iz) * cellH}
          width={cellW + 0.5}
          height={cellH + 0.5}
          fill={fluxColor(t)}
        />,
      )
    }
  }

  const rMax = heatmapRCm[nr - 1]
  const zMin = heatmapZCm[0]
  const zMax = heatmapZCm[nz - 1]
  const zSpan = zMax - zMin

  const xInner = (geometry.rInnerCm / rMax) * SVG_W
  const xFuel = (geometry.rFuelCm / rMax) * SVG_W
  const yActiveTop = ((zMax - geometry.hActiveCm / 2) / zSpan) * SVG_H
  const yActiveBot = ((zMax + geometry.hActiveCm / 2) / zSpan) * SVG_H

  return (
    <svg
      viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      className="flux-heatmap-svg"
      role="img"
      aria-label="2D r-z transient thermal flux distribution"
    >
      {cells}
      <line x1={xInner} y1={0} x2={xInner} y2={SVG_H}
        stroke="rgba(148,163,184,0.45)" strokeWidth="1.5" strokeDasharray="4 3" />
      <line x1={xFuel} y1={0} x2={xFuel} y2={SVG_H}
        stroke="rgba(148,163,184,0.45)" strokeWidth="1.5" strokeDasharray="4 3" />
      <line x1={0} y1={yActiveTop} x2={SVG_W} y2={yActiveTop}
        stroke="rgba(148,163,184,0.3)" strokeWidth="1" strokeDasharray="3 4" />
      <line x1={0} y1={yActiveBot} x2={SVG_W} y2={yActiveBot}
        stroke="rgba(148,163,184,0.3)" strokeWidth="1" strokeDasharray="3 4" />
      <text x={xInner / 2} y={SVG_H - 6} textAnchor="middle" className="flux-label">D₂O</text>
      <text x={(xInner + xFuel) / 2} y={SVG_H - 6} textAnchor="middle" className="flux-label">Fuel annulus</text>
      <text x={(xFuel + SVG_W) / 2} y={SVG_H - 6} textAnchor="middle" className="flux-label">Reflector</text>
      <text x={SVG_W - 4} y={SVG_H / 2} textAnchor="end" dominantBaseline="middle" className="flux-label">r →</text>
      <text x={6} y={8} textAnchor="start" className="flux-label">z ↑</text>
    </svg>
  )
}

// ---------------------------------------------------------------------------
// Radial profile
// ---------------------------------------------------------------------------

function RadialProfile({ profile, geometry }: { profile: CoreFluxProfile; geometry: CoreFluxGeometry }) {
  const { axisCm: rs, phiNorm: pts } = profile
  const n = rs.length
  const max = Math.max(...pts)
  const rMax = rs[n - 1]

  const xs = rs.map((r) => PROFILE_PAD + (r / rMax) * (PROFILE_W - PROFILE_PAD * 2))
  const ys = pts.map((v) => PROFILE_H - PROFILE_PAD - (v / max) * (PROFILE_H - PROFILE_PAD * 2))
  const linePath = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(' ')
  const areaPath = `${linePath} L${xs.at(-1)!.toFixed(1)},${(PROFILE_H - PROFILE_PAD).toFixed(1)} L${xs[0].toFixed(1)},${(PROFILE_H - PROFILE_PAD).toFixed(1)} Z`

  const xInner = PROFILE_PAD + (geometry.rInnerCm / rMax) * (PROFILE_W - PROFILE_PAD * 2)
  const xFuel = PROFILE_PAD + (geometry.rFuelCm / rMax) * (PROFILE_W - PROFILE_PAD * 2)

  return (
    <svg viewBox={`0 0 ${PROFILE_W} ${PROFILE_H}`} className="profile-svg" role="img" aria-label="Radial flux profile">
      <path d={areaPath} fill="#38bdf8" fillOpacity="0.14" />
      <path d={linePath} fill="none" stroke="#38bdf8" strokeWidth="2.5" strokeLinejoin="round" />
      <line x1={xInner} y1={PROFILE_PAD} x2={xInner} y2={PROFILE_H - PROFILE_PAD} stroke="rgba(148,163,184,0.4)" strokeWidth="1" strokeDasharray="3 3" />
      <line x1={xFuel} y1={PROFILE_PAD} x2={xFuel} y2={PROFILE_H - PROFILE_PAD} stroke="rgba(148,163,184,0.4)" strokeWidth="1" strokeDasharray="3 3" />
      <line x1={PROFILE_PAD} y1={PROFILE_H - PROFILE_PAD} x2={PROFILE_W - PROFILE_PAD} y2={PROFILE_H - PROFILE_PAD} stroke="rgba(148,163,184,0.25)" />
      <line x1={PROFILE_PAD} y1={PROFILE_PAD} x2={PROFILE_PAD} y2={PROFILE_H - PROFILE_PAD} stroke="rgba(148,163,184,0.25)" />
      <text x={xInner / 2 + PROFILE_PAD / 2} y={PROFILE_H - 3} textAnchor="middle" className="flux-label" fontSize="8">D₂O</text>
      <text x={(xInner + xFuel) / 2} y={PROFILE_H - 3} textAnchor="middle" className="flux-label" fontSize="8">Fuel</text>
      <text x={(xFuel + PROFILE_W - PROFILE_PAD) / 2} y={PROFILE_H - 3} textAnchor="middle" className="flux-label" fontSize="8">Refl.</text>
    </svg>
  )
}

// ---------------------------------------------------------------------------
// Axial profile
// ---------------------------------------------------------------------------

function AxialProfile({ profile, geometry }: { profile: CoreFluxProfile; geometry: CoreFluxGeometry }) {
  const { axisCm: zs, phiNorm: pts } = profile
  const n = zs.length
  const max = Math.max(...pts)
  const zMin = zs[0]
  const zMax = zs[n - 1]
  const zSpan = zMax - zMin

  const xs = zs.map((z) => PROFILE_PAD + ((z - zMin) / zSpan) * (PROFILE_W - PROFILE_PAD * 2))
  const ys = pts.map((v) => PROFILE_H - PROFILE_PAD - (v / max) * (PROFILE_H - PROFILE_PAD * 2))
  const linePath = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(' ')
  const areaPath = `${linePath} L${xs.at(-1)!.toFixed(1)},${(PROFILE_H - PROFILE_PAD).toFixed(1)} L${xs[0].toFixed(1)},${(PROFILE_H - PROFILE_PAD).toFixed(1)} Z`

  const xActiveL = PROFILE_PAD + ((-geometry.hActiveCm / 2 - zMin) / zSpan) * (PROFILE_W - PROFILE_PAD * 2)
  const xActiveR = PROFILE_PAD + ((geometry.hActiveCm / 2 - zMin) / zSpan) * (PROFILE_W - PROFILE_PAD * 2)

  return (
    <svg viewBox={`0 0 ${PROFILE_W} ${PROFILE_H}`} className="profile-svg" role="img" aria-label="Axial flux profile">
      <path d={areaPath} fill="#f59e0b" fillOpacity="0.14" />
      <path d={linePath} fill="none" stroke="#f59e0b" strokeWidth="2.5" strokeLinejoin="round" />
      <line x1={xActiveL} y1={PROFILE_PAD} x2={xActiveL} y2={PROFILE_H - PROFILE_PAD} stroke="rgba(148,163,184,0.4)" strokeWidth="1" strokeDasharray="3 3" />
      <line x1={xActiveR} y1={PROFILE_PAD} x2={xActiveR} y2={PROFILE_H - PROFILE_PAD} stroke="rgba(148,163,184,0.4)" strokeWidth="1" strokeDasharray="3 3" />
      <line x1={PROFILE_PAD} y1={PROFILE_H - PROFILE_PAD} x2={PROFILE_W - PROFILE_PAD} y2={PROFILE_H - PROFILE_PAD} stroke="rgba(148,163,184,0.25)" />
      <line x1={PROFILE_PAD} y1={PROFILE_PAD} x2={PROFILE_PAD} y2={PROFILE_H - PROFILE_PAD} stroke="rgba(148,163,184,0.25)" />
      <text x={(xActiveL) / 2 + PROFILE_PAD / 2} y={PROFILE_H - 3} textAnchor="middle" className="flux-label" fontSize="8">Refl.</text>
      <text x={(xActiveL + xActiveR) / 2} y={PROFILE_H - 3} textAnchor="middle" className="flux-label" fontSize="8">Active zone</text>
      <text x={(xActiveR + PROFILE_W - PROFILE_PAD) / 2} y={PROFILE_H - 3} textAnchor="middle" className="flux-label" fontSize="8">Refl.</text>
    </svg>
  )
}

// ---------------------------------------------------------------------------
// Trend chart  (scalar history vs simulation time)
// ---------------------------------------------------------------------------

const TREND_W = 480
const TREND_H = 130
const TREND_PAD_L = 44
const TREND_PAD_R = 12
const TREND_PAD_T = 14
const TREND_PAD_B = 24

interface TrendChartProps {
  history: TransientHistoryPoint[]
  field: 'reactivityPcm' | 'powerNorm'
  label: string
  unit: string
  color: string
  zeroLine?: boolean
}

function TrendChart({ history, field, label, unit, color, zeroLine = false }: TrendChartProps) {
  if (history.length < 2) {
    return (
      <svg viewBox={`0 0 ${TREND_W} ${TREND_H}`} className="profile-svg" aria-label={`${label} trend`}>
        <text x={TREND_W / 2} y={TREND_H / 2} textAnchor="middle" className="flux-label">No data yet</text>
      </svg>
    )
  }

  const times = history.map((h) => h.timeSeconds)
  const values = history.map((h) => h[field] as number)

  const tMin = times[0]
  const tMax = times[times.length - 1]
  const vMin = Math.min(...values)
  const vMax = Math.max(...values)
  const tSpan = tMax - tMin || 1
  const vSpan = vMax - vMin || 1

  const plotW = TREND_W - TREND_PAD_L - TREND_PAD_R
  const plotH = TREND_H - TREND_PAD_T - TREND_PAD_B

  const toX = (t: number) => TREND_PAD_L + ((t - tMin) / tSpan) * plotW
  const toY = (v: number) => TREND_PAD_T + (1 - (v - vMin) / vSpan) * plotH

  const linePath = times.map((t, i) => `${i === 0 ? 'M' : 'L'}${toX(t).toFixed(1)},${toY(values[i]).toFixed(1)}`).join(' ')
  const areaPath = `${linePath} L${toX(tMax).toFixed(1)},${(TREND_PAD_T + plotH).toFixed(1)} L${toX(tMin).toFixed(1)},${(TREND_PAD_T + plotH).toFixed(1)} Z`

  // Axis label ticks
  const nTicks = 4
  const yTicks = Array.from({ length: nTicks + 1 }, (_, i) => vMin + (i / nTicks) * vSpan)
  const xTicks = Array.from({ length: 5 }, (_, i) => tMin + (i / 4) * tSpan)

  return (
    <svg viewBox={`0 0 ${TREND_W} ${TREND_H}`} className="profile-svg" role="img" aria-label={`${label} history`}>
      {/* Grid lines */}
      {yTicks.map((v, i) => (
        <line key={i} x1={TREND_PAD_L} y1={toY(v)} x2={TREND_PAD_L + plotW} y2={toY(v)}
          stroke="rgba(148,163,184,0.1)" strokeWidth="1" />
      ))}
      {/* Zero line */}
      {zeroLine && vMin <= 0 && vMax >= 0 && (
        <line x1={TREND_PAD_L} y1={toY(0)} x2={TREND_PAD_L + plotW} y2={toY(0)}
          stroke="rgba(148,163,184,0.35)" strokeWidth="1" strokeDasharray="4 3" />
      )}
      {/* Area fill */}
      <path d={areaPath} fill={color} fillOpacity="0.12" />
      {/* Line */}
      <path d={linePath} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />
      {/* Axes */}
      <line x1={TREND_PAD_L} y1={TREND_PAD_T} x2={TREND_PAD_L} y2={TREND_PAD_T + plotH} stroke="rgba(148,163,184,0.3)" />
      <line x1={TREND_PAD_L} y1={TREND_PAD_T + plotH} x2={TREND_PAD_L + plotW} y2={TREND_PAD_T + plotH} stroke="rgba(148,163,184,0.3)" />
      {/* Y ticks */}
      {yTicks.map((v, i) => (
        <text key={i} x={TREND_PAD_L - 3} y={toY(v)} textAnchor="end" dominantBaseline="middle"
          className="flux-label" fontSize="8">
          {field === 'reactivityPcm' ? v.toFixed(0) : v.toFixed(2)}
        </text>
      ))}
      {/* X ticks */}
      {xTicks.map((t, i) => (
        <text key={i} x={toX(t)} y={TREND_PAD_T + plotH + 10} textAnchor="middle"
          className="flux-label" fontSize="8">
          {t.toFixed(0)}s
        </text>
      ))}
      {/* Labels */}
      <text x={TREND_PAD_L + plotW} y={TREND_PAD_T - 2} textAnchor="end"
        className="flux-label" fontSize="9" fill={color}>
        {label} ({unit})
      </text>
    </svg>
  )
}

// ---------------------------------------------------------------------------
// Rod slider control
// ---------------------------------------------------------------------------

function RodSlider({
  value,
  onChange,
  disabled,
}: {
  value: number
  onChange: (v: number) => void
  disabled: boolean
}) {
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const handle = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const v = Number(e.target.value)
      if (debounceRef.current) clearTimeout(debounceRef.current)
      debounceRef.current = setTimeout(() => onChange(v), 100)
    },
    [onChange],
  )

  return (
    <div className="transient-rod-control">
      <label className="section-label" htmlFor="transient-rod">
        Rod insertion — {value.toFixed(1)} %
      </label>
      <div className="rod-slider-track">
        <span className="rod-slider-cap">Withdrawn (0 %)</span>
        <input
          id="transient-rod"
          type="range"
          min={0}
          max={100}
          step={0.5}
          defaultValue={value}
          onChange={handle}
          disabled={disabled}
          className="rod-slider"
        />
        <span className="rod-slider-cap">Inserted (100 %)</span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Status badge
// ---------------------------------------------------------------------------

function StatusBadge({ state }: { state: TransientDiffusionState }) {
  const rhoColor = state.reactivityPcm > 5 ? '#4ade80' : state.reactivityPcm < -5 ? '#f87171' : '#fbbf24'
  const powerColor = state.powerNorm > 1.05 ? '#4ade80' : state.powerNorm < 0.95 ? '#f87171' : '#94a3b8'

  return (
    <div className="transient-status-row">
      <div className="transient-metric">
        <span className="transient-metric__label">Sim. time</span>
        <span className="transient-metric__value">{state.timeSeconds.toFixed(0)} s</span>
      </div>
      <div className="transient-metric">
        <span className="transient-metric__label">Reactivity ρ</span>
        <span className="transient-metric__value" style={{ color: rhoColor }}>
          {state.reactivityPcm > 0 ? '+' : ''}{state.reactivityPcm.toFixed(1)} pcm
        </span>
      </div>
      <div className="transient-metric">
        <span className="transient-metric__label">Power P/P₀</span>
        <span className="transient-metric__value" style={{ color: powerColor }}>
          {state.powerNorm.toFixed(4)}
        </span>
      </div>
      <div className="transient-metric">
        <span className="transient-metric__label">Steps</span>
        <span className="transient-metric__value">{state.stepCount}</span>
      </div>
      <div className="transient-metric">
        <span className="transient-metric__label">dt</span>
        <span className="transient-metric__value">{state.dt} s</span>
      </div>
      <div className="transient-metric">
        <span className="transient-metric__label">Mesh</span>
        <span className="transient-metric__value">Δr=Δz={state.meshDrCm} cm</span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export function TransientDiffusionPage() {
  const { state, loading, error, setRunning, setRodInsertion, reset, manualStep } = useTransientDiffusion()

  const isRunning = state?.running ?? false
  const isBusy = loading && state === null

  return (
    <div className="app-shell">
      <header className="hero-header panel">
        <div>
          <p className="eyebrow">Time-dependent spatial neutronics</p>
          <h1>Transient Diffusion</h1>
          <p className="hero-copy">
            True 2D r-z time-dependent one-group diffusion with six delayed-neutron
            precursor groups. Each second of simulation time advances one implicit
            Euler step; the flux amplitude and all six precursor fields evolve on the
            full spatial mesh. Adjust the control rod and observe how the spatial
            flux shape, reactivity, and power evolve over time.
          </p>
        </div>

        {error && (
          <div className="status-badge" style={{ borderColor: 'rgba(248,113,113,0.25)', background: 'rgba(127,29,29,0.2)' }}>
            <span className="status-badge__label">Backend unreachable</span>
            <span>{error}</span>
          </div>
        )}

        {isBusy && (
          <div className="status-badge" style={{ borderColor: 'rgba(59,130,246,0.25)', background: 'rgba(30,58,138,0.2)' }}>
            <span className="status-badge__label">Initializing…</span>
            <span>Running eigenvalue solve for initial critical state. First load may take ~10 s.</span>
          </div>
        )}

        {state && !isBusy && (
          <div className="status-badge" style={{ borderColor: isRunning ? 'rgba(74,222,128,0.25)' : 'rgba(148,163,184,0.2)', background: isRunning ? 'rgba(5,46,22,0.2)' : 'rgba(15,23,42,0.3)' }}>
            <span className="status-badge__label">{isRunning ? '▶ Running' : '⏸ Paused'}</span>
            <span>t = {state.timeSeconds.toFixed(0)} s · step {state.stepCount}</span>
          </div>
        )}
      </header>

      {/* Controls */}
      <section className="panel core-flux-panel">
        <div className="panel-heading">
          <div>
            <p className="section-label">Simulation controls</p>
            <h2>Rod & time controls</h2>
          </div>
          <div className="transient-buttons">
            <button
              className="btn-refresh"
              onClick={() => setRunning(!isRunning)}
              disabled={isBusy}
              aria-label={isRunning ? 'Pause simulation' : 'Run simulation'}
            >
              {isRunning ? '⏸ Pause' : '▶ Run'}
            </button>
            <button
              className="btn-refresh"
              onClick={() => manualStep()}
              disabled={isBusy || isRunning}
              title="Advance one time step (only available when paused)"
            >
              Step →
            </button>
            <button
              className="btn-refresh"
              onClick={() => reset()}
              disabled={isBusy}
            >
              ↺ Reset
            </button>
          </div>
        </div>

        {state && (
          <RodSlider
            value={state.rodInsertionPercent}
            onChange={setRodInsertion}
            disabled={isBusy}
          />
        )}

        {state && <StatusBadge state={state} />}
      </section>

      {/* 2D flux heatmap */}
      <section className="panel core-flux-panel">
        <div className="panel-heading">
          <div>
            <p className="section-label">2D r-z flux map</p>
            <h2>Transient flux distribution</h2>
          </div>
        </div>
        <p className="hero-copy" style={{ marginTop: '0.5rem', marginBottom: '1rem' }}>
          Peak-normalised thermal flux φ(r, z) at the current simulation time.
          The distribution evolves as the rod position and precursor fields change.
          Coarser mesh (Δr = Δz = 5 cm) than the steady-state Core page for faster
          LU factorization updates on rod movement.
        </p>

        {isBusy && <div className="flux-skeleton" />}
        {state && (
          <>
            <Heatmap2D data={state} />
            <div className="flux-colorbar-legend">
              <span className="flux-label">Low flux</span>
              <div className="flux-colorbar" aria-hidden="true" />
              <span className="flux-label">Peak flux</span>
            </div>
          </>
        )}
      </section>

      {/* Radial + axial profiles */}
      <section className="core-profiles-grid">
        <article className="panel core-profile-card">
          <div>
            <p className="section-label">Radial profile · φ(r)</p>
            <h2>Flux vs. radius</h2>
            <p className="hero-copy" style={{ fontSize: '0.88rem', marginTop: '0.5rem', marginBottom: '0.85rem' }}>
              Midplane (z = 0) radial cut from the transient solution.
            </p>
          </div>
          {isBusy && <div className="flux-skeleton" style={{ height: 120 }} />}
          {state && <RadialProfile profile={state.radial} geometry={state.geometry} />}
        </article>

        <article className="panel core-profile-card">
          <div>
            <p className="section-label">Axial profile · φ(z)</p>
            <h2>Flux vs. height</h2>
            <p className="hero-copy" style={{ fontSize: '0.88rem', marginTop: '0.5rem', marginBottom: '0.85rem' }}>
              Axial profile at the fuel annulus mid-radius from the transient solution.
            </p>
          </div>
          {isBusy && <div className="flux-skeleton" style={{ height: 120 }} />}
          {state && <AxialProfile profile={state.axial} geometry={state.geometry} />}
        </article>
      </section>

      {/* History trend charts */}
      <section className="core-profiles-grid">
        <article className="panel core-profile-card">
          <div>
            <p className="section-label">History · ρ(t)</p>
            <h2>Reactivity vs. time</h2>
          </div>
          {state ? (
            <TrendChart
              history={state.history}
              field="reactivityPcm"
              label="ρ"
              unit="pcm"
              color="#f59e0b"
              zeroLine
            />
          ) : (
            <div className="flux-skeleton" style={{ height: 130 }} />
          )}
        </article>

        <article className="panel core-profile-card">
          <div>
            <p className="section-label">History · P/P₀(t)</p>
            <h2>Normalised power vs. time</h2>
          </div>
          {state ? (
            <TrendChart
              history={state.history}
              field="powerNorm"
              label="P/P₀"
              unit="—"
              color="#4ade80"
              zeroLine={false}
            />
          ) : (
            <div className="flux-skeleton" style={{ height: 130 }} />
          )}
        </article>
      </section>

      {/* Info panel */}
      <section className="notes-grid">
        <article className="panel note-panel">
          <p className="section-label">Physics model</p>
          <h2>Implicit Euler diffusion</h2>
          <ul className="note-list">
            <li>One-group time-dependent diffusion equation with k_ref scaling to ensure exact criticality at the initial rod insertion on the discrete mesh.</li>
            <li>Six delayed-neutron precursor groups (IAEA standard β values for U-235 in D₂O).</li>
            <li>Fully implicit Euler time integration — unconditionally stable for arbitrary dt.</li>
            <li>LU factorization cached per rod insertion fraction; rebuilt only when the rod moves.</li>
          </ul>
        </article>

        <article className="panel note-panel">
          <p className="section-label">Solver parameters</p>
          <h2>Mesh &amp; timing</h2>
          <dl className="facts-grid" style={{ marginTop: '1rem' }}>
            {state ? (
              <>
                <div><dt>Mesh Δr = Δz</dt><dd>{state.meshDrCm} cm</dd></div>
                <div><dt>Time step dt</dt><dd>{state.dt} s</dd></div>
                <div><dt>Neutron velocity v</dt><dd>2.2 × 10⁵ cm/s</dd></div>
                <div><dt>Precursor groups</dt><dd>6</dd></div>
                <div><dt>β_total</dt><dd>0.00651</dd></div>
                <div><dt>Heatmap resolution</dt><dd>40 × 60 cells</dd></div>
              </>
            ) : (
              <>
                <div><dt>Mesh Δr = Δz</dt><dd>5 cm</dd></div>
                <div><dt>Time step dt</dt><dd>1 s</dd></div>
                <div><dt>Neutron velocity v</dt><dd>2.2 × 10⁵ cm/s</dd></div>
                <div><dt>Precursor groups</dt><dd>6</dd></div>
                <div><dt>β_total</dt><dd>0.00651</dd></div>
                <div><dt>Heatmap resolution</dt><dd>40 × 60 cells</dd></div>
              </>
            )}
          </dl>
        </article>
      </section>
    </div>
  )
}
