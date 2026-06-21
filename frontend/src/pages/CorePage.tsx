import { useMultigroupDiffusion } from '../hooks/useMultigroupDiffusion'
import type {
  MultigroupDiffusionProfile,
  MultigroupDiffusionResponse,
} from '../simulation/types'

const HEATMAP_PLOT_WIDTH = 306
const HEATMAP_PLOT_HEIGHT = 413
const PROFILE_WIDTH = 520
const PROFILE_HEIGHT = 78
const PROFILE_PAD = 16

const FLUX_COLORS = ['#000004', '#1b0c41', '#4a0c6b', '#781c6d', '#a52c60', '#cf4446', '#ed6925', '#fb9b06', '#f7d13d', '#fcffa4']
const POWER_COLORS = ['#241000', '#4a2500', '#773900', '#a64b13', '#c7633e', '#dc7f64', '#ee9b82', '#f6b59a']

function hexToRgb(hex: string) {
  const value = Number.parseInt(hex.slice(1), 16)
  return {
    r: (value >> 16) & 255,
    g: (value >> 8) & 255,
    b: value & 255,
  }
}

function fieldColor(value: number, power: boolean) {
  const t = Math.max(0, Math.min(1, value))
  const palette = power ? POWER_COLORS : FLUX_COLORS
  const scaled = t * (palette.length - 1)
  const index = Math.min(Math.floor(scaled), palette.length - 2)
  const local = scaled - index
  const left = hexToRgb(palette[index])
  const right = hexToRgb(palette[index + 1])
  const r = Math.round(left.r + (right.r - left.r) * local)
  const g = Math.round(left.g + (right.g - left.g) * local)
  const b = Math.round(left.b + (right.b - left.b) * local)
  return `rgb(${r} ${g} ${b})`
}

function formatCm(value: number) {
  return Math.abs(value) < 1e-6 ? '0' : value.toFixed(0)
}

function Heatmap({
  data,
  values,
  title,
  power = false,
}: {
  data: MultigroupDiffusionResponse
  values: number[][]
  title: string
  power?: boolean
}) {
  const nx = data.heatmapXCm.length
  const nz = data.heatmapZCm.length
  const xMinimum = data.heatmapXCm[0] ?? -1
  const xMaximum = data.heatmapXCm.at(-1) ?? 1
  const zMinimum = data.heatmapZCm[0] ?? -1
  const zMaximum = data.heatmapZCm.at(-1) ?? 1
  const xSpan = Math.max(xMaximum - xMinimum, 1)
  const zSpan = Math.max(zMaximum - zMinimum, 1)
  const plotHeight = HEATMAP_PLOT_HEIGHT
  const margin = { top: 12, right: 72, bottom: 42, left: 52 }
  const viewWidth = HEATMAP_PLOT_WIDTH + margin.left + margin.right
  const viewHeight = plotHeight + margin.top + margin.bottom
  const cellWidth = HEATMAP_PLOT_WIDTH / nx
  const cellHeight = plotHeight / nz
  const colorbarX = margin.left + HEATMAP_PLOT_WIDTH + 24
  const colorbarHeight = plotHeight
  const colorbarSegments = 36
  const xTicks = [xMinimum, 0, xMaximum]
  const zTicks = [zMinimum, 0, zMaximum]

  return (
    <article className="panel multigroup-map-card">
      <h2>{title}</h2>
      <svg
        viewBox={`0 0 ${viewWidth} ${viewHeight}`}
        className="multigroup-map"
        role="img"
        aria-label={title}
      >
        {values.flatMap((row, axialIndex) =>
          row.map((value, xIndex) => (
            <rect
              key={`${axialIndex}-${xIndex}`}
              x={margin.left + xIndex * cellWidth}
              y={margin.top + (nz - 1 - axialIndex) * cellHeight}
              width={cellWidth + 0.8}
              height={cellHeight + 0.8}
              fill={fieldColor(value, power)}
            />
          )),
        )}
        <line
          x1={margin.left}
          y1={margin.top + plotHeight}
          x2={margin.left + HEATMAP_PLOT_WIDTH}
          y2={margin.top + plotHeight}
          className="axis-line"
        />
        <line
          x1={margin.left}
          y1={margin.top}
          x2={margin.left}
          y2={margin.top + plotHeight}
          className="axis-line"
        />
        {xTicks.map((tick) => {
          const x = margin.left + ((tick - xMinimum) / xSpan) * HEATMAP_PLOT_WIDTH
          return (
            <g key={`x-${tick}`}>
              <line x1={x} y1={margin.top + plotHeight} x2={x} y2={margin.top + plotHeight + 5} className="axis-line" />
              <text x={x} y={margin.top + plotHeight + 19} textAnchor="middle" className="axis-label">
                {formatCm(tick)}
              </text>
            </g>
          )
        })}
        {zTicks.map((tick) => {
          const y = margin.top + plotHeight - ((tick - zMinimum) / zSpan) * plotHeight
          return (
            <g key={`z-${tick}`}>
              <line x1={margin.left - 5} y1={y} x2={margin.left} y2={y} className="axis-line" />
              <text x={margin.left - 10} y={y + 4} textAnchor="end" className="axis-label">
                {formatCm(tick)}
              </text>
            </g>
          )
        })}
        <text
          x={margin.left + HEATMAP_PLOT_WIDTH / 2}
          y={viewHeight - 7}
          textAnchor="middle"
          className="axis-title"
        >
          x [cm]
        </text>
        <text
          x={14}
          y={margin.top + plotHeight / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${margin.top + plotHeight / 2})`}
          className="axis-title"
        >
          z [cm]
        </text>
        {Array.from({ length: colorbarSegments }, (_, index) => {
          const t = index / (colorbarSegments - 1)
          return (
            <rect
              key={`scale-${index}`}
              x={colorbarX}
              y={margin.top + (1 - t) * colorbarHeight}
              width="12"
              height={colorbarHeight / colorbarSegments + 1}
              fill={fieldColor(t, power)}
            />
          )
        })}
        {[0, 0.5, 1].map((tick) => (
          <g key={`c-${tick}`}>
            <line
              x1={colorbarX + 12}
              y1={margin.top + (1 - tick) * colorbarHeight}
              x2={colorbarX + 17}
              y2={margin.top + (1 - tick) * colorbarHeight}
              className="axis-line"
            />
            <text
              x={colorbarX + 21}
              y={margin.top + (1 - tick) * colorbarHeight + 4}
              className="axis-label"
            >
              {tick.toFixed(tick === 0.5 ? 1 : 0)}
            </text>
          </g>
        ))}
        <text
          x={colorbarX + 48}
          y={margin.top + colorbarHeight / 2}
          textAnchor="middle"
          transform={`rotate(-90 ${colorbarX + 48} ${margin.top + colorbarHeight / 2})`}
          className="axis-title"
        >
          clean peak relative
        </text>
      </svg>
    </article>
  )
}

function ProfileChart({
  profile,
  title,
  color,
}: {
  profile: MultigroupDiffusionProfile
  title: string
  color: string
}) {
  const minimum = profile.axisCm[0]
  const maximum = profile.axisCm.at(-1) ?? minimum + 1
  const span = maximum - minimum || 1
  const points = profile.axisCm.map((axis, index) => {
    const x = PROFILE_PAD + ((axis - minimum) / span) * (PROFILE_WIDTH - 2 * PROFILE_PAD)
    const y =
      PROFILE_HEIGHT -
      PROFILE_PAD -
      profile.values[index] * (PROFILE_HEIGHT - 2 * PROFILE_PAD)
    return `${x.toFixed(2)},${y.toFixed(2)}`
  })

  return (
    <article className="panel multigroup-profile-card">
      <h3>{title}</h3>
      <svg viewBox={`0 0 ${PROFILE_WIDTH} ${PROFILE_HEIGHT}`} className="profile-svg">
        <polyline
          points={points.join(' ')}
          fill="none"
          stroke={color}
          strokeWidth="3"
          strokeLinejoin="round"
        />
        <line
          x1={PROFILE_PAD}
          y1={PROFILE_HEIGHT - PROFILE_PAD}
          x2={PROFILE_WIDTH - PROFILE_PAD}
          y2={PROFILE_HEIGHT - PROFILE_PAD}
          stroke="rgba(148,163,184,0.3)"
        />
      </svg>
    </article>
  )
}

export function CorePage() {
  const { state, recompute } = useMultigroupDiffusion()
  const data = state.status === 'success' ? state.data : null

  return (
    <div className="app-shell">
      <section className="panel core-summary-panel">
        <div className="core-result">
          {data ? (
            <>
              <span className="core-result__label">Result</span>
              <strong>diffusion {data.metadata.reactivityPcm.toFixed(1)} pcm</strong>
              <span>OpenMC CE / PK {data.metadata.openmcCeReactivityPcm.toFixed(1)} pcm</span>
              <span>rod {data.metadata.rodInsertionPercent.toFixed(1)}%</span>
            </>
          ) : (
            <strong>Core</strong>
          )}
          <button
            className="button"
            type="button"
            onClick={() => void recompute()}
            disabled={state.status === 'loading'}
          >
            {state.status === 'loading' ? 'Solving...' : 'Recompute current rod position'}
          </button>
        </div>
      </section>

      {state.status === 'error' && (
        <section className="panel loading-panel">
          <h2>Core result unavailable</h2>
          <p className="alert-banner">{state.message}</p>
        </section>
      )}

      {state.status === 'loading' && (
        <section className="panel loading-panel">
          <h2>Running core solve</h2>
          <p className="loading-copy">Loading or recomputing the current rod-position result.</p>
        </section>
      )}

      {data && (
        <>
          <section className="core-visual-grid">
            <Heatmap data={data} values={data.heatmapFlux} title="Total scalar flux" />
            <Heatmap data={data} values={data.heatmapPower} title="Fission power density" power />

            <section className="profile-stack" aria-label="Core line profiles">
              <ProfileChart profile={data.axialFlux} title="Axial flux" color="#7dd3fc" />
              <ProfileChart profile={data.axialPower} title="Axial power" color="#fb7185" />
              <ProfileChart profile={data.radialFlux} title="Radial flux" color="#38bdf8" />
              <ProfileChart profile={data.radialPower} title="Radial power shape" color="#f59e0b" />
            </section>
          </section>

          <section className="panel multigroup-details">
            <div>
              <span>Energy groups</span>
              <strong>{data.metadata.groupCount}</strong>
            </div>
            <div>
              <span>Resolved regions</span>
              <strong>{data.geometry.resolvedRegionCount}</strong>
            </div>
            <div>
              <span>Equivalent rod Delta Sigma_a</span>
              <strong>{data.metadata.rodDeltaAbsorptionCmInv.toFixed(3)} cm^-1</strong>
            </div>
          </section>
        </>
      )}
    </div>
  )
}
