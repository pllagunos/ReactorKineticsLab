import { useMultigroupDiffusion } from '../hooks/useMultigroupDiffusion'
import type {
  MultigroupDiffusionProfile,
  MultigroupDiffusionResponse,
} from '../simulation/types'

const HEATMAP_WIDTH = 440
const HEATMAP_HEIGHT = 310
const PROFILE_WIDTH = 520
const PROFILE_HEIGHT = 130
const PROFILE_PAD = 16

function fieldColor(value: number, power: boolean) {
  const t = Math.max(0, Math.min(1, value))
  const hue = power ? 36 - 24 * t : 205 - 35 * t
  const lightness = 15 + 58 * t
  return `hsl(${hue} 82% ${lightness}%)`
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
  const nr = data.heatmapRCm.length
  const nz = data.heatmapZCm.length
  const cellWidth = HEATMAP_WIDTH / nr
  const cellHeight = HEATMAP_HEIGHT / nz

  return (
    <article className="panel multigroup-map-card">
      <p className="section-label">Resolved r-z field</p>
      <h2>{title}</h2>
      <svg
        viewBox={`0 0 ${HEATMAP_WIDTH} ${HEATMAP_HEIGHT}`}
        className="multigroup-map"
        role="img"
        aria-label={title}
      >
        {values.flatMap((row, radialIndex) =>
          row.map((value, axialIndex) => (
            <rect
              key={`${radialIndex}-${axialIndex}`}
              x={radialIndex * cellWidth}
              y={(nz - 1 - axialIndex) * cellHeight}
              width={cellWidth + 0.4}
              height={cellHeight + 0.4}
              fill={fieldColor(value, power)}
            />
          )),
        )}
      </svg>
      <div className={`multigroup-scale${power ? ' multigroup-scale--power' : ''}`}>
        <span>Low</span>
        <span className="multigroup-scale__bar" />
        <span>Peak</span>
      </div>
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

export function MultigroupDiffusionPage() {
  const { state, recompute } = useMultigroupDiffusion()
  const data = state.status === 'success' ? state.data : null

  return (
    <div className="app-shell">
      <header className="panel hero-header">
        <div>
          <p className="eyebrow">OpenMC-informed clean-core calculation</p>
          <h1>Four-group diffusion</h1>
          <p className="hero-copy">
            Resolved OpenMC cells, P0 scattering, transport-derived diffusion coefficients,
            and optional CE-referenced SPH factors. Rod position is intentionally excluded.
          </p>
        </div>
        {data && (
          <div className={`status-badge ${data.metadata.qualified ? 'status-badge--near-critical' : 'status-badge--supercritical'}`}>
            <span className="status-badge__label">
              {data.metadata.qualified ? 'Qualified SPH result' : 'Provisional physics result'}
            </span>
            <span>
              k_eff {data.metadata.kEff.toFixed(6)} | OpenMC difference{' '}
              {data.metadata.differencePcm.toFixed(1)} pcm
            </span>
            <span>
              {data.metadata.cellCount.toLocaleString()} cells |{' '}
              {data.metadata.timingsSeconds.total?.toFixed(2) ?? '-'} s
            </span>
          </div>
        )}
      </header>

      <section className="panel multigroup-control-bar">
        <div>
          <p className="section-label">Explicit solve control</p>
          <strong>Clean core only</strong>
          {data && (
            <span className="multigroup-control-bar__detail">
              {data.metadata.sphApplied ? 'SPH factors applied' : 'Uncorrected diffusion'}
              {' | '}
              {data.metadata.cached ? 'persistent cache' : 'fresh recomputation'}
            </span>
          )}
        </div>
        <button
          className="button"
          type="button"
          onClick={() => void recompute()}
          disabled={state.status === 'loading'}
        >
          {state.status === 'loading' ? 'Solving...' : 'Recompute'}
        </button>
      </section>

      {state.status === 'error' && (
        <section className="panel loading-panel">
          <h2>Multigroup result unavailable</h2>
          <p className="alert-banner">{state.message}</p>
        </section>
      )}

      {state.status === 'loading' && (
        <section className="panel loading-panel">
          <h2>Running four-group solve</h2>
          <p className="loading-copy">Loading the persistent result or recomputing the clean core.</p>
        </section>
      )}

      {data && (
        <>
          <section className="multigroup-map-grid">
            <Heatmap data={data} values={data.heatmapFlux} title="Total scalar flux" />
            <Heatmap data={data} values={data.heatmapPower} title="Fission power density" power />
          </section>

          <section className="multigroup-profile-grid">
            <ProfileChart profile={data.radialFlux} title="Radial flux" color="#38bdf8" />
            <ProfileChart profile={data.axialFlux} title="Axial flux" color="#7dd3fc" />
            <ProfileChart profile={data.radialPower} title="Radial power" color="#f59e0b" />
            <ProfileChart profile={data.axialPower} title="Axial power" color="#fb7185" />
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
              <span>OpenMC uncertainty</span>
              <strong>{data.metadata.openmcReferenceStdDevPcm.toFixed(1)} pcm</strong>
            </div>
            <div>
              <span>Outer radius</span>
              <strong>{data.geometry.reflectorRadiusCm.toFixed(1)} cm</strong>
            </div>
          </section>
        </>
      )}
    </div>
  )
}
