import { useCoreFlux } from '../hooks/useCoreFlux'
import type { CoreFluxResponse } from '../simulation/types'

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

const SVG_W = 420
const SVG_H = 300
const PROFILE_W = 480
const PROFILE_H = 120
const PROFILE_PAD = 14

function Heatmap2D({ data }: { data: CoreFluxResponse }) {
  const { heatmapRCm, heatmapZCm, heatmapPhi, geometry } = data
  const nr = heatmapRCm.length
  const nz = heatmapZCm.length
  const cellW = SVG_W / nr
  const cellH = SVG_H / nz

  // Precompute max for normalisation (phi is already normalised to peak=1, but
  // the display subset might not include the peak cell)
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
      // SVG y=0 is top; z increases upward, so flip iz
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
  // Boundary of active zone in SVG coords (flip z)
  const yActiveTop = ((zMax - geometry.hActiveCm / 2) / zSpan) * SVG_H
  const yActiveBot = ((zMax + geometry.hActiveCm / 2) / zSpan) * SVG_H

  return (
    <svg
      viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      className="flux-heatmap-svg"
      role="img"
      aria-label="2D r-z thermal flux distribution computed by the diffusion solver"
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

function RadialProfile({ data }: { data: CoreFluxResponse }) {
  const { axisCm: rs, phiNorm: pts } = data.radial
  const { geometry } = data
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

function AxialProfile({ data }: { data: CoreFluxResponse }) {
  const { axisCm: zs, phiNorm: pts } = data.axial
  const { geometry } = data
  const n = zs.length
  const max = Math.max(...pts)
  const zMin = zs[0]
  const zMax = zs[n - 1]
  const zSpan = zMax - zMin

  const xs = zs.map((z) => PROFILE_PAD + ((z - zMin) / zSpan) * (PROFILE_W - PROFILE_PAD * 2))
  const ys = pts.map((v) => PROFILE_H - PROFILE_PAD - (v / max) * (PROFILE_H - PROFILE_PAD * 2))
  const linePath = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(' ')
  const areaPath = `${linePath} L${xs.at(-1)!.toFixed(1)},${(PROFILE_H - PROFILE_PAD).toFixed(1)} L${xs[0].toFixed(1)},${(PROFILE_H - PROFILE_PAD).toFixed(1)} Z`

  // Active zone boundaries in SVG x-coords
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

function SolveStatusBar({
  state,
  onRefresh,
}: {
  state: ReturnType<typeof useCoreFlux>['state']
  onRefresh: () => void
}) {
  if (state.status === 'loading') {
    return (
      <div className="flux-solve-bar flux-solve-bar--loading">
        <span>⏳ Running diffusion solve…  this may take a few seconds on first load.</span>
      </div>
    )
  }
  if (state.status === 'error') {
    return (
      <div className="flux-solve-bar flux-solve-bar--error">
        <span>⚠ {state.message}</span>
        <span className="flux-solve-bar__spacer" />
        <button className="btn-refresh" onClick={onRefresh}>Retry</button>
      </div>
    )
  }
  if (state.status === 'success') {
    const { metadata } = state.data
    const barClass = metadata.cached ? 'flux-solve-bar--cached' : 'flux-solve-bar--live'
    const label = metadata.cached ? '✓ Cached result' : '✓ Fresh solve'
    return (
      <div className={`flux-solve-bar ${barClass}`}>
        <span>{label}</span>
        <span style={{ opacity: 0.7 }}>
          · rod {metadata.rodInsertionPercent.toFixed(1)} %
          · k_eff {metadata.kEff.toFixed(6)}
          · {metadata.iterations} iter
          · Δr = Δz = {metadata.meshDrCm} cm
        </span>
        <span className="flux-solve-bar__spacer" />
        <button className="btn-refresh" onClick={onRefresh}>Refresh flux</button>
      </div>
    )
  }
  return null
}

export function CorePage() {
  const { state, refresh } = useCoreFlux()
  const data = state.status === 'success' ? state.data : null

  return (
    <div className="app-shell">
      <header className="hero-header panel">
        <div>
          <p className="eyebrow">Spatial flux distributions</p>
          <h1>Core</h1>
          <p className="hero-copy">
            Flux distributions computed by the 2D r-z one-group diffusion solver
            in <code>backend/reactor_backend/diffusion.py</code>. The solver mirrors
            the current rod insertion from the Overview page and caches results so
            repeated requests for the same state do not re-run the eigenvalue solve.
          </p>
        </div>
        {data && (
          <div className="status-badge" style={{ borderColor: 'rgba(74,222,128,0.25)', background: 'rgba(5,46,22,0.2)' }}>
            <span className="status-badge__label">Live solver output</span>
            <span>
              k_eff = {data.metadata.kEff.toFixed(6)}&ensp;·&ensp;
              rod {data.metadata.rodInsertionPercent.toFixed(1)} %&ensp;·&ensp;
              mesh {data.metadata.meshDrCm} cm
            </span>
          </div>
        )}
        {state.status === 'loading' && (
          <div className="status-badge" style={{ borderColor: 'rgba(59,130,246,0.25)', background: 'rgba(30,58,138,0.2)' }}>
            <span className="status-badge__label">Solving…</span>
            <span>Running 2D eigenvalue solve. First solve may take ~10 s.</span>
          </div>
        )}
        {state.status === 'error' && (
          <div className="status-badge" style={{ borderColor: 'rgba(248,113,113,0.25)', background: 'rgba(127,29,29,0.2)' }}>
            <span className="status-badge__label">Backend unreachable</span>
            <span>Start the Python backend to see live flux distributions.</span>
          </div>
        )}
      </header>

      {/* 2D flux heatmap */}
      <section className="panel core-flux-panel">
        <div className="panel-heading">
          <div>
            <p className="section-label">2D r-z flux map</p>
            <h2>Thermal flux distribution</h2>
          </div>
        </div>
        <p className="hero-copy" style={{ marginTop: '0.5rem', marginBottom: '1rem' }}>
          One-group thermal flux φ(r, z) for the current rod insertion state. The
          radial axis runs from the central D₂O channel through the fuel annulus to
          the outer reflector. The axial axis spans the full core height including
          top and bottom reflector slabs.
        </p>

        <SolveStatusBar state={state} onRefresh={refresh} />

        {state.status === 'loading' && <div className="flux-skeleton" />}
        {data && <Heatmap2D data={data} />}

        <div className="flux-colorbar-legend">
          <span className="flux-label">Low flux</span>
          <div className="flux-colorbar" aria-hidden="true" />
          <span className="flux-label">Peak flux</span>
        </div>
      </section>

      {/* Radial + axial profiles */}
      <section className="core-profiles-grid">
        <article className="panel core-profile-card">
          <div>
            <p className="section-label">Radial profile  ·  φ(r)</p>
            <h2>Flux vs. radius</h2>
            <p className="hero-copy" style={{ fontSize: '0.88rem', marginTop: '0.5rem', marginBottom: '0.85rem' }}>
              Midplane (z = 0) radial cut from the 2D solution. Central D₂O channel
              (0 – 0.8 m), fuel annulus (0.8 – 3.5 m), outer D₂O reflector
              (3.5 – 4.1 m).
            </p>
          </div>
          {state.status === 'loading' && <div className="flux-skeleton" style={{ height: 120 }} />}
          {data && <RadialProfile data={data} />}
        </article>

        <article className="panel core-profile-card">
          <div>
            <p className="section-label">Axial profile  ·  φ(z)</p>
            <h2>Flux vs. height</h2>
            <p className="hero-copy" style={{ fontSize: '0.88rem', marginTop: '0.5rem', marginBottom: '0.85rem' }}>
              Axial profile at the radial midpoint of the fuel annulus from the 2D
              solution. Active zone (6.9 m) with top and bottom reflector slabs (0.6 m each).
            </p>
          </div>
          {state.status === 'loading' && <div className="flux-skeleton" style={{ height: 120 }} />}
          {data && <AxialProfile data={data} />}
        </article>
      </section>

      {/* Geometry reference */}
      <section className="notes-grid">
        <article className="panel note-panel">
          <p className="section-label">Estimate 2 geometry</p>
          <h2>Core dimensions</h2>
          <dl className="facts-grid" style={{ marginTop: '1rem' }}>
            {data ? (
              <>
                <div><dt>Inner D₂O radius</dt><dd>{(data.geometry.rInnerCm / 100).toFixed(2)} m</dd></div>
                <div><dt>Fuel outer radius</dt><dd>{(data.geometry.rFuelCm / 100).toFixed(2)} m</dd></div>
                <div><dt>Outer reflector</dt><dd>{(data.geometry.rReflCm / 100).toFixed(2)} m</dd></div>
                <div><dt>Active height</dt><dd>{(data.geometry.hActiveCm / 100).toFixed(2)} m</dd></div>
                <div><dt>Axial reflector</dt><dd>{(data.geometry.hReflCm / 100).toFixed(2)} m per side</dd></div>
                <div><dt>Reference k_eff</dt><dd>{data.metadata.kEff.toFixed(6)}</dd></div>
              </>
            ) : (
              <>
                <div><dt>Inner D₂O radius</dt><dd>0.80 m</dd></div>
                <div><dt>Fuel outer radius</dt><dd>3.46 m</dd></div>
                <div><dt>Outer reflector</dt><dd>4.06 m</dd></div>
                <div><dt>Active height</dt><dd>6.91 m</dd></div>
                <div><dt>Axial reflector</dt><dd>0.60 m per side</dd></div>
                <div><dt>Reference k_eff</dt><dd>1.000 395 (unrodded)</dd></div>
              </>
            )}
          </dl>
        </article>

        <article className="panel note-panel">
          <p className="section-label">Diffusion solver</p>
          <h2>About the backend solver</h2>
          <ul className="note-list">
            <li>
              One-group finite-difference 2D r-z solver implemented in
              {' '}<code>backend/reactor_backend/diffusion.py</code>.
            </li>
            <li>
              Mesh spacing Δr = Δz = 3 cm; eigenvalue solved with a sparse power
              iteration method via SciPy. Results are cached per rod insertion state.
            </li>
            <li>
              Rod worth curve (Δρ vs. insertion) derived from repeated rodded vs.
              unrodded eigenvalue solves at 11 insertion fractions.
            </li>
            <li>
              The Core page mirrors the rod insertion from the running Overview
              simulation. Use the <em>Refresh flux</em> button to re-solve for the
              current state.
            </li>
          </ul>
        </article>
      </section>
    </div>
  )
}
