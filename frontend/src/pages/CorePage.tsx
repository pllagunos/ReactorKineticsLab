// Geometry constants from the 2D r-z diffusion model (Estimate 2)
const R_INNER_CM = 80.0
const R_FUEL_CM = 345.6
const R_TOTAL_CM = 405.6
const H_ACTIVE_CM = 691.2
const H_REFL_CM = 60.0

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

/** Map a normalised flux value [0,1] to an HSL/RGB colour string. */
function fluxColor(t: number): string {
  // Stops: deep navy → dark cyan → bright cyan → amber → warm cream
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

/** Approximate radial flux shape for the annular reflected core. */
function phiRadial(r_cm: number): number {
  if (r_cm <= R_INNER_CM) {
    return 0.35 + 0.25 * (r_cm / R_INNER_CM)
  } else if (r_cm <= R_FUEL_CM) {
    const rPeak = (R_INNER_CM + R_FUEL_CM) / 2
    const sigma = (R_FUEL_CM - R_INNER_CM) * 0.26
    return 0.5 + 0.5 * Math.exp(-0.5 * ((r_cm - rPeak) / sigma) ** 2)
  } else {
    const excess = r_cm - R_FUEL_CM
    return 0.5 * Math.exp(-3 * (excess / (R_TOTAL_CM - R_FUEL_CM)))
  }
}

/** Approximate axial flux shape (cosine in active zone, exponential in reflector). */
function phiAxial(z_cm: number): number {
  const zAbs = Math.abs(z_cm)
  const hHalf = H_ACTIVE_CM / 2
  if (zAbs <= hHalf) {
    return Math.cos((Math.PI / 2) * (zAbs / hHalf))
  }
  const excess = zAbs - hHalf
  return Math.exp(-2.5 * (excess / H_REFL_CM))
}

const NR = 42
const NZ = 28
const SVG_W = 420
const SVG_H = 300
const CELL_W = SVG_W / NR
const CELL_H = SVG_H / NZ

function Heatmap2D() {
  const hHalf = H_ACTIVE_CM / 2 + H_REFL_CM

  const rawValues: number[][] = Array.from({ length: NR }, (_, ir) => {
    const r = (ir / NR) * R_TOTAL_CM
    return Array.from({ length: NZ }, (_, iz) => {
      const z = ((iz / NZ) * 2 - 1) * hHalf
      return phiRadial(r) * phiAxial(z)
    })
  })

  const maxFlux = Math.max(...rawValues.flat())

  const cells: React.ReactNode[] = []
  for (let ir = 0; ir < NR; ir++) {
    for (let iz = 0; iz < NZ; iz++) {
      const t = rawValues[ir][iz] / (maxFlux || 1)
      cells.push(
        <rect
          key={`${ir}-${iz}`}
          x={ir * CELL_W}
          y={iz * CELL_H}
          width={CELL_W + 0.5}
          height={CELL_H + 0.5}
          fill={fluxColor(t)}
        />,
      )
    }
  }

  // Region boundary lines (r direction)
  const xInner = (R_INNER_CM / R_TOTAL_CM) * SVG_W
  const xFuel = (R_FUEL_CM / R_TOTAL_CM) * SVG_W
  // Active zone axial boundaries
  const yActiveTop = ((1 - (H_ACTIVE_CM / 2 + H_REFL_CM) / hHalf) / 2) * SVG_H
  const yActiveBot = SVG_H - yActiveTop

  return (
    <svg
      viewBox={`0 0 ${SVG_W} ${SVG_H}`}
      className="flux-heatmap-svg"
      role="img"
      aria-label="Mock 2D r-z thermal flux distribution for the annular core"
    >
      {cells}
      {/* Region boundaries */}
      <line
        x1={xInner} y1={0} x2={xInner} y2={SVG_H}
        stroke="rgba(148,163,184,0.45)" strokeWidth="1.5" strokeDasharray="4 3"
      />
      <line
        x1={xFuel} y1={0} x2={xFuel} y2={SVG_H}
        stroke="rgba(148,163,184,0.45)" strokeWidth="1.5" strokeDasharray="4 3"
      />
      {/* Active zone axial boundaries */}
      <line
        x1={0} y1={yActiveTop} x2={SVG_W} y2={yActiveTop}
        stroke="rgba(148,163,184,0.3)" strokeWidth="1" strokeDasharray="3 4"
      />
      <line
        x1={0} y1={yActiveBot} x2={SVG_W} y2={yActiveBot}
        stroke="rgba(148,163,184,0.3)" strokeWidth="1" strokeDasharray="3 4"
      />
      {/* Region labels */}
      <text x={xInner / 2} y={SVG_H - 6} textAnchor="middle" className="flux-label">
        D₂O
      </text>
      <text x={(xInner + xFuel) / 2} y={SVG_H - 6} textAnchor="middle" className="flux-label">
        Fuel annulus
      </text>
      <text x={(xFuel + SVG_W) / 2} y={SVG_H - 6} textAnchor="middle" className="flux-label">
        Reflector
      </text>
      {/* Axis labels */}
      <text x={SVG_W - 4} y={SVG_H / 2} textAnchor="end" dominantBaseline="middle" className="flux-label">
        r →
      </text>
      <text x={6} y={8} textAnchor="start" className="flux-label">
        z ↑
      </text>
    </svg>
  )
}

const NR_PROFILE = 60
const PROFILE_W = 480
const PROFILE_H = 120
const PROFILE_PAD = 14

function RadialProfile() {
  const pts = Array.from({ length: NR_PROFILE }, (_, i) => {
    const r = (i / (NR_PROFILE - 1)) * R_TOTAL_CM
    return phiRadial(r)
  })
  const max = Math.max(...pts)
  const xs = pts.map((_, i) =>
    PROFILE_PAD + (i / (NR_PROFILE - 1)) * (PROFILE_W - PROFILE_PAD * 2),
  )
  const ys = pts.map(
    (v) => PROFILE_H - PROFILE_PAD - ((v / max) * (PROFILE_H - PROFILE_PAD * 2)),
  )
  const linePath = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(' ')
  const areaPath = `${linePath} L${xs.at(-1)!.toFixed(1)},${(PROFILE_H - PROFILE_PAD).toFixed(1)} L${xs[0].toFixed(1)},${(PROFILE_H - PROFILE_PAD).toFixed(1)} Z`
  const xInner = PROFILE_PAD + (R_INNER_CM / R_TOTAL_CM) * (PROFILE_W - PROFILE_PAD * 2)
  const xFuel = PROFILE_PAD + (R_FUEL_CM / R_TOTAL_CM) * (PROFILE_W - PROFILE_PAD * 2)
  return (
    <svg viewBox={`0 0 ${PROFILE_W} ${PROFILE_H}`} className="profile-svg" role="img" aria-label="Mock radial flux profile">
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

const NZ_PROFILE = 50

function AxialProfile() {
  const hHalf = H_ACTIVE_CM / 2 + H_REFL_CM
  const pts = Array.from({ length: NZ_PROFILE }, (_, i) => {
    const z = ((i / (NZ_PROFILE - 1)) * 2 - 1) * hHalf
    return phiAxial(z)
  })
  const max = Math.max(...pts)
  const xs = pts.map((_, i) =>
    PROFILE_PAD + (i / (NZ_PROFILE - 1)) * (PROFILE_W - PROFILE_PAD * 2),
  )
  const ys = pts.map(
    (v) => PROFILE_H - PROFILE_PAD - ((v / max) * (PROFILE_H - PROFILE_PAD * 2)),
  )
  const linePath = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(' ')
  const areaPath = `${linePath} L${xs.at(-1)!.toFixed(1)},${(PROFILE_H - PROFILE_PAD).toFixed(1)} L${xs[0].toFixed(1)},${(PROFILE_H - PROFILE_PAD).toFixed(1)} Z`
  const hReflFrac = H_REFL_CM / hHalf
  const xReflL = PROFILE_PAD + (hReflFrac / 2) * (PROFILE_W - PROFILE_PAD * 2)
  const xReflR = PROFILE_W - PROFILE_PAD - (hReflFrac / 2) * (PROFILE_W - PROFILE_PAD * 2)
  return (
    <svg viewBox={`0 0 ${PROFILE_W} ${PROFILE_H}`} className="profile-svg" role="img" aria-label="Mock axial flux profile">
      <path d={areaPath} fill="#f59e0b" fillOpacity="0.14" />
      <path d={linePath} fill="none" stroke="#f59e0b" strokeWidth="2.5" strokeLinejoin="round" />
      <line x1={xReflL} y1={PROFILE_PAD} x2={xReflL} y2={PROFILE_H - PROFILE_PAD} stroke="rgba(148,163,184,0.4)" strokeWidth="1" strokeDasharray="3 3" />
      <line x1={xReflR} y1={PROFILE_PAD} x2={xReflR} y2={PROFILE_H - PROFILE_PAD} stroke="rgba(148,163,184,0.4)" strokeWidth="1" strokeDasharray="3 3" />
      <line x1={PROFILE_PAD} y1={PROFILE_H - PROFILE_PAD} x2={PROFILE_W - PROFILE_PAD} y2={PROFILE_H - PROFILE_PAD} stroke="rgba(148,163,184,0.25)" />
      <line x1={PROFILE_PAD} y1={PROFILE_PAD} x2={PROFILE_PAD} y2={PROFILE_H - PROFILE_PAD} stroke="rgba(148,163,184,0.25)" />
      <text x={(xReflL) / 2 + PROFILE_PAD / 2} y={PROFILE_H - 3} textAnchor="middle" className="flux-label" fontSize="8">Refl.</text>
      <text x={(xReflL + xReflR) / 2} y={PROFILE_H - 3} textAnchor="middle" className="flux-label" fontSize="8">Active zone</text>
      <text x={(xReflR + PROFILE_W - PROFILE_PAD) / 2} y={PROFILE_H - 3} textAnchor="middle" className="flux-label" fontSize="8">Refl.</text>
    </svg>
  )
}

export function CorePage() {
  return (
    <div className="app-shell">
      <header className="hero-header panel">
        <div>
          <p className="eyebrow">Spatial flux distributions</p>
          <h1>Core</h1>
          <p className="hero-copy">
            Flux distributions computed by the 2D r-z one-group diffusion solver
            in <code>backend/reactor_backend/diffusion.py</code>. The radial, axial,
            and 2D maps shown below are analytic mock profiles based on the Estimate 2
            geometry — real solver output is not yet wired to this page.
          </p>
        </div>
        <div className="status-badge" style={{ borderColor: 'rgba(245,158,11,0.3)', background: 'rgba(120,53,15,0.2)' }}>
          <span className="status-badge__label">Placeholder visuals</span>
          <span>
            These distributions are approximated from the clean-core geometry.
            Backend diffusion integration is planned for a future version.
          </span>
        </div>
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
          Approximate one-group thermal flux φ(r, z) for the unrodded clean-core
          operating state. The radial axis runs from the central D₂O channel through the
          fuel annulus to the outer reflector. The axial axis spans the full core height
          including top and bottom reflector slabs.
        </p>
        <Heatmap2D />
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
              Central D₂O channel (0 – 0.8 m), fuel annulus (0.8 – 3.5 m), outer D₂O
              reflector (3.5 – 4.1 m). Flux peaks in the fuel region and decays
              exponentially into the reflector.
            </p>
          </div>
          <RadialProfile />
        </article>

        <article className="panel core-profile-card">
          <div>
            <p className="section-label">Axial profile  ·  φ(z)</p>
            <h2>Flux vs. height</h2>
            <p className="hero-copy" style={{ fontSize: '0.88rem', marginTop: '0.5rem', marginBottom: '0.85rem' }}>
              Cosine-shaped profile across the 6.9 m active zone with exponential
              decay into the 0.6 m top/bottom reflector slabs. Peak flux occurs at
              the axial midplane.
            </p>
          </div>
          <AxialProfile />
        </article>
      </section>

      {/* Geometry reference */}
      <section className="notes-grid">
        <article className="panel note-panel">
          <p className="section-label">Estimate 2 geometry</p>
          <h2>Core dimensions</h2>
          <dl className="facts-grid" style={{ marginTop: '1rem' }}>
            <div><dt>Inner D₂O radius</dt><dd>0.80 m</dd></div>
            <div><dt>Fuel outer radius</dt><dd>3.46 m</dd></div>
            <div><dt>Outer reflector</dt><dd>4.06 m</dd></div>
            <div><dt>Active height</dt><dd>6.91 m</dd></div>
            <div><dt>Axial reflector</dt><dd>0.60 m per side</dd></div>
            <div><dt>Reference k_eff</dt><dd>1.000 395 (unrodded)</dd></div>
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
              iteration method via SciPy.
            </li>
            <li>
              Rod worth curve (Δρ vs. insertion) derived from repeated rodded vs.
              unrodded eigenvalue solves at 11 insertion fractions.
            </li>
            <li>
              This page will be wired to live diffusion solver output in a future
              version.
            </li>
          </ul>
        </article>
      </section>
    </div>
  )
}
