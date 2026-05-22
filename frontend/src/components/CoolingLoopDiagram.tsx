import type { ThermalSnapshot } from '../simulation/types'
import {
  formatMassFlow,
  formatPressureDropPa,
  formatTemperatureK,
} from '../utils/format'

type Props = {
  thermal: ThermalSnapshot | null
}

/** Map a normalised value t∈[0,1] through the app's flux colormap (dark-navy → blue → cyan → amber → warm-white). */
function heatmapColor(t: number): string {
  const stops: [number, number, number][] = [
    [2, 10, 35],      // 0.0  – dark navy
    [12, 74, 110],    // 0.25 – deep blue
    [8, 145, 178],    // 0.5  – cyan
    [245, 158, 11],   // 0.75 – amber
    [254, 243, 199],  // 1.0  – warm white
  ]
  const seg = Math.min(Math.floor(t * 4), 3)
  const u = t * 4 - seg
  const [r1, g1, b1] = stops[seg]
  const [r2, g2, b2] = stops[seg + 1]
  return `rgb(${Math.round(r1 + u * (r2 - r1))},${Math.round(g1 + u * (g2 - g1))},${Math.round(b1 + u * (b2 - b1))})`
}

function ValueBadge({
  cx,
  cy,
  label,
  value,
  accentColor,
}: {
  cx: number
  cy: number
  label: string
  value: string
  accentColor: string
}) {
  const w = 108
  const h = 46
  return (
    <g>
      <rect
        x={cx - w / 2}
        y={cy - h / 2}
        width={w}
        height={h}
        rx={7}
        fill="rgba(2,6,23,0.88)"
        stroke={accentColor}
        strokeWidth={1.5}
      />
      <text
        x={cx}
        y={cy - 6}
        textAnchor="middle"
        fill="rgba(148,163,184,0.8)"
        fontSize={9}
        letterSpacing="0.12em"
        fontFamily="inherit"
      >
        {label}
      </text>
      <text
        x={cx}
        y={cy + 12}
        textAnchor="middle"
        fill="#e2e8f0"
        fontSize={14}
        fontWeight="600"
        fontFamily="inherit"
      >
        {value}
      </text>
    </g>
  )
}

export function CoolingLoopDiagram({ thermal }: Props) {
  const tIn = formatTemperatureK(thermal?.inletTemperatureK ?? null)
  const tOut = formatTemperatureK(thermal?.outletTemperatureK ?? null)
  const mDot = formatMassFlow(thermal?.massFlowKgPerSecond ?? null)
  const dP = formatPressureDropPa(thermal?.corePressureDropPa ?? null)

  // Tube bundle colour ramp: amber (hot, left) → cyan (cold, right)
  const tubeBundleRows = [0, 1, 2, 3, 4, 5, 6] as const
  function tubeColor(i: number) {
    const r = Math.round(249 - i * 28)
    const g = Math.round(115 + i * 11)
    const b = Math.round(22 + i * 33)
    return `rgba(${r},${g},${b},0.55)`
  }

  return (
    <article className="panel th-diagram-panel">
      <p className="section-label">Primary cooling loop — HMI view</p>

      {/* ─────────────────────────────────────────────────────────────────────
          viewBox: 0 0 920 440
          Reactor vessel:  x=40  y=26  w=228 h=388 (right edge 268)
          Upper plenum:    x=92  y=68  w=128 h=50  (inlet, cold, center 156,93)
          Core fuel:       x=116 y=118 w=80  h=164
          Core channel:    x=138 y=128 w=36  h=144
          Lower plenum:    x=92  y=282 w=128 h=50  (outlet, hot,  center 156,307)
          HX shell:        x=680 y=178 w=140 h=118 (left 680, right 820, cy 237)
          Pump:            cx=530 cy=416
          Cold leg:  M 220 93  H 870 V 237 H 820   (top route, cyan)
          Hot  leg:  M 220 307 H 380 V 416 H 680 V 237  (bottom+ascent, amber)
      ──────────────────────────────────────────────────────────────────────── */}

      <svg
        className="th-diagram-svg"
        viewBox="0 0 920 440"
        role="img"
        aria-label="HMI schematic of the primary D₂O cooling loop"
      >
        <defs>
          {/* Clip path to preserve rounded corners when drawing 8-node power bands */}
          <clipPath id="thFuelNodeClip">
            <rect x={116} y={118} width={80} height={164} rx={8} />
          </clipPath>

          {/* Fuel region gradient kept for fallback – still used as base fill border */}
          <linearGradient id="thFuelGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#fbbf24" />
            <stop offset="100%" stopColor="#f87171" />
          </linearGradient>

          {/* Inlet plenum fill (cool) */}
          <linearGradient id="thInletGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="rgba(12,74,110,0.62)" />
            <stop offset="100%" stopColor="rgba(12,74,110,0.42)" />
          </linearGradient>

          {/* Outlet plenum fill (warm) */}
          <linearGradient id="thOutletGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="rgba(124,45,18,0.55)" />
            <stop offset="100%" stopColor="rgba(124,45,18,0.38)" />
          </linearGradient>

          {/* Pipe glow filter */}
          <filter id="thGlow" x="-20%" y="-60%" width="140%" height="220%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* ══════════════════════════════════════════════════════════════════
            PIPE RUNS  (drawn first so vessel + components render on top)
        ══════════════════════════════════════════════════════════════════ */}

        {/* Cold leg glow layer */}
        <polyline
          points="220,93 870,93 870,237 820,237"
          fill="none"
          stroke="#38bdf8"
          strokeWidth={28}
          strokeLinecap="square"
          strokeLinejoin="miter"
          opacity={0.07}
          filter="url(#thGlow)"
        />
        {/* Cold leg main */}
        <polyline
          points="220,93 870,93 870,237 820,237"
          fill="none"
          stroke="#38bdf8"
          strokeWidth={16}
          strokeLinecap="square"
          strokeLinejoin="miter"
          opacity={0.82}
        />

        {/* Hot leg glow layer */}
        <polyline
          points="220,307 380,307 380,416 680,416 680,237"
          fill="none"
          stroke="#f97316"
          strokeWidth={28}
          strokeLinecap="square"
          strokeLinejoin="miter"
          opacity={0.07}
          filter="url(#thGlow)"
        />
        {/* Hot leg main */}
        <polyline
          points="220,307 380,307 380,416 680,416 680,237"
          fill="none"
          stroke="#f97316"
          strokeWidth={16}
          strokeLinecap="square"
          strokeLinejoin="miter"
          opacity={0.82}
        />

        {/* ══════════════════════════════════════════════════════════════════
            FLOW DIRECTION ARROWS
        ══════════════════════════════════════════════════════════════════ */}

        {/* Cold leg: left along top (HX→reactor) */}
        <polygon points="559,87 543,93 559,99" fill="#38bdf8" opacity={0.9} />
        {/* Cold leg: up the right side */}
        <polygon points="876,176 870,160 864,176" fill="#38bdf8" opacity={0.9} />
        {/* Cold leg: right from HX outlet to corner */}
        <polygon points="830,231 846,237 830,243" fill="#38bdf8" opacity={0.9} />

        {/* Hot leg: right from lower plenum */}
        <polygon points="286,301 302,307 286,313" fill="#f97316" opacity={0.9} />
        {/* Hot leg: right along bottom (after pump) */}
        <polygon points="612,410 628,416 612,422" fill="#f97316" opacity={0.9} />
        {/* Hot leg: up into HX */}
        <polygon points="686,320 680,304 674,320" fill="#f97316" opacity={0.9} />

        {/* ══════════════════════════════════════════════════════════════════
            REACTOR VESSEL
        ══════════════════════════════════════════════════════════════════ */}

        {/* Pool boundary */}
        <rect
          x={40} y={26} width={228} height={388} rx={14}
          fill="rgba(8,47,73,0.2)"
          stroke="rgba(56,189,248,0.2)"
          strokeWidth={2}
        />
        {/* Subtle pool water fill gradient */}
        <rect
          x={42} y={28} width={224} height={384} rx={13}
          fill="rgba(12,74,110,0.07)"
        />

        {/* Upper plenum (inlet, cold) */}
        <rect
          x={92} y={68} width={128} height={50} rx={8}
          fill="url(#thInletGrad)"
          stroke="rgba(56,189,248,0.45)"
          strokeWidth={1.5}
        />
        <text
          x={156} y={62}
          textAnchor="middle"
          fill="rgba(148,163,184,0.6)"
          fontSize={9}
          letterSpacing="0.12em"
          fontFamily="inherit"
        >
          INLET PLENUM
        </text>

        {/* Core fuel annulus – 8-node axial power distribution heatmap */}
        {(() => {
          const fracs = thermal?.axialPowerFractions ?? Array(8).fill(0.125) as number[]
          const minF = Math.min(...fracs)
          const maxF = Math.max(...fracs)
          const span = maxF - minF > 1e-5 ? maxF - minF : 1
          const nodeH = 164 / 8
          return (
            <>
              {fracs.map((frac, i) => (
                <rect
                  key={i}
                  x={116}
                  y={118 + i * nodeH}
                  width={80}
                  height={nodeH + 0.5}
                  fill={heatmapColor((frac - minF) / span)}
                  clipPath="url(#thFuelNodeClip)"
                />
              ))}
              {/* Fuel border ring */}
              <rect
                x={116} y={118} width={80} height={164} rx={8}
                fill="none"
                stroke="rgba(245,158,11,0.35)"
                strokeWidth={1.5}
              />
            </>
          )
        })()}

        {/* Core coolant channel (hollow, where D₂O flows) */}
        <rect
          x={138} y={128} width={36} height={144} rx={5}
          fill="rgba(8,47,73,0.82)"
          stroke="rgba(56,189,248,0.22)"
          strokeWidth={1}
        />
        {/* Subtle flow striations in channel */}
        {[0, 1, 2].map(i => (
          <line
            key={i}
            x1={143} x2={169}
            y1={152 + i * 44} y2={152 + i * 44}
            stroke="rgba(56,189,248,0.15)"
            strokeWidth={1}
            strokeDasharray="4 4"
          />
        ))}

        {/* Lower plenum (outlet, hot) */}
        <rect
          x={92} y={282} width={128} height={50} rx={8}
          fill="url(#thOutletGrad)"
          stroke="rgba(249,115,22,0.45)"
          strokeWidth={1.5}
        />
        <text
          x={156} y={348}
          textAnchor="middle"
          fill="rgba(148,163,184,0.6)"
          fontSize={9}
          letterSpacing="0.12em"
          fontFamily="inherit"
        >
          OUTLET PLENUM
        </text>

        {/* Reactor label */}
        <text
          x={154} y={424}
          textAnchor="middle"
          fill="rgba(148,163,184,0.4)"
          fontSize={9}
          letterSpacing="0.12em"
          fontFamily="inherit"
        >
          RESEARCH REACTOR
        </text>

        {/* ── ΔP differential reference lines ── */}
        {/* Connects the ΔP badge (at ~310,200) to the two plenum pipe exits */}
        <line
          x1={256} y1={200} x2={220} y2={93}
          stroke="rgba(167,139,250,0.28)"
          strokeWidth={1}
          strokeDasharray="4 3"
        />
        <line
          x1={256} y1={200} x2={220} y2={307}
          stroke="rgba(167,139,250,0.28)"
          strokeWidth={1}
          strokeDasharray="4 3"
        />

        {/* ══════════════════════════════════════════════════════════════════
            HEAT EXCHANGER
        ══════════════════════════════════════════════════════════════════ */}

        {/* HX outer shell */}
        <rect
          x={680} y={178} width={140} height={118} rx={10}
          fill="rgba(15,28,63,0.92)"
          stroke="rgba(148,163,184,0.22)"
          strokeWidth={1.5}
        />

        {/* HX tube bundle: hot (amber) left → cold (cyan) right */}
        {tubeBundleRows.map(i => (
          <line
            key={i}
            x1={698} x2={800}
            y1={196 + i * 16} y2={196 + i * 16}
            stroke={tubeColor(i)}
            strokeWidth={2.5}
          />
        ))}

        {/* HX left end cap (primary hot inlet) */}
        <ellipse
          cx={680} cy={237} rx={14} ry={55}
          fill="rgba(124,45,18,0.55)"
          stroke="rgba(249,115,22,0.45)"
          strokeWidth={1.5}
        />

        {/* HX right end cap (primary cold outlet) */}
        <ellipse
          cx={820} cy={237} rx={14} ry={55}
          fill="rgba(12,74,110,0.55)"
          stroke="rgba(56,189,248,0.45)"
          strokeWidth={1.5}
        />

        {/* Secondary side stubs (top and bottom of HX shell) */}
        <rect
          x={726} y={158} width={44} height={20} rx={4}
          fill="rgba(15,28,63,0.75)"
          stroke="rgba(56,189,248,0.28)"
          strokeWidth={1.2}
        />
        <text
          x={748} y={152}
          textAnchor="middle"
          fill="rgba(56,189,248,0.55)"
          fontSize={8}
          letterSpacing="0.08em"
          fontFamily="inherit"
        >
          20 °C SECONDARY IN
        </text>

        <rect
          x={726} y={296} width={44} height={20} rx={4}
          fill="rgba(15,28,63,0.75)"
          stroke="rgba(56,189,248,0.28)"
          strokeWidth={1.2}
        />
        <text
          x={748} y={332}
          textAnchor="middle"
          fill="rgba(56,189,248,0.45)"
          fontSize={8}
          letterSpacing="0.08em"
          fontFamily="inherit"
        >
          SECONDARY OUT
        </text>

        {/* HX label */}
        <text
          x={750} y={170}
          textAnchor="middle"
          fill="rgba(148,163,184,0.45)"
          fontSize={9}
          letterSpacing="0.12em"
          fontFamily="inherit"
        >
          HX-01
        </text>

        {/* ══════════════════════════════════════════════════════════════════
            PRIMARY PUMP
        ══════════════════════════════════════════════════════════════════ */}

        {/* Pump outer casing */}
        <circle
          cx={530} cy={416} r={26}
          fill="rgba(15,28,63,0.92)"
          stroke="rgba(148,163,184,0.35)"
          strokeWidth={1.5}
        />
        {/* Impeller symbol */}
        <circle
          cx={530} cy={416} r={11}
          fill="none"
          stroke="rgba(148,163,184,0.45)"
          strokeWidth={1.2}
        />
        <line
          x1={530} y1={405} x2={538} y2={416}
          stroke="rgba(148,163,184,0.6)"
          strokeWidth={1.5}
        />
        <line
          x1={538} y1={416} x2={530} y2={425}
          stroke="rgba(148,163,184,0.6)"
          strokeWidth={1.5}
        />
        <line
          x1={530} y1={425} x2={522} y2={416}
          stroke="rgba(148,163,184,0.6)"
          strokeWidth={1.5}
        />
        {/* Pump label */}
        <text
          x={530} y={434}
          textAnchor="middle"
          fill="rgba(148,163,184,0.45)"
          fontSize={9}
          letterSpacing="0.1em"
          fontFamily="inherit"
        >
          P-01
        </text>

        {/* ══════════════════════════════════════════════════════════════════
            MEASUREMENT BADGES
        ══════════════════════════════════════════════════════════════════ */}

        {/* T_inlet — overlaid on upper plenum (center 156,93) */}
        <ValueBadge
          cx={156} cy={93}
          label="T INLET"
          value={tIn}
          accentColor="rgba(56,189,248,0.72)"
        />

        {/* T_outlet — overlaid on lower plenum (center 156,307) */}
        <ValueBadge
          cx={156} cy={307}
          label="T OUTLET"
          value={tOut}
          accentColor="rgba(249,115,22,0.72)"
        />

        {/* Mass flow — above pump */}
        <ValueBadge
          cx={530} cy={382}
          label="FLOW"
          value={mDot}
          accentColor="rgba(74,222,128,0.72)"
        />

        {/* ΔP — between pipe exits, right of vessel */}
        <ValueBadge
          cx={310} cy={200}
          label="ΔP CORE"
          value={dP}
          accentColor="rgba(167,139,250,0.72)"
        />
      </svg>
    </article>
  )
}
