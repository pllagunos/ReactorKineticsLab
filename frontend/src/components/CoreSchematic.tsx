type CoreSchematicProps = {
  insertionPercent: number
  scramLatched: boolean
}

export function CoreSchematic({
  insertionPercent,
  scramLatched,
}: CoreSchematicProps) {
  const rodTravel = 68 + (insertionPercent / 100) * 168

  return (
    <div className="core-card">
      <svg
        className="core-svg"
        viewBox="0 0 280 320"
        role="img"
        aria-label="Schematic of a hollow cylindrical reactor core with control rod insertion"
      >
        <defs>
          <linearGradient id="fuelGradient" x1="0%" x2="0%" y1="0%" y2="100%">
            <stop offset="0%" stopColor="#fbbf24" />
            <stop offset="100%" stopColor="#fb7185" />
          </linearGradient>
        </defs>

        <text x="140" y="22" textAnchor="middle" className="core-label">
          simplified longitudinal core section
        </text>

        <rect x="54" y="48" width="172" height="220" rx="84" className="core-shell" />
        <rect x="76" y="64" width="128" height="188" rx="62" className="core-fuel" />
        <rect x="116" y="74" width="48" height="168" rx="24" className="core-channel" />
        <line x1="140" y1="40" x2="140" y2="268" className="core-guide" />
        <rect
          x="132"
          y="40"
          width="16"
          height={rodTravel}
          rx="6"
          className={scramLatched ? 'core-rod core-rod--scrammed' : 'core-rod'}
        />

        <text x="140" y="290" textAnchor="middle" className="core-label">
          heavy-water-moderated annulus
        </text>
      </svg>

      <div className="core-legend" aria-hidden="true">
        <span>
          <span className="legend-swatch legend-swatch--fuel"></span>
          Fuel region
        </span>
        <span>
          <span className="legend-swatch legend-swatch--moderator"></span>
          Hollow moderator channel
        </span>
        <span>
          <span className="legend-swatch legend-swatch--rod"></span>
          Rod bank
        </span>
      </div>
    </div>
  )
}
