export function ThermalHydraulicsPage() {
  return (
    <div className="app-shell">
      <header className="hero-header panel">
        <div>
          <p className="eyebrow">Thermal hydraulics</p>
          <h1>Coolant & heat transfer</h1>
          <p className="hero-copy">
            This page is reserved for the thermal-hydraulic model. Once implemented, it
            will show coolant circuit conditions, heat removal rate, and temperature
            feedback coupling to the neutronics. No thermal-hydraulic solver exists yet.
          </p>
        </div>
        <div className="status-badge" style={{ borderColor: 'rgba(56,189,248,0.28)', background: 'rgba(12,74,110,0.22)' }}>
          <span className="status-badge__label">Not yet modeled</span>
          <span>
            Thermal-hydraulic feedback is planned for a future development phase.
          </span>
        </div>
      </header>

      <section className="notes-grid">
        <article className="panel note-panel">
          <p className="section-label">Coolant circuit</p>
          <h2>Primary D₂O loop</h2>
          <ul className="note-list">
            <li>Flow rate through fuel annulus — not yet coupled.</li>
            <li>Coolant inlet / outlet temperatures — placeholder.</li>
            <li>Pressure and void fraction — not modeled.</li>
            <li>Pump coast-down transients — future work.</li>
          </ul>
        </article>

        <article className="panel note-panel">
          <p className="section-label">Heat removal</p>
          <h2>Secondary side</h2>
          <ul className="note-list">
            <li>Steam generator / heat exchanger model — not yet implemented.</li>
            <li>Secondary-side feedwater flow — placeholder.</li>
            <li>Loss-of-heat-sink transients — future work.</li>
          </ul>
        </article>

        <article className="panel note-panel">
          <p className="section-label">Feedback</p>
          <h2>Temperature reactivity coupling</h2>
          <ul className="note-list">
            <li>Fuel temperature Doppler coefficient — not yet coupled to kinetics.</li>
            <li>Moderator temperature coefficient — placeholder.</li>
            <li>Void reactivity coefficient — not modeled.</li>
            <li>
              When implemented, these coefficients will feed back into the
              point-kinetics engine already running in the Python backend.
            </li>
          </ul>
        </article>

        <article className="panel note-panel">
          <p className="section-label">Roadmap</p>
          <h2>Planned development</h2>
          <ul className="note-list">
            <li>
              <strong style={{ color: 'var(--text-strong)' }}>Phase 1</strong> — add
              heat conduction model from fuel to coolant (simple lumped capacitance).
            </li>
            <li>
              <strong style={{ color: 'var(--text-strong)' }}>Phase 2</strong> — couple
              Doppler and moderator temperature feedback to the reactivity model.
            </li>
            <li>
              <strong style={{ color: 'var(--text-strong)' }}>Phase 3</strong> — add
              single-channel coolant flow model and heat-exchanger balance.
            </li>
            <li>
              <strong style={{ color: 'var(--text-strong)' }}>Phase 4</strong> — extend
              to loss-of-coolant and loss-of-heat-sink transient scenarios.
            </li>
          </ul>
        </article>
      </section>
    </div>
  )
}
