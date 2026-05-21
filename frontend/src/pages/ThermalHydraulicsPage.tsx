export function ThermalHydraulicsPage() {
  return (
    <div className="app-shell">
      <header className="hero-header panel">
        <div>
          <p className="eyebrow">Thermal hydraulics</p>
          <h1>Coolant &amp; heat transfer</h1>
          <p className="hero-copy">
            Architecture decided; Modelica implementation pending. The selected design
            is a simplified forced-circulation D₂O primary loop (25 °C inlet,
            45 °C nominal outlet, 237 kg/s) feeding a light-water secondary side via a
            single heat exchanger. See <code>theory/ThermalHydraulics.tex</code> for
            the full sizing note.
          </p>
        </div>
        <div className="status-badge" style={{ borderColor: 'rgba(56,189,248,0.28)', background: 'rgba(12,74,110,0.22)' }}>
          <span className="status-badge__label">Architecture defined — not yet coupled</span>
          <span>
            Modelica model under development. Live data will appear here once the
            FMI interface is wired up.
          </span>
        </div>
      </header>

      <section className="notes-grid">
        <article className="panel note-panel">
          <p className="section-label">Primary loop — selected design</p>
          <h2>D₂O forced circulation</h2>
          <ul className="note-list">
            <li><strong style={{ color: 'var(--text-strong)' }}>Flow direction:</strong> top-to-bottom through effective core channel.</li>
            <li><strong style={{ color: 'var(--text-strong)' }}>Inlet temperature:</strong> 25 °C.</li>
            <li><strong style={{ color: 'var(--text-strong)' }}>Nominal outlet:</strong> 45 °C (upper limit 55 °C).</li>
            <li><strong style={{ color: 'var(--text-strong)' }}>Nominal mass flow:</strong> 237 kg/s (776 m³/h).</li>
            <li><strong style={{ color: 'var(--text-strong)' }}>Pressure regime:</strong> ≤ 3 bar gauge.</li>
            <li>Single pump, single HX — no parallel trains in first pass.</li>
          </ul>
        </article>

        <article className="panel note-panel">
          <p className="section-label">Core abstraction</p>
          <h2>Single effective channel</h2>
          <ul className="note-list">
            <li>One axially discretised DynamicPipe (8 nodes) carries the full primary flow.</li>
            <li>Inlet and outlet plena provide hydraulic decoupling.</li>
            <li>Axial power profile from neutronics drives distributed heat input.</li>
            <li>Outer reflector and pool are stagnant — not in the forced-flow path.</li>
            <li>Pool (≈ 2 535 m³) modelled as a fixed-temperature thermal boundary.</li>
          </ul>
        </article>

        <article className="panel note-panel">
          <p className="section-label">Secondary loop — selected design</p>
          <h2>Light-water heat sink</h2>
          <ul className="note-list">
            <li>Single H₂O loop: source tank → pump → HX secondary side → sink tank.</li>
            <li>Closes the energy balance without modelling a full balance-of-plant.</li>
            <li>No tertiary loop in first implementation pass.</li>
            <li>Secondary-side temperatures set by HX sizing (≈ 10–15 K approach).</li>
          </ul>
        </article>

        <article className="panel note-panel">
          <p className="section-label">Implementation roadmap</p>
          <h2>Next steps</h2>
          <ul className="note-list">
            <li>
              <strong style={{ color: 'var(--text-strong)' }}>Step 1</strong> — build
              reactor-scale Modelica TH model (plena + channel + pump + HX + secondary).
            </li>
            <li>
              <strong style={{ color: 'var(--text-strong)' }}>Step 2</strong> — wire FMI
              interface: Python sends power &amp; axial fractions; Modelica returns
              T_in, T_out, ṁ, ΔP.
            </li>
            <li>
              <strong style={{ color: 'var(--text-strong)' }}>Step 3</strong> — couple
              Doppler and moderator temperature feedback into the point-kinetics engine.
            </li>
            <li>
              <strong style={{ color: 'var(--text-strong)' }}>Step 4</strong> — extend
              to loss-of-flow and loss-of-heat-sink transient scenarios.
            </li>
          </ul>
        </article>
      </section>
    </div>
  )
}
