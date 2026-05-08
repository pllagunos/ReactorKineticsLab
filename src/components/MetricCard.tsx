type MetricCardProps = {
  label: string
  value: string
  detail: string
  tone?: 'neutral' | 'cool' | 'warning' | 'danger'
}

export function MetricCard({
  label,
  value,
  detail,
  tone = 'neutral',
}: MetricCardProps) {
  return (
    <article className={`metric-card metric-card--${tone}`}>
      <span className="metric-card__label">{label}</span>
      <p className="metric-card__value">{value}</p>
      <p className="metric-card__detail">{detail}</p>
    </article>
  )
}
