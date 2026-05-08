import type { HistoryPoint } from '../simulation/types'
import { formatSimTime } from '../utils/format'

type LineChartProps = {
  title: string
  subtitle: string
  color: string
  data: HistoryPoint[]
  valueAccessor: (point: HistoryPoint) => number
  valueFormatter: (value: number) => string
}

const width = 720
const height = 220
const padding = 16

function buildChartPath(values: number[]) {
  if (values.length === 0) {
    return {
      areaPath: '',
      linePath: '',
      minimum: 0,
      maximum: 0,
    }
  }

  const minimum = Math.min(...values)
  const maximum = Math.max(...values)
  const innerWidth = width - padding * 2
  const innerHeight = height - padding * 2
  const span = maximum - minimum

  const points = values.map((value, index) => {
    const x =
      padding + (index / Math.max(values.length - 1, 1)) * innerWidth
    const y =
      span === 0
        ? padding + innerHeight / 2
        : padding + innerHeight - ((value - minimum) / span) * innerHeight

    return { x, y }
  })

  const linePath = points
    .map((point, index) =>
      `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`,
    )
    .join(' ')

  const areaPath = `${linePath} L ${points[points.length - 1].x.toFixed(2)} ${(height - padding).toFixed(2)} L ${points[0].x.toFixed(2)} ${(height - padding).toFixed(2)} Z`

  return {
    areaPath,
    linePath,
    minimum,
    maximum,
  }
}

export function LineChart({
  title,
  subtitle,
  color,
  data,
  valueAccessor,
  valueFormatter,
}: LineChartProps) {
  const values = data.map(valueAccessor)
  const latestValue = values.at(-1) ?? 0
  const { areaPath, linePath, maximum, minimum } = buildChartPath(values)
  const historyWindowSeconds =
    data.length > 1 ? data[data.length - 1].timeSeconds - data[0].timeSeconds : 0

  return (
    <article className="panel chart-card">
      <div className="chart-card__header">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
        <p className="chart-card__current">{valueFormatter(latestValue)}</p>
      </div>

      <svg
        className="chart-card__svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${title} over time`}
      >
        <line
          x1={padding}
          x2={width - padding}
          y1={height / 2}
          y2={height / 2}
          stroke="rgba(148, 163, 184, 0.16)"
          strokeDasharray="6 6"
        />
        <line
          x1={padding}
          x2={width - padding}
          y1={height - padding}
          y2={height - padding}
          stroke="rgba(148, 163, 184, 0.18)"
        />
        <line
          x1={padding}
          x2={padding}
          y1={padding}
          y2={height - padding}
          stroke="rgba(148, 163, 184, 0.18)"
        />
        {areaPath ? <path d={areaPath} fill={color} fillOpacity="0.16" /> : null}
        {linePath ? (
          <path
            d={linePath}
            fill="none"
            stroke={color}
            strokeWidth="3"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ) : null}
      </svg>

      <div className="chart-card__footer">
        <span>min {valueFormatter(minimum)}</span>
        <span>window {formatSimTime(historyWindowSeconds)}</span>
        <span>max {valueFormatter(maximum)}</span>
      </div>
    </article>
  )
}
