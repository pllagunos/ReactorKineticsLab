import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LineController,
  LinearScale,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js'
import type { ChartOptions, ScaleOptions, TooltipItem } from 'chart.js'
import { formatSimTime } from '../utils/format'

// ── Tree-shakable registration ────────────────────────────
ChartJS.register(LineElement, PointElement, LineController, LinearScale, Tooltip, Legend, Filler)

// ── Dark-theme defaults ───────────────────────────────────
ChartJS.defaults.color = 'rgba(148, 163, 184, 0.6)'
ChartJS.defaults.borderColor = 'rgba(148, 163, 184, 0.08)'
ChartJS.defaults.font.family = 'Inter, ui-sans-serif, system-ui, sans-serif'

// ── Public API ────────────────────────────────────────────

export type TrendSeries<T> = {
  id: string
  label: string
  unit: string
  color: string
  data: T[]
  valueAccessor: (point: T) => number
  valueFormatter: (value: number) => string
  tickValueFormatter?: (value: number) => string
}

type TrendChartProps<T extends { timeSeconds: number }> = {
  title: string
  series: TrendSeries<T>[]
}

const MAX_WINDOW_SECONDS = 300

// ── Helpers ───────────────────────────────────────────────

function padRange(min: number, max: number): [number, number] {
  if (min === max) {
    // Flat line: give more room below than above so the line sits
    // above the chart centre and different series don't all overlap at 50%.
    const mag = min === 0 ? 10 : Math.abs(min)
    const padUp = Math.max(mag * 0.03, 2)
    const padDown = Math.max(mag * 0.12, padUp * 3)
    return [min - padDown, max + padUp]
  }
  const span = max - min
  const pad = span * 0.12
  let lo = min - pad
  let hi = max + pad
  const mag = Math.max(Math.abs(lo), Math.abs(hi), 1)
  if (hi - lo < mag * 0.06) {
    const mid = (lo + hi) / 2
    lo = mid - (mag * 0.06) / 2
    hi = mid + (mag * 0.06) / 2
  }
  return [lo, hi]
}

// ── Component ─────────────────────────────────────────────

export function TrendChart<T extends { timeSeconds: number }>({
  title,
  series,
}: TrendChartProps<T>) {
  // ── 5‑minute rolling time window ────────────────────────
  const allLatest = series.flatMap((s) =>
    s.data.length > 0 ? [s.data[s.data.length - 1].timeSeconds] : [],
  )
  const latestTime = allLatest.length > 0 ? Math.max(...allLatest) : 0
  const windowStart = latestTime - MAX_WINDOW_SECONDS

  const windowed = series.map((s) => ({
    ...s,
    data: s.data.filter((p) => p.timeSeconds >= windowStart),
  }))
  const active = windowed.filter((s) => s.data.length > 1)

  if (active.length === 0) {
    return (
      <article className="panel chart-card">
        <div className="chart-card__header">
          <h3>{title}</h3>
          <p className="chart-card__current">No data yet</p>
        </div>
      </article>
    )
  }

  // ── Clamp X-axis min so the window doesn't show empty space
  //    before the first data point (e.g. early in a simulation).
  const firstTimes = active.flatMap((s) =>
    s.data.length > 0 ? [s.data[0].timeSeconds] : [],
  )
  const dataFirst = firstTimes.length > 0 ? Math.min(...firstTimes) : 0
  const xMin = Math.max(windowStart, dataFirst)

  // ── Build datasets ──────────────────────────────────────
  const datasets = active.map((s, i) => ({
    label: s.label,
    data: s.data.map((p) => ({ x: p.timeSeconds, y: s.valueAccessor(p) })),
    borderColor: s.color,
    backgroundColor: s.color,
    yAxisID: `y${i}`,
    pointRadius: 0,
    pointHoverRadius: 4,
    pointHoverBackgroundColor: s.color,
    borderWidth: 2,
    tension: 0,
  }))

  // ── Build Y‑axes (one per series, alternating sides) ────
  const yAxes: Record<string, ScaleOptions<'linear'>> = {}
  active.forEach((s, i) => {
    const values = s.data.map(s.valueAccessor)
    const [lo, hi] = padRange(Math.min(...values), Math.max(...values))
    const fmt = s.tickValueFormatter ?? s.valueFormatter

    yAxes[`y${i}`] = {
      type: 'linear' as const,
      position: i % 2 === 0 ? 'left' : 'right',
      min: lo,
      max: hi,
      ticks: {
        callback: (v: string | number) => fmt(Number(v)),
        color: s.color,
        font: { size: 9 },
        maxTicksLimit: 5,
      },
      title: {
        display: true,
        text: s.unit,
        color: s.color,
        font: { size: 9, weight: 'normal' },
      },
      grid: {
        display: i === 0,
        color: 'rgba(148, 163, 184, 0.06)',
      },
    }
  })

  // ── Chart options ───────────────────────────────────────
  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false as const,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        align: 'end',
        labels: {
          boxWidth: 10,
          boxHeight: 10,
          padding: 14,
          usePointStyle: true,
          pointStyleWidth: 10,
          font: { size: 10 },
          color: 'rgba(203, 213, 225, 0.8)',
        },
      },
      tooltip: {
        backgroundColor: 'rgba(2, 6, 23, 0.94)',
        titleColor: 'rgba(226, 232, 240, 0.9)',
        bodyColor: 'rgba(203, 213, 225, 0.85)',
        borderColor: 'rgba(148, 163, 184, 0.2)',
        borderWidth: 1,
        padding: 10,
        titleFont: { size: 10 },
        bodyFont: { size: 10 },
        callbacks: {
          title: (items: TooltipItem<'line'>[]) =>
            formatSimTime((items[0]?.parsed as { x: number })?.x ?? 0),
          label: (item: TooltipItem<'line'>) => {
            const ds = active[item.datasetIndex]
            const fmt = ds.valueFormatter
            return ` ${ds.label}: ${fmt((item.parsed as { y: number }).y)}`
          },
        },
      },
    },
    scales: {
      x: {
        type: 'linear',
        min: xMin,
        max: latestTime,
        ticks: {
          callback: (v: string | number) => formatSimTime(Number(v)),
          maxTicksLimit: 5,
          font: { size: 9 },
        },
        grid: { display: false },
      },
      ...yAxes,
    },
  }

  return (
    <article className="panel chart-card">
      <div className="chart-card__header">
        <h3>{title}</h3>
      </div>
      <div className="chart-card__canvas-wrap">
        <Line data={{ datasets }} options={options} />
      </div>
    </article>
  )
}
