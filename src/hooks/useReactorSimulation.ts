import { useCallback, useEffect, useRef, useState } from 'react'
import { ReactorEngine } from '../simulation/engine'
import { simulationTuning } from '../simulation/model'
import type { HistoryPoint, ReactorSnapshot } from '../simulation/types'

function toHistoryPoint(snapshot: ReactorSnapshot): HistoryPoint {
  return {
    timeSeconds: snapshot.timeSeconds,
    reactivityPcm: snapshot.reactivity.totalPcm,
    totalFlux: snapshot.totalFlux,
    thermalPowerMw: snapshot.thermalPowerMw,
  }
}

export function useReactorSimulation() {
  const [engine] = useState(() => new ReactorEngine())
  const [snapshot, setSnapshot] = useState(() => engine.getSnapshot())
  const [history, setHistory] = useState<HistoryPoint[]>(() => [
    toHistoryPoint(engine.getSnapshot()),
  ])
  const [running, setRunning] = useState(true)

  const animationFrameRef = useRef<number | null>(null)
  const lastFrameRef = useRef<number | null>(null)
  const historyAccumulatorRef = useRef(0)

  const refreshSnapshot = useCallback((appendHistory: boolean) => {
    const nextSnapshot = engine.getSnapshot()
    setSnapshot(nextSnapshot)

    if (appendHistory) {
      setHistory((current) => {
        const nextHistory = [...current, toHistoryPoint(nextSnapshot)]
        return nextHistory.slice(-simulationTuning.historyPointLimit)
      })
    }
  }, [engine])

  const reset = useCallback(() => {
    engine.reset()
    historyAccumulatorRef.current = 0
    lastFrameRef.current = null

    const nextSnapshot = engine.getSnapshot()
    setSnapshot(nextSnapshot)
    setHistory([toHistoryPoint(nextSnapshot)])
    setRunning(true)
  }, [engine])

  const scram = useCallback(() => {
    engine.scram()
    refreshSnapshot(true)
    setRunning(true)
  }, [engine, refreshSnapshot])

  const setRodInsertionPercent = useCallback(
    (insertionPercent: number) => {
      engine.setRodInsertion(insertionPercent)
      refreshSnapshot(false)
    },
    [engine, refreshSnapshot],
  )

  useEffect(() => {
    if (!running) {
      lastFrameRef.current = null
      return
    }

    const animate = (timestamp: number) => {
      if (lastFrameRef.current === null) {
        lastFrameRef.current = timestamp
      }

      const elapsedWallSeconds = Math.min(
        (timestamp - lastFrameRef.current) / 1000,
        simulationTuning.maxWallStepSeconds,
      )

      lastFrameRef.current = timestamp

      let simulatedSecondsRemaining =
        elapsedWallSeconds * simulationTuning.timeScale
      const pendingHistoryPoints: HistoryPoint[] = []

      while (simulatedSecondsRemaining > 0) {
        const stepSeconds = Math.min(
          simulatedSecondsRemaining,
          simulationTuning.integratorStepSeconds,
        )

        engine.step(stepSeconds)
        simulatedSecondsRemaining -= stepSeconds
        historyAccumulatorRef.current += stepSeconds

        if (historyAccumulatorRef.current >= simulationTuning.historySampleSeconds) {
          historyAccumulatorRef.current -= simulationTuning.historySampleSeconds
          pendingHistoryPoints.push(toHistoryPoint(engine.getSnapshot()))
        }
      }

      const nextSnapshot = engine.getSnapshot()
      setSnapshot(nextSnapshot)

      if (pendingHistoryPoints.length > 0) {
        setHistory((current) => {
          const nextHistory = [...current, ...pendingHistoryPoints]
          return nextHistory.slice(-simulationTuning.historyPointLimit)
        })
      }

      animationFrameRef.current = requestAnimationFrame(animate)
    }

    animationFrameRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [engine, running])

  return {
    history,
    reset,
    running,
    scram,
    setRodInsertionPercent,
    setRunning,
    snapshot,
  }
}
