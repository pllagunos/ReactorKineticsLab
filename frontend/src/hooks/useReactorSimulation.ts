import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type SetStateAction,
} from 'react'
import { simulationApi } from '../simulation/api'
import type { SimulationState, ThermalSnapshot } from '../simulation/types'

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : 'Unknown backend error'
}

export function useReactorSimulation() {
  const [state, setState] = useState<SimulationState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [thermalHistory, setThermalHistory] = useState<ThermalSnapshot[]>([])
  const pollInFlightRef = useRef(false)
  const actionTokenRef = useRef(0)

  const applyState = useCallback((nextState: SimulationState) => {
    setState(nextState)
    setError(null)
    if (nextState.thermal.available) {
      setThermalHistory(prev => {
        const next = [...prev, nextState.thermal]
        return next.length > 300 ? next.slice(-300) : next
      })
    }
  }, [])

  const runAction = useCallback(
    async (action: () => Promise<SimulationState>) => {
      const token = ++actionTokenRef.current

      try {
        const nextState = await action()

        if (token === actionTokenRef.current) {
          applyState(nextState)
        }
      } catch (nextError) {
        setError(getErrorMessage(nextError))
      }
    },
    [applyState],
  )

  const reset = useCallback(() => {
    void runAction(() => simulationApi.reset())
  }, [runAction])

  const scram = useCallback(() => {
    void runAction(() => simulationApi.scram())
  }, [runAction])

  const setRodInsertionPercent = useCallback(
    (insertionPercent: number) => {
      void runAction(() => simulationApi.setRodInsertionPercent(insertionPercent))
    },
    [runAction],
  )

  const setRunning = useCallback(
    (nextState: SetStateAction<boolean>) => {
      const current = state?.running ?? false
      const running =
        typeof nextState === 'function' ? nextState(current) : nextState

      void runAction(() => simulationApi.setRunning(running))
    },
    [runAction, state?.running],
  )

  useEffect(() => {
    let cancelled = false

    const poll = async () => {
      if (cancelled || pollInFlightRef.current) {
        return
      }

      pollInFlightRef.current = true

      try {
        const nextState = await simulationApi.getState()

        if (!cancelled) {
          applyState(nextState)
        }
      } catch (nextError) {
        if (!cancelled) {
          setError(getErrorMessage(nextError))
        }
      } finally {
        pollInFlightRef.current = false
      }
    }

    void poll()

    const intervalId = window.setInterval(() => {
      void poll()
    }, state?.tuning.pollIntervalMs ?? 100)

    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
  }, [applyState, state?.tuning.pollIntervalMs])

  return {
    error,
    history: state?.history ?? [],
    loading: state === null,
    model: state?.model ?? null,
    reset,
    running: state?.running ?? false,
    scram,
    setRodInsertionPercent,
    setRunning,
    snapshot: state?.snapshot ?? null,
    thermal: state?.thermal ?? null,
    thermalHistory,
  }
}
