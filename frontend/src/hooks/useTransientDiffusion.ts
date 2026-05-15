import { useCallback, useEffect, useRef, useState } from 'react'
import { transientApi } from '../simulation/api'
import type { TransientDiffusionState } from '../simulation/types'

const POLL_MS = 2000

export function useTransientDiffusion() {
  const [state, setState] = useState<TransientDiffusionState | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Trigger counter allows the polling effect to re-run for immediate refreshes
  const [trigger, setTrigger] = useState(0)
  const refresh = useCallback(() => setTrigger((n) => n + 1), [])

  const runningRef = useRef(false)

  // Polling effect
  useEffect(() => {
    let cancelled = false

    const poll = async () => {
      try {
        const data = await transientApi.getState()
        if (cancelled) return
        setState(data)
        runningRef.current = data.running
        setError(null)
      } catch (e) {
        if (cancelled) return
        setError(e instanceof Error ? e.message : 'Backend unreachable')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    poll()

    const interval = setInterval(() => {
      if (runningRef.current) poll()
    }, POLL_MS)

    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [trigger])

  const setRunning = useCallback(async (running: boolean) => {
    try {
      const data = await transientApi.setRunning(running)
      setState(data)
      runningRef.current = data.running
      if (running) refresh() // restart polling loop
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed')
    }
  }, [refresh])

  const setRodInsertion = useCallback(async (insertionPercent: number) => {
    try {
      const data = await transientApi.setRodInsertion(insertionPercent)
      setState(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed')
    }
  }, [])

  const reset = useCallback(async () => {
    try {
      setLoading(true)
      const data = await transientApi.reset()
      setState(data)
      runningRef.current = false
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }, [])

  const manualStep = useCallback(async () => {
    try {
      const data = await transientApi.step()
      setState(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed')
    }
  }, [])

  return { state, loading, error, setRunning, setRodInsertion, reset, manualStep }
}
