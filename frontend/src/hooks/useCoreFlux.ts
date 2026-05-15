import { useCallback, useEffect, useState } from 'react'
import { simulationApi } from '../simulation/api'
import type { CoreFluxResponse } from '../simulation/types'

export type CoreFluxState =
  | { status: 'loading' }
  | { status: 'success'; data: CoreFluxResponse }
  | { status: 'error'; message: string }

export function useCoreFlux() {
  const [state, setState] = useState<CoreFluxState>({ status: 'loading' })
  const [trigger, setTrigger] = useState(0)

  // refresh is an event handler — setting state here is fine
  const refresh = useCallback(() => {
    setState({ status: 'loading' })
    setTrigger((n) => n + 1)
  }, [])

  useEffect(() => {
    let cancelled = false

    const run = async () => {
      try {
        const data = await simulationApi.getCoreFlux()
        if (!cancelled) {
          setState({ status: 'success', data })
        }
      } catch (err: unknown) {
        if (!cancelled) {
          const message =
            err instanceof Error ? err.message : 'Failed to fetch flux distribution'
          setState({ status: 'error', message })
        }
      }
    }

    void run()

    return () => {
      cancelled = true
    }
  }, [trigger])

  return { state, refresh }
}
