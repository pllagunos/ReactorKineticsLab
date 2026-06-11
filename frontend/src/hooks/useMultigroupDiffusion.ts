import { useEffect, useState } from 'react'
import { multigroupDiffusionApi } from '../simulation/api'
import type { MultigroupDiffusionResponse } from '../simulation/types'

export type MultigroupDiffusionState =
  | { status: 'loading' }
  | { status: 'success'; data: MultigroupDiffusionResponse }
  | { status: 'error'; message: string }

export function useMultigroupDiffusion() {
  const [state, setState] = useState<MultigroupDiffusionState>({ status: 'loading' })

  useEffect(() => {
    let cancelled = false
    void multigroupDiffusionApi.getState()
      .then((data) => {
        if (!cancelled) setState({ status: 'success', data })
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setState({
            status: 'error',
            message: error instanceof Error ? error.message : 'Failed to load multigroup result',
          })
        }
      })
    return () => {
      cancelled = true
    }
  }, [])

  const recompute = async () => {
    setState({ status: 'loading' })
    try {
      const data = await multigroupDiffusionApi.recompute()
      setState({ status: 'success', data })
    } catch (error: unknown) {
      setState({
        status: 'error',
        message: error instanceof Error ? error.message : 'Failed to recompute multigroup result',
      })
    }
  }

  return { state, recompute }
}
