import type { SimulationState, CoreFluxResponse } from './types'

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(/\/$/, '')

async function request<T>(path: string, init?: RequestInit) {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
    ...init,
  })

  if (!response.ok) {
    throw new Error(`Backend request failed with status ${response.status}`)
  }

  return (await response.json()) as T
}

export const simulationApi = {
  getState() {
    return request<SimulationState>('/simulation/state')
  },
  reset() {
    return request<SimulationState>('/simulation/reset', { method: 'POST' })
  },
  scram() {
    return request<SimulationState>('/simulation/scram', { method: 'POST' })
  },
  setRodInsertionPercent(insertionPercent: number) {
    return request<SimulationState>('/simulation/rod-insertion', {
      method: 'POST',
      body: JSON.stringify({ insertionPercent }),
    })
  },
  setRunning(running: boolean) {
    return request<SimulationState>('/simulation/running', {
      method: 'POST',
      body: JSON.stringify({ running }),
    })
  },
  getCoreFlux() {
    return request<CoreFluxResponse>('/core/flux')
  },
}
