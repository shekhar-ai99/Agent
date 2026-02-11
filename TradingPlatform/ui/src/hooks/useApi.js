import { useState, useEffect, useCallback } from 'react'
import apiClient from '../api/client'

export const useFetch = (fetchFn, dependencies = []) => {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetch = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetchFn()
      setData(response.data)
    } catch (err) {
      setError(err.message || 'An error occurred')
      console.error('Fetch error:', err)
    } finally {
      setLoading(false)
    }
  }, [fetchFn])

  useEffect(() => {
    fetch()
  }, dependencies)

  return { data, loading, error, refetch: fetch }
}

export const useRunBacktest = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [runId, setRunId] = useState(null)

  const run = useCallback(async (config) => {
    setLoading(true)
    setError(null)
    try {
      const response = await apiClient.runBacktest(config)
      setRunId(response.data.run_id)
      return response.data
    } catch (err) {
      const errorMsg = err.response?.data?.message || err.message || 'Backtest failed'
      setError(errorMsg)
      console.error('Backtest error:', err)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  return { run, loading, error, runId }
}

export const useRunSimulation = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [runId, setRunId] = useState(null)

  const run = useCallback(async (config) => {
    setLoading(true)
    setError(null)
    try {
      const response = await apiClient.runSimulation(config)
      setRunId(response.data.run_id)
      return response.data
    } catch (err) {
      const errorMsg = err.response?.data?.message || err.message || 'Simulation failed'
      setError(errorMsg)
      console.error('Simulation error:', err)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  return { run, loading, error, runId }
}

export const usePollRunStatus = (runId, pollInterval = 2000) => {
  const [status, setStatus] = useState('running')
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    if (!runId) return

    const poll = async () => {
      try {
        const response = await apiClient.checkRunStatus(runId)
        setStatus(response.data.status)
        setProgress(response.data.progress || 0)
      } catch (err) {
        console.error('Status check failed:', err)
      }
    }

    const interval = setInterval(poll, pollInterval)
    return () => clearInterval(interval)
  }, [runId, pollInterval])

  return { status, progress }
}
