import { useState, useEffect } from 'react'

// Run history management (localStorage)
const RUN_HISTORY_KEY = 'tp.run.history'
const ACTIVE_RUN_KEY = 'tp.run.active'

export const useRunHistory = () => {
  const [history, setHistory] = useState([])

  // Load history on mount
  useEffect(() => {
    const raw = localStorage.getItem(RUN_HISTORY_KEY)
    if (raw) {
      setHistory(JSON.parse(raw))
    }
  }, [])

  const addRun = (runData) => {
    const newHistory = [runData, ...history].slice(0, 10) // Keep last 10
    setHistory(newHistory)
    localStorage.setItem(RUN_HISTORY_KEY, JSON.stringify(newHistory))
  }

  const clearHistory = () => {
    setHistory([])
    localStorage.removeItem(RUN_HISTORY_KEY)
  }

  return { history, addRun, clearHistory }
}

// Active run (sessionStorage)
export const useActiveRun = () => {
  const [activeRun, setActiveRun] = useState(null)

  // Load active run on mount
  useEffect(() => {
    const stored = sessionStorage.getItem(ACTIVE_RUN_KEY)
    if (stored) {
      setActiveRun(stored)
    }
  }, [])

  const setRun = (runId) => {
    setActiveRun(runId)
    sessionStorage.setItem(ACTIVE_RUN_KEY, runId)
  }

  const clearRun = () => {
    setActiveRun(null)
    sessionStorage.removeItem(ACTIVE_RUN_KEY)
  }

  return { activeRun, setRun, clearRun }
}
