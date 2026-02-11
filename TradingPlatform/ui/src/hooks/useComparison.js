import { useState, useCallback } from 'react'

export const useComparison = (runs = []) => {
  const [selected, setSelected] = useState([])

  const toggleRun = useCallback((runId) => {
    setSelected(prev => {
      if (prev.includes(runId)) {
        return prev.filter(id => id !== runId)
      } else {
        return [...prev, runId]
      }
    })
  }, [])

  const getSelectedRuns = useCallback(() => {
    return runs.filter(r => selected.includes(r.run_id))
  }, [runs, selected])

  const clearSelection = useCallback(() => {
    setSelected([])
  }, [])

  return {
    selected,
    toggleRun,
    getSelectedRuns,
    clearSelection,
    canCompare: selected.length >= 2
  }
}

// Format comparison table data
export const formatComparisonData = (runs) => {
  return runs.map(run => ({
    run_id: run.run_id,
    instrument: run.instrument || run.meta?.instrument || '--',
    mode: run.mode || run.meta?.mode || '--',
    timeframe: run.timeframe || run.meta?.timeframe || '--',
    net_pnl: run.total_pnl || run.net_pnl || 0,
    win_rate: run.win_rate || 0,
    max_drawdown: run.max_drawdown || 0,
    sharpe_ratio: run.sharpe_ratio || 0,
  }))
}
