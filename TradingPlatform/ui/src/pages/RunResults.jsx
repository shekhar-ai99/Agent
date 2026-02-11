import { useEffect, useState } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { Card, MetricCard, Spinner, Alert, Button, EmptyState, Table, Badge } from '../components/common/index'
import {
  EquityCurveChart,
  DrawdownChart,
  TradesPerDayChart,
  PnLDistributionChart,
} from '../components/charts/Charts'
import { useFetch, usePollRunStatus } from '../hooks/useApi'
import { useRunHistory } from '../hooks/useStorage'
import { useComparison, formatComparisonData } from '../hooks/useComparison'
import { exportComparison } from '../utils/export'
import apiClient from '../api/client'
import { formatCurrency, formatPercent, formatDateTime } from '../utils/config'

function RunResults() {
  const navigate = useNavigate()
  const { runId } = useParams()
  const { data: results, loading: resultsLoading, error: resultsError } = useFetch(
    () => apiClient.getResults(runId),
    [runId]
  )
  const { status, progress } = usePollRunStatus(runId)
  const { history, addRun } = useRunHistory()
  const { selected, toggleRun, getSelectedRuns, canCompare } = useComparison(history)

  if (resultsError) {
    return (
      <div className="space-y-6">
        <h1 className="text-3xl font-bold text-gray-900">Run Results</h1>
        <Alert type="error" message={`Failed to load results: ${resultsError}`} />
        <Link to="/">
          <Button>Back to Configuration</Button>
        </Link>
      </div>
    )
  }

  if (resultsLoading || (status === 'running' && !results)) {
    return (
      <div className="space-y-6">
        <h1 className="text-3xl font-bold text-gray-900">Run Results</h1>
        <Card className="text-center py-12">
          <Spinner />
          <p className="mt-4 text-gray-600">Running backtest... {progress}%</p>
          <div className="mt-4 w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-trading-600 h-2 rounded-full transition-all"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </Card>
      </div>
    )
  }

  if (!results) {
    return (
      <div className="space-y-6">
        <h1 className="text-3xl font-bold text-gray-900">Run Results</h1>
        <EmptyState
          title="No Results Available"
          description="The backtest results could not be loaded."
          action={<Link to="/"><Button>Back to Configuration</Button></Link>}
        />
      </div>
    )
  }

  // Save run to history when results load
  useEffect(() => {
    if (results && runId) {
      addRun({
        run_id: runId,
        instrument: results.meta?.instrument || '--',
        mode: results.meta?.mode || '--',
        timeframe: results.meta?.timeframe || '--',
        timestamp: new Date().toISOString(),
        net_pnl: results.total_pnl || results.net_pnl,
        win_rate: results.win_rate,
        max_drawdown: results.max_drawdown,
        sharpe_ratio: results.sharpe_ratio
      })
    }
  }, [results, runId, addRun])

  const {
    total_pnl = 0,
    net_pnl = 0,
    return_pct = 0,
    win_rate = 0,
    num_trades = 0,
    max_drawdown = 0,
    sharpe_ratio = 0,
    profit_factor = 0,
    equity_curve = [],
    drawdown_curve = [],
    trades_per_day = [],
    pnl_trades = [],
  } = results

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Run Results</h1>
          <p className="mt-2 text-gray-600">Run ID: {runId}</p>
        </div>
        <Link to="/">
          <Button variant="outline">← Back</Button>
        </Link>
      </div>

      {/* Status Badge */}
      {status === 'completed' && (
        <Alert type="success" message="✅ Backtest completed successfully" />
      )}
      {status === 'failed' && (
        <Alert type="error" message="❌ Backtest failed. Please check the configuration and try again." />
      )}

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <MetricCard
          label="Total P&L"
          value={formatCurrency(total_pnl || net_pnl)}
          icon={total_pnl >= 0 ? '📈' : '📉'}
        />
        <MetricCard
          label="Return"
          value={formatPercent(return_pct)}
          icon="💰"
        />
        <MetricCard
          label="Total Trades"
          value={num_trades}
          icon="📊"
        />
        <MetricCard
          label="Win Rate"
          value={formatPercent(win_rate)}
          icon="✅"
        />
        <MetricCard
          label="Max Drawdown"
          value={formatPercent(Math.abs(max_drawdown))}
          icon="📉"
        />
        <MetricCard
          label="Sharpe Ratio"
          value={sharpe_ratio.toFixed(2)}
          icon="📈"
        />
      </div>

      {/* Additional Metrics */}
      <Card>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-sm text-gray-600">Profit Factor</p>
            <p className="text-2xl font-bold text-gray-900">{profit_factor.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Total Trades</p>
            <p className="text-2xl font-bold text-gray-900">{num_trades}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Max Drawdown</p>
            <p className="text-2xl font-bold text-red-600">{formatPercent(Math.abs(max_drawdown))}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Return on Investment</p>
            <p className="text-2xl font-bold text-green-600">{formatPercent(return_pct)}</p>
          </div>
        </div>
      </Card>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Equity Curve</h3>
          {equity_curve && equity_curve.length > 0 ? (
            <EquityCurveChart data={equity_curve} />
          ) : (
            <p className="text-gray-500">No equity curve data</p>
          )}
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Drawdown</h3>
          {drawdown_curve && drawdown_curve.length > 0 ? (
            <DrawdownChart data={drawdown_curve} />
          ) : (
            <p className="text-gray-500">No drawdown data</p>
          )}
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Trades per Day</h3>
          {trades_per_day && trades_per_day.length > 0 ? (
            <TradesPerDayChart data={trades_per_day} />
          ) : (
            <p className="text-gray-500">No daily trades data</p>
          )}
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-gray-900 mb-4">P&L Distribution</h3>
          {pnl_trades && pnl_trades.length > 0 ? (
            <PnLDistributionChart data={pnl_trades} />
          ) : (
            <p className="text-gray-500">No trade data</p>
          )}
        </Card>
      </div>

      {/* Run History */}
      {history.length > 0 && (
        <Card>
          <h3 className="text-lg font-semibold text-gray-900 mb-4">📊 Run History</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {history.slice(0, 6).map(run => (
              <div key={run.run_id} className="p-4 border border-gray-200 rounded-lg hover:border-trading-600 transition cursor-pointer"
                onClick={() => navigate(`/results/${run.run_id}`)}>
                <p className="font-semibold text-gray-900">{run.instrument}</p>
                <p className="text-sm text-gray-500">{run.mode} • {run.timeframe}</p>
                <p className="text-sm text-gray-500">{new Date(run.timestamp).toLocaleDateString()}</p>
                <p className={`text-sm font-bold mt-2 ${run.net_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  P&L: {formatCurrency(run.net_pnl || 0)}
                </p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Run Comparison */}
      {history.length > 1 && (
        <Card>
          <h3 className="text-lg font-semibold text-gray-900 mb-4">🔄 Compare Runs</h3>
          <p className="text-sm text-gray-600 mb-4">Select runs to compare performance metrics</p>
          
          <div className="space-y-3 mb-4 max-h-60 overflow-y-auto">
            {history.map(run => (
              <label key={run.run_id} className="flex items-center p-3 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer">
                <input
                  type="checkbox"
                  checked={selected.includes(run.run_id)}
                  onChange={() => toggleRun(run.run_id)}
                  className="w-4 h-4"
                />
                <div className="ml-3 flex-1">
                  <p className="font-semibold text-gray-900">{run.instrument}</p>
                  <p className="text-sm text-gray-500">{run.mode} • {run.timeframe}</p>
                </div>
                <p className={`text-sm font-bold ${run.net_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {formatCurrency(run.net_pnl || 0)}
                </p>
              </label>
            ))}
          </div>

          {canCompare && (
            <>
              <div className="overflow-x-auto mb-4">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 bg-gray-50">
                      <th className="px-4 py-2 text-left font-semibold">Run ID</th>
                      <th className="px-4 py-2 text-left font-semibold">Instrument</th>
                      <th className="px-4 py-2 text-right font-semibold">Net P&L</th>
                      <th className="px-4 py-2 text-right font-semibold">Win Rate</th>
                      <th className="px-4 py-2 text-right font-semibold">Max DD</th>
                      <th className="px-4 py-2 text-right font-semibold">Sharpe</th>
                    </tr>
                  </thead>
                  <tbody>
                    {getSelectedRuns().map(run => (
                      <tr key={run.run_id} className="border-b border-gray-200">
                        <td className="px-4 py-2 font-mono text-xs">{run.run_id.substring(0, 20)}...</td>
                        <td className="px-4 py-2">{run.instrument}</td>
                        <td className={`px-4 py-2 text-right font-bold ${run.net_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {formatCurrency(run.net_pnl || 0)}
                        </td>
                        <td className="px-4 py-2 text-right">{formatPercent(run.win_rate || 0)}</td>
                        <td className="px-4 py-2 text-right text-red-600">{formatPercent(Math.abs(run.max_drawdown || 0))}</td>
                        <td className="px-4 py-2 text-right">{(run.sharpe_ratio || 0).toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => exportComparison(formatComparisonData(getSelectedRuns()), 'csv')}
              >
                📥 Export Comparison
              </Button>
            </>
          )}
        </Card>
      )}

      {/* Action Buttons */}
      <div className="flex gap-4">
        <Link to={`/trades/${runId}`}>
          <Button variant="primary">📋 View Trade Details</Button>
        </Link>
        <Link to="/rankings">
          <Button variant="secondary">⭐ View Rankings</Button>
        </Link>
      </div>
    </div>
  )
}

export default RunResults
