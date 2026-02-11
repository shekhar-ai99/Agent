import { useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { Table, Card, Button, Select, Badge, EmptyState, Spinner } from '../components/common/index'
import { useFetch } from '../hooks/useApi'
import { exportRanking } from '../utils/export'
import apiClient from '../api/client'
import { formatPercent, formatCurrency } from '../utils/config'

function StrategyRanking() {
  const { data: rankingData, loading } = useFetch(
    () => apiClient.getStrategyRanking(),
    []
  )

  const [filterMarket, setFilterMarket] = useState('')
  const [filterDay, setFilterDay] = useState('')
  const [filterSession, setFilterSession] = useState('')
  const [filterRegime, setFilterRegime] = useState('')
  const [filterVolatility, setFilterVolatility] = useState('')
  const [sortField, setSortField] = useState('rank')
  const [sortOrder, setSortOrder] = useState('asc')

  const strategies = rankingData?.strategies || []

  // Extract unique values for filters
  const markets = useMemo(() => [...new Set(strategies.map(s => s.market))], [strategies])
  const days = useMemo(() => [...new Set(strategies.flatMap(s => s.days || []))], [strategies])
  const sessions = useMemo(() => [...new Set(strategies.flatMap(s => s.sessions || []))], [strategies])
  const regimes = useMemo(() => [...new Set(strategies.flatMap(s => s.regimes || []))], [strategies])
  const volatilities = useMemo(() => [...new Set(strategies.flatMap(s => s.volatilities || []))], [strategies])

  // Filter strategies
  const filteredStrategies = useMemo(() => {
    let filtered = strategies.filter(strategy => {
      if (filterMarket && strategy.market !== filterMarket) return false
      if (filterDay && !(strategy.days || []).includes(filterDay)) return false
      if (filterSession && !(strategy.sessions || []).includes(filterSession)) return false
      if (filterRegime && !(strategy.regimes || []).includes(filterRegime)) return false
      if (filterVolatility && !(strategy.volatilities || []).includes(filterVolatility)) return false
      return true
    })

    // Sort
    filtered.sort((a, b) => {
      let aVal = a[sortField] ?? 0
      let bVal = b[sortField] ?? 0

      if (typeof aVal === 'string') {
        aVal = aVal.toLowerCase()
        bVal = bVal.toLowerCase()
      }

      if (aVal < bVal) return sortOrder === 'asc' ? -1 : 1
      if (aVal > bVal) return sortOrder === 'asc' ? 1 : -1
      return 0
    })

    return filtered
  }, [strategies, filterMarket, filterDay, filterSession, filterRegime, filterVolatility, sortField, sortOrder])

  const columns = [
    {
      id: 'rank',
      label: 'Rank',
      render: (row) => (
        <div className="flex items-center gap-2">
          <span className="text-2xl">
            {row.rank === 1 ? '🥇' : row.rank === 2 ? '🥈' : row.rank === 3 ? '🥉' : '✓'}
          </span>
          <span className="text-lg font-bold">{row.rank}</span>
        </div>
      )
    },
    {
      id: 'name',
      label: 'Strategy Name',
      render: (row) => (
        <div>
          <p className="font-semibold text-gray-900">{row.name}</p>
          {row.market && <p className="text-xs text-gray-500">{row.market.toUpperCase()}</p>}
        </div>
      )
    },
    {
      id: 'win_rate',
      label: 'Win Rate',
      render: (row) => (
        <span className={row.win_rate >= 50 ? 'text-green-600 font-bold' : 'text-gray-700'}>
          {formatPercent(row.win_rate)}
        </span>
      )
    },
    {
      id: 'profit_factor',
      label: 'Profit Factor',
      render: (row) => (
        <span className={row.profit_factor >= 1.5 ? 'text-green-600 font-bold' : 'text-gray-700'}>
          {row.profit_factor.toFixed(2)}
        </span>
      )
    },
    {
      id: 'avg_pnl',
      label: 'Avg P&L',
      render: (row) => (
        <span className={row.avg_pnl >= 0 ? 'text-green-600 font-bold' : 'text-red-600 font-bold'}>
          {formatCurrency(row.avg_pnl)}
        </span>
      )
    },
    {
      id: 'total_pnl',
      label: 'Total P&L',
      render: (row) => (
        <span className={row.total_pnl >= 0 ? 'text-green-600 font-bold' : 'text-red-600 font-bold'}>
          {formatCurrency(row.total_pnl)}
        </span>
      )
    },
    {
      id: 'max_drawdown',
      label: 'Max Drawdown',
      render: (row) => <span className="text-red-600">{formatPercent(Math.abs(row.max_drawdown))}</span>
    },
    {
      id: 'trades_count',
      label: 'Trades',
      render: (row) => <span className="font-mono">{row.trades_count}</span>
    },
  ]

  if (loading) {
    return (
      <div className="space-y-6">
        <h1 className="text-3xl font-bold text-gray-900">Strategy Rankings</h1>
        <Card className="flex items-center justify-center py-12">
          <Spinner />
          <span className="ml-3 text-gray-600">Loading rankings...</span>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Strategy Rankings</h1>
          <p className="mt-2 text-gray-600">
            These rankings are derived from historical backtests on various market conditions
          </p>
        </div>
        <Link to="/">
          <Button variant="outline">← Back to Configure</Button>
        </Link>
      </div>

      {/* Info Card */}
      <Card className="bg-amber-50 border-l-4 border-amber-500">
        <h3 className="font-semibold text-amber-900">📊 About These Rankings</h3>
        <ul className="text-sm text-amber-700 mt-2 space-y-1">
          <li>• Based on backtests across different market conditions</li>
          <li>• Ranked by risk-adjusted returns (Sharpe ratio + Profit Factor)</li>
          <li>• Use filters to find best performers in specific conditions</li>
          <li>• Past performance does not guarantee future results</li>
        </ul>
      </Card>

      {/* Filters */}
      <Card>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Filters</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          {markets.length > 0 && (
            <Select
              label="Market"
              value={filterMarket}
              onChange={(e) => setFilterMarket(e.target.value)}
              options={markets.map(m => ({ id: m, name: m.toUpperCase() }))}
            />
          )}
          {days.length > 0 && (
            <Select
              label="Day of Week"
              value={filterDay}
              onChange={(e) => setFilterDay(e.target.value)}
              options={days.map(d => ({ id: d, name: d }))}
            />
          )}
          {sessions.length > 0 && (
            <Select
              label="Session"
              value={filterSession}
              onChange={(e) => setFilterSession(e.target.value)}
              options={sessions.map(s => ({ id: s, name: s }))}
            />
          )}
          {regimes.length > 0 && (
            <Select
              label="Regime"
              value={filterRegime}
              onChange={(e) => setFilterRegime(e.target.value)}
              options={regimes.map(r => ({ id: r, name: r }))}
            />
          )}
          {volatilities.length > 0 && (
            <Select
              label="Volatility"
              value={filterVolatility}
              onChange={(e) => setFilterVolatility(e.target.value)}
              options={volatilities.map(v => ({ id: v, name: v }))}
            />
          )}
        </div>
        <div className="mt-4">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              setFilterMarket('')
              setFilterDay('')
              setFilterSession('')
              setFilterRegime('')
              setFilterVolatility('')
            }}
          >
            Clear All Filters
          </Button>
        </div>
      </Card>

      {/* Summary Card */}
      <Card className="bg-blue-50">
        <p className="text-sm text-blue-700">
          <strong>Showing {filteredStrategies.length} strategies</strong>
          {filterMarket && ` • Market: ${filterMarket.toUpperCase()}`}
          {filterDay && ` • Day: ${filterDay}`}
          {filterSession && ` • Session: ${filterSession}`}
          {filterRegime && ` • Regime: ${filterRegime}`}
          {filterVolatility && ` • Volatility: ${filterVolatility}`}
        </p>
      </Card>

      {/* Rankings Table */}
      {filteredStrategies.length > 0 ? (
        <Card className="overflow-hidden">
          <Table
            columns={columns}
            data={filteredStrategies}
            loading={loading}
          />
        </Card>
      ) : (
        <EmptyState
          title="No Strategies Found"
          description="Try adjusting your filters to see strategy rankings"
        />
      )}

      {/* Export Section */}
      {filteredStrategies.length > 0 && (
        <div className="flex gap-3 items-center">
          <Card className="flex-1 bg-green-50 p-4">
            <p className="text-sm text-green-700 mb-3">
              💾 Export {filteredStrategies.length} strategies as CSV or JSON for your analysis
            </p>
            <div className="flex gap-2">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => exportRanking(filteredStrategies, 'csv')}
              >
                📥 Export CSV
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => exportRanking(filteredStrategies, 'json')}
              >
                📥 Export JSON
              </Button>
            </div>
          </Card>
        </div>
      )}

      {/* Key Metrics explanation */}
      <Card className="bg-gray-50">
        <h3 className="font-semibold text-gray-900 mb-3">📈 Key Metrics Explained</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-700">
          <div>
            <p><strong>Win Rate:</strong> Percentage of profitable trades</p>
            <p><strong>Profit Factor:</strong> Gross profit ÷ Gross loss (higher is better)</p>
          </div>
          <div>
            <p><strong>Max Drawdown:</strong> Largest peak-to-trough decline (lower is better)</p>
            <p><strong>Avg P&L:</strong> Average profit/loss per trade</p>
          </div>
        </div>
      </Card>
    </div>
  )
}

export default StrategyRanking
