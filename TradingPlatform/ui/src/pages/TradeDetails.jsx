import { useState, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { Table, Card, Button, Select, EmptyState } from '../components/common/index'
import { useFetch } from '../hooks/useApi'
import { exportTrades } from '../utils/export'
import apiClient from '../api/client'
import { formatCurrency, formatDateTime, formatTime, Badge } from '../utils/config'

function TradeDetails() {
  const { runId } = useParams()
  const { data: tradesData, loading } = useFetch(
    () => apiClient.getTrades(runId),
    [runId]
  )

  const [filterStrategy, setFilterStrategy] = useState('')
  const [filterDay, setFilterDay] = useState('')
  const [filterSession, setFilterSession] = useState('')
  const [sortField, setSortField] = useState('entry_time')
  const [sortOrder, setSortOrder] = useState('asc')

  const trades = tradesData?.trades || []

  // Extract unique values for filters
  const strategies = useMemo(
    () => [...new Set(trades.map(t => t.strategy))],
    [trades]
  )

  const days = useMemo(
    () => [...new Set(trades.map(t => t.day))],
    [trades]
  )

  const sessions = useMemo(
    () => [...new Set(trades.map(t => t.session))],
    [trades]
  )

  // Filter and sort trades
  const filteredTrades = useMemo(() => {
    let filtered = trades.filter(trade => {
      if (filterStrategy && trade.strategy !== filterStrategy) return false
      if (filterDay && trade.day !== filterDay) return false
      if (filterSession && trade.session !== filterSession) return false
      return true
    })

    // Sort
    filtered.sort((a, b) => {
      let aVal = a[sortField]
      let bVal = b[sortField]

      if (typeof aVal === 'string') {
        aVal = aVal.toLowerCase()
        bVal = bVal.toLowerCase()
      }

      if (aVal < bVal) return sortOrder === 'asc' ? -1 : 1
      if (aVal > bVal) return sortOrder === 'asc' ? 1 : -1
      return 0
    })

    return filtered
  }, [trades, filterStrategy, filterDay, filterSession, sortField, sortOrder])

  const columns = [
    {
      id: 'trade_id',
      label: 'Trade ID',
      render: (row) => <span className="font-mono text-sm">{row.trade_id}</span>
    },
    {
      id: 'strategy',
      label: 'Strategy',
      render: (row) => <Badge variant="primary">{row.strategy}</Badge>
    },
    {
      id: 'entry_time',
      label: 'Entry Time',
      render: (row) => formatDateTime(row.entry_time)
    },
    {
      id: 'entry_price',
      label: 'Entry Price',
      render: (row) => `${row.entry_price.toFixed(2)}`
    },
    {
      id: 'exit_time',
      label: 'Exit Time',
      render: (row) => formatDateTime(row.exit_time)
    },
    {
      id: 'exit_price',
      label: 'Exit Price',
      render: (row) => `${row.exit_price.toFixed(2)}`
    },
    {
      id: 'quantity',
      label: 'Quantity',
      render: (row) => row.quantity
    },
    {
      id: 'pnl',
      label: 'P&L',
      render: (row) => (
        <span className={row.pnl >= 0 ? 'text-green-600 font-bold' : 'text-red-600 font-bold'}>
          {formatCurrency(row.pnl)}
        </span>
      )
    },
    {
      id: 'regime',
      label: 'Regime',
      render: (row) => (
        <Badge variant={row.regime === 'TRENDING' ? 'success' : row.regime === 'RANGING' ? 'warning' : 'error'}>
          {row.regime}
        </Badge>
      )
    },
    {
      id: 'volatility',
      label: 'Volatility',
      render: (row) => (
        <Badge variant={row.volatility === 'LOW' ? 'success' : row.volatility === 'HIGH' ? 'error' : 'warning'}>
          {row.volatility}
        </Badge>
      )
    },
    {
      id: 'session',
      label: 'Session',
      render: (row) => row.session || '-'
    },
    {
      id: 'day',
      label: 'Day',
      render: (row) => row.day
    },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Trade Details</h1>
          <p className="mt-2 text-gray-600">Run ID: {runId}</p>
        </div>
        <Link to={`/results/${runId}`}>
          <Button variant="outline">← Back to Results</Button>
        </Link>
      </div>

      {/* Summary Card */}
      <Card className="bg-blue-50">
        <h3 className="font-semibold text-blue-900">Total Trades: {trades.length}</h3>
        <p className="text-sm text-blue-700 mt-1">
          Winning trades: {trades.filter(t => t.pnl > 0).length} | Losing trades: {trades.filter(t => t.pnl < 0).length}
        </p>
      </Card>

      {/* Filters */}
      <Card>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Filters & Sorting</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Select
            label="Strategy"
            value={filterStrategy}
            onChange={(e) => setFilterStrategy(e.target.value)}
            options={strategies.map(s => ({ id: s, name: s }))}
          />
          <Select
            label="Day"
            value={filterDay}
            onChange={(e) => setFilterDay(e.target.value)}
            options={days.map(d => ({ id: d, name: d }))}
          />
          <Select
            label="Session"
            value={filterSession}
            onChange={(e) => setFilterSession(e.target.value)}
            options={sessions.map(s => ({ id: s, name: s }))}
          />
          <div className="flex flex-col pt-6">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                setFilterStrategy('')
                setFilterDay('')
                setFilterSession('')
              }}
            >
              Clear Filters
            </Button>
          </div>
        </div>
      </Card>

      {/* Sorting Info */}
      <Card className="bg-gray-50 text-sm text-gray-600">
        <p>
          Showing <strong>{filteredTrades.length}</strong> trades
          {filterStrategy && ` (Strategy: ${filterStrategy})`}
          {filterDay && ` (Day: ${filterDay})`}
          {filterSession && ` (Session: ${filterSession})`}
        </p>
      </Card>

      {/* Trades Table */}
      {filteredTrades.length > 0 ? (
        <Card className="overflow-hidden">
          <Table
            columns={columns}
            data={filteredTrades}
            loading={loading}
          />
        </Card>
      ) : (
        <EmptyState
          title="No Trades Found"
          description="Try adjusting your filters"
        />
      )}

      {/* Export Options */}
      <div className="flex gap-3 items-center">
        <Card className="flex-1 bg-blue-50 p-4">
          <p className="text-sm text-blue-700 mb-3">
            💡 Export {filteredTrades.length} trades as CSV or JSON for further analysis
          </p>
          <div className="flex gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => exportTrades(filteredTrades, 'csv')}
              disabled={filteredTrades.length === 0}
            >
              📥 Export CSV
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => exportTrades(filteredTrades, 'json')}
              disabled={filteredTrades.length === 0}
            >
              📥 Export JSON
            </Button>
          </div>
        </Card>
      </div>
    </div>
  )
}

export default TradeDetails
