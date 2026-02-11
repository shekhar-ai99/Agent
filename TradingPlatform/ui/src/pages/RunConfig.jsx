import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Card, Button, Input, Alert, Spinner } from '../components/common/index'
import { ConfigurationPanel } from '../components/selectors/ConfigSelectors'
import { useRunBacktest, useRunSimulation } from '../hooks/useApi'

function RunConfig() {
  const navigate = useNavigate()
  const { run: runBacktest, loading: backtestLoading, error: backtestError } = useRunBacktest()
  const { run: runSimulation, loading: simulationLoading, error: simulationError } = useRunSimulation()

  const [config, setConfig] = useState({
    market: '',
    exchange: '',
    instrument: '',
    timeframe: '',
    mode: '',
    capital: 100000,
    risk_per_trade: 2,
    start_date: '2025-01-01',
    end_date: '2025-12-31',
  })

  const [showDateFields, setShowDateFields] = useState(false)
  const loading = backtestLoading || simulationLoading
  const error = backtestError || simulationError

  const handleConfigChange = (newConfig) => {
    setConfig(newConfig)
    setShowDateFields(newConfig.mode === 'backtest')
  }

  const validateConfig = () => {
    if (!config.market || !config.exchange || !config.instrument || !config.timeframe || !config.mode) {
      return false
    }
    if (config.mode === 'backtest' && (!config.start_date || !config.end_date)) {
      return false
    }
    if (config.capital <= 0 || config.risk_per_trade <= 0 || config.risk_per_trade > 100) {
      return false
    }
    return true
  }

  const handleRun = async () => {
    if (!validateConfig()) {
      alert('Please fill in all required fields correctly')
      return
    }

    try {
      let result
      if (config.mode === 'backtest') {
        result = await runBacktest(config)
      } else {
        result = await runSimulation(config)
      }

      if (result?.run_id) {
        navigate(`/results/${result.run_id}`)
      }
    } catch (err) {
      console.error('Run failed:', err)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Configure Backtest / Simulation</h1>
        <p className="mt-2 text-gray-600">
          Select your market parameters and run a backtest or simulation
        </p>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert 
          type="error" 
          message={error}
          onClose={() => setConfig(config)}
        />
      )}

      {/* Form Card */}
      <Card className="space-y-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Market Configuration</h2>
          <ConfigurationPanel config={config} onChange={handleConfigChange} />
        </div>

        <hr className="border-gray-200" />

        {/* Parameters */}
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Strategy Parameters</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Input
              label="Initial Capital (₹)"
              type="number"
              value={config.capital}
              onChange={(e) => setConfig({ ...config, capital: parseFloat(e.target.value) })}
              placeholder="100000"
              required
            />
            <Input
              label="Risk per Trade (%)"
              type="number"
              value={config.risk_per_trade}
              onChange={(e) => setConfig({ ...config, risk_per_trade: parseFloat(e.target.value) })}
              placeholder="2"
              required
            />
          </div>
        </div>

        {/* Backtest Date Range */}
        {showDateFields && config.mode === 'backtest' && (
          <>
            <hr className="border-gray-200" />
            <div>
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Backtest Period</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Input
                  label="Start Date"
                  type="date"
                  value={config.start_date}
                  onChange={(e) => setConfig({ ...config, start_date: e.target.value })}
                  required
                />
                <Input
                  label="End Date"
                  type="date"
                  value={config.end_date}
                  onChange={(e) => setConfig({ ...config, end_date: e.target.value })}
                  required
                />
              </div>
            </div>
          </>
        )}

        {/* Action Buttons */}
        <div className="flex gap-3 pt-6">
          <Button
            variant="primary"
            size="lg"
            onClick={handleRun}
            disabled={loading}
            className="flex-1"
          >
            {loading ? (
              <>
                <Spinner />
                <span className="ml-2">Running...</span>
              </>
            ) : (
              `Run ${config.mode === 'backtest' ? 'Backtest' : 'Simulation'}`
            )}
          </Button>
          <Button
            variant="secondary"
            size="lg"
            onClick={() => setConfig({
              market: '',
              exchange: '',
              instrument: '',
              timeframe: '',
              mode: '',
              capital: 100000,
              risk_per_trade: 2,
              start_date: '2025-01-01',
              end_date: '2025-12-31',
            })}
            disabled={loading}
          >
            Reset
          </Button>
        </div>
      </Card>

      {/* Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-blue-50">
          <h3 className="font-semibold text-blue-900">💡 Backtest</h3>
          <p className="text-sm text-blue-700 mt-2">
            Run strategy on historical data to analyze performance
          </p>
        </Card>
        <Card className="bg-green-50">
          <h3 className="font-semibold text-green-900">📊 Simulation</h3>
          <p className="text-sm text-green-700 mt-2">
            Execute your strategy on live market data with paper trades
          </p>
        </Card>
        <Card className="bg-purple-50">
          <h3 className="font-semibold text-purple-900">🎯 Live</h3>
          <p className="text-sm text-purple-700 mt-2">
            Execute real trades with actual capital (coming soon)
          </p>
        </Card>
      </div>
    </div>
  )
}

export default RunConfig
