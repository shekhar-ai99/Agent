import { useState, useMemo } from 'react'
import { Select } from '../common/index'
import { MARKET_CONFIG, TIMEFRAMES, RUN_MODES, getInstruments, getExchanges } from '../../utils/config'

export function MarketSelector({ value, onChange }) {
  return (
    <Select
      label="Market"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      options={[
        { id: 'india', name: 'India' },
        { id: 'crypto', name: 'Cryptocurrencies' },
      ]}
      required
    />
  )
}

export function ExchangeSelector({ market, value, onChange }) {
  const exchanges = useMemo(() => getExchanges(market), [market])

  return (
    <Select
      label="Exchange"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      options={exchanges}
      disabled={!market}
      required
    />
  )
}

export function InstrumentSelector({ market, exchange, value, onChange }) {
  const instruments = useMemo(() => getInstruments(market, exchange), [market, exchange])

  return (
    <Select
      label="Instrument"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      options={instruments}
      disabled={!market || !exchange}
      required
    />
  )
}

export function TimeframeSelector({ value, onChange }) {
  return (
    <Select
      label="Timeframe"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      options={TIMEFRAMES}
      required
    />
  )
}

export function ModeSelector({ value, onChange }) {
  return (
    <Select
      label="Run Mode"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      options={RUN_MODES.filter(m => m.enabled)}
      required
    />
  )
}

// Combined Configuration Panel
export function ConfigurationPanel({ config, onChange }) {
  const handleChange = (field, value) => {
    onChange({ ...config, [field]: value })
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <MarketSelector 
          value={config.market} 
          onChange={(value) => {
            handleChange('market', value)
            // Reset dependent fields
            handleChange('exchange', '')
            handleChange('instrument', '')
          }}
        />
        
        {config.market && (
          <ExchangeSelector
            market={config.market}
            value={config.exchange}
            onChange={(value) => {
              handleChange('exchange', value)
              handleChange('instrument', '')
            }}
          />
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {config.market && config.exchange && (
          <InstrumentSelector
            market={config.market}
            exchange={config.exchange}
            value={config.instrument}
            onChange={(value) => handleChange('instrument', value)}
          />
        )}
        
        <TimeframeSelector
          value={config.timeframe}
          onChange={(value) => handleChange('timeframe', value)}
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <ModeSelector
          value={config.mode}
          onChange={(value) => handleChange('mode', value)}
        />
      </div>
    </div>
  )
}
