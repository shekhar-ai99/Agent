// Market and instrument definitions
export const MARKET_CONFIG = {
  india: {
    exchanges: {
      nse: {
        name: 'NSE (National Stock Exchange)',
        instruments: [
          { id: 'NIFTY', name: 'NIFTY 50', type: 'index' },
          { id: 'BANKNIFTY', name: 'BANK NIFTY', type: 'index' },
          { id: 'FINNIFTY', name: 'FIN NIFTY', type: 'index' },
          { id: 'MIDCPNIFTY', name: 'MIDCAP NIFTY', type: 'index' },
          { id: 'NIFTYNXT50', name: 'NIFTY NEXT 50', type: 'index' },
          { id: 'NIFTY100', name: 'NIFTY 100', type: 'index' },
        ]
      },
      bse: {
        name: 'BSE (Bombay Stock Exchange)',
        instruments: [
          { id: 'SENSEX', name: 'SENSEX', type: 'index' },
          { id: 'BANKEX', name: 'BANK NX', type: 'index' },
        ]
      }
    }
  },
  crypto: {
    exchanges: {
      global: {
        name: 'Global',
        instruments: [
          { id: 'BTCUSD', name: 'Bitcoin', type: 'crypto' },
          { id: 'ETHUSD', name: 'Ethereum', type: 'crypto' },
          { id: 'BNBUSD', name: 'Binance Coin', type: 'crypto' },
          { id: 'SOLUSD', name: 'Solana', type: 'crypto' },
          { id: 'XAUUSD', name: 'Gold', type: 'commodity' },
        ]
      }
    }
  }
}

export const TIMEFRAMES = [
  { id: '1m', label: '1 Minute', value: '1m' },
  { id: '3m', label: '3 Minutes', value: '3m' },
  { id: '5m', label: '5 Minutes', value: '5m' },
  { id: '15m', label: '15 Minutes', value: '15m' },
  { id: '1h', label: '1 Hour', value: '1h' },
  { id: 'daily', label: 'Daily', value: 'daily' },
]

export const RUN_MODES = [
  { id: 'backtest', label: 'Backtest', description: 'Historical data analysis', enabled: true },
  { id: 'simulation', label: 'Simulation', description: 'Live data with paper trades', enabled: true },
  { id: 'live', label: 'Live Trading', description: 'Real money trading', enabled: false },
]

export const SESSIONS = {
  india: [
    { id: 'morning', label: 'Morning (9:15-11:00)', duration: '1h 45m' },
    { id: 'midday', label: 'Mid-day (11:00-14:00)', duration: '3h' },
    { id: 'afternoon', label: 'Afternoon (14:00-15:30)', duration: '1h 30m' },
  ],
  crypto: [
    { id: 'all', label: '24/7 Trading', duration: '24h' },
  ]
}

// Format utilities
export const formatCurrency = (value, decimals = 2) => {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value)
}

export const formatPercent = (value, decimals = 2) => {
  return `${(value).toFixed(decimals)}%`
}

export const formatDate = (date) => {
  if (!date) return '-'
  const d = new Date(date)
  return d.toLocaleDateString('en-IN', { year: 'numeric', month: 'short', day: 'numeric' })
}

export const formatDateTime = (dateTime) => {
  if (!dateTime) return '-'
  const d = new Date(dateTime)
  return d.toLocaleString('en-IN', { 
    year: 'numeric', 
    month: 'short', 
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

export const formatTime = (time) => {
  if (!time) return '-'
  const d = new Date(time)
  return d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })
}

// Get instruments for selected exchange
export const getInstruments = (market, exchange) => {
  const marketConfig = MARKET_CONFIG[market]
  if (!marketConfig || !marketConfig.exchanges[exchange]) {
    return []
  }
  return marketConfig.exchanges[exchange].instruments
}

// Get exchanges for selected market
export const getExchanges = (market) => {
  const marketConfig = MARKET_CONFIG[market]
  if (!marketConfig) return []
  return Object.entries(marketConfig.exchanges).map(([key, value]) => ({
    id: key,
    name: value.name
  }))
}
