// Export utilities for CSV and JSON

export const exportToCsv = (data, headers, filename = 'export.csv') => {
  const escape = (value) => `"${String(value ?? '').replace(/"/g, '""')}"`
  
  const headerRow = headers.join(',')
  const rows = data.map(row => 
    headers.map(header => escape(row[header] ?? '')).join(',')
  )
  
  const csv = [headerRow, ...rows].join('\n')
  downloadFile(csv, filename, 'text/csv')
}

export const exportToJson = (data, filename = 'export.json') => {
  const json = JSON.stringify(data, null, 2)
  downloadFile(json, filename, 'application/json')
}

const downloadFile = (content, filename, mimeType) => {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

// Trade export
export const exportTrades = (trades, format = 'csv') => {
  const columns = ['trade_id', 'strategy', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'quantity', 'pnl', 'session', 'day']
  
  const data = trades.map((trade, idx) => ({
    trade_id: trade.trade_id || trade.id || idx + 1,
    strategy: trade.strategy || '--',
    entry_time: trade.entry_time || '--',
    entry_price: trade.entry_price || '--',
    exit_time: trade.exit_time || '--',
    exit_price: trade.exit_price || '--',
    quantity: trade.quantity || '--',
    pnl: trade.pnl || '--',
    session: trade.session || '--',
    day: trade.day || '--'
  }))
  
  if (format === 'csv') {
    exportToCsv(data, columns, 'trades_export.csv')
  } else {
    exportToJson(data, 'trades_export.json')
  }
}

// Strategy ranking export
export const exportRanking = (strategies, format = 'csv') => {
  const columns = ['rank', 'name', 'win_rate', 'profit_factor', 'avg_pnl', 'total_pnl', 'max_drawdown', 'trades_count']
  
  const data = strategies.map(s => ({
    rank: s.rank || '--',
    name: s.name || '--',
    win_rate: s.win_rate || '--',
    profit_factor: s.profit_factor || '--',
    avg_pnl: s.avg_pnl || '--',
    total_pnl: s.total_pnl || '--',
    max_drawdown: s.max_drawdown || '--',
    trades_count: s.trades_count || '--'
  }))
  
  if (format === 'csv') {
    exportToCsv(data, columns, 'strategy_ranking_export.csv')
  } else {
    exportToJson(data, 'strategy_ranking_export.json')
  }
}

// Comparison export
export const exportComparison = (runData, format = 'csv') => {
  const columns = ['run_id', 'instrument', 'mode', 'timeframe', 'net_pnl', 'win_rate', 'max_drawdown', 'sharpe_ratio']
  
  const data = runData.map(run => ({
    run_id: run.run_id || '--',
    instrument: run.instrument || '--',
    mode: run.mode || '--',
    timeframe: run.timeframe || '--',
    net_pnl: run.net_pnl || '--',
    win_rate: run.win_rate || '--',
    max_drawdown: run.max_drawdown || '--',
    sharpe_ratio: run.sharpe_ratio || '--'
  }))
  
  if (format === 'csv') {
    exportToCsv(data, columns, 'comparison_export.csv')
  } else {
    exportToJson(data, 'comparison_export.json')
  }
}
