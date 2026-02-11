# TradingPlatform React UI - Setup & API Guide

## Quick Start

```bash
cd TradingPlatform/ui

# Install dependencies
npm install

# Start development server
npm run dev

# Navigate to http://localhost:3000
```

## Backend API Endpoints Required

The UI expects these FastAPI endpoints to be available:

### 1. Run Endpoints

**POST /api/run/backtest**
```json
Request:
{
  "market": "india",
  "exchange": "nse",
  "instrument": "NIFTY",
  "timeframe": "5m",
  "capital": 100000,
  "risk_per_trade": 2,
  "start_date": "2025-01-01",
  "end_date": "2025-12-31"
}

Response:
{
  "run_id": "run_20250211_123456",
  "message": "Backtest started"
}
```

**POST /api/run/simulation**
```json
Request:
{
  "market": "india",
  "exchange": "nse",
  "instrument": "NIFTY",
  "timeframe": "5m",
  "capital": 100000,
  "risk_per_trade": 2
}

Response:
{
  "run_id": "sim_20250211_123456",
  "message": "Simulation started"
}
```

### 2. Status Endpoint

**GET /api/results/{run_id}/status**
```json
Response:
{
  "run_id": "run_20250211_123456",
  "status": "running|completed|failed",
  "progress": 45
}
```

### 3. Results Endpoint

**GET /api/results/{run_id}**
```json
Response:
{
  "run_id": "run_20250211_123456",
  "status": "completed",
  "total_pnl": 15000,
  "net_pnl": 15000,
  "return_pct": 15.0,
  "win_rate": 55.0,
  "num_trades": 100,
  "max_drawdown": -5.0,
  "sharpe_ratio": 1.8,
  "profit_factor": 1.6,
  "equity_curve": [
    {
      "date": "2025-01-01",
      "equity": 100000
    },
    {
      "date": "2025-01-02",
      "equity": 101500
    }
  ],
  "drawdown_curve": [
    {
      "date": "2025-01-01",
      "drawdown": 0
    }
  ],
  "trades_per_day": [
    {
      "day": "2025-01-01",
      "count": 5
    }
  ],
  "pnl_trades": [
    {
      "pnl": 1000
    },
    {
      "pnl": -500
    }
  ]
}
```

### 4. Trade Details Endpoint

**GET /api/results/{run_id}/trades**
```json
Response:
{
  "trades": [
    {
      "trade_id": 1,
      "strategy": "RSI_MeanReversion",
      "entry_time": "2025-01-01T10:00:00",
      "entry_price": 22500.0,
      "exit_time": "2025-01-01T11:00:00",
      "exit_price": 22600.0,
      "quantity": 50,
      "pnl": 5000.0,
      "regime": "TRENDING",
      "volatility": "HIGH",
      "session": "MORNING",
      "day": "Monday"
    }
  ]
}
```

### 5. Strategy Ranking Endpoint

**GET /api/results/strategy_ranking.json**
```json
Response:
{
  "strategies": [
    {
      "rank": 1,
      "name": "RSI_MeanReversion",
      "market": "india",
      "win_rate": 58.5,
      "profit_factor": 1.8,
      "avg_pnl": 450.0,
      "total_pnl": 54000.0,
      "max_drawdown": -4.2,
      "trades_count": 120,
      "days": ["Monday", "Tuesday"],
      "sessions": ["MORNING", "MIDDAY"],
      "regimes": ["TRENDING", "RANGING"],
      "volatilities": ["LOW", "MEDIUM", "HIGH"]
    }
  ]
}
```

## File Structure

```
ui/
├── index.html              # HTML entry point
├── package.json            # Dependencies
├── vite.config.js          # Vite config
├── tailwind.config.js      # Tailwind config
├── postcss.config.js       # PostCSS config
├── .env.local              # Environment variables
├── README.md               # Main documentation
│
└── src/
    ├── main.jsx            # React entry point
    ├── App.jsx             # Main app component & routing
    │
    ├── api/
    │   └── client.js       # Axios client & API wrapper
    │
    ├── pages/
    │   ├── RunConfig.jsx       # Configuration page (/)
    │   ├── RunResults.jsx      # Results page (/results/:runId)
    │   ├── TradeDetails.jsx    # Trades page (/trades/:runId)
    │   └── StrategyRanking.jsx # Rankings page (/rankings)
    │
    ├── components/
    │   ├── common/
    │   │   ├── Layout.jsx      # App header/footer layout
    │   │   └── index.jsx       # UI components (Button, Card, etc.)
    │   ├── selectors/
    │   │   └── ConfigSelectors.jsx  # Market/exchange/instrument dropdowns
    │   ├── charts/
    │   │   └── Charts.jsx      # Chart.js components
    │   └── tables/
    │       └── TradeTable.jsx  # (for future)
    │
    ├── hooks/
    │   └── useApi.js       # Custom hooks for API calls
    │
    ├── utils/
    │   └── config.js       # Config constants & format utilities
    │
    └── styles/
        └── index.css       # Global Tailwind styles
```

## Component API Reference

### Pages

#### RunConfig
- Location: `/`
- Purpose: Configure and execute backtests/simulations
- Props: None
- State: Market, exchange, instrument, timeframe, mode, capital, dates

#### RunResults
- Location: `/results/:runId`
- Purpose: Display backtest results and charts
- Props: `:runId` (URL param)
- Features: Equity chart, drawdown, metrics, trade distribution

#### TradeDetails
- Location: `/trades/:runId`
- Purpose: Detailed trade analysis with filters
- Props: `:runId` (URL param)
- Features: Sortable table, filtering by strategy/day/session

#### StrategyRanking
- Location: `/rankings`
- Purpose: Compare strategy performance
- Props: None
- Features: Multi-filter ranking, metrics breakdown

### Components

All components in `src/components/common/index.jsx`:

```jsx
<Button variant="primary" size="lg">Click me</Button>
<Card>Content</Card>
<MetricCard label="P&L" value="15000" unit="₹" icon="📈" />
<Input label="Capital" type="number" onChange={...} />
<Select label="Market" options={[...]} onChange={...} />
<Table columns={[...]} data={[...]} loading={false} />
<Badge variant="success">TRENDING</Badge>
<Alert type="error" message="Failed" />
<Spinner />
<EmptyState title="No data" description="Try again" />
```

### Custom Hooks

```jsx
// Generic fetch hook
const { data, loading, error, refetch } = useFetch(
  () => apiClient.getTrades(runId),
  [runId]
)

// Backtest runner
const { run, loading, error, runId } = useRunBacktest()
const result = await run(config)

// Simulation runner
const { run, loading, error, runId } = useRunSimulation()
const result = await run(config)

// Poll run status
const { status, progress } = usePollRunStatus(runId, 2000)
```

## Configuration

Edit `.env.local` to change:
```
VITE_API_URL=http://localhost:8000/api
NODE_ENV=development
```

For production:
```
VITE_API_URL=https://api.yourserver.com
NODE_ENV=production
```

## Build & Deploy

```bash
# Production build
npm run build

# Output in dist/
# Deploy dist/ folder to your web server
```

## Navigation Flow

```
/ (RunConfig)
  ↓ [Run Backtest/Simulation]
  ↓
/results/:runId (RunResults)
  ├→ [View Trade Details] → /trades/:runId (TradeDetails)
  └→ [View Rankings] → /rankings (StrategyRanking)

/rankings (StrategyRanking)
  ←→ [Back to Configure] → / (RunConfig)
```

## Error Handling

All API calls include error handling:
- Network errors → Show alert
- API errors → Display error message
- Validation errors → Highlight form field

## Performance Considerations

1. Charts render with reasonable data size (100-1000 points)
2. Trade tables paginate or virtualize for large datasets
3. Filters use useMemo to prevent unnecessary recalculations
4. API responses are cached in component state

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Debugging

Enable debug logging:
```jsx
// In api/client.js
client.interceptors.response.use(response => {
  console.log('API Response:', response)
  return response
})
```

Check browser DevTools:
- Network tab: API requests/responses
- Console: JavaScript errors
- Application: Stored state & cookies

## Next Steps

1. Implement backend FastAPI endpoints
2. Run `npm run dev` to start frontend
3. Test each page with sample data
4. Configure CORS on backend
5. Deploy to production

---

For questions, check the main README.md or create an issue.
