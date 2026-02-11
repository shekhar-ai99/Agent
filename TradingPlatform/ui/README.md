# TradingPlatform UI

A production-grade web interface for configuring, executing, and analyzing trading backtests and simulations.

## 🎯 Features

- **Market & Instrument Selection**: Choose between India (NSE/BSE) and Crypto markets with dynamic instrument loading
- **Flexible Run Configuration**: Support for backtesting, paper trading, and live simulation modes
- **Real-time Results Dashboard**: Equity curves, drawdown analysis, and performance metrics
- **Trade-level Analysis**: Drill down into individual trades with filters and sorting
- **Strategy Ranking**: Comparative performance analysis across all tested strategies
- **Interactive Charts**: Chart.js-powered visualizations (equity curve, PnL distribution, etc.)
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Demo Mode**: Pre-loaded sample data for UI exploration without a backend

## 📁 File Structure

```
TradingPlatform/ui/
├── index.html              # Main SPA (Single Page Application)
├── README.md               # This file
└── assets/
    ├── css/
    │   └── style.css       # All styles (Tailwind-inspired custom CSS)
    └── js/
        ├── api.js          # Backend API client
        ├── charts.js       # Chart.js initialization & helpers
        └── app.js          # Main application logic & state management
```

## 🚀 Quick Start

### 1. Open the UI

```bash
# From the TradingPlatform directory
cd ui
open index.html  # macOS
# or
xdg-open index.html  # Linux
# or double-click in Windows Explorer
```

Or serve it with a local server:

```bash
cd /workspaces/Agent/TradingPlatform/ui
python3 -m http.server 8000
# Then open http://localhost:8000
```

### 2. Basic Workflow

1. **Configuration Tab**: Select market, exchange, instrument, timeframe, and run mode
2. **Submit Run**: Click "Submit Run" to send to backend OR use "Load Demo Results" for testing
3. **Results Tab**: View equity curves, metrics, and summary statistics
4. **Trade Details Tab**: Drill down into individual trades with filters
5. **Strategy Ranking Tab**: Compare strategy performance

### 3. Demo Mode

Without a backend, use the **Demo Mode** button to load sample data:
- 60 days of backtest data
- 45 sample trades across 4 strategies
- Complete equity curve and metrics

Perfect for UI testing and exploration!

## 🔌 Backend Integration

The UI communicates with the backend via REST-style JSON endpoints. All calls are handled in `assets/js/api.js`.

### Required Endpoints

#### 1. Submit Backtest
```
POST /api/run/backtest
Request Body:
{
  "market": "india|crypto",
  "exchange": "nse|bse|global",
  "instrument": "NIFTY50|etc",
  "timeframe": "1m|5m|15m|1h|1d",
  "mode": "backtest",
  "capital": 100000,
  "risk_per_trade": 2.0,
  "start_date": "2025-01-01",
  "end_date": "2025-03-10"
}

Response:
{
  "run_id": "run-123",
  "status": "completed|running|failed",
  "total_trades": 45,
  "win_rate": 62.5,
  "profit_factor": 2.15,
  "gross_profit": 25000,
  "gross_loss": 11000,
  "max_drawdown": 5.2,
  "sharpe_ratio": 1.45,
  "equity_curve": [{date, equity}, ...],
  "trades": [{trade data}, ...]
}
```

#### 2. Submit Simulation
```
POST /api/run/simulation
(Same request/response schema as backtest)
```

#### 3. Get Run Results
```
GET /api/results/{run_id}
Response: (Same as backtest response above)
```

#### 4. Get Trade Details
```
GET /api/results/{run_id}/trades
Response:
[
  {
    "trade_id": 1,
    "strategy": "EMACrossover",
    "entry_time": "2025-01-15T10:30:00",
    "entry_price": 23150.50,
    "exit_time": "2025-01-15T11:45:00",
    "exit_price": 23200.75,
    "quantity": 50,
    "pnl": 2512.50,
    "regime": "TRENDING",
    "volatility": "MEDIUM",
    "session": "MORNING"
  },
  ...
]
```

#### 5. Get Strategy Ranking
```
GET /api/results/strategy_ranking.json
Response:
[
  {
    "strategy_name": "EMACrossover",
    "rank": 1,
    "trades_count": 15,
    "win_rate": 73.3,
    "profit_factor": 2.8,
    "gross_profit": 18000,
    "gross_loss": 6500,
    "avg_pnl": 520.5,
    "max_pnl": 2000,
    "min_pnl": -1200
  },
  ...
]
```

### Customizing API Base URL

In `assets/js/api.js`, update the `BASE_URL`:

```javascript
const API = {
  BASE_URL: 'http://your-backend-url:5000/api',
  // or dynamically detect:
  // BASE_URL: window.location.origin.includes('localhost') 
  //   ? 'http://localhost:5000/api'
  //   : 'https://api.tradingplatform.com/api',
};
```

## 🎨 UI Components

### Navigation

- **Configuration**: Set up backtest parameters
- **Results**: View summary metrics and charts
- **Trade Details**: Analyze individual trades
- **Strategy Ranking**: Compare strategy performance

### Pages/Sections

#### Configuration Page
- Market selector (India / Crypto)
- Exchange selector (dynamic based on market)
- Instrument selector (dynamic based on exchange)
- Timeframe selector (1m, 5m, 15m, 1h, daily)
- Run mode selector (Backtest / Simulation / Live)
- Capital input
- Risk per trade (%)
- Date range (for backtests)

#### Results Page
- Run status with live indicator
- 6 metric cards: Trades, Win Rate, Profit Factor, Gross Profit, Max Drawdown, Sharpe
- 4 Charts:
  - Equity Curve (line chart with cumulative P&L)
  - Trade Distribution (doughnut: Wins vs Losses)
  - PnL Distribution (histogram of trade outcomes)
  - Trades Per Day (bar chart)

#### Trade Details Page
- Strategy filter dropdown
- Sortable table with 10 columns
- Click rows for more information
- Real-time trade count display

#### Strategy Ranking Page
- Filter by market, volatility, regime (extensible)
- Ranked table with 9 columns
- Top 10 strategies by profit factor
- Bar chart visualization

## 💻 Technology Stack

- **HTML5**: Semantic markup
- **CSS**: Custom stylesheet (no framework dependency, but Tailwind-inspired design)
- **JavaScript (ES6+)**: Vanilla JS, no jQuery or heavy frameworks
- **Chart.js**: Lightweight charting library (via CDN)
- **Responsive**: Mobile-first, works on all devices

## 🔧 Customization

### Change Color Scheme

Edit CSS variables in `assets/css/style.css`:

```css
:root {
  --primary: #2563eb;        /* Blue */
  --secondary: #10b981;      /* Green */
  --danger: #ef4444;         /* Red */
  --success: #10b981;        /* Green */
  --warning: #f59e0b;        /* Amber */
  /* ... etc ... */
}
```

### Add New Chart Types

In `assets/js/charts.js`, follow the pattern:

```javascript
Charts.initMyChart = function(containerId, data) {
  const ctx = document.getElementById(containerId);
  if (!ctx) return;

  if (this.instances[containerId]) {
    this.instances[containerId].destroy();
  }

  this.instances[containerId] = new Chart(ctx, {
    type: 'line', // or 'bar', 'doughnut', etc.
    data: { /* ... */ },
    options: { /* ... */ }
  });
};
```

### Extend API Client

In `assets/js/api.js`:

```javascript
API.myNewMethod = async function(param1, param2) {
  return this.request('/my/endpoint', {
    method: 'POST',
    body: JSON.stringify({ param1, param2 })
  });
};
```

### Add New Navigation Section

1. Add button in `<nav>`:
   ```html
   <button data-section="mynewsection">My Section</button>
   ```

2. Add section HTML:
   ```html
   <section id="section-mynewsection" class="section">
     <!-- ... -->
   </section>
   ```

3. Add navigation logic in `assets/js/app.js`:
   ```javascript
   case 'mynewsection':
     // load data or initialize section
     break;
   ```

## 📊 Test Data Generators

### Generate Demo Results

Pre-built function in `App.generateDemoResults()`:
- 60 days of daily equity curve
- 45 sample trades
- 4 different strategies
- Random P&L within realistic bounds

### Load from JSON File

Place a `results.json` file in `/data/`:

```javascript
const results = await API.loadLocalResults('results.json');
App.state.runResults = results;
App.renderResults();
```

## 🐛 Debugging

### Enable Console Logging

All API calls and state changes are logged to browser console.

```javascript
console.log('API request:', endpoint);
console.log('State update:', App.state);
```

### Check Browser DevTools

- **Network Tab**: Monitor API requests
- **Console Tab**: View logs and errors
- **Application Tab**: Inspect local storage / session storage

### Common Issues

| Issue | Solution |
|-------|----------|
| CORS errors | Ensure backend includes `Access-Control-Allow-Origin` headers |
| 404 Not Found | Check `API.BASE_URL` matches backend URL |
| No data showing | Use "Load Demo Results" to test UI independently |
| Charts not rendering | Ensure Canvas IDs match between HTML and JavaScript |

## 📱 Responsive Design

The UI is fully responsive with breakpoints at:
- **768px**: Tablet layout adjustments
- **480px**: Mobile layout adjustments

Tested on:
- Desktop (Chrome, Firefox, Safari)
- iPad (iOS)
- iPhone (iOS)
- Android devices

## 📝 State Management

All application state is managed in `App.state`:

```javascript
App.state = {
  currentSection: 'config',     // Current tab
  currentRun: null,              // Current run ID
  runResults: null,              // Full results object
  allTrades: null,               // Trades array
  strategyRanking: null,         // Strategy rankings
};
```

State is updated via methods like:
- `App.submitRun()`
- `App.renderResults()`
- `App.showSection()`

## 🔐 Security Notes

- **No sensitive data storage**: Credentials are not stored in localStorage
- **CORS protected**: Frontend assumes backend enforces CORS
- **Input validation**: Form inputs are validated before submission
- **XSS protection**: Chart.js and DOM methods protect against injection

## 📚 Resources

- [Chart.js Documentation](https://www.chartjs.org/docs/latest/)
- [MDN Web Docs](https://developer.mozilla.org/)
- [JSON Schema](https://json-schema.org/)

## 🤝 Contributing

To add features:

1. Keep CSS in `style.css` (maintain single-source-of-truth)
2. Keep API logic in `api.js`
3. Keep chart code in `charts.js`
4. Keep app logic in `app.js`
5. Test in Demo Mode first
6. Maintain responsive layout

## 📄 License

Part of TradingPlatform. See main repo for license information.

---

**Last Updated**: February 10, 2025
**Status**: Production Ready
**Tested On**: Chrome, Firefox, Safari, Edge
