# TradingPlatform React UI - Complete Getting Started Guide

## 🎯 Overview

This is a **production-grade React Single Page Application (SPA)** for:
- ✅ Configuring backtests and simulations
- ✅ Running strategy backtests
- ✅ Analyzing trade performance
- ✅ Comparing strategy rankings
- ✅ Deep-diving into trade details

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Frontend Setup](#frontend-setup)
4. [Backend Setup (Stub)](#backend-setup-stub)
5. [Full System Test](#full-system-test)
6. [Project Structure](#project-structure)
7. [Feature Overview](#feature-overview)
8. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Frontend
- Node.js 16.x or higher
- npm 8.x or higher

### Backend (Optional - for stub server)
- Python 3.8+
- pip

### Browsers Supported
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

---

## 🚀 Quick Start

### Option A: Frontend Only (with stub backend)

```bash
# Terminal 1: Start the backend stub
cd TradingPlatform/ui
pip install -r requirements.txt
python fastapi_stub.py

# Terminal 2: Start the frontend
cd TradingPlatform/ui
npm install
npm run dev
```

Then open: **http://localhost:3000**

### Option B: Frontend Only (with existing FastAPI backend)

Edit `.env.local`:
```
VITE_API_URL=http://your-api-server:8000/api
```

Then:
```bash
cd TradingPlatform/ui
npm install
npm run dev
```

---

## 📦 Frontend Setup

### Step 1: Install Dependencies

```bash
cd TradingPlatform/ui
npm install
```

This installs:
- React & React Router (routing)
- Tailwind CSS (styling)
- Chart.js (charting)
- Axios (HTTP client)
- Vite (build tool)

### Step 2: Configure API Endpoint

Edit or create `.env.local`:

**Development:**
```bash
VITE_API_URL=http://localhost:8000/api
NODE_ENV=development
```

**Production:**
```bash
VITE_API_URL=https://api.yourserver.com
NODE_ENV=production
```

### Step 3: Start Development Server

```bash
npm run dev
```

Output:
```
  VITE v5.0.0  ready in 234 ms

  ➜  Local:   http://localhost:3000/
  ➜  press h to show help
```

### Step 4: Open in Browser

Navigate to **http://localhost:3000**

You should see the **Configuration Page** with:
- Market selector (India / Crypto)
- Exchange selector (NSE / BSE / Global)
- Instrument selector
- Timeframe selector
- Mode selector (Backtest / Simulation)
- Capital & risk inputs
- Run button

---

## 🔧 Backend Setup (Stub)

### Why Use the Stub?

The included stub server (`fastapi_stub.py`) provides:
- ✅ All required API endpoints
- ✅ Sample data generation
- ✅ CORS support for frontend
- ✅ Real-time status polling
- ✅ Complete response schemas

**Perfect for:** Testing the UI without a real backend

### Step 1: Install Dependencies

```bash
cd TradingPlatform/ui
pip install -r requirements.txt
```

Installs:
- FastAPI (web framework)
- Uvicorn (ASGI server)
- Pydantic (data validation)

### Step 2: Start Stub Server

```bash
python fastapi_stub.py
```

Output:
```
🚀 Starting TradingPlatform API...
📍 Available at: http://localhost:8000
📚 API Docs at: http://localhost:8000/docs
💬 ReDoc at: http://localhost:8000/redoc
```

### Step 3: Test API Endpoints

**In browser or curl:**

```bash
# Health check
curl http://localhost:8000/api/health

# Get strategy rankings
curl http://localhost:8000/api/results/strategy_ranking.json

# Get sample results
curl http://localhost:8000/api/results/run_12345
```

### Step 4: View API Documentation

Open: **http://localhost:8000/docs**

Interactive Swagger UI with:
- List of all endpoints
- Request/response examples
- Try-it-out feature

---

## ✅ Full System Test

### Test Scenario: Run a Backtest

#### Step 1: Open UI
```
http://localhost:3000
```

#### Step 2: Configure Backtest
1. Market: **India**
2. Exchange: **NSE**
3. Instrument: **NIFTY**
4. Timeframe: **5m**
5. Mode: **Backtest**
6. Capital: **100,000** (₹)
7. Risk: **2%**
8. Start Date: **2025-01-01**
9. End Date: **2025-12-31**

#### Step 3: Click "Run Backtest"
You should see:
- Spinner: "Running..."
- Status updates every 2 seconds
- After completion: redirected to `/results/{run_id}`

#### Step 4: View Results
Should display:
- **Metrics:** Total P&L, Return %, Trades, Win Rate, Drawdown, Sharpe
- **Charts:**
  - Equity curve (line chart)
  - Drawdown (bar chart)
  - Trades per day (bar chart)
  - P&L distribution (histogram)

#### Step 5: View Trade Details
Click "📋 View Trade Details" → `/trades/{run_id}`

Should show:
- Table of 40-100 trades with:
  - Trade ID, Strategy, Entry/Exit times and prices
  - Quantity, P&L, Regime, Volatility, Session, Day
- Filters: Strategy, Day, Session
- Sorting: Click column headers

#### Step 6: View Strategy Rankings
Click "⭐ View Rankings" → `/rankings`

Should show:
- Table of 3+ strategies ranked by performance
- Metrics: Win rate, Profit factor, Avg P&L, Max Drawdown
- Filters: Market, Day, Session, Regime, Volatility
- Info: "Rankings are derived from historical backtests"

---

## 📁 Project Structure

```
TradingPlatform/ui/
│
├── 📄 package.json              # Dependencies & scripts
├── 📄 vite.config.js            # Vite build config
├── 📄 tailwind.config.js        # Tailwind CSS config
├── 📄 postcss.config.js         # PostCSS config
├── 📄 index.html                # HTML entry point
├── 📄 .env.local                # Environment variables
├── 📄 requirements.txt          # Python dependencies
├── 📄 fastapi_stub.py           # Reference backend server
├── 📄 SETUP_GUIDE.md            # API integration guide
│
└── 📁 src/
    │
    ├── 📄 main.jsx              # React entry point
    ├── 📄 App.jsx               # Main app & routing
    │
    ├── 📁 api/
    │   └── 📄 client.js         # Axios client & API methods
    │
    ├── 📁 pages/                # Page components
    │   ├── 📄 RunConfig.jsx     # Configuration page (/)
    │   ├── 📄 RunResults.jsx    # Results page (/results/:runId)
    │   ├── 📄 TradeDetails.jsx  # Trades page (/trades/:runId)
    │   └── 📄 StrategyRanking.jsx # Rankings (/rankings)
    │
    ├── 📁 components/
    │   ├── 📁 common/
    │   │   ├── 📄 Layout.jsx
    │   │   └── 📄 index.jsx     # UI components
    │   ├── 📁 selectors/
    │   │   └── 📄 ConfigSelectors.jsx
    │   └── 📁 charts/
    │       └── 📄 Charts.jsx    # Chart.js components
    │
    ├── 📁 hooks/
    │   └── 📄 useApi.js         # React hooks for API
    │
    ├── 📁 utils/
    │   └── 📄 config.js         # Config & format utilities
    │
    └── 📁 styles/
        └── 📄 index.css         # Global Tailwind styles
```

---

## 🎨 Feature Overview

### Page 1: Configuration (`/`)

**Purpose:** Set up and execute a backtest or simulation

**Controls:**
- Market selector (India / Crypto)
- Exchange selector (NSE / BSE / Global)
- Instrument selector (20+ instruments)
- Timeframe selector (6 options)
- Mode selector (Backtest / Simulation)
- Capital input (₹)
- Risk per trade (%)
- Date range (backtest only)

**Actions:**
- Run Backtest → POST `/api/run/backtest`
- Run Simulation → POST `/api/run/simulation`
- Validation: Prevents invalid configs
- Error handling: Shows alerts

---

### Page 2: Results (`/results/:runId`)

**Purpose:** Visualize backtest performance

**Displays:**
- **Status:** Running / Completed / Failed
- **Metrics Cards:** P&L, Return %, Trades, Win Rate, Drawdown, Sharpe
- **Charts:** Equity curve, Drawdown, Trades/day, P&L distribution
- **Navigation:** Links to Trades & Rankings pages

**Data Source:**
- GET `/api/results/{run_id}`

---

### Page 3: Trade Details (`/trades/:runId`)

**Purpose:** Analyze individual trades

**Table Columns:**
| Column | Content |
|--------|---------|
| Trade ID | Sequential ID |
| Strategy | Strategy name (badge) |
| Entry Time | YYYY-MM-DD HH:MM |
| Entry Price | Price at entry |
| Exit Time | YYYY-MM-DD HH:MM |
| Exit Price | Price at exit |
| Quantity | Number of units |
| P&L | Profit/Loss (₹) |
| Regime | TRENDING / RANGING / VOLATILE |
| Volatility | LOW / MEDIUM / HIGH |
| Session | MORNING / MIDDAY / AFTERNOON |
| Day | Day of week |

**Features:**
- ✅ Sortable columns
- ✅ Filter by strategy, day, session
- ✅ Color-coded wins/losses
- ✅ Total trade count summary

**Data Source:**
- GET `/api/results/{run_id}/trades`

---

### Page 4: Strategy Rankings (`/rankings`)

**Purpose:** Compare strategy performance across conditions

**Table Columns:**
| Column | Content |
|--------|---------|
| Rank | 1 (🥇) / 2 (🥈) / 3 (🥉) / ... |
| Strategy | Strategy name |
| Win Rate | % of profitable trades |
| Profit Factor | Gross profit ÷ Gross loss |
| Avg P&L | Average per trade (₹) |
| Total P&L | Total cumulative (₹) |
| Max Drawdown | Largest decline (%) |
| Trades | Total count |

**Filters:**
- Market (India / Crypto)
- Day of week (Mon-Fri)
- Session (Morning / Midday / Afternoon)
- Regime (Trending / Ranging / Volatile)
- Volatility (Low / Medium / High)

**Data Source:**
- GET `/api/results/strategy_ranking.json`

---

## 🎛️ Configuration Options

### Markets & Instruments

**India - NSE:**
- NIFTY (NIFTY 50)
- BANKNIFTY
- FINNIFTY
- MIDCPNIFTY
- NIFTYNXT50
- NIFTY100

**India - BSE:**
- SENSEX
- BANKEX

**Crypto - Global:**
- BTCUSD (Bitcoin)
- ETHUSD (Ethereum)
- BNBUSD (Binance Coin)
- SOLUSD (Solana)
- XAUUSD (Gold)

### Timeframes

- 1m (1 minute)
- 3m (3 minutes)
- 5m (5 minutes) ← Default
- 15m (15 minutes)
- 1h (1 hour)
- daily (Daily closes)

### Run Modes

| Mode | Data | Orders | Status |
|------|------|--------|--------|
| **Backtest** | Historical | Simulated | ✅ Available |
| **Simulation** | Live | Paper trades | ✅ Available |
| **Live** | Live | Real money | 🔄 Coming soon |

---

## 📊 Components & Hooks

### Reusable Components

```jsx
// Button variants
<Button variant="primary|secondary|outline|danger|success" />

// Card container
<Card>Content</Card>

// Metric display
<MetricCard label="P&L" value="15000" unit="₹" icon="📈" />

// Form inputs
<Input label="Capital" type="number" onChange={...} />
<Select label="Market" options={[...]} onChange={...} />

// Data display
<Table columns={cols} data={rows} loading={false} />
<Badge variant="success">TRENDING</Badge>

// States
<Spinner /> {/* Loading indicator */}
<Alert type="error" message="Failed" />
<EmptyState title="No data" description="Try again" />
```

### Custom Hooks

```jsx
// Fetch data with loading/error
const { data, loading, error, refetch } = useFetch(fetchFn)

// Run backtest
const { run, loading, error, runId } = useRunBacktest()
await run(config) → navigate to results

// Run simulation
const { run, loading, error, runId } = useRunSimulation()
await run(config) → navigate to results

// Poll status updates every N ms
const { status, progress } = usePollRunStatus(runId, 2000)
```

---

## 🔌 API Integration Points

### What the UI Expects

The following FastAPI endpoints must exist:

```python
# Run endpoints
POST /api/run/backtest      → Start backtest
POST /api/run/simulation    → Start simulation

# Status check
GET /api/results/{run_id}/status

# Results
GET /api/results/{run_id}            → Full results + charts
GET /api/results/{run_id}/trades     → Trade details
GET /api/results/strategy_ranking.json → Strategy rankings
```

### Response Formats

**GET /api/results/{run_id}** must return:
```json
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
  "equity_curve": [{"date": "...", "equity": 100000}, ...],
  "drawdown_curve": [{"date": "...", "drawdown": 0}, ...],
  "trades_per_day": [{"day": "...", "count": 5}, ...],
  "pnl_trades": [{"pnl": 1000}, ...],
  "trades": [...]
}
```

See `SETUP_GUIDE.md` for exact schemas.

---

## 🐛 Troubleshooting

### Frontend Issues

**Q: Blank page after `npm run dev`**
```
A: Check:
1. http://localhost:3000 is the correct URL
2. npm run dev completed without errors
3. Browser console (F12) for errors
```

**Q: API calls failing (CORS error)**
```
A: Backend must send CORS headers:
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"])
```

**Q: Charts not rendering**
```
A: Check:
1. Chart.js is installed: npm list react-chartjs-2
2. No console errors in DevTools (F12)
3. Data includes at least 2 points
```

**Q: Can't select instruments**
```
A: Verify:
1. Market is selected
2. Exchange is selected (should auto-populate instruments)
3. Check browser console for errors
```

### Backend Issues

**Q: 404 errors on API calls**
```
A: Check:
1. Backend running: http://localhost:8000/docs
2. VITE_API_URL matches backend URL
3. API endpoint paths match (include /api prefix)
```

**Q: 500 errors**
```
A: Check backend logs:
python fastapi_stub.py
Look for exception traceback
```

**Q: Status not updating during backtest**
```
A: Check:
1. usePollRunStatus hook is polling (/api/results/{runId}/status)
2. Backend returns status: "running" → "completed"
3. Network tab shows requests every 2 seconds
```

### Performance Issues

**Q: Slow to load results**
```
A: Optimize:
1. Use pagination for large trade tables (>1000 trades)
2. Limit equity curve points (show every Nth bar)
3. Lazy load charts
```

**Q: High CPU usage**
```
A: Check:
1. Are filters recalculating unnecessarily? (add useMemo)
2. Are charts re-rendering? (useMemo for data)
3. Browser DevTools → Rendering → Enable paint flashing
```

---

## 📦 Build for Production

### Create optimized build

```bash
npm run build
```

Output: `dist/` folder

### Deploy

```bash
# Copy dist/ to your web server
  
# Update VITE_API_URL in .env before building:
VITE_API_URL=https://api.yourserver.com
npm run build
```

### Verify

```bash
npm run preview
# Opens http://localhost:4173 with production build
```

---

## 🚀 Next Steps

1. **Test the stub:** Run `python fastapi_stub.py`
2. **Start frontend:** Run `npm run dev`
3. **Test a backtest:** Follow "Full System Test" above
4. **Integrate with real backend:** Update API endpoints in SETUP_GUIDE.md
5. **Deploy:** Build and serve to production

---

## 💡 Tips & Tricks

### Development Tips

1. **Hot reload:** Edit any `.jsx` or `.css` file → auto-refreshes
2. **DevTools:** F12 → Network tab to see API calls
3. **Redux DevTools:** Install Redux DevTools extension (optional)
4. **Lighthouse:** F12 → Lighthouse tab for performance audit

### Customization

1. **Change colors:** Edit `tailwind.config.js`
2. **Add filters:** Edit `StrategyRanking.jsx`
3. **Change date format:** Edit `utils/config.js` formatDate()
4. **Add new pages:** Create page in `src/pages/`, add route in `App.jsx`

### Performance

1. **Code splitting:** Vite does this automatically
2. **Lazy routes:** Use React.lazy() for heavy pages
3. **Image optimization:** Use WebP format
4. **Caching:** Cache API responses in localStorage

---

## 🎓 Learning Resources

- [React Docs](https://react.dev)
- [React Router](https://reactrouter.com)
- [Tailwind CSS](https://tailwindcss.com)
- [Chart.js](https://www.chartjs.org)
- [Axios](https://axios-http.com)
- [Vite](https://vite.dev)

---

## 📈 Roadmap

**Coming Soon:**
- [ ] Real-time live trading dashboard
- [ ] Custom strategy builder
- [ ] Historical backtest archive browser
- [ ] Multi-run comparison view
- [ ] Export reports to PDF
- [ ] Dark mode
- [ ] Mobile app (React Native)

---

## 📝 License

TradingPlatform UI - 2026

---

**Questions?** Check `SETUP_GUIDE.md` for API details or review component code in `src/`.

**Ready to start?** Run:
```bash
npm install && npm run dev
```

Enjoy your TradingPlatform! 🎉
