# Trading Bot Phase 1

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your credentials.
3. Run:
   ```
   python main.py
   ```
4. Open http://localhost:5000 in your browser.

## Features
- Simulator mode with EMA Crossover & RSI+MACD strategies.
- Offline historical data support (3min, 5min, 15min CSV).
- Real-time analytics: PnL, trades, capital used, entry price, status.
- Logs actions to `logs/trading.log`.
