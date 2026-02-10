# Datasets for Backtesting

This directory contains historical market data for backtesting strategies.

## Directory Structure

```
datasets/
├── nifty/
│   ├── 1min/          - 1-minute OHLCV data
│   ├── 5min/          - 5-minute OHLCV data
│   ├── 15min/         - 15-minute OHLCV data
│   └── daily/         - Daily OHLCV data
└── README.md          - This file
```

## Available Datasets

### NIFTY 50 Index

| Timeframe | File | Status | Rows |
|-----------|------|--------|------|
| 1 minute | `nifty/1min/nifty_historical_data_1min.csv` | ✅ Ready | Check locally |
| 5 minute | `nifty/5min/nifty_historical_data_5min.csv` | ✅ Ready | Check locally |
| 15 minute | `nifty/15min/nifty_historical_data_15min.csv` | ✅ Ready | Check locally |
| Daily | `nifty/daily/` | ⏳ Ready to add | - |

## CSV Format Expected

```
timestamp,open,high,low,close,volume
2024-01-01 09:15:00,19200.00,19300.00,19150.00,19250.00,1000000
2024-01-01 09:16:00,19250.00,19350.00,19200.00,19300.00,900000
...
```

## How to Use in Backtesting

### Load Data
```python
import pandas as pd
from pathlib import Path

# Load NIFTY 5min data
data_path = Path("datasets/nifty/5min/nifty_historical_data_5min.csv")
df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")

print(f"Loaded {len(df)} rows")
print(df.head())
```

### Backtest with TradingPlatform
```python
from simulation import PaperTradingEngine

# Create data source
data_source = {"NIFTY50": df}

# Run backtest
engine = PaperTradingEngine(data_source=data_source)
result = engine.run("2024-01-01", "2024-12-31")

print(f"Return: {result['total_return']:.2f}%")
print(f"Win Rate: {result['win_rate']:.1f}%")
```

### Walk-Forward Optimization
```python
from simulation import MultiMonthRunner

# Run walk-forward test
runner = MultiMonthRunner()
result = runner.run(
    data_source={"NIFTY50": df},
    train_months=3,
    test_months=1,
    step_months=1
)

print(f"Avg Return: {result['avg_return_per_iteration']:.2f}%")
```

## Adding New Datasets

1. **Create folder** for new market/timeframe:
   ```bash
   mkdir -p datasets/banknifty/5min
   ```

2. **Copy CSV file** with proper format:
   ```bash
   cp /path/to/data.csv datasets/banknifty/5min/
   ```

3. **Update this README** with new dataset info

## Data Preparation Checklist

When adding new CSV files:
- [ ] CSV has `timestamp` column (datetime format)
- [ ] CSV has OHLCV columns: `open`, `high`, `low`, `close`, `volume`
- [ ] Timestamp is sorted (oldest to newest)
- [ ] No missing data in OHLCV columns
- [ ] Timestamp index is unique (no duplicates)

## Adding Indicators

If your CSV already includes indicators:
```
timestamp,open,high,low,close,volume,sma20,sma50,rsi,atr,bb_upper,bb_lower
```

Otherwise, add them during data loading:
```python
import pandas as pd

df = pd.read_csv("data.csv", parse_dates=["timestamp"], index_col="timestamp")

# Calculate indicators
df['sma20'] = df['close'].rolling(20).mean()
df['sma50'] = df['close'].rolling(50).mean()

# RSI calculation
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# ATR calculation
tr = pd.concat([
    df['high'] - df['low'],
    abs(df['high'] - df['close'].shift()),
    abs(df['low'] - df['close'].shift()),
], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean()
```

## Data Statistics

Check your data quality:
```python
import pandas as pd

df = pd.read_csv("datasets/nifty/5min/nifty_historical_data_5min.csv")

print(f"Rows: {len(df)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
print(f"Missing OHLCV: {df[['open','high','low','close','volume']].isnull().sum().sum()}")
print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
```

## Tips for Best Results

1. **Use consistent timeframe** - Don't mix 5min and daily
2. **Ensure continuous data** - No gaps in timestamps
3. **Verify price ranges** - Check for data anomalies
4. **Include at least 3-6 months** - For reliable backtesting
5. **Separate train/test** - Use MultiMonthRunner for walk-forward

---

**Last Updated**: February 6, 2026
**Status**: Ready for backtesting
