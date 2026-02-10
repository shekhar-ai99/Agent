# Trading Platform - Production-Grade Multi-Market System

A modular, extensible trading platform supporting multiple markets (Indian equities, cryptocurrency) with clean separation of concerns, configuration-driven behavior, and simulation-first design.

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                     TradingPlatform (main.py)                  │
│                  Orchestrates all components                    │
└─────────┬─────────────────────────────────────────────────────┘
          │
    ┌─────┴──────────────────────────────────────────────────┐
    │                                                          │
┌───▼────────────┐  ┌──────────────┐  ┌─────────────────┐  ┌──▼──────────────┐
│   DataFeed     │  │    Markets   │  │    Strategies   │  │  RiskManager    │
│                │  │              │  │                 │  │                 │
│ • OHLCV data   │  │ • IndianMkt  │  │ • RSIStrategy   │  │ • Position size │
│ • Indicators   │  │ • CryptoMkt  │  │ • MAStrategy    │  │ • Daily loss    │
│ • Signals      │  │ • Sessions   │  │ • BBStrategy    │  │ • Concurrent #  │
└────────────────┘  │ • Expiry     │  │ • Custom...     │  └─────────────────┘
                    └──────────────┘  └─────────────────┘
                           │                  │
                    ┌──────▼──────────────────▼──────────┐
                    │   ExecutionEngine (core logic)     │
                    │                                    │
                    │  1. Check market hours            │
                    │  2. Select best strategy          │
                    │  3. Generate signal               │
                    │  4. Validate with risk manager    │
                    │  5. Place order via broker        │
                    │  6. Manage positions              │
                    │  7. Log performance               │
                    └──────────┬───────────────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │    PaperBroker         │
                    │                        │
                    │ • Order management     │
                    │ • Position tracking    │
                    │ • P&L calculation      │
                    │ • Fill simulation      │
                    │ • Slippage injection   │
                    └────────────────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │ PerformanceTracker     │
                    │                        │
                    │ • Trade history        │
                    │ • Win rate tracking    │
                    │ • Profit factor        │
                    │ • Equity curve         │
                    │ • Max drawdown         │
                    └────────────────────────┘
```

## Module Structure

### `/core/` - Abstract Interfaces
**Responsibility**: Define contracts for all market-specific and strategy-specific implementations.

- **`base_strategy.py`**: Abstract strategy interface
  - Signal types: BUY, SELL, HOLD with confidence levels
  - Filtering: Market type, regime, volatility, session awareness
  - Context: Provides all data needed for signal generation
  
- **`base_market.py`**: Market-specific rules
  - Sessions: Trading hours and breaks
  - Expiry: Instrument expiry dates
  - Risk multipliers: SL/TP adjustments per regime/volatility
  
- **`base_broker.py`**: Order execution interface
  - Order lifecycle: PENDING → PLACED → FILLED → CLOSED
  - Position tracking with P&L
  - Support for market/limit orders with SL/TP
  
- **`base_risk.py`**: Risk controls
  - Position sizing based on account percentage
  - Daily loss limits
  - Concurrent trade limits
  
- **`base_selector.py`**: Strategy ranking
  - Performance table per market/regime/volatility/session
  - Win rate and profit factor tracking
  - Automatic strategy selection based on historical data

### `/markets/` - Market Implementations
**Responsibility**: Implement market-specific rules (sessions, expiry, risk multipliers).

- **`indian_market.py`**: NSE equity trading
  - Sessions: Pre-market (9:00-9:15), Morning (9:15-11:59), Midday (12:00-13:59), Afternoon (14:00-15:30), Post-market
  - Expiry: Last Thursday of month
  - Risk multipliers adjusted for trending/ranging/volatile regimes
  
- **`crypto_market.py`**: 24/7 cryptocurrency trading
  - No sessions, always open
  - No expiry
  - Higher volatility thresholds (20.0/35.0 vs 15.0/25.0)

### `/brokers/` - Broker Implementations
**Responsibility**: Simulate or execute orders.

- **`paper_broker.py`**: Simulation-only broker (NO REAL ORDERS)
  - All orders are simulated fills
  - Slippage injection (0-1% per config)
  - Supports instant/next-bar execution modes
  - Full position tracking with P&L calculation
  - Trailing stop-loss/take-profit support

### `/strategies/` - Trading Strategies
**Responsibility**: Generate BUY/SELL signals based on market data.

- **`example_strategies.py`**: Reference implementations
  - RSIStrategy: Mean reversion (RSI < 30 buy, > 70 sell)
  - MAStrategy: Trend following (Fast MA > Slow MA)
  - BollingerBandsStrategy: Range trading (bands mean reversion)
  
  All strategies inherit BaseStrategy and implement:
  ```python
  required_indicators() -> List[str]  # What indicators needed
  supports_market(market_type) -> bool  # Filter by market
  supports_regime(regime) -> bool  # Filter by market regime
  supports_volatility(volatility) -> bool  # Filter by volatility
  generate_signal(context) -> Signal  # Generate BUY/SELL/HOLD
  ```

### `/execution/` - Trading Logic Orchestrator
**Responsibility**: Coordinate strategy selection, risk validation, and order placement.

- **`execution_engine.py`**: Main trading loop
  - Flow: Market hours check → Strategy selection → Signal generation → Risk validation → Order placement
  - Handles position management and exit conditions
  - Logs all trades for performance tracking

### `/config/` - Configuration System
**Responsibility**: Centralize all parameters (no magic numbers in code).

- **`settings.py`**: PlatformConfig dataclass with 16+ parameters
  - Account settings (initial balance, risk limits)
  - Market selection
  - Strategy selection
  - Simulation dates
  - Broker parameters
  - Logging configuration
  
- **`default_config.yaml`**: Sample configuration template
  - Load with `PlatformConfig.load_from_yaml("path/to/config.yaml")`

### `/data/` - Data and Indicators
**Responsibility**: Provide OHLCV data and technical indicators.

- **`DataFeed`**: Load historical/live data (CSV, API, etc.)
- **`IndicatorCalculator`**: Compute RSI, SMA, ATR, Bollinger Bands, etc.
- Currently stubs - integrate with your data sources

### `/analytics/` - Performance Tracking
**Responsibility**: Log trades and compute performance metrics.

- **`PerformanceTracker`**: Track trades, calculate P&L, drawdown, win rate
- Integrates with StrategySelector for performance ranking

### `/simulation/` - Backtesting Engine
**Responsibility**: Run platform over historical data.

- **`PaperTradingEngine`**: Single-month backtest
- **`MultiMonthRunner`**: Multi-month simulation with walk-forward optimization
- Currently stubs - implement to iterate through historical bars

## Adding New Components

### Adding a New Market Type
```python
# 1. Create /markets/exotic_market.py
from core import BaseMarket, MarketSession, RiskMultipliers

class ExoticMarket(BaseMarket):
    def __init__(self, symbol: str):
        super().__init__(symbol, market_type="exotic")
        # Define sessions, multipliers, etc.

# 2. Register in /markets/__init__.py
from exotic_market import ExoticMarket

# 3. Add to PlatformConfig.enabled_markets
# 4. TradingPlatform will auto-instantiate
```

### Adding a New Strategy
```python
# 1. Create in /strategies/example_strategies.py or new file
from core import BaseStrategy, Signal, StrategyContext

class MyStrategy(BaseStrategy):
    def required_indicators(self) -> List[str]:
        return ["close", "RSI", "ATR"]
    
    def supports_market(self, market_type: str) -> bool:
        return market_type == "indian"
    
    def supports_regime(self, regime: MarketRegime) -> bool:
        return regime in [MarketRegime.TRENDING, MarketRegime.RANGING]
    
    def supports_volatility(self, volatility: VolatilityBucket) -> bool:
        return volatility != VolatilityBucket.EXTREME
    
    def generate_signal(self, context: StrategyContext) -> Signal:
        # Your logic here
        return Signal(direction="BUY", confidence=0.75, ...)

# 2. Register in STRATEGY_REGISTRY
STRATEGY_REGISTRY["MyStrategy"] = MyStrategy()

# 3. Instantiate in /strategies/__init__.py
# 4. TradingPlatform will auto-instantiate
```

### Adding a New Broker
```python
# 1. Create /brokers/real_broker.py
from core import BaseBroker, Order, OrderStatus, Position

class RealBroker(BaseBroker):
    def __init__(self, api_key: str):
        super().__init__(broker_type="real")
        self.api_key = api_key
        # Connect to real broker API
    
    def place_order(self, order: Order) -> str:
        # Call real broker API
        pass

# 2. Modify TradingPlatform._initialize_broker()
# 3. Update config to select which broker to use
```

## Configuration

### Key Parameters (in `default_config.yaml`)
```yaml
# Account Risk Settings
initial_account_balance: 100000.0  # Start capital
risk_per_trade_percent: 1.0        # Risk 1% of balance per trade
max_position_size_percent: 5.0      # Max 5% per position
max_concurrent_trades: 5             # Max 5 open trades
max_daily_loss_percent: 3.0          # Max 3% daily loss

# Market Selection
enabled_markets: ["indian", "crypto"]
indian_symbols: ["NIFTY50", "BANKNIFTY"]
crypto_symbols: ["BTCUSDT", "ETHUSDT"]

# Strategy Selection
use_all_strategies: true
# OR
strategy_names: ["RSIStrategy", "MAStrategy"]

# Simulation
simulation_start_date: "2024-01-01"
simulation_end_date: "2024-12-31"
bar_frequency: "1h"  # 1h, 4h, 1d, etc.

# Broker Settings
execution_mode: "instant"  # "instant" or "next_bar"
slippage_percent: 0.05     # 0.05% slippage per order
```

### Market-Specific Risk Multipliers
Each market defines SL/TP multipliers per regime/volatility:
```python
IndianMarket.risk_multipliers = {
    "trending": RiskMultipliers(
        sl_atr_multiple=1.2,   # SL 1.2x ATR
        tp_atr_multiple=1.5,   # TP 1.5x ATR
        max_concurrent=3,      # Max 3 concurrent trending trades
    ),
    "ranging": RiskMultipliers(sl_atr_multiple=0.8, ...),
    "volatile": RiskMultipliers(sl_atr_multiple=1.5, ...),
}
```

## Usage Examples

### Basic: Initialize Platform
```python
from main import TradingPlatform
from config.settings import config

platform = TradingPlatform()
print(platform)
# TradingPlatform | Markets: 2 | Strategies: 3 | Balance: $100000.00
```

### Simulate a Bar
```python
import pandas as pd

bar = pd.Series({
    "symbol": "NIFTY50",
    "open": 19200,
    "high": 19300,
    "low": 19150,
    "close": 19250,
    "volume": 1000000,
    "rsi": 55,
    "sma20": 19100,
    "atr": 150,
})

result = platform.process_bar(
    symbol="NIFTY50",
    bar_data=bar,
    regime="trending",
    volatility="medium",
    session="morning"
)

if result:
    print(f"Trade: {result['direction']} {result['quantity']} @ {result['entry_price']}")
```

### Close All Positions and Export Results
```python
exit_prices = {
    "NIFTY50": 19350,
    "BANKNIFTY": 45000,
}

platform.close_positions(exit_prices)
summary = platform.get_performance_summary()

print(f"Total Return: {summary['total_return']:.2f}%")
print(f"Win Rate: {summary['win_rate']:.1f}%")
print(f"Max Drawdown: {summary['max_drawdown']:.2f}%")

platform.export_results()
```

### Load Custom Configuration
```python
from config.settings import PlatformConfig

custom_config = PlatformConfig.load_from_yaml("my_config.yaml")
platform = TradingPlatform(custom_config)
```

## Data Requirements

### Input Data Format
OHLCV + Indicators in DataFrame:
```
timestamp | open   | high  | low   | close | volume   | rsi | sma20 | atr
2024-01-01 00:00 | 19200 | 19300 | 19150 | 19250 | 1000000 | 55  | 19100 | 150
```

### Indicators Required (by strategy)
- **RSIStrategy**: close, RSI (14)
- **MAStrategy**: close, SMA (20), SMA (50)
- **BollingerBandsStrategy**: close, BB_upper, BB_lower, BB_middle

### Data Sources
Integrate with your own:
- CSV files (see `IndianMarket/strategy_tester_app/data/`)
- APIs (NSE, Kucoin, Binance, etc.)
- Real-time feeds (WebSocket, etc.)

## Performance Tracking

### Strategy Performance Table
Automatically built by StrategySelector:
```
Strategy | Market | Regime | Session | Volatility | WinRate | Trades | ProfitFactor | Confidence
RSI_MeanReversion | indian | ranging | morning | low | 55% | 45 | 1.2 | 92%
```

### Trade-Level Metrics
Tracked by PerformanceTracker:
- Entry/exit prices and times
- P&L absolute and percentage
- Win/loss status
- Trade duration
- Max drawdown per trade

### Account-Level Metrics
```
Final Balance: $105,000
Total Return: 5.0%
Total Trades: 120
Win Rate: 52.5%
Profit Factor: 1.3
Max Drawdown: 8.2%
Sharpe Ratio: 1.15
```

## Best Practices

1. **Always use configuration files** - No hardcoded values
2. **Test strategies on paper first** - PaperBroker is your friend
3. **Use market-specific implementations** - IndianMarket and CryptoMarket have different rules
4. **Monitor max drawdown** - Set reasonable daily loss limits
5. **Walk-forward test** - Use MultiMonthRunner for rolling window validation
6. **Track strategy performance** - StrategySelector ranks by historical win rate
7. **Implement trailing stops** - PaperBroker supports them
8. **Log everything** - Use logging module extensively

## Known Limitations

1. **PaperBroker only** - No real broker integration yet
2. **Single symbol per execution** - Modify ExecutionEngine to support multiple symbols
3. **No multi-timeframe** - Currently processes one timeframe at a time
4. **No news/macro events** - Purely technical indicators
5. **No ML models** - Only traditional technical strategies implemented

## Future Roadmap

1. **Real Broker Integration**: Zerodha, Angel One, etc.
2. **Advanced Strategies**: ML models, sentiment analysis
3. **Multi-timeframe Analysis**: 1m, 5m, 15m, 1h, 4h, 1d simultaneously
4. **Position Sizing**: Kelly Criterion, Optimal F, etc.
5. **Portfolio Optimization**: Correlation matrices, weights optimization
6. **Live Monitoring**: Real-time P&L, alerts, notifications
7. **Backtesting Reports**: HTML reports with equity curves, drawdown charts

## Troubleshooting

### "No markets initialized"
- Check `enabled_markets` in config.yaml
- Verify `indian_symbols` and `crypto_symbols` are populated

### "Strategy not generating signals"
- Verify required indicators are present in bar_data
- Check `supports_market()`, `supports_regime()`, `supports_volatility()` filters
- Enable DEBUG logging to see filtering decisions

### "Orders not filling"
- Verify market is in trading hours (check `is_trading_hours()`)
- Check daily loss limit not exceeded
- Verify position size calculated correctly by RiskManager

### Performance tracking shows no trades
- Ensure signals have confidence > 0.5 (configurable)
- Verify risk checks passing (try increasing max_concurrent_trades)
- Check bar_data has all required indicator columns

## Contributing

To add new features:
1. Extend abstract base class (BaseStrategy, BaseMarket, etc.)
2. Implement all abstract methods
3. Register in appropriate registry (STRATEGY_REGISTRY, etc.)
4. Add tests in `/tests/`
5. Update this README

## License

Proprietary - Internal Use Only

---

**Last Updated**: 2025-01-16
**Version**: 1.0.0 - Production-Grade Multi-Market Trading Platform
