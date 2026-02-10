"""
Getting Started with TradingPlatform

Quick setup guide and examples.
"""

# ============================================================================
# EXAMPLE 1: Basic Platform Initialization
# ============================================================================

from main import TradingPlatform
from config.settings import config

# Initialize with default config
platform = TradingPlatform()

print(f"Platform Status: {platform}")
print(f"Markets: {list(platform.markets.keys())}")
print(f"Strategies: {list(platform.strategies.keys())}")
print(f"Broker Balance: ${platform.broker.current_balance:,.2f}")


# ============================================================================
# EXAMPLE 2: Load Custom Configuration
# ============================================================================

from config.settings import PlatformConfig

# Load from YAML
custom_config = PlatformConfig.load_from_yaml("my_config.yaml")

# Or load from JSON
custom_config = PlatformConfig.load_from_json("my_config.json")

# Or create programmatically
custom_config = PlatformConfig(
    initial_account_balance=50000.0,
    risk_per_trade_percent=2.0,
    max_concurrent_trades=3,
    enabled_markets=["indian"],
    indian_symbols=["NIFTY50"],
)

platform = TradingPlatform(custom_config)


# ============================================================================
# EXAMPLE 3: Create Sample OHLCV Data with Indicators
# ============================================================================

import pandas as pd
import numpy as np

# Generate sample market data
dates = pd.date_range('2024-01-01', periods=250, freq='D')
n = len(dates)

# Create OHLCV
prices = 19200 + np.cumsum(np.random.randn(n) * 10)

df = pd.DataFrame({
    'open': prices + np.random.randn(n) * 5,
    'high': prices + abs(np.random.randn(n) * 20),
    'low': prices - abs(np.random.randn(n) * 20),
    'close': prices,
    'volume': np.random.randint(500000, 1500000, n),
}, index=dates)

# Calculate simple indicators
df['sma20'] = df['close'].rolling(20).mean()
df['sma50'] = df['close'].rolling(50).mean()

# RSI calculation
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# ATR calculation (simplified)
tr = pd.concat([
    df['high'] - df['low'],
    abs(df['high'] - df['close'].shift()),
    abs(df['low'] - df['close'].shift()),
], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean()

# Bollinger Bands
df['bb_middle'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

print(f"Sample data created: {len(df)} bars")
print(df.head())


# ============================================================================
# EXAMPLE 4: Single Bar Processing
# ============================================================================

import pandas as pd

# Process a single bar
bar = df.iloc[100]  # Get bar at index 100

result = platform.process_bar(
    symbol="NIFTY50",
    bar_data=bar,
    regime="trending",
    volatility="medium",
    session="morning"
)

if result:
    print(f"\nTrade Executed:")
    print(f"  Direction: {result['direction']}")
    print(f"  Entry Price: {result['entry_price']}")
    print(f"  Quantity: {result['quantity']}")
    print(f"  Stop Loss: {result['stop_loss']}")
    print(f"  Take Profit: {result['take_profit']}")
else:
    print("No trade signal this bar")


# ============================================================================
# EXAMPLE 5: Backtesting with Simulation Engine
# ============================================================================

from simulation import PaperTradingEngine

# Create data source: dict of symbol -> DataFrame
data_source = {
    "NIFTY50": df,
}

# Run single month backtest
engine = PaperTradingEngine(data_source=data_source)
result = engine.run(
    start_date="2024-01-15",
    end_date="2024-03-31"
)

print(f"\nBacktest Results:")
print(f"  Final Balance: ${result['final_balance']:,.2f}")
print(f"  Return: {result['total_return']:.2f}%")
print(f"  Trades: {result['total_trades_executed']}")
print(f"  Win Rate: {result['win_rate']:.1f}%")


# ============================================================================
# EXAMPLE 6: Walk-Forward Optimization
# ============================================================================

from simulation import MultiMonthRunner

# Run 6 months training, 1 month test, rolling monthly
runner = MultiMonthRunner()
agg_result = runner.run(
    data_source={"NIFTY50": df},
    train_months=6,
    test_months=1,
    step_months=1
)

print(f"\nWalk-Forward Results:")
print(f"  Iterations: {agg_result['iterations']}")
print(f"  Total Return: {agg_result['total_return']:.2f}%")
print(f"  Avg Per Iteration: {agg_result['avg_return_per_iteration']:.2f}%")
print(f"  Avg Win Rate: {agg_result['avg_win_rate']:.1f}%")


# ============================================================================
# EXAMPLE 7: Accessing Strategy Performance Table
# ============================================================================

# After trading, check strategy rankings
perf_table = platform.strategy_selector.performance_table

print(f"\nStrategy Performance Table:")
print(perf_table)

# Query by specific context
from core import StrategyContext, MarketRegime, VolatilityBucket

# Get best strategy for trending + high volatility market
best = platform.strategy_selector.select_best_strategy(
    market_type="indian",
    regime=MarketRegime.TRENDING,
    volatility=VolatilityBucket.HIGH,
    session="morning"
)

if best:
    print(f"\nBest strategy for trending/high vol: {best[0]}")


# ============================================================================
# EXAMPLE 8: Getting Performance Summary
# ============================================================================

summary = platform.get_performance_summary()

print(f"\nPerformance Summary:")
print(f"  Final Balance: ${summary['final_balance']:,.2f}")
print(f"  Total Return: {summary['total_return']:.2f}%")
print(f"  Total Trades: {summary.get('total_trades', 0)}")
print(f"  Win Rate: {summary.get('win_rate', 0):.1f}%")
print(f"  Profit Factor: {summary.get('profit_factor', 0):.2f}")
print(f"  Max Drawdown: {summary.get('max_drawdown', 0):.2f}%")


# ============================================================================
# EXAMPLE 9: Closing Positions and Exporting
# ============================================================================

# Close all open positions at specific prices
exit_prices = {
    "NIFTY50": 19500,
    "BANKNIFTY": 45000,
}

platform.close_positions(exit_prices, reason="End of backtest")

# Export results to files
platform.export_results()

print("Results exported to:")
print(f"  - strategy_performance.csv")
print(f"  - trade_history.csv")


# ============================================================================
# EXAMPLE 10: Creating a Custom Strategy
# ============================================================================

from core import BaseStrategy, Signal, StrategyContext, MarketRegime, VolatilityBucket
from typing import List

class MACrossoverStrategy(BaseStrategy):
    """
    Simple moving average crossover strategy.
    Buy when SMA20 > SMA50, Sell when SMA20 < SMA50
    """
    
    def __init__(self):
        super().__init__(
            name="MACrossoverStrategy",
            version="1.0",
            description="SMA20 > SMA50 trend following"
        )
    
    def required_indicators(self) -> List[str]:
        """This strategy needs SMA20 and SMA50"""
        return ["close", "sma20", "sma50"]
    
    def supports_market(self, market_type: str) -> bool:
        """Works on any market"""
        return True
    
    def supports_regime(self, regime: MarketRegime) -> bool:
        """Best in trending markets"""
        return regime == MarketRegime.TRENDING
    
    def supports_volatility(self, volatility: VolatilityBucket) -> bool:
        """Works in all volatility regimes"""
        return True
    
    def generate_signal(self, context: StrategyContext) -> Signal:
        """Generate buy/sell signal"""
        
        # Get indicators from context
        sma20 = context.current_bar.get('sma20')
        sma50 = context.current_bar.get('sma50')
        
        if sma20 is None or sma50 is None:
            return Signal(
                strategy_name=self.name,
                direction="HOLD",
                confidence=0.0,
                reasoning="Missing indicators"
            )
        
        # Buy signal
        if sma20 > sma50:
            return Signal(
                strategy_name=self.name,
                direction="BUY",
                confidence=0.7,
                reasoning=f"SMA20({sma20:.0f}) > SMA50({sma50:.0f})"
            )
        
        # Sell signal
        elif sma20 < sma50:
            return Signal(
                strategy_name=self.name,
                direction="SELL",
                confidence=0.7,
                reasoning=f"SMA20({sma20:.0f}) < SMA50({sma50:.0f})"
            )
        
        # No signal
        else:
            return Signal(
                strategy_name=self.name,
                direction="HOLD",
                confidence=0.5,
                reasoning="SMAs converging"
            )


# Register and use custom strategy
from strategies.example_strategies import STRATEGY_REGISTRY

custom_strat = MACrossoverStrategy()
STRATEGY_REGISTRY["MACrossoverStrategy"] = custom_strat

platform = TradingPlatform()
print(f"Strategies: {list(platform.strategies.keys())}")


# ============================================================================
# EXAMPLE 11: Accessing Broker and Position Details
# ============================================================================

# Get open positions
open_positions = platform.broker.get_open_positions()

for pos in open_positions:
    print(f"\nPosition: {pos.symbol}")
    print(f"  Quantity: {pos.quantity}")
    print(f"  Entry Price: {pos.entry_price:.2f}")
    print(f"  Current Price: {pos.current_price:.2f}")
    print(f"  P&L: ${pos.current_pnl:.2f} ({pos.pnl_percent:.2f}%)")
    print(f"  Stop Loss: {pos.stop_loss}")
    print(f"  Take Profit: {pos.take_profit}")

# Get trade history
trades = platform.broker.trade_history

for trade in trades[-5:]:  # Last 5 trades
    print(f"\nTrade: {trade['symbol']}")
    print(f"  Side: {trade['side']}")
    print(f"  Entry: {trade['entry_price']} x {trade['quantity']}")
    print(f"  Exit: {trade['exit_price']}")
    print(f"  P&L: ${trade['pnl']:.2f}")


# ============================================================================
# EXAMPLE 12: Monitoring Real-Time Performance
# ============================================================================

# After each bar, check account status
def monitor_account(platform):
    balance = platform.broker.current_balance
    initial = platform.config.initial_account_balance
    
    return_pct = ((balance - initial) / initial) * 100
    
    print(f"\nAccount Status:")
    print(f"  Initial: ${initial:,.2f}")
    print(f"  Current: ${balance:,.2f}")
    print(f"  Return: {return_pct:.2f}%")
    print(f"  Open Positions: {len(platform.broker.get_open_positions())}")


# ============================================================================
# Configuration File Examples
# ============================================================================

"""
# config.yaml

# Account Settings
initial_account_balance: 100000.0
risk_per_trade_percent: 1.0
max_position_size_percent: 5.0
max_concurrent_trades: 5
max_daily_loss_percent: 3.0

# Markets
enabled_markets:
  - indian
  - crypto

indian_symbols:
  - NIFTY50
  - BANKNIFTY
  - FINNIFTY

crypto_symbols:
  - BTCUSDT
  - ETHUSDT

# Strategies
use_all_strategies: true

# Simulation
simulation_start_date: "2024-01-01"
simulation_end_date: "2024-12-31"
bar_frequency: "1h"

# Broker
execution_mode: "instant"
slippage_percent: 0.05

# Logging
log_level: "INFO"
log_file: "logs/trading_platform.log"

# Output
save_performance_table: true
save_trade_history: true
output_dir: "results"
"""

print("=" * 70)
print("QUICK START GUIDE - TradingPlatform")
print("=" * 70)
print("\n1. Initialize Platform:")
print("   platform = TradingPlatform()")
print("\n2. Process bars:")
print("   result = platform.process_bar(symbol, bar_data, regime, volatility, session)")
print("\n3. Run backtest:")
print("   engine = PaperTradingEngine(data_source=data)")
print("   result = engine.run('2024-01-01', '2024-12-31')")
print("\n4. Get performance:")
print("   summary = platform.get_performance_summary()")
print("\n5. Export results:")
print("   platform.export_results()")
print("=" * 70)
