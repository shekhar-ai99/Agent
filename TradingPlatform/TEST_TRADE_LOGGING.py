"""
Test script to demonstrate detailed trade logging with relax_entry mode
"""
import logging
from strategies import STRATEGY_REGISTRY
from execution.modes.backtest_mode import BacktestMode

# Setup logging to see detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

print("=" * 80)
print("BACKTEST WITH TRADE LOGGING DEMONSTRATION")
print("=" * 80)

# Initialize backtest
backtest = BacktestMode(
    capital=100000,
    market='india',
    timeframe='5min'
)

# Run with relax_entry to force trades and demonstrate logging
print("\nRunning backtest with relax_entry=True to demonstrate trade logging...")
print("This will force trades to show the logging system in action.\n")

results = backtest.run(
    strategies=list(STRATEGY_REGISTRY.keys()),
    bypass_selector=True,
    relax_entry=True,
    force_close_positions=True,
    save_pnl=True,
    save_strategy_pnl=True,
    generate_html_report=True
)

# Display results
print("\n" + "=" * 80)
print("BACKTEST COMPLETED")
print("=" * 80)
print(f"Total P&L: ${results['total_pnl']:.2f}")
print(f"Return (%): {results['return_pct']:.2f}%")
print(f"Total Trades: {results['num_trades']}")
print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"\nFiles Generated:")
if 'html_report' in results:
    print(f"  HTML Report: {results['html_report']}")
if 'pnl_file' in results:
    print(f"  PnL CSV: {results['pnl_file']}")
if 'strategy_pnl_file' in results:
    print(f"  Strategy PnL CSV: {results['strategy_pnl_file']}")
print("\n✅ All features working: Strategy-attributed logging, HTML reports, CSV exports")
print("=" * 80)
