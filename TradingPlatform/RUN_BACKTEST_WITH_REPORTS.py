"""
Test script for running backtests with detailed logging and HTML report generation.

Features:
- Detailed trade logging showing which strategy takes which trade
- CSV export of PnL data
- Per-strategy PnL breakdown
- HTML report generation with market/timeframe breakdown
- Trade log table
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategies import STRATEGY_REGISTRY
from execution.modes.backtest_mode import BacktestMode

# Configure logging to show INFO level with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backtest_detailed.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

def run_backtest_with_reports():
    """Run backtest with full logging and report generation"""
    
    print("=" * 80)
    print("BACKTEST WITH DETAILED LOGGING & HTML REPORTS")
    print("=" * 80)
    
    # Configuration
    market = "india"
    timeframe = "5min"
    capital = 100000
    strategies = list(STRATEGY_REGISTRY.keys())
    
    print(f"\nConfiguration:")
    print(f"  Market: {market}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Capital: ${capital:,}")
    print(f"  Strategies: {', '.join(strategies)}")
    print("\n" + "=" * 80)
    print("Starting backtest...\n")
    
    # Initialize backtest
    bt = BacktestMode(capital=capital, market=market, timeframe=timeframe)
    
    # Run with all features enabled
    results = bt.run(
        strategies=strategies,
        bypass_selector=True,          # Use all strategies
        relax_entry=False,             # Use real signal generation (not relaxed)
        force_close_positions=True,    # Close all positions at end
        save_pnl=True,                 # Save per-bar PnL CSV
        save_strategy_pnl=True,        # Save per-strategy summary CSV
        generate_html_report=True,     # Generate HTML report
    )
    
    print("\n" + "=" * 80)
    print("BACKTEST COMPLETED")
    print("=" * 80)
    print("\nResults:")
    print(f"  Strategies: {', '.join(results.get('strategies', []))}")
    print(f"  Total P&L: ${results.get('total_pnl', 0):,.2f}")
    print(f"  Return %: {results.get('return_pct', 0):.2f}%")
    print(f"  Total Trades: {results.get('num_trades', 0):,}")
    print(f"  Win Rate: {results.get('win_rate', 0):.2f}%")
    print(f"  Profit Factor: {results.get('profit_factor', 0):.2f}")
    print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
    print(f"  Total Bars: {results.get('total_bars', 0):,}")
    
    print("\n" + "=" * 80)
    print("GENERATED FILES")
    print("=" * 80)
    
    if results.get('pnl_csv'):
        print(f"\n📊 Per-bar PnL CSV:")
        print(f"   {results['pnl_csv']}")
    
    if results.get('strategy_pnl_csv'):
        print(f"\n📈 Per-strategy PnL CSV:")
        print(f"   {results['strategy_pnl_csv']}")
    
    if results.get('html_report'):
        print(f"\n📄 HTML Report:")
        print(f"   {results['html_report']}")
        print(f"\n   Open in browser: file://{results['html_report']}")
    
    print(f"\n📝 Detailed log file:")
    print(f"   {Path('backtest_detailed.log').resolve()}")
    
    print("\n" + "=" * 80)
    print("TRADE LOG SUMMARY")
    print("=" * 80)
    
    # Show trade summary from broker
    trade_history = bt.broker.get_trade_history()
    if trade_history:
        print(f"\nTotal trades executed: {len(trade_history)}")
        print("\nFirst 10 trades:")
        print(f"{'#':<4} {'Strategy':<25} {'Symbol':<10} {'Dir':<5} {'P&L':<12} {'Exit Reason':<20}")
        print("-" * 80)
        for i, trade in enumerate(trade_history[:10], 1):
            pnl = trade.get('pnl', 0)
            pnl_str = f"${pnl:,.2f}"
            pnl_icon = "✅" if pnl > 0 else "❌" if pnl < 0 else "⚪"
            print(
                f"{i:<4} {trade.get('strategy', 'unknown'):<25} "
                f"{trade.get('symbol', ''):<10} {trade.get('direction', ''):<5} "
                f"{pnl_icon} {pnl_str:<10} {trade.get('exit_reason', ''):<20}"
            )
        
        if len(trade_history) > 10:
            print(f"\n... and {len(trade_history) - 10} more trades")
    
    print("\n" + "=" * 80)
    print("✅ BACKTEST COMPLETE - Check the generated files above")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_backtest_with_reports()
