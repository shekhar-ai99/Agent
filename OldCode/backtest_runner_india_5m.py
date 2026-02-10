#!/usr/bin/env python3
"""
🏆 NON-DESTRUCTIVE INDIAN MARKET BACKTEST RUNNER

This orchestrates the GOLD backtest code without modifying it.
Runs all strategies on 5-minute Indian market data.

Reference: INDIAN_MARKET_BACKTEST_AUDIT.md
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add paths to import gold code
WORKSPACE_ROOT = Path(__file__).parent
INDIAN_MARKET_PATH = WORKSPACE_ROOT / "IndianMarket" / "strategy_tester_app"
COMMON_PATH = WORKSPACE_ROOT / "Common"

# Important: Add strategy_tester_app to path so 'app' can be imported as a package
sys.path.insert(0, str(INDIAN_MARKET_PATH))
sys.path.insert(0, str(COMMON_PATH))

# Import gold components
# These will import from IndianMarket/strategy_tester_app/app/
from app.backtest_engine import SimpleBacktester, save_trades_to_csv
from app.compute_indicators import compute_indicators
from strategies.strategies import strategy_factories, default_strategy_functions

# Setup logging
LOG_DIR = WORKSPACE_ROOT / "runs" / f"backtest_india_5m_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_nifty_5m_data():
    """
    Load NIFTY 5-minute data from TradingPlatform datasets.
    
    Returns:
        pd.DataFrame: OHLCV data with datetime index
    """
    logger.info("Loading NIFTY 5-minute data...")
    
    # Check multiple possible locations
    possible_paths = [
        WORKSPACE_ROOT / "TradingPlatform" / "datasets" / "nifty" / "5min" / "nifty_historical_data_5min.csv",
        WORKSPACE_ROOT / "TradingPlatform" / "datasets" / "nifty" / "5min.csv",
        WORKSPACE_ROOT / "datasets" / "nifty" / "5min.csv",
        INDIAN_MARKET_PATH / "data" / "nifty_5min.csv",
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"Found data at: {path}")
            df = pd.read_csv(path)
            
            # Ensure datetime column
            if 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            else:
                # Try to use index
                df['timestamp'] = pd.to_datetime(df.index)
            
            df = df.set_index('timestamp')
            df = df.sort_index()
            
            logger.info(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
            return df
    
    logger.error("No NIFTY 5-minute data found!")
    logger.info("Searched locations:")
    for path in possible_paths:
        logger.info(f"  - {path}")
    
    raise FileNotFoundError("NIFTY 5-minute data not found. Please ensure data exists in TradingPlatform/datasets/nifty/5min.csv")


def prepare_data(df):
    """
    Add technical indicators using gold code.
    
    Args:
        df: Raw OHLCV DataFrame
        
    Returns:
        pd.DataFrame: Data with indicators
    """
    logger.info("Computing indicators...")
    
    # Ensure required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            raise ValueError(f"DataFrame must have '{col}' column")
    
    # Use gold indicator computation (requires output_path)
    temp_output = LOG_DIR / "data_with_indicators.csv"
    df = compute_indicators(df, str(temp_output))
    
    logger.info(f"Added {len(df.columns) - 5} indicators")
    return df


def get_default_params():
    """
    Get default parameters for strategies (from gold code).
    
    Returns:
        dict: Strategy parameters
    """
    return {
        'sl_atr_mult': 1.5,      # Stop Loss = Entry ± (ATR * 1.5)
        'tp_atr_mult': 2.0,      # Take Profit = Entry ± (ATR * 2.0)
        'tsl_atr_mult': 1.0,     # Trailing SL = Entry ± (ATR * 1.0)
        'same_day_exit': True,   # Force exit at 15:30
    }


def run_strategy_backtest(strategy_name, strategy_func, df, params, timeframe="5min", symbol="NIFTY"):
    """
    Run backtest for a single strategy using GOLD code.
    
    Args:
        strategy_name: Name of strategy
        strategy_func: Strategy function (from gold code)
        df: Data with indicators
        params: Strategy parameters
        timeframe: Timeframe string
        symbol: Symbol name
        
    Returns:
        dict: Performance results
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"RUNNING STRATEGY: {strategy_name}")
    logger.info(f"{'=' * 80}\n")
    
    run_id = f"{strategy_name}_{datetime.now().strftime('%H%M%S')}"
    
    # For index backtesting (without option premium files), set dummy expiry/strike
    # OptionTradeExecutor will skip if CE/PE files not found
    expiry_date = "2025-05-29"  # Dummy expiry
    strike_price = 23000  # Dummy ATM strike for NIFTY
    
    # Create gold backtest engine
    backtester = SimpleBacktester(
        df=df,
        strategy_name=strategy_name,
        strategy_func=strategy_func,
        params=params,
        timeframe=timeframe,
        symbol=symbol,
        exchange="NSE",
        run_id=run_id,
        expiry_date=expiry_date,
        strike_price=strike_price,
    )
    
    # Run simulation (gold code)
    results = backtester.run_simulation()
    
    # Save trades
    if backtester.trades:
        save_trades_to_csv(
            trades=backtester.trades,
            run_id=run_id,
            output_dir=str(LOG_DIR / "trades")
        )
    
    logger.info(f"\nSTRATEGY: {strategy_name}")
    logger.info(f"  Total Trades: {results.get('total_trades', 0)}")
    logger.info(f"  Profitable: {results.get('profitable_trades', 0)}")
    logger.info(f"  Losing: {results.get('losing_trades', 0)}")
    logger.info(f"  Win Rate: {results.get('win_rate', 0):.2f}%")
    logger.info(f"  Total P&L: {results.get('total_pnl', 0):.2f}")
    logger.info(f"  Performance Score: {results.get('performance_score', 0):.2f}")
    
    # Log exit reasons
    exit_reasons = results.get('exit_reasons', {})
    logger.info(f"  Exit Reasons:")
    logger.info(f"    SL: {exit_reasons.get('sl', 0)}")
    logger.info(f"    TSL: {exit_reasons.get('tsl', 0)}")
    logger.info(f"    TP: {exit_reasons.get('tp', 0)}")
    logger.info(f"    Signal: {exit_reasons.get('signal', 0)}")
    logger.info(f"    Session End: {exit_reasons.get('session_end', 0)}")
    
    return results


def generate_summary_report(all_results):
    """
    Generate a comprehensive performance report.
    
    Args:
        all_results: Dict of strategy_name -> results
    """
    logger.info(f"\n{'=' * 80}")
    logger.info("📊 COMPREHENSIVE PERFORMANCE REPORT")
    logger.info(f"{'=' * 80}\n")
    
    # Create summary DataFrame
    summary_data = []
    for strategy_name, results in all_results.items():
        summary_data.append({
            'Strategy': strategy_name,
            'Trades': results.get('total_trades', 0),
            'Win %': results.get('win_rate', 0),
            'Net P&L': results.get('total_pnl', 0),
            'Profitable': results.get('profitable_trades', 0),
            'Losing': results.get('losing_trades', 0),
            'Score': results.get('performance_score', 0),
            'SL Hits': results.get('exit_reasons', {}).get('sl', 0),
            'TP Hits': results.get('exit_reasons', {}).get('tp', 0),
            'TSL Hits': results.get('exit_reasons', {}).get('tsl', 0),
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Sort by performance score
    df_summary = df_summary.sort_values('Score', ascending=False)
    
    # Print table
    print("\n" + df_summary.to_string(index=False))
    print()
    
    # Save to CSV
    report_path = LOG_DIR / "performance_report.csv"
    df_summary.to_csv(report_path, index=False)
    logger.info(f"✅ Performance report saved to: {report_path}")
    
    # Print statistics
    logger.info("\n📈 AGGREGATE STATISTICS:")
    logger.info(f"  Total Strategies Tested: {len(all_results)}")
    logger.info(f"  Strategies with Trades: {sum(1 for r in all_results.values() if r.get('total_trades', 0) > 0)}")
    logger.info(f"  Total Trades Across All: {sum(r.get('total_trades', 0) for r in all_results.values())}")
    logger.info(f"  Average Win Rate: {df_summary['Win %'].mean():.2f}%")
    logger.info(f"  Total P&L: {df_summary['Net P&L'].sum():.2f}")
    logger.info(f"  Best Strategy: {df_summary.iloc[0]['Strategy']} (Score: {df_summary.iloc[0]['Score']:.2f})")
    
    return df_summary


def main():
    """Main execution function"""
    try:
        logger.info("🚀 STARTING INDIAN MARKET BACKTEST (GOLD CODE)")
        logger.info(f"Run Directory: {LOG_DIR}")
        logger.info(f"Timestamp: {datetime.now()}")
        
        # Step 1: Load data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: LOADING DATA")
        logger.info("=" * 80)
        df = load_nifty_5m_data()
        
        # Step 2: Prepare data (add indicators)
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: COMPUTING INDICATORS")
        logger.info("=" * 80)
        df = prepare_data(df)
        
        # Step 3: Get parameters
        params = get_default_params()
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: STRATEGY PARAMETERS")
        logger.info("=" * 80)
        for key, value in params.items():
            logger.info(f"  {key}: {value}")
        
        # Step 4: Get strategies from gold code
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: LOADING GOLD STRATEGIES")
        logger.info("=" * 80)
        
        # Instantiate strategies from factories
        strategies_to_test = {}
        for strategy_name, factory in strategy_factories.items():
            logger.info(f"  ✅ {strategy_name}")
            # Call factory to create strategy function
            # The factory bakes in the parameters, so the returned function only needs (row, history)
            # But backtest_engine.py calls with (row, history, params) and expects dict return
            base_strategy = factory()  # This returns a function(row, history) -> string
            
            # Wrap to make compatible with backtest_engine expectations
            def make_compatible_wrapper(strat_func):
                def wrapped_strategy(row, history, params=None):
                    # Call the gold strategy (returns string)
                    signal_str = strat_func(row, history)
                    
                    # Convert 'hold' -> 'none' for compatibility
                    if signal_str == 'hold':
                        signal_str = 'none'
                    
                    # Compute SL/TP/TSL based on ATR and params
                    if params is None:
                        params = get_default_params()
                    
                    atr = row.get('atr_14', row.get('atr', 1.0))
                    close = row['close']
                    
                    # Calculate risk parameters
                    sl_val = close - (atr * params['sl_atr_mult'])
                    tp_val = close + (atr * params['tp_atr_mult']) 
                    tsl_val = close - (atr * params['tsl_atr_mult'])
                    
                    # Return dict format expected by backtest_engine
                    return {
                        'signal': signal_str,
                        'sl': sl_val,
                        'tp': tp_val,
                        'tsl': tsl_val
                    }
                return wrapped_strategy
            
            strategies_to_test[strategy_name] = make_compatible_wrapper(base_strategy)
        
        logger.info(f"\nTotal strategies loaded: {len(strategies_to_test)}")
        
        # Step 5: Run backtests
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: RUNNING BACKTESTS")
        logger.info("=" * 80)
        
        all_results = {}
        for strategy_name, strategy_func in strategies_to_test.items():
            try:
                results = run_strategy_backtest(
                    strategy_name=strategy_name,
                    strategy_func=strategy_func,
                    df=df.copy(),  # Pass copy to avoid mutations
                    params=params,
                    timeframe="5min",
                    symbol="NIFTY"
                )
                all_results[strategy_name] = results
            except Exception as e:
                logger.error(f"❌ Strategy {strategy_name} failed: {e}", exc_info=True)
                all_results[strategy_name] = {
                    'total_trades': 0,
                    'error': str(e)
                }
        
        # Step 6: Generate report
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: GENERATING PERFORMANCE REPORT")
        logger.info("=" * 80)
        df_summary = generate_summary_report(all_results)
        
        # Step 7: Final summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ BACKTEST COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📁 Results saved to: {LOG_DIR}")
        logger.info(f"📊 Performance report: {LOG_DIR / 'performance_report.csv'}")
        logger.info(f"📝 Trade logs: {LOG_DIR / 'trades'}")
        logger.info(f"📋 Full log: {LOG_DIR / 'backtest.log'}")
        
        return all_results
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        sys.exit(1)
