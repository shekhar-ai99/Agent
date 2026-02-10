#!/usr/bin/env python3
"""
QUICK START: Run backtests with CSV data from datasets folder

This script shows how to:
1. Load NIFTY data from CSV files
2. Add technical indicators
3. Run a backtest with strategies
4. Compare strategy performance
"""

import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import TradingPlatform modules
from data import DatasetLoader
from config import PlatformConfig
from core import BaseStrategy
from execution import ExecutionEngine
from simulation import PaperTradingEngine, MultiMonthRunner


def example_1_load_and_explore_data():
    """Example 1: Load data and explore it"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Load and Explore Data")
    print("="*70)
    
    # Load NIFTY 5min data
    df = DatasetLoader.load_nifty(timeframe="5min")
    
    print(f"\nData loaded:")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    return df


def example_2_load_with_indicators():
    """Example 2: Load data and add indicators"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Load Data with Technical Indicators")
    print("="*70)
    
    # Load data
    df = DatasetLoader.load_nifty(timeframe="5min")
    
    # Add indicators
    print("\nAdding technical indicators...")
    df = DatasetLoader.add_indicators(df)
    
    print(f"Data shape after adding indicators: {df.shape}")
    print(f"New indicators: SMA20, SMA50, RSI, ATR, Bollinger Bands")
    print(f"\nData sample with indicators:")
    cols_to_show = ['close', 'sma20', 'sma50', 'rsi', 'atr', 'bb_upper', 'bb_lower']
    print(df[cols_to_show].head(10))
    
    return df


def example_3_prepare_for_backtest():
    """Example 3: Prepare data for backtesting"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Prepare Data for Backtesting")
    print("="*70)
    
    # Load data in backtest-ready format
    data_source = DatasetLoader.load_for_backtest(symbol="NIFTY50", timeframe="5min")
    
    # Add indicators
    data_source["NIFTY50"] = DatasetLoader.add_indicators(data_source["NIFTY50"])
    
    print(f"\nBacktest-ready data prepared:")
    print(f"  Symbols: {list(data_source.keys())}")
    print(f"  NIFTY50 rows: {len(data_source['NIFTY50'])}")
    print(f"  Date range: {data_source['NIFTY50'].index.min()} to {data_source['NIFTY50'].index.max()}")
    
    return data_source


def example_4_run_single_strategy_backtest():
    """Example 4: Run backtest with a single strategy"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Run Single Strategy Backtest")
    print("="*70)
    
    # Load data
    data_source = DatasetLoader.load_for_backtest(symbol="NIFTY50", timeframe="5min")
    data_source["NIFTY50"] = DatasetLoader.add_indicators(data_source["NIFTY50"])
    
    # Create config
    config = PlatformConfig()
    config.initial_capital = 100000  # Start with 1 lakh
    config.risk_per_trade = 0.02    # 2% risk per trade
    
    # Create engine
    engine = PaperTradingEngine(data_source=data_source)
    
    print(f"\nBacktest Configuration:")
    print(f"  Initial Capital: ₹{config.initial_capital:,.0f}")
    print(f"  Risk per Trade: {config.risk_per_trade*100}%")
    print(f"  Data: NIFTY50 {len(data_source['NIFTY50'])} bars (5min)")
    print(f"  Date Range: {data_source['NIFTY50'].index.min()} to {data_source['NIFTY50'].index.max()}")
    
    print("\n⚠️  Note: To run actual backtest, you would:")
    print("  1. Select a strategy (RSI, MovingAverage, BollingerBands)")
    print("  2. Call engine.run(strategy)")
    print("  3. Analyze performance metrics")
    
    return engine, config


def example_5_list_available_data():
    """Example 5: List all available datasets"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Available Datasets")
    print("="*70)
    
    DatasetLoader.list_available_datasets()


def example_6_load_multiple_timeframes():
    """Example 6: Load and compare multiple timeframes"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Load Multiple Timeframes")
    print("="*70)
    
    timeframes = ["1min", "5min", "15min"]
    datasets = {}
    
    for tf in timeframes:
        try:
            df = DatasetLoader.load_nifty(timeframe=tf)
            datasets[tf] = df
            print(f"\n{tf:6s}: {len(df):6,d} bars | {df.index.min()} to {df.index.max()}")
        except FileNotFoundError:
            print(f"\n{tf:6s}: NOT FOUND")
    
    return datasets


# ============================================================================
# MAIN - Run examples
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRADINGPLATFORM - BACKTEST QUICK START")
    print("="*70)
    print("Loading CSV data from datasets folder and running examples...")
    
    try:
        # Example 1: Load and explore
        example_1_load_and_explore_data()
        
        # Example 2: Load with indicators
        example_2_load_with_indicators()
        
        # Example 3: Prepare for backtest
        example_3_prepare_for_backtest()
        
        # Example 4: Run single strategy
        example_4_run_single_strategy_backtest()
        
        # Example 5: List available data
        example_5_list_available_data()
        
        # Example 6: Multiple timeframes
        example_6_load_multiple_timeframes()
        
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)
        print("\nNext steps:")
        print("1. Review the dataset_loader.py module for all available methods")
        print("2. Update your strategies in strategies/example_strategies.py")
        print("3. Run actual backtests by calling engine.run(strategy)")
        print("4. Check simulation/backtest.py for walk-forward testing")
        print("\n")
        
    except Exception as e:
        logger.error(f"Error in examples: {e}", exc_info=True)
        print("\n" + "="*70)
        print("❌ Error occurred - see details above")
        print("="*70 + "\n")
