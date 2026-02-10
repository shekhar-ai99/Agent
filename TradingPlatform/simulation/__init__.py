"""
Simulation Engine

Backtesting and simulation for the trading platform.
Runs platform over historical data to validate strategies.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from main import TradingPlatform
from config.settings import PlatformConfig, config
from core import MarketRegime, VolatilityBucket

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """
    Run a single month/period of backtesting.
    Iterates through historical data bars and executes the platform.
    """
    
    def __init__(self, config: PlatformConfig = None, data_source: Dict[str, pd.DataFrame] = None):
        """
        Initialize paper trading engine.
        
        Args:
            config: PlatformConfig (uses global if None)
            data_source: Dict of symbol -> OHLCV DataFrame with indicators
        """
        self.config = config or config
        self.data_source = data_source or {}
        self.platform: TradingPlatform = None
        self.logger = logging.getLogger("PaperTradingEngine")
        
    def run(self, start_date: str, end_date: str) -> Dict:
        """
        Run simulation for date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Simulation summary with performance metrics
        """
        self.logger.info("=" * 70)
        self.logger.info(f"BACKTESTING: {start_date} to {end_date}")
        self.logger.info("=" * 70)
        
        # Initialize platform
        self.platform = TradingPlatform(self.config)
        
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        total_bars = 0
        total_trades = 0
        
        # Process each symbol's data
        for symbol, data_df in self.data_source.items():
            if symbol not in self.platform.markets:
                self.logger.warning(f"Symbol {symbol} not in platform markets")
                continue
            
            # Filter data to date range
            mask = (data_df.index >= start) & (data_df.index <= end)
            period_data = data_df[mask]
            
            self.logger.info(f"\nProcessing {symbol}: {len(period_data)} bars")
            
            # Iterate through each bar
            for idx, (timestamp, row) in enumerate(period_data.iterrows()):
                try:
                    # Calculate market regime and volatility
                    regime = self._calculate_regime(period_data, idx)
                    volatility = self._calculate_volatility(period_data, idx)
                    session = self._get_session(timestamp, symbol)
                    
                    # Process bar
                    result = self.platform.process_bar(
                        symbol=symbol,
                        bar_data=row,
                        regime=regime,
                        volatility=volatility,
                        session=session,
                    )
                    
                    if result:
                        total_trades += 1
                    
                    total_bars += 1
                    
                    if (total_bars % 100) == 0:
                        self.logger.info(f"  Processed {total_bars} bars, {total_trades} trades")
                
                except Exception as e:
                    self.logger.error(f"Error processing bar {timestamp}: {str(e)}")
                    continue
        
        # Close all positions at end of simulation
        last_prices = {symbol: data.iloc[-1]['close'] for symbol, data in self.data_source.items()}
        self.platform.close_positions(last_prices, reason="End of simulation")
        
        # Export results
        self.platform.export_results()
        
        # Return summary
        summary = self.platform.get_performance_summary()
        summary.update({
            "total_bars_processed": total_bars,
            "total_trades_executed": total_trades,
            "simulation_start": start_date,
            "simulation_end": end_date,
        })
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("SIMULATION COMPLETE")
        self.logger.info("=" * 70)
        self._print_summary(summary)
        
        return summary
    
    def _calculate_regime(self, data_df: pd.DataFrame, current_idx: int) -> str:
        """
        Calculate market regime based on trend.
        
        Simplified: Trending if close > SMA50, Ranging otherwise
        """
        if current_idx < 50 or 'close' not in data_df.columns:
            return "ranging"
        
        recent = data_df.iloc[current_idx-50:current_idx+1]['close']
        sma50 = recent.mean()
        current_close = recent.iloc[-1]
        
        if current_close > sma50 * 1.02:
            return "trending"
        elif current_close < sma50 * 0.98:
            return "trending"
        else:
            return "ranging"
    
    def _calculate_volatility(self, data_df: pd.DataFrame, current_idx: int) -> str:
        """
        Calculate volatility bucket based on ATR or standard deviation.
        
        Simplified: Low if ATR < 1.0, High if > 2.0, else Medium
        """
        if current_idx < 14 or 'atr' not in data_df.columns:
            return "medium"
        
        recent_atr = data_df.iloc[max(0, current_idx-14):current_idx+1]['atr'].mean()
        
        if recent_atr < 1.0:
            return "low"
        elif recent_atr > 2.0:
            return "high"
        else:
            return "medium"
    
    def _get_session(self, timestamp: pd.Timestamp, symbol: str) -> str:
        """
        Get current trading session for the symbol.
        
        Simplified: Morning (9:15-12:00), Afternoon (12:00-15:30)
        """
        hour = timestamp.hour
        minute = timestamp.minute
        
        if symbol.endswith("USDT"):  # Crypto, always same session
            return "default"
        
        # Indian market sessions
        time_minutes = hour * 60 + minute
        morning_start = 9 * 60 + 15
        midday_start = 12 * 60
        afternoon_end = 15 * 60 + 30
        
        if morning_start <= time_minutes < midday_start:
            return "morning"
        elif midday_start <= time_minutes < afternoon_end:
            return "afternoon"
        else:
            return "post_market"
    
    def _print_summary(self, summary: Dict):
        """Pretty-print performance summary"""
        print("\n" + "=" * 70)
        print("PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"Period: {summary.get('simulation_start')} to {summary.get('simulation_end')}")
        print(f"Bars Processed: {summary.get('total_bars_processed', 0)}")
        print(f"Trades Executed: {summary.get('total_trades_executed', 0)}")
        print(f"\nFinal Balance: ${summary.get('final_balance', 0):,.2f}")
        print(f"Total Return: {summary.get('total_return', 0):.2f}%")
        print(f"Win Rate: {summary.get('win_rate', 0):.1f}%")
        print(f"Profit Factor: {summary.get('profit_factor', 0):.2f}")
        print(f"Max Drawdown: {summary.get('max_drawdown', 0):.2f}%")
        print("=" * 70 + "\n")


class MultiMonthRunner:
    """
    Run platform over multiple months/years.
    Supports walk-forward optimization: train on historical, validate on recent.
    """
    
    def __init__(self, config: PlatformConfig = None):
        """
        Initialize multi-month runner.
        
        Args:
            config: PlatformConfig
        """
        self.config = config or config
        self.logger = logging.getLogger("MultiMonthRunner")
        self.results: List[Dict] = []
    
    def run(
        self,
        data_source: Dict[str, pd.DataFrame],
        train_months: int = 6,
        test_months: int = 1,
        step_months: int = 1,
    ) -> Dict:
        """
        Run walk-forward simulation.
        
        Train on recent N months, test on next M months, then slide window forward.
        
        Args:
            data_source: Dict of symbol -> OHLCV DataFrame
            train_months: Months of data to train on
            test_months: Months to test on
            step_months: Months to slide window forward
            
        Returns:
            Aggregated results across all test periods
        """
        self.logger.info("=" * 70)
        self.logger.info("WALK-FORWARD OPTIMIZATION")
        self.logger.info(f"Train: {train_months}m | Test: {test_months}m | Step: {step_months}m")
        self.logger.info("=" * 70)
        
        # Get date range
        first_date = min(df.index.min() for df in data_source.values())
        last_date = max(df.index.max() for df in data_source.values())
        
        current_date = first_date + timedelta(days=30 * train_months)
        test_start = current_date
        test_end = current_date + timedelta(days=30 * test_months)
        
        iteration = 0
        while test_end <= last_date:
            iteration += 1
            self.logger.info(f"\n[Iteration {iteration}] Test: {test_start.date()} to {test_end.date()}")
            
            # Run simulation for this test period
            engine = PaperTradingEngine(self.config, data_source)
            result = engine.run(
                start_date=test_start.strftime("%Y-%m-%d"),
                end_date=test_end.strftime("%Y-%m-%d"),
            )
            
            self.results.append(result)
            
            # Slide window forward
            test_start = test_start + timedelta(days=30 * step_months)
            test_end = test_end + timedelta(days=30 * step_months)
        
        # Aggregate results
        return self._aggregate_results()
    
    def _aggregate_results(self) -> Dict:
        """Aggregate results from all iterations"""
        if not self.results:
            return {}
        
        total_return = sum(r.get('total_return', 0) for r in self.results)
        avg_return = total_return / len(self.results)
        
        total_trades = sum(r.get('total_trades_executed', 0) for r in self.results)
        win_rates = [r.get('win_rate', 0) for r in self.results]
        
        aggregated = {
            "iterations": len(self.results),
            "total_return": total_return,
            "avg_return_per_iteration": avg_return,
            "total_trades": total_trades,
            "avg_win_rate": sum(win_rates) / len(win_rates) if win_rates else 0,
            "iteration_results": self.results,
        }
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("WALK-FORWARD AGGREGATED RESULTS")
        self.logger.info("=" * 70)
        self.logger.info(f"Iterations: {aggregated['iterations']}")
        self.logger.info(f"Total Return (cumulative): {aggregated['total_return']:.2f}%")
        self.logger.info(f"Avg Return per Iteration: {aggregated['avg_return_per_iteration']:.2f}%")
        self.logger.info(f"Total Trades: {aggregated['total_trades']}")
        self.logger.info(f"Avg Win Rate: {aggregated['avg_win_rate']:.1f}%")
        self.logger.info("=" * 70 + "\n")
        
        return aggregated
