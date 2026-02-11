"""
COMPREHENSIVE BACKTEST ENGINE - PHASE 1 & 2

Runs ALL generic strategies on NIFTY 5min historical data.
Computes and persists comprehensive performance metrics per strategy.
Prepares ranking data for live emulator.

Features:
- Auto-discovers all generic strategies
- Runs each strategy individually (isolated backtests)
- Computes detailed performance metrics
- Persists metrics to performance engine
- Generates ranking tables by context
- Handles NaN-safe indicator computation
- Safe execution with error handling per strategy
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset_loader import DatasetLoader
from strategies.strategy_registry import StrategyRegistry
from markets.indian_market import IndianMarket
from brokers.paper_broker import PaperBroker
from core.base_risk import RiskManager
from core.base_selector import StrategySelector
from execution.execution_engine import ExecutionEngine
from analytics.performance_engine import (
    StrategyPerformanceEngine,
    StrategyPerformanceMetrics,
    StrategyContext
)
from core.base_strategy import (
    Signal,
    StrategyContext as StrategySignalContext,
    MarketRegime,
    VolatilityBucket
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('comprehensive_backtest.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)


class ComprehensiveBacktester:
    """
    Runs comprehensive backtests across all strategies.
    
    Approach:
    1. Auto-discover all generic strategies
    2. Load NIFTY 5min data with all indicators
    3. For each strategy:
       - Run isolated backtest
       - Collect trades and performance
       - Compute comprehensive metrics
       - Persist to performance engine
    4. Generate ranking tables
    """
    
    def __init__(
        self,
        market: str = "india",
        symbol: str = "NIFTY50",
        timeframe: str = "5min",
        initial_capital: float = 100000.0,
        storage_dir: str = "./performance_data"
    ):
        """
        Initialize comprehensive backtester.
        
        Args:
            market: Market type
            symbol: Symbol to backtest
            timeframe: Data timeframe
            initial_capital: Starting capital per strategy
            storage_dir: Directory to store performance metrics
        """
        self.market_name = market
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.storage_dir = Path(storage_dir)
        
        # Initialize performance engine
        self.perf_engine = StrategyPerformanceEngine(storage_dir=str(self.storage_dir))
        
        # Load market
        self.market = IndianMarket()
        
        logger.info(f"ComprehensiveBacktester initialized")
        logger.info(f"  Market: {market}")
        logger.info(f"  Symbol: {symbol}")
        logger.info(f"  Timeframe: {timeframe}")
        logger.info(f"  Initial Capital: ${initial_capital:,.2f}")
        logger.info(f"  Storage: {self.storage_dir}")
    
    def load_data_with_indicators(self) -> pd.DataFrame:
        """Load historical data and compute all indicators."""
        logger.info("Loading historical data...")
        
        # Load NIFTY 5min data
        df = DatasetLoader.load_nifty(timeframe=self.timeframe)
        logger.info(f"Loaded {len(df)} bars | {df.index.min()} to {df.index.max()}")
        
        # Add all indicators
        logger.info("Computing indicators...")
        df = DatasetLoader.add_indicators(df)
        logger.info(f"Added indicators | Shape: {df.shape}")
        
        # Add regime and volatility classification
        logger.info("Classifying regime and volatility...")
        df = self._classify_market_context(df)
        
        return df
    
    def _classify_market_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each bar's market regime and volatility.
        
        Simple heuristics:
        - Regime: based on ADX and EMA alignment
        - Volatility: based on ATR percentile
        - Session: based on timestamp
        """
        df = df.copy()
        
        # Regime classification
        regime = []
        for idx, row in df.iterrows():
            if pd.isna(row.get('adx_14')):
                regime.append('unknown')
            elif row['adx_14'] > 25:
                regime.append('trending')
            elif row['adx_14'] > 15:
                regime.append('ranging')
            else:
                regime.append('choppy')
        
        df['regime'] = regime
        
        # Volatility classification (based on ATR percentile)
        if 'atr_14' in df.columns and not df['atr_14'].isna().all():
            atr_20pct = df['atr_14'].quantile(0.20)
            atr_80pct = df['atr_14'].quantile(0.80)
            
            volatility = []
            for atr_val in df['atr_14']:
                if pd.isna(atr_val):
                    volatility.append('medium')
                elif atr_val < atr_20pct:
                    volatility.append('low')
                elif atr_val > atr_80pct:
                    volatility.append('high')
                else:
                    volatility.append('medium')
            
            df['volatility'] = volatility
        else:
            df['volatility'] = 'medium'
        
        # Session classification
        session = []
        for ts in df.index:
            session_name = self.market.get_session_name(ts)
            session.append(session_name if session_name else 'Session_1')
        
        df['session'] = session
        
        logger.info(f"Classified {len(df)} bars")
        logger.info(f"  Regimes: {df['regime'].value_counts().to_dict()}")
        logger.info(f"  Volatility: {df['volatility'].value_counts().to_dict()}")
        logger.info(f"  Sessions: {df['session'].value_counts().to_dict()}")
        
        return df
    
    def run_strategy_backtest(
        self,
        strategy_name: str,
        strategy_instance: Any,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run backtest for a single strategy.
        
        Args:
            strategy_name: Name of the strategy
            strategy_instance: Instance of the strategy
            data: Historical data with indicators
            
        Returns:
            Dictionary with trade history and metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Running backtest: {strategy_name}")
        logger.info(f"{'='*80}")
        
        # Initialize broker for this strategy
        broker = PaperBroker(initial_balance=self.initial_capital)
        
        # Track state
        position = None
        trades = []
        equity_curve = [self.initial_capital]
        
        # Simulate bar-by-bar execution
        for i in range(len(data)):
            current_bar = data.iloc[i]
            current_timestamp = data.index[i]
            
            # Skip if insufficient history
            required_lookback = getattr(strategy_instance, 'required_lookback', 50)
            if i < required_lookback:
                equity_curve.append(equity_curve[-1])
                continue
            
            # Skip if key indicators are NaN
            if pd.isna(current_bar.get('close')):
                equity_curve.append(equity_curve[-1])
                continue
            
            # Get historical window
            history = data.iloc[:i+1].copy()
            
            # Build context
            try:
                context = StrategySignalContext(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    timestamp=current_timestamp,
                    current_price=float(current_bar['close']),
                    data=current_bar,
                    historical_data=history,
                    regime=MarketRegime(current_bar['regime']),
                    volatility=VolatilityBucket(current_bar['volatility']),
                    session=current_bar['session'],
                    market_type=self.market_name
                )
            except Exception as e:
                # Skip bars with invalid context (e.g., NaN indicators)
                equity_curve.append(equity_curve[-1])
                continue
            
            # Generate signal (NaN-safe)
            try:
                signal = strategy_instance.generate_signal(context)
            except Exception as e:
                logger.error(f"Signal generation failed for {strategy_name} at {current_timestamp}: {e}")
                signal = Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=strategy_name,
                    reasoning=f"Error: {str(e)}"
                )
            
            # Execute trade logic
            current_price = float(current_bar['close'])
            
            if position is None:
                # No position - check for entry
                if signal.direction == "BUY" and signal.confidence > 0.1:  # Lowered threshold
                    # Enter long position
                    position = {
                        'direction': 'LONG',
                        'entry_price': current_price,
                        'entry_time': current_timestamp,
                        'entry_reason': signal.reasoning,
                        'quantity': 1,
                    }
                    logger.debug(f"ENTER LONG {strategy_name} @ {current_price:.2f}")
                
                elif signal.direction == "SELL" and signal.confidence > 0.1:  # Lowered threshold
                    # Enter short position
                    position = {
                        'direction': 'SHORT',
                        'entry_price': current_price,
                        'entry_time': current_timestamp,
                        'entry_reason': signal.reasoning,
                        'quantity': 1,
                    }
                    logger.debug(f"ENTER SHORT {strategy_name} @ {current_price:.2f}")
            
            else:
                # Have position - check for exit
                exit_signal = False
                exit_reason = ""
                
                if position['direction'] == 'LONG':
                    # Exit long on opposite signal or stop loss
                    if signal.direction == "SELL":
                        exit_signal = True
                        exit_reason = "Opposite signal"
                    # Simple stop loss: -2%
                    elif current_price < position['entry_price'] * 0.98:
                        exit_signal = True
                        exit_reason = "Stop loss"
                    # Simple take profit: +3%
                    elif current_price > position['entry_price'] * 1.03:
                        exit_signal = True
                        exit_reason = "Take profit"
                
                elif position['direction'] == 'SHORT':
                    # Exit short on opposite signal or stop loss
                    if signal.direction == "BUY":
                        exit_signal = True
                        exit_reason = "Opposite signal"
                    # Simple stop loss: +2%
                    elif current_price > position['entry_price'] * 1.02:
                        exit_signal = True
                        exit_reason = "Stop loss"
                    # Simple take profit: -3%
                    elif current_price < position['entry_price'] * 0.97:
                        exit_signal = True
                        exit_reason = "Take profit"
                
                # Execute exit
                if exit_signal:
                    # Calculate P&L
                    if position['direction'] == 'LONG':
                        pnl = (current_price - position['entry_price']) * position['quantity']
                    else:  # SHORT
                        pnl = (position['entry_price'] - current_price) * position['quantity']
                    
                    # Record trade
                    trade = {
                        'strategy': strategy_name,
                        'symbol': self.symbol,
                        'direction': position['direction'],
                        'entry_time': position['entry_time'],
                        'entry_price': position['entry_price'],
                        'exit_time': current_timestamp,
                        'exit_price': current_price,
                        'quantity': position['quantity'],
                        'pnl': pnl,
                        'return_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100,
                        'exit_reason': exit_reason,
                        'regime': current_bar['regime'],
                        'volatility': current_bar['volatility'],
                        'session': current_bar['session'],
                    }
                    
                    trades.append(trade)
                    
                    logger.debug(
                        f"EXIT {position['direction']} {strategy_name} @ {current_price:.2f} | "
                        f"P&L: {pnl:.2f} | {exit_reason}"
                    )
                    
                    # Clear position
                    position = None
            
            # Update equity curve
            current_equity = equity_curve[-1]
            if trades:
                current_equity = self.initial_capital + sum(t['pnl'] for t in trades)
            equity_curve.append(current_equity)
        
        # Close any remaining position at end
        if position is not None:
            final_price = float(data.iloc[-1]['close'])
            if position['direction'] == 'LONG':
                pnl = (final_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - final_price) * position['quantity']
            
            trade = {
                'strategy': strategy_name,
                'symbol': self.symbol,
                'direction': position['direction'],
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': data.index[-1],
                'exit_price': final_price,
                'quantity': position['quantity'],
                'pnl': pnl,
                'return_pct': (pnl / (position['entry_price'] * position['quantity'])) * 100,
                'exit_reason': 'End of backtest',
                'regime': data.iloc[-1]['regime'],
                'volatility': data.iloc[-1]['volatility'],
                'session': data.iloc[-1]['session'],
            }
            trades.append(trade)
        
        # Compute summary metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
        
        profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades and avg_loss != 0 else 0.0
        
        logger.info(f"\n{strategy_name} Results:")
        logger.info(f"  Total Trades: {total_trades}")
        logger.info(f"  Win Rate: {win_rate*100:.2f}%")
        logger.info(f"  Total P&L: ${total_pnl:.2f}")
        logger.info(f"  Profit Factor: {profit_factor:.2f}")
        
        return {
            'strategy_name': strategy_name,
            'trades': trades,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'equity_curve': equity_curve,
        }
    
    def compute_comprehensive_metrics(
        self,
        strategy_name: str,
        backtest_results: Dict[str, Any],
        data: pd.DataFrame
    ) -> List[StrategyPerformanceMetrics]:
        """
        Compute comprehensive metrics per context.
        
        Returns metrics broken down by:
        - Overall
        - Per regime
        - Per session
        - Per volatility bucket
        """
        metrics_list = []
        
        trades_df = pd.DataFrame(backtest_results['trades']) if backtest_results['trades'] else pd.DataFrame()
        
        if trades_df.empty:
            logger.warning(f"No trades for {strategy_name}, skipping metrics")
            return metrics_list
        
        # Helper function to compute metrics for a trade subset
        def compute_metrics_for_subset(
            subset_trades: pd.DataFrame,
            context_regime: str = "ALL",
            context_session: str = "ALL",
            context_volatility: str = "ALL"
        ) -> StrategyPerformanceMetrics:
            
            if subset_trades.empty:
                # Return zero metrics
                return StrategyPerformanceMetrics(
                    strategy_name=strategy_name,
                    market=self.market_name,
                    timeframe=self.timeframe,
                    regime=context_regime,
                    session=context_session,
                    volatility_bucket=context_volatility,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    total_pnl=0.0,
                    avg_win=0.0,
                    avg_loss=0.0,
                    avg_pnl_per_trade=0.0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    sharpe_ratio=0.0,
                    sortino_ratio=0.0,
                    max_drawdown=0.0,
                    max_drawdown_pct=0.0,
                    avg_trade_duration_seconds=0.0,
                    consistency_score=0.0,
                    reliability_score=0.0,
                    last_updated=datetime.now().isoformat(),
                )
            
            total_trades = len(subset_trades)
            winning = subset_trades[subset_trades['pnl'] > 0]
            losing = subset_trades[subset_trades['pnl'] < 0]
            
            total_pnl = subset_trades['pnl'].sum()
            avg_win = winning['pnl'].mean() if len(winning) > 0 else 0.0
            avg_loss = losing['pnl'].mean() if len(losing) > 0 else 0.0
            avg_pnl = subset_trades['pnl'].mean()
            
            win_rate = len(winning) / total_trades if total_trades > 0 else 0.0
            profit_factor = abs((avg_win * len(winning)) / (avg_loss * len(losing))) if len(losing) > 0 and avg_loss != 0 else 0.0
            
            # Sharpe ratio (simple approximation)
            returns = subset_trades['return_pct']
            sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0.0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            sortino = returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0.0
            
            # Max drawdown (simple)
            cumulative_pnl = subset_trades['pnl'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = cumulative_pnl - running_max
            max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
            max_dd_pct = (max_dd / self.initial_capital) * 100 if self.initial_capital > 0 else 0.0
            
            # Trade duration
            subset_trades['duration'] = (subset_trades['exit_time'] - subset_trades['entry_time']).dt.total_seconds()
            avg_duration = subset_trades['duration'].mean() if 'duration' in subset_trades.columns else 0.0
            
            # Consistency score (how consistent are the returns?)
            consistency = max(0, 100 - (returns.std() * 10)) if returns.std() > 0 else 50.0
            
            # Reliability score (based on win rate and sample size)
            reliability = (win_rate * 100 * 0.7) + (min(total_trades / 50.0, 1.0) * 30)
            
            metrics = StrategyPerformanceMetrics(
                strategy_name=strategy_name,
                market=self.market_name,
                timeframe=self.timeframe,
                regime=context_regime,
                session=context_session,
                volatility_bucket=context_volatility,
                total_trades=int(total_trades),
                winning_trades=int(len(winning)),
                losing_trades=int(len(losing)),
                total_pnl=float(total_pnl),
                avg_win=float(avg_win),
                avg_loss=float(avg_loss),
                avg_pnl_per_trade=float(avg_pnl),
                win_rate=float(win_rate),
                profit_factor=float(profit_factor),
                sharpe_ratio=float(sharpe),
                sortino_ratio=float(sortino),
                max_drawdown=float(max_dd),
                max_drawdown_pct=float(max_dd_pct),
                avg_trade_duration_seconds=float(avg_duration),
                consistency_score=float(consistency),
                reliability_score=float(reliability),
                last_updated=datetime.now().isoformat(),
                first_trade_date=str(subset_trades['entry_time'].min()) if 'entry_time' in subset_trades.columns else None,
                last_trade_date=str(subset_trades['exit_time'].max()) if 'exit_time' in subset_trades.columns else None,
            )
            
            # Calculate composite score
            metrics.calculate_composite_score()
            
            return metrics
        
        # Overall metrics
        metrics_list.append(compute_metrics_for_subset(trades_df))
        
        # Per regime
        for regime in trades_df['regime'].unique():
            regime_trades = trades_df[trades_df['regime'] == regime]
            metrics_list.append(compute_metrics_for_subset(regime_trades, context_regime=regime))
        
        # Per session
        for session in trades_df['session'].unique():
            session_trades = trades_df[trades_df['session'] == session]
            metrics_list.append(compute_metrics_for_subset(session_trades, context_session=session))
        
        # Per volatility
        for vol in trades_df['volatility'].unique():
            vol_trades = trades_df[trades_df['volatility'] == vol]
            metrics_list.append(compute_metrics_for_subset(vol_trades, context_volatility=vol))
        
        return metrics_list
    
    def run_all_strategies(self) -> Dict[str, Any]:
        """
        Main execution: Run backtests for all strategies.
        
        Returns:
            Summary results
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE BACKTEST ENGINE - PHASE 1 & 2")
        print("="*80)
        print(f"\nMarket: {self.market_name}")
        print(f"Symbol: {self.symbol}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print("\n" + "="*80)
        
        # Load data with indicators
        data = self.load_data_with_indicators()
        
        # Discover all strategies
        logger.info("\nDiscovering strategies...")
        all_strategies = StrategyRegistry.get_all()
        logger.info(f"Found {len(all_strategies)} strategies")
        
        # Results tracking
        all_results = {}
        successful_strategies = []
        failed_strategies = []
        
        # Run each strategy
        for i, (strategy_class_name, strategy_class) in enumerate(sorted(all_strategies.items()), 1):
            print(f"\n[{i}/{len(all_strategies)}] Processing: {strategy_class_name}")
            
            try:
                # Instantiate strategy
                strategy_instance = strategy_class()
                strategy_name = strategy_instance.name
                
                # Run backtest
                backtest_results = self.run_strategy_backtest(
                    strategy_name=strategy_name,
                    strategy_instance=strategy_instance,
                    data=data
                )
                
                # Compute comprehensive metrics
                metrics_list = self.compute_comprehensive_metrics(
                    strategy_name=strategy_name,
                    backtest_results=backtest_results,
                    data=data
                )
                
                # Persist metrics to performance engine
                for metrics in metrics_list:
                    self.perf_engine.performance_db[
                        self.perf_engine._make_key(
                            metrics.strategy_name,
                            metrics.market,
                            metrics.timeframe,
                            metrics.regime,
                            metrics.session,
                            metrics.volatility_bucket
                        )
                    ] = metrics
                
                # Persist to disk
                self.perf_engine.persist()
                
                all_results[strategy_name] = {
                    'backtest': backtest_results,
                    'metrics': metrics_list
                }
                
                successful_strategies.append(strategy_name)
                logger.info(f"✅ {strategy_name} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ {strategy_class_name} failed: {e}")
                logger.error(traceback.format_exc())
                failed_strategies.append(strategy_class_name)
        
        # Generate summary report
        print("\n" + "="*80)
        print("BACKTEST SUMMARY")
        print("="*80)
        print(f"\nTotal Strategies: {len(all_strategies)}")
        print(f"Successful: {len(successful_strategies)}")
        print(f"Failed: {len(failed_strategies)}")
        
        if successful_strategies:
            print(f"\n✅ Successful Strategies:")
            for name in successful_strategies:
                result = all_results.get(name)
                if result:
                    bt = result['backtest']
                    print(f"  • {name}: {bt['total_trades']} trades, "
                          f"${bt['total_pnl']:.2f} P&L, "
                          f"{bt['win_rate']*100:.1f}% WR")
        
        if failed_strategies:
            print(f"\n❌ Failed Strategies:")
            for name in failed_strategies:
                print(f"  • {name}")
        
        # Show performance data location
        print(f"\n📊 Performance Data:")
        print(f"  CSV: {self.perf_engine.csv_path}")
        print(f"  JSON: {self.perf_engine.json_path}")
        
        print("\n" + "="*80)
        print("TOP 10 STRATEGIES (by composite score)")
        print("="*80)
        
        # Query top strategies
        top_strategies = self.perf_engine.get_best_strategies(
            market=self.market_name,
            timeframe=self.timeframe,
            regime="ALL",
            min_trades=5,
            top_n=10
        )
        
        if top_strategies:
            print(f"\n{'Rank':<6} {'Strategy':<30} {'Score':<8} {'Trades':<8} {'Win Rate':<10} {'P&L':<12}")
            print("-"*80)
            for rank, metrics in enumerate(top_strategies, 1):
                print(
                    f"{rank:<6} {metrics.strategy_name:<30} "
                    f"{metrics.composite_score:<8.2f} {metrics.total_trades:<8} "
                    f"{metrics.win_rate*100:<10.1f} ${metrics.total_pnl:<12.2f}"
                )
        else:
            print("No top strategies found")
        
        print("\n" + "="*80)
        print("✅ COMPREHENSIVE BACKTEST COMPLETE")
        print("="*80 + "\n")
        
        return {
            'total_strategies': len(all_strategies),
            'successful': len(successful_strategies),
            'failed': len(failed_strategies),
            'results': all_results,
            'top_strategies': top_strategies,
        }


def main():
    """Main entry point"""
    backtester = ComprehensiveBacktester(
        market="india",
        symbol="NIFTY50",
        timeframe="5min",
        initial_capital=100000.0,
        storage_dir="./performance_data"
    )
    
    results = backtester.run_all_strategies()
    
    return results


if __name__ == "__main__":
    main()
