"""
Backtest Mode

Runs historical data through all strategies and computes performance metrics.
Uses ExecutionEngine internally for order/position/risk logic.

[Consolidates logic from:
 - Bitcoin/bitcoin_strategy_tester_package/bitcoin_strategy_tester/backtester.py
 - IndianMarket/strategy_tester_app/main.py
 - Common/backtest_runner.py
]
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from data.dataset_loader import DatasetLoader
from markets import IndianMarket, CryptoMarket
from strategies import STRATEGY_REGISTRY, instantiate_all_strategies
from core import StrategyContext, MarketRegime, VolatilityBucket
from core.base_selector import StrategyScore
from brokers.paper_broker import PaperBroker
from execution.execution_engine import ExecutionEngine
from core import RiskManager, StrategySelector
from analytics.report_generator import BacktestReportGenerator

logger = logging.getLogger(__name__)


class PassThroughSelector:
    """
    Selector that always returns the passed strategies.
    Used to bypass ranking for deterministic backtests.
    """

    def __init__(self, strategy_names: List[str]):
        self.strategy_names = strategy_names

    def select_strategies(
        self,
        market_type: str,
        regime: str,
        session: str,
        volatility: str,
        top_n: int = 3,
    ) -> List[StrategyScore]:
        return [
            StrategyScore(
                strategy_name=name,
                market_type=market_type,
                regime=regime,
                session=session,
                volatility=volatility,
                win_rate=0.0,
                trades_count=0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                confidence_score=1.0,
            )
            for name in self.strategy_names
        ]


class BacktestMode:
    """
    Run backtest on historical data.
    
    Approach:
    1. Load historical OHLCV data
    2. Add indicators
    3. Classify regime/volatility for each bar
    4. For each bar: Run strategies → Pick best → Place virtual order
    5. Aggregate metrics and return results
    """
    
    def __init__(self, capital: float, market: str = "india", timeframe: str = "5min"):
        """
        Initialize backtest mode.
        
        Args:
            capital: Starting capital
            market: "india" or "crypto"
            timeframe: "1min", "5min", "15min", "daily"
        """
        self.capital = capital
        self.market_name = market
        self.timeframe = timeframe
        self.logger = logging.getLogger("BacktestMode")
        
        # Initialize market
        if market.lower() == "india":
            self.market = IndianMarket()
        elif market.lower() == "crypto":
            self.market = CryptoMarket()
        else:
            raise ValueError(f"Unknown market: {market}")
        
        # Initialize broker (paper for backtest)
        self.broker = PaperBroker(initial_balance=capital)
        
        # Initialize managers
        self.risk_manager = RiskManager()
        self.strategy_selector = StrategySelector()
        
        # Track results
        self.trade_log: List[Dict] = []
        self.bar_log: List[Dict] = []
        self.equity_curve: List[float] = [capital]
    
    def run(self, strategies: List[str], **kwargs) -> Dict[str, Any]:
        """
        Run backtest with given strategies.
        
        Args:
            strategies: List of strategy names to use
            **kwargs: Additional parameters
                - start_date: "YYYY-MM-DD" (optional)
                - end_date: "YYYY-MM-DD" (optional)
                
        Returns:
            BacktestResults with metrics
        """
        
        self.logger.info(f"Starting backtest: {self.market_name} {self.timeframe}")
        self.logger.info(f"Strategies: {strategies}")
        self.logger.info(f"Capital: {self.capital}")
        
        # Load data
        self.logger.info("Loading data...")
        data = self._load_data(**kwargs)
        
        if data.empty:
            raise ValueError("No data loaded")
        
        self.logger.info(f"Loaded {len(data)} bars | {data.index.min()} to {data.index.max()}")
        
        # Validate strategies are available
        available_strategies = instantiate_all_strategies()
        selected_strategies = {}
        
        for strategy_name in strategies:
            if strategy_name not in available_strategies:
                self.logger.warning(f"Strategy {strategy_name} not found, skipping")
                continue
            selected_strategies[strategy_name] = available_strategies[strategy_name]
        
        if not selected_strategies:
            raise ValueError("No valid strategies provided")
        
        self.logger.info(f"Using {len(selected_strategies)} strategies")
        
        # Initialize execution engine
        execution_engine = ExecutionEngine(
            market=self.market,
            broker=self.broker,
            risk_manager=self.risk_manager,
            strategy_selector=self.strategy_selector,
            available_strategies=selected_strategies,
        )

        if kwargs.get("bypass_selector"):
            execution_engine.strategy_selector = PassThroughSelector(list(selected_strategies.keys()))
        
        # Run backtest bar-by-bar
        self.logger.info("Running backtest...")
        bar_count = 0

        debug_signals = bool(kwargs.get("debug_signals"))
        if debug_signals:
            self.signal_debug: List[Dict] = []
        
        for timestamp, row in data.iterrows():
            bar_count += 1
            
            if bar_count % 1000 == 0:
                self.logger.info(f"  Bar {bar_count}... | Trades: {len(self.trade_log)}")
            
            # Get regime/volatility
            regime = self._classify_regime(row)
            volatility = self._classify_volatility(row)
            
            # Get session name
            session = self.market.get_session_name(timestamp) if hasattr(self.market, 'get_session_name') else "default"
            
            # Create context
            # For backtest, historical_data is a recent window
            history_window = 100  # Last N bars
            hist_start = max(0, data.index.get_loc(timestamp) - history_window)
            hist_end = data.index.get_loc(timestamp) + 1
            historical_data = data.iloc[hist_start:hist_end].copy()
            
            context = StrategyContext(
                symbol="NIFTY50" if self.market_name == "india" else "BTC",
                market_type=self.market_name,
                timeframe=self.timeframe,
                current_bar=row,
                historical_data=historical_data,
                regime=regime,
                volatility=volatility,
                session=session,
                is_expiry_day=self._is_expiry_day(timestamp),
                timestamp=pd.Timestamp(timestamp),
                additional_info={"relax_entry": bool(kwargs.get("relax_entry"))},
            )

            if debug_signals:
                for strategy_name, strategy in selected_strategies.items():
                    signal = strategy.execute_signal_generation(context)
                    self.signal_debug.append(
                        {
                            "timestamp": timestamp,
                            "strategy": strategy_name,
                            "direction": signal.direction,
                            "confidence": signal.confidence,
                            "reason": signal.reasoning,
                        }
                    )
            
            # Execute cycle
            execution_summary = execution_engine.execute_cycle(context)
            
            # Log trade execution with detailed info
            if execution_summary:
                if 'signal' in execution_summary:  # New position opened
                    self.logger.info(
                        f"✅ ENTRY: {execution_summary['strategy']} | "
                        f"{execution_summary['signal']} {execution_summary['position_size']} {execution_summary['symbol']} @ "
                        f"${execution_summary['entry_price']:.2f} | "
                        f"Confidence: {execution_summary['confidence']:.2f} | "
                        f"Regime: {execution_summary['regime']} | Volatility: {execution_summary['volatility']}"
                    )
                elif 'exit_reason' in execution_summary:  # Position closed
                    self.logger.info(
                        f"❌ EXIT: {execution_summary['strategy']} | "
                        f"{execution_summary['direction']} {execution_summary['quantity']} {execution_summary['symbol']} @ "
                        f"${execution_summary['exit_price']:.2f} | "
                        f"PnL: ${execution_summary['pnl']:.2f} ({execution_summary['pnl_percent']:.2f}%) | "
                        f"Reason: {execution_summary['exit_reason']}"
                    )
            
            # Track equity
            current_equity = self.broker.get_balance()
            self.equity_curve.append(current_equity)
            
            # Log bar
            self.bar_log.append({
                "timestamp": timestamp,
                "close": row["close"],
                "equity": current_equity,
                "trades_count": len(self.broker.get_orders()),
                "regime": regime.value,
                "volatility": volatility.value,
            })

        if kwargs.get("force_close_positions"):
            self._force_close_open_positions(data)
        
        # Get final results
        self.logger.info(f"Backtest completed: {bar_count} bars processed")
        
        results = self._compute_metrics()
        results["total_bars"] = bar_count
        results["strategies"] = list(selected_strategies.keys())

        if kwargs.get("save_pnl"):
            results["pnl_csv"] = self._save_pnl_data(kwargs.get("save_pnl_path"))

        if kwargs.get("save_strategy_pnl"):
            results["strategy_pnl_csv"] = self._save_strategy_pnl(kwargs.get("strategy_pnl_path"))

        if kwargs.get("generate_html_report"):
            results["html_report"] = self._generate_html_report(
                market=self.market_name,
                timeframe=self.timeframe,
                output_path=kwargs.get("html_report_path")
            )

        if debug_signals:
            print("\nFirst 50 signals (debug):")
            for i, entry in enumerate(self.signal_debug[:50], 1):
                print(
                    f"{i:02d} | {entry['timestamp']} | {entry['strategy']} | "
                    f"{entry['direction']} | {entry['confidence']:.2f} | {entry['reason']}"
                )
        
        self.logger.info(f"Final P&L: {results.get('total_pnl', 0):.2f}")
        self.logger.info(f"Total Trades: {results.get('num_trades', 0)}")
        
        return results

    def _save_pnl_data(self, output_path: Optional[str] = None) -> str:
        """Save per-bar equity and PnL data to CSV."""
        if not self.bar_log:
            return ""

        df = pd.DataFrame(self.bar_log)
        df["pnl"] = df["equity"] - self.capital

        if output_path:
            path = Path(output_path)
        else:
            runs_dir = Path(__file__).resolve().parents[2] / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = runs_dir / f"backtest_pnl_{timestamp}.csv"

        df.to_csv(path, index=False)
        return str(path)

    def _save_strategy_pnl(self, output_path: Optional[str] = None) -> str:
        """Save per-strategy PnL summary to CSV."""
        trade_history = self.broker.get_trade_history()
        if not trade_history:
            return ""

        df = pd.DataFrame(trade_history)
        if "strategy" not in df.columns or "pnl" not in df.columns:
            return ""

        summary = df.groupby("strategy").agg(
            total_pnl=("pnl", "sum"),
            trades=("pnl", "count"),
            avg_pnl=("pnl", "mean"),
        ).reset_index()

        if output_path:
            path = Path(output_path)
        else:
            runs_dir = Path(__file__).resolve().parents[2] / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = runs_dir / f"backtest_strategy_pnl_{timestamp}.csv"

        summary.to_csv(path, index=False)
        return str(path)

    def _generate_html_report(self, market: str, timeframe: str, output_path: Optional[str] = None) -> str:
        """Generate HTML report with market/timeframe breakdown."""
        trade_history = self.broker.get_trade_history()
        
        generator = BacktestReportGenerator(
            results=self._compute_metrics(),
            trade_history=trade_history,
            equity_curve=self.equity_curve
        )
        
        path = Path(output_path) if output_path else None
        report_path = generator.generate_html(market, timeframe, path)
        
        self.logger.info(f"HTML report generated: {report_path}")
        return report_path

    def _force_close_open_positions(self, data: pd.DataFrame) -> None:
        """Close any open positions at the final bar price."""
        if data.empty:
            return

        last_close = data["close"].iloc[-1]
        if pd.isna(last_close):
            return

        for pos in list(self.broker.get_open_positions()):
            self.broker.close_position(
                symbol=pos.symbol,
                exit_price=last_close,
                exit_reason="forced_end_of_backtest",
            )
    
    def _load_data(self, **kwargs) -> pd.DataFrame:
        """
        Load historical data.
        
        Supports:
        - NIFTY50 from CSV (India)
        - BTC from CSV (Crypto)
        """
        
        if self.market_name.lower() == "india":
            try:
                data = DatasetLoader.load_nifty(self.timeframe)
            except Exception as e:
                self.logger.error(f"Failed to load NIFTY data: {e}")
                return pd.DataFrame()
        
        elif self.market_name.lower() == "crypto":
            try:
                # For crypto, we'll use Bitcoin data if available
                # Fallback to NIFTY for now
                data = DatasetLoader.load_nifty(self.timeframe)
            except Exception as e:
                self.logger.error(f"Failed to load crypto data: {e}")
                return pd.DataFrame()
        
        else:
            return pd.DataFrame()
        
        # Filter by date range if provided
        if "start_date" in kwargs and kwargs["start_date"]:
            start = pd.to_datetime(kwargs["start_date"])
            if data.index.tz is not None and start.tzinfo is None:
                start = start.tz_localize(data.index.tz)
            data = data[data.index >= start]
        
        if "end_date" in kwargs and kwargs["end_date"]:
            end = pd.to_datetime(kwargs["end_date"])
            if data.index.tz is not None and end.tzinfo is None:
                end = end.tz_localize(data.index.tz)
            data = data[data.index <= end]
        
        # Add indicators
        try:
            data = DatasetLoader.add_indicators(data)
        except Exception as e:
            self.logger.warning(f"Could not add indicators: {e}")
        
        return data
    
    def _classify_regime(self, row: pd.Series) -> MarketRegime:
        """
        Classify market regime based on technical indicators.
        
        Rules:
        - TRENDING: SMA20 > SMA50 OR SMA20 < SMA50, ADX > 25
        - RANGING: SMA20 ≈ SMA50, ADX < 25
        - VOLATILE: high ATR or high standard deviation
        """
        
        sma20 = row.get("SMA_20", 0)
        sma50 = row.get("SMA_50", 0)
        adx = row.get("ADX_14", 0)
        atr = row.get("ATR", 0)
        close = row.get("close", 0)
        
        # Avoid division by zero
        if close == 0 or sma50 == 0:
            return MarketRegime.UNKNOWN
        
        # Check volatility first
        atr_pct = (atr / close) * 100 if close > 0 else 0
        if atr_pct > 2:  # More than 2% ATR
            return MarketRegime.VOLATILE
        
        # Check trend
        if adx > 25:  # Strong trend
            return MarketRegime.TRENDING
        
        # Check range
        ma_distance = abs(sma20 - sma50) / sma50 * 100 if sma50 > 0 else 0
        if ma_distance < 1:  # MAs are close
            return MarketRegime.RANGING
        
        return MarketRegime.UNKNOWN
    
    def _classify_volatility(self, row: pd.Series) -> VolatilityBucket:
        """
        Classify volatility level.
        
        Based on ATR as % of close price.
        """
        
        atr = row.get("ATR", 0)
        close = row.get("close", 1)
        
        if close <= 0:
            return VolatilityBucket.MEDIUM
        
        atr_pct = (atr / close) * 100
        
        if atr_pct < 0.5:
            return VolatilityBucket.LOW
        elif atr_pct < 2.0:
            return VolatilityBucket.MEDIUM
        else:
            return VolatilityBucket.HIGH
    
    def _is_expiry_day(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if this is an expiry day (for Indian market).
        
        Indian expiry: Last Thursday of each month
        """
        
        if self.market_name.lower() != "india":
            return False
        
        # Simple check: Is it Thursday with less than 7 days to end of month?
        if timestamp.weekday() != 3:  # Thursday = 3
            return False
        
        month_start = pd.Timestamp(timestamp.year, timestamp.month, 1)
        if timestamp.tzinfo is not None:
            month_start = month_start.tz_localize(timestamp.tzinfo)
        month_end = month_start + pd.DateOffset(months=1)
        days_til_month_end = (month_end - timestamp).days
        
        return days_til_month_end <= 7
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """
        Compute backtest metrics.
        
        Returns:
            Dict with:
            - total_pnl
            - return_pct
            - num_trades
            - win_rate
            - max_drawdown
            - sharpe_ratio
            - profit_factor
        """
        
        if hasattr(self.broker, "get_trade_history"):
            trades = self.broker.get_trade_history()
        else:
            trades = []
        final_equity = self.broker.get_balance()
        
        # Basic metrics
        total_pnl = final_equity - self.capital
        return_pct = (total_pnl / self.capital * 100) if self.capital > 0 else 0
        num_trades = len(trades)
        
        # Win rate
        if num_trades > 0:
            winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
            win_rate = (winning_trades / num_trades) * 100
        else:
            win_rate = 0
        
        # Drawdown
        max_equity = max(self.equity_curve) if self.equity_curve else self.capital
        min_equity = min(self.equity_curve) if self.equity_curve else self.capital
        max_drawdown = ((max_equity - min_equity) / max_equity * 100) if max_equity > 0 else 0
        
        # Profit factor
        if num_trades > 0:
            gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
            gross_loss = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        else:
            profit_factor = 0
        
        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        return {
            "total_pnl": total_pnl,
            "return_pct": return_pct,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "profit_factor": profit_factor,
            "final_equity": final_equity,
            "initial_capital": self.capital,
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run simple backtest
    backtest = BacktestMode(capital=100_000, market="india", timeframe="5min")
    results = backtest.run(
        strategies=["RSI_MeanReversion", "MA_Crossover"],
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:>15,.2f}")
        elif isinstance(value, list):
            print(f"{key:30s}: {len(value)} items")
        else:
            print(f"{key:30s}: {value}")
    print("=" * 60)
