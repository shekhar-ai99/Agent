"""
Simulation Mode

Paper trading with realistic order fills and trade logging.
Uses ExecutionEngine for order/position/risk logic.

[Consolidates logic from:
 - IndianMarket/agentic_bot_fresh/main.py (paper trading)
 - IndianMarket/strategy_tester_app/main.py (simulation)
]
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from data.dataset_loader import DatasetLoader
from markets import IndianMarket, CryptoMarket
from strategies import instantiate_all_strategies
from core import StrategyContext, MarketRegime, VolatilityBucket
from brokers.paper_broker import PaperBroker
from execution.execution_engine import ExecutionEngine
from core import RiskManager, StrategySelector

logger = logging.getLogger(__name__)


class SimulationMode:
    """
    Run paper trading simulation.
    
    Features:
    - Virtual order fills with realistic latency/slippage
    - Trade journal logging
    - No real money or broker connections
    - Useful for testing strategy logic before live
    """
    
    def __init__(self, capital: float, market: str = "india", timeframe: str = "5min"):
        """
        Initialize simulation mode.
        
        Args:
            capital: Starting capital
            market: "india" or "crypto"
            timeframe: "1min", "5min", "15min", "1h", "daily"
        """
        self.capital = capital
        self.market_name = market
        self.timeframe = timeframe
        self.logger = logging.getLogger("SimulationMode")
        
        # Initialize market
        if market.lower() == "india":
            self.market = IndianMarket()
        elif market.lower() == "crypto":
            self.market = CryptoMarket()
        else:
            raise ValueError(f"Unknown market: {market}")
        
        # Initialize broker (paper but with realistic fills)
        self.broker = PaperBroker(initial_balance=capital, slippage_percent=0.02)
        
        # Initialize managers
        self.risk_manager = RiskManager()
        self.strategy_selector = StrategySelector()
        
        # Results tracking
        self.trade_journal: List[Dict] = []
        self.signal_log: List[Dict] = []
        self.error_log: List[Dict] = []
    
    def run(self, strategies: List[str], **kwargs) -> Dict[str, Any]:
        """
        Run simulation with given strategies.
        
        Args:
            strategies: List of strategy names to use
            **kwargs: Additional parameters
                - live: bool (True=use live data, False=use replayed historical)
                - duration: int (minutes, default=60)
                
        Returns:
            SimulationResults with trade journal and metrics
        """
        
        self.logger.info(f"Starting simulation: {self.market_name} {self.timeframe}")
        self.logger.info(f"Strategies: {strategies}")
        self.logger.info(f"Capital: {self.capital}")
        
        # For now, use replayed historical data
        use_live = kwargs.get("live", False)
        
        if use_live:
            self.logger.warning("Live mode not yet implemented, using replayed data")
        
        # Load data (use last N bars as if it's streaming)
        self.logger.info("Loading data for simulation...")
        data = self._load_data(**kwargs)
        
        if data.empty:
            self.logger.error("No data loaded")
            return {"error": "No data loaded"}
        
        self.logger.info(f"Loaded {len(data)} bars")
        
        # Validate strategies
        available_strategies = instantiate_all_strategies()
        selected_strategies = {}
        
        for strategy_name in strategies:
            if strategy_name not in available_strategies:
                self.logger.warning(f"Strategy {strategy_name} not found")
                continue
            selected_strategies[strategy_name] = available_strategies[strategy_name]
        
        if not selected_strategies:
            return {"error": "No valid strategies"}
        
        # Initialize execution engine
        execution_engine = ExecutionEngine(
            market=self.market,
            broker=self.broker,
            risk_manager=self.risk_manager,
            strategy_selector=self.strategy_selector,
            available_strategies=selected_strategies,
        )
        
        # Simulate bar-by-bar
        self.logger.info("Starting simulation...")
        bar_count = 0
        
        for timestamp, row in data.iterrows():
            bar_count += 1
            
            if bar_count % 100 == 0:
                self.logger.info(f"  Bar {bar_count}... | Equity: {self.broker.get_balance():.2f}")
            
            # Get regime/volatility  (same as backtest)
            regime = self._classify_regime(row)
            volatility = self._classify_volatility(row)
            
            # Get session
            session = self.market.get_session_name(timestamp) if hasattr(self.market, 'get_session_name') else "default"
            
            # Create context
            hist_start = max(0, data.index.get_loc(timestamp) - 100)
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
            )
            
            # Execute cycle
            try:
                execution_summary = execution_engine.execute_cycle(context)
                
                if execution_summary:
                    self.signal_log.append({
                        "timestamp": timestamp,
                        "signal": execution_summary,
                        "equity": self.broker.get_balance(),
                    })
            
            except Exception as e:
                self.logger.error(f"Error at {timestamp}: {e}")
                self.error_log.append({
                    "timestamp": timestamp,
                    "error": str(e),
                })
        
        self.logger.info(f"Simulation completed: {bar_count} bars")
        
        # Compute results
        results = self._compute_results()
        results["total_bars"] = bar_count
        results["strategies"] = list(selected_strategies.keys())
        
        return results
    
    def _load_data(self, **kwargs) -> pd.DataFrame:
        """Load data for simulation."""
        
        if self.market_name.lower() == "india":
            try:
                data = DatasetLoader.load_nifty(self.timeframe)
            except Exception as e:
                self.logger.error(f"Failed to load data: {e}")
                return pd.DataFrame()
        else:
            try:
                data = DatasetLoader.load_nifty(self.timeframe)
            except Exception as e:
                self.logger.error(f"Failed to load data: {e}")
                return pd.DataFrame()
        
        # Limit to last N bars (more realistic for live scenario)
        max_bars = kwargs.get("max_bars", 1000)
        if len(data) > max_bars:
            data = data.tail(max_bars)
        
        # Add indicators
        try:
            data = DatasetLoader.add_indicators(data)
        except Exception as e:
            self.logger.warning(f"Could not add indicators: {e}")
        
        return data
    
    def _classify_regime(self, row: pd.Series) -> MarketRegime:
        """Classify regime (same as backtest)."""
        
        sma20 = row.get("SMA_20", 0)
        sma50 = row.get("SMA_50", 0)
        adx = row.get("ADX_14", 0)
        atr = row.get("ATR", 0)
        close = row.get("close", 0)
        
        if close == 0 or sma50 == 0:
            return MarketRegime.UNKNOWN
        
        atr_pct = (atr / close) * 100 if close > 0 else 0
        if atr_pct > 2:
            return MarketRegime.VOLATILE
        
        if adx > 25:
            return MarketRegime.TRENDING
        
        ma_distance = abs(sma20 - sma50) / sma50 * 100 if sma50 > 0 else 0
        if ma_distance < 1:
            return MarketRegime.RANGING
        
        return MarketRegime.UNKNOWN
    
    def _classify_volatility(self, row: pd.Series) -> VolatilityBucket:
        """Classify volatility (same as backtest)."""
        
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
        """Check if expiry day (india only)."""
        
        if self.market_name.lower() != "india":
            return False
        
        if timestamp.weekday() != 3:  # Thursday
            return False
        
        month_start = pd.Timestamp(timestamp.year, timestamp.month, 1)
        if timestamp.tzinfo is not None:
            month_start = month_start.tz_localize(timestamp.tzinfo)
        month_end = month_start + pd.DateOffset(months=1)
        days_til_month_end = (month_end - timestamp).days
        
        return days_til_month_end <= 7
    
    def _compute_results(self) -> Dict[str, Any]:
        """Compute simulation metrics."""
        
        final_equity = self.broker.get_balance()
        pnl = final_equity - self.capital
        return_pct = (pnl / self.capital * 100) if self.capital > 0 else 0
        
        return {
            "initial_capital": self.capital,
            "final_equity": final_equity,
            "total_pnl": pnl,
            "return_pct": return_pct,
            "num_signals": len(self.signal_log),
            "num_errors": len(self.error_log),
            "signal_log": self.signal_log[-10:],  # Last 10 signals
            "errors": self.error_log[-5:] if self.error_log else [],
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run simple simulation
    sim = SimulationMode(capital=100_000, market="india", timeframe="5min")
    results = sim.run(
        strategies=["RSI_MeanReversion", "MA_Crossover"],
        live=False,
        max_bars=500,
    )
    
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    for key, value in results.items():
        if key not in ["signal_log", "errors"]:
            if isinstance(value, float):
                print(f"{key:30s}: {value:>15,.2f}")
            else:
                print(f"{key:30s}: {value}")
    print("=" * 60)
