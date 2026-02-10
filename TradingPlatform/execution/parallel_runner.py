"""
Parallel Market Runner

Executes trading operations across multiple markets simultaneously.
Each market runs in isolation with separate state, data feeds, and brokers.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
from pathlib import Path
import time
from datetime import datetime

# UnifiedTradingSystem will be integrated later
# from execution.unified_system import UnifiedTradingSystem

from config.settings import PlatformConfig

logger = logging.getLogger(__name__)


@dataclass
class MarketRunResult:
    """Result from running a single market"""
    market: str
    success: bool
    error_message: Optional[str]
    total_pnl: float
    num_trades: int
    runtime_seconds: float
    start_time: datetime
    end_time: datetime
    metrics: Dict[str, Any]


class ParallelMarketRunner:
    """
    Runs multiple markets in parallel with isolated state.
    
    **KEY PRINCIPLES:**
    - One market crash does NOT stop others
    - Each market has separate logs
    - Shared strategy registry
    - Separate data feeds and brokers
    - Thread-safe operation
    
    Use cases:
    - Run India + Crypto simultaneously
    - Run multiple timeframes in parallel
    - Run multiple symbols in parallel
    """
    
    def __init__(
        self,
        config: PlatformConfig,
        max_workers: Optional[int] = None
    ):
        """
        Initialize parallel runner.
        
        Args:
            config: Platform configuration
            max_workers: Max parallel workers (default: number of markets)
        """
        self.config = config
        self.max_workers = max_workers or len(config.enabled_markets)
        
        # Thread-safe storage for results
        self._results_lock = threading.Lock()
        self._market_results: Dict[str, MarketRunResult] = {}
        
        # Thread-safe storage for market systems
        self._systems_lock = threading.Lock()
        self._market_systems: Dict[str, Any] = {}  # Will hold UnifiedTradingSystem instances
        
        logger.info(
            f"ParallelMarketRunner initialized | "
            f"markets={config.enabled_markets} | max_workers={self.max_workers}"
        )
    
    def run_all_markets(
        self,
        mode: str = "simulation",
        timeframes: Optional[List[str]] = None,
        duration_days: Optional[int] = None
    ) -> Dict[str, MarketRunResult]:
        """
        Run all enabled markets in parallel.
        
        Args:
            mode: "simulation", "backtest", or "live"
            timeframes: List of timeframes to run (e.g., ["5min", "15min"])
            duration_days: Duration to run (for simulation mode)
            
        Returns:
            Dict mapping market name to result
        """
        logger.info("=" * 80)
        logger.info(f"PARALLEL MARKET EXECUTION | Mode: {mode}")
        logger.info(f"Markets: {self.config.enabled_markets}")
        logger.info(f"Timeframes: {timeframes or ['default']}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Create tasks for each market
        tasks = []
        for market in self.config.enabled_markets:
            for tf in (timeframes or [self.config.bar_frequency]):
                task = {
                    "market": market,
                    "timeframe": tf,
                    "mode": mode,
                    "duration_days": duration_days
                }
                tasks.append(task)
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._run_single_market, task): task
                for task in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    
                    with self._results_lock:
                        key = f"{task['market']}_{task['timeframe']}"
                        self._market_results[key] = result
                    
                    if result.success:
                        logger.info(
                            f"✅ {result.market} completed | "
                            f"P&L: ${result.total_pnl:.2f} | "
                            f"Trades: {result.num_trades} | "
                            f"Time: {result.runtime_seconds:.1f}s"
                        )
                    else:
                        logger.error(
                            f"❌ {result.market} failed | "
                            f"Error: {result.error_message}"
                        )
                        
                except Exception as e:
                    logger.error(f"CRITICAL: Task {task} raised exception: {e}")
                    
                    # Store failed result
                    with self._results_lock:
                        key = f"{task['market']}_{task['timeframe']}"
                        self._market_results[key] = MarketRunResult(
                            market=key,
                            success=False,
                            error_message=str(e),
                            total_pnl=0.0,
                            num_trades=0,
                            runtime_seconds=0.0,
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            metrics={}
                        )
        
        total_time = time.time() - start_time
        
        # Summary
        logger.info("=" * 80)
        logger.info("PARALLEL EXECUTION COMPLETE")
        logger.info(f"Total runtime: {total_time:.1f}s")
        logger.info(f"Markets completed: {len([r for r in self._market_results.values() if r.success])}/{len(self._market_results)}")
        
        # Aggregate metrics
        total_pnl = sum(r.total_pnl for r in self._market_results.values())
        total_trades = sum(r.num_trades for r in self._market_results.values())
        
        logger.info(f"Aggregate P&L: ${total_pnl:.2f}")
        logger.info(f"Aggregate Trades: {total_trades}")
        logger.info("=" * 80)
        
        return self._market_results
    
    def _run_single_market(self, task: Dict) -> MarketRunResult:
        """
        Run a single market in isolation.
        
        This is the worker function executed in each thread.
        """
        market = task["market"]
        timeframe = task["timeframe"]
        mode = task["mode"]
        duration_days = task.get("duration_days")
        
        market_key = f"{market}_{timeframe}"
        
        # Setup market-specific logging
        market_logger = logging.getLogger(f"Market.{market_key}")
        market_logger.setLevel(logging.INFO)
        
        market_logger.info(f"Starting {market} execution | TF: {timeframe} | Mode: {mode}")
        
        start_time = datetime.now()
        
        try:
            # Create isolated trading system for this market
            system = self._create_market_system(market, timeframe)
            
            with self._systems_lock:
                self._market_systems[market_key] = system
            
            # Run the system
            if mode == "backtest":
                results = self._run_backtest(system, market_logger)
            elif mode == "simulation":
                results = self._run_simulation(system, duration_days, market_logger)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds()
            
            return MarketRunResult(
                market=market_key,
                success=True,
                error_message=None,
                total_pnl=results.get("total_pnl", 0.0),
                num_trades=results.get("num_trades", 0),
                runtime_seconds=runtime,
                start_time=start_time,
                end_time=end_time,
                metrics=results
            )
            
        except Exception as e:
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds()
            
            market_logger.error(f"Market execution failed: {e}", exc_info=True)
            
            return MarketRunResult(
                market=market_key,
                success=False,
                error_message=str(e),
                total_pnl=0.0,
                num_trades=0,
                runtime_seconds=runtime,
                start_time=start_time,
                end_time=end_time,
                metrics={}
            )
    
    def _create_market_system(
        self,
        market: str,
        timeframe: str
    ) -> Any:  # Will return UnifiedTradingSystem
        """
        Create an isolated UnifiedTradingSystem for a market.
        
        Each system has:
        - Separate broker
        - Separate data loader
        - Separate execution engine
        - Shared strategy registry
        - Shared performance engine
        """
        # Create market-specific config
        market_config = self._make_market_config(market, timeframe)
        
        # Create system
        # Note: UnifiedTradingSystem integration pending
        raise NotImplementedError(
            "UnifiedTradingSystem integration pending. "
            "Will create separate system per market with isolated state."
        )
    
    def _make_market_config(self, market: str, timeframe: str) -> Dict:
        """Create market-specific configuration"""
        return {
            "market": market,
            "timeframe": timeframe,
            "capital": self.config.initial_account_balance,
            "risk_per_trade": self.config.risk_per_trade_percent,
            "max_positions": self.config.max_concurrent_trades,
            "log_level": self.config.log_level
        }
    
    def _run_backtest(
        self,
        system: Any,  # UnifiedTradingSystem
        logger: logging.Logger
    ) -> Dict:
        """Run backtest mode for a market"""
        logger.info("Starting backtest...")
        
        # Execute backtest
        # Placeholder - actual implementation depends on UnifiedTradingSystem API
        results = {
            "mode": "backtest",
            "total_pnl": 0.0,
            "num_trades": 0,
            "win_rate": 0.0
        }
        
        logger.info(f"Backtest complete | P&L: ${results['total_pnl']:.2f}")
        
        return results
    
    def _run_simulation(
        self,
        system: Any,  # UnifiedTradingSystem
        duration_days: Optional[int],
        logger: logging.Logger
    ) -> Dict:
        """Run simulation mode for a market"""
        logger.info(f"Starting simulation | Duration: {duration_days} days")
        
        # Execute simulation
        # Placeholder - actual implementation depends on UnifiedTradingSystem API
        results = {
            "mode": "simulation",
            "total_pnl": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "duration_days": duration_days
        }
        
        logger.info(f"Simulation complete | P&L: ${results['total_pnl']:.2f}")
        
        return results
    
    def get_market_system(self, market: str, timeframe: str) -> Optional[Any]:
        """Get the running system for a market (will return UnifiedTradingSystem)"""
        with self._systems_lock:
            return self._market_systems.get(f"{market}_{timeframe}")
    
    def get_all_results(self) -> Dict[str, MarketRunResult]:
        """Get all market results"""
        with self._results_lock:
            return self._market_results.copy()
    
    def get_aggregate_metrics(self) -> Dict:
        """Get aggregate metrics across all markets"""
        with self._results_lock:
            successful_markets = [r for r in self._market_results.values() if r.success]
            
            if not successful_markets:
                return {
                    "total_markets": 0,
                    "successful_markets": 0,
                    "total_pnl": 0.0,
                    "total_trades": 0,
                    "avg_runtime_s": 0.0
                }
            
            return {
                "total_markets": len(self._market_results),
                "successful_markets": len(successful_markets),
                "failed_markets": len(self._market_results) - len(successful_markets),
                "total_pnl": sum(r.total_pnl for r in successful_markets),
                "total_trades": sum(r.num_trades for r in successful_markets),
                "avg_runtime_s": sum(r.runtime_seconds for r in successful_markets) / len(successful_markets),
                "markets": [r.market for r in successful_markets]
            }
    
    def shutdown(self):
        """Gracefully shutdown all market systems"""
        logger.info("Shutting down parallel market runner...")
        
        with self._systems_lock:
            for market_key, system in self._market_systems.items():
                try:
                    # Placeholder for system shutdown
                    logger.info(f"Shutting down {market_key}")
                    # system.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down {market_key}: {e}")
        
        logger.info("Parallel market runner shutdown complete")
    
    def __repr__(self) -> str:
        metrics = self.get_aggregate_metrics()
        return (
            f"ParallelMarketRunner("
            f"markets={metrics['total_markets']}, "
            f"successful={metrics['successful_markets']}, "
            f"total_pnl=${metrics['total_pnl']:.2f})"
        )
