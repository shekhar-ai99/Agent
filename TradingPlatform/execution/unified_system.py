"""
Unified Trading System Interface

Single entry point for backtest, simulation, and real trading modes.
All modes use the same ExecutionEngine and strategy logic internally.

Example usage:
    system = TradingSystem()
    
    # Backtest
    results = system.run(
        mode="backtest",
        market="india",
        timeframe="5min",
        strategies=["RSI_MeanReversion", "MA_Crossover"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        capital=100_000
    )
    
    # Simulation (paper trading)
    journal = system.run(
        mode="simulation",
        market="crypto",
        timeframe="15min",
        strategies=["SuperTrend_ADX_Trend"],
        capital=50_000,
        live=False  # Use replayed data for testing
    )
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class TradingSystem:
    """
    Unified interface for all trading modes.
    
    Modes:
    - backtest: Historical data, all strategies, fast
    - simulation: Paper trading, realistic fills, real-time
    - real: live trading (Phase 2+)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize TradingSystem.
        
        Args:
            config_path: Path to execution config YAML (optional)
        """
        self.config_path = config_path
        self.logger = logging.getLogger("TradingSystem")
        self.logger.info("TradingSystem initialized")
    
    def run(
        self,
        mode: str,
        market: str,
        timeframe: str,
        strategies: Optional[List[str]] = None,
        capital: float = 100_000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute trading in specified mode.
        
        Args:
            mode: "backtest", "simulation", or "real"
            market: "india", "crypto", or "india+crypto"
            timeframe: "1min", "5min", "15min", "1h", "daily"
            strategies: List of strategy names to use
            capital: Starting capital in units
            **kwargs: Mode-specific parameters
            
        Returns:
            Results dict with mode-specific structure
            
        Raises:
            ValueError: If mode is invalid or required parameters missing
        """
        
        # Validate inputs
        self._validate_inputs(mode, market, timeframe)
        
        # Default strategies if not provided
        if strategies is None:
            strategies = ["RSI_MeanReversion", "MA_Crossover"]
        
        self.logger.info(f"Starting {mode} on {market} {timeframe}")
        self.logger.info(f"Strategies: {strategies}")
        self.logger.info(f"Capital: {capital}")
        
        # Dispatch to mode handler
        if mode.lower() == "backtest":
            from modes.backtest_mode import BacktestMode
            mode_handler = BacktestMode(capital=capital, market=market, timeframe=timeframe)
            results = mode_handler.run(strategies=strategies, **kwargs)
        
        elif mode.lower() == "simulation":
            from modes.simulation_mode import SimulationMode
            mode_handler = SimulationMode(capital=capital, market=market, timeframe=timeframe)
            results = mode_handler.run(strategies=strategies, **kwargs)
        
        elif mode.lower() == "real":
            from modes.real_mode import RealMode
            mode_handler = RealMode(capital=capital, market=market, timeframe=timeframe)
            results = mode_handler.run(strategies=strategies, **kwargs)
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be backtest, simulation, or real")
        
        self.logger.info(f"{mode} completed successfully")
        return results
    
    def _validate_inputs(self, mode: str, market: str, timeframe: str) -> None:
        """Validate input parameters."""
        
        valid_modes = ["backtest", "simulation", "real"]
        if mode.lower() not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")
        
        valid_markets = ["india", "crypto", "india+crypto"]
        if market.lower() not in valid_markets:
            raise ValueError(f"market must be one of {valid_markets}, got {market}")
        
        valid_timeframes = ["1min", "5min", "15min", "1h", "daily"]
        if timeframe.lower() not in valid_timeframes:
            raise ValueError(f"timeframe must be one of {valid_timeframes}, got {timeframe}")
    
    def list_strategies(self) -> List[str]:
        """List all available strategies."""
        from strategies import STRATEGY_REGISTRY
        return list(STRATEGY_REGISTRY.keys())
    
    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """Get detailed information about a strategy."""
        from strategies import STRATEGY_REGISTRY
        
        if strategy_name not in STRATEGY_REGISTRY:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        strategy_class = STRATEGY_REGISTRY[strategy_name]
        instance = strategy_class()
        
        return {
            "name": instance.name,
            "version": instance.version,
            "required_indicators": instance.required_indicators(),
            "description": instance.__doc__ or "No description",
        }
    
    def compare_modes(self, strategies: List[str], market: str = "india") -> Dict[str, Any]:
        """
        Compare results across different modes (for testing).
        
        Runs same strategies in backtest and simulation, compares outputs.
        """
        self.logger.info(f"Comparing modes for {strategies}")
        
        # Run backtest
        backtest_results = self.run(
            mode="backtest",
            market=market,
            timeframe="5min",
            strategies=strategies,
            start_date="2024-01-01",
            end_date="2024-01-31",
        )
        
        # Run simulation (with replayed data)
        simulation_results = self.run(
            mode="simulation",
            market=market,
            timeframe="5min",
            strategies=strategies,
            live=False,  # Use replayed data
        )
        
        # Compare
        comparison = {
            "backtest_pnl": backtest_results.get("total_pnl"),
            "simulation_pnl": simulation_results.get("total_pnl"),
            "difference_pct": abs(
                (backtest_results.get("total_pnl", 0) - simulation_results.get("total_pnl", 0))
                / (backtest_results.get("total_pnl", 1) + 1e-10)
                * 100
            ),
            "backtest_trades": backtest_results.get("num_trades"),
            "simulation_trades": simulation_results.get("num_trades"),
        }
        
        return comparison


# Command-line interface
if __name__ == "__main__":
    import sys
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Unified Trading System")
    parser.add_argument("mode", choices=["backtest", "simulation", "real", "list-strategies"],
                        help="Trading mode or action")
    parser.add_argument("--market", default="india", choices=["india", "crypto", "india+crypto"],
                        help="Market to trade")
    parser.add_argument("--timeframe", default="5min", 
                        choices=["1min", "5min", "15min", "1h", "daily"],
                        help="Timeframe (default: 5min)")
    parser.add_argument("--strategies", default="RSI_MeanReversion,MA_Crossover",
                        help="Comma-separated strategy names")
    parser.add_argument("--capital", type=float, default=100_000,
                        help="Starting capital (default: 100000)")
    parser.add_argument("--start-date", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Backtest end date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Create system
    system = TradingSystem()
    
    # Handle list-strategies
    if args.mode == "list-strategies":
        print("\nAvailable Strategies:")
        print("=" * 60)
        for strategy_name in system.list_strategies():
            info = system.get_strategy_info(strategy_name)
            print(f"\n{strategy_name}")
            print(f"  Version: {info['version']}")
            print(f"  Indicators: {', '.join(info['required_indicators'])}")
        print("\n" + "=" * 60)
        sys.exit(0)
    
    # Parse strategies
    strategies = [s.strip() for s in args.strategies.split(",")]
    
    # Run requested mode
    try:
        kwargs = {}
        if args.start_date:
            kwargs["start_date"] = args.start_date
        if args.end_date:
            kwargs["end_date"] = args.end_date
        
        results = system.run(
            mode=args.mode,
            market=args.market,
            timeframe=args.timeframe,
            strategies=strategies,
            capital=args.capital,
            **kwargs
        )
        
        # Pretty-print results
        print("\n" + "=" * 60)
        print(f"{args.mode.upper()} RESULTS")
        print("=" * 60)
        
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key:30s}: {value:>15,.2f}")
            else:
                print(f"{key:30s}: {value}")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
