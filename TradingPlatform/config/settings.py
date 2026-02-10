"""
Global configuration for the trading platform.
Centralizes all configurable parameters.

**CRITICAL: All runtime behavior must be controlled via config.**
**NO magic numbers. NO hardcoded switches.**
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlatformConfig:
    """
    Main platform configuration.
    
    Controls all aspects of the trading system:
    - Markets and symbols
    - Strategies and selection
    - Risk management
    - Execution modes
    - Performance tracking
    - Parallel execution
    - Logging and output
    """
    
    # === Account & Risk ===
    initial_account_balance: float = 100000.0
    max_position_size_percent: float = 5.0
    max_concurrent_trades: int = 5
    max_daily_loss_percent: float = 2.0
    risk_per_trade_percent: float = 1.0
    
    # === Markets ===
    enabled_markets: List[str] = None  # ["india", "crypto"]
    india_symbols: List[str] = None  # ["NIFTY50", "BANKNIFTY"]
    crypto_symbols: List[str] = None  # ["BTCUSDT", "ETHUSDT"]
    
    # === Strategies ===
    enabled_strategies: List[str] = None  # Names of strategies to enable (empty = all)
    strategy_min_trades_for_ranking: int = 10  # Min trades to include in ranking
    strategy_selection_top_n: int = 3  # Number of strategies to select per context
    strategy_auto_discovery: bool = True  # Auto-discover strategies from registry
    
    # === Timeframes ===
    enabled_timeframes: List[str] = None  # ["1min", "5min", "15min"]
    default_timeframe: str = "5min"
    
    # === Simulation & Backtest ===
    simulation_start_date: str = "2024-01-01"
    simulation_end_date: str = "2024-12-31"
    simulation_mode: str = "paper"  # "paper" or "backtest"
    backtest_initial_capital: float = 100000.0
    
    # === Long-Run Simulation ===
    long_run_checkpoint_frequency_days: int = 7  # Checkpoint every N days
    long_run_enable_daily_reports: bool = True
    long_run_enable_monthly_reports: bool = True
    
    # === Parallel Execution ===
    parallel_max_workers: Optional[int] = None  # None = auto (one per market)
    parallel_enable: bool = True  # Whether to run markets in parallel
    
    # === Performance Tracking ===
    performance_engine_storage_dir: str = "./performance_data"
    performance_auto_save_frequency_trades: int = 50  # Auto-save every N trades
    performance_enable_persistence: bool = True
    
    # === Logging & Output ===
    log_level: str = "INFO"
    log_file: Optional[str] = "trading.log"
    log_to_console: bool = True
    output_dir: str = "./results"
    save_performance_table: bool = True
    save_trade_history: bool = True
    save_html_reports: bool = True
    
    # === Broker ===
    broker_type: str = "paper"  # Only "paper" for now
    slippage_percent: float = 0.0
    execution_mode: str = "instant"  # "instant" or "next_bar"
    commission_per_trade: float = 0.0
    
    # === Data Sources ===
    data_base_dir: str = "./datasets"
    data_cache_enabled: bool = True
    
    # === Safety & Validation ===
    enable_sanity_checks: bool = True
    fail_fast_on_errors: bool = True  # Stop on critical errors
    max_consecutive_losses: int = 10  # Halt if hit consecutive losses
    emergency_stop_loss_pct: float = 10.0  # Halt if account drops by this %
    
    # === Feature Flags ===
    enable_regime_detection: bool = True
    enable_volatility_filtering: bool = True
    enable_session_filtering: bool = True
    enable_dynamic_position_sizing: bool = True
    
    def __post_init__(self):
        """Set defaults for list fields and validate configuration"""
        # Set default lists
        if self.enabled_markets is None:
            self.enabled_markets = ["india", "crypto"]
        
        if self.india_symbols is None:
            self.india_symbols = ["NIFTY50"]
        
        if self.crypto_symbols is None:
            self.crypto_symbols = ["BTCUSDT"]
        
        if self.enabled_strategies is None:
            self.enabled_strategies = []  # Empty = all strategies
        
        if self.enabled_timeframes is None:
            self.enabled_timeframes = ["5min", "15min"]
        
        # Validate configuration
        if self.enable_sanity_checks:
            self._validate()
    
    def _validate(self):
        """Validate configuration sanity"""
        errors = []
        
        # Validate markets
        valid_markets = ["india", "crypto"]
        for market in self.enabled_markets:
            if market not in valid_markets:
                errors.append(f"Invalid market: {market}. Must be one of {valid_markets}")
        
        # Validate risk parameters
        if self.risk_per_trade_percent <= 0 or self.risk_per_trade_percent > 10:
            errors.append(f"risk_per_trade_percent must be between 0 and 10, got {self.risk_per_trade_percent}")
        
        if self.max_concurrent_trades < 1:
            errors.append(f"max_concurrent_trades must be >= 1, got {self.max_concurrent_trades}")
        
        if self.max_daily_loss_percent <= 0 or self.max_daily_loss_percent > 50:
            errors.append(f"max_daily_loss_percent must be between 0 and 50, got {self.max_daily_loss_percent}")
        
        # Validate balance
        if self.initial_account_balance <= 0:
            errors.append(f"initial_account_balance must be positive, got {self.initial_account_balance}")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            errors.append(f"log_level must be one of {valid_log_levels}, got {self.log_level}")
        
        # Validate strategy selection
        if self.strategy_selection_top_n < 1:
            errors.append(f"strategy_selection_top_n must be >= 1, got {self.strategy_selection_top_n}")
        
        # Raise if errors found
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            if self.fail_fast_on_errors:
                raise ValueError(error_msg)
            else:
                logger.warning(error_msg)
    
    def get_symbols_for_market(self, market: str) -> List[str]:
        """Get symbol list for a specific market"""
        if market == "india":
            return self.india_symbols
        elif market == "crypto":
            return self.crypto_symbols
        else:
            logger.warning(f"Unknown market: {market}")
            return []
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            # Account & Risk
            "initial_account_balance": self.initial_account_balance,
            "max_position_size_percent": self.max_position_size_percent,
            "max_concurrent_trades": self.max_concurrent_trades,
            "max_daily_loss_percent": self.max_daily_loss_percent,
            "risk_per_trade_percent": self.risk_per_trade_percent,
            
            # Markets
            "enabled_markets": self.enabled_markets,
            "india_symbols": self.india_symbols,
            "crypto_symbols": self.crypto_symbols,
            
            # Strategies
            "enabled_strategies": self.enabled_strategies,
            "strategy_min_trades_for_ranking": self.strategy_min_trades_for_ranking,
            "strategy_selection_top_n": self.strategy_selection_top_n,
            "strategy_auto_discovery": self.strategy_auto_discovery,
            
            # Timeframes
            "enabled_timeframes": self.enabled_timeframes,
            "default_timeframe": self.default_timeframe,
            
            # Simulation
            "simulation_start_date": self.simulation_start_date,
            "simulation_end_date": self.simulation_end_date,
            "simulation_mode": self.simulation_mode,
            
            # Parallel
            "parallel_max_workers": self.parallel_max_workers,
            "parallel_enable": self.parallel_enable,
            
            # Logging
            "log_level": self.log_level,
            "log_file": self.log_file,
            "output_dir": self.output_dir,
            
            # Safety
            "enable_sanity_checks": self.enable_sanity_checks,
            "fail_fast_on_errors": self.fail_fast_on_errors,
            "max_consecutive_losses": self.max_consecutive_losses,
            "emergency_stop_loss_pct": self.emergency_stop_loss_pct
        }
    
    def __repr__(self) -> str:
        return (
            f"PlatformConfig("
            f"markets={self.enabled_markets}, "
            f"balance=${self.initial_account_balance}, "
            f"risk={self.risk_per_trade_percent}%)"
        )


# Global instance
config = PlatformConfig()


def load_from_yaml(filepath: str) -> PlatformConfig:
    """Load configuration from YAML file"""
    try:
        import yaml
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        cfg = PlatformConfig(**data)
        logger.info(f"Configuration loaded from {filepath}")
        return cfg
        
    except Exception as e:
        logger.error(f"Failed to load config from {filepath}: {e}")
        logger.warning("Using default configuration")
        return config


def load_from_json(filepath: str) -> PlatformConfig:
    """Load configuration from JSON file"""
    try:
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        cfg = PlatformConfig(**data)
        logger.info(f"Configuration loaded from {filepath}")
        return cfg
        
    except Exception as e:
        logger.error(f"Failed to load config from {filepath}: {e}")
        logger.warning("Using default configuration")
        return config


def save_to_yaml(config: PlatformConfig, filepath: str):
    """Save configuration to YAML file"""
    try:
        import yaml
        with open(filepath, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save config to {filepath}: {e}")


def save_to_json(config: PlatformConfig, filepath: str):
    """Save configuration to JSON file"""
    try:
        import json
        with open(filepath, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save config to {filepath}: {e}")

