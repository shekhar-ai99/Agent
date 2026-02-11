"""
Strategies module - exposes the StrategyRegistry and BaseStrategy abstractions.
"""

from .strategy_registry import StrategyRegistry
# Re-export BaseStrategy abstractions so strategies can import from TradingPlatform.strategies
from TradingPlatform.core.base_strategy import (
    BaseStrategy,
    Signal,
    StrategyContext,
    MarketRegime,
    VolatilityBucket,
)

__all__ = [
    "StrategyRegistry",
    "BaseStrategy",
    "Signal",
    "StrategyContext",
    "MarketRegime",
    "VolatilityBucket",
]
