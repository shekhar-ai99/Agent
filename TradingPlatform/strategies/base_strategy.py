"""
BaseStrategy re-export for strategies module.

This file allows strategies to import BaseStrategy from TradingPlatform.strategies
while the actual implementation lives in TradingPlatform.core.
"""

from TradingPlatform.core.base_strategy import (
    BaseStrategy,
    Signal,
    StrategyContext,
    MarketRegime,
    VolatilityBucket,
)

__all__ = [
    "BaseStrategy",
    "Signal",
    "StrategyContext",
    "MarketRegime",
    "VolatilityBucket",
]
