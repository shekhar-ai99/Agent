"""
Core module - contains fundamental abstractions for the trading platform.

Exports:
- BaseStrategy: Abstract strategy interface
- BaseMarket: Abstract market interface
- BaseBroker: Abstract broker interface
- RiskManager: Risk management logic
- StrategySelector: Strategy selection engine
"""

from .base_strategy import (
    BaseStrategy,
    Signal,
    StrategyContext,
    MarketRegime,
    VolatilityBucket,
)
from .base_market import (
    BaseMarket,
    MarketConfig,
    MarketSession,
    RiskMultipliers,
)
from .base_broker import (
    BaseBroker,
    Order,
    OrderStatus,
    OrderType,
    Position,
)
from .base_risk import RiskManager, RiskMetrics
from .base_selector import StrategySelector, SelectedStrategy

__all__ = [
    "BaseStrategy",
    "Signal",
    "StrategyContext",
    "MarketRegime",
    "VolatilityBucket",
    "BaseMarket",
    "MarketConfig",
    "MarketSession",
    "RiskMultipliers",
    "BaseBroker",
    "Order",
    "OrderStatus",
    "OrderType",
    "Position",
    "RiskManager",
    "RiskMetrics",
    "StrategySelector",
    "SelectedStrategy",
]
