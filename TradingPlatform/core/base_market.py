"""
Base Market Abstract Class

Markets define trading rules, session logic, expiry logic, and risk parameters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketSession:
    """Defines a trading session within a market"""
    name: str  # "Morning", "Midday", "Afternoon", "Pre-Market", etc.
    start_time: time
    end_time: time
    allow_trading: bool = True
    strategy_restrictions: Optional[List[str]] = None  # Strategies allowed in this session


@dataclass
class RiskMultipliers:
    """Market-specific risk parameter multipliers"""
    sl_atr_multiple: float  # SL = current_price ± (ATR * this)
    tp_atr_multiple: float  # TP = current_price ± (ATR * this)
    tsl_atr_multiple: float  # TSL = (ATR * this)
    max_position_size: float  # % of account
    max_concurrent_trades: int


@dataclass
class MarketConfig:
    """Market-wide configuration"""
    symbol: str
    market_type: str  # "indian", "crypto"
    tick_size: float  # Minimum price movement (e.g., 0.05 for NSE)
    lot_size: int  # Minimum quantity
    trading_hours: List[MarketSession]
    risk_multipliers: Dict[str, RiskMultipliers]  # By regime: "trending", "ranging", "volatile"
    expiry_enabled: bool = False
    expiry_days: Optional[List[int]] = None  # Days of month when expiry occurs
    volatility_thresholds: Optional[Dict[str, float]] = None  # "low_high", "medium_high"


class BaseMarket(ABC):
    """
    Abstract base class for markets.
    
    Responsibilities:
    - Define trading hours and sessions
    - Manage expiry logic (if applicable)
    - Inject risk parameters based on market regime
    - Validate orders before execution
    - Track market-specific state
    """

    def __init__(self, config: MarketConfig):
        self.config = config
        self.symbol = config.symbol
        self.market_type = config.market_type
        self.logger = logging.getLogger(f"Market.{self.market_type}")
        self._validate_config()

    def _validate_config(self):
        """Validate that configuration is complete"""
        if not self.config.trading_hours:
            raise ValueError(f"Market {self.market_type} has no trading sessions")
        if not self.config.risk_multipliers:
            raise ValueError(f"Market {self.market_type} has no risk multipliers")
        self.logger.info(f"Market {self.market_type} initialized with {len(self.config.trading_hours)} sessions")

    @abstractmethod
    def is_trading_hours(self, timestamp: datetime) -> bool:
        """Check if current time is within trading hours"""
        pass

    @abstractmethod
    def get_current_session(self, timestamp: datetime) -> Optional[MarketSession]:
        """Get the current trading session"""
        pass

    @abstractmethod
    def is_expiry_day(self, timestamp: datetime) -> bool:
        """Check if today is an expiry day (if applicable)"""
        pass

    @abstractmethod
    def get_risk_multipliers(self, regime: str) -> RiskMultipliers:
        """
        Get risk parameters for the given market regime.
        regime: "trending", "ranging", "volatile"
        """
        pass

    @abstractmethod
    def validate_order(self, symbol: str, quantity: int, price: float) -> Tuple[bool, str]:
        """
        Validate if an order can be placed.
        Returns: (is_valid, reason_if_invalid)
        """
        pass

    @abstractmethod
    def adjust_order_quantity(self, quantity: int) -> int:
        """Adjust quantity to conform to market rules (lot size, etc.)"""
        pass

    @abstractmethod
    def adjust_order_price(self, price: float, direction: str) -> float:
        """
        Adjust price to conform to tick size.
        direction: "buy" or "sell"
        """
        pass

    def round_to_tick_size(self, price: float) -> float:
        """Round price to nearest tick size"""
        return round(price / self.config.tick_size) * self.config.tick_size

    def get_trading_sessions_today(self, timestamp: datetime) -> List[MarketSession]:
        """Get all trading sessions for a given day"""
        return [s for s in self.config.trading_hours if s.allow_trading]

    def get_next_trading_session(self, timestamp: datetime) -> Optional[MarketSession]:
        """Get the next trading session after the given timestamp"""
        current_time = timestamp.time()
        for session in self.config.trading_hours:
            if session.allow_trading and session.start_time > current_time:
                return session
        return None

    def __str__(self) -> str:
        return f"{self.market_type.upper()} Market ({self.symbol})"
