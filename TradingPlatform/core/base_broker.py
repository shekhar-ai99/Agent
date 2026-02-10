"""
Base Broker Abstract Class

All brokers (paper, real) inherit from this.
Initial implementation: Paper broker for simulation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order lifecycle states"""
    PENDING = "pending"
    PLACED = "placed"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class Order:
    """Represents a single trade order"""
    order_id: str
    symbol: str
    market_type: str  # "indian" or "crypto"
    direction: str  # "BUY" or "SELL"
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_price: Optional[float] = None
    filled_quantity: int = 0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)  # Strategy name, entry reason, etc.

    def is_filled(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]

    def is_complete(self) -> bool:
        return self.filled_quantity >= self.quantity


@dataclass
class Position:
    """Represents an open position in an instrument"""
    symbol: str
    market_type: str
    quantity: int
    entry_price: float
    entry_time: datetime
    direction: str  # "LONG" or "SHORT"
    strategy_name: str
    entry_signal: str  # "BUY" or "SELL"
    pnl: float = 0.0
    pnl_percent: float = 0.0
    current_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    def update_pnl(self, current_price: float):
        """Update P&L based on current price"""
        self.current_price = current_price
        if self.direction == "LONG":
            self.pnl = (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.pnl = (self.entry_price - current_price) * self.quantity
        
        self.pnl_percent = (self.pnl / (self.entry_price * self.quantity)) * 100 if self.entry_price > 0 else 0


class BaseBroker(ABC):
    """
    Abstract base class for brokers.
    
    Responsibilities:
    - Order placement and management
    - Position tracking
    - Account balance management
    - Trade execution simulation (for paper broker)
    """

    def __init__(self, broker_name: str, initial_balance: float = 100000.0):
        self.broker_name = broker_name
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.logger = logging.getLogger(f"Broker.{broker_name}")
        
        # State tracking
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.order_counter = 0

    def get_next_order_id(self) -> str:
        """Generate a unique order ID"""
        self.order_counter += 1
        return f"{self.broker_name}_ORD_{self.order_counter:06d}"

    def get_balance(self) -> float:
        """Get current account balance."""
        return self.current_balance

    def get_orders(self) -> List[Order]:
        """Get all orders."""
        return list(self.orders.values())

    @abstractmethod
    def place_order(self, symbol: str, market_type: str, direction: str, quantity: int,
                   order_type: OrderType = OrderType.MARKET, limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None, **metadata) -> Order:
        """
        Place a new order.
        Returns: Order object with status.
        """
        pass

    @abstractmethod
    def fill_order(self, order_id: str, filled_price: float, filled_quantity: int) -> bool:
        """Simulate order fill"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass

    @abstractmethod
    def update_position(self, symbol: str, current_price: float):
        """Update P&L for a position based on current market price"""
        pass

    @abstractmethod
    def close_position(self, symbol: str, exit_price: float, exit_reason: str) -> Optional[Dict]:
        """
        Close a position.
        Returns: Trade summary dict (entry, exit, pnl, etc.)
        """
        pass

    def get_open_positions(self) -> List[Position]:
        """Return all open positions"""
        return list(self.positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get a specific position"""
        return self.positions.get(symbol)

    def get_portfolio_value(self, current_prices: Dict[str, float] = None) -> float:
        """Calculate total portfolio value (cash + open positions)"""
        total = self.current_balance
        for pos in self.positions.values():
            if current_prices and pos.symbol in current_prices:
                pos.update_pnl(current_prices[pos.symbol])
            total += pos.pnl
        return total

    def get_account_equity(self, current_prices: Dict[str, float] = None) -> float:
        """Alias for portfolio value"""
        return self.get_portfolio_value(current_prices)

    def add_to_trade_history(self, trade_summary: Dict):
        """Record a completed trade"""
        self.trade_history.append(trade_summary)
        self.logger.info(f"Trade recorded: {trade_summary}")

    def get_trade_history(self) -> List[Dict]:
        """Return all completed trades"""
        return self.trade_history

    def __str__(self) -> str:
        return f"{self.broker_name} Broker (Balance: ${self.current_balance:.2f})"
