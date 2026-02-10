"""
Paper Broker - Simulates trading for backtesting and paper trading.

NO REAL ORDERS ARE PLACED.
This is for simulation only.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import uuid

from core.base_broker import BaseBroker, Order, OrderStatus, OrderType, Position

logger = logging.getLogger(__name__)


class PaperBroker(BaseBroker):
    """
    Paper trading broker for backtesting and simulation.
    
    Features:
    - Simulates order fills based on market price
    - Tracks positions and P&L
    - No slippage (configurable)
    - Market-agnostic (supports indian, crypto, etc.)
    """

    def __init__(
        self,
        initial_balance: float = 100000.0,
        slippage_percent: float = 0.0,
        execution_mode: str = "instant",  # "instant" or "next_bar"
    ):
        """
        Args:
            initial_balance: Starting account balance
            slippage_percent: Simulate slippage (% of entry price)
            execution_mode: When to fill orders (instant or next bar)
        """
        super().__init__("PaperBroker", initial_balance)
        self.slippage_percent = slippage_percent
        self.execution_mode = execution_mode
        self.logger.info(
            f"PaperBroker initialized - Balance: ${initial_balance:.2f}, "
            f"Slippage: {slippage_percent}%, Mode: {execution_mode}"
        )

    def place_order(
        self,
        symbol: str,
        market_type: str,
        direction: str,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **metadata,
    ) -> Order:
        """
        Place an order.
        
        For paper trading, MARKET orders are filled immediately.
        LIMIT and STOP orders are queued until conditions are met.
        """
        order_id = self.get_next_order_id()
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            market_type=market_type,
            direction=direction,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            metadata=metadata,
        )
        
        self.orders[order_id] = order
        order.status = OrderStatus.PLACED
        
        self.logger.info(
            f"Order placed: {order_id} | {direction} {quantity} {symbol} @ "
            f"{order_type.value} (limit: {limit_price})"
        )
        
        return order

    def fill_order(
        self,
        order_id: str,
        filled_price: float,
        filled_quantity: int,
    ) -> bool:
        """
        Simulate order fill.
        
        For paper trading, this is called by the execution engine
        when a market order should be filled or a limit/stop condition is met.
        """
        if order_id not in self.orders:
            self.logger.error(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        # Apply slippage
        if self.slippage_percent > 0:
            if order.direction == "BUY":
                filled_price *= (1 + self.slippage_percent / 100.0)
            else:  # SELL
                filled_price *= (1 - self.slippage_percent / 100.0)
        
        # Update order
        order.filled_price = filled_price
        order.filled_quantity = filled_quantity
        order.filled_at = datetime.utcnow()
        order.status = OrderStatus.FILLED if filled_quantity >= order.quantity else OrderStatus.PARTIALLY_FILLED
        
        # Update account balance (deduct position cost)
        position_cost = filled_price * filled_quantity
        self.current_balance -= position_cost
        
        # Open or update position
        if order.symbol not in self.positions:
            position = Position(
                symbol=order.symbol,
                market_type=order.market_type,
                quantity=filled_quantity,
                entry_price=filled_price,
                entry_time=order.filled_at,
                direction="LONG" if order.direction == "BUY" else "SHORT",
                strategy_name=order.metadata.get("strategy", "unknown"),
                entry_signal=order.direction,
                stop_loss=order.metadata.get("stop_loss"),
                take_profit=order.metadata.get("take_profit"),
                metadata=order.metadata,
            )
            self.positions[order.symbol] = position
        else:
            # Add to existing position
            pos = self.positions[order.symbol]
            if pos.direction == ("LONG" if order.direction == "BUY" else "SHORT"):
                # Same direction, increase position
                avg_price = (pos.entry_price * pos.quantity + filled_price * filled_quantity) / (
                    pos.quantity + filled_quantity
                )
                pos.entry_price = avg_price
                pos.quantity += filled_quantity
            else:
                # Opposite direction, decrease position
                pos.quantity -= filled_quantity
                if pos.quantity == 0:
                    del self.positions[order.symbol]
        
        self.logger.info(
            f"Order filled: {order_id} | {order.direction} {filled_quantity} "
            f"{order.symbol} @ ${filled_price:.2f} | "
            f"Strategy: {order.metadata.get('strategy', 'unknown')}"
        )
        
        return True

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if order_id not in self.orders:
            self.logger.error(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        if order.status == OrderStatus.PENDING or order.status == OrderStatus.PLACED:
            order.status = OrderStatus.CANCELLED
            self.logger.info(f"Order cancelled: {order_id}")
            return True
        
        self.logger.warning(f"Cannot cancel order {order_id} with status {order.status.value}")
        return False

    def update_position(self, symbol: str, current_price: float):
        """Update P&L for a position"""
        if symbol in self.positions:
            self.positions[symbol].update_pnl(current_price)

    def close_position(self, symbol: str, exit_price: float, exit_reason: str) -> Optional[Dict]:
        """
        Close a position and record the trade.
        
        Returns: Trade summary dict
        """
        if symbol not in self.positions:
            self.logger.warning(f"No position found for {symbol}")
            return None
        
        pos = self.positions[symbol]
        
        # Calculate P&L
        if pos.direction == "LONG":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:  # SHORT
            pnl = (pos.entry_price - exit_price) * pos.quantity
        
        # Calculate other metrics
        duration = datetime.utcnow() - pos.entry_time
        pnl_percent = (pnl / (pos.entry_price * pos.quantity)) * 100
        
        trade_summary = {
            "symbol": symbol,
            "market_type": pos.market_type,
            "direction": pos.entry_signal,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "quantity": pos.quantity,
            "entry_time": pos.entry_time,
            "exit_time": datetime.utcnow(),
            "duration": duration.total_seconds(),
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "strategy": pos.strategy_name,
            "exit_reason": exit_reason,
            "stop_loss_hit": pos.stop_loss and exit_price <= pos.stop_loss if pos.direction == "LONG" else exit_price >= pos.stop_loss,
            "take_profit_hit": pos.take_profit and exit_price >= pos.take_profit if pos.direction == "LONG" else exit_price <= pos.take_profit,
        }
        
        # Update balance
        self.current_balance += (exit_price * pos.quantity)
        
        # Remove position
        del self.positions[symbol]
        
        # Record in history
        self.add_to_trade_history(trade_summary)
        
        self.logger.info(
            f"❌ Position closed: {symbol} | Strategy: {pos.strategy_name} | "
            f"PnL: ${pnl:.2f} ({pnl_percent:.2f}%) | "
            f"Entry: ${pos.entry_price:.2f} -> Exit: ${exit_price:.2f} | "
            f"Duration: {duration.total_seconds():.0f}s | "
            f"Reason: {exit_reason}"
        )
        
        return trade_summary

    def simulate_market_fill(
        self,
        symbol: str,
        market_type: str,
        direction: str,
        quantity: int,
        market_price: float,
        **metadata,
    ) -> Optional[Dict]:
        """
        Convenience method: Place a market order and fill it immediately.
        Used in backtesting loops.
        
        Returns: Trade dict or None
        """
        order = self.place_order(
            symbol=symbol,
            market_type=market_type,
            direction=direction,
            quantity=quantity,
            order_type=OrderType.MARKET,
            **metadata,
        )
        
        self.fill_order(order.order_id, market_price, quantity)
        return order

    def get_portfolio_value(self, current_prices: Dict[str, float] = None) -> float:
        """Calculate total portfolio value"""
        total = self.current_balance
        
        if current_prices:
            for symbol, price in current_prices.items():
                if symbol in self.positions:
                    self.update_position(symbol, price)
        
        # Add open position values
        for pos in self.positions.values():
            if pos.current_price:
                if pos.direction == "LONG":
                    total += (pos.current_price - pos.entry_price) * pos.quantity
                else:  # SHORT
                    total += (pos.entry_price - pos.current_price) * pos.quantity
        
        return total

    def __str__(self) -> str:
        return (
            f"PaperBroker | Balance: ${self.current_balance:.2f} | "
            f"Positions: {len(self.positions)} | "
            f"Trades: {len(self.trade_history)}"
        )
