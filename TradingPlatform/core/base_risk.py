"""
Risk Management Layer

Controls position sizing, drawdowns, and order validation before execution.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Real-time risk tracking"""
    account_balance: float
    portfolio_value: float
    open_position_count: int
    total_open_pnl: float
    daily_realized_pnl: float
    max_daily_loss: float
    max_drawdown_percent: float
    risk_per_trade_percent: float


class RiskManager:
    """
    Manages position sizing and risk controls.
    """

    def __init__(
        self,
        max_position_size_percent: float = 5.0,
        max_concurrent_trades: int = 5,
        max_daily_loss_percent: float = 2.0,
        risk_per_trade_percent: float = 1.0,
    ):
        """
        Args:
            max_position_size_percent: Max % of account per trade
            max_concurrent_trades: Max number of open trades
            max_daily_loss_percent: Stop trading if daily loss exceeds this
            risk_per_trade_percent: Max loss per trade as % of account
        """
        self.max_position_size_percent = max_position_size_percent
        self.max_concurrent_trades = max_concurrent_trades
        self.max_daily_loss_percent = max_daily_loss_percent
        self.risk_per_trade_percent = risk_per_trade_percent
        self.daily_pnl = 0.0
        self.logger = logging.getLogger("RiskManager")

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> int:
        """
        Calculate position size based on risk per trade.
        
        Formula:
        position_size = (account_balance * risk_per_trade_percent) / (entry_price - stop_loss_price)
        
        Returns: Quantity to trade
        """
        risk_amount = account_balance * (self.risk_per_trade_percent / 100.0)
        price_range = abs(entry_price - stop_loss_price)
        
        if price_range == 0:
            self.logger.warning("Stop loss equals entry price. Cannot calculate position size.")
            return 0
        
        position_size = int(risk_amount / price_range)
        
        # Cap by max position size
        max_size = int(account_balance * (self.max_position_size_percent / 100.0) / entry_price)
        position_size = min(position_size, max_size)
        
        self.logger.info(
            f"Position size: {position_size} "
            f"(Risk: ${risk_amount:.2f}, PriceRange: ${price_range:.2f})"
        )
        return max(position_size, 1)  # At least 1 unit

    def can_open_trade(
        self,
        current_open_trades: int,
        account_balance: float,
        daily_loss: float,
    ) -> Tuple[bool, str]:
        """
        Check if a new trade is allowed.
        Returns: (is_allowed, reason_if_denied)
        """
        # Check concurrent trades limit
        if current_open_trades >= self.max_concurrent_trades:
            return False, f"Concurrent trades limit ({self.max_concurrent_trades}) reached"
        
        # Check daily loss limit
        if daily_loss / account_balance < -(self.max_daily_loss_percent / 100.0):
            return False, f"Daily loss limit ({self.max_daily_loss_percent}%) exceeded"
        
        return True, "OK"

    def validate_order(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        account_balance: float,
    ) -> Tuple[bool, str]:
        """
        Validate order parameters.
        Returns: (is_valid, reason_if_invalid)
        """
        if quantity <= 0:
            return False, "Quantity must be positive"
        
        if entry_price <= 0:
            return False, "Entry price must be positive"
        
        if stop_loss <= 0:
            return False, "Stop loss must be positive"
        
        # Ensure stop loss is actually protective
        if abs(entry_price - stop_loss) < entry_price * 0.001:  # Less than 0.1%
            return False, "Stop loss too close to entry price"
        
        # Check position size against account
        position_value = quantity * entry_price
        if position_value > account_balance * (self.max_position_size_percent / 100.0):
            return False, f"Position size exceeds max ({self.max_position_size_percent}% of account)"
        
        return True, "OK"

    def get_risk_metrics(
        self,
        account_balance: float,
        portfolio_value: float,
        open_positions: int,
        total_open_pnl: float,
        daily_realized_pnl: float,
        peak_portfolio_value: float,
    ) -> RiskMetrics:
        """Generate current risk metrics snapshot"""
        max_drawdown = max(0, (peak_portfolio_value - portfolio_value) / peak_portfolio_value * 100)
        
        return RiskMetrics(
            account_balance=account_balance,
            portfolio_value=portfolio_value,
            open_position_count=open_positions,
            total_open_pnl=total_open_pnl,
            daily_realized_pnl=daily_realized_pnl,
            max_daily_loss=self.max_daily_loss_percent,
            max_drawdown_percent=max_drawdown,
            risk_per_trade_percent=self.risk_per_trade_percent,
        )

    def update_daily_pnl(self, trade_pnl: float):
        """Update daily P&L tracking"""
        self.daily_pnl += trade_pnl
        self.logger.info(f"Daily P&L updated: ${self.daily_pnl:.2f}")

    def reset_daily_pnl(self):
        """Reset daily P&L (call at market open)"""
        self.daily_pnl = 0.0
        self.logger.info("Daily P&L reset")
