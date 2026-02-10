"""
Indian Market Implementation

Includes NSE (Equity & Index) trading rules, sessions, expiry logic, and risk parameters.
"""

from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from core.base_market import BaseMarket, MarketConfig, MarketSession, RiskMultipliers

logger = logging.getLogger(__name__)


class IndianMarket(BaseMarket):
    """
    Indian equity and index market (NSE).
    
    Features:
    - Pre-market: 9:00 - 9:15 AM
    - Morning session: 9:15 AM - 11:59 AM
    - Midday: 12:00 PM - 1:59 PM
    - Afternoon: 2:00 PM - 3:30 PM
    - Index expiry: Last Thursday of every month
    """

    @staticmethod
    def create_default_config(symbol: str = "NIFTY50") -> MarketConfig:
        """Create default NSE configuration"""
        return MarketConfig(
            symbol=symbol,
            market_type="indian",
            tick_size=0.05,  # NSE tick size
            lot_size=1,
            trading_hours=[
                MarketSession("Pre-Market", time(9, 0), time(9, 15), allow_trading=False),
                MarketSession("Morning", time(9, 15), time(11, 59), allow_trading=True),
                MarketSession("Midday", time(12, 0), time(13, 59), allow_trading=True),
                MarketSession("Afternoon", time(14, 0), time(15, 30), allow_trading=True),
                MarketSession("Post-Market", time(15, 30), time(16, 0), allow_trading=False),
            ],
            risk_multipliers={
                "trending": RiskMultipliers(
                    sl_atr_multiple=1.2,
                    tp_atr_multiple=1.5,
                    tsl_atr_multiple=0.8,
                    max_position_size=5.0,
                    max_concurrent_trades=3,
                ),
                "ranging": RiskMultipliers(
                    sl_atr_multiple=0.8,
                    tp_atr_multiple=0.8,
                    tsl_atr_multiple=0.6,
                    max_position_size=3.0,
                    max_concurrent_trades=2,
                ),
                "volatile": RiskMultipliers(
                    sl_atr_multiple=1.5,
                    tp_atr_multiple=1.0,
                    tsl_atr_multiple=1.0,
                    max_position_size=2.0,
                    max_concurrent_trades=1,
                ),
            },
            expiry_enabled=True,
            expiry_days=[],  # Will be calculated dynamically
            volatility_thresholds={
                "low_high": 15.0,
                "medium_high": 25.0,
            },
        )

    def __init__(self, symbol: str = "NIFTY50"):
        config = self.create_default_config(symbol)
        super().__init__(config)
        self._update_expiry_days()

    def _update_expiry_days(self):
        """Calculate all expiry days for current and next few months"""
        expiry_days = []
        today = datetime.now()
        
        # Calculate for next 6 months
        for month_offset in range(6):
            # Get last day of month
            first_day = today.replace(
                month=((today.month - 1 + month_offset) % 12) + 1,
                year=today.year + ((today.month - 1 + month_offset) // 12),
                day=1
            )
            last_day = (first_day + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            
            # Find last Thursday
            while last_day.weekday() != 3:  # 3 = Thursday
                last_day -= timedelta(days=1)
            
            expiry_days.append(last_day.day)
        
        self.config.expiry_days = sorted(set(expiry_days))
        self.logger.info(f"Expiry days: {self.config.expiry_days}")

    def is_trading_hours(self, timestamp: datetime) -> bool:
        """Check if current time is within trading hours"""
        current_time = timestamp.time()
        
        # Skip weekends
        if timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if within any trading session
        for session in self.config.trading_hours:
            if session.allow_trading and session.start_time <= current_time <= session.end_time:
                return True
        
        return False

    def get_current_session(self, timestamp: datetime) -> Optional[MarketSession]:
        """Get the current trading session"""
        current_time = timestamp.time()
        
        for session in self.config.trading_hours:
            if session.start_time <= current_time <= session.end_time:
                return session
        
        return None

    def is_expiry_day(self, timestamp: datetime) -> bool:
        """Check if today is an expiry day"""
        if not self.config.expiry_enabled:
            return False
        
        return timestamp.day in self.config.expiry_days

    def get_risk_multipliers(self, regime: str) -> RiskMultipliers:
        """Get risk parameters for the given market regime"""
        return self.config.risk_multipliers.get(
            regime,
            self.config.risk_multipliers["ranging"]  # Default fallback
        )

    def validate_order(self, symbol: str, quantity: int, price: float) -> Tuple[bool, str]:
        """Validate if an order can be placed"""
        if quantity <= 0:
            return False, "Quantity must be positive"
        
        if quantity % self.config.lot_size != 0:
            return False, f"Quantity must be multiple of lot size ({self.config.lot_size})"
        
        if price <= 0:
            return False, "Price must be positive"
        
        return True, "Valid"

    def adjust_order_quantity(self, quantity: int) -> int:
        """Adjust quantity to nearest valid lot size"""
        return (quantity // self.config.lot_size) * self.config.lot_size

    def adjust_order_price(self, price: float, direction: str) -> float:
        """Round price to nearest tick size"""
        return self.round_to_tick_size(price)

    def get_session_name(self, timestamp: datetime) -> str:
        """Get current session name for logging/reporting"""
        session = self.get_current_session(timestamp)
        return session.name if session else "Outside Trading Hours"

    def is_morning_session(self, timestamp: datetime) -> bool:
        """Check if in morning session"""
        session = self.get_current_session(timestamp)
        return session and session.name == "Morning"

    def is_midday_session(self, timestamp: datetime) -> bool:
        """Check if in midday session"""
        session = self.get_current_session(timestamp)
        return session and session.name == "Midday"

    def is_afternoon_session(self, timestamp: datetime) -> bool:
        """Check if in afternoon session"""
        session = self.get_current_session(timestamp)
        return session and session.name == "Afternoon"
