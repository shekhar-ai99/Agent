"""
Crypto Market Implementation

Crypto-specific trading rules: 24/7 trading, no expiry, higher volatility.
"""

from datetime import datetime, time
from typing import Optional, Tuple
import logging

from core.base_market import BaseMarket, MarketConfig, MarketSession, RiskMultipliers

logger = logging.getLogger(__name__)


class CryptoMarket(BaseMarket):
    """
    Cryptocurrency market (Bitcoin, Ethereum, etc.).
    
    Features:
    - 24/7 trading (no market hours)
    - No expiry logic
    - Higher volatility thresholds
    - Smaller lot sizes (can trade 0.001 BTC, etc.)
    """

    @staticmethod
    def create_default_config(symbol: str = "BTCUSDT") -> MarketConfig:
        """Create default crypto configuration"""
        return MarketConfig(
            symbol=symbol,
            market_type="crypto",
            tick_size=0.01,  # Small tick size for crypto
            lot_size=1,  # Can adjust per symbol (e.g., 0.001 for BTC)
            trading_hours=[
                MarketSession("24/7", time(0, 0), time(23, 59), allow_trading=True),
            ],
            risk_multipliers={
                "trending": RiskMultipliers(
                    sl_atr_multiple=1.5,  # Higher for crypto volatility
                    tp_atr_multiple=2.0,
                    tsl_atr_multiple=1.0,
                    max_position_size=3.0,  # More conservative for crypto
                    max_concurrent_trades=2,
                ),
                "ranging": RiskMultipliers(
                    sl_atr_multiple=1.0,
                    tp_atr_multiple=1.2,
                    tsl_atr_multiple=0.8,
                    max_position_size=2.0,
                    max_concurrent_trades=2,
                ),
                "volatile": RiskMultipliers(
                    sl_atr_multiple=2.0,
                    tp_atr_multiple=1.5,
                    tsl_atr_multiple=1.2,
                    max_position_size=1.0,  # Very conservative in volatile conditions
                    max_concurrent_trades=1,
                ),
            },
            expiry_enabled=False,
            expiry_days=None,
            volatility_thresholds={
                "low_high": 20.0,  # Higher thresholds for crypto
                "medium_high": 35.0,
            },
        )

    def __init__(self, symbol: str = "BTCUSDT"):
        config = self.create_default_config(symbol)
        super().__init__(config)

    def is_trading_hours(self, timestamp: datetime) -> bool:
        """Crypto trades 24/7"""
        return True

    def get_current_session(self, timestamp: datetime) -> Optional[MarketSession]:
        """Crypto has one continuous session"""
        return self.config.trading_hours[0]

    def is_expiry_day(self, timestamp: datetime) -> bool:
        """Crypto has no expiry"""
        return False

    def get_risk_multipliers(self, regime: str) -> RiskMultipliers:
        """Get risk parameters for the given market regime"""
        return self.config.risk_multipliers.get(
            regime,
            self.config.risk_multipliers["ranging"]  # Default fallback
        )

    def validate_order(self, symbol: str, quantity: float, price: float) -> Tuple[bool, str]:
        """Validate if an order can be placed"""
        if quantity <= 0:
            return False, "Quantity must be positive"
        
        if price <= 0:
            return False, "Price must be positive"
        
        # Crypto allows fractional quantities
        return True, "Valid"

    def adjust_order_quantity(self, quantity: float) -> float:
        """Crypto usually doesn't have strict lot size restrictions"""
        return quantity

    def adjust_order_price(self, price: float, direction: str) -> float:
        """Round price to nearest tick size"""
        return self.round_to_tick_size(price)

    def get_session_name(self, timestamp: datetime) -> str:
        """Crypto doesn't have sessions"""
        return "24/7"

    def is_high_volatility_period(self, timestamp: datetime) -> bool:
        """
        Crypto has certain high volatility periods (usually during major economic news).
        Can be extended with more sophisticated detection.
        """
        hour = timestamp.hour
        # Typically high volatility during NYC market open (1-4 PM UTC)
        return hour in [13, 14, 15, 16]
