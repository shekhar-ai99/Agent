"""
Bollinger Mean Reversion Strategy (Close vs Bands)

Legacy logic: buy when close is below lower band; sell when close is above upper band.

MIGRATED FROM: IndianMarket/strategy_tester_app/app/strategies.py
ORIGINAL: BollingerMeanReversionStrategy
"""

import pandas as pd
from typing import List
import logging

from TradingPlatform.core.base_strategy import (
    BaseStrategy,
    Signal,
    StrategyContext,
    MarketRegime,
    VolatilityBucket
)
from TradingPlatform.utils.strategy_utils import ensure_min_history

logger = logging.getLogger(__name__)


class BollingerMeanReversion(BaseStrategy):
    """Close-based Bollinger mean reversion (legacy parity)."""

    def __init__(
        self,
        name: str = "IM_BollingerMeanReversion",
        version: str = "2.0"
    ):
        super().__init__(name=name, version=version)
        self.required_lookback = 1

    def required_indicators(self) -> List[str]:
        return ["close", "bb_lower", "bb_upper", "atr_14"]

    def supports_market(self, market_type: str) -> bool:
        return True

    def supports_regime(self, regime: MarketRegime) -> bool:
        return True

    def supports_volatility(self, volatility: VolatilityBucket) -> bool:
        return True

    def generate_signal(self, context: StrategyContext) -> Signal:
        if not ensure_min_history(context.historical_data, self.required_lookback):
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=f"Insufficient data: need {self.required_lookback} bars"
            )

        current_close = context.current_bar.get("close")
        current_lower = context.current_bar.get("bb_lower")
        current_upper = context.current_bar.get("bb_upper")
        current_atr = context.current_bar.get("atr_14")

        if (
            pd.isna(current_close)
            or pd.isna(current_lower)
            or pd.isna(current_upper)
            or pd.isna(current_atr)
        ):
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning="Invalid indicator values"
            )

        if current_close < current_lower:
            return Signal(
                direction="BUY",
                confidence=0.6,
                strategy_name=self.name,
                reasoning=(
                    f"Bollinger mean reversion BUY: close={current_close:.2f} "
                    f"< lower_band={current_lower:.2f}"
                ),
                additional_context={
                    "bb_lower": current_lower,
                    "bb_upper": current_upper,
                    "atr_14": current_atr,
                    "sl": current_close * 0.995,
                    "tp": current_close * 1.01,
                    "tsl": current_close - current_atr
                }
            )

        if current_close > current_upper:
            return Signal(
                direction="SELL",
                confidence=0.6,
                strategy_name=self.name,
                reasoning=(
                    f"Bollinger mean reversion SELL: close={current_close:.2f} "
                    f"> upper_band={current_upper:.2f}"
                ),
                additional_context={
                    "bb_lower": current_lower,
                    "bb_upper": current_upper,
                    "atr_14": current_atr,
                    "sl": current_close * 1.005,
                    "tp": current_close * 0.99,
                    "tsl": current_close - current_atr
                }
            )

        return Signal(
            direction="HOLD",
            confidence=0.0,
            strategy_name=self.name,
            reasoning=(
                f"No band breach: close={current_close:.2f}, "
                f"bands=[{current_lower:.2f}, {current_upper:.2f}]"
            )
        )
