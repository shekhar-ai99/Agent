"""
ATR Breakout Strategy

Buy when close exceeds prior close + ATR*mult; sell when close is below prior close - ATR*mult.

MIGRATED FROM: IndianMarket/strategy_tester_app/app/strategies.py (ATRBreakoutStrategy)
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


class ATRBreakout(BaseStrategy):
    """ATR-based breakout using prior close +/- ATR multiplier bands."""

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        name: str = "ATR_Breakout",
        version: str = "2.0"
    ):
        super().__init__(name=name, version=version)
        self.atr_multiplier = atr_multiplier
        self.required_lookback = 2

        self.logger.info(
            f"Initialized {self.name} with atr_multiplier={atr_multiplier}"
        )

    def required_indicators(self) -> List[str]:
        return ["close", "atr_14"]

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

        try:
            history = context.historical_data

            try:
                current_idx = history.index.get_loc(context.current_bar.name)
            except KeyError:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Current bar not in history"
                )

            if current_idx < 1:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Not enough history for ATR breakout"
                )

            current_close = history["close"].iloc[current_idx]
            prev_close = history["close"].iloc[current_idx - 1]
            current_atr = history["atr_14"].iloc[current_idx]

            if pd.isna(current_close) or pd.isna(prev_close) or pd.isna(current_atr):
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Invalid close/atr values"
                )

            upper_band = prev_close + (self.atr_multiplier * current_atr)
            lower_band = prev_close - (self.atr_multiplier * current_atr)

            if current_close > upper_band:
                return Signal(
                    direction="BUY",
                    confidence=0.6,
                    strategy_name=self.name,
                    reasoning=(
                        f"ATR breakout BUY: close={current_close:.2f} "
                        f"> upper_band={upper_band:.2f}"
                    ),
                    additional_context={
                        "upper_band": upper_band,
                        "lower_band": lower_band,
                        "atr_14": current_atr,
                        "sl": current_close * 0.995,
                        "tp": current_close * 1.01,
                        "tsl": current_close - current_atr
                    }
                )

            if current_close < lower_band:
                return Signal(
                    direction="SELL",
                    confidence=0.6,
                    strategy_name=self.name,
                    reasoning=(
                        f"ATR breakout SELL: close={current_close:.2f} "
                        f"< lower_band={lower_band:.2f}"
                    ),
                    additional_context={
                        "upper_band": upper_band,
                        "lower_band": lower_band,
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
                    f"No breakout: close={current_close:.2f}, "
                    f"band=[{lower_band:.2f}, {upper_band:.2f}]"
                )
            )

        except Exception as e:
            self.logger.error(f"ATR breakout error: {e}", exc_info=True)
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=f"Exception: {str(e)}"
            )


def create_atr_breakout_strategy(
    atr_multiplier: float = 2.0
) -> ATRBreakout:
    return ATRBreakout(atr_multiplier=atr_multiplier)
