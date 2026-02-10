"""
SuperTrend + ADX Strategy

Logic: buy when ADX is above threshold and close is above SuperTrend; sell when close is below SuperTrend.

MIGRATED FROM: IndianMarket/strategy_tester_app/app/strategies.py
ORIGINAL: SuperTrendADXStrategy
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


class SuperTrendADX(BaseStrategy):
    """SuperTrend with ADX threshold (legacy parity)."""

    def __init__(
        self,
        supert_period: int = 10,
        supert_multiplier: float = 2.0,
        adx_period: int = 14,
        adx_threshold: float = 20,
        name: str = "IM_SuperTrendADX",
        version: str = "2.0"
    ):
        super().__init__(name=name, version=version)
        self.supert_period = supert_period
        self.supert_multiplier = supert_multiplier
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.required_lookback = 1

    def required_indicators(self) -> List[str]:
        return [
            "close",
            "atr_14",
            f"adx_{self.adx_period}",
            f"supertrend_{self.supert_period}_{self.supert_multiplier}"
        ]

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

        st_col = f"supertrend_{self.supert_period}_{self.supert_multiplier}"
        adx_col = f"adx_{self.adx_period}"

        current_close = context.current_bar.get("close")
        current_atr = context.current_bar.get("atr_14")
        current_adx = context.current_bar.get(adx_col)
        current_st = context.current_bar.get(st_col)

        if (
            pd.isna(current_close)
            or pd.isna(current_atr)
            or pd.isna(current_adx)
            or pd.isna(current_st)
        ):
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning="Invalid indicator values"
            )

        if current_adx > self.adx_threshold and current_close > current_st:
            return Signal(
                direction="BUY",
                confidence=0.6,
                strategy_name=self.name,
                reasoning="SuperTrendADX BUY: ADX strong and close above SuperTrend",
                additional_context={
                    "adx": current_adx,
                    "supertrend": current_st,
                    "sl": current_close * 0.995,
                    "tp": current_close * 1.01,
                    "tsl": current_close - current_atr
                }
            )

        if current_adx > self.adx_threshold and current_close < current_st:
            return Signal(
                direction="SELL",
                confidence=0.6,
                strategy_name=self.name,
                reasoning="SuperTrendADX SELL: ADX strong and close below SuperTrend",
                additional_context={
                    "adx": current_adx,
                    "supertrend": current_st,
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
                f"No signal: adx={current_adx:.2f}, close={current_close:.2f}, "
                f"supertrend={current_st:.2f}"
            )
        )


def create_supertrend_adx_strategy(
    supert_period: int = 10,
    supert_multiplier: float = 2.0,
    adx_period: int = 14,
    adx_threshold: float = 20
) -> SuperTrendADX:
    return SuperTrendADX(
        supert_period=supert_period,
        supert_multiplier=supert_multiplier,
        adx_period=adx_period,
        adx_threshold=adx_threshold
    )
