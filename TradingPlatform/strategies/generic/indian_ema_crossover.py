"""
Indian EMA Crossover Strategy (legacy parity)

Logic: buy when EMA short crosses above EMA long; sell on cross below.

MIGRATED FROM: IndianMarket/strategy_tester_app/app/strategies.py
ORIGINAL: EMACrossoverStrategy
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


class IndianEMACrossover(BaseStrategy):
    """EMA crossover using precomputed EMA columns (legacy parity)."""

    def __init__(
        self,
        short_window: int = 9,
        long_window: int = 21,
        name: str = "IM_EMACrossover",
        version: str = "2.0"
    ):
        super().__init__(name=name, version=version)
        self.short_window = short_window
        self.long_window = long_window
        self.required_lookback = 2

    def required_indicators(self) -> List[str]:
        return [
            "close",
            "atr_14",
            f"ema_{self.short_window}",
            f"ema_{self.long_window}"
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

        history = context.historical_data
        ema_short_col = f"ema_{self.short_window}"
        ema_long_col = f"ema_{self.long_window}"

        if ema_short_col not in history.columns or ema_long_col not in history.columns:
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=f"Missing EMA columns {ema_short_col} or {ema_long_col}"
            )

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
                reasoning="Not enough history for crossover detection"
            )

        current_close = history["close"].iloc[current_idx]
        current_atr = history["atr_14"].iloc[current_idx]
        current_short = history[ema_short_col].iloc[current_idx]
        current_long = history[ema_long_col].iloc[current_idx]
        prev_short = history[ema_short_col].iloc[current_idx - 1]
        prev_long = history[ema_long_col].iloc[current_idx - 1]

        if (
            pd.isna(current_close)
            or pd.isna(current_atr)
            or pd.isna(current_short)
            or pd.isna(current_long)
            or pd.isna(prev_short)
            or pd.isna(prev_long)
        ):
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning="Invalid EMA/close/ATR values"
            )

        if current_short > current_long and prev_short <= prev_long:
            return Signal(
                direction="BUY",
                confidence=0.6,
                strategy_name=self.name,
                reasoning=(
                    f"EMA crossover BUY: {ema_short_col} crossed above {ema_long_col}"
                ),
                additional_context={
                    "ema_short": current_short,
                    "ema_long": current_long,
                    "sl": current_close * 0.995,
                    "tp": current_close * 1.01,
                    "tsl": current_close - current_atr
                }
            )

        if current_short < current_long and prev_short >= prev_long:
            return Signal(
                direction="SELL",
                confidence=0.6,
                strategy_name=self.name,
                reasoning=(
                    f"EMA crossover SELL: {ema_short_col} crossed below {ema_long_col}"
                ),
                additional_context={
                    "ema_short": current_short,
                    "ema_long": current_long,
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
                f"No crossover: {ema_short_col}={current_short:.2f}, "
                f"{ema_long_col}={current_long:.2f}"
            )
        )
