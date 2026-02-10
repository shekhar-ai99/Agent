"""
Momentum Breakout Strategy

Logic: buy on 20-bar rolling high breakout; sell on 20-bar rolling low breakdown.

MIGRATED FROM: IndianMarket/strategy_tester_app/app/strategies.py
ORIGINAL: MomentumBreakoutStrategy
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


class MomentumBreakout(BaseStrategy):
    """Momentum breakout using rolling highs/lows (legacy parity)."""

    def __init__(
        self,
        breakout_window: int = 20,
        name: str = "Momentum_Breakout",
        version: str = "2.0"
    ):
        super().__init__(name=name, version=version)
        self.breakout_window = breakout_window
        self.required_lookback = breakout_window + 1

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
                    reasoning="Not enough history for breakout detection"
                )

            close_series = history["close"]
            current_close = close_series.iloc[current_idx]
            current_atr = history["atr_14"].iloc[current_idx]

            if pd.isna(current_close) or pd.isna(current_atr):
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Invalid close/ATR values"
                )

            rolling_max = close_series.rolling(self.breakout_window).max().shift(1)
            rolling_min = close_series.rolling(self.breakout_window).min().shift(1)

            if current_close > rolling_max.iloc[current_idx]:
                return Signal(
                    direction="BUY",
                    confidence=0.6,
                    strategy_name=self.name,
                    reasoning=(
                        f"Momentum breakout BUY: close={current_close:.2f} "
                        f"> rolling_max={rolling_max.iloc[current_idx]:.2f}"
                    ),
                    additional_context={
                        "rolling_max": rolling_max.iloc[current_idx],
                        "rolling_min": rolling_min.iloc[current_idx],
                        "sl": current_close * 0.995,
                        "tp": current_close * 1.01,
                        "tsl": current_close - current_atr
                    }
                )

            if current_close < rolling_min.iloc[current_idx]:
                return Signal(
                    direction="SELL",
                    confidence=0.6,
                    strategy_name=self.name,
                    reasoning=(
                        f"Momentum breakout SELL: close={current_close:.2f} "
                        f"< rolling_min={rolling_min.iloc[current_idx]:.2f}"
                    ),
                    additional_context={
                        "rolling_max": rolling_max.iloc[current_idx],
                        "rolling_min": rolling_min.iloc[current_idx],
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
                    f"rolling_max={rolling_max.iloc[current_idx]:.2f}, "
                    f"rolling_min={rolling_min.iloc[current_idx]:.2f}"
                )
            )

        except Exception as e:
            self.logger.error(f"Momentum breakout error: {e}", exc_info=True)
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=f"Exception: {str(e)}"
            )


def create_momentum_breakout_strategy(
    breakout_window: int = 20
) -> MomentumBreakout:
    return MomentumBreakout(breakout_window=breakout_window)
