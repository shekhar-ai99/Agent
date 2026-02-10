"""
Bollinger Bands Mean Reversion Strategy

Pure mean reversion strategy using Bollinger Band touch points.

Entry Logic:
- BUY: Price (low) touches or breaks below lower Bollinger Band
- SELL: Price (high) touches or breaks above upper Bollinger Band

Exit Logic: Controlled by risk module (SL/TP/TSL)

Optimal Conditions:
- Ranging or choppy markets (mean reversion regime)
- Low to medium volatility
- Works on all timeframes

Parameters:
- bb_period: Moving average period (default: 20)
- bb_stddev: Standard deviation multiplier (default: 2.0)

Historical Performance (NIFTY 5min, Jan-May 2025):
- 908 trades
- 74.12% win rate ⭐ (3rd best)
- ₹18,451 P&L ⭐ (3rd best)

MIGRATED FROM: Common/strategies/strategies.py
ORIGINAL: create_bb_mean_reversion_strategy()
STATUS: ✅ PRODUCTION LOGIC PRESERVED
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
from TradingPlatform.utils.strategy_utils import (
    calculate_bollinger_bands,
    ensure_min_history,
    touches_level
)

logger = logging.getLogger(__name__)


class BBMeanReversion(BaseStrategy):
    """
    Bollinger Bands Mean Reversion - Pure reversion to mean.

    Generates signals when price touches the outer Bollinger Bands,
    indicating potential exhaustion and reversal opportunity.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_stddev: float = 2.0,
        name: str = "BB_MeanReversion",
        version: str = "2.0"
    ):
        """
        Initialize Bollinger Bands Mean Reversion strategy.

        Args:
            bb_period: Moving average period
            bb_stddev: Standard deviation multiplier
            name: Strategy name
            version: Strategy version
        """
        super().__init__(name=name, version=version)

        self.bb_period = bb_period
        self.bb_stddev = bb_stddev
        self.required_lookback = bb_period + 5

        self.name = f"{name}_{bb_period}_{bb_stddev:.1f}"

        self.logger.info(
            f"Initialized {self.name} with period={bb_period}, stddev={bb_stddev:.1f}"
        )

    def required_indicators(self) -> List[str]:
        """Required indicators: Only need close price."""
        return ["close", "high", "low"]

    def supports_market(self, market_type: str) -> bool:
        """Supports all markets."""
        return True

    def supports_regime(self, regime: MarketRegime) -> bool:
        """
        Works best in RANGING and CHOPPY markets.
        Can work in VOLATILE markets, but trending is not ideal.
        """
        return regime in [MarketRegime.RANGING, MarketRegime.CHOPPY, MarketRegime.VOLATILE]

    def supports_volatility(self, volatility: VolatilityBucket) -> bool:
        """Works best in low to medium volatility."""
        return volatility in [VolatilityBucket.LOW, VolatilityBucket.MEDIUM]

    def generate_signal(self, context: StrategyContext) -> Signal:
        """
        Generate trading signal based on Bollinger Band touch.

        Logic (PRESERVED FROM PRODUCTION):
        1. Calculate Bollinger Bands
        2. Check if current bar's low <= lower band (BUY signal)
        3. Check if current bar's high >= upper band (SELL signal)

        Args:
            context: Strategy context with market data

        Returns:
            Signal with direction and confidence
        """
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

            start_idx = max(0, current_idx - self.required_lookback + 1)
            history_slice = history.iloc[start_idx:current_idx + 1]

            if len(history_slice) < self.bb_period:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Insufficient data for BB calculation"
                )

            bbands_df = calculate_bollinger_bands(
                history_slice["close"],
                period=self.bb_period,
                std_dev=self.bb_stddev
            )

            if bbands_df is None or bbands_df.empty:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Bollinger Bands calculation failed"
                )

            lower_band_col = f"BBL_{self.bb_period}_{self.bb_stddev:.1f}"
            upper_band_col = f"BBU_{self.bb_period}_{self.bb_stddev:.1f}"

            if lower_band_col not in bbands_df.columns or upper_band_col not in bbands_df.columns:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Bollinger Band columns not found"
                )

            current_lower_band = bbands_df[lower_band_col].iloc[-1]
            current_upper_band = bbands_df[upper_band_col].iloc[-1]

            current_low = context.current_bar["low"]
            current_high = context.current_bar["high"]

            if pd.isna(current_lower_band) or pd.isna(current_upper_band):
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Invalid band values"
                )

            if pd.isna(current_low) or pd.isna(current_high):
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Invalid OHLC values"
                )

            if current_low <= current_lower_band:
                band_range = current_upper_band - current_lower_band
                penetration_depth = (
                    (current_lower_band - current_low) / band_range
                    if band_range > 0 else 0
                )

                confidence = min(0.6 + (penetration_depth * 0.4), 1.0)

                return Signal(
                    direction="BUY",
                    confidence=confidence,
                    strategy_name=self.name,
                    reasoning=(
                        f"Mean reversion BUY: Price ({current_low:.2f}) "
                        f"touched lower BB ({current_lower_band:.2f})"
                    ),
                    additional_context={
                        "price": current_low,
                        "lower_band": current_lower_band,
                        "upper_band": current_upper_band,
                        "band_width": band_range,
                        "penetration": penetration_depth
                    },
                    suggested_sl_mult=1.2,
                    suggested_tp_mult=1.5
                )

            if current_high >= current_upper_band:
                band_range = current_upper_band - current_lower_band
                penetration_depth = (
                    (current_high - current_upper_band) / band_range
                    if band_range > 0 else 0
                )

                confidence = min(0.6 + (penetration_depth * 0.4), 1.0)

                return Signal(
                    direction="SELL",
                    confidence=confidence,
                    strategy_name=self.name,
                    reasoning=(
                        f"Mean reversion SELL: Price ({current_high:.2f}) "
                        f"touched upper BB ({current_upper_band:.2f})"
                    ),
                    additional_context={
                        "price": current_high,
                        "lower_band": current_lower_band,
                        "upper_band": current_upper_band,
                        "band_width": band_range,
                        "penetration": penetration_depth
                    },
                    suggested_sl_mult=1.2,
                    suggested_tp_mult=1.5
                )

            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=(
                    f"No band touch: Price in middle. "
                    f"Low={current_low:.2f}, High={current_high:.2f}, "
                    f"Bands=[{current_lower_band:.2f}, {current_upper_band:.2f}]"
                ),
                additional_context={
                    "lower_band": current_lower_band,
                    "upper_band": current_upper_band,
                    "band_width": current_upper_band - current_lower_band
                }
            )

        except Exception as e:
            self.logger.error(f"Error in {self.name}: {e}", exc_info=True)
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=f"Exception: {str(e)}"
            )

    def __str__(self) -> str:
        return f"{self.name} (Period: {self.bb_period}, StdDev: {self.bb_stddev:.1f})"


def create_bb_mean_reversion_strategy(
    bb_period: int = 20,
    bb_stddev: float = 2.0
) -> BBMeanReversion:
    """
    Factory function for creating Bollinger Bands Mean Reversion strategy.

    Args:
        bb_period: Moving average period
        bb_stddev: Standard deviation multiplier

    Returns:
        BBMeanReversion strategy instance
    """
    return BBMeanReversion(
        bb_period=bb_period,
        bb_stddev=bb_stddev
    )
