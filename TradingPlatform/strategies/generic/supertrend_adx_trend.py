"""
SuperTrend + ADX Strategy

Trend-following strategy combining SuperTrend direction with ADX confirmation.

Entry Logic:
- BUY: SuperTrend direction = 1 (bullish) AND ADX >= threshold
- SELL: SuperTrend direction = -1 (bearish) AND ADX >= threshold

Exit Logic: Controlled by risk module (SL/TP/TSL)

Optimal Conditions:
- Trending markets only (ADX > 25 required for signals)
- Medium to high volatility
- Works on all timeframes

Parameters:
- st_period: SuperTrend ATR period (default: 10)
- st_multiplier: SuperTrend multiplier (default: 3.0)
- adx_period: ADX calculation period (default: 14)
- adx_threshold: Minimum ADX for signals (default: 25)

Historical Performance (NIFTY 5min, Jan-May 2025):
- 1,454 trades
- 84.53% win rate ⭐ (2nd best)
- ₹48,747 P&L ⭐ (2nd best)

MIGRATED FROM: Common/strategies/strategies.py
ORIGINAL: create_supertrend_adx_strategy()
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
    calculate_supertrend,
    calculate_adx,
    ensure_min_history
)

logger = logging.getLogger(__name__)


class SuperTrendADXTrend(BaseStrategy):
    """
    SuperTrend + ADX Filter - Trend following with trend strength confirmation.

    Generates signals when SuperTrend direction is confirmed by strong ADX.
    Only trades when market is trending (ADX >= threshold).
    """

    def __init__(
        self,
        st_period: int = 10,
        st_multiplier: float = 3.0,
        adx_period: int = 14,
        adx_threshold: int = 25,
        name: str = "SuperTrend_ADX",
        version: str = "2.0"
    ):
        """
        Initialize SuperTrend + ADX strategy.

        Args:
            st_period: SuperTrend ATR period
            st_multiplier: SuperTrend ATR multiplier
            adx_period: ADX calculation period
            adx_threshold: ADX minimum threshold for signals
            name: Strategy name
            version: Strategy version
        """
        super().__init__(name=name, version=version)

        self.st_period = st_period
        self.st_multiplier = st_multiplier
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.required_lookback = max(st_period, adx_period) + 50

        self.name = f"{name}_{st_period}_{st_multiplier:.1f}_{adx_period}_{adx_threshold}"

        self.logger.info(
            f"Initialized {self.name} with ST({st_period},{st_multiplier:.1f}), "
            f"ADX({adx_period},{adx_threshold})"
        )

    def required_indicators(self) -> List[str]:
        """Required indicators: Will calculate dynamically."""
        return ["high", "low", "close"]

    def supports_market(self, market_type: str) -> bool:
        """Supports all markets."""
        return True

    def supports_regime(self, regime: MarketRegime) -> bool:
        """
        ONLY works in TRENDING markets.
        ADX threshold filter ensures we only trade strong trends.
        """
        return regime == MarketRegime.TRENDING

    def supports_volatility(self, volatility: VolatilityBucket) -> bool:
        """Works best in medium to high volatility."""
        return volatility in [VolatilityBucket.MEDIUM, VolatilityBucket.HIGH]

    def generate_signal(self, context: StrategyContext) -> Signal:
        """
        Generate trading signal based on SuperTrend + ADX.

        Logic (PRESERVED FROM PRODUCTION):
        1. Verify sufficient historical data
        2. Calculate SuperTrend from high/low/close
        3. Calculate ADX from high/low/close
        4. Check SuperTrend direction and ADX strength
        5. BUY if direction=1 and ADX >= threshold
        6. SELL if direction=-1 and ADX >= threshold
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

            if len(history_slice) < max(self.st_period, self.adx_period) + 1:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Insufficient data for indicator calculation"
                )

            supertrend_df = calculate_supertrend(
                history_slice["high"],
                history_slice["low"],
                history_slice["close"],
                period=self.st_period,
                multiplier=self.st_multiplier
            )

            if supertrend_df is None or supertrend_df.empty:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="SuperTrend calculation failed"
                )

            adx_df = calculate_adx(
                history_slice["high"],
                history_slice["low"],
                history_slice["close"],
                period=self.adx_period
            )

            if adx_df is None or adx_df.empty:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="ADX calculation failed"
                )

            st_direction_col = f"SUPERTd_{self.st_period}_{self.st_multiplier}"
            adx_val_col = f"ADX_{self.adx_period}"

            if st_direction_col not in supertrend_df.columns:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning=f"SuperTrend direction column not found: {st_direction_col}"
                )

            if adx_val_col not in adx_df.columns:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning=f"ADX column not found: {adx_val_col}"
                )

            current_st_direction = supertrend_df[st_direction_col].iloc[-1]
            current_adx = adx_df[adx_val_col].iloc[-1]

            if pd.isna(current_st_direction) or pd.isna(current_adx):
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Invalid indicator values"
                )

            if current_adx >= self.adx_threshold:
                if current_st_direction == 1:
                    adx_strength = (current_adx - self.adx_threshold) / 25
                    confidence = min(0.6 + (adx_strength * 0.4), 1.0)

                    return Signal(
                        direction="BUY",
                        confidence=confidence,
                        strategy_name=self.name,
                        reasoning=(
                            f"Bullish SuperTrend (dir={current_st_direction}) "
                            f"+ Strong ADX ({current_adx:.2f} >= {self.adx_threshold})"
                        ),
                        additional_context={
                            "st_direction": current_st_direction,
                            "adx": current_adx,
                            "adx_strength": adx_strength
                        },
                        suggested_sl_mult=1.5,
                        suggested_tp_mult=2.5
                    )

                if current_st_direction == -1:
                    adx_strength = (current_adx - self.adx_threshold) / 25
                    confidence = min(0.6 + (adx_strength * 0.4), 1.0)

                    return Signal(
                        direction="SELL",
                        confidence=confidence,
                        strategy_name=self.name,
                        reasoning=(
                            f"Bearish SuperTrend (dir={current_st_direction}) "
                            f"+ Strong ADX ({current_adx:.2f} >= {self.adx_threshold})"
                        ),
                        additional_context={
                            "st_direction": current_st_direction,
                            "adx": current_adx,
                            "adx_strength": adx_strength
                        },
                        suggested_sl_mult=1.5,
                        suggested_tp_mult=2.5
                    )

            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=(
                    f"ADX too weak ({current_adx:.2f} < {self.adx_threshold}). "
                    f"ST direction: {current_st_direction}"
                ),
                additional_context={
                    "st_direction": current_st_direction,
                    "adx": current_adx,
                    "adx_strength": current_adx - self.adx_threshold
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
        return (
            f"{self.name} (ST: {self.st_period}/{self.st_multiplier}, "
            f"ADX: {self.adx_period}/{self.adx_threshold})"
        )


def create_supertrend_adx_strategy(
    st_period: int = 10,
    st_multiplier: float = 3.0,
    adx_period: int = 14,
    adx_threshold: int = 25
) -> SuperTrendADXTrend:
    return SuperTrendADXTrend(
        st_period=st_period,
        st_multiplier=st_multiplier,
        adx_period=adx_period,
        adx_threshold=adx_threshold
    )
