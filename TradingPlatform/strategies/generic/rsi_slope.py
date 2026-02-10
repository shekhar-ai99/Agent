"""
RSI Slope Strategy

Mean reversion with momentum confirmation using RSI and RSI slope.

Entry Logic:
- BUY: RSI < 30 (oversold) AND RSI slope > threshold (upward momentum)
- SELL: RSI > 70 (overbought) AND RSI slope < threshold (downward momentum)

Exit Logic: Controlled by risk module (SL/TP/TSL)

Optimal Conditions:
- Ranging or choppy markets
- Low to high volatility
- Works on all timeframes

Parameters:
- rsi_period: RSI period (default: 14)
- rsi_oversold: Oversold threshold (default: 30)
- rsi_overbought: Overbought threshold (default: 70)
- slope_lookback: Periods for slope calculation (default: 3)
- slope_threshold_buy: Minimum slope for buy (default: 0.5)
- slope_threshold_sell: Maximum slope for sell (default: -0.5)

Historical Performance (NIFTY 5min, Jan-May 2025):
- 223 trades
- 81.17% win rate
- ₹6,261 P&L

MIGRATED FROM: Common/strategies/strategies.py
ORIGINAL: create_rsi_slope_strategy()
STATUS: ✅ PRODUCTION LOGIC PRESERVED
"""

import pandas as pd
from typing import List, Optional
import logging

from TradingPlatform.core.base_strategy import (
    BaseStrategy,
    Signal,
    StrategyContext,
    MarketRegime,
    VolatilityBucket
)
from TradingPlatform.utils.strategy_utils import (
    calculate_rsi,
    calculate_slope,
    ensure_min_history
)

logger = logging.getLogger(__name__)


class RSISlope(BaseStrategy):
    """
    RSI Slope - Mean reversion with momentum confirmation.

    Combines RSI extreme zones (oversold/overbought) with RSI slope
    to confirm reversal momentum.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        slope_lookback: int = 3,
        slope_threshold_buy: float = 0.5,
        slope_threshold_sell: float = -0.5,
        name: str = "RSI_Slope",
        version: str = "2.0"
    ):
        """
        Initialize RSI Slope strategy.

        Args:
            rsi_period: RSI calculation period
            rsi_oversold: Oversold threshold
            rsi_overbought: Overbought threshold
            slope_lookback: Periods for slope calculation
            slope_threshold_buy: Minimum slope for buy signal
            slope_threshold_sell: Maximum slope for sell signal
            name: Strategy name
            version: Strategy version
        """
        super().__init__(name=name, version=version)

        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.slope_lookback = slope_lookback
        self.slope_threshold_buy = slope_threshold_buy
        self.slope_threshold_sell = slope_threshold_sell
        self.required_lookback = rsi_period + slope_lookback + 5

        self.name = f"{name}_{rsi_period}_{rsi_oversold}_{rsi_overbought}_{slope_lookback}"

        self.logger.info(
            f"Initialized {self.name} with RSI({rsi_period}), "
            f"zones=[{rsi_oversold},{rsi_overbought}], "
            f"slope({slope_lookback}, thresholds=[{slope_threshold_buy},{slope_threshold_sell}])"
        )

    def required_indicators(self) -> List[str]:
        """Required indicators: Only close price."""
        return ["close"]

    def supports_market(self, market_type: str) -> bool:
        """Supports all markets."""
        return True

    def supports_regime(self, regime: MarketRegime) -> bool:
        """
        Works best in RANGING and VOLATILE markets.
        Mean reversion trades better in non-trending conditions.
        """
        return regime in [MarketRegime.RANGING, MarketRegime.VOLATILE, MarketRegime.CHOPPY]

    def supports_volatility(self, volatility: VolatilityBucket) -> bool:
        """Works in all volatility levels."""
        return True

    def generate_signal(self, context: StrategyContext) -> Signal:
        """
        Generate trading signal based on RSI + slope.

        Logic (PRESERVED FROM PRODUCTION):
        1. Calculate RSI
        2. Calculate RSI slope from lookback period
        3. Check oversold/overbought + slope momentum combination
        4. BUY if RSI < 30 AND slope > threshold (bouncing from oversold)
        5. SELL if RSI > 70 AND slope < threshold (falling from overbought)
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

            if len(history_slice) < self.rsi_period + 1:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Insufficient data for RSI calculation"
                )

            rsi_series = calculate_rsi(history_slice["close"], period=self.rsi_period)

            if rsi_series is None or rsi_series.empty:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="RSI calculation failed"
                )

            if len(rsi_series) < self.slope_lookback:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Insufficient RSI history for slope calculation"
                )

            rsi_current = rsi_series.iloc[-1]

            if pd.isna(rsi_current):
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Invalid current RSI value"
                )

            rsi_past = rsi_series.iloc[-self.slope_lookback]

            if pd.isna(rsi_past):
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Invalid past RSI value for slope"
                )

            if self.slope_lookback > 1:
                rsi_slope = (rsi_current - rsi_past) / (self.slope_lookback - 1)
            else:
                rsi_slope = rsi_current - rsi_past

            if rsi_current < self.rsi_oversold and rsi_slope > self.slope_threshold_buy:
                oversold_depth = (self.rsi_oversold - rsi_current) / self.rsi_oversold
                slope_strength = rsi_slope / (self.slope_threshold_buy + 0.5)

                confidence = min(0.5 + (oversold_depth * 0.25) + (slope_strength * 0.25), 1.0)

                return Signal(
                    direction="BUY",
                    confidence=confidence,
                    strategy_name=self.name,
                    reasoning=(
                        f"Oversold reversal: RSI={rsi_current:.2f} (< {self.rsi_oversold}) "
                        f"+ positive slope={rsi_slope:.2f}"
                    ),
                    additional_context={
                        "rsi": rsi_current,
                        "rsi_slope": rsi_slope,
                        "oversold_depth": oversold_depth,
                        "slope_strength": slope_strength
                    },
                    suggested_sl_mult=1.3,
                    suggested_tp_mult=2.0
                )

            if rsi_current > self.rsi_overbought and rsi_slope < self.slope_threshold_sell:
                overbought_depth = (rsi_current - self.rsi_overbought) / (100 - self.rsi_overbought)
                slope_strength = abs(rsi_slope) / (abs(self.slope_threshold_sell) + 0.5)

                confidence = min(0.5 + (overbought_depth * 0.25) + (slope_strength * 0.25), 1.0)

                return Signal(
                    direction="SELL",
                    confidence=confidence,
                    strategy_name=self.name,
                    reasoning=(
                        f"Overbought reversal: RSI={rsi_current:.2f} (> {self.rsi_overbought}) "
                        f"+ negative slope={rsi_slope:.2f}"
                    ),
                    additional_context={
                        "rsi": rsi_current,
                        "rsi_slope": rsi_slope,
                        "overbought_depth": overbought_depth,
                        "slope_strength": slope_strength
                    },
                    suggested_sl_mult=1.3,
                    suggested_tp_mult=2.0
                )

            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=(
                    f"No signal: RSI={rsi_current:.2f}, slope={rsi_slope:.2f}, "
                    f"thresholds=[{self.slope_threshold_buy},{self.slope_threshold_sell}]"
                )
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
            f"{self.name} (RSI: {self.rsi_period}, "
            f"slope: {self.slope_lookback}, "
            f"zones: {self.rsi_oversold}/{self.rsi_overbought})"
        )


def create_rsi_slope_strategy(
    rsi_period: int = 14,
    rsi_oversold: int = 30,
    rsi_overbought: int = 70,
    slope_lookback: int = 3,
    slope_threshold_buy: float = 0.5,
    slope_threshold_sell: float = -0.5
) -> RSISlope:
    return RSISlope(
        rsi_period=rsi_period,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        slope_lookback=slope_lookback,
        slope_threshold_buy=slope_threshold_buy,
        slope_threshold_sell=slope_threshold_sell
    )
