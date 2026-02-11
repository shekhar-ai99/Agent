"""
Volatility Breakout with Bollinger Band Squeeze (VolatilityBreakout_BBS)

Two-phase strategy that detects volatility squeeze followed by expansion breakout.

Entry Logic (TWO PHASES):
1. SQUEEZE DETECTION: Bollinger Band bandwidth narrows (lower than last N bars)
   - Signals market consolidation
   - Confirms reduced volatility and potential coiling

2. BREAKOUT EXPANSION: Price breaks out as volatility expands
   - BUY: Close above upper band (when previous squeeze detected)
   - SELL: Close below lower band (when previous squeeze detected)

Critical Behavior:
- SQUEEZE checked on PREVIOUS bar bandwidth (t-1)
- BREAKOUT checked on CURRENT bar price vs bands (t)
- Requires bandwidth to be at lowest for N periods (squeeze_lookback)

Optimal Conditions:
- Works best in choppy/consolidating markets
- Ideal for low-to-medium volatility transition to breakout
- Works on all timeframes

Parameters:
- bb_period: Bollinger Bands period (default: 20)
- bb_stddev: BB standard deviations (default: 2.0)
- squeeze_lookback: Periods to check for squeeze (default: 20)
- squeeze_tolerance: Tolerance for lowest bandwidth (default: 1.05 = 5% above lowest)

Historical Performance (NIFTY 5min, Jan-May 2025):
- 79 trades (low frequency, high accuracy trades)
- 60.76% win rate
- ₹1,015 P&L

MIGRATED FROM: Common/strategies/strategies.py
ORIGINAL: create_volatility_breakout_bbs_strategy()
STATUS: ✅ PRODUCTION LOGIC PRESERVED
        ✅ TWO-PHASE SQUEEZE→BREAKOUT LOGIC IMPLEMENTED
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
    calculate_bollinger_bands,
    ensure_min_history,
    detect_volatility_squeeze
)

logger = logging.getLogger(__name__)


class VolatilityBreakoutBBS(BaseStrategy):
    """
    Volatility Breakout with Bollinger Band Squeeze.

    Two-phase strategy:
    1. Detects squeeze (narrow bandwidth)
    2. Trades breakout when volatility expands

    CRITICAL: Squeeze state tracked from previous bar, breakout from current.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_stddev: float = 2.0,
        squeeze_lookback: int = 20,
        squeeze_tolerance: float = 1.05,
        name: str = "VolatilityBreakout_BBS",
        version: str = "2.0"
    ):
        """
        Initialize Volatility Breakout BBS strategy.

        Args:
            bb_period: Bollinger Bands period
            bb_stddev: Standard deviations for bands
            squeeze_lookback: Periods to check for squeeze
            squeeze_tolerance: Tolerance % for lowest bandwidth
            name: Strategy name
            version: Strategy version
        """
        super().__init__(name=name, version=version)

        self.bb_period = bb_period
        self.bb_stddev = bb_stddev
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_tolerance = squeeze_tolerance

        self.required_lookback = bb_period + squeeze_lookback + 100

        self.name = (
            f"{name}_{bb_period}_{bb_stddev:.1f}_"
            f"sq{squeeze_lookback}_{squeeze_tolerance}"
        )

        self.logger.info(
            f"Initialized {self.name}: BB({bb_period},{bb_stddev}), "
            f"squeeze_window={squeeze_lookback}, "
            f"tolerance={squeeze_tolerance}"
        )

    def required_indicators(self) -> List[str]:
        """Required: close price for BB calculation."""
        return ["close"]

    def supports_market(self, market_type: str) -> bool:
        """Works in all markets."""
        return True

    def supports_regime(self, regime: MarketRegime) -> bool:
        """
        Best in RANGING or CHOPPY markets where squeeze is meaningful.
        Works in VOLATILE but less optimal (already wide).
        """
        return regime in [MarketRegime.RANGING, MarketRegime.CHOPPY, MarketRegime.VOLATILE]

    def supports_volatility(self, volatility: VolatilityBucket) -> bool:
        """Works in low-to-medium volatility (squeeze more likely)."""
        return volatility in [VolatilityBucket.LOW, VolatilityBucket.MEDIUM]

    def _calculate_bandwidth(self, upper_band: float, lower_band: float) -> float:
        """
        Calculate Bollinger Band bandwidth (width of bands).

        Args:
            upper_band: Upper band value
            lower_band: Lower band value

        Returns:
            Bandwidth (upper - lower)
        """
        if upper_band <= lower_band:
            return 0.0
        return upper_band - lower_band

    def _find_min_bandwidth(
        self,
        history: pd.Series,
        lookback_bars: int
    ) -> Optional[float]:
        """
        Find the minimum bandwidth in the last N bars.

        Args:
            history: Historical bandwidth data
            lookback_bars: Number of bars to check

        Returns:
            Minimum bandwidth or None if insufficient data
        """
        if len(history) < lookback_bars:
            return None

        recent_bandwidth = history.iloc[-lookback_bars:]
        if recent_bandwidth.empty:
            return None

        min_bandwidth = recent_bandwidth.min()

        if pd.isna(min_bandwidth) or min_bandwidth <= 0:
            return None

        return min_bandwidth

    def generate_signal(self, context: StrategyContext) -> Signal:
        """
        Generate trading signal based on BB squeeze + breakout.

        Logic (PRESERVED FROM PRODUCTION - TWO PHASE):

        PHASE 1 - SQUEEZE DETECTION (checked PREVIOUS bar):
        1. Calculate BB bandwidth for previous bar
        2. Check if previous bandwidth <= min_bandwidth × tolerance
        3. If squeeze detected on previous bar, set up for breakout trading

        PHASE 2 - BREAKOUT EXPANSION (checked CURRENT bar):
        1. Check if previous bar had squeeze
        2. If current price breaks outside bands:
           - BUY if close > upper_band (expansion above)
           - SELL if close < lower_band (expansion below)

        CRITICAL:
        - Squeeze detection uses PREVIOUS bar bandwidth
        - Breakout detection uses CURRENT bar price
        - This two-phase approach prevents instant entries
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

            if current_idx < self.bb_period + self.squeeze_lookback:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning=(
                        f"Insufficient history for squeeze detection: "
                        f"need {self.bb_period + self.squeeze_lookback}, have {current_idx}"
                    )
                )

            start_idx = max(0, current_idx - self.bb_period - self.squeeze_lookback + 1)
            history_slice = history.iloc[start_idx:current_idx + 1]

            bb_result = calculate_bollinger_bands(
                history_slice["close"],
                period=self.bb_period,
                std_dev=self.bb_stddev
            )

            if bb_result is None or bb_result.empty:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Bollinger Bands calculation failed"
                )

            # Extract BB columns - ta.bbands returns DataFrame with BBL_{period}_{std}, BBM_{period}_{std}, BBU_{period}_{std}, BBB_{period}_{std}
            bb_cols = bb_result.columns
            lower_col = [c for c in bb_cols if 'BBL_' in c]
            middle_col = [c for c in bb_cols if 'BBM_' in c]
            upper_col = [c for c in bb_cols if 'BBU_' in c]
            
            if not lower_col or not middle_col or not upper_col:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Bollinger Bands columns not found"
                )
            
            lower_band_series = bb_result[lower_col[0]]
            middle_band_series = bb_result[middle_col[0]]
            upper_band_series = bb_result[upper_col[0]]

            if len(upper_band_series) < 2:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Insufficient BB history"
                )

            current_upper = upper_band_series.iloc[-1]
            current_lower = lower_band_series.iloc[-1]
            previous_upper = upper_band_series.iloc[-2]
            previous_lower = lower_band_series.iloc[-2]

            if pd.isna(current_upper) or pd.isna(current_lower):
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Invalid current BB values"
                )

            previous_bandwidth = self._calculate_bandwidth(previous_upper, previous_lower)

            if previous_bandwidth <= 0:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Invalid previous bandwidth"
                )

            bandwidth_series = pd.Series([
                self._calculate_bandwidth(u, l)
                for u, l in zip(upper_band_series, lower_band_series)
            ])

            min_bandwidth = self._find_min_bandwidth(
                bandwidth_series,
                min(self.squeeze_lookback, len(bandwidth_series))
            )

            if min_bandwidth is None or min_bandwidth <= 0:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Cannot determine minimum bandwidth"
                )

            squeeze_threshold = min_bandwidth * self.squeeze_tolerance
            was_previous_squeezed = previous_bandwidth <= squeeze_threshold

            if not was_previous_squeezed:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning=(
                        f"No squeeze detected: prev_bandwidth={previous_bandwidth:.2f} > "
                        f"threshold={squeeze_threshold:.2f} "
                        f"(min_bandwidth={min_bandwidth:.2f})"
                    ),
                    additional_context={
                        "previous_bandwidth": previous_bandwidth,
                        "min_bandwidth": min_bandwidth,
                        "squeeze_threshold": squeeze_threshold
                    }
                )

            current_close = context.current_bar["close"]
            current_high = context.current_bar["high"]
            current_low = context.current_bar["low"]

            if current_close > current_upper:
                current_bandwidth = self._calculate_bandwidth(current_upper, current_lower)
                breakout_strength = (
                    (current_close - current_upper) / current_bandwidth
                    if current_bandwidth > 0 else 0.5
                )
                confidence = min(0.5 + (breakout_strength * 0.5), 1.0)

                return Signal(
                    direction="BUY",
                    confidence=confidence,
                    strategy_name=self.name,
                    reasoning=(
                        f"Squeeze breakout (bullish): squeeze_prev({previous_bandwidth:.2f}) "
                        f"+ breakout_current({current_close:.2f} > {current_upper:.2f})"
                    ),
                    additional_context={
                        "previous_bandwidth": previous_bandwidth,
                        "min_bandwidth": min_bandwidth,
                        "current_bandwidth": current_bandwidth,
                        "upper_band": current_upper,
                        "lower_band": current_lower,
                        "breakout_strength": breakout_strength
                    },
                    suggested_sl_mult=1.3,
                    suggested_tp_mult=2.0
                )

            if current_close < current_lower:
                current_bandwidth = self._calculate_bandwidth(current_upper, current_lower)
                breakout_strength = (
                    (current_lower - current_close) / current_bandwidth
                    if current_bandwidth > 0 else 0.5
                )
                confidence = min(0.5 + (breakout_strength * 0.5), 1.0)

                return Signal(
                    direction="SELL",
                    confidence=confidence,
                    strategy_name=self.name,
                    reasoning=(
                        f"Squeeze breakout (bearish): squeeze_prev({previous_bandwidth:.2f}) "
                        f"+ breakout_current({current_close:.2f} < {current_lower:.2f})"
                    ),
                    additional_context={
                        "previous_bandwidth": previous_bandwidth,
                        "min_bandwidth": min_bandwidth,
                        "current_bandwidth": current_bandwidth,
                        "upper_band": current_upper,
                        "lower_band": current_lower,
                        "breakout_strength": breakout_strength
                    },
                    suggested_sl_mult=1.3,
                    suggested_tp_mult=2.0
                )

            current_bandwidth = self._calculate_bandwidth(current_upper, current_lower)
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=(
                    f"Squeeze detected (coiling): squeeze_prev={previous_bandwidth:.2f}, "
                    f"wait for breakout. Current price within bands "
                    f"[{current_lower:.2f}, {current_upper:.2f}]"
                ),
                additional_context={
                    "previous_bandwidth": previous_bandwidth,
                    "min_bandwidth": min_bandwidth,
                    "current_bandwidth": current_bandwidth,
                    "upper_band": current_upper,
                    "lower_band": current_lower,
                    "squeeze_state": "active"
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
            f"{self.name} (BB: {self.bb_period}, "
            f"squeeze: {self.squeeze_lookback}x{self.squeeze_tolerance})"
        )


def create_volatility_breakout_bbs_strategy(
    bb_period: int = 20,
    bb_stddev: float = 2.0,
    squeeze_lookback: int = 20,
    squeeze_tolerance: float = 1.05
) -> VolatilityBreakoutBBS:
    """
    Factory function for creating Volatility Breakout BBS strategy.

    Args:
        bb_period: Bollinger Bands period
        bb_stddev: Standard deviations
        squeeze_lookback: Squeeze detection window
        squeeze_tolerance: Tolerance for minimum bandwidth

    Returns:
        VolatilityBreakoutBBS strategy instance
    """
    return VolatilityBreakoutBBS(
        bb_period=bb_period,
        bb_stddev=bb_stddev,
        squeeze_lookback=squeeze_lookback,
        squeeze_tolerance=squeeze_tolerance
    )
