"""
EMA Crossover Strategy

Trend-following strategy based on exponential moving average crossovers.

Entry Logic:
- BUY: When fast EMA crosses above slow EMA (bullish crossover)
- SELL: When fast EMA crosses below slow EMA (bearish crossover)

Exit Logic: Controlled by risk module (SL/TP/TSL)

Optimal Conditions:
- Trending markets (ADX > 25)
- Medium to low volatility
- Works on all timeframes

Parameters:
- ema_short_period: Fast EMA period (default: 9)
- ema_long_period: Slow EMA period (default: 21)

Historical Performance (NIFTY 5min, Jan-May 2025):
- 161 trades
- 66.46% win rate
- ₹2,670 P&L

MIGRATED FROM: Common/strategies/strategies.py
ORIGINAL: strategy_ema_crossover_factory()
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
    calculate_ema,
    detect_crossover,
    ensure_min_history
)

logger = logging.getLogger(__name__)


class EMACrossover(BaseStrategy):
    """
    EMA Crossover - Classic trend following strategy.

    Generates signals when fast EMA crosses slow EMA.
    Requires sufficient historical data for EMA calculation.
    """

    def __init__(
        self,
        ema_short_period: int = 9,
        ema_long_period: int = 21,
        name: str = "EMA_Crossover",
        version: str = "2.0"
    ):
        """
        Initialize EMA Crossover strategy.

        Args:
            ema_short_period: Fast EMA period
            ema_long_period: Slow EMA period
            name: Strategy name
            version: Strategy version
        """
        super().__init__(name=name, version=version)

        self.ema_short_period = ema_short_period
        self.ema_long_period = ema_long_period
        self.required_lookback = ema_long_period + 5

        self.name = f"{name}_{ema_short_period}_{ema_long_period}"

        self.logger.info(
            f"Initialized {self.name} with short={ema_short_period}, long={ema_long_period}"
        )

    def required_indicators(self) -> List[str]:
        """
        Required indicators: EMAs will be calculated dynamically.
        Only need basic OHLCV.
        """
        return ["close"]

    def supports_market(self, market_type: str) -> bool:
        """Supports all markets."""
        return True

    def supports_regime(self, regime: MarketRegime) -> bool:
        """
        Works best in trending markets.
        Can operate in ranging but with reduced confidence.
        """
        return regime in [
            MarketRegime.TRENDING,
            MarketRegime.RANGING,
            MarketRegime.UNKNOWN
        ]

    def supports_volatility(self, volatility: VolatilityBucket) -> bool:
        """
        Works in all volatility conditions.
        Performs best in medium volatility.
        """
        return True

    def generate_signal(self, context: StrategyContext) -> Signal:
        """
        Generate trading signal based on EMA crossover.

        Logic (PRESERVED FROM PRODUCTION):
        1. Calculate short and long EMAs using pandas_ta
        2. Detect crossover between EMAs
        3. BUY on bullish crossover (fast > slow after fast <= slow)
        4. SELL on bearish crossover (fast < slow after fast >= slow)

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

            ema_short = calculate_ema(history["close"], self.ema_short_period)
            ema_long = calculate_ema(history["close"], self.ema_long_period)

            if ema_short is None or ema_long is None:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="EMA calculation failed"
                )

            if ema_short.empty or ema_long.empty or len(ema_short) < 2 or len(ema_long) < 2:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Insufficient EMA data"
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

            current_short = ema_short.iloc[current_idx]
            current_long = ema_long.iloc[current_idx]
            prev_short = ema_short.iloc[current_idx - 1]
            prev_long = ema_long.iloc[current_idx - 1]

            if pd.isna(current_short) or pd.isna(current_long):
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Invalid current EMA values"
                )

            if pd.isna(prev_short) or pd.isna(prev_long):
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Invalid previous EMA values"
                )

            if prev_short <= prev_long and current_short > current_long:
                separation = (current_short - current_long) / current_long
                confidence = min(0.5 + (separation * 100), 1.0)

                if context.regime == MarketRegime.TRENDING:
                    confidence = min(confidence * 1.2, 1.0)

                return Signal(
                    direction="BUY",
                    confidence=confidence,
                    strategy_name=self.name,
                    reasoning=(
                        f"Bullish EMA crossover: EMA{self.ema_short_period}={current_short:.2f} "
                        f"crossed above EMA{self.ema_long_period}={current_long:.2f}"
                    ),
                    additional_context={
                        "ema_short": current_short,
                        "ema_long": current_long,
                        "separation_pct": separation * 100
                    }
                )

            if prev_short >= prev_long and current_short < current_long:
                separation = (current_long - current_short) / current_long
                confidence = min(0.5 + (separation * 100), 1.0)

                if context.regime == MarketRegime.TRENDING:
                    confidence = min(confidence * 1.2, 1.0)

                return Signal(
                    direction="SELL",
                    confidence=confidence,
                    strategy_name=self.name,
                    reasoning=(
                        f"Bearish EMA crossover: EMA{self.ema_short_period}={current_short:.2f} "
                        f"crossed below EMA{self.ema_long_period}={current_long:.2f}"
                    ),
                    additional_context={
                        "ema_short": current_short,
                        "ema_long": current_long,
                        "separation_pct": separation * 100
                    }
                )

            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=(
                    f"No crossover: EMA{self.ema_short_period}={current_short:.2f}, "
                    f"EMA{self.ema_long_period}={current_long:.2f}"
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
            f"{self.name} (Short: {self.ema_short_period}, "
            f"Long: {self.ema_long_period})"
        )


def create_ema_crossover_strategy(
    ema_short_period: int = 9,
    ema_long_period: int = 21
) -> EMACrossover:
    """
    Factory function for creating EMA Crossover strategy.

    Provides backward compatibility with old factory pattern.

    Args:
        ema_short_period: Fast EMA period
        ema_long_period: Slow EMA period

    Returns:
        EMACrossover strategy instance
    """
    return EMACrossover(
        ema_short_period=ema_short_period,
        ema_long_period=ema_long_period
    )
