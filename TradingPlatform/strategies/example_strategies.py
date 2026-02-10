"""
Example Strategies

These demonstrate how strategies should be implemented.
All strategies inherit from BaseStrategy and are market-agnostic.
"""

import logging
import pandas as pd
from core.base_strategy import BaseStrategy, Signal, StrategyContext, MarketRegime, VolatilityBucket

logger = logging.getLogger(__name__)


class RSIStrategy(BaseStrategy):
    """
    RSI-based mean reversion strategy.
    
    - Buys when RSI < 30 (oversold)
    - Sells when RSI > 70 (overbought)
    - Works in ranging/volatile markets
    """

    def __init__(self, rsi_period: int = 14):
        super().__init__(name="RSI_MeanReversion", version="1.0")
        self.rsi_period = rsi_period

    def required_indicators(self):
        return ["RSI", "close"]

    def supports_market(self, market_type: str) -> bool:
        return market_type in ["indian", "crypto"]

    def supports_regime(self, regime: MarketRegime) -> bool:
        return regime in [MarketRegime.RANGING, MarketRegime.VOLATILE]

    def supports_volatility(self, volatility: VolatilityBucket) -> bool:
        return volatility in [VolatilityBucket.MEDIUM, VolatilityBucket.HIGH]

    def generate_signal(self, context: StrategyContext) -> Signal:
        rsi = context.current_bar.get("RSI", 50)
        close = context.current_bar.get("close", 0)

        if rsi < 30:
            return Signal(
                direction="BUY",
                confidence=min((30 - rsi) / 30, 1.0),  # Higher confidence when more oversold
                strategy_name=self.name,
                reasoning=f"RSI {rsi:.2f} is oversold (< 30), potential reversal",
            )
        elif rsi > 70:
            return Signal(
                direction="SELL",
                confidence=min((rsi - 70) / 30, 1.0),  # Higher confidence when more overbought
                strategy_name=self.name,
                reasoning=f"RSI {rsi:.2f} is overbought (> 70), potential reversal",
            )
        else:
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning="RSI in neutral zone",
            )


class MAStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    
    - Buy when fast MA > slow MA
    - Sell when fast MA < slow MA
    - Works in trending markets
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__(name="MA_Crossover", version="1.0")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def required_indicators(self):
        return [f"SMA_{self.fast_period}", f"SMA_{self.slow_period}", "close"]

    def supports_market(self, market_type: str) -> bool:
        return market_type in ["indian", "crypto"]

    def supports_regime(self, regime: MarketRegime) -> bool:
        return regime == MarketRegime.TRENDING

    def supports_volatility(self, volatility: VolatilityBucket) -> bool:
        return volatility in [VolatilityBucket.LOW, VolatilityBucket.MEDIUM]

    def generate_signal(self, context: StrategyContext) -> Signal:
        fast_ma = context.current_bar.get(f"SMA_{self.fast_period}", None)
        slow_ma = context.current_bar.get(f"SMA_{self.slow_period}", None)
        close = context.current_bar.get("close", 0)

        if fast_ma is None or slow_ma is None:
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning="Required moving averages not available",
            )

        distance_percent = abs(fast_ma - slow_ma) / slow_ma * 100

        if fast_ma > slow_ma:
            return Signal(
                direction="BUY",
                confidence=min(distance_percent / 5, 1.0),
                strategy_name=self.name,
                reasoning=f"Fast MA ({fast_ma:.2f}) > Slow MA ({slow_ma:.2f}), trending up",
            )
        elif fast_ma < slow_ma:
            return Signal(
                direction="SELL",
                confidence=min(distance_percent / 5, 1.0),
                strategy_name=self.name,
                reasoning=f"Fast MA ({fast_ma:.2f}) < Slow MA ({slow_ma:.2f}), trending down",
            )
        else:
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning="Moving averages are equal",
            )


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy.
    
    - Buy when price touches lower band
    - Sell when price touches upper band
    - Works in ranging markets
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(name="BB_MeanReversion", version="1.0")
        self.period = period
        self.std_dev = std_dev

    def required_indicators(self):
        return ["BB_Upper", "BB_Lower", "close"]

    def supports_market(self, market_type: str) -> bool:
        return market_type in ["indian", "crypto"]

    def supports_regime(self, regime: MarketRegime) -> bool:
        return regime in [MarketRegime.RANGING]

    def supports_volatility(self, volatility: VolatilityBucket) -> bool:
        return volatility in [VolatilityBucket.LOW, VolatilityBucket.MEDIUM]

    def generate_signal(self, context: StrategyContext) -> Signal:
        upper_band = context.current_bar.get("BB_Upper", None)
        lower_band = context.current_bar.get("BB_Lower", None)
        close = context.current_bar.get("close", 0)

        if upper_band is None or lower_band is None:
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning="Bollinger Bands not available",
            )

        band_width = upper_band - lower_band

        if close <= lower_band:
            confidence = min((lower_band - close) / band_width, 1.0)
            return Signal(
                direction="BUY",
                confidence=confidence,
                strategy_name=self.name,
                reasoning=f"Price ({close:.2f}) at lower band ({lower_band:.2f}), mean reversion buy",
            )
        elif close >= upper_band:
            confidence = min((close - upper_band) / band_width, 1.0)
            return Signal(
                direction="SELL",
                confidence=confidence,
                strategy_name=self.name,
                reasoning=f"Price ({close:.2f}) at upper band ({upper_band:.2f}), mean reversion sell",
            )
        else:
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning="Price within bands",
            )


class SuperTrendADXStrategy(BaseStrategy):
    """
    SuperTrend + ADX Trend Following Strategy.
    
    - Buy when SuperTrend=1 AND ADX >= threshold
    - Sell when SuperTrend=-1 AND ADX >= threshold
    - Only trades when trend is strong (ADX confirms)
    - Works in strong trending markets
    
    [Preserved from legacy: Common/strategies.py]
    """

    def __init__(self, st_period: int = 10, st_multiplier: float = 3.0, adx_period: int = 14, adx_threshold: int = 25):
        super().__init__(name="SuperTrend_ADX_Trend", version="1.0")
        self.st_period = st_period
        self.st_multiplier = st_multiplier
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

    def required_indicators(self):
        return [f"SUPERT_{self.st_period}", f"SUPERTD_{self.st_period}", f"ADX_{self.adx_period}", "close"]

    def supports_market(self, market_type: str) -> bool:
        return market_type in ["indian", "crypto"]

    def supports_regime(self, regime: MarketRegime) -> bool:
        return regime == MarketRegime.TRENDING

    def supports_volatility(self, volatility: VolatilityBucket) -> bool:
        return volatility in [VolatilityBucket.MEDIUM, VolatilityBucket.HIGH]

    def generate_signal(self, context: StrategyContext) -> Signal:
        st_direction = context.current_bar.get(f"SUPERTD_{self.st_period}", 0)
        adx = context.current_bar.get(f"ADX_{self.adx_period}", 0)
        close = context.current_bar.get("close", 0)

        if pd.isna(adx) or pd.isna(st_direction):
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning="ADX/SuperTrend not ready",
            )

        # Only trade when ADX is strong enough
        if adx < self.adx_threshold:
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=f"ADX {adx:.2f} below threshold {self.adx_threshold}, trend weak",
            )

        if st_direction == 1:
            confidence = min(max((adx - self.adx_threshold) / (50 - self.adx_threshold), 0.0), 1.0)
            return Signal(
                direction="BUY",
                confidence=confidence,
                strategy_name=self.name,
                reasoning=f"SuperTrend=UP, ADX {adx:.2f} confirms strong trend",
            )
        elif st_direction == -1:
            confidence = min(max((adx - self.adx_threshold) / (50 - self.adx_threshold), 0.0), 1.0)
            return Signal(
                direction="SELL",
                confidence=confidence,
                strategy_name=self.name,
                reasoning=f"SuperTrend=DOWN, ADX {adx:.2f} confirms strong trend",
            )
        else:
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning="SuperTrend direction unclear",
            )


class RSISlopeStrategy(BaseStrategy):
    """
    RSI + Slope Mean Reversion Strategy.
    
    - Buy when RSI < 30 AND RSI slope is positive
    - Sell when RSI > 70 AND RSI slope is negative
    - Adds momentum confirmation to RSI extremes
    - Works in ranging/volatile markets
    
    [Preserved from legacy: Common/strategies.py]
    """

    def __init__(self, rsi_period: int = 14, slope_lookback: int = 3, 
                 rsi_oversold: int = 30, rsi_overbought: int = 70,
                 slope_threshold_buy: float = 0.5, slope_threshold_sell: float = -0.5):
        super().__init__(name="RSI_Slope_Momentum", version="1.0")
        self.rsi_period = rsi_period
        self.slope_lookback = slope_lookback
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.slope_threshold_buy = slope_threshold_buy
        self.slope_threshold_sell = slope_threshold_sell

    def required_indicators(self):
        return ["RSI", "close"]

    def supports_market(self, market_type: str) -> bool:
        return market_type in ["indian", "crypto"]

    def supports_regime(self, regime: MarketRegime) -> bool:
        return regime in [MarketRegime.RANGING, MarketRegime.VOLATILE]

    def supports_volatility(self, volatility: VolatilityBucket) -> bool:
        return volatility in [VolatilityBucket.MEDIUM, VolatilityBucket.HIGH]

    def generate_signal(self, context: StrategyContext) -> Signal:
        rsi = context.current_bar.get("RSI", 50)
        close = context.current_bar.get("close", 0)
        
        # Calculate slope from history if available
        rsi_history = context.get_indicator_history("RSI", self.slope_lookback)
        slope = 0.0
        
        if rsi_history and len(rsi_history) >= 2:
            # Simple slope: (current - lookback_ago) / periods
            slope = (rsi_history[-1] - rsi_history[0]) / (len(rsi_history) - 1) if len(rsi_history) > 1 else 0

        if rsi < self.rsi_oversold and slope > self.slope_threshold_buy:
            confidence = min((self.rsi_oversold - rsi) / self.rsi_oversold * 0.7 + slope / 2, 1.0)
            return Signal(
                direction="BUY",
                confidence=confidence,
                strategy_name=self.name,
                reasoning=f"RSI {rsi:.2f} oversold + slope {slope:.2f} positive, momentum reversal",
            )
        elif rsi > self.rsi_overbought and slope < self.slope_threshold_sell:
            confidence = min((rsi - self.rsi_overbought) / (100 - self.rsi_overbought) * 0.7 + abs(slope) / 2, 1.0)
            return Signal(
                direction="SELL",
                confidence=confidence,
                strategy_name=self.name,
                reasoning=f"RSI {rsi:.2f} overbought + slope {slope:.2f} negative, momentum reversal",
            )
        else:
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=f"RSI {rsi:.2f} neutral or slope insufficient",
            )


# Strategy registry
STRATEGY_REGISTRY = {
    "RSI_MeanReversion": RSIStrategy,
    "RSI_Slope_Momentum": RSISlopeStrategy,
    "MA_Crossover": MAStrategy,
    "SuperTrend_ADX_Trend": SuperTrendADXStrategy,
    "BB_MeanReversion": BollingerBandsStrategy,
}


def instantiate_all_strategies():
    """Create instances of all registered strategies"""
    return {name: strategy_class() for name, strategy_class in STRATEGY_REGISTRY.items()}
