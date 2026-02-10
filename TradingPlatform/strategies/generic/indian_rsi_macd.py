"""
Indian RSI + MACD Strategy (legacy parity)

Logic: buy when RSI oversold and MACD above signal; sell when RSI overbought and MACD below signal.

MIGRATED FROM: IndianMarket/strategy_tester_app/app/strategies.py
ORIGINAL: RSIMACDStrategy
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


class IndianRSIMACD(BaseStrategy):
    """RSI oversold/overbought with MACD confirmation (legacy parity)."""

    def __init__(
        self,
        rsi_oversold: int = 35,
        rsi_overbought: int = 65,
        name: str = "IM_RSIMACD",
        version: str = "2.0"
    ):
        super().__init__(name=name, version=version)
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.required_lookback = 1

    def required_indicators(self) -> List[str]:
        return [
            "close",
            "atr_14",
            "rsi_14",
            "macd_12_26_9",
            "macds_12_26_9",
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

        current_rsi = context.current_bar.get("rsi_14")
        current_macd = context.current_bar.get("macd_12_26_9")
        current_macds = context.current_bar.get("macds_12_26_9")
        current_close = context.current_bar.get("close")
        current_atr = context.current_bar.get("atr_14")

        if (
            pd.isna(current_rsi)
            or pd.isna(current_macd)
            or pd.isna(current_macds)
            or pd.isna(current_close)
            or pd.isna(current_atr)
        ):
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning="Invalid indicator values"
            )

        if current_rsi < self.rsi_oversold and current_macd > current_macds:
            return Signal(
                direction="BUY",
                confidence=0.6,
                strategy_name=self.name,
                reasoning="RSI oversold with MACD confirmation",
                additional_context={
                    "rsi_14": current_rsi,
                    "macd_12_26_9": current_macd,
                    "macds_12_26_9": current_macds,
                    "sl": current_close * 0.995,
                    "tp": current_close * 1.01,
                    "tsl": current_close - current_atr
                }
            )

        if current_rsi > self.rsi_overbought and current_macd < current_macds:
            return Signal(
                direction="SELL",
                confidence=0.6,
                strategy_name=self.name,
                reasoning="RSI overbought with MACD confirmation",
                additional_context={
                    "rsi_14": current_rsi,
                    "macd_12_26_9": current_macd,
                    "macds_12_26_9": current_macds,
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
                f"No signal: rsi_14={current_rsi:.2f}, "
                f"macd_12_26_9={current_macd:.2f}, "
                f"macds_12_26_9={current_macds:.2f}"
            )
        )
