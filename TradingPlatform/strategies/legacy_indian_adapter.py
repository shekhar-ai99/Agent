"""
Legacy IndianMarket strategy adapter.

Wraps IndianMarket/strategy_tester_app/app/strategies.py classes so they can
run in the unified BaseStrategy framework without changing legacy logic.
"""

from typing import Any, Dict, List, Optional, Type
import pandas as pd

from TradingPlatform.core.base_strategy import BaseStrategy, Signal, StrategyContext, MarketRegime, VolatilityBucket
from IndianMarket.strategy_tester_app.app import strategies as legacy_strategies


class LegacyIndianAdapter(BaseStrategy):
    """
    Adapter that delegates signal generation to a legacy strategy class.

    Subclasses must set:
    - LEGACY_CLASS: class from IndianMarket.strategy_tester_app.app.strategies
    - DEFAULT_NAME: strategy name
    - REQUIRED_INDICATORS: list of required columns (optional)
    - DESCRIPTION: simple logic description
    """

    LEGACY_CLASS: Optional[Type] = None
    DEFAULT_NAME: str = "LegacyIndianStrategy"
    REQUIRED_INDICATORS: List[str] = []
    DESCRIPTION: str = "Legacy IndianMarket strategy."

    def __init__(self, name: Optional[str] = None, version: str = "1.0", **legacy_params: Any):
        super().__init__(name=name or self.DEFAULT_NAME, version=version)
        if self.LEGACY_CLASS is None:
            raise ValueError("LEGACY_CLASS must be set in subclass")
        self._legacy = self.LEGACY_CLASS(**legacy_params)
        self._legacy_params = legacy_params

    def required_indicators(self) -> List[str]:
        return list(self.REQUIRED_INDICATORS)

    def supports_market(self, market_type: str) -> bool:
        return market_type in ["indian", "crypto", "all"]

    def supports_regime(self, regime: MarketRegime) -> bool:
        return True

    def supports_volatility(self, volatility: VolatilityBucket) -> bool:
        return True

    def generate_signal(self, context: StrategyContext) -> Signal:
        try:
            legacy_df = self._legacy.generate_signals(context.historical_data)
            if legacy_df is None or "signal" not in legacy_df.columns:
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Legacy strategy did not return signal column"
                )

            signal_value = None
            if context.current_bar.name in legacy_df.index:
                signal_value = legacy_df.loc[context.current_bar.name, "signal"]
            else:
                signal_value = legacy_df["signal"].iloc[-1]

            if pd.isna(signal_value):
                signal_value = 0

            if signal_value == 1:
                return Signal(
                    direction="BUY",
                    confidence=0.6,
                    strategy_name=self.name,
                    reasoning=self.DESCRIPTION,
                    additional_context={"legacy_signal": int(signal_value)}
                )
            if signal_value == -1:
                return Signal(
                    direction="SELL",
                    confidence=0.6,
                    strategy_name=self.name,
                    reasoning=self.DESCRIPTION,
                    additional_context={"legacy_signal": int(signal_value)}
                )

            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning="No legacy signal",
                additional_context={"legacy_signal": int(signal_value)}
            )
        except Exception as exc:
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=f"Legacy adapter error: {exc}"
            )


LEGACY = legacy_strategies
