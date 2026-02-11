from TradingPlatform.strategies.base_strategy import (
    BaseStrategy,
    Signal,
    StrategyContext,
    MarketRegime,
    VolatilityBucket,
)


class SuperMomentumStrategy(BaseStrategy):
    def __init__(
        self,
        name="SuperMomentum_Score",
        version="1.0",
        sl_mult=None,
        tp_mult=None,
        tsl_mult=None,
        **params
    ):
        super().__init__(name=name, version=version)
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
        self.tsl_mult = tsl_mult
        self.params = params

        self.params.setdefault("rsi_low", 45)
        self.params.setdefault("rsi_high", 65)
        self.params.setdefault("adx_threshold", 20)

    def required_indicators(self):
        return [
            "rsi_14",
            "macd_12_26_9",
            "macds_12_26_9",
            "supertrend_10_3.0",
            "adx_14",
            "atr_14",
            "close",
        ]

    def supports_market(self, market_type: str) -> bool:
        return market_type in {"india", "crypto"}

    def supports_regime(self, regime: MarketRegime) -> bool:
        return regime in {
            MarketRegime.TRENDING,
            MarketRegime.RANGING,
            MarketRegime.VOLATILE,
            MarketRegime.CHOPPY,
        }

    def supports_volatility(self, volatility: VolatilityBucket) -> bool:
        return volatility in {
            VolatilityBucket.LOW,
            VolatilityBucket.MEDIUM,
            VolatilityBucket.HIGH,
        }

    def generate_signal(self, context: StrategyContext) -> Signal:
        return Signal(
            direction="HOLD",
            confidence=0.0,
            strategy_name=self.name,
            reasoning="Strategy logic not yet implemented"
        )
