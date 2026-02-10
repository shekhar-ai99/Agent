from TradingPlatform.strategies.base_strategy import (
    BaseStrategy,
    Signal,
    StrategyContext,
    MarketRegime,
    VolatilityBucket,
)


class TripleEMAMomentumStrategy(BaseStrategy):
    def __init__(
        self,
        name="Triple_EMA_Momentum",
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

        self.params.setdefault("ema_fast", 5)
        self.params.setdefault("ema_mid", 13)
        self.params.setdefault("ema_slow", 34)
        self.params.setdefault("rsi_momentum", 50)
        self.params.setdefault("rsi_exit", 40)

    def required_indicators(self):
        return ["ema_5", "ema_13", "ema_34", "rsi_14", "atr_14", "close"]

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
        raise NotImplementedError("Logic to be inserted after approval.")
