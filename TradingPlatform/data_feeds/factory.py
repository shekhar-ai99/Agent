"""
DataFeedFactory

Centralized routing for data feed instances.
No scattered if/else across codebase.
"""

from typing import Optional

from data_feeds.angelone.angelone_live_feed import AngelOneLiveFeed
from data_feeds.angelone.angelone_historical_fetcher import AngelOneHistoricalFeed
from data_feeds.coindcx.coindcx_live_feed import CoinDCXLiveFeed
from data_feeds.coindcx.coindcx_historical_fetcher import CoinDCXHistoricalFeed
from data_feeds.delta_exchange.delta_live_feed import DeltaLiveFeed
from data_feeds.delta_exchange.delta_historical_fetcher import DeltaHistoricalFeed


class DataFeedFactory:
    @staticmethod
    def create(market: str, broker: str, mode: str, **kwargs):
        market = market.lower()
        broker = broker.lower()
        mode = mode.lower()

        if broker == "angelone":
            if mode in ["live", "simulation"]:
                return AngelOneLiveFeed(**kwargs)
            if mode in ["backtest", "historical"]:
                source = kwargs.pop("source", "data_fetcher")
                return AngelOneHistoricalFeed(source=source)

        if broker == "coindcx":
            if mode in ["live", "simulation"]:
                return CoinDCXLiveFeed()
            if mode in ["backtest", "historical"]:
                return CoinDCXHistoricalFeed()

        if broker == "delta":
            if mode in ["live", "simulation"]:
                return DeltaLiveFeed()
            if mode in ["backtest", "historical"]:
                return DeltaHistoricalFeed()

        raise ValueError(f"Unsupported feed request: market={market}, broker={broker}, mode={mode}")
