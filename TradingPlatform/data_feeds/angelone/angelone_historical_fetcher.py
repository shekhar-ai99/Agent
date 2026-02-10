"""
Angel One historical data adapters.

Wraps legacy historical fetchers without modifying their internal logic.
"""

import sys
from pathlib import Path
from typing import Optional, Any

from data_feeds.base import BaseDataFeed


def _add_legacy_path(*parts: str) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    legacy_path = repo_root.joinpath(*parts)
    sys.path.append(str(legacy_path))


# Legacy modules
_add_legacy_path("IndianMarket", "BOT", "trading_bot_final", "data")
_add_legacy_path("IndianMarket", "strategy_tester_app", "app")
_add_legacy_path("IndianMarket", "strategy_tester_app", "angel_one")

try:
    import data_fetcher as legacy_data_fetcher  # type: ignore
except Exception as exc:
    legacy_data_fetcher = None
    _DATA_FETCHER_ERR = exc
else:
    _DATA_FETCHER_ERR = None

try:
    import fetcher_v2 as legacy_fetcher_v2  # type: ignore
except Exception as exc:
    legacy_fetcher_v2 = None
    _FETCHER_V2_ERR = exc
else:
    _FETCHER_V2_ERR = None

try:
    import historical_data as legacy_hist_app  # type: ignore
except Exception as exc:
    legacy_hist_app = None
    _HIST_APP_ERR = exc
else:
    _HIST_APP_ERR = None

try:
    import angel_data_fetcher as legacy_angel_data_fetcher  # type: ignore
except Exception as exc:
    legacy_angel_data_fetcher = None
    _ANGEL_DATA_ERR = exc
else:
    _ANGEL_DATA_ERR = None


class AngelOneHistoricalFeed(BaseDataFeed):
    """Historical feed adapter that delegates to selected legacy source."""

    def __init__(self, source: str = "data_fetcher"):
        self.source = source

    def connect(self) -> None:
        return None

    def fetch_live(self):
        raise NotImplementedError("AngelOneHistoricalFeed does not support live streaming")

    def fetch_historical(self, start: Optional[str], end: Optional[str], timeframe: str) -> Any:
        if self.source == "data_fetcher":
            if legacy_data_fetcher is None:
                raise ImportError(f"legacy data_fetcher not available: {_DATA_FETCHER_ERR}")
            return legacy_data_fetcher.fetch_data(
                source="online",
                interval=timeframe,
                start_date=start,
                end_date=end,
            )

        if self.source == "fetcher_v2":
            if legacy_fetcher_v2 is None:
                raise ImportError(f"legacy fetcher_v2 not available: {_FETCHER_V2_ERR}")
            fetcher = legacy_fetcher_v2.HistoricalDataFetcher(interval=timeframe)
            return fetcher.fetch_data(source="online", interval=timeframe)

        if self.source == "historical_data_app":
            if legacy_hist_app is None:
                raise ImportError(f"legacy historical_data not available: {_HIST_APP_ERR}")
            fetcher = legacy_hist_app.HistoricalDataFetcher(interval=timeframe)
            smart_api = fetcher.initialize_api()
            return fetcher.fetch_historical_data(smart_api, days=30)

        if self.source == "angel_data_fetcher":
            if legacy_angel_data_fetcher is None:
                raise ImportError(f"legacy angel_data_fetcher not available: {_ANGEL_DATA_ERR}")
            fetcher = legacy_angel_data_fetcher.AngelDataFetcher(config={})
            return fetcher.fetch_historical_candles(symbol="NIFTY", timeframe=timeframe)

        raise ValueError(f"Unknown AngelOne historical source: {self.source}")

    def subscribe(self, symbol: str, timeframe: str) -> None:
        return None

    def stop(self) -> None:
        return None
