"""
Delta Exchange live feed adapter.

No live feed implementation found in legacy code.
"""

from typing import Any, Iterable, Optional, Dict

from data_feeds.base import BaseDataFeed


class DeltaLiveFeed(BaseDataFeed):
    def connect(self) -> None:
        raise NotImplementedError("Delta Exchange live feed not found in legacy code")

    def fetch_live(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError("Delta Exchange live feed not found in legacy code")

    def fetch_historical(self, start: Optional[str], end: Optional[str], timeframe: str) -> Any:
        raise NotImplementedError("Delta live feed does not support historical fetching")

    def subscribe(self, symbol: str, timeframe: str) -> None:
        raise NotImplementedError("Delta Exchange live feed not found in legacy code")

    def stop(self) -> None:
        return None
