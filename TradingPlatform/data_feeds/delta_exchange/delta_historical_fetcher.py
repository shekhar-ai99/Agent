"""
Delta Exchange historical data adapter.

Legacy repo only contains an API client (no OHLC feed found).
"""

import sys
from pathlib import Path
from typing import Optional, Any

from data_feeds.base import BaseDataFeed


def _add_legacy_path(*parts: str) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    legacy_path = repo_root.joinpath(*parts)
    sys.path.append(str(legacy_path))


_add_legacy_path("Bitcoin", "bitcoin_strategy_tester_package", "bitcoin_strategy_tester")

try:
    import api as legacy_delta_api  # type: ignore
except Exception as exc:
    legacy_delta_api = None
    _DELTA_ERR = exc
else:
    _DELTA_ERR = None


class DeltaHistoricalFeed(BaseDataFeed):
    def connect(self) -> None:
        return None

    def fetch_live(self):
        raise NotImplementedError("DeltaHistoricalFeed does not support live streaming")

    def fetch_historical(self, start: Optional[str], end: Optional[str], timeframe: str) -> Any:
        raise NotImplementedError("Delta historical feed not found in legacy code")

    def subscribe(self, symbol: str, timeframe: str) -> None:
        return None

    def stop(self) -> None:
        return None
