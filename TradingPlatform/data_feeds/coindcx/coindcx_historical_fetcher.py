"""
CoinDCX historical data adapter.

Wraps legacy getCoinDCX.py without modifying its internal logic.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
import pandas as pd

from data_feeds.base import BaseDataFeed


def _add_legacy_path(*parts: str) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    legacy_path = repo_root.joinpath(*parts)
    sys.path.append(str(legacy_path))


_add_legacy_path("IndianMarket", "strategy_tester_app", "app")

try:
    import getCoinDCX as legacy_coindcx  # type: ignore
except Exception as exc:
    legacy_coindcx = None
    _COINDCX_ERR = exc
else:
    _COINDCX_ERR = None


class CoinDCXHistoricalFeed(BaseDataFeed):
    def connect(self) -> None:
        return None

    def fetch_live(self):
        raise NotImplementedError("CoinDCXHistoricalFeed does not support live streaming")

    def fetch_historical(self, start: Optional[str], end: Optional[str], timeframe: str) -> Any:
        if legacy_coindcx is None:
            raise ImportError(f"legacy getCoinDCX not available: {_COINDCX_ERR}")

        start_dt = datetime.fromisoformat(start) if start else None
        end_dt = datetime.fromisoformat(end) if end else None
        if not start_dt or not end_dt:
            raise ValueError("start and end are required for CoinDCX historical fetch")

        data = legacy_coindcx.get_candlestick_data(
            pair="B-BTC_USDT",
            interval=timeframe,
            start_time=start_dt,
            end_time=end_dt,
        )

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df

    def subscribe(self, symbol: str, timeframe: str) -> None:
        return None

    def stop(self) -> None:
        return None
