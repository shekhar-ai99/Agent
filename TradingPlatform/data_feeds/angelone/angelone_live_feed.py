"""
Angel One live data feed adapter.

Wraps the migrated LiveDataFeeder (tick aggregation + WebSocket streaming).
"""

import time
import threading
from collections import deque
from typing import Deque, Dict, Iterable, Optional, Any

from data_feeds.base import BaseDataFeed
from .live_data_feeder import LiveDataFeeder
from .tick_aggregator import TickAggregator


class AngelOneLiveFeed(BaseDataFeed):
    """Adapter for Angel One live feed using migrated LiveDataFeeder."""

    def __init__(
        self,
        tokens,
        interval_minutes: int = 5,
        candle_callback=None,
        tick_callback=None,
        mode: int = 1,
        correlation_id: str = "live_correlation_1",
        api_key: Optional[str] = None,
        client_code: Optional[str] = None,
        password: Optional[str] = None,
        totp_secret: Optional[str] = None,
    ):

        self._queue: Deque[Dict[str, Any]] = deque()
        self._running = False

        def _candle_cb(candle):
            self._queue.append(candle)
            if candle_callback:
                candle_callback(candle)

        def _tick_cb(tick):
            if tick_callback:
                tick_callback(tick)

        self._feeder = LiveDataFeeder(
            tokens=tokens,
            interval_minutes=interval_minutes,
            candle_callback=_candle_cb,
            tick_callback=_tick_cb,
            mode=mode,
            correlation_id=correlation_id,
            api_key=api_key,
            client_code=client_code,
            password=password,
            totp_secret=totp_secret,
        )

    def connect(self) -> None:
        """Start the legacy live feed in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._feeder.start, daemon=True)
        self._thread.start()

    def fetch_live(self) -> Iterable[Dict[str, Any]]:
        """Yield live candles as they arrive."""
        self.connect()
        while self._running:
            if self._queue:
                yield self._queue.popleft()
            else:
                time.sleep(0.1)

    def fetch_historical(self, start: Optional[str], end: Optional[str], timeframe: str) -> Any:
        raise NotImplementedError("AngelOneLiveFeed does not support historical fetching")

    def subscribe(self, symbol: str, timeframe: str) -> None:
        """Subscription is handled by legacy LiveDataFeeder via tokens list."""
        # Tokens must be provided at init; keep behavior unchanged.
        return None

    def stop(self) -> None:
        self._running = False
        if self._feeder:
            self._feeder.stop()
