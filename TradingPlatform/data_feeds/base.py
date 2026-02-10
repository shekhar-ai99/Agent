"""
Base data feed interface.

Mode-agnostic interface for live, simulation, and backtest data sources.
Do NOT embed broker-specific logic here.
"""

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, Any


class BaseDataFeed(ABC):
    """Abstract base class for all data feeds."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connections or initialize resources."""
        raise NotImplementedError

    @abstractmethod
    def fetch_live(self) -> Iterable[Dict[str, Any]]:
        """
        Stream live ticks or candles.

        Returns:
            Iterable of tick/candle dicts.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_historical(
        self,
        start: Optional[str],
        end: Optional[str],
        timeframe: str
    ) -> Any:
        """
        Fetch historical candles for the given time range.

        Args:
            start: ISO date/time string or None
            end: ISO date/time string or None
            timeframe: timeframe string (e.g., "1min", "5min")
        """
        raise NotImplementedError

    @abstractmethod
    def subscribe(self, symbol: str, timeframe: str) -> None:
        """Subscribe to a symbol/timeframe for live updates."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming and release resources."""
        raise NotImplementedError
