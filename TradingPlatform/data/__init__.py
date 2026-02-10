"""
Data module for TradingPlatform.

Provides:
- DatasetLoader: Load CSV data from datasets folder
- DataFeed: Interface for data providers
"""

import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from .dataset_loader import DatasetLoader

__all__ = ["DatasetLoader", "DataFeed"]


class DataFeed:
    """
    Provides market data (OHLCV) and indicators.
    Should be implemented based on your data source (CSV, APIs, etc.)
    """

    def get_latest_bar(self, symbol: str, market_type: str) -> Optional[pd.Series]:
        """Get latest bar (OHLCV + indicators) for a symbol"""
        pass

    def get_historical_data(self, symbol: str, market_type: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get last N bars of historical data"""
        pass

    def get_current_price(self, symbol: str, market_type: str) -> Optional[float]:
        """Get current price"""
        pass


class IndicatorCalculator:
    """Calculates technical indicators"""

    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        pass

    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> float:
        """Calculate Simple Moving Average"""
        pass

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range"""
        pass

    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0):
        """Calculate Bollinger Bands"""
        pass
