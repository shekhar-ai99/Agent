"""
Data Router

Abstracts data source (CSV, API, Broker feeds).
Used by backtest, simulation, and real modes.

Ensures all modes get data in same format regardless of source.
"""

import logging
from typing import Dict, Optional
from pathlib import Path
import pandas as pd

from data.dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


class DataRouter:
    """
    Route data requests to appropriate source.
    
    Supports:
    - CSV files (backtest)
    - APIs (simulation/real)
    - Broker feeds (real)
    """
    
    def __init__(self, source: str = "csv", **config):
        """
        Initialize DataRouter.
        
        Args:
            source: "csv", "api", or "broker"
            **config: Source-specific config
        """
        self.source = source.lower()
        self.config = config
        self.logger = logging.getLogger("DataRouter")
    
    def get_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get OHLCV data.
        
        Args:
            symbol: "NIFTY50", "BTC", etc.
            timeframe: "1min", "5min", "15min", "1h", "daily"
            start_date: Optional filter (YYYY-MM-DD)
            end_date: Optional filter (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        
        if self.source == "csv":
            return self._get_from_csv(symbol, timeframe, start_date, end_date)
        
        elif self.source == "api":
            return self._get_from_api(symbol, timeframe, start_date, end_date)
        
        elif self.source == "broker":
            return self._get_from_broker(symbol, timeframe)
        
        else:
            raise ValueError(f"Unknown source: {self.source}")
    
    def _get_from_csv(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load data from CSV files."""
        
        if symbol in ["NIFTY50", "NIFTY", "INDIA"]:
            data = DatasetLoader.load_nifty(timeframe)
        
        elif symbol in ["BTC", "CRYPTO"]:
            # For now, fall back to NIFTY data
            # In Phase 2+, will have Bitcoin CSV files
            self.logger.warning(f"Bitcoin CSV not available, using NIFTY")
            data = DatasetLoader.load_nifty(timeframe)
        
        else:
            raise ValueError(f"Symbol {symbol} not available as CSV")
        
        # Filter by date
        if start_date:
            start = pd.to_datetime(start_date)
            data = data[data.index >= start]
        
        if end_date:
            end = pd.to_datetime(end_date)
            data = data[data.index <= end]
        
        # Add indicators
        data = DatasetLoader.add_indicators(data)
        
        return data
    
    def _get_from_api(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch data from API (Phase 2+).
        
        Supported APIs:
        - Angel One (India)
        - Delta Exchange (Crypto)
        - Kite API (Zero­da­sh)
        """
        
        raise NotImplementedError("API data source not yet implemented (Phase 2+)")
    
    def _get_from_broker(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Get live data from broker (Phase 2+).
        
        Used by simulation and real modes.
        """
        
        raise NotImplementedError("Broker data source not yet implemented (Phase 2+)")
    
    def get_live_tick(self, symbol: str) -> Optional[Dict]:
        """
        Get latest tick for a symbol (for live trading).
        
        Returns:
            Dict with timestamp, open, high, low, close, volume
        """
        
        if self.source == "broker":
            return self._fetch_live_tick_from_broker(symbol)
        else:
            raise ValueError(f"Cannot get live tick from {self.source}")
    
    def _fetch_live_tick_from_broker(self, symbol: str) -> Optional[Dict]:
        """Fetch from broker (Phase 2+)."""
        raise NotImplementedError("Live tick not yet implemented (Phase 2+)")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # CSV-based router
    router = DataRouter(source="csv")
    
    # Load day of data
    data = router.get_data(
        symbol="NIFTY50",
        timeframe="5min",
        start_date="2024-01-15",
        end_date="2024-01-15",
    )
    
    print(f"Loaded {len(data)} rows")
    print(data.head())
