"""
Dataset Loader Utility

Load CSV datasets for backtesting with TradingPlatform.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load OHLCV data from CSV files in the datasets folder."""
    
    # Base path to datasets directory
    BASE_PATH = Path(__file__).resolve().parent.parent / "datasets"
    
    @staticmethod
    def load_nifty(timeframe: str = "5min") -> pd.DataFrame:
        """
        Load NIFTY historical data.
        
        Args:
            timeframe: "1min", "5min", "15min", or "daily"
            
        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        file_map = {
            "1min": "nifty/1min/nifty_historical_data_1min.csv",
            "5min": "nifty/5min/nifty_historical_data_5min.csv",
            "15min": "nifty/15min/nifty_historical_data_15min.csv",
            "daily": "nifty/daily/nifty_historical_data_daily.csv",
        }
        
        if timeframe not in file_map:
            raise ValueError(f"Timeframe must be one of: {list(file_map.keys())}")
        
        file_path = DatasetLoader.BASE_PATH / file_map[timeframe]
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        
        logger.info(f"Loading NIFTY {timeframe} data from {file_path}")
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Parse timestamp (try different column names)
        timestamp_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if timestamp_cols:
            df['timestamp'] = pd.to_datetime(df[timestamp_cols[0]])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            logger.warning("No timestamp column found, using index")
        
        # Set index
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        # Ensure OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
        
        logger.info(f"Loaded {len(df)} rows | Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    @staticmethod
    def load_custom(file_path: str) -> pd.DataFrame:
        """
        Load custom CSV file.
        
        Args:
            file_path: Path to CSV file (relative to datasets folder or absolute)
            
        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        # Try relative path first
        relative_path = DatasetLoader.BASE_PATH / file_path
        if relative_path.exists():
            path = relative_path
        else:
            # Try absolute path
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading data from {path}")
        
        df = pd.read_csv(path)
        
        # Try to find and parse timestamp
        timestamp_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if timestamp_cols:
            df['timestamp'] = pd.to_datetime(df[timestamp_cols[0]])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        logger.info(f"Loaded {len(df)} rows")
        return df
    
    @staticmethod
    def load_for_backtest(symbol: str = "NIFTY50", timeframe: str = "5min") -> Dict[str, pd.DataFrame]:
        """
        Load data in format ready for PaperTradingEngine.
        
        Args:
            symbol: "NIFTY50", "BANKNIFTY", etc.
            timeframe: "1min", "5min", "15min", "daily"
            
        Returns:
            Dict with symbol as key, DataFrame as value
        """
        if symbol == "NIFTY50":
            df = DatasetLoader.load_nifty(timeframe)
        else:
            raise NotImplementedError(f"Symbol {symbol} not yet implemented")
        
        return {symbol: df}
    
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators to OHLCV data.
        
        Adds:
        - SMA20, SMA50 (Simple Moving Average)
        - EMA20, EMA50 (Exponential Moving Average) [Preserved from legacy]
        - RSI14 (Relative Strength Index)
        - ATR14 (Average True Range)
        - Bollinger Bands (20-period, 2 std)
        - SuperTrend (10-period, 3.0 multiplier) [Preserved from legacy]
        - ADX14 (Average Directional Index) [Preserved from legacy]
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        df = df.copy()
        
        # Simple Moving Averages
        logger.info("Calculating SMA20, SMA50...")
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages [Preserved from legacy: analysis_engine.py]
        logger.info("Calculating EMA20, EMA50...")
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # RSI (14-period)
        logger.info("Calculating RSI14...")
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR (14-period)
        logger.info("Calculating ATR14...")
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift()),
        ], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # Bollinger Bands (20-period, 2 std)
        logger.info("Calculating Bollinger Bands...")
        bb_middle = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_Middle'] = bb_middle
        df['BB_Upper'] = bb_middle + (bb_std * 2)
        df['BB_Lower'] = bb_middle - (bb_std * 2)
        
        # SuperTrend (10-period, 3.0 multiplier) [Preserved from legacy: Common/strategies.py]
        logger.info("Calculating SuperTrend...")
        df = DatasetLoader._add_supertrend(df, period=10, multiplier=3.0)
        
        # ADX (14-period) [Preserved from legacy: Common/strategies.py]
        logger.info("Calculating ADX...")
        df = DatasetLoader._add_adx(df, period=14)
        
        logger.info("Indicators added successfully")
        return df
    
    @staticmethod
    def _add_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """
        Calculate SuperTrend indicator.
        
        Preserves logic from: Common/strategies.py::create_supertrend_adx_strategy
        """
        hl_avg = (df['high'] + df['low']) / 2
        atr = DatasetLoader._calculate_atr(df, period)
        
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        # Initialize SuperTrend
        supertrend = [0] * len(df)
        direction = [0] * len(df)
        
        for i in range(period, len(df)):
            if i == period:
                supertrend[i] = lower_band.iloc[i]
                direction[i] = 1
            else:
                if supertrend[i-1] == upper_band.iloc[i-1]:
                    supertrend[i] = upper_band.iloc[i]
                    direction[i] = -1
                else:
                    supertrend[i] = lower_band.iloc[i]
                    direction[i] = 1
                
                # Update bands
                if direction[i] == 1:
                    if lower_band.iloc[i] < supertrend[i]:
                        supertrend[i] = lower_band.iloc[i]
                else:
                    if upper_band.iloc[i] > supertrend[i]:
                        supertrend[i] = upper_band.iloc[i]
        
        df[f'SUPERT_{period}'] = supertrend
        df[f'SUPERTD_{period}'] = direction
        
        return df
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range for SuperTrend."""
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift()),
        ], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def _add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate ADX (Average Directional Index).
        
        Preserves logic from: Common/strategies.py::create_supertrend_adx_strategy
        """
        # Calculate Directional Movements
        up_move = df['high'].diff()
        down_move = -df['low'].diff()
        
        pos_dm = [0.0] * len(df)
        neg_dm = [0.0] * len(df)
        
        for i in range(1, len(df)):
            if up_move.iloc[i] > down_move.iloc[i] and up_move.iloc[i] > 0:
                pos_dm[i] = up_move.iloc[i]
            if down_move.iloc[i] > up_move.iloc[i] and down_move.iloc[i] > 0:
                neg_dm[i] = down_move.iloc[i]
        
        atr = DatasetLoader._calculate_atr(df, period)
        
        pos_di = 100 * pd.Series(pos_dm).rolling(period).mean() / (atr + 1e-10)
        neg_di = 100 * pd.Series(neg_dm).rolling(period).mean() / (atr + 1e-10)
        
        di_diff = abs(pos_di - neg_di)
        di_sum = pos_di + neg_di
        
        dx = 100 * di_diff / (di_sum + 1e-10)
        df[f'ADX_{period}'] = dx.rolling(period).mean()
        
        return df
    
    @staticmethod
    def list_available_datasets() -> None:
        """Print all available datasets."""
        print("\n" + "="*70)
        print("AVAILABLE DATASETS")
        print("="*70)
        
        nifty_1min = DatasetLoader.BASE_PATH / "nifty/1min/nifty_historical_data_1min.csv"
        nifty_5min = DatasetLoader.BASE_PATH / "nifty/5min/nifty_historical_data_5min.csv"
        nifty_15min = DatasetLoader.BASE_PATH / "nifty/15min/nifty_historical_data_15min.csv"
        
        datasets = [
            ("NIFTY 1min", nifty_1min),
            ("NIFTY 5min", nifty_5min),
            ("NIFTY 15min", nifty_15min),
        ]
        
        for name, path in datasets:
            if path.exists():
                rows = len(pd.read_csv(path))
                print(f"✓ {name:20s} | {rows:,} rows | {path.relative_to(Path.cwd())}")
            else:
                print(f"✗ {name:20s} | Not found")
        
        print("="*70 + "\n")


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # List available datasets
    DatasetLoader.list_available_datasets()
    
    # Load NIFTY 5min data
    print("Loading NIFTY 5min data...")
    df = DatasetLoader.load_nifty("5min")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    
    # Add indicators
    print("\n" + "="*70)
    print("Adding technical indicators...")
    df = DatasetLoader.add_indicators(df)
    print(f"New shape with indicators: {df.shape}")
    print(f"\nFirst few rows with indicators:\n{df[['close', 'sma20', 'sma50', 'rsi', 'atr']].head()}")
    
    # Prepare for backtesting
    print("\n" + "="*70)
    print("Preparing data for backtesting...")
    data_source = DatasetLoader.load_for_backtest(symbol="NIFTY50", timeframe="5min")
    data_source["NIFTY50"] = DatasetLoader.add_indicators(data_source["NIFTY50"])
    
    from simulation import PaperTradingEngine
    engine = PaperTradingEngine(data_source=data_source)
    print(f"Ready to backtest with {len(data_source)} symbols")
    print(f"NIFTY50 data: {len(data_source['NIFTY50'])} bars")
