"""
Dataset Loader Utility

Load CSV datasets for backtesting with TradingPlatform.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import numpy as np

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
        
        TIER 1 (CORE - Required for all strategies):
        - rsi_14 (Relative Strength Index, 14-period)
        - atr_14 (Average True Range, 14-period)
        - ema_20, ema_50 (Exponential Moving Averages)
        - bb_upper, bb_lower (Bollinger Bands 20, 2σ)
        - supertrend_10_3.0 (SuperTrend 10-period, 3.0 multiplier)
        - adx_14 (Average Directional Index)
        - macd_12_26_9, macds_12_26_9 (MACD)
        
        TIER 2 (EXTENDED):
        - ema_5, ema_9, ema_13, ema_21, ema_34 (EMA variants)
        - stoch_k_14, stoch_d_14 (Stochastic)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        df = df.copy()
        
        # ===== TIER 1: CORE INDICATORS =====
        
        # SMA (Simple Moving Averages)
        logger.info("Calculating SMA20, SMA50...")
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # EMA (Exponential Moving Averages) - TIER 1 & 2
        logger.info("Calculating EMAs (5, 9, 13, 20, 21, 34, 50)...")
        for period in [5, 9, 13, 20, 21, 34, 50]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # RSI (14-period) - RENAMED: RSI → rsi_14
        logger.info("Calculating rsi_14...")
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # RSI with 10-period (for strategies needing rsi_10)
        logger.info("Calculating rsi_10...")
        delta10 = df['close'].diff()
        gain10 = (delta10.where(delta10 > 0, 0)).rolling(window=10).mean()
        loss10 = (-delta10.where(delta10 < 0, 0)).rolling(window=10).mean()
        rs10 = gain10 / loss10
        df['rsi_10'] = 100 - (100 / (1 + rs10))
        
        # ATR (14-period) - RENAMED: ATR → atr_14
        logger.info("Calculating atr_14...")
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift()),
        ], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()
        
        # Bollinger Bands (20-period, 2 std)
        logger.info("Calculating Bollinger Bands...")
        bb_middle = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_middle'] = bb_middle
        df['bb_upper'] = bb_middle + (bb_std * 2)
        df['bb_lower'] = bb_middle - (bb_std * 2)
        
        # SuperTrend (10-period, 3.0 multiplier) - RENAMED: SUPERT_10 → supertrend_10_3.0
        logger.info("Calculating supertrend_10_3.0...")
        df = DatasetLoader._add_supertrend(df, period=10, multiplier=3.0)
        
        # SuperTrend (10-period, 2.0 multiplier) - For strategies with lower multiplier
        logger.info("Calculating supertrend_10_2.0...")
        df = DatasetLoader._add_supertrend(df, period=10, multiplier=2.0)
        
        # ADX (14-period)
        logger.info("Calculating adx_14...")
        df = DatasetLoader._add_adx(df, period=14)
        
        # MACD (12/26/9) - NEW
        logger.info("Calculating MACD (12, 26, 9)...")
        df = DatasetLoader._add_macd(df, fast=12, slow=26, signal=9)
        
        # ===== TIER 2: EXTENDED INDICATORS =====
        
        # Stochastic K & D (14-period)
        logger.info("Calculating Stochastic K & D (14-period)...")
        df = DatasetLoader._add_stochastic(df, period=14)
        
        # Rate of Change (5-period)
        logger.info("Calculating ROC (5-period)...")
        df['roc_5'] = df['close'].pct_change(periods=5) * 100
        
        # VWAP (Volume Weighted Average Price)
        logger.info("Calculating VWAP...")
        df = DatasetLoader._add_vwap(df)
        
        # Volume Average (for volume-based strategies)
        logger.info("Calculating volume_avg...")
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
        
        logger.info(f"Indicators added successfully. DataFrame shape: {df.shape}")
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
        
        # Use snake_case naming: supertrend_10_3.0
        supertrend_col = f'supertrend_{period}_{multiplier}'
        df[supertrend_col] = supertrend
        df[f'{supertrend_col}_direction'] = direction
        
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
        Calculate ADX (Average Directional Index) using pandas_ta.
        
        Uses vectorized operations for efficiency.
        """
        try:
            import pandas_ta as ta
            # Use pandas_ta for ADX calculation
            adx_result = ta.adx(df['high'], df['low'], df['close'], length=period)
            
            if adx_result is not None and not adx_result.empty:
                # pandas_ta returns DataFrame with ADX_{period}, DMP_{period}, DMN_{period} columns
                adx_col = f'ADX_{period}'
                if adx_col in adx_result.columns:
                    df[f'adx_{period}'] = adx_result[adx_col]
                    logger.info(f"ADX calculated successfully using pandas_ta")
                    return df
        except Exception as e:
            logger.warning(f"pandas_ta ADX failed: {e}, falling back to manual calculation")
        
        # Fallback: Manual calculation using pandas vectorized operations
        up_move = df['high'].diff()
        down_move = -df['low'].diff()
        
        # Directional movements (vectorized)
        pos_dm = pd.Series(0.0, index=df.index)
        neg_dm = pd.Series(0.0, index=df.index)
        
        pos_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
        neg_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
        
        # ATR
        atr = DatasetLoader._calculate_atr(df, period)
        
        # Directional indicators (smooth with EMA)
        pos_di = 100 * pos_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10)
        neg_di = 100 * neg_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10)
        
        # DX and ADX
        di_diff = abs(pos_di - neg_di)
        di_sum = pos_di + neg_di
        dx = 100 * di_diff / (di_sum + 1e-10)
        
        # Smooth DX to get ADX
        df[f'adx_{period}'] = dx.ewm(span=period, adjust=False).mean()
        
        return df
    
    @staticmethod
    def _add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Parameters:
            df: DataFrame with OHLCV data
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)
            
        Returns:
            DataFrame with MACD columns added
        """
        # Calculate EMA lines
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        # MACD line = fast EMA - slow EMA
        macd = ema_fast - ema_slow
        
        # Signal line = 9-period EMA of MACD
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        # Histogram = MACD - Signal
        histogram = macd - signal_line
        
        # Add to DataFrame
        df[f'macd_{fast}_{slow}_{signal}'] = macd
        df[f'macds_{fast}_{slow}_{signal}'] = signal_line
        df[f'macdh_{fast}_{slow}_{signal}'] = histogram
        
        return df
    
    @staticmethod
    def _add_stochastic(df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Parameters:
            df: DataFrame with OHLCV data
            period: Period for K calculation (default 14)
            smooth_k: Smoothing period for K line (default 3)
            smooth_d: Smoothing period for D line (default 3)
            
        Returns:
            DataFrame with Stochastic columns added
        """
        # Find the lowest low and highest high over the period
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        
        # Calculate raw K
        k_raw = (df['close'] - low_min) / (high_max - low_min) * 100
        
        # Smooth K
        k_line = k_raw.rolling(window=smooth_k).mean()
        
        # D line = SMA of K
        d_line = k_line.rolling(window=smooth_d).mean()
        
        # Add to DataFrame
        df[f'stoch_k_{period}'] = k_line
        df[f'stoch_d_{period}'] = d_line
        
        return df
    
    @staticmethod
    def _add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP (Volume Weighted Average Price).
        
        VWAP resets daily. For intraday data without proper date grouping,
        we calculate session-based VWAP.
        """
        df = df.copy()
        
        # Check if we have volume data
        if df['volume'].isna().all() or (df['volume'] == 0).all():
            logger.warning("Volume data is missing or zero, using close price for VWAP")
            df['vwap'] = df['close']
            return df
        
        # Typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Try to group by date for proper daily VWAP
        try:
            if hasattr(df.index, 'date'):
                # Group by date
                df['date'] = df.index.date
                df['tp_vol'] = typical_price * df['volume']
                
                # Calculate cumulative sums per date
                df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
                df['cum_vol'] = df.groupby('date')['volume'].cumsum()
                
                # VWAP
                df['vwap'] = df['cum_tp_vol'] / (df['cum_vol'] + 1e-10)
                
                # Clean up temporary columns
                df = df.drop(columns=['date', 'tp_vol', 'cum_tp_vol', 'cum_vol'])
            else:
                # Fallback: cumulative VWAP
                df['tp_vol'] = typical_price * df['volume']
                df['vwap'] = df['tp_vol'].cumsum() / (df['volume'].cumsum() + 1e-10)
                df = df.drop(columns=['tp_vol'])
        except Exception as e:
            logger.warning(f"VWAP date grouping failed: {e}, using cumulative VWAP")
            df['tp_vol'] = typical_price * df['volume']
            df['vwap'] = df['tp_vol'].cumsum() / (df['volume'].cumsum() + 1e-10)
            df = df.drop(columns=['tp_vol'])
        
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
