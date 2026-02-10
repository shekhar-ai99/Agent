import sys
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

import pandas_ta as ta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('indicator_calculator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IndicatorCalculator:
    """Optimized technical indicator calculator with chaining support and parallel processing"""
    
    DEFAULT_PARAMS = {
        'sma_periods': [20, 50, 200],
        'ema_periods': [9, 14, 21, 50],
        'macd_params': (12, 26, 9),
        'rsi_period': 14,
        'stochastic_period': 14,
        'bollinger_period': 20,
        'bollinger_std': 2,
        'atr_period': 14,
        'dmi_length': 14,
        'adx_smoothing': 14,
        'obv_ema_period': 21,
        'vol_ma_len': 50,
        'vwap_enabled': True,
        'vwma_period': 20,
        'vwap_type': 'cumulative',
        'vol_ma_enabled': True,
        'parallel_workers': max(1, cpu_count() - 1),  # Leave one core free
        'enable_parallel': True  # Can be disabled for chaining
    }

    def __init__(self, params: Optional[Dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._validate_params()
        logger.info(f"Initialized with {self.params['parallel_workers']} parallel workers")

    def _validate_params(self) -> None:
        """Validate all parameters"""
        all_periods = [
            *self.params['sma_periods'], *self.params['ema_periods'],
            self.params['rsi_period'], self.params['stochastic_period'],
            self.params['bollinger_period'], self.params['atr_period'],
            self.params['dmi_length'], self.params['adx_smoothing'],
            self.params['vol_ma_len']
        ]
        if any(p <= 0 for p in all_periods):
            raise ValueError("All periods must be positive integers")
        if self.params['bollinger_std'] <= 0:
            raise ValueError("Bollinger standard deviation must be positive")

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate input dataframe structure and data integrity"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Convert to numeric efficiently
        numeric_cols = df[required_columns].apply(pd.to_numeric, errors='coerce')
        if numeric_cols.isna().any().any():
            raise ValueError("Non-numeric values found in price/volume data")
        df[required_columns] = numeric_cols

        min_period = max([
            *self.params['sma_periods'], *self.params['ema_periods'],
            self.params['rsi_period'], self.params['stochastic_period'],
            self.params['bollinger_period'], self.params['atr_period'],
            self.params['dmi_length'], self.params['adx_smoothing'],
            self.params['vol_ma_len'],
            self.params['macd_params'][1], self.params['macd_params'][2]
        ])
        
        if len(df) < min_period:
            logger.warning(f"Limited data points ({len(df)}) may affect indicator accuracy (min {min_period} recommended)")

        # Vectorized validation
        invalid_high_low = (df['high'] < df['low']).any()
        invalid_close = ((df['close'] > df['high']) | (df['close'] < df['low'])).any()
        invalid_volume = (df['volume'] < 0).any()
        
        if invalid_high_low:
            raise ValueError("High price cannot be less than low price")
        if invalid_close:
            raise ValueError("Close price must be between high and low")
        if invalid_volume:
            raise ValueError("Volume cannot be negative")

    def _parallel_calculate(self, func, df: pd.DataFrame, *args) -> pd.DataFrame:
        """Helper method for parallel calculation"""
        with ThreadPoolExecutor(max_workers=self.params['parallel_workers']) as executor:
            future = executor.submit(func, df.copy(), *args)
            return future.result()

    def calculate_dmi_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate DMI/ADX indicators with chaining support"""
        try:
            dmi_len = self.params['dmi_length']
            adx_smooth = self.params['adx_smoothing']
            
            if not all(col in df for col in ['high', 'low', 'close']):
                raise ValueError("Missing HLC columns for DMI/ADX calculation.")
            
            logger.info(f"Calculating DMI(len={dmi_len})/ADX(smooth={adx_smooth})...")
            
            # Use parallel processing for larger datasets when enabled
            if len(df) > 10000 and self.params['enable_parallel']:
                logger.info("Using parallel processing for DMI/ADX calculation")
                return self._parallel_calculate(self._calculate_dmi_adx_helper, df)
            else:
                return self._calculate_dmi_adx_helper(df)
                
        except Exception as e:
            logger.error(f"Error calculating DMI/ADX: {str(e)}", exc_info=True)
            raise

    def _calculate_dmi_adx_helper(self, df: pd.DataFrame) -> pd.DataFrame:
        """Actual DMI/ADX calculation logic"""
        dmi_len = self.params['dmi_length']
        adx_smooth = self.params['adx_smoothing']
        
        adx_results = ta.adx(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            length=dmi_len,
            adx_length=adx_smooth
        )

        if adx_results is None or adx_results.empty:
            logger.error("ADX calculation returned empty results.")
            df['plus_di'] = df['minus_di'] = df['adx'] = np.nan
            return df

        # Efficient column mapping
        col_mapping = {
            f'DMP_{dmi_len}': 'plus_di',
            f'DMN_{dmi_len}': 'minus_di',
            f'ADX_{adx_smooth}': 'adx'
        }
        
        for src, dest in col_mapping.items():
            if src in adx_results.columns:
                df[dest] = adx_results[src]
            else:
                logger.warning(f"Column {src} not found in ADX results")
                df[dest] = np.nan

        return df

    def calculate_vwma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Weighted Moving Average with chaining support"""
        try:
            period = self.params['vwma_period']
            price_volume = df['close'] * df['volume']
            vol_sum = df['volume'].rolling(window=period, min_periods=1).sum()
            df[f'vwma_{period}'] = price_volume.rolling(window=period, min_periods=1).sum() / vol_sum.replace(0, np.nan)
            return df
        except Exception as e:
            logger.error(f"Error calculating VWMA: {e}")
            raise

    def calculate_volume_ma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Moving Average with chaining support"""
        try:
            period = self.params['vol_ma_len']
            col_name = f'vol_ma_{period}'
            df[col_name] = df['volume'].rolling(period, min_periods=1).mean()
            df['vol_ma'] = df[col_name]  # Optional alias for convenience
            return df
        except Exception as e:
            logger.error(f"Error calculating Volume MA: {e}")
            raise
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all volume-related indicators with chaining support"""
        try:
            # Convert volume to numeric efficiently
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
            
            # OBV calculation
            price_diff = df['close'].diff().fillna(0)
            df['obv'] = (np.sign(price_diff) * df['volume']).cumsum()
            df['obv_ema'] = df['obv'].ewm(
                span=self.params['obv_ema_period'], 
                adjust=False
            ).mean()
            
            # VWAP calculation
            if self.params['vwap_enabled']:
                # 🛡️ Ensure datetime column is present
                if 'datetime' not in df.columns:
                    if 'Date' in df.columns:
                        df['datetime'] = pd.to_datetime(df['Date'])
                    elif 'timestamp' in df.columns:
                        df['datetime'] = pd.to_datetime(df['timestamp'])
                    else:
                        raise ValueError("Column 'datetime' is required for VWAP calculation.")

                
                if self.params['vwap_type'] == 'session':
                    df = self.calculate_session_vwap(df)
                else:
                    # Cumulative VWAP
                    tpv = df['close'] * df['volume']
                    df['vwap'] = tpv.cumsum() / df['volume'].cumsum().replace(0, np.nan)

            return df
        except Exception as e:
            logger.error(f"Volume indicator error: {e}")
            raise

    def calculate_session_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate session-based VWAP with chaining support"""
        try:
            df['date'] = pd.to_datetime(df['datetime']).dt.date
            tpv = df['close'] * df['volume']
            
            # Group by session and calculate cumulative VWAP
            def session_vwap(group):
                vol_cumsum = group['volume'].cumsum()
                return (group['close'] * group['volume']).cumsum() / vol_cumsum.replace(0, np.nan)
            
            df['vwap'] = df.groupby('date', group_keys=False).apply(session_vwap)
            df.drop(columns=['date'], inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Session VWAP calculation error: {e}")
            raise

    def calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all trend indicators with chaining support"""
        try:
            if self.params['enable_parallel'] and len(df) > 5000:
                with ThreadPoolExecutor(max_workers=self.params['parallel_workers']) as executor:
                    futures = []
                    
                    # Submit SMA calculations
                    for period in self.params['sma_periods']:
                        futures.append(executor.submit(
                            lambda p, d: d.assign(**{f'sma_{p}': d['close'].rolling(window=p, min_periods=1).mean()}),
                            period, df.copy()
                        ))
                    
                    # Submit EMA calculations
                    for period in self.params['ema_periods']:
                        futures.append(executor.submit(
                            lambda p, d: d.assign(**{f'ema_{p}': d['close'].ewm(span=p, adjust=False, min_periods=1).mean()}),
                            period, df.copy()
                        ))
                    
                    # Combine results
                    for future in as_completed(futures):
                        result = future.result()
                        for col in result.columns.difference(df.columns):
                            df[col] = result[col]
            else:
                # Sequential calculation for smaller datasets or when parallel is disabled
                for period in self.params['sma_periods']:
                    df[f'sma_{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
                for period in self.params['ema_periods']:
                    df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False, min_periods=1).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {str(e)}")
            raise

    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators with chaining support"""
        try:
            fast, slow, signal = self.params['macd_params']
            
            # Calculate in parallel if dataset is large and parallel enabled
            if len(df) > 5000 and self.params['enable_parallel']:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    ema_fast_future = executor.submit(
                        lambda: df['close'].ewm(span=fast, adjust=False, min_periods=1).mean()
                    )
                    ema_slow_future = executor.submit(
                        lambda: df['close'].ewm(span=slow, adjust=False, min_periods=1).mean()
                    )
                    
                    ema_fast = ema_fast_future.result()
                    ema_slow = ema_slow_future.result()
            else:
                ema_fast = df['close'].ewm(span=fast, adjust=False, min_periods=1).mean()
                ema_slow = df['close'].ewm(span=slow, adjust=False, min_periods=1).mean()
            
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False, min_periods=1).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            return df
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            raise

    def calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator with chaining support"""
        try:
            period = self.params['rsi_period']
            delta = df['close'].diff()
            
            if len(delta) < period:
                logger.warning(f"Insufficient data for RSI ({len(delta)} points)")
                df['rsi'] = np.nan
                return df
                
            # Vectorized RSI calculation
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
            
            rs = avg_gain / avg_loss.replace(0, 1)
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi'].fillna(50).clip(0, 100)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            raise

    def calculate_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator with chaining support"""
        try:
            period = self.params['stochastic_period']
            
            # Vectorized calculation
            low_min = df['low'].rolling(window=period, min_periods=1).min()
            high_max = df['high'].rolling(window=period, min_periods=1).max()
            
            df['stochastic_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min).replace(0, 1))
            df['stochastic_d'] = df['stochastic_k'].rolling(window=3, min_periods=1).mean()
            
            df['stochastic_k'] = df['stochastic_k'].fillna(50)
            df['stochastic_d'] = df['stochastic_d'].fillna(50)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {str(e)}")
            raise
    def _calculate_supertrend(self, df: pd.DataFrame, length=10, multiplier=3.0) -> pd.DataFrame:
        supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=length, multiplier=multiplier)
        df = df.join(supertrend)
        return df

    def _calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])

        if isinstance(ichimoku, tuple):
            # Unpack the tuple and join as DataFrame
            ichimoku_df = pd.concat(ichimoku, axis=1)
        else:
            # Sometimes it returns a single DataFrame directly
            ichimoku_df = ichimoku

        df = df.join(ichimoku_df)
        return df



    def _calculate_mfi(self, df: pd.DataFrame, length=14) -> pd.DataFrame:
        df[f'MFI_{length}'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=length)
        return df

    def _calculate_chaikin_oscillator(self, df: pd.DataFrame, fast=3, slow=10) -> pd.DataFrame:
        high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
        ad = ((2 * close - high - low) / (high - low + 1e-9)) * volume
        ad = ad.fillna(0)
        ad_cumsum = ad.cumsum()
        chaikin = ad_cumsum.ewm(span=fast, adjust=False).mean() - ad_cumsum.ewm(span=slow, adjust=False).mean()
        df['Chaikin_Osc'] = chaikin
        return df

    def _calculate_keltner(self, df: pd.DataFrame, length=20, scalar=1.5) -> pd.DataFrame:
        keltner = ta.kc(df['high'], df['low'], df['close'], length=length, scalar=scalar)
        df = df.join(keltner)
        return df

    def _calculate_donchian(self, df: pd.DataFrame, lower_length=20, upper_length=20) -> pd.DataFrame:
        donchian = ta.donchian(df['high'], df['low'], lower_length=lower_length, upper_length=upper_length)
        df = df.join(donchian)
        return df

    def _calculate_cci(self, df: pd.DataFrame, length=20) -> pd.DataFrame:
        df[f'CCI_{length}'] = ta.cci(df['high'], df['low'], df['close'], length=length)
        return df


    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands with chaining support"""
        try:
            period = self.params['bollinger_period']
            std_dev = self.params['bollinger_std']
            
            # Calculate in parallel if dataset is large and parallel enabled
            if len(df) > 10000 and self.params['enable_parallel']:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    sma_future = executor.submit(
                        lambda: df['close'].rolling(window=period, min_periods=1).mean()
                    )
                    std_future = executor.submit(
                        lambda: df['close'].rolling(window=period, min_periods=1).std()
                    )
                    
                    sma = sma_future.result()
                    rolling_std = std_future.result()
            else:
                sma = df['close'].rolling(window=period, min_periods=1).mean()
                rolling_std = df['close'].rolling(window=period, min_periods=1).std()
            
            df['bollinger_upper'] = sma + (std_dev * rolling_std)
            df['bollinger_lower'] = sma - (std_dev * rolling_std)
            df['bollinger_mid'] = sma
            df['bollinger_bandwidth'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['bollinger_mid']
            df['bollinger_pctb'] = (df['close'] - df['bollinger_lower']) / (
                (df['bollinger_upper'] - df['bollinger_lower']).replace(0, 1))
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            raise

    def calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range with chaining support"""
        try:
            period = self.params['atr_period']
            
            # Vectorized calculation
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.ewm(span=period, adjust=False).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            raise

    def calculate_anchored_vwap(self, df: pd.DataFrame, anchor_index: int) -> pd.Series:
        """Calculate Anchored VWAP from a specific index with chaining support"""
        if not 0 <= anchor_index < len(df):
            raise ValueError("Anchor index out of bounds")

        price_volume = (df['close'] * df['volume']).copy()
        price_volume[:anchor_index] = 0
        volume_cumsum = df['volume'].copy()
        volume_cumsum[:anchor_index] = 0

        return (price_volume.cumsum() / volume_cumsum.cumsum().replace(0, np.nan)).ffill()

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators with chaining support"""
        try:
            self.validate_dataframe(df)
            df_out = df.copy()

            logger.info("Starting indicator calculations...")
            
            # Calculate indicators in sequence to maintain chaining
            df_out = (df_out
                .pipe(self.calculate_trend_indicators)
                .pipe(lambda d: self.calculate_macd(d)
                          .pipe(self.calculate_rsi)
                          .pipe(self.calculate_stochastic))
                .pipe(self.calculate_bollinger_bands)
                .pipe(self.calculate_atr)
                .pipe(self.calculate_volume_indicators)
                .pipe(self.calculate_dmi_adx)
                .pipe(self.calculate_vwma)
                .pipe(self.calculate_volume_ma)
                .pipe(self._calculate_supertrend)
                .pipe(self._calculate_ichimoku)
                .pipe(self._calculate_mfi)
                .pipe(self._calculate_chaikin_oscillator)
                .pipe(self._calculate_keltner)
                .pipe(self._calculate_cci)
                .pipe(self._calculate_donchian)

            )

            # Summary
            original_cols = set(df.columns)
            calculated_cols = set(df_out.columns) - original_cols
            logger.info(f"Successfully calculated {len(calculated_cols)} indicators")
            
            return df_out

        except Exception as e:
            logger.error(f"Overall indicator calculation failed: {str(e)}", exc_info=True)
            raise


def main(input_file: Path, output_file: Path) -> bool:
    """Optimized main function with chaining support"""
    try:
        logger.info(f"Loading data from {input_file}")
        
        # Read CSV with optimized parameters
        df = pd.read_csv(
            input_file,
            parse_dates=['datetime'],
            index_col='datetime',
            dtype={
                'open': 'float32',
                'high': 'float32',
                'low': 'float32',
                'close': 'float32',
                'volume': 'float32'
            },
            engine='c'  # Use C engine for faster parsing
        )
        if df.index.name == 'datetime':
            df.reset_index(inplace=True)
        if 'datetime' not in df.columns:
            if 'Date' in df.columns:
                df['datetime'] = pd.to_datetime(df['Date'])
            elif 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'])
            else:
                raise ValueError(f"Column 'datetime' is required for VWAP calculation, but none of 'datetime', 'Date', or 'timestamp' found. Columns: {df.columns.tolist()}")


        if df.empty:
            logger.error("Input file contains no data")
            return False
            
        logger.info(f"Processing {len(df)} records from {df.index.min()} to {df.index.max()}")
        
        # Initialize calculator and process data
        calculator = IndicatorCalculator()
        df_with_indicators = calculator.calculate_all_indicators(df)
        
        # Save with compression for large files
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if len(df) > 100000:
            df_with_indicators.to_csv(output_file, compression='gzip')
        else:
            df_with_indicators.to_csv(output_file)
            
        logger.info(f"Results saved to {output_file}")
        return True
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        return False
    except pd.errors.EmptyDataError:
        logger.error("Input file is empty or corrupt")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return False

def parse_args() -> argparse.Namespace:
    """Parse command line arguments with validation"""
    parser = argparse.ArgumentParser(
        description='Calculate technical indicators from price data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input', 
        required=True,
        type=Path,
        help='Path to input CSV file with columns: datetime,open,high,low,close,volume'
    )
    parser.add_argument(
        '--output', 
        required=True,
        type=Path,
        help='Path to save the output CSV file with indicators'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=max(1, cpu_count() - 1),
        help='Number of parallel workers to use'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing for chaining operations'
    )
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_args()
        params = {
            'parallel_workers': args.workers,
            'enable_parallel': not args.no_parallel
        }
        success = main(args.input, args.output)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)