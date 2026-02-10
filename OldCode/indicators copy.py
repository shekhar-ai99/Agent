import signal
import sys
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
# Removed functools.lru_cache, partial as the direct lru_cache on Series was problematic

# Attempt to import numba for optimized RSI
try:
    import numba
    # Simple check if Numba works on a basic operation
    @numba.jit(nopython=True)
    def _test_numba(arr):
        return arr + 1
    _test_numba(np.array([1, 2])) # Test call
except Exception:
    print("Numba not found or not functional. RSI calculation will use pandas_ta implementation.", file=sys.stderr)
    print("Install Numba for performance improvements: pip install numba", file=sys.stderr)
    numba = None # Set to None if not available

# Attempt to import pandas_ta
try:
    import pandas_ta as ta
except ImportError:
    print("Please install pandas_ta: pip install pandas_ta", file=sys.stderr)
    sys.exit(1)

# Attempt to import pyarrow (for potential future Parquet support, or just keep check)
try:
    import pyarrow
except ImportError:
    print("Warning: pyarrow not found. While this script now saves to CSV, it's needed for Parquet support.", file=sys.stderr)


# Configure logging
logger = logging.getLogger(__name__)
# Only configure if no handlers exist (prevents duplicate logs if script is imported)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # Add handlers only if they don't exist
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
         file_handler = logging.FileHandler('indicator_calculator.log')
         file_handler.setFormatter(formatter)
         logger.addHandler(file_handler)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
         stream_handler = logging.StreamHandler()
         stream_handler.setFormatter(formatter)
         logger.addHandler(stream_handler)


class IndicatorCalculator:
    """
    Optimized technical indicator calculator with vectorization,
    chaining support, and parallel processing capabilities using pandas_ta and Numba.
    Handles CSV input/output and ensures DatetimeIndex for time-series operations.
    LRU caching for MAs is currently disabled due to pandas Series hashability issues.
    """

    DEFAULT_PARAMS = {
        # User requested periods + common ones
        'sma_periods': [10, 20, 50, 200],
        'ema_periods': [9, 14, 21, 50],
        'macd_params': (12, 26, 9),  # fast, slow, signal
        'rsi_period': 14,
        'stochastic_period': 14,
        'stochastic_smooth_k': 3, # Smoothing period for %D
        'bollinger_period': 20,
        'bollinger_std': 2.0,  # Ensure float
        'atr_period': 14,
        'dmi_length': 14,
        'adx_smoothing': 14,
        'obv_ema_period': 21,
        'vol_ma_len': 50,
        'vwap_enabled': True,
        'vwma_period': 20,
        'vwap_type': 'session', # 'cumulative' or 'session' - session requires DatetimeIndex
        'vol_ma_enabled': True,  # Controls separate volume MA calc
        'supertrend_length': 10,
        'supertrend_multiplier': 3.0,
        'ichimoku_tenkan': 9,
        'ichimoku_kijun': 26,
        'ichimoku_senkou': 52, # Displacement period for Senkou Span B/Chikou Span
        'mfi_length': 14,
        'chaikin_fast': 3,
        'chaikin_slow': 10,
        'keltner_length': 20,
        'keltner_scalar': 1.5,
        'keltner_atr_length': 10,  # Typically use separate ATR length for KC
        'donchian_lower_length': 20,
        'donchian_upper_length': 20,
        'cci_length': 20,
        'parallel_workers': max(1, cpu_count() - 1),
        'enable_parallel': True,
        'parallel_threshold': 5000,  # Row count threshold to enable parallelism
        # Removed caching parameters as the feature is temporarily disabled
        # 'enable_caching': True,
        # 'cache_maxsize': 64
    }

    def __init__(self, params: Optional[Dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._validate_params()
        # Removed _setup_caching call as the caching logic is adjusted

        logger.info(
            f"IndicatorCalculator initialized. "
            f"Parallelism: {self.params['enable_parallel']} ({self.params['parallel_workers']} workers, threshold: {self.params['parallel_threshold']}). "
            f"LRU Caching: Disabled (due to Series hashability)."
        )

    # Removed _setup_caching method

    def _validate_params(self) -> None:
        """Validate parameters loaded during initialization."""
        periods_to_check = [
            *self.params.get('sma_periods', []), *self.params.get('ema_periods', []),
            self.params.get('rsi_period'), self.params.get('stochastic_period'),
            self.params.get('stochastic_smooth_k'), self.params.get('bollinger_period'),
            self.params.get('atr_period'), self.params.get('dmi_length'),
            self.params.get('adx_smoothing'), self.params.get('obv_ema_period'),
            self.params.get('vol_ma_len'), self.params.get('vwma_period'),
            self.params.get('supertrend_length'),
            self.params.get('ichimoku_tenkan'), self.params.get('ichimoku_kijun'), self.params.get('ichimoku_senkou'),
            self.params.get('mfi_length'), self.params.get('chaikin_fast'),
            self.params.get('chaikin_slow'), self.params.get('keltner_length'),
            self.params.get('keltner_atr_length'), self.params.get('donchian_lower_length'),
            self.params.get('donchian_upper_length'), self.params.get('cci_length'),
        ]
        macd_params = self.params.get('macd_params')
        if isinstance(macd_params, (tuple, list)) and len(macd_params) == 3:
            periods_to_check.extend(macd_params)

        positive_check_values = [p for p in periods_to_check if isinstance(p, (int, float))]
        if any(p is not None and p <= 0 for p in positive_check_values):
            raise ValueError("All indicator periods/lengths must be positive numbers.")

        if self.params.get('bollinger_std', 0) <= 0: raise ValueError("Bollinger std dev must be positive.")
        if self.params.get('keltner_scalar', 0) <= 0: raise ValueError("Keltner scalar must be positive.")
        if self.params.get('supertrend_multiplier', 0) <= 0: raise ValueError("Supertrend multiplier must be positive.")

        chaikin_fast = self.params.get('chaikin_fast')
        chaikin_slow = self.params.get('chaikin_slow')
        if chaikin_fast is not None and chaikin_slow is not None and (chaikin_fast <= 0 or chaikin_slow <= 0 or chaikin_fast >= chaikin_slow):
             raise ValueError("Chaikin requires fast > 0, slow > 0, and fast < slow.")

        if self.params.get('vwap_type') not in ['cumulative', 'session']:
             raise ValueError("vwap_type must be 'cumulative' or 'session'")

        logger.debug("Parameters validated successfully.")

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Enhanced dataframe validation: checks type, columns, numeric types, OHLCV logic,
        and verifies DatetimeIndex.
        """
        logger.debug(f"Validating DataFrame with shape {df.shape}...")
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # --- DatetimeIndex Validation ---
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex.")

        # --- Required OHLCV Columns ---
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame missing required columns: {missing}")

        # --- Optimized Type Conversion and NaN Check for OHLCV ---
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        try:
            for col in ohlcv_columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 if df[col].isna().any():
                      logger.warning(f"NaN values found in column '{col}' after numeric conversion.")

        except Exception as e:
             logger.error(f"Unexpected error during numeric conversion/validation: {e}")
             raise ValueError(f"Failed to convert columns to numeric: {e}")

        # --- Vectorized Logical Validation ---
        validation_checks = [
            (df['high'] < df['low'], "High < Low"),
            (df['close'] > df['high'], "Close > High"),
            (df['close'] < df['low'], "Close < Low"),
            (df['volume'] < 0, "Negative volume")
        ]

        for check, msg in validation_checks:
            if check.any():
                raise ValueError(f"Invalid data detected: {msg}")

        # --- Check sufficient data length ---
        max_period = max([1, *self.params.get('sma_periods', []), *self.params.get('ema_periods', []),
                          self.params.get('rsi_period', 1), self.params.get('stochastic_period', 1),
                          self.params.get('bollinger_period', 1), self.params.get('atr_period', 1),
                          self.params.get('dmi_length', 1), self.params.get('adx_smoothing', 1),
                          self.params.get('vol_ma_len', 1), self.params.get('vwma_period', 1),
                          self.params.get('supertrend_length', 1),
                          self.params.get('ichimoku_tenkan', 1), self.params.get('ichimoku_kijun', 1), self.params.get('ichimoku_senkou', 1),
                          self.params.get('mfi_length', 1), self.params.get('keltner_length', 1), self.params.get('keltner_atr_length', 1),
                          self.params.get('donchian_lower_length', 1), self.params.get('donchian_upper_length', 1), self.params.get('cci_length', 1),
                         ])
        macd_slow = self.params.get('macd_params', (1,1,1))[1]
        macd_signal = self.params.get('macd_params', (1,1,1))[2]
        max_period = max(max_period, macd_slow + macd_signal) # Typo: was macd_signal, should be signal

        if len(df) < max_period:
            logger.warning(
                f"Input DataFrame has {len(df)} rows, which might be less than the "
                f"minimum data points needed ({max_period}) for some indicator calculations. "
                f"Initial indicator values may be NaN."
            )
        logger.debug("DataFrame validation passed.")


    # --- Numba-Optimized RSI ---
    if numba:
        @staticmethod
        @numba.jit(nopython=True, fastmath=True)
        def _rsi_numba(prices: np.ndarray, period: int) -> np.ndarray:
            deltas = np.diff(prices)
            if period >= len(deltas):
                 return np.full_like(prices, np.nan)

            seed_deltas = deltas[:period]
            up_avg = np.sum(seed_deltas[seed_deltas >= 0]) / period
            down_avg = np.sum(-seed_deltas[seed_deltas < 0]) / period

            rsi = np.full_like(prices, np.nan)
            # Use 1e-9 for safe division
            rs = up_avg / (down_avg + 1e-9)
            rsi[period] = 100.0 - 100.0 / (1.0 + rs)

            for i in range(period, len(prices) - 1):
                delta = deltas[i]
                upval = delta if delta > 0 else 0.0
                downval = -delta if delta < 0 else 0.0

                up_avg = (up_avg * (period - 1) + upval) / period
                down_avg = (down_avg * (period - 1) + downval) / period

                rs = up_avg / (down_avg + 1e-9)
                rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

            return rsi
    else:
         @staticmethod
         def _rsi_numba(prices, period):
              """Placeholder if Numba is not available."""
              # logger.warning("Numba not available, _rsi_numba placeholder called.") # Avoid spamming logs
              return None # Explicitly return None


    # --- MA Calculation Methods (NOT using LRU Cache directly) ---
    @staticmethod
    def _calculate_ema(series: pd.Series, span: int) -> pd.Series:
        """EMA calculation logic."""
        logger.debug(f"Calculating EMA with span={span}")
        span_int = int(span)
        # min_periods=span_int ensures NaN until enough data
        return series.ewm(span=span_int, adjust=False, min_periods=span_int).mean()

    @staticmethod
    def _calculate_sma(series: pd.Series, window: int) -> pd.Series:
        """SMA calculation logic."""
        logger.debug(f"Calculating SMA with window={window}")
        window_int = int(window)
        # min_periods=window_int ensures NaN until enough data
        return series.rolling(window=window_int, min_periods=window_int).mean()


    # --- Indicator Calculation Methods ---

    def calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates SMA and EMA."""
        logger.debug("Calculating trend indicators (SMA, EMA)...")
        try:
            close_prices = df['close']

            # Call the simple static methods directly - no caching issue
            for period in self.params['ema_periods']:
                df[f'ema_{int(period)}'] = self._calculate_ema(close_prices, int(period))
            for period in self.params['sma_periods']:
                df[f'sma_{int(period)}'] = self._calculate_sma(close_prices, int(period))

            return df
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}", exc_info=True)
            for period in self.params['ema_periods']: df[f'ema_{int(period)}'] = np.nan
            for period in self.params['sma_periods']: df[f'sma_{int(period)}'] = np.nan
            raise # Re-raise


    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates MACD indicators."""
        logger.debug("Calculating MACD...")
        try:
            fast, slow, signal = map(int, self.params['macd_params'])
            close_prices = df['close']

            # Call the simple static methods directly for base EMAs
            ema_fast = self._calculate_ema(close_prices, fast)
            ema_slow = self._calculate_ema(close_prices, slow)

            df['macd'] = ema_fast - ema_slow

            # Calculate signal line using EMA on the MACD series
            # Call the simple static method directly for signal EMA
            df['macd_signal'] = self._calculate_ema(df['macd'].fillna(0), signal)
            df['macd_hist'] = df['macd'] - df['macd_signal']

            return df
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}", exc_info=True)
            df['macd'] = df['macd_signal'] = df['macd_hist'] = np.nan
            raise # Re-raise


    def calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates RSI using Numba if available, otherwise pandas_ta."""
        logger.debug("Calculating RSI...")
        try:
            period = int(self.params['rsi_period'])

            if len(df) <= period:
                logger.warning(f"Insufficient data for RSI ({len(df)} <= {period}). Setting RSI to NaN.")
                df['rsi'] = np.nan
                return df

            # Attempt Numba calculation first
            rsi_values = None
            if numba:
                 try:
                     # Pass numpy array to Numba function
                     rsi_values = self._rsi_numba(df['close'].values.astype(np.float64), period)
                     if rsi_values is not None and not np.isnan(rsi_values).all():
                          logger.info("Used Numba-optimized RSI calculation.")
                     else:
                          # If Numba returned all NaNs unexpectedly, fallback
                          rsi_values = None # Force fallback
                 except Exception as numba_e:
                     logger.warning(f"Numba RSI calculation failed: {numba_e}. Falling back to pandas_ta.", exc_info=True)
                     rsi_values = None # Force fallback


            if rsi_values is None:
                # Fallback to pandas_ta if numba is not available or fails
                logger.info("Using pandas_ta for RSI calculation.")
                rsi_series = ta.rsi(close=df['close'], length=period)
                if rsi_series is not None:
                     rsi_values = rsi_series.values # Get numpy array
                else:
                     logger.warning("pandas_ta RSI calculation returned None.")
                     rsi_values = np.full(len(df), np.nan) # Create NaN array

            df['rsi'] = rsi_values # Assign numpy array

            # Final cleaning: fill initial NaNs and clip to 0-100 range.
            df['rsi'] = df['rsi'].fillna(50).clip(0, 100)

            return df
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}", exc_info=True)
            df['rsi'] = np.nan  # Assign NaN on error
            raise # Re-raise

    def calculate_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Stochastic Oscillator (%K and %D) using pandas_ta."""
        logger.debug("Calculating Stochastic Oscillator...")
        try:
            k_period = int(self.params['stochastic_period'])
            d_period = int(self.params['stochastic_smooth_k'])

            if len(df) < max(k_period, d_period):
                 logger.warning(f"Insufficient data for Stochastic ({len(df)}). Need at least {max(k_period, d_period)} points. Setting to NaN.")
                 df['stochastic_k'] = df['stochastic_d'] = np.nan
                 return df

            stoch = ta.stoch(
                high=df['high'], low=df['low'], close=df['close'],
                k=k_period, d=d_period, smooth_k=d_period
            )

            if stoch is not None and not stoch.empty:
                k_col = next((col for col in stoch.columns if col.startswith('STOCHk')), None)
                d_col = next((col for col in stoch.columns if col.startswith('STOCHd')), None)

                if k_col is not None:
                    df['stochastic_k'] = stoch[k_col].fillna(50)
                else:
                    logger.warning(f"Could not find %K column in pandas_ta stoch result for k={k_period}, d={d_period}.")
                    df['stochastic_k'] = np.nan

                if d_col is not None:
                    df['stochastic_d'] = stoch[d_col].fillna(50)
                else:
                     logger.warning(f"Could not find %D column in pandas_ta stoch result for k={k_period}, d={d_period}.")
                     df['stochastic_d'] = np.nan

                df['stochastic_k'] = df['stochastic_k'].clip(0, 100)
                df['stochastic_d'] = df['stochastic_d'].clip(0, 100)

            else:
                logger.warning("Stochastic calculation via pandas_ta returned None or empty.")
                df['stochastic_k'] = np.nan
                df['stochastic_d'] = np.nan

            return df
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}", exc_info=True)
            df['stochastic_k'] = df['stochastic_d'] = np.nan
            raise


    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Bollinger Bands using pandas_ta."""
        logger.debug("Calculating Bollinger Bands...")
        try:
            period = int(self.params['bollinger_period'])
            std_dev = float(self.params['bollinger_std'])

            if len(df) < period:
                 logger.warning(f"Insufficient data for Bollinger Bands ({len(df)} < {period}). Setting to NaN.")
                 df['bollinger_mid'] = df['bollinger_upper'] = df['bollinger_lower'] = np.nan
                 df['bollinger_bandwidth'] = df['bollinger_pctb'] = np.nan
                 return df

            bb = ta.bbands(close=df['close'], length=period, std=std_dev, mamode='sma') # Using SMA

            if bb is not None and not bb.empty:
                 # pandas_ta columns: BBL_..., BBM_..., BBU_..., BBB_..., BBP_...
                 df['bollinger_lower'] = bb.iloc[:, 0]
                 df['bollinger_mid'] = bb.iloc[:, 1]
                 df['bollinger_upper'] = bb.iloc[:, 2]
                 df['bollinger_bandwidth'] = bb.iloc[:, 3]
                 df['bollinger_pctb'] = bb.iloc[:, 4]
            else:
                 logger.warning("Bollinger Bands calculation via pandas_ta returned None or empty.")
                 df['bollinger_mid'] = df['bollinger_upper'] = df['bollinger_lower'] = np.nan
                 df['bollinger_bandwidth'] = df['bollinger_pctb'] = np.nan

            return df
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}", exc_info=True)
            df['bollinger_mid'] = df['bollinger_upper'] = df['bollinger_lower'] = np.nan
            df['bollinger_bandwidth'] = df['bollinger_pctb'] = np.nan
            raise
    def calculate_atr_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Calculating ATR SMA (5-period)...")
        try:
            period = 5
            if 'atr' not in df.columns:
                raise ValueError("Missing 'atr' column for ATR SMA calculation.")
            if len(df) < period:
                logger.warning(f"Insufficient data for ATR SMA ({len(df)} < {period}). Setting to NaN.")
                df['atr_sma_5'] = np.nan
                return df
            df['atr_sma_5'] = self._calculate_sma(df['atr'], period)
            return df
        except Exception as e:
            logger.error(f"Error calculating ATR SMA: {e}", exc_info=True)
            df['atr_sma_5'] = np.nan
            raise

    def calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Average True Range (ATR) using pandas_ta."""
        logger.debug("Calculating ATR...")
        try:
            period = int(self.params['atr_period'])
            if not all(col in df for col in ['high', 'low', 'close']):
                 raise ValueError("Missing HLC columns for ATR calculation.")

            if len(df) < period:
                 logger.warning(f"Insufficient data for ATR ({len(df)} < {period}). Setting to NaN.")
                 df['atr'] = np.nan
                 return df

            atr_series = ta.atr(
                high=df['high'], low=df['low'], close=df['close'],
                length=period
            )
            if atr_series is not None:
                 df['atr'] = atr_series
            else:
                 logger.warning("ATR calculation returned None.")
                 df['atr'] = np.nan
            return df
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}", exc_info=True)
            df['atr'] = np.nan
            raise


    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates volume-based indicators: OBV, OBV_EMA, VWAP, VWMA, Volume MA."""
        logger.debug("Calculating volume indicators (OBV, VWAP, VWMA, Vol MA)...")
        try:
            df_processed = df.copy()

            if not all(col in df_processed.columns for col in ['close', 'volume']):
                raise ValueError("Missing 'close' or 'volume' columns for Volume Indicators.")

            df_processed['volume'] = pd.to_numeric(df_processed['volume'], errors='coerce').fillna(0)

            # --- OBV calculation (using pandas_ta) ---
            obv_series = ta.obv(close=df_processed['close'], volume=df_processed['volume'])
            if obv_series is not None:
                df_processed['obv'] = obv_series
                obv_ema_period = int(self.params['obv_ema_period'])
                df_processed['obv_ema'] = df_processed['obv'].ewm(
                    span=obv_ema_period,
                    adjust=False
                ).mean()
            else:
                logger.warning("OBV calculation returned None.")
                df_processed['obv'] = np.nan
                df_processed['obv_ema'] = np.nan

            # --- VWAP calculation ---
            if self.params['vwap_enabled']:
                if not isinstance(df_processed.index, pd.DatetimeIndex):
                     logger.error("DatetimeIndex required for VWAP calculation.")
                     df_processed['vwap'] = np.nan
                else:
                    if self.params['vwap_type'] == 'session':
                        df_processed = self.calculate_session_vwap(df_processed)
                    else:
                        tpv = df_processed['close'] * df_processed['volume']
                        vol_cumsum = df_processed['volume'].cumsum().replace(0, np.nan)
                        df_processed['vwap'] = tpv.cumsum() / vol_cumsum

            else:
                 df_processed['vwap'] = np.nan

            # --- VWMA calculation (using pandas_ta) ---
            vwma_period = int(self.params['vwma_period'])
            vwma_series = ta.vwma(close=df_processed['close'], volume=df_processed['volume'], length=vwma_period)
            if vwma_series is not None: df_processed[f'vwma_{vwma_period}'] = vwma_series
            else: logger.warning(f"VWMA calculation (period={vwma_period}) returned None."); df_processed[f'vwma_{vwma_period}'] = np.nan

            # --- Volume MA calculation (using pandas_ta SMA on volume) ---
            if self.params.get('vol_ma_enabled', True):
                vol_ma_period = int(self.params['vol_ma_len'])
                col_name_vol_ma = f'vol_ma_{vol_ma_period}'
                if 'volume' not in df_processed.columns: raise ValueError("Missing volume column for Volume MA.")

                vol_ma_series = ta.sma(close=df_processed['volume'], length=vol_ma_period)
                if vol_ma_series is not None:
                     df_processed[col_name_vol_ma] = vol_ma_series
                     df_processed['vol_ma'] = df_processed[col_name_vol_ma]
                else:
                     logger.warning(f"Volume MA calculation (period={vol_ma_period}) returned None.")
                     df_processed['vol_ma'] = np.nan; df_processed[col_name_vol_ma] = np.nan
            else:
                df_processed['vol_ma'] = np.nan
                vol_ma_period = int(self.params.get('vol_ma_len', 50))
                df_processed[f'vol_ma_{vol_ma_period}'] = np.nan


            return df_processed

        except Exception as e:
            logger.error(f"Volume indicator calculation error: {e}", exc_info=True)
            vwma_period = int(self.params.get('vwma_period', 1))
            vol_ma_period = int(self.params.get('vol_ma_len', 1))
            df['obv'] = df['obv_ema'] = df['vwap'] = df[f'vwma_{vwma_period}'] = df['vol_ma'] = df[f'vol_ma_{vol_ma_period}'] = np.nan
            raise


    def calculate_session_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate session-based VWAP using the DatetimeIndex."""
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                 raise TypeError("DataFrame index is not DatetimeIndex, cannot calculate session VWAP.")

            vwap_series = df.groupby(df.index.date, group_keys=False).apply(
                 lambda group: (group['close'] * group['volume']).cumsum() / group['volume'].cumsum().replace(0, np.nan)
            )
            df['vwap'] = vwap_series
            return df
        except Exception as e:
            logger.error(f"Session VWAP calculation error: {e}", exc_info=True)
            df['vwap'] = np.nan
            raise # Re-raise the exception


    # --- Other Parameterized Indicator Methods using pandas_ta ---

    def calculate_dmi_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates DMI (+DI, -DI) and ADX using pandas_ta."""
        logger.debug("Calculating DMI/ADX...")
        try:
            dmi_len = int(self.params['dmi_length'])
            adx_smooth = int(self.params['adx_smoothing'])
            if not all(col in df for col in ['high', 'low', 'close']):
                raise ValueError("Missing HLC for DMI/ADX.")

            if len(df) < max(dmi_len, adx_smooth):
                 logger.warning(f"Insufficient data for DMI/ADX ({len(df)} < {max(dmi_len, adx_smooth)}). Setting to NaN.")
                 df['plus_di'] = df['minus_di'] = df['adx'] = np.nan
                 return df

            adx_results = ta.adx(
                high=df['high'], low=df['low'], close=df['close'],
                length=dmi_len, adx_length=adx_smooth
            )

            if adx_results is None or adx_results.empty:
                logger.error("ADX calculation via pandas_ta returned empty results.")
                df['plus_di'], df['minus_di'], df['adx'] = np.nan, np.nan, np.nan
                return df

            col_mapping = {
                f'DMP_{dmi_len}': 'plus_di', f'DMN_{dmi_len}': 'minus_di', f'ADX_{adx_smooth}': 'adx'
            }
            for src_col, dest_col in col_mapping.items():
                if src_col in adx_results.columns: df[dest_col] = adx_results[src_col]
                else: logger.warning(f"Column '{src_col}' not found in ADX results for len={dmi_len}, smooth={adx_smooth}."); df[dest_col] = np.nan
            return df
        except Exception as e: logger.error(f"Error calculating DMI/ADX: {e}", exc_info=True); raise


    def calculate_vwma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Volume Weighted Moving Average (VWMA) using pandas_ta."""
        # Note: VWMA is also calculated in calculate_volume_indicators.
        # This standalone method is kept for potential individual use or different pipeline chaining.
        # If called in the main pipeline *after* calculate_volume_indicators, it will overwrite the 'vwma_xx' column.
        # The cleaned pipeline removes the duplicate call.
        logger.debug("Calculating VWMA...")
        try:
            period = int(self.params['vwma_period'])
            col_name = f'vwma_{period}'
            if not all(col in df for col in ['close', 'volume']): raise ValueError("Missing close/volume for VWMA.")

            if len(df) < period:
                 logger.warning(f"Insufficient data for VWMA ({len(df)} < {period}). Setting to NaN.")
                 df[col_name] = np.nan
                 return df

            vwma_series = ta.vwma(close=df['close'], volume=df['volume'], length=period)
            if vwma_series is not None: df[col_name] = vwma_series
            else: logger.warning(f"VWMA calculation (period={period}) returned None."); df[col_name] = np.nan
            return df
        except Exception as e: logger.error(f"Error calculating VWMA: {e}", exc_info=True); raise


    def calculate_volume_ma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Simple Moving Average of Volume using pandas_ta."""
        # Note: Volume MA is also calculated in calculate_volume_indicators.
        # This standalone method is kept for potential individual use.
        # The cleaned pipeline removes the duplicate call.
        logger.debug("Calculating Volume MA...")
        try:
            if self.params.get('vol_ma_enabled', True):
                period = int(self.params['vol_ma_len'])
                col_name = f'vol_ma_{period}'
                if 'volume' not in df.columns: raise ValueError("Missing volume column for Volume MA.")

                if len(df) < period:
                     logger.warning(f"Insufficient data for Volume MA ({len(df)} < {period}). Setting to NaN.")
                     df[col_name] = np.nan
                     df['vol_ma'] = np.nan # Alias
                     return df

                vol_ma_series = ta.sma(close=df['volume'], length=period)
                if vol_ma_series is not None:
                     df[col_name] = vol_ma_series
                     df['vol_ma'] = df[col_name]
                else:
                     logger.warning(f"Volume MA calculation (period={period}) returned None.")
                     df['vol_ma'] = np.nan; df[col_name] = np.nan
            else:
                df['vol_ma'] = np.nan
                period = int(self.params.get('vol_ma_len', 50))
                df[f'vol_ma_{period}'] = np.nan
            return df
        except Exception as e: logger.error(f"Error calculating Volume MA: {e}", exc_info=True); raise


    # --- Parameterized Helper methods using pandas_ta for chaining ---

    def _calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Calculating Supertrend...")
        try:
            length = int(self.params['supertrend_length'])
            multiplier = float(self.params['supertrend_multiplier'])
            if not all(col in df for col in ['high', 'low', 'close']): raise ValueError("Missing HLC for Supertrend.")

            # Supertrend requires data length >= length
            if len(df) < length:
                 logger.warning(f"Insufficient data for Supertrend ({len(df)} < {length}). Setting to NaN.")
                 df[f'SUPERT_{length}_{multiplier}'] = np.nan
                 df[f'SUPERTd_{length}_{multiplier}'] = np.nan
                 df[f'SUPERTr_{length}_{multiplier}'] = np.nan
                 return df

            st = ta.supertrend(high=df['high'], low=df['low'], close=df['close'], length=length, multiplier=multiplier)
            if st is not None and not st.empty:
                 df = df.join(st)
            else:
                 logger.warning(f"Supertrend calculation (len={length}, mult={multiplier}) returned None or empty.")
                 df[f'SUPERT_{length}_{multiplier}'] = np.nan
                 df[f'SUPERTd_{length}_{multiplier}'] = np.nan
                 df[f'SUPERTr_{length}_{multiplier}'] = np.nan
            return df
        except Exception as e: logger.error(f"Error in Supertrend: {e}", exc_info=True); raise


    def _calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Calculating Ichimoku...")
        try:
            tenkan, kijun, senkou = map(int, [self.params['ichimoku_tenkan'], self.params['ichimoku_kijun'], self.params['ichimoku_senkou']])
            if not all(col in df for col in ['high', 'low', 'close']): raise ValueError("Missing HLC for Ichimoku.")

            required_len = max(tenkan, kijun, senkou) + senkou # Approx requirement for spans
            if len(df) < required_len:
                 logger.warning(f"Insufficient data for Ichimoku ({len(df)} < approx {required_len}). Setting to NaN.")
                 df['ICH_CS'] = df[f'ICH_KS_{kijun}'] = df[f'ICH_SA_{tenkan},{kijun}'] = df[f'ICH_SB_{senkou}'] = df[f'ICH_TS_{tenkan}'] = np.nan
                 return df


            ichi_tuple = ta.ichimoku(high=df['high'], low=df['low'], close=df['close'], tenkan=tenkan, kijun=kijun, senkou=senkou)

            if ichi_tuple and isinstance(ichi_tuple, tuple) and len(ichi_tuple) > 0:
                df_lines = ichi_tuple[0] # Contains TS, KS, CS (Tenkan, Kijun, Chikou Span)

                if not df_lines.empty:
                    df = df.join(df_lines)
                else:
                    logger.warning(f"Ichimoku lines DataFrame (params={tenkan},{kijun},{senkou}) is empty.")
                    df['ICH_CS'] = df[f'ICH_KS_{kijun}'] = df[f'ICH_SA_{tenkan},{kijun}'] = df[f'ICH_SB_{senkou}'] = df[f'ICH_TS_{tenkan}'] = np.nan
            else:
                 logger.warning(f"Ichimoku calculation (params={tenkan},{kijun},{senkou}) returned None or unexpected result type: {type(ichi_tuple)}.")
                 df['ICH_CS'] = df[f'ICH_KS_{kijun}'] = df[f'ICH_KS_{kijun}'] = df[f'ICH_SA_{tenkan},{kijun}'] = df[f'ICH_SB_{senkou}'] = df[f'ICH_TS_{tenkan}'] = np.nan
            return df
        except Exception as e: logger.error(f"Error in Ichimoku: {e}", exc_info=True); raise

    def _calculate_mfi(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Calculating MFI...")
        try:
            length = int(self.params['mfi_length'])
            col_name = f'MFI_{length}'
            if not all(col in df for col in ['high', 'low', 'close', 'volume']): raise ValueError("Missing HLCV for MFI.")

            if len(df) < length:
                 logger.warning(f"Insufficient data for MFI ({len(df)} < {length}). Setting to NaN.")
                 df[col_name] = np.nan
                 return df

            mfi = ta.mfi(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], length=length)
            if mfi is not None: df[col_name] = mfi.clip(0, 100)
            else: logger.warning(f"MFI calculation (length={length}) returned None."); df[col_name] = np.nan
            return df
        except Exception as e: logger.error(f"Error in MFI: {e}", exc_info=True); raise

    def _calculate_chaikin_oscillator(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Calculating Chaikin Oscillator...")
        try:
            fast, slow = map(int, [self.params['chaikin_fast'], self.params['chaikin_slow']])
            col_name = f'chaikin_osc_{fast}_{slow}'
            if not all(col in df for col in ['high', 'low', 'close', 'volume']): raise ValueError("Missing HLCV for Chaikin.")
            if fast <= 0 or slow <= 0 or fast >= slow: raise ValueError("Invalid Chaikin params: fast > 0, slow > 0, fast < slow.")

            if len(df) < slow:
                 logger.warning(f"Insufficient data for Chaikin Osc ({len(df)} < {slow}). Setting to NaN.")
                 df[col_name] = np.nan
                 return df

            adosc = ta.adosc(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], fast=fast, slow=slow)
            if adosc is not None: df[col_name] = adosc
            else: logger.warning(f"Chaikin Oscillator calculation (fast={fast}, slow={slow}) returned None."); df[col_name] = np.nan
            return df
        except Exception as e: logger.error(f"Error in Chaikin Oscillator: {e}", exc_info=True); raise

    def _calculate_keltner(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Calculating Keltner Channels...")
        try:
            length = int(self.params['keltner_length'])
            scalar = float(self.params['keltner_scalar'])
            atr_len = int(self.params['keltner_atr_length'])
            if not all(col in df for col in ['high', 'low', 'close']): raise ValueError("Missing HLC for Keltner.")

            required_len = max(length, atr_len)
            if len(df) < required_len:
                 logger.warning(f"Insufficient data for Keltner Channels ({len(df)} < {required_len}). Setting to NaN.")
                 df[f'KCBe_{length}_{scalar}'] = df[f'KCBh_{length}_{scalar}'] = df[f'KCBl_{length}_{scalar}'] = np.nan
                 return df

            kc = ta.kc(high=df['high'], low=df['low'], close=df['close'], length=length, scalar=scalar, atr_length=atr_len, mamode='ema')
            if kc is not None and not kc.empty: df = df.join(kc)
            else:
                 logger.warning(f"Keltner calculation (len={length}, scal={scalar}, atr_len={atr_len}) returned None or empty.")
                 df[f'KCBe_{length}_{scalar}'] = df[f'KCBh_{length}_{scalar}'] = df[f'KCBl_{length}_{scalar}'] = np.nan

            return df
        except Exception as e: logger.error(f"Error in Keltner: {e}", exc_info=True); raise

    def _calculate_donchian(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Calculating Donchian Channels...")
        try:
            lower_len = int(self.params['donchian_lower_length'])
            upper_len = int(self.params['donchian_upper_length'])
            if not all(col in df for col in ['high', 'low']): raise ValueError("Missing HL for Donchian.")

            required_len = max(lower_len, upper_len)
            if len(df) < required_len:
                 logger.warning(f"Insufficient data for Donchian Channels ({len(df)} < {required_len}). Setting to NaN.")
                 df[f'DCHe_{lower_len},{upper_len}'] = df[f'DCHl_{lower_len},{upper_len}'] = df[f'DCHm_{lower_len},{upper_len}'] = np.nan
                 return df

            dc = ta.donchian(high=df['high'], low=df['low'], lower_length=lower_len, upper_length=upper_len)
            if dc is not None and not dc.empty: df = df.join(dc)
            else:
                 logger.warning(f"Donchian calculation (lower={lower_len}, upper={upper_len}) returned None or empty.")
                 df[f'DCHe_{lower_len},{upper_len}'] = df[f'DCHl_{lower_len},{upper_len}'] = df[f'DCHm_{lower_len},{upper_len}'] = np.nan
            return df
        except Exception as e: logger.error(f"Error in Donchian: {e}", exc_info=True); raise

    def _calculate_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Calculating CCI...")
        try:
            length = int(self.params['cci_length'])
            col_name = f'CCI_{length}'
            if not all(col in df for col in ['high', 'low', 'close']): raise ValueError("Missing HLC for CCI.")

            if len(df) < length:
                 logger.warning(f"Insufficient data for CCI ({len(df)} < {length}). Setting to NaN.")
                 df[col_name] = np.nan
                 return df

            cci_s = ta.cci(high=df['high'], low=df['low'], close=df['close'], length=length)
            if cci_s is not None: df[col_name] = cci_s
            else: logger.warning(f"CCI calculation (length={length}) returned None."); df[col_name] = np.nan
            return df
        except Exception as e: logger.error(f"Error in CCI: {e}", exc_info=True); raise


    # --- Main Calculation Method ---
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all configured technical indicators using optimized methods.

        Args:
            df (pd.DataFrame): Input DataFrame with a DatetimeIndex and
                               'open', 'high', 'low', 'close', 'volume' columns.

        Returns:
            pd.DataFrame: DataFrame with original data and calculated indicator columns.
        """
        logger.info(f"Starting calculation of all indicators for DataFrame with shape {df.shape}...")
        start_time = pd.Timestamp.now()

        try:
            self.validate_dataframe(df)
            df_out = df.copy()

            # --- Define Pipeline of Calculation Functions ---
            # Calculate indicators in sequence.
            # Using .pipe() for chained operations.
            pipeline = [
                self.calculate_trend_indicators,  # SMA, EMA
                self.calculate_macd,             # Uses EMA
                self.calculate_rsi,              # Numba/pandas_ta
                self.calculate_stochastic,       # pandas_ta
                self.calculate_bollinger_bands,  # pandas_ta
                self.calculate_atr,              # pandas_ta
                # Volume indicators calculated together
                self.calculate_atr_sma,
                self.calculate_volume_indicators, # OBV, OBV_EMA, VWAP (session/cumulative), VWMA, Volume MA
                self.calculate_dmi_adx,          # pandas_ta
                self._calculate_supertrend,      # pandas_ta
                self._calculate_ichimoku,        # pandas_ta
                self._calculate_mfi,             # pandas_ta
                self._calculate_chaikin_oscillator,# pandas_ta
                self._calculate_keltner,         # pandas_ta (uses ATR implicitly)
                self._calculate_cci,             # pandas_ta
                self._calculate_donchian,        # pandas_ta
            ]

            # --- Execute Pipeline ---
            logger.info("Executing indicator calculation pipeline...")
            for func in pipeline:
                logger.debug(f"Running: {func.__name__}")
                try:
                    df_out = func(df_out)
                except Exception as e:
                    logger.error(f"Calculation failed for {func.__name__}: {e}", exc_info=True)
                    # Re-raise to stop the process if any calculation fails after validation
                    raise


            logger.info("Defragmenting final DataFrame...")
            df_out = df_out.copy()

            # --- Summary Logging ---
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            original_cols = set(df.columns)
            calculated_cols = set(df_out.columns) - original_cols
            logger.info(
                f"Successfully calculated {len(calculated_cols)} indicator columns "
                f"in {duration:.2f} seconds. Final DataFrame shape: {df_out.shape}"
            )
            logger.debug(f"Calculated columns: {sorted(list(calculated_cols))}")

            return df_out

        except Exception as e:
            logger.critical(f"Overall indicator calculation failed: {e}", exc_info=True)
            raise


# --- Standalone Script Execution Logic ---

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for standalone execution."""
    parser = argparse.ArgumentParser(
        description='Calculate technical indicators from price data (CSV input/output)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        required=True,
        type=Path,
        help='Path to input CSV file. Must contain a datetime column (named "datetime", "Date", "timestamp", or similar) and open, high, low, close, volume columns.'
    )
    parser.add_argument(
        '--output',
        required=True,
        type=Path,
        help='Path to save the output CSV file with indicators. Automatically gets .csv extension.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=max(1, cpu_count() - 1),
        help='Max parallel workers for indicators that support internal parallelism (e.g., some pandas_ta functions or custom logic).'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable internal parallel processing within indicator calculations.'
    )
     # Removed --no-cache argument
    return parser.parse_args()

def main(input_path: Path, output_path: Path, calc_params: Optional[Dict] = None) -> bool:
    """Main function for standalone script execution."""
    logger.info(f"Initiating standalone indicator calculation: {input_path} -> {output_path}")
    try:
        logger.info(f"Loading data from {input_path}...")

        # --- Load CSV and Handle DatetimeIndex ---
        df = None
        # Try reading with 'datetime' as index
        try:
             df = pd.read_csv(
                 input_path,
                 parse_dates=True,
                 index_col='datetime',
                 dtype={'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float64'},
                 engine='c', memory_map=True
             )
             logger.info("Loaded CSV assuming 'datetime' column as index.")
        except KeyError:
             # If 'datetime' not found as index_col, try loading without it
             logger.warning("'datetime' not found as index_col, attempting to load without and find date column.")
             df = pd.read_csv(
                 input_path,
                 parse_dates=True,
                 dtype={'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float64'},
                 engine='c', memory_map=True
             )

             # Now, find a suitable datetime column and set it as index
             datetime_col_found = False
             possible_dt_cols = ['datetime', 'Date', 'timestamp', 'Timestamp', 'time']
             for col in possible_dt_cols:
                 if col in df.columns:
                     logger.info(f"Using column '{col}' as datetime source and setting as index.")
                     if not pd.api.types.is_datetime64_any_dtype(df[col]):
                         try: df[col] = pd.to_datetime(df[col])
                         except Exception as e: raise ValueError(f"Failed to convert column '{col}' to datetime: {e}")
                     df.set_index(col, inplace=True)
                     datetime_col_found = True
                     break

             if not datetime_col_found:
                  # If no suitable column found, check if the *current* index is datetime-like after loading
                 if isinstance(df.index, pd.DatetimeIndex):
                      logger.info("Using loaded DataFrame index as datetime source.")
                 else:
                      # Last resort: try converting the current index
                      try:
                           df.index = pd.to_datetime(df.index)
                           if isinstance(df.index, pd.DatetimeIndex): logger.info("Converted existing index to DatetimeIndex.")
                           else: raise TypeError("Index could not be converted to DatetimeIndex after loading.")
                      except Exception as e: raise ValueError(f"Could not identify or convert any column/index to DatetimeIndex: {e}")

        # Ensure the index is a DatetimeIndex after loading and setting
        if not isinstance(df.index, pd.DatetimeIndex):
             raise TypeError("DataFrame index is not a DatetimeIndex after load and conversion attempts.")

        # Handle potential duplicates in index
        if not df.index.is_unique:
             logger.warning("Duplicate timestamps found in index. Aggregating duplicates (using mean for OHLC, sum for volume).")
             agg_logic = {col: 'mean' for col in ['open', 'high', 'low', 'close'] if col in df.columns}
             if 'volume' in df.columns: agg_logic['volume'] = 'sum'
             other_cols = df.columns.difference(['open', 'high', 'low', 'close', 'volume'])
             for col in other_cols: agg_logic[col] = 'first'
             df = df.groupby(df.index).agg(agg_logic)
             df.sort_index(inplace=True)

        # Localize naive DatetimeIndex to Asia/Kolkata (assuming this is the source timezone)
        if df.index.tz is None:
            logger.info("Localizing naive DatetimeIndex to Asia/Kolkata")
            try:
                df.index = df.index.tz_localize('Asia/Kolkata', ambiguous='infer')
            except Exception as e:
                 logger.warning(f"Could not localize DatetimeIndex to Asia/Kolkata: {e}. Proceeding with naive index.")

        if df.empty:
            logger.error("Input file contains no data after loading")
            return False

        logger.info(f"Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")

        # Initialize calculator with params from args/defaults
        indicator_params = {
             'parallel_workers': calc_params.get('parallel_workers', IndicatorCalculator.DEFAULT_PARAMS['parallel_workers']),
             'enable_parallel': calc_params.get('enable_parallel', IndicatorCalculator.DEFAULT_PARAMS['enable_parallel']),
             # 'enable_caching' is now internally disabled, no need to pass from args
        }
        calculator = IndicatorCalculator(params=indicator_params)

        # Calculate indicators
        df_with_indicators = calculator.calculate_all_indicators(df)

        # Efficient saving to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_csv_path = output_path.with_suffix('.csv')

        compression = 'gzip' if len(df_with_indicators) > 100000 else None
        logger.info(f"Saving results to {output_csv_path} with compression: {compression}")
        # Ensure the index is saved as a column named 'datetime' in the CSV
        df_with_indicators.to_csv(output_csv_path, compression=compression, index=True, index_label='datetime')

        logger.info(f"Indicator results saved successfully.")
        return True

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        return False
    except pd.errors.EmptyDataError:
        logger.error(f"Input file is empty or corrupt: {input_path}")
        return False
    except ImportError as e:
        logger.error(f"Missing required library: {e}. Please install requirements.")
        return False
    except Exception as e:
        logger.critical(f"Error in main execution: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    root_logger = logging.getLogger()
    if not root_logger.handlers:
         root_logger.setLevel(logging.INFO)
         formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
         file_handler = logging.FileHandler('indicator_calculator.log')
         file_handler.setFormatter(formatter)
         root_logger.addHandler(file_handler)
         stream_handler = logging.StreamHandler()
         stream_handler.setFormatter(formatter)
         root_logger.addHandler(stream_handler)

    try:
        args = parse_args()
        run_params = {
            'parallel_workers': args.workers,
            'enable_parallel': not args.no_parallel,
        }
        success = main(args.input, args.output, calc_params=run_params)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.critical(f"Script failed before/during main execution: {e}", exc_info=True)
        sys.exit(1)