import sys
import pandas as pd

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
npNaN = np.nan

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
    """Enhanced technical indicator calculator with complete implementations"""
    
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
        'vwap_enabled': True
    }
    
    def __init__(self, params: Optional[Dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._validate_params()
    # --- CORRECTED DMI/ADX Calculation Method ---
    def calculate_dmi_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate DMI (+DI, -DI) and ADX using pandas_ta."""
        try:
            dmi_len = self.params['dmi_length']
            adx_smooth = self.params['adx_smoothing']
            if not all(col in df for col in ['high', 'low', 'close']):
                raise ValueError("Missing HLC columns for DMI/ADX calculation.")
            logger.info(f"Calculating DMI(len={dmi_len})/ADX(smooth={adx_smooth}) using pandas-ta...")

            # Correct usage: Call function from library (imported as ta), passing columns
            adx_results = ta.adx(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                length=dmi_len,
                adx_length=adx_smooth # Use adx_length for pandas-ta v0.3.14+
            )

            if adx_results is None or adx_results.empty:
                logger.error("pandas-ta adx calculation returned empty results."); df['plus_di']=np.nan; df['minus_di']=np.nan; df['adx']=np.nan; return df

            # Define expected column names from pandas_ta
            dmp_col=f'DMP_{dmi_len}'; dmn_col=f'DMN_{dmi_len}'; adx_col=f'ADX_{adx_smooth}'
            cols_to_rename = {}; final_cols = ['plus_di', 'minus_di', 'adx'] # Target column names

            # Check and prepare renaming map
            if dmp_col in adx_results.columns: cols_to_rename[dmp_col] = final_cols[0]
            else: logger.warning(f"Column {dmp_col} not found in ta.adx results.")
            if dmn_col in adx_results.columns: cols_to_rename[dmn_col] = final_cols[1]
            else: logger.warning(f"Column {dmn_col} not found in ta.adx results.")
            if adx_col in adx_results.columns: cols_to_rename[adx_col] = final_cols[2]
            else: logger.warning(f"Column {adx_col} not found in ta.adx results.")

            # Check if all target columns can be created
            if len(cols_to_rename) != 3:
                logger.error("Essential DMI/ADX columns missing from pandas-ta results. Assigning NaNs.")
                for col in final_cols: df[col]=np.nan
                return df

            # Rename and select only the needed columns before merging
            adx_results_renamed = adx_results[list(cols_to_rename.keys())].rename(columns=cols_to_rename)
            logger.info(f"Renamed ADX columns to: {final_cols}")

            # Merge results
            df = pd.concat([df, adx_results_renamed], axis=1)
            return df
        except AttributeError: # Catch if 'ta' still doesn't have 'adx'
            logger.error(f"AttributeError calculating DMI/ADX. Is 'import pandas_ta as ta' correct and at the top?", exc_info=True); raise
        except Exception as e: logger.error(f"Error calculating DMI/ADX: {str(e)}", exc_info=True); raise

    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # (Implementation remains the same)
        try:
            # Ensure volume column is numeric, handling potential errors
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
            df['obv'] = (np.sign(df['close'].diff().fillna(0)) * df['volume']).cumsum() # Fill diff NaN
            df['obv_ema'] = df['obv'].ewm(span=self.params['obv_ema_period'], adjust=False).mean()
            if self.params['vwap_enabled']:
                 # Ensure volume sum is not zero before dividing
                 vol_cumsum = df['volume'].cumsum()
                 #df['vwap'] = ((df['close'] * df['volume']).cumsum() / vol_cumsum.replace(0, np.nan)).fillna(method='ffill') # Avoid zero division and forward fill NaNs
                 df['vwap'] = ((df['close'] * df['volume']).cumsum() / vol_cumsum.replace(0, np.nan)).ffill()

            return df
        except Exception as e: logger.error(f"Volume indicator error: {e}"); raise



    def _validate_params(self) -> None:
        """Validate indicator parameters"""
        all_periods = [
            *self.params['sma_periods'], *self.params['ema_periods'],
            self.params['rsi_period'], self.params['stochastic_period'],
            self.params['bollinger_period'], self.params['atr_period'],
            self.params['dmi_length'], self.params['adx_smoothing'],
            self.params['vol_ma_len'] # <--- Added vol_ma_len check
        ]
        
        if any(p <= 0 for p in all_periods):
            raise ValueError("All periods must be positive integers")
        if self.params['bollinger_std'] <= 0:
            raise ValueError("Bollinger standard deviation must be positive")
    
    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate input dataframe structure and data quality"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check for numeric data types
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} must be numeric")
        
        # Check for sufficient data
        min_period = max([ *self.params['sma_periods'], *self.params['ema_periods'],
             self.params['rsi_period'], self.params['stochastic_period'],
             self.params['bollinger_period'], self.params['atr_period'],
             self.params['dmi_length'], self.params['adx_smoothing'],
             self.params['vol_ma_len'], # <--- Added vol_ma_len
             self.params['macd_params'][1], self.params['macd_params'][2]
        ]);
        min_required_rows = min_period + self.params['adx_smoothing'] + 5
        if len(df) < min_required_rows: logger.warning(f"Limited data points ({len(df)}) may affect indicator accuracy (min ~{min_required_rows} recommended)")
        if (df['high'] < df['low']).any(): raise ValueError("High price cannot be less than low price")
        if (df['close'] > df['high']).any() or (df['close'] < df['low']).any(): raise ValueError("Close price must be between high and low")
        if (df['volume'] < 0).any(): raise ValueError("Volume cannot be negative")


    def calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-following indicators"""
        try:
            # Simple Moving Averages
            for period in self.params['sma_periods']:
                df[f'sma_{period}'] = df['close'].rolling(
                    window=period, 
                    min_periods=1
                ).mean()
            
            # Exponential Moving Averages
            for period in self.params['ema_periods']:
                df[f'ema_{period}'] = df['close'].ewm(
                    span=period, 
                    adjust=False, 
                    min_periods=1
                ).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {str(e)}")
            raise

    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD with validation"""
        try:
            fast, slow, signal = self.params['macd_params']
            
            ema_fast = df['close'].ewm(span=fast, adjust=False, min_periods=1).mean()
            ema_slow = df['close'].ewm(span=slow, adjust=False, min_periods=1).mean()
            
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(
                span=signal, 
                adjust=False, 
                min_periods=1
            ).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            return df
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            raise

    def calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI with robustness checks"""
        try:
            period = self.params['rsi_period']
            delta = df['close'].diff()
            
            # Handle edge cases
            if len(delta) < period:
                logger.warning(f"Insufficient data for RSI ({len(delta)} points)")
                df['rsi'] = np.nan
                return df
            
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            
            rs = avg_gain / avg_loss.replace(0, 1)  # Avoid division by zero
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Fill initial NaN values with neutral 50
            df['rsi'] = df['rsi'].fillna(50)
            
            # Clip values to 0-100 range
            df['rsi'] = df['rsi'].clip(0, 100)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            raise

    def calculate_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        try:
            period = self.params['stochastic_period']
            
            low_min = df['low'].rolling(window=period, min_periods=1).min()
            high_max = df['high'].rolling(window=period, min_periods=1).max()
            
            df['stochastic_k'] = 100 * ((df['close'] - low_min) / 
                                      (high_max - low_min).replace(0, 1))  # Avoid division by zero
            
            # Smooth with 3-period SMA for %D
            df['stochastic_d'] = df['stochastic_k'].rolling(window=3, min_periods=1).mean()
            
            # Fill NaN values with 50 (neutral)
            df['stochastic_k'] = df['stochastic_k'].fillna(50)
            df['stochastic_d'] = df['stochastic_d'].fillna(50)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {str(e)}")
            raise

    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        try:
            period = self.params['bollinger_period']
            std_dev = self.params['bollinger_std']
            
            sma = df['close'].rolling(window=period, min_periods=1).mean()
            rolling_std = df['close'].rolling(window=period, min_periods=1).std()
            
            df['bollinger_upper'] = sma + (std_dev * rolling_std)
            df['bollinger_lower'] = sma - (std_dev * rolling_std)
            df['bollinger_mid'] = sma
            
            # Calculate bandwidth and %b
            df['bollinger_bandwidth'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['bollinger_mid']
            df['bollinger_pctb'] = (df['close'] - df['bollinger_lower']) / (
                df['bollinger_upper'] - df['bollinger_lower']).replace(0, 1)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            raise

    def calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range"""
        try:
            period = self.params['atr_period']
            
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(window=period, min_periods=1).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            raise

    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # (Implementation remains the same)
        try:
            # Ensure volume column is numeric, handling potential errors
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
            df['obv'] = (np.sign(df['close'].diff().fillna(0)) * df['volume']).cumsum() # Fill diff NaN
            df['obv_ema'] = df['obv'].ewm(span=self.params['obv_ema_period'], adjust=False).mean()
            if self.params['vwap_enabled']:
                 # Ensure volume sum is not zero before dividing
                 vol_cumsum = df['volume'].cumsum()
                # df['vwap'] = ((df['close'] * df['volume']).cumsum() / vol_cumsum.replace(0, np.nan)).fillna(method='ffill') # Avoid zero division and forward fill NaNs
                 df['vwap'] = ((df['close'] * df['volume']).cumsum() / vol_cumsum.replace(0, np.nan)).ffill()
            

            vol_ma_len = self.params['vol_ma_len']
            df['vol_ma'] = ta.sma(df['volume'], length=vol_ma_len)
            logger.info(f"Calculated Volume MA (len={vol_ma_len})")
            return df
        except Exception as e: logger.error(f"Volume indicator error: {e}"); raise

    # def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """Calculate all technical indicators using pandas_ta where possible"""
    #     try:
    #         self.validate_dataframe(df); df_out = df.copy()
            
    #         logger.info("Calculating indicators using pandas_ta...")

    #         # EMAs (includes 14 now)
    #         for period in self.params['ema_periods']: df_out[f'ema_{period}'] = ta.ema(df_out['close'], length=period)

    #         # MACD
    #         fast, slow, signal_len = self.params['macd_params']; macd_df = ta.macd(df_out['close'], fast=fast, slow=slow, signal=signal_len)
    #         if macd_df is not None: df_out['macd']=macd_df[f'MACD_{fast}_{slow}_{signal_len}']; df_out['macd_signal']=macd_df[f'MACDs_{fast}_{slow}_{signal_len}']; df_out['macd_hist']=macd_df[f'MACDh_{fast}_{slow}_{signal_len}']
    #         # RSI
    #         df_out['rsi'] = ta.rsi(df_out['close'], length=self.params['rsi_period']).fillna(50).clip(0, 100)
    #         # Stochastic
    #         k_period=self.params['stochastic_period']; d_period=3; stoch = ta.stoch(df_out['high'], df_out['low'], df_out['close'], k=k_period, d=d_period)
    #         if stoch is not None: df_out['stochastic_k']=stoch[f'STOCHk_{k_period}_{d_period}_{d_period}'].fillna(50); df_out['stochastic_d']=stoch[f'STOCHd_{k_period}_{d_period}_{d_period}'].fillna(50)
    #         # Bollinger Bands
    #         period=self.params['bollinger_period']; std=self.params['bollinger_std']; bbands = ta.bbands(df_out['close'], length=period, std=std)
    #         if bbands is not None: df_out['bollinger_upper']=bbands[f'BBU_{period}_{std}.0']; df_out['bollinger_lower']=bbands[f'BBL_{period}_{std}.0']; df_out['bollinger_mid']=bbands[f'BBM_{period}_{std}.0']; df_out['bollinger_bandwidth']=bbands[f'BBB_{period}_{std}.0']/100; df_out['bollinger_pctb']=bbands[f'BBP_{period}_{std}.0']
    #         # ATR
    #         df_out['atr'] = ta.atr(df_out['high'], df_out['low'], df_out['close'], length=self.params['atr_period'])
    #         # DMI/ADX
    #         typical_price = (df['high'] + df['low'] + df['close']) / 3
    #         df['vwap'] = ((typical_price * df['volume']).cumsum() / df['volume'].replace(0, np.nan).cumsum()).ffill()


    #         df_out = self.calculate_dmi_adx(df_out)
    #         # Volume Indicators (now includes vol_ma)
    #         df_out = self.calculate_volume_indicators(df_out)

    #         original_cols = set(df.columns); calculated_count = len(set(df_out.columns) - original_cols)
    #         logger.info(f"Successfully calculated {calculated_count} total indicators.")
    #         logger.info("Returning DataFrame with initial NaNs for warm-up periods.")
    #         return df_out
    #     except Exception as e: logger.error(f"Overall indicator calculation failed: {str(e)}", exc_info=True); raise
def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators using pandas_ta where possible."""
    try:
        self.validate_dataframe(df)
        df_out = df.copy()

        logger.info("Calculating indicators using pandas_ta...")

        # EMAs
        for period in self.params['ema_periods']:
            df_out[f'ema_{period}'] = ta.ema(df_out['close'], length=period)

        # MACD
        fast, slow, signal_len = self.params['macd_params']
        macd_df = ta.macd(df_out['close'], fast=fast, slow=slow, signal=signal_len)
        if macd_df is not None:
            df_out['macd'] = macd_df.iloc[:, 0]
            df_out['macd_signal'] = macd_df.iloc[:, 1]
            df_out['macd_hist'] = macd_df.iloc[:, 2]

        # RSI
        df_out['rsi'] = ta.rsi(df_out['close'], length=self.params['rsi_period']).fillna(50).clip(0, 100)

        # Stochastic Oscillator
        k_period = self.params['stochastic_period']
        d_period = 3
        stoch = ta.stoch(df_out['high'], df_out['low'], df_out['close'], k=k_period, d=d_period)
        if stoch is not None:
            df_out['stochastic_k'] = stoch.iloc[:, 0].fillna(50)
            df_out['stochastic_d'] = stoch.iloc[:, 1].fillna(50)

        # Bollinger Bands
        period = self.params['bollinger_period']
        std = self.params['bollinger_std']
        bbands = ta.bbands(df_out['close'], length=period, std=std)
        if bbands is not None:
            df_out['bollinger_upper'] = bbands.iloc[:, 0]
            df_out['bollinger_mid'] = bbands.iloc[:, 1]
            df_out['bollinger_lower'] = bbands.iloc[:, 2]
            df_out['bollinger_bandwidth'] = bbands.iloc[:, 3] / 100
            df_out['bollinger_pctb'] = bbands.iloc[:, 4]

        # ATR
        df_out['atr'] = ta.atr(df_out['high'], df_out['low'], df_out['close'], length=self.params['atr_period'])

        # VWAP (custom)
        typical_price = (df_out['high'] + df_out['low'] + df_out['close']) / 3
        df_out['vwap'] = ((typical_price * df_out['volume']).cumsum() / df_out['volume'].replace(0, np.nan).cumsum()).ffill()

        # DMI/ADX
        df_out = self.calculate_dmi_adx(df_out)

        # Volume Indicators
        df_out = self.calculate_volume_indicators(df_out)

        # Summary
        original_cols = set(df.columns)
        calculated_cols = set(df_out.columns) - original_cols
        logger.info(f"Successfully calculated {len(calculated_cols)} new indicators.")
        logger.info("Returning DataFrame with initial NaNs due to warm-up periods.")

        return df_out

    except Exception as e:
        logger.error(f"Overall indicator calculation failed: {str(e)}", exc_info=True)
        raise


def main(input_file: Path, output_file: Path) -> bool:
    """Enhanced main function with better error handling"""
    try:
            logger.info(f"Loading data from {input_file}")
            df = pd.read_csv(input_file, parse_dates=['datetime'], index_col='datetime', dtype={'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
            if df.empty: logger.error("Input file contains no data"); return False
            logger.info(f"Found {len(df)} records from {df.index.min()} to {df.index.max()}")
            logger.info("Calculating technical indicators")
            calculator = IndicatorCalculator()
            df_with_indicators = calculator.calculate_all_indicators(df)
            logger.info(f"Saving indicators to {output_file}")
            output_file.parent.mkdir(parents=True, exist_ok=True); df_with_indicators.to_csv(output_file)
            logger.info("Indicator calculation completed successfully"); return True
    except FileNotFoundError: logger.error(f"Input file not found: {input_file}"); return False
    except pd.errors.EmptyDataError: logger.error("Input file is empty or corrupt"); return False
    except Exception as e: logger.error(f"Unexpected error in main: {str(e)}", exc_info=True); return False

            
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
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_args()
        success = main(args.input, args.output)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)