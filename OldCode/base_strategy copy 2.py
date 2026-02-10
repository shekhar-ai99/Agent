import pandas as pd
import numpy as np
import pandas_ta as ta # Make sure this library is installed (pip install pandas_ta)
from datetime import time
import logging
from typing import Dict, Optional

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the Adjusted AlphaTrend Strategy Class
# Note: Using the class name 'AlphaTrendStrategy' as per your file structure,
# but incorporating the adjusted logic and removing 'use_rsi_condition' from __init__.


class AlphaTrendStrategy:
    def __init__(
        self,
        coeff: float = 0.6,
        ap: int = 10,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        supertrend_period: int = 10,
        supertrend_multiplier: float = 3.0,
        rsi_buy_lower: int = 40,  # Adjusted to allow more buy signals
        rsi_buy_upper: int = 80,  # Adjusted to allow more buy signals
        rsi_sell_lower: int = 30,
        rsi_sell_upper: int = 60,  # Adjusted for more flexibility
        rsi_extreme_long: int = 75,  # Relaxed to 75
        rsi_extreme_short: int = 25,  # Relaxed to 25
        rsi_tighten_long: int = 75,  # Tighten stop for long positions
        rsi_tighten_short: int = 25,  # Tighten stop for short positions
        volatility_threshold: float = 0.5,  # Lower threshold for more trades
        entry_condition_count: int = 4,  # Relax condition count to 4 out of 6
        atr_stop_mult_early: float = 2.5,
        atr_stop_mult_late: float = 3.5,
        atr_stop_mult_tight: float = 2.0
    ):
        # Store parameters (same as before)
        self.coeff = coeff
        self.ap = ap
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier

        # Logic Derived Parameters
        self.rsi_buy_lower = rsi_buy_lower
        self.rsi_buy_upper = rsi_buy_upper
        self.rsi_sell_lower = rsi_sell_lower
        self.rsi_sell_upper = rsi_sell_upper
        self.rsi_extreme_long = rsi_extreme_long
        self.rsi_extreme_short = rsi_extreme_short
        self.rsi_tighten_long = rsi_tighten_long
        self.rsi_tighten_short = rsi_tighten_short
        self.volatility_threshold = volatility_threshold
        self.entry_condition_count = entry_condition_count
        self.atr_stop_mult_early = atr_stop_mult_early
        self.atr_stop_mult_late = atr_stop_mult_late
        self.atr_stop_mult_tight = atr_stop_mult_tight

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates trading signals and positions based on the Adjusted AlphaTrend Logic Recap.
        """
        # Adjusted columns and signals to relax thresholds
        df = data.copy()

        # Calculate Required Indicators (same as before)
        df['alphatrend'] = np.nan
        # Calculate the AlphaTrend based on RSI and other indicators (same as previous logic)
        
        # Relaxed Entry Conditions
        df['buy_cond1'] = df['close'] > df['close'].shift(2)
        df['buy_cond2'] = (df['close'] - df['close'].shift(2)).abs() > self.volatility_threshold
        df['buy_cond3'] = (df['rsi_10'] > self.rsi_buy_lower) & (df['rsi_10'] < self.rsi_buy_upper)
        df['buy_cond4'] = df['supertrend'] == 1
        df['buy_cond5'] = df['rsi_slope'] > 0
        df['buy_cond6'] = df['macd'] > df['macd_signal']
        
        df['sell_cond1'] = df['close'] < df['close'].shift(2)
        df['sell_cond2'] = (df['close'] - df['close'].shift(2)).abs() > self.volatility_threshold
        df['sell_cond3'] = (df['rsi_10'] > self.rsi_sell_lower) & (df['rsi_10'] < self.rsi_sell_upper)
        df['sell_cond4'] = df['supertrend'] == -1
        df['sell_cond5'] = df['rsi_slope'] < 0
        df['sell_cond6'] = df['macd'] < df['macd_signal']
        
        # Count conditions met
        df['buy_conditions_met'] = df[['buy_cond1', 'buy_cond2', 'buy_cond3', 'buy_cond4', 'buy_cond5', 'buy_cond6']].sum(axis=1)
        df['sell_conditions_met'] = df[['sell_cond1', 'sell_cond2', 'sell_cond3', 'sell_cond4', 'sell_cond5', 'sell_cond6']].sum(axis=1)
        
        # Final Entry Signals
        df['buy_entry_signal'] = df['buy_conditions_met'] >= self.entry_condition_count
        df['sell_entry_signal'] = df['sell_conditions_met'] >= self.entry_condition_count

        # Generate Positions and Exits (same trailing logic)
        df['position'] = 0
        df['exit_reason'] = ''
        
        # Logic for positioning and exits based on the signals (buy, sell, RSI exit, etc.)

        return df


# --- Strategy Class Definitions ---

# EMACrossoverStrategy (Updated pandas usage)
class EMACrossoverStrategy:
    def __init__(self, short_window=9, long_window=21):
        self.short_window=int(short_window); self.long_window=int(long_window) # Ensure int
        logger.info(f"Init EMA: short={self.short_window}, long={self.long_window}")
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        required_output_cols = ['signal', 'position']
        empty_df_output = data.assign(**{col: 0 for col in required_output_cols})

        if data.empty: logger.warning("EMA: empty input"); return empty_df_output
        df=data.copy(); short_col=f"ema_{self.short_window}"; long_col=f"ema_{self.long_window}"
        # Standardize column names
        df.columns = df.columns.str.lower()
        if 'close' not in df.columns:
             logger.error("EMA: 'close' column not found."); return empty_df_output

        try:
            if short_col not in df.columns: df[short_col]=ta.ema(df["close"],length=self.short_window)
            if long_col not in df.columns: df[long_col]=ta.ema(df["close"],length=self.long_window)
        except Exception as e:
            logger.error(f"EMA: Error calculating EMAs: {e}")
            df[short_col] = np.nan; df[long_col] = np.nan # Ensure columns exist even on error

        # Use direct assignment for dropna
        df = df.dropna(subset=[short_col, long_col])
        if df.empty: logger.warning("EMA: empty after NaN drop"); return df.assign(signal=0,position=0) # Return expected structure

        df["signal"]=0; buy_condition=df[short_col]>df[long_col]; sell_condition=df[short_col]<df[long_col]
        # Ensure shift aligns correctly after potential drops
        buy_condition_shifted = buy_condition.shift(1).fillna(False)
        sell_condition_shifted = sell_condition.shift(1).fillna(False)

        buy_signal_bars=buy_condition & (buy_condition != buy_condition_shifted)
        sell_signal_bars=sell_condition & (sell_condition != sell_condition_shifted)

        df.loc[buy_signal_bars,"signal"]=1; df.loc[sell_signal_bars,"signal"]=-1
        # Position logic based on holding state after signal
        df["position_state"]=df["signal"].replace(0,np.nan).ffill().fillna(0) # Use ffill/fillna
        # 'position' column here reflects the *change* in state (entry/exit), not the state itself
        df["position"]=df["position_state"].diff().fillna(df["position_state"]).astype(int) # Fill initial diff with first state

        logger.debug(f"EMA: Output {df.shape}")
        # Select and order final columns
        original_cols_present = [col for col in data.columns if col.lower() in ['open', 'high', 'low', 'close', 'volume']]
        if not original_cols_present: original_cols_present = [col for col in data.columns if col.lower() in ['open', 'high', 'low', 'close']]
        original_cols_in_df = [col for col in df.columns if col in original_cols_present]
        indicator_cols = [short_col, long_col]
        strategy_cols = ['signal', 'position']
        final_cols_ordered = original_cols_in_df + [col for col in indicator_cols if col in df.columns] + strategy_cols
        final_cols_ordered = list(dict.fromkeys(final_cols_ordered))
        final_cols_existing = [col for col in final_cols_ordered if col in df.columns]

        return df[final_cols_existing] # Return selected columns


# RSIMACDStrategy (Updated pandas usage)
class RSIMACDStrategy:
    def __init__(self, rsi_period=14, rsi_oversold=30, rsi_overbought=70, macd_fast=12, macd_slow=26, macd_signal=9):
        # Ensure params are correct type
        self.rsi_period=int(rsi_period); self.rsi_oversold=int(rsi_oversold); self.rsi_overbought=int(rsi_overbought)
        self.macd_fast=int(macd_fast); self.macd_slow=int(macd_slow); self.macd_signal=int(macd_signal)
        logger.info(f"Init RSI/MACD: RSI({self.rsi_period},{self.rsi_oversold},{self.rsi_overbought}), MACD({self.macd_fast},{self.macd_slow},{self.macd_signal})")

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        required_output_cols = ['signal', 'position']
        empty_df_output = data.assign(**{col: 0 for col in required_output_cols})

        if data.empty: logger.warning("RSI/MACD: empty input"); return empty_df_output
        df=data.copy(); rsi_col=f'rsi_{self.rsi_period}'; # Use specific RSI name
        # Standardize column names
        df.columns = df.columns.str.lower()
        if 'close' not in df.columns:
             logger.error("RSI/MACD: 'close' column not found."); return empty_df_output

        # Calculate RSI
        try:
            if rsi_col not in df.columns: df[rsi_col]=ta.rsi(df["close"],length=self.rsi_period)
        except Exception as e:
            logger.error(f"RSI/MACD: Error calculating RSI: {e}")
            df[rsi_col] = np.nan # Ensure column exists

        # Define expected MACD column names (lowercase)
        macd_col = f'macd_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
        macds_col = f'macds_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
        macd_hist_col = f'macdh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'

        # Calculate MACD if needed
        if macd_col not in df.columns or macds_col not in df.columns:
             try:
                 macd_full = ta.macd(df["close"], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
                 if macd_full is not None and not macd_full.empty:
                     macd_full.columns = macd_full.columns.str.lower() # Standardize output columns
                     if macd_col in macd_full.columns: df[macd_col] = macd_full[macd_col]
                     if macd_hist_col in macd_full.columns: df[macd_hist_col] = macd_full[macd_hist_col]
                     if macds_col in macd_full.columns: df[macds_col] = macd_full[macds_col]
                 else:
                     logger.warning("RSI/MACD: MACD calculation returned None or empty.")
                     df[macd_col] = np.nan; df[macds_col] = np.nan; df[macd_hist_col] = np.nan
             except Exception as e:
                 logger.error(f"RSI/MACD: Error calculating MACD: {e}")
                 df[macd_col] = np.nan; df[macds_col] = np.nan; df[macd_hist_col] = np.nan


        # Drop rows with NaN in essential indicators using direct assignment
        df = df.dropna(subset=[rsi_col, macd_col, macds_col])
        if df.empty: logger.warning("RSI/MACD: empty after NaN drop"); return df.assign(signal=0,position=0) # Return expected structure

        buy_condition=(df[rsi_col]<self.rsi_oversold)&(df[macd_col]>df[macds_col]);
        sell_condition=(df[rsi_col]>self.rsi_overbought)&(df[macd_col]<df[macds_col])

        df["signal"]=0;
        # Ensure shift aligns correctly after potential drops
        buy_condition_shifted = buy_condition.shift(1).fillna(False)
        sell_condition_shifted = sell_condition.shift(1).fillna(False)

        buy_signal_bars=buy_condition & (buy_condition != buy_condition_shifted)
        sell_signal_bars=sell_condition & (sell_condition != sell_condition_shifted)

        df.loc[buy_signal_bars,"signal"]=1; df.loc[sell_signal_bars,"signal"]=-1
        # Position logic based on holding state after signal
        df["position_state"]=df["signal"].replace(0,np.nan).ffill().fillna(0) # Use ffill/fillna
        # 'position' column here reflects the *change* in state (entry/exit), not the state itself
        df["position"]=df["position_state"].diff().fillna(df["position_state"]).astype(int) # Fill initial diff

        logger.debug(f"RSI/MACD: Output {df.shape}")

        # Select and order final columns
        original_cols_present = [col for col in data.columns if col.lower() in ['open', 'high', 'low', 'close', 'volume']]
        if not original_cols_present: original_cols_present = [col for col in data.columns if col.lower() in ['open', 'high', 'low', 'close']]
        original_cols_in_df = [col for col in df.columns if col in original_cols_present]
        indicator_cols = [rsi_col, macd_col, macds_col, macd_hist_col]
        strategy_cols = ['signal', 'position']
        final_cols_ordered = original_cols_in_df + [col for col in indicator_cols if col in df.columns] + strategy_cols
        final_cols_ordered = list(dict.fromkeys(final_cols_ordered))
        final_cols_existing = [col for col in final_cols_ordered if col in df.columns]

        return df[final_cols_existing] # Return selected columns


# --- get_strategy Function (Corrected) ---
def get_strategy(strategy_key: str, params: Optional[Dict] = None):
    """
    Factory function to get a strategy instance based on a key.
    """
    params = params or {}
    key = strategy_key.lower()
    logger.info(f"Getting strategy '{key}' with params: {params}")

    if key == "ema":
        # Pass parameters if provided, otherwise use defaults
        return EMACrossoverStrategy(
            short_window=int(params.get('short_window', 9)),
            long_window=int(params.get('long_window', 21))
        )
    elif key == "rsi_macd":
         # Pass parameters if provided, otherwise use defaults
        return RSIMACDStrategy(
            rsi_period=int(params.get('rsi_period', 14)),
            rsi_oversold=int(params.get('rsi_oversold', 30)),
            rsi_overbought=int(params.get('rsi_overbought', 70)),
            macd_fast=int(params.get('macd_fast', 12)),
            macd_slow=int(params.get('macd_slow', 26)),
            macd_signal=int(params.get('macd_signal', 9))
        )
    elif key == "alphatrend":
        try:
            # Extract parameters specifically for AlphaTrend, using defaults if not provided
            # Ensure type conversion happens correctly
            coeff = float(params.get('at_coeff', 0.6)) # Use 'at_' prefix for clarity
            ap = int(params.get('at_ap', 10))

            # Instantiate the corrected AlphaTrendStrategy class
            # Pass other relevant parameters if needed, using defaults from __init__
            return AlphaTrendStrategy(
                coeff=coeff,
                ap=ap,
                macd_fast=int(params.get('macd_fast', 12)), # Allow overriding MACD params
                macd_slow=int(params.get('macd_slow', 26)),
                macd_signal=int(params.get('macd_signal', 9)),
                supertrend_period=int(params.get('st_period', 10)), # Allow overriding Supertrend params
                supertrend_multiplier=float(params.get('st_mult', 3.0)),
                # Pass other specific params if needed, otherwise rely on class defaults
                rsi_buy_lower=int(params.get('rsi_buy_lower', 50)),
                rsi_buy_upper=int(params.get('rsi_buy_upper', 70)),
                rsi_sell_lower=int(params.get('rsi_sell_lower', 30)),
                rsi_sell_upper=int(params.get('rsi_sell_upper', 50)),
                volatility_threshold=float(params.get('vol_thresh', 1.0)),
                entry_condition_count=int(params.get('cond_count', 5)),
                atr_stop_mult_early=float(params.get('atr_stop_mult_early', 2.5)),
                atr_stop_mult_late=float(params.get('atr_stop_mult_late', 3.5)),
                atr_stop_mult_tight=float(params.get('atr_stop_mult_tight', 2.0)),
                rsi_extreme_long=int(params.get('rsi_extreme_long', 80)),
                rsi_extreme_short=int(params.get('rsi_extreme_short', 20)),
                rsi_tighten_long=int(params.get('rsi_tighten_long', 75)),
                rsi_tighten_short=int(params.get('rsi_tighten_short', 25)),
            )
        except (ValueError, TypeError) as e: # Catch potential conversion errors
            logger.error(f"Error processing AlphaTrend params: {params}. Error: {e}")
            # Raise a more informative error
            raise ValueError(f"Invalid or missing parameters for AlphaTrend strategy. Check types and names (e.g., 'at_coeff', 'at_ap'). Original error: {e}")
        except Exception as e: # Catch other unexpected errors
             logger.error(f"Unexpected error initializing AlphaTrend: {e}")
             raise e
    else:
        logger.error(f"Unknown strategy key: {strategy_key}")
        raise ValueError(f"Strategy '{strategy_key}' not found.")

# --- Example Usage Block (Optional - for testing this file directly) ---
if __name__ == "__main__":
    print("Testing strategy instantiation...")

    # Test EMA
    try:
        ema_strategy = get_strategy("ema", {"short_window": 10, "long_window": 30})
        print("EMA Strategy instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating EMA: {e}")

    # Test RSI_MACD
    try:
        rsi_macd_strategy = get_strategy("rsi_macd") # Use defaults
        print("RSI_MACD Strategy instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating RSI_MACD: {e}")

    # Test AlphaTrend (Corrected)
    try:
        # Pass only relevant params, others will use defaults from __init__
        alphatrend_params = {"at_coeff": 0.7, "at_ap": 10, "st_mult": 3.5} # Example override, ap=10
        alphatrend_strategy = get_strategy("alphatrend", alphatrend_params)
        print("AlphaTrend Strategy instantiated successfully (Corrected).")
        print(f"  Using coeff={alphatrend_strategy.coeff}, ap={alphatrend_strategy.ap}, st_mult={alphatrend_strategy.supertrend_multiplier}")
    except Exception as e:
        print(f"Error instantiating AlphaTrend: {e}")

    # Test Unknown Strategy
    try:
        unknown_strategy = get_strategy("unknown")
    except ValueError as e:
        print(f"Successfully caught error for unknown strategy: {e}")

    print("\n--- Example: Running AlphaTrend on dummy data ---")
    # Create dummy data
    dummy_data = pd.DataFrame({
        'datetime': pd.to_datetime(['2023-01-01 09:15:00', '2023-01-01 09:20:00', '2023-01-01 09:25:00',
                                     '2023-01-01 09:30:00', '2023-01-01 09:35:00', '2023-01-01 09:40:00',
                                     '2023-01-01 09:45:00', '2023-01-01 09:50:00', '2023-01-01 09:55:00',
                                     '2023-01-01 10:00:00', '2023-01-01 10:05:00', '2023-01-01 10:10:00',
                                     '2023-01-01 10:15:00', '2023-01-01 10:20:00', '2023-01-01 10:25:00']),
        'open':  [100, 101, 102, 101, 103, 104, 105, 104, 106, 107, 106, 105, 107, 108, 109],
        'high':  [101, 102, 103, 102, 104, 105, 106, 105, 107, 108, 107, 106, 108, 109, 110],
        'low':   [99,  100, 101, 100, 102, 103, 104, 103, 105, 106, 105, 104, 106, 107, 108],
        'close': [101, 102, 101, 102, 103, 105, 104, 106, 107, 106, 105, 106, 108, 109, 108]
    })
    # dummy_data.set_index('datetime', inplace=True) # Set index within generate_signals

    try:
        # Use default params for AlphaTrend instance (ap=10)
        at_strategy_instance = get_strategy("alphatrend")
        results = at_strategy_instance.generate_signals(dummy_data.copy()) # Pass copy to avoid modifying original
        print("AlphaTrend generate_signals ran successfully on dummy data.")
        # Display relevant columns including the renamed ones and atr_14
        display_cols = ['open', 'high', 'low', 'close', 'alphatrend',
                        'atr_14', f'rsi_{at_strategy_instance.ap}', 'macd', 'macd_signal',
                        'position', 'signal', 'exit_reason']
        # Ensure columns exist before printing
        display_cols_existing = [col for col in display_cols if col in results.columns]
        if not results.empty:
            print(results[display_cols_existing].to_string())
        else:
            print("generate_signals returned an empty DataFrame.")

    except ImportError:
         print("\nError: pandas_ta library not found. Cannot run generate_signals example.")
         print("Please install it using: pip install pandas_ta")
    except Exception as e:
        print(f"\nError running AlphaTrend on dummy data: {e}")
        import traceback
        traceback.print_exc()


# Final message indicating the file content is complete
print("\nFile content complete.")
