import pandas as pd
import numpy as np
import pandas_ta as ta # Still needed by the strategy class
from datetime import time, datetime
import logging
from io import StringIO # To read string data
from typing import Optional, List, Dict, Any # Added for type hinting

# --- Configure Logging ---
# Configure basic logging to show info level messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Define Modified AlphaTrendStrategy Class ---
class AlphaTrendStrategy:
    """
    Implements the AlphaTrend trading strategy with ADJUSTED parameters
    to potentially generate more signals.

    Key Logic Points:
    - AlphaTrend Formula: 0.6 * close + 0.3 * prev_AlphaTrend + 0.1 * RSI
    - Entry Conditions: 4 out of 6 specific conditions must be met (Relaxed).
    - Trailing Stops: Time and RSI dependent ATR-based stops.
    - Morning Rule: Fixed Long entry at 9:15, Exit at 9:20.
    - Other Exits: Opposite signal, RSI extremes (Relaxed), Session End (15:25).
    """
    def __init__(
        self,
        coeff: float = 0.6,
        ap: int = 10,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        supertrend_period: int = 10,
        supertrend_multiplier: float = 3.0,
        # --- Adjusted Parameters ---
        rsi_buy_lower: int = 40,  # Relaxed: Was 50
        rsi_buy_upper: int = 80,  # Relaxed: Was 70
        rsi_sell_lower: int = 30, # Kept same
        rsi_sell_upper: int = 60,  # Relaxed: Was 50
        rsi_extreme_long: int = 75,  # Relaxed: Was 80
        rsi_extreme_short: int = 25,  # Relaxed: Was 20
        rsi_tighten_long: int = 75,  # Kept same
        rsi_tighten_short: int = 25,  # Kept same
        volatility_threshold: float = 0.5,  # Relaxed: Was 1.0
        entry_condition_count: int = 4,  # Relaxed: Was 5
        # --- Stop Parameters (Unchanged) ---
        atr_stop_mult_early: float = 2.5,
        atr_stop_mult_late: float = 3.5,
        atr_stop_mult_tight: float = 2.0
    ):
        """
        Initializes the strategy with potentially adjusted parameters.
        """
        # Store parameters
        self.coeff = coeff
        self.ap = ap
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier

        # Store logic-derived parameters (using the potentially adjusted values)
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

        # Log initialization parameters
        logger.info(
            f"Init MODIFIED AlphaTrend: coeff={coeff}, ap={ap}, "
            f"macd=({macd_fast},{macd_slow},{macd_signal}), "
            f"supertrend=({supertrend_period},{supertrend_multiplier}), "
            f"RSI ranges Buy=({rsi_buy_lower}-{rsi_buy_upper}), Sell=({rsi_sell_lower}-{rsi_sell_upper}), " # Adjusted
            f"Volatility Threshold={volatility_threshold}, Conditions Needed={entry_condition_count}, " # Adjusted
            f"Stop Multipliers Early={atr_stop_mult_early}, Late={atr_stop_mult_late}, Tight={atr_stop_mult_tight}, "
            f"RSI Extremes=({rsi_extreme_short}/{rsi_extreme_long}), RSI Tighten=({rsi_tighten_short}/{rsi_tighten_long})" # Adjusted
        )

    def _calculate_true_range(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Helper to calculate True Range."""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)
        return true_range

    # --- generate_signals Method (Integrating Adjusted Parameters) ---
    def generate_signals(self, data: pd.DataFrame, prev_close: Optional[float] = None) -> pd.DataFrame:
        """
        Generates trading signals and positions based on the Adjusted AlphaTrend Logic Recap,
        using the potentially modified parameters from __init__.
        Adds indicator columns needed for backtesting.
        """
        required_output_cols = ['signal', 'position', 'exit_reason', 'buy_conditions_met', 'sell_conditions_met'] # Add condition counts
        empty_df_output = pd.DataFrame(columns=list(data.columns) + required_output_cols)
        for col in required_output_cols:
             if col == 'exit_reason': empty_df_output[col] = empty_df_output[col].astype(str)
             else: empty_df_output[col] = empty_df_output[col].astype(float) # Use float for counts

        if data.empty:
            logger.warning("Adjusted AlphaTrend: empty input dataframe.")
            return empty_df_output

        df = data.copy()

        # --- 1. Ensure DatetimeIndex and Timezone ---
        if not isinstance(df.index, pd.DatetimeIndex):
             if 'datetime' in df.columns:
                 try:
                     df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                     df = df.dropna(subset=['datetime'])
                     if df.empty: return empty_df_output
                     df = df.set_index('datetime')
                 except Exception as e: logger.error(f"Failed parse 'datetime': {e}"); return empty_df_output
             elif 'Date' in df.columns and 'Time' in df.columns:
                 try:
                     datetime_col = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
                     df['temp_datetime'] = datetime_col
                     df = df.dropna(subset=['temp_datetime'])
                     if df.empty: return empty_df_output
                     df.index = df['temp_datetime']
                     df = df.drop(columns=['Date', 'Time', 'temp_datetime'])
                 except Exception as e: logger.error(f"Failed combine Date/Time: {e}"); return empty_df_output
             else: logger.error("Requires DatetimeIndex or 'datetime' col."); return empty_df_output

        if not df.index.is_monotonic_increasing:
            logger.info("Sorting data by index."); df = df.sort_index()

        target_tz = 'Asia/Kolkata'
        if df.index.tz is None:
             try: df.index = df.index.tz_localize(target_tz, ambiguous='infer')
             except Exception as e: logger.warning(f"TZ localize failed: {e}")
        elif str(df.index.tz) != target_tz:
            try: df.index = df.index.tz_convert(target_tz)
            except Exception as e: logger.warning(f"TZ convert failed: {e}")

        # --- 2. Calculate Required Indicators ---
        required_cols = ['high', 'low', 'close']
        df.columns = df.columns.str.lower()
        if not all(col in df.columns for col in required_cols):
             logger.error(f"Missing OHLC columns: {required_cols}"); return empty_df_output

        min_data_length = max(self.ap, self.macd_slow, self.supertrend_period, 14) + 5
        if len(df) < min_data_length: logger.warning(f"Short data ({len(df)}<{min_data_length})")

        # ATR (Strategy AP and Sim 14)
        atr_strategy_col = f'atr_sma_{self.ap}'
        atr_sim_col = 'atr_14'; atr_sim_period = 14
        try:
            df['tr'] = self._calculate_true_range(df['high'], df['low'], df['close'])
            df['tr'] = pd.to_numeric(df['tr'], errors='coerce')
            df[atr_strategy_col] = ta.sma(df['tr'], length=self.ap); df[atr_strategy_col] = df[atr_strategy_col].bfill()
            df[atr_sim_col] = ta.sma(df['tr'], length=atr_sim_period); df[atr_sim_col] = df[atr_sim_col].bfill()
            df = df.drop(columns=['tr'], errors='ignore')
        except Exception as e: logger.error(f"ATR Error: {e}"); df[atr_strategy_col]=np.nan; df[atr_sim_col]=np.nan

        # RSI and Slope (Strategy AP)
        rsi_col = f'rsi_{self.ap}'
        try:
            df[rsi_col] = ta.rsi(df['close'], length=self.ap)
            df['rsi_slope'] = df[rsi_col].diff()
            df[rsi_col] = df[rsi_col].bfill(); df['rsi_slope'] = df['rsi_slope'].fillna(0)
        except Exception as e: logger.error(f"RSI Error: {e}"); df[rsi_col]=np.nan; df['rsi_slope']=np.nan

        # MACD
        macd_internal_col = f'macd_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
        macds_internal_col = f'macds_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
        macd_hist_internal_col = f'macdh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
        macd_final_col = 'macd'; macds_final_col = 'macd_signal'
        try:
            macd = ta.macd(df['close'], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            if macd is not None and not macd.empty:
                macd.columns = macd.columns.str.lower()
                macd_col_ta = f'macd_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
                macds_col_ta = f'macds_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
                macd_hist_col_ta = f'macdh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
                if macd_col_ta in macd.columns: df[macd_internal_col] = macd[macd_col_ta]
                if macds_col_ta in macd.columns: df[macds_internal_col] = macd[macds_col_ta]
                if macd_hist_col_ta in macd.columns: df[macd_hist_internal_col] = macd[macd_hist_col_ta]
                if macd_internal_col in df.columns: df[macd_internal_col] = df[macd_internal_col].bfill()
                if macds_internal_col in df.columns: df[macds_internal_col] = df[macds_internal_col].bfill()
                if macd_hist_internal_col in df.columns: df[macd_hist_internal_col] = df[macd_hist_internal_col].fillna(0)
            else: df[macd_internal_col]=np.nan; df[macds_internal_col]=np.nan; df[macd_hist_internal_col]=np.nan; logger.warning("MACD calc failed.")
        except Exception as e: logger.error(f"MACD Error: {e}"); df[macd_internal_col]=np.nan; df[macds_internal_col]=np.nan; df[macd_hist_internal_col]=np.nan

        # Supertrend
        expected_st_dir_col = f'SUPERTd_{self.supertrend_period}_{self.supertrend_multiplier}'
        supertrend_calculated = False
        try:
            st = ta.supertrend(high=df['high'], low=df['low'], close=df['close'], length=self.supertrend_period, multiplier=self.supertrend_multiplier)
            if st is not None and not st.empty:
                actual_st_dir_col = next((col for col in st.columns if col.lower() == expected_st_dir_col.lower()), None)
                if actual_st_dir_col:
                     df[expected_st_dir_col] = st[actual_st_dir_col]
                     df[expected_st_dir_col] = df[expected_st_dir_col].ffill().fillna(0)
                     logger.info(f"Supertrend direction found: {expected_st_dir_col}")
                     supertrend_calculated = True
                else: logger.warning(f"Expected ST col '{expected_st_dir_col}' not found. Cols: {st.columns.tolist()}. Defaulting ST to 0."); df[expected_st_dir_col] = 0
            else: logger.warning("ST calc failed. Defaulting ST to 0."); df[expected_st_dir_col] = 0
        except Exception as e: logger.error(f"ST Error: {e}"); df[expected_st_dir_col] = 0

        # AlphaTrend Calculation
        df['alphatrend'] = np.nan
        first_valid_rsi_index = df[rsi_col].first_valid_index() if rsi_col in df.columns else None
        if first_valid_rsi_index is not None:
            first_valid_loc = df.index.get_loc(first_valid_rsi_index)
            if 'close' in df.columns and rsi_col in df.columns:
                close_val_at_first = df.iloc[first_valid_loc]['close']
                if pd.notna(close_val_at_first): df.iloc[first_valid_loc, df.columns.get_loc('alphatrend')] = close_val_at_first
                if 'alphatrend' in df.columns:
                    alpha_trend_col_idx = df.columns.get_loc('alphatrend'); close_col_idx = df.columns.get_loc('close'); rsi_col_idx = df.columns.get_loc(rsi_col)
                    for i in range(first_valid_loc + 1, len(df)):
                        prev_at = df.iloc[i-1, alpha_trend_col_idx]; current_close = df.iloc[i, close_col_idx]; current_rsi = df.iloc[i, rsi_col_idx]
                        if pd.notna(prev_at) and pd.notna(current_close) and pd.notna(current_rsi):
                             df.iloc[i, alpha_trend_col_idx] = (self.coeff * current_close + 0.3 * prev_at + 0.1 * current_rsi)
                        else: df.iloc[i, alpha_trend_col_idx] = prev_at
            if 'alphatrend' in df.columns: df['alphatrend'] = df['alphatrend'].ffill()
        else: logger.warning("No valid RSI for AlphaTrend init.")

        # --- 4. Calculate Entry Conditions (Using Adjusted Parameters) ---
        required_cond_cols_internal = [rsi_col, 'rsi_slope', macd_internal_col, macds_internal_col, expected_st_dir_col]
        if not all(col in df.columns and pd.api.types.is_numeric_dtype(df[col]) for col in required_cond_cols_internal):
            missing = [col for col in required_cond_cols_internal if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col])]
            logger.error(f"Entry condition indicators missing/non-numeric: {missing}. Cannot generate signals."); return empty_df_output

        df['close_shift2'] = df['close'].shift(2); df['high_shift1'] = df['high'].shift(1); df['low_shift1'] = df['low'].shift(1)

        # Buy Conditions (Uses parameters from __init__, e.g., self.rsi_buy_lower)
        df['buy_cond1'] = df['close'] > df['close_shift2']
        df['buy_cond2'] = (df['close'] - df['close_shift2']).abs() > self.volatility_threshold # Adjusted
        df['buy_cond3'] = (df[rsi_col] > self.rsi_buy_lower) & (df[rsi_col] < self.rsi_buy_upper) # Adjusted
        df['buy_cond4'] = df[expected_st_dir_col] == 1
        df['buy_cond5'] = df['rsi_slope'] > 0
        df['buy_cond6'] = df[macd_internal_col] > df[macds_internal_col]

        # Sell Conditions (Uses parameters from __init__, e.g., self.rsi_sell_upper)
        df['sell_cond1'] = df['close'] < df['close_shift2']
        df['sell_cond2'] = (df['close'] - df['close_shift2']).abs() > self.volatility_threshold # Adjusted
        df['sell_cond3'] = (df[rsi_col] > self.rsi_sell_lower) & (df[rsi_col] < self.rsi_sell_upper) # Adjusted
        df['sell_cond4'] = df[expected_st_dir_col] == -1
        df['sell_cond5'] = df['rsi_slope'] < 0
        df['sell_cond6'] = df[macd_internal_col] < df[macds_internal_col]

        # Count conditions met
        buy_cond_cols = ['buy_cond1', 'buy_cond2', 'buy_cond3', 'buy_cond4', 'buy_cond5', 'buy_cond6']
        sell_cond_cols = ['sell_cond1', 'sell_cond2', 'sell_cond3', 'sell_cond4', 'sell_cond5', 'sell_cond6']
        df['buy_conditions_met'] = df[buy_cond_cols].fillna(False).sum(axis=1)
        df['sell_conditions_met'] = df[sell_cond_cols].fillna(False).sum(axis=1)

        # Final Entry Signals (Uses self.entry_condition_count - Adjusted)
        df['buy_entry_signal'] = df['buy_conditions_met'] >= self.entry_condition_count
        df['sell_entry_signal'] = df['sell_conditions_met'] >= self.entry_condition_count

        # --- 5. Generate Position based on Signals, Stops, and Exits ---
        df['position'] = 0; df['exit_reason'] = ''
        position_active = 0; trailing_stop_price = np.nan; morning_trade_active = False
        session_end_time = time(15, 25, 0); time_0915 = time(9, 15, 0); time_0920 = time(9, 20, 0); time_0930 = time(9, 30, 0)

        col_indices = {}; required_loop_cols = {'low': 'low', 'high': 'high', 'close': 'close', rsi_col: rsi_col, atr_strategy_col: atr_strategy_col,'high_shift1': 'high_shift1', 'low_shift1': 'low_shift1','buy_entry_signal': 'buy_entry_signal', 'sell_entry_signal': 'sell_entry_signal','position': 'position', 'exit_reason': 'exit_reason'}
        if not all(col in df.columns and (col_indices.update({name: df.columns.get_loc(col)}) or True) for name, col in required_loop_cols.items()):
            logger.error(f"Missing loop columns: {[col for name, col in required_loop_cols.items() if col not in df.columns]}"); return empty_df_output

        for i, idx in enumerate(df.index):
            current_time = idx.time()
            try: current_low = df.iloc[i, col_indices['low']]; current_high = df.iloc[i, col_indices['high']]; current_close = df.iloc[i, col_indices['close']]; current_rsi = df.iloc[i, col_indices[rsi_col]]; current_atr = df.iloc[i, col_indices[atr_strategy_col]]; prev_high = df.iloc[i, col_indices['high_shift1']]; prev_low = df.iloc[i, col_indices['low_shift1']]; is_buy = df.iloc[i, col_indices['buy_entry_signal']]; is_sell = df.iloc[i, col_indices['sell_entry_signal']]
            except IndexError: logger.error(f"IndexError at {idx}");
            if i > 0: df.iloc[i, col_indices['position']] = df.iloc[i-1, col_indices['position']]; continue
            if pd.isna(current_low) or pd.isna(current_high) or pd.isna(current_close) or pd.isna(current_rsi) or pd.isna(current_atr):
                if i > 0: df.iloc[i, col_indices['position']] = df.iloc[i-1, col_indices['position']]; continue

            exit_triggered = False; exit_reason_str = ''
            if position_active == 1:
                if pd.notna(trailing_stop_price) and current_low <= trailing_stop_price: exit_reason_str = 'Stop Loss'; exit_triggered = True
                elif morning_trade_active and current_time == time_0920: exit_reason_str = 'Morning Exit Time'; exit_triggered = True
                elif not morning_trade_active and is_sell: exit_reason_str = 'Opposite Signal (Sell)'; exit_triggered = True
                elif current_rsi > self.rsi_extreme_long: exit_reason_str = f'RSI Extreme (> {self.rsi_extreme_long})'; exit_triggered = True # Adjusted threshold
                elif current_rsi < self.rsi_extreme_short: exit_reason_str = f'RSI Extreme (< {self.rsi_extreme_short})'; exit_triggered = True # Adjusted threshold
                elif current_time >= session_end_time: exit_reason_str = 'Session End'; exit_triggered = True
            elif position_active == -1: # Basic short exit logic
                 if pd.notna(trailing_stop_price) and current_high >= trailing_stop_price: exit_reason_str = 'Stop Loss'; exit_triggered = True
                 elif not morning_trade_active and is_buy: exit_reason_str = 'Opposite Signal (Buy)'; exit_triggered = True
                 elif current_rsi < self.rsi_extreme_short: exit_reason_str = f'RSI Extreme (< {self.rsi_extreme_short})'; exit_triggered = True
                 elif current_rsi > self.rsi_extreme_long: exit_reason_str = f'RSI Extreme (> {self.rsi_extreme_long})'; exit_triggered = True
                 elif current_time >= session_end_time: exit_reason_str = 'Session End'; exit_triggered = True


            if exit_triggered: position_active = 0; trailing_stop_price = np.nan; morning_trade_active = False; df.iloc[i, col_indices['exit_reason']] = exit_reason_str

            if position_active == 0 and not exit_triggered:
                if current_time == time_0915: position_active = 1; morning_trade_active = True; trailing_stop_price = np.nan
                elif is_buy and current_time < session_end_time:
                    position_active = 1; morning_trade_active = False; stop_offset = np.nan
                    if pd.notna(current_atr) and current_atr > 0:
                         time_check = current_time < time_0930; atr_mult = self.atr_stop_mult_early if time_check else self.atr_stop_mult_late
                         if time_check and pd.notna(prev_high) and pd.notna(prev_low): prev_range = prev_high - prev_low; stop_offset = (0.5 * prev_range + atr_mult * current_atr) if prev_range >= 0 else (atr_mult * current_atr)
                         else: stop_offset = atr_mult * current_atr
                         trailing_stop_price = (current_close - stop_offset) if pd.notna(stop_offset) and stop_offset > 0 else np.nan
                    else: trailing_stop_price = np.nan
                elif is_sell and current_time < session_end_time: # Add short entry
                    position_active = -1; morning_trade_active = False; stop_offset = np.nan
                    if pd.notna(current_atr) and current_atr > 0:
                        time_check = current_time < time_0930; atr_mult = self.atr_stop_mult_early if time_check else self.atr_stop_mult_late
                        if time_check and pd.notna(prev_high) and pd.notna(prev_low): prev_range = prev_high - prev_low; stop_offset = (0.5 * prev_range + atr_mult * current_atr) if prev_range >= 0 else (atr_mult * current_atr)
                        else: stop_offset = atr_mult * current_atr
                        trailing_stop_price = (current_close + stop_offset) if pd.notna(stop_offset) and stop_offset > 0 else np.nan
                    else: trailing_stop_price = np.nan


            if pd.notna(current_atr) and current_atr > 0 and not exit_triggered and current_time >= time_0930:
                if position_active == 1 and not morning_trade_active:
                    atr_mult = self.atr_stop_mult_tight if current_rsi > self.rsi_tighten_long else self.atr_stop_mult_late # Adjusted threshold
                    potential_new_stop = current_close - (atr_mult * current_atr)
                    if pd.notna(potential_new_stop): trailing_stop_price = np.nanmax([trailing_stop_price, potential_new_stop])
                elif position_active == -1: # Add short trailing stop logic
                    atr_mult = self.atr_stop_mult_tight if current_rsi < self.rsi_tighten_short else self.atr_stop_mult_late
                    potential_new_stop = current_close + (atr_mult * current_atr)
                    if pd.notna(potential_new_stop): trailing_stop_price = np.nanmin([trailing_stop_price, potential_new_stop])


            df.iloc[i, col_indices['position']] = position_active

        # --- 6. Generate Final Signal Column and Rename Columns ---
        df['position_prev'] = df['position'].shift(1).fillna(0); df['signal'] = 0
        df.loc[(df['position'] == 1) & (df['position_prev'] == 0), 'signal'] = 1
        df.loc[(df['position'] == -1) & (df['position_prev'] == 0), 'signal'] = -1
        df.loc[(df['position'] == 0) & (df['position_prev'] == 1), 'signal'] = 2
        df.loc[(df['position'] == 0) & (df['position_prev'] == -1), 'signal'] = -2

        rename_map = {}
        if macd_internal_col in df.columns: rename_map[macd_internal_col] = macd_final_col
        if macds_internal_col in df.columns: rename_map[macds_internal_col] = macds_final_col
        df = df.rename(columns=rename_map)
        logger.info(f"Renamed columns: {rename_map}")

        # Select final columns
        original_cols_present = [col for col in data.columns if col.lower() in ['open', 'high', 'low', 'close', 'volume']]
        if not original_cols_present: original_cols_present = [col for col in data.columns if col.lower() in ['open', 'high', 'low', 'close']]
        original_cols_in_df = [col for col in df.columns if col in original_cols_present]
        # Include condition counts in output for backtester
        final_indicator_cols = ['alphatrend', atr_sim_col, rsi_col, macd_final_col, macds_final_col, 'rsi_slope', 'buy_conditions_met', 'sell_conditions_met']
        if supertrend_calculated: final_indicator_cols.append(expected_st_dir_col)
        strategy_cols = ['position', 'signal', 'exit_reason']
        final_cols_ordered = original_cols_in_df + [col for col in final_indicator_cols if col in df.columns] + strategy_cols
        final_cols_ordered = list(dict.fromkeys(final_cols_ordered))
        final_cols_existing = [col for col in final_cols_ordered if col in df.columns]
        final_df = df[final_cols_existing].copy()

        sim_required = ['close', 'position', 'high', 'low', rsi_col, 'rsi_slope', macd_final_col, macds_final_col, atr_sim_col]
        missing_for_sim = [col for col in sim_required if col not in final_df.columns]
        if missing_for_sim: logger.error(f"Final DF missing sim cols: {missing_for_sim}")

        logger.info(f"generate_signals returning cols: {final_df.columns.tolist()}")
        return final_df


# --- Detailed Backtesting Function ---
def run_detailed_backtest(df_signals: pd.DataFrame, strategy: AlphaTrendStrategy) -> None:
    """
    Runs a detailed backtest simulation based on signals from AlphaTrendStrategy.

    Args:
        df_signals (pd.DataFrame): DataFrame with OHLC, indicators, position, signal, exit_reason.
        strategy (AlphaTrendStrategy): The instantiated strategy object to access parameters.
    """
    trades = []
    backtest_log = []
    active_trade: Optional[Dict[str, Any]] = None # Holds current open trade details
    trade_counter = 0

    # Get necessary parameters from strategy object
    atr_strategy_col = f'atr_sma_{strategy.ap}'
    rsi_col = f'rsi_{strategy.ap}'
    sl_pct = 1.0 # Example SL %, could be made dynamic or passed if needed

    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'signal', atr_strategy_col, rsi_col, 'buy_conditions_met', 'sell_conditions_met']
    if not all(col in df_signals.columns for col in required_cols):
        logger.error(f"Backtest input missing columns: {[c for c in required_cols if c not in df_signals.columns]}")
        return

    logger.info("Starting detailed backtest...")
    print("\n--- Detailed Backtest Log ---")

    for idx, row in df_signals.iterrows():
        current_time = idx.time()
        current_price = row['close'] # Use close for logging, high/low for checks
        signal = row['signal']
        current_atr = row[atr_strategy_col]
        current_rsi = row[rsi_col]

        log_entry = f"[{idx}] Px:{current_price:.2f} | "
        trade_info = ""

        # --- TSL Update ---
        if active_trade:
            side = active_trade['side']
            # Update highest/lowest price since entry
            if side == 'long':
                active_trade['high_since_entry'] = max(active_trade['high_since_entry'], row['high'])
            elif side == 'short':
                active_trade['low_since_entry'] = min(active_trade['low_since_entry'], row['low'])

            # Calculate potential new TSL only after 09:30
            if current_time >= time(9, 30) and pd.notna(current_atr) and current_atr > 0:
                if side == 'long':
                    atr_mult = strategy.atr_stop_mult_tight if current_rsi > strategy.rsi_tighten_long else strategy.atr_stop_mult_late
                    potential_tsl = active_trade['high_since_entry'] - (atr_mult * current_atr)
                    active_trade['tsl'] = np.nanmax([active_trade['tsl'], potential_tsl]) # Trail up
                elif side == 'short':
                    atr_mult = strategy.atr_stop_mult_tight if current_rsi < strategy.rsi_tighten_short else strategy.atr_stop_mult_late
                    potential_tsl = active_trade['low_since_entry'] + (atr_mult * current_atr)
                    active_trade['tsl'] = np.nanmin([active_trade['tsl'], potential_tsl]) # Trail down

            trade_info = (f"Running {active_trade['id']} ({side.upper()}) | "
                          f"Entry:{active_trade['entry_price']:.2f} | "
                          f"SL:{active_trade['initial_sl']:.2f} | "
                          f"TSL:{active_trade['tsl']:.2f}")


        # --- Exit Check ---
        if active_trade:
            side = active_trade['side']
            initial_sl = active_trade['initial_sl']
            tsl = active_trade['tsl']
            exit_reason = None
            exit_price = None

            if side == 'long':
                if row['low'] <= initial_sl: exit_reason, exit_price = "SL Hit", initial_sl
                elif row['low'] <= tsl: exit_reason, exit_price = "TSL Hit", tsl
                elif signal == 2: exit_reason, exit_price = "Strategy Exit Signal", current_price # Exit at close on signal
                # Add other strategy exit reasons if needed (e.g., RSI extreme from strategy output)
                elif row['exit_reason'] and row['exit_reason'] != '': exit_reason, exit_price = row['exit_reason'], current_price
            elif side == 'short':
                if row['high'] >= initial_sl: exit_reason, exit_price = "SL Hit", initial_sl
                elif row['high'] >= tsl: exit_reason, exit_price = "TSL Hit", tsl
                elif signal == -2: exit_reason, exit_price = "Strategy Exit Signal", current_price
                elif row['exit_reason'] and row['exit_reason'] != '': exit_reason, exit_price = row['exit_reason'], current_price

            if exit_reason:
                # Clip exit price to row's high/low
                exit_price = np.clip(exit_price, row['low'], row['high'])
                profit = (exit_price - active_trade['entry_price']) if side == 'long' else (active_trade['entry_price'] - exit_price)
                active_trade['exit_time'] = idx
                active_trade['exit_price'] = exit_price
                active_trade['exit_reason'] = exit_reason
                active_trade['profit'] = profit
                trades.append(active_trade)
                log_entry += f"EXIT {active_trade['id']} ({side.upper()}) @ {exit_price:.2f} ({exit_reason}). PnL: {profit:.2f}"
                active_trade = None # Clear active trade
            else:
                 log_entry += trade_info # Log running trade info if not exited

        # --- Entry Check ---
        if not active_trade: # Only enter if flat
            entry_price = row['close'] # Enter at close for simplicity, could use open of next bar
            side = None
            conditions_met = 0

            if signal == 1: # Enter Long
                side = 'long'
                conditions_met = row['buy_conditions_met']
                initial_sl_offset = np.nan
                if pd.notna(current_atr) and current_atr > 0:
                    time_check = current_time < time(9, 30)
                    atr_mult = strategy.atr_stop_mult_early if time_check else strategy.atr_stop_mult_late
                    prev_high = row['high_shift1'] if 'high_shift1' in row.index else np.nan # Need shifted cols
                    prev_low = row['low_shift1'] if 'low_shift1' in row.index else np.nan
                    if time_check and pd.notna(prev_high) and pd.notna(prev_low):
                        prev_range = prev_high - prev_low
                        initial_sl_offset = (0.5 * prev_range + atr_mult * current_atr) if prev_range >= 0 else (atr_mult * current_atr)
                    else:
                        initial_sl_offset = atr_mult * current_atr
                initial_sl = (entry_price - initial_sl_offset) if pd.notna(initial_sl_offset) and initial_sl_offset > 0 else np.nan

            elif signal == -1: # Enter Short
                side = 'short'
                conditions_met = row['sell_conditions_met']
                initial_sl_offset = np.nan
                if pd.notna(current_atr) and current_atr > 0:
                    time_check = current_time < time(9, 30)
                    atr_mult = strategy.atr_stop_mult_early if time_check else strategy.atr_stop_mult_late
                    prev_high = row['high_shift1'] if 'high_shift1' in row.index else np.nan
                    prev_low = row['low_shift1'] if 'low_shift1' in row.index else np.nan
                    if time_check and pd.notna(prev_high) and pd.notna(prev_low):
                        prev_range = prev_high - prev_low
                        initial_sl_offset = (0.5 * prev_range + atr_mult * current_atr) if prev_range >= 0 else (atr_mult * current_atr)
                    else:
                        initial_sl_offset = atr_mult * current_atr
                initial_sl = (entry_price + initial_sl_offset) if pd.notna(initial_sl_offset) and initial_sl_offset > 0 else np.nan


            if side: # If entry signal triggered
                trade_counter += 1
                trade_id = f"{side.capitalize()}_{trade_counter}_{idx.strftime('%H%M')}"
                active_trade = {
                    'id': trade_id,
                    'side': side,
                    'entry_time': idx,
                    'entry_price': entry_price,
                    'initial_sl': initial_sl if pd.notna(initial_sl) else (entry_price * (1 - sl_pct/100) if side=='long' else entry_price * (1 + sl_pct/100)), # Fallback SL
                    'tsl': initial_sl if pd.notna(initial_sl) else (entry_price * (1 - sl_pct/100) if side=='long' else entry_price * (1 + sl_pct/100)), # Init TSL at SL
                    'high_since_entry': entry_price if side == 'long' else -np.inf, # Track high/low for TSL
                    'low_since_entry': entry_price if side == 'short' else np.inf,
                    'conditions_met': conditions_met,
                    'exit_time': None, 'exit_price': None, 'exit_reason': None, 'profit': None
                }
                log_entry += (f"ENTER {side.upper()} | ID:{trade_id} | Entry:{entry_price:.2f} | "
                              f"SL:{active_trade['initial_sl']:.2f} | CondMet:{conditions_met:.1f}")
            elif trade_info == "": # Only log Hold if not already logging a running trade
                log_entry += "Hold"

        # Append log entry for the current bar
        backtest_log.append(log_entry)

    # --- Post-Backtest Summary ---
    print("\n--- Backtest Summary ---")
    total_trades = len(trades)
    if total_trades > 0:
        total_pnl = sum(t['profit'] for t in trades if pd.notna(t['profit']))
        winning_trades = sum(1 for t in trades if pd.notna(t['profit']) and t['profit'] > 0)
        accuracy = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {total_trades - winning_trades}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Total PnL: {total_pnl:.2f}")

        # Save detailed trades to CSV
        trades_df = pd.DataFrame(trades)
        trades_filename = f"detailed_trades_output_{datetime.now():%Y%m%d_%H%M%S}.csv"
        try:
            trades_df.to_csv(trades_filename, index=False)
            logger.info(f"Detailed trade list saved to {trades_filename}")
        except Exception as e:
            logger.error(f"Failed to save detailed trades: {e}")
    else:
        print("No trades were executed.")

    # Save detailed log to file
    log_filename = f"detailed_backtest_log_{datetime.now():%Y%m%d_%H%M%S}.txt"
    try:
        with open(log_filename, 'w') as f:
            for line in backtest_log:
                f.write(line + '\n')
        logger.info(f"Detailed backtest log saved to {log_filename}")
    except Exception as e:
        logger.error(f"Failed to save backtest log: {e}")

data = '''datetime,open,high,low,close,volume
2025-04-29 09:15:00+05:30,24370.7,24442.25,24364.35,24439.25,0
2025-04-29 09:20:00+05:30,24438.5,24455.05,24424.5,24453.7,0
2025-04-29 09:25:00+05:30,24453.6,24457.65,24413.3,24417.1,0
2025-04-29 09:30:00+05:30,24418.25,24452.9,24393.55,24440.2,0
2025-04-29 09:35:00+05:30,24440.85,24442.5,24359.2,24369.45,0
2025-04-29 09:40:00+05:30,24370.1,24387.05,24303.9,24316.6,0
2025-04-29 09:45:00+05:30,24317.85,24364.5,24308.1,24315.55,0
2025-04-29 09:50:00+05:30,24316.75,24340.05,24290.75,24329.1,0
2025-04-29 09:55:00+05:30,24330.15,24395,24326.5,24390.25,0
2025-04-29 10:00:00+05:30,24390,24396.15,24362.2,24378.3,0
2025-04-29 10:05:00+05:30,24379.15,24379.75,24333.8,24344.65,0
2025-04-29 10:10:00+05:30,24345.2,24365.1,24338.55,24351.55,0
2025-04-29 10:15:00+05:30,24351.3,24354.1,24319,24319,0
2025-04-29 10:20:00+05:30,24318.1,24347.1,24317.4,24346.4,0
2025-04-29 10:25:00+05:30,24346.45,24348.15,24311.3,24317.8,0
2025-04-29 10:30:00+05:30,24318.85,24329.7,24305.5,24314.95,0
2025-04-29 10:35:00+05:30,24316.2,24335.25,24314.55,24331.5,0
2025-04-29 10:40:00+05:30,24331.55,24333.9,24312.8,24321.5,0
2025-04-29 10:45:00+05:30,24321.85,24352.35,24317.95,24350.7,0
2025-04-29 10:50:00+05:30,24349.85,24353.7,24328.45,24330.85,0
2025-04-29 10:55:00+05:30,24330.25,24349.95,24330.25,24333.8,0
2025-04-29 11:00:00+05:30,24334.7,24351.9,24332.8,24351.9,0
2025-04-29 11:05:00+05:30,24353.05,24369.85,24351.9,24367.1,0
2025-04-29 11:10:00+05:30,24367.25,24369.7,24350.6,24356.05,0
2025-04-29 11:15:00+05:30,24356.55,24358.5,24338.95,24348.8,0
2025-04-29 11:20:00+05:30,24348.3,24360.35,24347.05,24348.95,0
2025-04-29 11:25:00+05:30,24349.05,24358.75,24347.6,24349.25,0
2025-04-29 11:30:00+05:30,24348.65,24351.6,24334.95,24347.65,0
2025-04-29 11:35:00+05:30,24347.95,24353.6,24341.6,24346.45,0
2025-04-29 11:40:00+05:30,24346.7,24356.8,24342.2,24354.4,0
2025-04-29 11:45:00+05:30,24354.65,24355.8,24335.3,24340.15,0
2025-04-29 11:50:00+05:30,24340.8,24340.85,24321,24322.95,0
2025-04-29 11:55:00+05:30,24323.4,24329.1,24311.95,24313.25,0
2025-04-29 12:00:00+05:30,24313.4,24322.5,24302.45,24322.05,0
2025-04-29 12:05:00+05:30,24322.15,24323.75,24310.65,24315.5,0
2025-04-29 12:10:00+05:30,24315.3,24322.3,24309.2,24316.6,0
2025-04-29 12:15:00+05:30,24316.4,24321.6,24310.65,24313.1,0
2025-04-29 12:20:00+05:30,24315.1,24315.85,24304.1,24313.65,0
2025-04-29 12:25:00+05:30,24312.95,24317.25,24306.6,24309.75,0
2025-04-29 12:30:00+05:30,24309.25,24330.8,24309.15,24329.95,0
2025-04-29 12:35:00+05:30,24328.75,24347.95,24325.4,24341.35,0
2025-04-29 12:40:00+05:30,24340.6,24346.8,24336.4,24341.85,0
2025-04-29 12:45:00+05:30,24338.8,24344.85,24330.6,24339.55,0
2025-04-29 12:50:00+05:30,24339.15,24358.35,24336.45,24357.9,0
2025-04-29 12:55:00+05:30,24357.75,24363.55,24353,24358.95,0
2025-04-29 13:00:00+05:30,24358.25,24369.5,24352.2,24368.4,0
2025-04-29 13:05:00+05:30,24368.05,24368.5,24350.6,24360.1,0
2025-04-29 13:10:00+05:30,24359.4,24363.55,24353.5,24356.85,0
2025-04-29 13:15:00+05:30,24355.45,24359.35,24341.35,24354.1,0
2025-04-29 13:20:00+05:30,24352.2,24363.15,24346.55,24362.9,0
2025-04-29 13:25:00+05:30,24362.1,24376.05,24356,24366.05,0
2025-04-29 13:30:00+05:30,24366.65,24371.7,24361.45,24366.8,0
2025-04-29 13:35:00+05:30,24366.5,24368.8,24357.4,24366.15,0
2025-04-29 13:40:00+05:30,24367.15,24372.7,24351,24353.9,0
2025-04-29 13:45:00+05:30,24352.7,24366.95,24351.75,24363.45,0
2025-04-29 13:50:00+05:30,24364,24368.75,24355.1,24355.65,0
2025-04-29 13:55:00+05:30,24356.65,24363.4,24347.65,24358.25,0
2025-04-29 14:00:00+05:30,24360.15,24368.3,24356.85,24363.55,0
2025-04-29 14:05:00+05:30,24363.8,24375.5,24360.6,24365.2,0
2025-04-29 14:10:00+05:30,24365.35,24374.8,24360.25,24365.95,0
2025-04-29 14:15:00+05:30,24366.9,24368.3,24355.05,24361.65,0
2025-04-29 14:20:00+05:30,24361.9,24367.55,24356.25,24357.15,0
2025-04-29 14:25:00+05:30,24357.75,24359.7,24342.4,24346,0
2025-04-29 14:30:00+05:30,24346.45,24350.5,24339,24343.45,0
2025-04-29 14:35:00+05:30,24343.75,24361.95,24343.75,24348.1,0
2025-04-29 14:40:00+05:30,24348.8,24360.7,24347.2,24356.45,0
2025-04-29 14:45:00+05:30,24356.2,24357.95,24347.85,24352.95,0
2025-04-29 14:50:00+05:30,24353.35,24358,24344.85,24346.85,0
2025-04-29 14:55:00+05:30,24346.8,24355.7,24339.45,24340.2,0
2025-04-29 15:00:00+05:30,24340.55,24347.8,24325.2,24331.4,0
2025-04-29 15:05:00+05:30,24331.55,24336.5,24324.9,24332.7,0
2025-04-29 15:10:00+05:30,24332.6,24335.6,24324.45,24326.45,0
2025-04-29 15:15:00+05:30,24326.2,24341.45,24310.6,24339.65,0
2025-04-29 15:20:00+05:30,24338.85,24348.5,24325.85,24333.1,0
2025-04-29 15:25:00+05:30,24331.55,24340.85,24317.6,24325.45,0'''
# load into DataFrame

# df = pd.read_csv(StringIO(data), parse_dates=['datetime'])
# df.set_index('datetime', inplace=True)

# --- Script Execution ---
# Load Data
df = pd.read_csv(StringIO(data))
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime'])
df = df.set_index('datetime', inplace=True)

# Ensure Timezone
target_tz = 'Asia/Kolkata'
if df.index.tz is None:
    try: df.index = df.index.tz_localize(target_tz, ambiguous='infer'); logger.info(f"Localized to {target_tz}")
    except Exception as e: logger.warning(f"TZ localize failed: {e}")
elif str(df.index.tz) != target_tz:
    try: df.index = df.index.tz_convert(target_tz); logger.info(f"Converted to {target_tz}")
    except Exception as e: logger.warning(f"TZ convert failed: {e}")

# Instantiate and Run Strategy to get signals
try:
    logger.info("Instantiating MODIFIED AlphaTrendStrategy...")
    strategy = AlphaTrendStrategy() # Uses the modified __init__ defaults

    logger.info("Running MODIFIED AlphaTrendStrategy.generate_signals...")
    df_with_signals = strategy.generate_signals(df.copy()) # Pass a copy
    logger.info("Signal generation complete.")

    # --- Run Detailed Backtest ---
    if not df_with_signals.empty:
        run_detailed_backtest(df_with_signals, strategy)
    else:
        print("\nStrategy did not generate signals. Cannot run backtest.")

except Exception as e:
    logger.error(f"Error during MODIFIED strategy execution or backtest: {e}", exc_info=True)

