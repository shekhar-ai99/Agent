import pandas as pd
import numpy as np
import pandas_ta as ta # Strategy class needs this
from datetime import time
import logging
from typing import Optional

# Configure logging for the strategy file (optional, but can be helpful)
# Use a unique name for the logger if running alongside other modules
strategy_logger = logging.getLogger('AlphaTrendStrategyLogger')
# Prevent duplicate handlers if basicConfig was called elsewhere
if not strategy_logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    strategy_logger = logging.getLogger('AlphaTrendStrategyLogger') # Re-get logger after config


# --- Define Modified AlphaTrendStrategy Class ---
class AlphaTrendStrategy:
    """
    Implements the AlphaTrend trading strategy with ADJUSTED parameters
    to potentially generate more signals.

    Key Logic Points:
    - AlphaTrend Formula: 0.6 * close + 0.3 * prev_AlphaTrend + 0.1 * RSI
    - Entry Conditions: 3 out of 6 specific conditions must be met (Relaxed).
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
        # --- MORE Adjusted Parameters ---
        rsi_buy_lower: int = 35,  # Relaxed: Was 40
        rsi_buy_upper: int = 85,  # Relaxed: Was 80
        rsi_sell_lower: int = 25, # Relaxed: Was 30
        rsi_sell_upper: int = 65,  # Relaxed: Was 60
        rsi_extreme_long: int = 80,  # Relaxed: Was 75 -> Back to 80
        rsi_extreme_short: int = 20,  # Relaxed: Was 25 -> Back to 20
        rsi_tighten_long: int = 70,  # Relaxed: Was 75
        rsi_tighten_short: int = 30,  # Relaxed: Was 25
        volatility_threshold: float = 0.25, # Relaxed: Was 0.5
        entry_condition_count: int = 3,  # Relaxed: Was 4
        # --- Stop Parameters (Unchanged) ---
        atr_stop_mult_early: float = 2.5,
        atr_stop_mult_late: float = 3.5,
        atr_stop_mult_tight: float = 2.0 # Used for RSI extremes AND score flips
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

        # Log initialization parameters using the specific logger
        strategy_logger.info(
            f"Init MORE Relaxed AlphaTrend: coeff={coeff}, ap={ap}, "
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
            strategy_logger.warning("Adjusted AlphaTrend: empty input dataframe.")
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
                 except Exception as e: strategy_logger.error(f"Failed parse 'datetime': {e}"); return empty_df_output
             elif 'Date' in df.columns and 'Time' in df.columns:
                 try:
                     datetime_col = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
                     df['temp_datetime'] = datetime_col
                     df = df.dropna(subset=['temp_datetime'])
                     if df.empty: return empty_df_output
                     df.index = df['temp_datetime']
                     df = df.drop(columns=['Date', 'Time', 'temp_datetime'])
                 except Exception as e: strategy_logger.error(f"Failed combine Date/Time: {e}"); return empty_df_output
             else: strategy_logger.error("Requires DatetimeIndex or 'datetime' col."); return empty_df_output

        if not df.index.is_monotonic_increasing:
            strategy_logger.info("Sorting data by index."); df = df.sort_index()

        target_tz = 'Asia/Kolkata'
        if df.index.tz is None:
             try: df.index = df.index.tz_localize(target_tz, ambiguous='infer')
             except Exception as e: strategy_logger.warning(f"TZ localize failed: {e}")
        elif str(df.index.tz) != target_tz:
            try: df.index = df.index.tz_convert(target_tz)
            except Exception as e: strategy_logger.warning(f"TZ convert failed: {e}")

        # --- 2. Calculate Required Indicators ---
        required_cols = ['high', 'low', 'close']
        df.columns = df.columns.str.lower()
        if not all(col in df.columns for col in required_cols):
             strategy_logger.error(f"Missing OHLC columns: {required_cols}"); return empty_df_output

        min_data_length = max(self.ap, self.macd_slow, self.supertrend_period, 14) + 5
        if len(df) < min_data_length: strategy_logger.warning(f"Short data ({len(df)}<{min_data_length})")

        # ATR (Strategy AP and Sim 14)
        atr_strategy_col = f'atr_sma_{self.ap}'
        atr_sim_col = 'atr_14'; atr_sim_period = 14
        try:
            df['tr'] = self._calculate_true_range(df['high'], df['low'], df['close'])
            df['tr'] = pd.to_numeric(df['tr'], errors='coerce')
            df[atr_strategy_col] = ta.sma(df['tr'], length=self.ap); df[atr_strategy_col] = df[atr_strategy_col].bfill()
            df[atr_sim_col] = ta.sma(df['tr'], length=atr_sim_period); df[atr_sim_col] = df[atr_sim_col].bfill()
            df = df.drop(columns=['tr'], errors='ignore')
        except Exception as e: strategy_logger.error(f"ATR Error: {e}"); df[atr_strategy_col]=np.nan; df[atr_sim_col]=np.nan

        # RSI and Slope (Strategy AP)
        rsi_col = f'rsi_{self.ap}'
        try:
            df[rsi_col] = ta.rsi(df['close'], length=self.ap)
            df['rsi_slope'] = df[rsi_col].diff()
            df[rsi_col] = df[rsi_col].bfill(); df['rsi_slope'] = df['rsi_slope'].fillna(0)
        except Exception as e: strategy_logger.error(f"RSI Error: {e}"); df[rsi_col]=np.nan; df['rsi_slope']=np.nan

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
            else: df[macd_internal_col]=np.nan; df[macds_internal_col]=np.nan; df[macd_hist_internal_col]=np.nan; strategy_logger.warning("MACD calc failed.")
        except Exception as e: strategy_logger.error(f"MACD Error: {e}"); df[macd_internal_col]=np.nan; df[macds_internal_col]=np.nan; df[macd_hist_internal_col]=np.nan

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
                     strategy_logger.info(f"Supertrend direction found: {expected_st_dir_col}")
                     supertrend_calculated = True
                else: strategy_logger.warning(f"Expected ST col '{expected_st_dir_col}' not found. Cols: {st.columns.tolist()}. Defaulting ST to 0."); df[expected_st_dir_col] = 0
            else: strategy_logger.warning("ST calc failed. Defaulting ST to 0."); df[expected_st_dir_col] = 0
        except Exception as e: strategy_logger.error(f"ST Error: {e}"); df[expected_st_dir_col] = 0

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
        else: strategy_logger.warning("No valid RSI for AlphaTrend init.")

        # --- 4. Calculate Entry Conditions (Using Adjusted Parameters) ---
        required_cond_cols_internal = [rsi_col, 'rsi_slope', macd_internal_col, macds_internal_col, expected_st_dir_col]
        if not all(col in df.columns and pd.api.types.is_numeric_dtype(df[col]) for col in required_cond_cols_internal):
            missing = [col for col in required_cond_cols_internal if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col])]
            strategy_logger.error(f"Entry condition indicators missing/non-numeric: {missing}. Cannot generate signals."); return empty_df_output

        # *** Include high_shift1 and low_shift1 calculation here ***
        df['close_shift2'] = df['close'].shift(2)
        df['high_shift1'] = df['high'].shift(1)
        df['low_shift1'] = df['low'].shift(1)

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
            strategy_logger.error(f"Missing loop columns: {[col for name, col in required_loop_cols.items() if col not in df.columns]}"); return empty_df_output

        for i, idx in enumerate(df.index):
            current_time = idx.time()
            try: current_low = df.iloc[i, col_indices['low']]; current_high = df.iloc[i, col_indices['high']]; current_close = df.iloc[i, col_indices['close']]; current_rsi = df.iloc[i, col_indices[rsi_col]]; current_atr = df.iloc[i, col_indices[atr_strategy_col]]; prev_high = df.iloc[i, col_indices['high_shift1']]; prev_low = df.iloc[i, col_indices['low_shift1']]; is_buy = df.iloc[i, col_indices['buy_entry_signal']]; is_sell = df.iloc[i, col_indices['sell_entry_signal']]
            except IndexError: strategy_logger.error(f"IndexError at {idx}");
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
        # strategy_logger.info(f"Renamed columns: {rename_map}") # Reduced logging verbosity

        # --- Select final columns, ensuring required backtest columns are included ---
        original_cols_present = [col for col in data.columns if col.lower() in ['open', 'high', 'low', 'close', 'volume']]
        if not original_cols_present: original_cols_present = [col for col in data.columns if col.lower() in ['open', 'high', 'low', 'close']]
        original_cols_in_df = [col for col in df.columns if col in original_cols_present]

        # Define columns needed for indicators, strategy logic, and backtesting
        final_indicator_cols = ['alphatrend', atr_strategy_col, atr_sim_col, rsi_col, macd_final_col, macds_final_col, 'rsi_slope',
                                'buy_conditions_met', 'sell_conditions_met', 'high_shift1', 'low_shift1'] # Added required cols
        if supertrend_calculated: final_indicator_cols.append(expected_st_dir_col)
        strategy_cols = ['position', 'signal', 'exit_reason']

        # Combine, ensure existence, remove duplicates
        final_cols_ordered = original_cols_in_df + [col for col in final_indicator_cols if col in df.columns] + strategy_cols
        final_cols_ordered = list(dict.fromkeys(final_cols_ordered))
        final_cols_existing = [col for col in final_cols_ordered if col in df.columns]
        final_df = df[final_cols_existing].copy()

        # Verify required simulation columns are present before returning
        sim_required = ['close', 'position', 'high', 'low', rsi_col, 'rsi_slope', macd_final_col, macds_final_col, atr_sim_col, atr_strategy_col, 'high_shift1', 'low_shift1']
        missing_for_sim = [col for col in sim_required if col not in final_df.columns]
        if missing_for_sim: strategy_logger.error(f"Final DF missing sim cols: {missing_for_sim}")

        # strategy_logger.info(f"generate_signals returning cols: {final_df.columns.tolist()}") # Reduced logging verbosity
        return final_df
