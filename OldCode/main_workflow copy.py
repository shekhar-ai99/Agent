# main_workflow.py
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np # Added numpy import
import logging
from pathlib import Path
import sys

# Import the classes from the other files
# Assuming EnhancedMultiStrategyBacktester is correctly modified for TSL
from backtester_engine import EnhancedMultiStrategyBacktester 
from backtester_analyzer import BacktestAnalyzerReporter

# --- Configure logging (can be configured once here) ---
# (Keep your existing logging setup)
logging.basicConfig(
    level=logging.INFO, # Set to DEBUG to see AI_v6 detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Include logger name
    handlers=[
        logging.FileHandler('main_workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__) # Use __name__ for logger


# --- Example Strategies (Should ideally live in a separate 'strategies.py' file) ---
# Import strategies or define them here for simplicity in this example
def strategy_ema_crossover(current_row: pd.Series, data: pd.DataFrame = None,params: Dict = None) -> str:
    # (Keep existing strategy code)
    if pd.isna(current_row['ema_9']) or pd.isna(current_row['ema_21']): return 'hold'
    idx = data.index.get_loc(current_row.name);
    if idx == 0: return 'hold'
    prev_row = data.iloc[idx-1]
    if current_row['ema_9'] > current_row['ema_21'] and prev_row['ema_9'] <= prev_row['ema_21']: return 'buy'
    elif current_row['ema_9'] < current_row['ema_21'] and prev_row['ema_9'] >= prev_row['ema_21']: return 'sell'
    return 'hold'

def strategy_rsi_confirmed(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    # (Keep existing strategy code)
    rsi_oversold = 35; rsi_overbought = 65
    required_indicators = ['rsi', 'macd', 'macd_signal']
    if current_row[required_indicators].isna().any(): return 'hold'
    idx = data.index.get_loc(current_row.name)
    if idx == 0: return 'hold'
    prev_row = data.iloc[idx-1]
    if pd.isna(prev_row['rsi']): return 'hold'
    rsi_crossed_above_os = current_row['rsi'] > rsi_oversold and prev_row['rsi'] <= rsi_oversold
    rsi_crossed_below_ob = current_row['rsi'] < rsi_overbought and prev_row['rsi'] >= rsi_overbought
    macd_bullish = current_row['macd'] > current_row['macd_signal']
    macd_bearish = current_row['macd'] < current_row['macd_signal']
    if rsi_crossed_above_os and macd_bullish: return 'buy'
    elif rsi_crossed_below_ob and macd_bearish: return 'sell'
    return 'hold'

def strategy_rsi_threshold(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    # (Keep existing strategy code)
    rsi_oversold = 35; rsi_overbought = 65
    if pd.isna(current_row['rsi']): return 'hold'
    idx = data.index.get_loc(current_row.name);
    if idx == 0: return 'hold'
    prev_row = data.iloc[idx-1]
    if current_row['rsi'] > rsi_oversold and prev_row['rsi'] <= rsi_oversold: return 'buy'
    elif current_row['rsi'] < rsi_overbought and prev_row['rsi'] >= rsi_overbought: return 'sell'
    return 'hold'

def strategy_ema_crossover_filtered(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    # (Keep existing strategy code)
    required_indicators = ['ema_9', 'ema_21', 'ema_50', 'close']
    if current_row[required_indicators].isna().any(): return 'hold'
    idx = data.index.get_loc(current_row.name)
    if idx == 0: return 'hold'
    prev_row = data.iloc[idx-1]
    if pd.isna(prev_row['ema_9']) or pd.isna(prev_row['ema_21']): return 'hold'
    crossed_up = current_row['ema_9'] > current_row['ema_21'] and prev_row['ema_9'] <= prev_row['ema_21']
    crossed_down = current_row['ema_9'] < current_row['ema_21'] and prev_row['ema_9'] >= prev_row['ema_21']
    is_uptrend = current_row['close'] > current_row['ema_50']
    is_downtrend = current_row['close'] < current_row['ema_50']
    if crossed_up and is_uptrend: return 'buy'
    elif crossed_down and is_downtrend: return 'sell'
    return 'hold'

def strategy_macd_cross(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    # (Keep existing strategy code)
    if pd.isna(current_row['macd']) or pd.isna(current_row['macd_signal']): return 'hold'
    idx = data.index.get_loc(current_row.name);
    if idx == 0: return 'hold'
    prev_row = data.iloc[idx-1]
    if current_row['macd'] > current_row['macd_signal'] and prev_row['macd'] <= prev_row['macd_signal']: return 'buy'
    elif current_row['macd'] < current_row['macd_signal'] and prev_row['macd'] >= prev_row['macd_signal']: return 'sell'
    return 'hold'

def strategy_bb_squeeze_breakout(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    # (Keep existing strategy code)
    if pd.isna(current_row['bollinger_bandwidth']) or pd.isna(current_row['close']) or pd.isna(current_row['bollinger_upper']) or pd.isna(current_row['bollinger_lower']): return 'hold'
    idx = data.index.get_loc(current_row.name); squeeze_lookback = 20
    if idx < squeeze_lookback: return 'hold'
    prev_rows = data.iloc[idx-squeeze_lookback:idx]; bandwidth_threshold = prev_rows['bollinger_bandwidth'].quantile(0.15)
    prev_bandwidth = data.iloc[idx-1]['bollinger_bandwidth'];
    if pd.isna(prev_bandwidth): return 'hold'
    in_squeeze_prev = prev_bandwidth < bandwidth_threshold; breaking_out_up = current_row['close'] > current_row['bollinger_upper']; breaking_out_down = current_row['close'] < current_row['bollinger_lower']
    if in_squeeze_prev and breaking_out_up: return 'buy'
    elif in_squeeze_prev and breaking_out_down: return 'sell'
    return 'hold'

def strategy_combined_momentum(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    # (Keep existing strategy code)
    if pd.isna(current_row['ema_9']) or pd.isna(current_row['ema_21']) or pd.isna(current_row['rsi']): return 'hold'
    ema_crossed_up = False; ema_crossed_down = False; idx = data.index.get_loc(current_row.name)
    if idx > 0: prev_row = data.iloc[idx-1]; ema_crossed_up = current_row['ema_9'] > current_row['ema_21'] and prev_row['ema_9'] <= prev_row['ema_21']; ema_crossed_down = current_row['ema_9'] < current_row['ema_21'] and prev_row['ema_9'] >= prev_row['ema_21']
    rsi_confirm_buy = current_row['rsi'] > 50; rsi_confirm_sell = current_row['rsi'] < 50
    if ema_crossed_up and rsi_confirm_buy: return 'buy'
    elif ema_crossed_down and rsi_confirm_sell: return 'sell'
    return 'hold'

def strategy_consolidated_simplified(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    # (Keep existing strategy code)
    emaFastLen = 9; emaMedLen = 14; emaSlowLen = 21; rsiBuyLevel = 55.0; rsiSellLevel = 45.0
    volMultiplier = 1.5; useEmaTrendFilter = True; useAdxFilter = True; adxThreshold = 20.0
    useAdxDirectionFilter = True; useVolBreakoutBuy = True; useVolBreakoutSell = True
    required_indicators = [f'ema_{emaFastLen}', f'ema_{emaMedLen}', f'ema_{emaSlowLen}', 'macd', 'macd_signal', 'rsi', 'volume', 'vol_ma', 'plus_di', 'minus_di', 'adx', 'close', 'open']
    if current_row[required_indicators].isna().any(): return 'hold'
    emaFast = current_row[f'ema_{emaFastLen}']; emaMed = current_row[f'ema_{emaMedLen}']; emaSlow = current_row[f'ema_{emaSlowLen}']
    macdLine = current_row['macd']; signalLine = current_row['macd_signal']; priceRsi = current_row['rsi']
    volume = current_row['volume']; volMA = current_row['vol_ma']; diPos = current_row['plus_di']
    diNeg = current_row['minus_di']; adxVal = current_row['adx']; close = current_row['close']; open_ = current_row['open']
    idx = data.index.get_loc(current_row.name);
    if idx == 0: return 'hold'
    prev_row = data.iloc[idx - 1]
    condEmaFastSlowCrossBuy = macdLine > signalLine and prev_row['macd'] <= prev_row['macd_signal']
    condEmaFastSlowCrossSell = macdLine < signalLine and prev_row['macd'] >= prev_row['macd_signal']
    condRsiOkBuy = priceRsi > rsiBuyLevel; condRsiOkSell = priceRsi < rsiSellLevel
    condHighVol = volume > volMA * volMultiplier if pd.notna(volMA) and volMA > 0 else False
    condVolBreakoutBuy = useVolBreakoutBuy and condHighVol and (close > open_) and (close > emaSlow)
    condVolBreakoutSell = useVolBreakoutSell and condHighVol and (close < open_) and (close < emaSlow)
    condEmaTrendOkBuy = not useEmaTrendFilter or (emaMed > emaSlow)
    condEmaTrendOkSell = not useEmaTrendFilter or (emaMed < emaSlow)
    condAdxStrengthOk = not useAdxFilter or (adxVal > adxThreshold)
    condAdxDirectionOkBuy = not useAdxDirectionFilter or (diPos > diNeg)
    condAdxDirectionOkSell = not useAdxDirectionFilter or (diNeg > diPos)
    condAdxFilterOkBuy = condAdxStrengthOk and condAdxDirectionOkBuy
    condAdxFilterOkSell = condAdxStrengthOk and condAdxDirectionOkSell
    allFiltersOkBuy = condEmaTrendOkBuy and condAdxFilterOkBuy
    allFiltersOkSell = condEmaTrendOkSell and condAdxFilterOkSell
    isPotentialBuy = (condEmaFastSlowCrossBuy or condVolBreakoutBuy) and condRsiOkBuy and allFiltersOkBuy
    isPotentialSell = (condEmaFastSlowCrossSell or condVolBreakoutSell) and condRsiOkSell and allFiltersOkSell
    if isPotentialBuy: return 'buy'
    elif isPotentialSell: return 'sell'
    else: return 'hold'

def strategy_ai_powered_v6(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    # (Keep existing strategy code, including the debug logging if desired)
    ema9Length = 9; ema50Length = 50; w_trend = 1.0; w_rsi = 1.0;
    aiBuyThreshold = 0.2; aiSellThreshold = -0.2; adxThreshold = 20
    required_indicators = [f'ema_{ema9Length}', f'ema_{ema50Length}', 'macd', 'macd_signal', 'macd_hist', 'rsi', 'plus_di', 'minus_di', 'adx', 'close']
    if current_row[required_indicators].isna().any(): return 'hold'
    ema9 = current_row[f'ema_{ema9Length}']; ema50 = current_row[f'ema_{ema50Length}']
    macd_line = current_row['macd']; macd_signal_line = current_row['macd_signal']
    macd_hist = current_row['macd_hist']; rsi = current_row['rsi']
    plus_di = current_row['plus_di']; minus_di = current_row['minus_di']
    adx = current_row['adx']; close = current_row['close']
    idx = data.index.get_loc(current_row.name); hist_lookback = 5
    if idx >= hist_lookback and 'macd_hist' in data.columns:
        trendFactor = data['macd_hist'].iloc[idx-hist_lookback+1 : idx+1].mean();
        if pd.isna(trendFactor): trendFactor = 0
    else: trendFactor = macd_hist
    rsiFactor = (rsi - 50) / 10; ai_confidence = (w_trend * trendFactor + w_rsi * rsiFactor) / max((w_trend + w_rsi), 1e-6)
    ai_buy_signal = ai_confidence > aiBuyThreshold; ai_sell_signal = ai_confidence < aiSellThreshold; isTrending = adx > adxThreshold
    if idx == 0: return 'hold'
    prev_row = data.iloc[idx - 1];
    if pd.isna(prev_row['macd']) or pd.isna(prev_row['macd_signal']): return 'hold'
    macdCrossover = macd_line > macd_signal_line and prev_row['macd'] <= prev_row['macd_signal']
    macdCrossunder = macd_line < macd_signal_line and prev_row['macd'] >= prev_row['macd_signal']
    close_gt_ema9 = close > ema9; close_gt_ema50 = close > ema50
    close_lt_ema9 = close < ema9; close_lt_ema50 = close < ema50
    # Use try-except for robustness if needed
    if (macdCrossover and ai_buy_signal and close_gt_ema9 and close_gt_ema50 and isTrending): return 'buy'
    elif (macdCrossunder and ai_sell_signal and close_lt_ema9 and close_lt_ema50 and isTrending): return 'sell'
    else: return 'hold'

def strategy_bb_adx_trend(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    # (Keep existing strategy code)
    adx_trend_threshold = 25
    required_indicators = ['bollinger_upper', 'bollinger_lower', 'adx', 'plus_di', 'minus_di', 'close']
    if current_row[required_indicators].isna().any(): return 'hold'
    bb_upper = current_row['bollinger_upper']; bb_lower = current_row['bollinger_lower']
    adx = current_row['adx']; plus_di = current_row['plus_di']; minus_di = current_row['minus_di']
    close = current_row['close']
    is_trending = adx > adx_trend_threshold; is_uptrend_dmi = plus_di > minus_di; is_downtrend_dmi = minus_di > plus_di
    idx = data.index.get_loc(current_row.name);
    if idx == 0: return 'hold'
    prev_row = data.iloc[idx - 1]
    # Need to check if prev_row indicators are NaN too
    if pd.isna(prev_row['close']) or pd.isna(prev_row['bollinger_upper']) or pd.isna(prev_row['bollinger_lower']): return 'hold'
    prev_close_inside_bands = (prev_row['close'] < prev_row['bollinger_upper']) and (prev_row['close'] > prev_row['bollinger_lower'])
    long_condition = (close > bb_upper) and prev_close_inside_bands and is_trending and is_uptrend_dmi
    short_condition = (close < bb_lower) and prev_close_inside_bands and is_trending and is_downtrend_dmi
    if long_condition: return 'buy'
    elif short_condition: return 'sell'
    else: return 'hold'


# --- NEW STRATEGY FUNCTION (ICT Turtle Soup) ---
def strategy_ict_turtle_soup(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    """
    Strategy based on ICT Turtle Soup concept.
    Looks for a liquidity grab above/below a previous HTF high/low,
    followed by a market structure shift (break of recent swing high/low).

    Requires pre-calculated columns:
    'htf_high_shifted', 'htf_low_shifted', 'mss_high_shifted', 'mss_low_shifted'
    """
    signal = 'hold'
    params = params or {} # Ensure params is a dict

    # Get required parameters from the passed dictionary
    bar_length = params.get('bar_length', 20) # Default if not passed
    mss_offset = params.get('mss_offset', 10)
    breakout_method = params.get('breakout_method', 'Wick') # 'Wick' or 'Close'

    # Required columns check
    required_columns = [
        'high', 'low', 'close', 'open',
        'htf_high_shifted', 'htf_low_shifted',
        'mss_high_shifted', 'mss_low_shifted'
    ]
    if current_row[required_columns].isna().any():
        # logger.debug(f"Turtle Soup - NaN check failed @ {current_row.name}")
        return 'hold'

    # Get current index location safely
    try:
        idx_loc = data.index.get_loc(current_row.name)
    except KeyError:
        logger.warning(f"Could not find index {current_row.name} in DataFrame for Turtle Soup.")
        return 'hold'

    # Ensure we have enough history for lookbacks
    if idx_loc < bar_length: # Need at least bar_length history
        # logger.debug(f"Turtle Soup - Not enough history @ {current_row.name} (idx={idx_loc}, bar_length={bar_length})")
        return 'hold'

    # Get current and previous data points needed
    current_high = current_row['high']
    current_low = current_row['low']
    current_close = current_row['close']
    # open_ = current_row['open'] # Not used in this simplified logic, keep if needed later
    prev_mss_high = current_row['mss_high_shifted'] # Already shifted in preprocessing
    prev_mss_low = current_row['mss_low_shifted']   # Already shifted in preprocessing

    # Determine the price to use for checking breaks based on method
    break_high_price = current_high if breakout_method == 'Wick' else current_close
    break_low_price = current_low if breakout_method == 'Wick' else current_close

    # --- Logic ---
    # Check for recent liquidity grabs within the HTF lookback period *ending before the current bar*
    lookback_start_index = idx_loc - bar_length
    if lookback_start_index < 0: lookback_start_index = 0
    lookback_end_index = idx_loc # Python slicing is exclusive at the end

    recent_df_slice = data.iloc[lookback_start_index:lookback_end_index]

    # Check if any high in the lookback period went above its corresponding prev HTF high
    # Use .values to avoid potential index alignment issues if slice is very small
    recent_high_liq_grab = (recent_df_slice['high'].values > recent_df_slice['htf_high_shifted'].values).any()

    # Check if any low in the lookback period went below its corresponding prev HTF low
    recent_low_liq_grab = (recent_df_slice['low'].values < recent_df_slice['htf_low_shifted'].values).any()

    # Sell Condition: Recent High Liq Grab occurred *AND* MSS broke down on the *current* bar
    if recent_high_liq_grab and break_low_price < prev_mss_low:
        signal = 'sell'
        # logger.debug(f"Turtle Soup SELL signal @ {current_row.name}: HighGrab={recent_high_liq_grab}, LowBreak={break_low_price} < PrevMSSLow={prev_mss_low}")


    # Buy Condition: Recent Low Liq Grab occurred *AND* MSS broke up on the *current* bar
    elif recent_low_liq_grab and break_high_price > prev_mss_high:
        signal = 'buy'
        # logger.debug(f"Turtle Soup BUY signal @ {current_row.name}: LowGrab={recent_low_liq_grab}, HighBreak={break_high_price} > PrevMSSHigh={prev_mss_high}")


    return signal
# --- END OF NEW STRATEGY ---


# --- (Keep generate_consolidated_html_report function as is) ---
def calculate_tr(df: pd.DataFrame) -> pd.Series:
    """Calculates True Range."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    # tr.fillna(0, inplace=True) # First value will be NaN, fill with 0 or handle later
    return tr

def calculate_kalman_filter_iterative(src_series: pd.Series, tr_series: pd.Series) -> pd.Series:
    """
    Iteratively calculates the Ehlers Kalman Filter value.
    Handles potential NaNs during iteration.
    """
    # Create empty series with the same index
    value1 = pd.Series(index=src_series.index, dtype=float)
    value2 = pd.Series(index=src_series.index, dtype=float)
    filtered_series = pd.Series(index=src_series.index, dtype=float)
    
    # Find the first index where both src and tr are valid
    first_valid_src_index = src_series.first_valid_index()
    first_valid_tr_index = tr_series.first_valid_index()
    
    if first_valid_src_index is None or first_valid_tr_index is None:
        logger.warning("Kalman Filter: Not enough valid data in source or TR series.")
        return filtered_series # Return empty series

    # Determine the actual start index for calculation (need src[i] and src[i-1])
    start_index = max(first_valid_src_index, first_valid_tr_index)
    start_loc = src_series.index.get_loc(start_index)

    if start_loc == 0: # Cannot calculate diff on the very first point
        logger.warning("Kalman Filter: Cannot calculate on the first data point. Shifting start.")
        if len(src_series.index) > 1:
             start_loc = 1
             start_index = src_series.index[start_loc]
        else:
             logger.warning("Kalman Filter: Only one data point available.")
             # Return series with the source value? Or NaN? Let's return NaN for filter.
             return filtered_series

    # Initialize the state at the location *before* the loop start
    init_loc = start_loc - 1
    init_index = src_series.index[init_loc]
    
    value1.iloc[init_loc] = 0.0 # Initial state guess
    value2.iloc[init_loc] = tr_series.iloc[init_loc] if pd.notna(tr_series.iloc[init_loc]) else 0.0 # Initial state guess
    filtered_series.iloc[init_loc] = src_series.iloc[init_loc] if pd.notna(src_series.iloc[init_loc]) else 0.0 # Start with source

    logger.debug(f"Kalman Filter Init @ {init_index}: v1={value1.iloc[init_loc]:.4f}, v2={value2.iloc[init_loc]:.4f}, filt={filtered_series.iloc[init_loc]:.4f}")

    # Iterate from the determined start location
    for i in range(start_loc, len(src_series.index)):
        current_index = src_series.index[i]
        prev_index = src_series.index[i-1]

        src_curr = src_series.loc[current_index]
        src_prev = src_series.loc[prev_index]
        tr_curr = tr_series.loc[current_index]
        
        # Get previous state values safely, handling potential NaNs from init/previous steps
        prev_value1 = value1.loc[prev_index] if pd.notna(value1.loc[prev_index]) else 0.0
        prev_value2 = value2.loc[prev_index] if pd.notna(value2.loc[prev_index]) else 0.0
        prev_filtered = filtered_series.loc[prev_index] if pd.notna(filtered_series.loc[prev_index]) else src_prev # Fallback to prev source

        # Check for NaNs in current inputs needed for this step's calculation
        if pd.isna(src_curr) or pd.isna(src_prev) or pd.isna(tr_curr):
             # Carry forward previous calculated values if inputs are missing
             value1.loc[current_index] = prev_value1
             value2.loc[current_index] = prev_value2
             filtered_series.loc[current_index] = prev_filtered
             # logger.debug(f"Kalman Filter Skip NaN @ {current_index}")
             continue

        # Calculate value1, value2
        v1 = 0.2 * (src_curr - src_prev) + 0.8 * prev_value1
        v2 = 0.1 * tr_curr + 0.8 * prev_value2
        value1.loc[current_index] = v1
        value2.loc[current_index] = v2

        # Calculate lambda and alpha
        lambda_val = abs(v1 / v2) if v2 != 0 else 0
        # Handle potential negative values under sqrt for lambda_val near 0
        lambda_pow4 = lambda_val**4
        lambda_pow2 = lambda_val**2
        sqrt_term = lambda_pow4 + 16 * lambda_pow2
        alpha = (-lambda_pow2 + np.sqrt(max(0, sqrt_term))) / 8.0 # Ensure sqrt term is non-negative
        alpha = max(0, min(1, alpha)) # Clamp alpha between 0 and 1 for stability

        # Calculate filtered value (value3)
        v3 = alpha * src_curr + (1 - alpha) * prev_filtered
        filtered_series.loc[current_index] = v3
        # logger.debug(f"Kalman Filter Calc @ {current_index}: alpha={alpha:.4f}, v3={v3:.4f}")


    return filtered_series
# --- NEW STRATEGY FUNCTION (ZLEMA Kalman Cross) ---
def strategy_zlema_kalman_cross(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    """
    Trades on the crossover/crossunder of the fast and medium ZLEMA lines.
    Optionally uses a Kalman filter on the source data.
    Requires pre-calculated ZLEMA columns, e.g., 'zlema_8', 'zlema_21'.
    """
    params = params or {}
    # Get parameters
    show_cross = params.get('show_cross', True)
    period1 = params.get('period1', 8)
    period2 = params.get('period2', 21)

    # Check if signals are enabled
    if not show_cross:
        return 'hold'

    # Define required column names based on periods
    zlema_fast_col = f'zlema_{period1}'
    zlema_medium_col = f'zlema_{period2}'

    # Check if required columns exist and have valid data
    required_cols = [zlema_fast_col, zlema_medium_col]
    if not all(col in current_row.index for col in required_cols):
        # logger.warning(f"ZLEMA columns missing: {required_cols}") # Can be noisy
        return 'hold'

    if current_row[required_cols].isna().any():
        return 'hold'

    # Get index location safely
    try:
        idx_loc = data.index.get_loc(current_row.name)
    except KeyError:
        logger.warning(f"Could not find index {current_row.name} in DataFrame for ZLEMA Cross.")
        return 'hold'

    # Need previous bar for crossover check
    if idx_loc == 0:
        return 'hold'

    prev_row = data.iloc[idx_loc - 1]

    # Check previous row validity
    if prev_row[required_cols].isna().any():
        return 'hold'

    # Current values
    zlema1_curr = current_row[zlema_fast_col]
    zlema2_curr = current_row[zlema_medium_col]

    # Previous values
    zlema1_prev = prev_row[zlema_fast_col]
    zlema2_prev = prev_row[zlema_medium_col]

    # Crossover/Crossunder Logic
    crossed_above = zlema1_curr > zlema2_curr and zlema1_prev <= zlema2_prev
    crossed_below = zlema1_curr < zlema2_curr and zlema1_prev >= zlema2_prev

    if crossed_above:
        return 'buy'
    elif crossed_below:
        return 'sell'
    else:
        return 'hold'
# --- END OF ZLEMA STRATEGY ---
def calculate_zlema(series: pd.Series, period: int) -> pd.Series:
    """Calculates Zero Lag Exponential Moving Average (ZLEMA)."""
    if period < 1: return pd.Series(index=series.index, dtype=float) # Return empty if period invalid
    lag = int(round((period - 1) / 2)) # Calculate lag, ensure integer
    
    # Calculate ema_data = src + (src - src[lag])
    # Need to handle NaNs introduced by shift
    shifted_series = series.shift(lag)
    ema_data = series + (series - shifted_series)
    
    # Calculate EMA of ema_data
    # Use adjust=False for behavior closer to TradingView's default EMA
    zl = ema_data.ewm(span=period, adjust=False).mean()
    return zl

def generate_consolidated_html_report(
    all_metrics: Dict[str, Optional[pd.DataFrame]],
    all_plot_uris: Dict[str, Dict[str, Optional[str]]],
    all_best_strategies: Dict[str, Optional[Dict]],
    output_file: Path):
    # (Keep existing report generation code)
    logger.info(f"Generating consolidated HTML report at {output_file}...")
    timeframes = list(all_metrics.keys())
    if not timeframes: logger.warning("No metrics data found to generate consolidated report."); return
    html_start = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Consolidated Backtest Report</title><style>body{{font-family:'Segoe UI',sans-serif;margin:20px;background-color:#f4f4f4;color:#333;line-height:1.6;}}.container{{max-width:1400px;margin:auto;background-color:#fff;padding:30px;box-shadow:0 2px 15px rgba(0,0,0,0.1);border-radius:8px}}h1,h2,h3{{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:8px;margin-top:35px;margin-bottom:20px}}h1{{text-align:center;margin-bottom:25px;border-bottom:none;}}.tab-buttons button {{ background-color: #eee; border: 1px solid #ccc; padding: 10px 15px; cursor: pointer; transition: background-color 0.3s; border-radius: 4px 4px 0 0; margin-right: 2px; font-size: 1em; }}.tab-buttons button.active {{ background-color: #3498db; color: white; border-bottom: 1px solid #3498db; }}.tab-content {{ display: none; padding: 20px; border: 1px solid #ccc; border-top: none; border-radius: 0 0 4px 4px; background-color: #fff; animation: fadeIn 0.5s; }}.tab-content.active {{ display: block; }}@keyframes fadeIn {{ from {{opacity: 0;}} to {{opacity: 1;}} }}table.performance-table{{border-collapse:collapse;width:100%;margin-bottom:25px;font-size:0.9em;}}th,td{{border:1px solid #ddd;padding:10px 12px;text-align:right;}}th{{background-color:#3498db;color:white;text-align:center;font-weight:600;}}tr:nth-child(even){{background-color:#f9f9f9;}} tr:hover {{background-color: #f1f1f1;}}.metric-card{{border:1px solid #e0e0e0;border-radius:6px;padding:20px;margin-bottom:25px;background-color:#fff;box-shadow:0 1px 4px rgba(0,0,0,0.06);}}.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:25px;}}img{{max-width:100%;height:auto;border-radius:4px;margin-top:15px;border:1px solid #eee;display:block;margin-left:auto;margin-right:auto;background-color:#fafafa;}}.positive{{color:#27ae60;font-weight:bold;}} .negative{{color:#c0392b;font-weight:bold;}} .neutral{{color:#7f8c8d;}}.timestamp{{text-align:center;color:#7f8c8d;margin-bottom:35px;font-size:0.9em;}}details{{margin-top:15px;border:1px solid #eee;padding:10px;border-radius:4px;background-color:#f9f9f9;}}details>summary{{cursor:pointer;font-weight:bold;color:#3498db;padding:5px;display:inline-block;}}details[open]>summary {{ border-bottom: 1px solid #ddd; margin-bottom: 10px; }}</style></head><body><div class="container"><h1>Consolidated Multi-Timeframe Backtest Report</h1><p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p><div class="tab-buttons">"""
    for i, tf in enumerate(timeframes): active_class = 'active' if i == 0 else ''; html_start += f'<button class="tab-button {active_class}" onclick="openTab(event, \'{tf}\')">{tf}</button>'
    html_start += "</div>"; html_content = ""
    for i, tf in enumerate(timeframes):
        active_class = 'active' if i == 0 else ''; html_content += f'<div id="{tf}" class="tab-content {active_class}">\n<h2>Results for Timeframe: {tf}</h2>\n'; metrics_df = all_metrics.get(tf); plot_uris = all_plot_uris.get(tf, {}); best_strat_info = all_best_strategies.get(tf)
        if best_strat_info: bs_name=best_strat_info['name']; bs_metric=best_strat_info['metric'].replace('_',' ').title(); bs_score=best_strat_info['score']; score_format = {'total_pnl_points': '{:,.2f}', 'avg_win_points': '{:,.2f}', 'avg_loss_points': '{:,.2f}', 'win_rate': '{:.2f}%', 'profit_factor': '{:.2f}', 'expectancy_points': '{:.2f}', 'max_drawdown_points': '{:,.2f}'}.get(best_strat_info['metric'], '{:.2f}'); bs_score_str = score_format.format(bs_score) if pd.notna(bs_score) else 'N/A'; html_content += f'<div class="metric-card" style="border-left: 5px solid #27ae60;"><p>Best Strategy ({tf}, based on max. \'{bs_metric}\'): <strong>{bs_name}</strong> (Score: {bs_score_str})</p></div>'
        else: html_content += '<div class="metric-card"><p>Could not determine best strategy (requires >= 5 trades).</p></div>'
        html_content += "<h3>Performance Summary</h3><div class=\"metric-card\">";
        if metrics_df is not None and not metrics_df.empty:
             try:
                metrics_to_format = {'total_pnl_points': '{:,.2f}', 'avg_win_points': '{:,.2f}', 'avg_loss_points': '{:,.2f}', 'win_rate': '{:.2f}%', 'profit_factor': '{:.2f}', 'expectancy_points': '{:.2f}', 'max_drawdown_points': '{:,.2f}'}; metrics_display_df = metrics_df.copy()
                for col, fmt in metrics_to_format.items():
                     if col in metrics_display_df.columns: metrics_display_df[col] = metrics_display_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')
                if 'total_trades' in metrics_display_df.columns: metrics_display_df['total_trades'] = metrics_display_df['total_trades'].astype(int); best_strategy_name = best_strat_info['name'] if best_strat_info else None
                def highlight_best(s): return ['background-color: #e0ffe0' if s.name == best_strategy_name else '' for v in s]
                metrics_table_html = metrics_display_df.style.apply(highlight_best, axis=1).to_html(classes='performance-table', border=1, justify='right'); html_content += metrics_table_html
             except Exception as e: logger.error(f"HTML metrics table error for {tf}: {e}"); html_content += "<p>Error displaying metrics table.</p>"
        else: html_content += "<p>No metrics data available.</p>"
        html_content += '<p style="font-size: 0.8em; color: #555;text-align:center;">Note: PnL & Drawdown in points.</p></div>'
        html_content += "<h3>Equity Curves (Cumulative PnL Points)</h3><div class=\"metric-card\">"; equity_uri = plot_uris.get('equity_curves', ''); html_content += f'<img src="{equity_uri}" alt="Equity Curves {tf}">' if equity_uri else "<p>Equity curve plot unavailable.</p>"; html_content += "</div>"
        html_content += "<h3>Individual Strategy Analysis</h3><div class=\"metric-grid\">"; strategies_in_metrics = metrics_df.index if metrics_df is not None else []
        for name in strategies_in_metrics:
             if metrics_df is None or name not in metrics_df.index or pd.isna(metrics_df.loc[name, 'total_trades']) or metrics_df.loc[name, 'total_trades']==0: html_content += f'<div class="metric-card"><h4>{name}</h4><p>No trades or error.</p></div>'; continue
             metrics=metrics_df.loc[name]; win_rate_str=f"{metrics.get('win_rate',0):.2f}%"; pf_str=f"{metrics.get('profit_factor',0):.2f}"; pnl_str=f"{metrics.get('total_pnl_points',0):,.2f}"; pnl_dist_uri = plot_uris.get(f'{name}_pnl_distribution', ''); cum_pnl_uri = plot_uris.get(f'{name}_cumulative_pnl', ''); best_style = ' style="border-left: 5px solid #27ae60;"' if name == best_strategy_name else ''; html_content += f"""<div class="metric-card"{best_style}><h4>{name}{' (Best)' if name == best_strategy_name else ''}</h4><p><strong>Total Trades:</strong> {int(metrics.get('total_trades',0))}</p><p><strong>Win Rate:</strong> {win_rate_str}</p><p><strong>Profit Factor:</strong> {pf_str}</p><p><strong>Total PnL (Points):</strong> {pnl_str}</p><details><summary>Show Plots ({name})</summary>{'<img src="' + pnl_dist_uri + '" alt="' + name + ' PnL Dist">' if pnl_dist_uri else '<p>PnL Distribution plot unavailable.</p>'}{'<img src="' + cum_pnl_uri + '" alt="' + name + ' Cum PnL">' if cum_pnl_uri else '<p>Cumulative PnL plot unavailable.</p>'}</details></div>"""
        html_content += "</div>"; html_content += "</div>\n"
    html_end = """</div> <script>function openTab(evt, timeframeName) { var i, tabcontent, tablinks; tabcontent = document.getElementsByClassName("tab-content"); for (i = 0; i < tabcontent.length; i++) { tabcontent[i].style.display = "none"; tabcontent[i].classList.remove("active"); } tablinks = document.getElementsByClassName("tab-button"); for (i = 0; i < tablinks.length; i++) { tablinks[i].classList.remove("active"); } document.getElementById(timeframeName).style.display = "block"; document.getElementById(timeframeName).classList.add("active"); evt.currentTarget.classList.add("active"); }</script></body></html>"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f: f.write(html_start + html_content + html_end)
        logger.info(f"Consolidated HTML report saved successfully to {output_file}")
    except Exception as e: logger.error(f"Failed to write consolidated HTML report: {e}", exc_info=True)
# --- Add this function to main_workflow.py ---
# In main_workflow.py (add this function)
# In main_workflow.py
import math # Add this import at the top of the file

# --- Replace the previous strategy_voting_ensemble function with this ---

def strategy_voting_ensemble(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
    """
    Generates signals based on voting:
    - ENTRY: Enter Long/Short if >= 'min_votes_entry' strategies agree.
    - EXIT: Exit Long/Short if >= 'exit_vote_percentage' strategies vote opposite.
           Exit is triggered by returning the opposing signal (Signal Reversal).

    Requires constituent strategy signals in 'current_row' and current 'state' dict.
    """
    params = params or {}
    state = state or {'active_trade': False, 'position': ''} # Default state if first call

    # --- Parameters ---
    min_votes_entry = params.get('min_votes_entry', 3) 
    constituent_strategies = params.get('constituent_strategies', []) 
    exit_vote_percentage = params.get('exit_vote_percentage', 0.40) # 40%

    if not constituent_strategies:
        logger.warning("Voting_Ensemble strategy configured with no constituent_strategies.")
        return 'hold'
        
    total_strategies = len(constituent_strategies)
    if total_strategies == 0: return 'hold' # Avoid division by zero
        
    # Calculate minimum opposing votes needed to exit (round up)
    min_opposing_votes_exit = math.ceil(total_strategies * exit_vote_percentage) 

    buy_votes = 0
    sell_votes = 0

    # --- Count Votes ---
    for name in constituent_strategies:
        signal_col = f'{name}_signal'.lower() 
        if signal_col in current_row.index: 
            signal = current_row[signal_col]
            if signal == 'buy':
                buy_votes += 1
            elif signal == 'sell':
                sell_votes += 1
        # else: logger.debug(f"Voting_Ensemble: Signal column '{signal_col}' not found.")

    # --- Logic ---
    current_position = state.get('position', '') # Get current position from state

    # 1. Check EXIT Condition (only if currently in a trade)
    if current_position == 'long':
        if sell_votes >= min_opposing_votes_exit:
            logger.debug(f"Voting EXIT LONG @ {current_row.name}: Sells={sell_votes} >= Threshold={min_opposing_votes_exit}")
            return 'sell' # Signal reversal to exit long
    elif current_position == 'short':
        if buy_votes >= min_opposing_votes_exit:
            logger.debug(f"Voting EXIT SHORT @ {current_row.name}: Buys={buy_votes} >= Threshold={min_opposing_votes_exit}")
            return 'buy' # Signal reversal to exit short

    # 2. Check ENTRY Condition (only if NOT currently in a trade)
    #    Also prevents immediate re-entry on the same bar an exit signal was generated above
    signal_to_return = 'hold' # Default if no entry/exit
    if not state.get('active_trade', False): 
        # Prioritize Buy slightly if both thresholds met simultaneously (unlikely)
        if buy_votes >= min_votes_entry:
            signal_to_return = 'buy'
            # logger.debug(f"Voting ENTRY BUY @ {current_row.name}: Buys={buy_votes} >= Threshold={min_votes_entry}")
        elif sell_votes >= min_votes_entry:
            signal_to_return = 'sell'
            # logger.debug(f"Voting ENTRY SELL @ {current_row.name}: Sells={sell_votes} >= Threshold={min_votes_entry}")
            
    return signal_to_return

# --- END of modified strategy_voting_ensemble definition ---
# --- END of strategy_voting_ensemble definition ---
def strategy_channel_breakout(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    """
    Trades on breakouts of the highest high or lowest low over a lookback period.
    Triggers buy if current high > previous highest high.
    Triggers sell if current low < previous lowest low.
    Relies on engine SL/TP/TSL for exits.

    Requires pre-calculated columns like 'highest_5_shifted', 'lowest_5_shifted'
    (where 5 is the 'length' parameter).
    """
    signal = 'hold'
    params = params or {}
    length = params.get('length', 5) # Get length from params, default 5

    # Define column names based on the length parameter
    highest_col = f'highest_{length}_shifted'
    lowest_col = f'lowest_{length}_shifted'

    # Check if required pre-calculated columns exist and have valid data
    if highest_col not in current_row or lowest_col not in current_row or \
       pd.isna(current_row[highest_col]) or pd.isna(current_row[lowest_col]):
        # logger.debug(f"ChannelBreakout - Missing/NaN prerequisites @ {current_row.name}")
        return 'hold' # Cannot determine signal without lookback data

    # Get necessary values from the current row
    prev_upBound = current_row[highest_col]
    prev_downBound = current_row[lowest_col]
    current_high = current_row['high']
    current_low = current_row['low']

    # --- Entry Logic ---
    # Long entry: Current high breaks above the highest high of the previous 'length' bars
    if current_high > prev_upBound:
        signal = 'buy'
        # logger.debug(f"ChannelBreakout BUY signal @ {current_row.name}: High={current_high} > PrevHigh={prev_upBound}")

    # Short entry: Current low breaks below the lowest low of the previous 'length' bars
    elif current_low < prev_downBound:
        signal = 'sell'
        # logger.debug(f"ChannelBreakout SELL signal @ {current_row.name}: Low={current_low} < PrevLow={prev_downBound}")

    return signal
# --- End of strategy definition ---
# --- Main Workflow Function ---
# --- Main Workflow Function ---
def run_full_backtest_workflow(data_paths: Dict[str, str], output_base_dir: str = 'backtest_runs'):
    """
    Runs backtest, collects results, and generates a single consolidated report.
    """
    # --- Set up base output directory ---
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_path = Path(output_base_dir) / f'run_{run_timestamp}' # Unique run folder
    base_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output base directory: {base_path}")


    # --- Dictionaries to collect results from each timeframe ---
    all_metrics_dfs: Dict[str, Optional[pd.DataFrame]] = {}
    all_plot_uris: Dict[str, Dict[str, Optional[str]]] = {}
    all_best_strategies: Dict[str, Optional[Dict]] = {}
    all_raw_results: Dict[str, Optional[pd.DataFrame]] = {}

    # --- Define Strategies and their configurations ---
    # Default parameters shared across strategies (can be overridden)
    default_sl_tp = {"sl_atr_mult": 1.5, "tp_atr_mult": 2.0}
    default_tsl = {"use_trailing_sl": True, "trailing_sl_atr_mult": 1.5} 
    all_strategy_names = [ # List all your individual strategy function names here
        'EMA_Crossover', 'RSI_Threshold', 'MACD_Cross', 'BB_Breakout', 
        'Combined_Mom', 'AI_Powered_v6', 'BB_ADX_Trend', 'Rsi_confirmed', 
        'Ema_crossover_filtered', 'Consolidated_Simp', 'ZLEMA_Kalman_Cross', 
        'ICT_Turtle_Soup', 'Channel_Breakout'
    ]
    # Combine strategy function with its specific parameters
    strategies_config = {
        # --- Keep all your existing strategy definitions ---
        'EMA_Crossover': {"function": strategy_ema_crossover, "params": {**default_sl_tp, **default_tsl}},
        'RSI_Threshold': {"function": strategy_rsi_threshold, "params": {**default_sl_tp, **default_tsl}},
        'MACD_Cross': {"function": strategy_macd_cross, "params": {**default_sl_tp, **default_tsl}},
        'BB_Breakout': {"function": strategy_bb_squeeze_breakout, "params": {**default_sl_tp, **default_tsl}},
        'Combined_Mom': {"function": strategy_combined_momentum, "params": {**default_sl_tp, **default_tsl}},
        'AI_Powered_v6': {"function": strategy_ai_powered_v6, "params": {**default_sl_tp, **default_tsl}},
        'BB_ADX_Trend': {"function": strategy_bb_adx_trend, "params": {**default_sl_tp, **default_tsl}},
        'Rsi_confirmed':{"function": strategy_rsi_confirmed, "params": {**default_sl_tp, **default_tsl}},
        'Ema_crossover_filtered':{"function": strategy_ema_crossover_filtered, "params": {**default_sl_tp, **default_tsl}},
        'Consolidated_Simp': {"function": strategy_consolidated_simplified, "params": {**default_sl_tp, **default_tsl}},
        'ZLEMA_Kalman_Cross': {
            "function": strategy_zlema_kalman_cross,
            "params": { "period1": 8, "period2": 21, "period3": 55, "enable_kalman": "ON", "show_cross": True, **default_sl_tp, **default_tsl }
        },
        'ICT_Turtle_Soup': {
            "function": strategy_ict_turtle_soup,
            "params": { "mss_offset": 10, "breakout_method": "Wick", "sl_atr_mult": 3.5, "tp_atr_mult": 3.5 * 0.9, "use_trailing_sl": True, "trailing_sl_atr_mult": 2.0 } # bar_length added dynamically
        },
        'Channel_Breakout': {
            "function": strategy_channel_breakout,
            "params": { "length": 5, **default_sl_tp, **default_tsl }
        },
        # --- ADD Voting Ensemble Strategy ---
        # IMPORTANT: Place this entry LAST if relying on dict iteration order 
        # for the engine to process other signals first (safer in Python 3.7+)
        'Voting_Ensemble': {
            "function": strategy_voting_ensemble,
            "params": {
                # --- Configure the Voting Strategy ---
                "min_votes_entry": 3,   # Enter on >= 3 votes
                "exit_vote_percentage": 0.40, # Exit on >= 40% opposing votes
                "constituent_strategies": all_strategy_names, # Use all other strategies
                # --- SL/TP/TSL for the Ensemble Strategy ---
                **default_sl_tp, 
                **default_tsl  
            }
        
        }
    }
    strategy_names = list(strategies_config.keys()) # Get names from config
 
    # --- Loop through each timeframe ---
    for timeframe, data_path_str in data_paths.items():
        logger.info(f"\n===== Processing Timeframe: {timeframe} =====")
        data_path = Path(data_path_str)
        # Per-timeframe output dir (e.g., for detailed results)
        timeframe_output_dir = base_path / timeframe
        timeframe_output_dir.mkdir(parents=True, exist_ok=True)

        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}. Skipping.")
            continue

        try:
            # --- 1. Load Data ---
            logger.info(f"Loading data from {data_path}")
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            
            # --- Check and Convert Index ---
            if not isinstance(data.index, pd.DatetimeIndex):
                logger.warning(f"Index for {timeframe} is not DatetimeIndex. Attempting conversion.")
                data.index = pd.to_datetime(data.index)
            data.index.name = 'datetime' # Ensure index name is set

            # --- !!! PREPROCESSING Section (Consolidated) !!! ---
            logger.info(f"Preprocessing data for {timeframe} strategies...")

            # --- Turtle Soup Preprocessing ---
            try: 
                current_tf_minutes = int(timeframe.replace('min', ''))
            except ValueError: 
                logger.error(f"Could not determine minutes from timeframe string: {timeframe}. Using default 1.")
                current_tf_minutes = 1
            
            htf_minutes = 60 # Example HTF
            mss_offset = strategies_config.get('ICT_Turtle_Soup', {}).get('params', {}).get('mss_offset', 10) # Get from config if possible
            
            if current_tf_minutes > 0 and htf_minutes >= current_tf_minutes and htf_minutes % current_tf_minutes == 0: 
                bar_length = htf_minutes // current_tf_minutes
                logger.info(f"Calculated bar_length for {timeframe}: {bar_length} (HTF={htf_minutes}min)")
            else: 
                bar_length = 20 # Fallback default
                logger.warning(f"Timeframes incompatible. Using default bar_length: {bar_length} for Turtle Soup.")
                
            # Add/Update bar_length in params dynamically
            if 'ICT_Turtle_Soup' in strategies_config: 
                 strategies_config['ICT_Turtle_Soup']['params']['bar_length'] = bar_length
                 
            data['htf_high_shifted'] = data['high'].rolling(window=bar_length, min_periods=bar_length).max().shift(1)
            data['htf_low_shifted'] = data['low'].rolling(window=bar_length, min_periods=bar_length).min().shift(1)
            data['mss_high_shifted'] = data['high'].rolling(window=mss_offset, min_periods=mss_offset).max().shift(1)
            data['mss_low_shifted'] = data['low'].rolling(window=mss_offset, min_periods=mss_offset).min().shift(1)
            # --- End Turtle Soup Preprocessing ---

            # --- ZLEMA / Kalman Preprocessing ---
            if 'ZLEMA_Kalman_Cross' in strategies_config: # Only run if strategy is configured
                logger.info(f"Calculating ZLEMA/Kalman prerequisites for {timeframe}...")
                zlema_kalman_params = strategies_config['ZLEMA_Kalman_Cross']['params']
                zlema_p1 = zlema_kalman_params.get('period1', 8)
                zlema_p2 = zlema_kalman_params.get('period2', 21)
                zlema_p3 = zlema_kalman_params.get('period3', 55)
                kalman_enabled = zlema_kalman_params.get('enable_kalman', "ON") == "ON"

                data['hlc3'] = (data['high'] + data['low'] + data['close']) / 3.0
                source_col_name = 'hlc3' 

                if kalman_enabled:
                    logger.info(f"Calculating TR and Kalman Filter for {timeframe}...")
                    data['tr'] = calculate_tr(data)
                    # FIXED FutureWarning:
                    data['tr'] = data['tr'].bfill() # Backfill first NaN without inplace
                    data['kalman_src'] = calculate_kalman_filter_iterative(data[source_col_name], data['tr'])
                    final_src_col_name = 'kalman_src'
                    logger.info(f"Finished Kalman Filter calculation for {timeframe}.")
                else:
                    final_src_col_name = source_col_name 
                    logger.info(f"Kalman Filter disabled, using '{source_col_name}' as source.")

                if final_src_col_name not in data.columns or data[final_src_col_name].isnull().all():
                     logger.error(f"Source column '{final_src_col_name}' is missing or all NaN after calculation. Cannot calculate ZLEMA.")
                else:
                     final_src_series = data[final_src_col_name]
                     data[f'zlema_{zlema_p1}'] = calculate_zlema(final_src_series, zlema_p1)
                     data[f'zlema_{zlema_p2}'] = calculate_zlema(final_src_series, zlema_p2)
                     data[f'zlema_{zlema_p3}'] = calculate_zlema(final_src_series, zlema_p3)
                     logger.info(f"Calculated ZLEMA lines ({zlema_p1}, {zlema_p2}, {zlema_p3}) for {timeframe}.")
            # --- End ZLEMA / Kalman Preprocessing ---
             # --- Channel Breakout Preprocessing (NEW) ---
            if 'Channel_Breakout' in strategies_config: # Only calculate if strategy is used
                logger.info(f"Calculating Channel Breakout prerequisites for {timeframe}...")
                cb_params = strategies_config['Channel_Breakout']['params']
                cb_length = cb_params.get('length', 5) # Get length from config
                
                # Calculate highest high and lowest low for the period
                highest_col_temp = f'highest_{cb_length}'
                lowest_col_temp = f'lowest_{cb_length}'
                data[highest_col_temp] = data['high'].rolling(window=cb_length, min_periods=cb_length).max()
                data[lowest_col_temp] = data['low'].rolling(window=cb_length, min_periods=cb_length).min()
                
                # Create the shifted versions needed by the strategy
                data[f'highest_{cb_length}_shifted'] = data[highest_col_temp].shift(1)
                data[f'lowest_{cb_length}_shifted'] = data[lowest_col_temp].shift(1)
                logger.info(f"Calculated Channel Breakout lines (Length={cb_length}) for {timeframe}.")
            # --- End Channel Breakout Preprocessing ---

            # --- Ensure ATR exists (CRITICAL) ---
            if 'atr' not in data.columns: # Check lowercase
                 logger.error(f"'atr' column not found in data for {timeframe} after loading/preprocessing. Backtest will likely fail on SL/TP/TSL. Please fix indicators.py.")
                 # Decide whether to continue or skip
                 # continue # Option 1: Skip this timeframe
                 # Option 2: Add a dummy ATR (will give bad results but might prevent crash)
                 data['atr'] = 1.0 
                 logger.warning("Added dummy 'atr' column with value 1.0. SL/TP/TSL results will be incorrect!")
            
           # --- Final NaN Drop (Consolidated) ---
            all_required_columns = set(['open', 'high', 'low', 'close', 'volume', 'atr']) # Base requirements
            # Add columns needed by specific strategies only if they are in the config
            if 'ICT_Turtle_Soup' in strategies_config: all_required_columns.update(['htf_high_shifted', 'htf_low_shifted', 'mss_high_shifted', 'mss_low_shifted'])
            if 'ZLEMA_Kalman_Cross' in strategies_config:
                 z_params = strategies_config['ZLEMA_Kalman_Cross']['params']; z_p1 = z_params.get('period1', 8); z_p2 = z_params.get('period2', 21); z_p3 = z_params.get('period3', 55)
                 all_required_columns.add('hlc3'); 
                 if z_params.get('enable_kalman', "ON") == "ON": all_required_columns.update(['tr', 'kalman_src'])
                 all_required_columns.update([f"zlema_{z_p1}", f"zlema_{z_p2}", f"zlema_{z_p3}"])
            # Add columns for Channel Breakout (NEW)
            if 'Channel_Breakout' in strategies_config:
                 cb_params = strategies_config['Channel_Breakout']['params']; cb_len = cb_params.get('length', 5)
                 all_required_columns.update([f"highest_{cb_len}_shifted", f"lowest_{cb_len}_shifted"])
            # Add required indicators for ALL other strategies being run
            all_required_columns.update(['ema_9', 'ema_21', 'ema_50', 'macd', 'macd_signal', 'macd_hist', 'rsi', 'plus_di', 'minus_di', 'adx', 'bollinger_upper', 'bollinger_lower', 'bollinger_bandwidth', 'vol_ma'])


            initial_rows = len(data)
            present_required_columns = [col for col in all_required_columns if col in data.columns]
            missing_indicator_cols = all_required_columns - set(data.columns)
            if missing_indicator_cols: logger.warning(f"Missing expected indicator columns for {timeframe}: {missing_indicator_cols}. NaN drop might be incomplete.")
            logger.info(f"Dropping rows with NaN in columns: {present_required_columns}")
            data.dropna(subset=present_required_columns, inplace=True)
            logger.info(f"Preprocessing: Dropped {initial_rows - len(data)} total rows with NaN values for {timeframe}.")
            if data.empty: logger.error(f"No data remaining after NaN drop for {timeframe}. Skipping backtest."); continue
            # --- END OF PREPROCESSING ---.")
            
            


            # --- 2. Run Backtest Engine ---
            logger.info(f"Initializing & Running backtester engine for {timeframe}...")
            backtester = EnhancedMultiStrategyBacktester(strategies_config) # Pass the config
            results_df = backtester.run_backtest(data) # Pass the preprocessed data

            # --- 3. Analyze Results ---
            # (Rest of the analysis and reporting code remains the same)
            if results_df is not None and not results_df.empty:
                all_raw_results[timeframe] = results_df # Store raw results
                detailed_csv_path = timeframe_output_dir / 'backtest_results_detailed.csv'
                results_df.to_csv(detailed_csv_path)
                logger.info(f"Saved detailed results for {timeframe} to {detailed_csv_path}")

                logger.info(f"Initializing analyzer for {timeframe}...")
                # Pass strategy_names derived from config
                analyzer = BacktestAnalyzerReporter(results_df, strategy_names)

                # --- 4. Collect Metrics and Plots ---
                logger.info(f"Collecting metrics and plot data for {timeframe}...")
                metrics_df = analyzer.get_performance_metrics_df()
                plot_uris = analyzer.generate_plot_data_uris()
                best_strat_info = analyzer.best_strategy_info

                all_metrics_dfs[timeframe] = metrics_df
                all_plot_uris[timeframe] = plot_uris
                all_best_strategies[timeframe] = best_strat_info

                logger.info(f"Finished analysis for {timeframe}.")
            else:
                logger.error(f"Backtest engine did not produce results for {timeframe}.")
                all_metrics_dfs[timeframe] = None
                all_plot_uris[timeframe] = {}
                all_best_strategies[timeframe] = None

        except Exception as e:
            logger.error(f"Workflow failed for timeframe {timeframe}: {e}", exc_info=True)
            all_metrics_dfs[timeframe] = None; all_plot_uris[timeframe] = {}; all_best_strategies[timeframe] = None

    # --- 5. Generate ONE Consolidated Report AFTER loop ---
    # (Keep existing report generation code)
    if any(df is not None for df in all_metrics_dfs.values()):
         consolidated_report_path = base_path / "consolidated_backtest_report.html"
         generate_consolidated_html_report(
             all_metrics=all_metrics_dfs,
             all_plot_uris=all_plot_uris,
             all_best_strategies=all_best_strategies,
             output_file=consolidated_report_path
         )
    else:
         logger.warning("No successful backtest results obtained. Skipping consolidated report generation.")


    # --- 6. Overall Console Summary ---
    # (Keep existing summary code)
    logger.info("\n===== Overall Timeframe Comparison (Console Summary) =====")
    for timeframe, metrics_data in all_metrics_dfs.items():
         print(f"\n--- {timeframe} Summary ---")
         if metrics_data is not None and not metrics_data.empty:
              metrics_data_sorted = metrics_data.sort_index()
              for strategy_name, strategy_metrics in metrics_data_sorted.iterrows():
                   pnl = strategy_metrics.get('total_pnl_points', 'N/A'); trades = strategy_metrics.get('total_trades', 'N/A')
                   win_rate = strategy_metrics.get('win_rate', 'N/A'); pf = strategy_metrics.get('profit_factor', 'N/A')
                   wr_str = f"{win_rate:.2f}%" if isinstance(win_rate, (int, float)) and pd.notna(win_rate) else 'N/A'
                   pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) and pd.notna(pf) else 'N/A'
                   pnl_str = f"{pnl:.2f}" if isinstance(pnl, (int, float)) and pd.notna(pnl) else 'N/A'
                   trade_str = str(int(trades)) if isinstance(trades, (int, float)) and pd.notna(trades) else 'N/A'
                   print(f"  Strategy: {strategy_name:<25} | Trades: {trade_str:<5} | Win Rate: {wr_str:<8} | Profit Factor: {pf_str:<5} | PnL Points: {pnl_str}")
         else: print(f"  No metrics calculated for {timeframe}.")


if __name__ == "__main__":
    # --- Configuration ---
    data_files_by_timeframe = {
        "3min": "/Users/shekhar/Desktop/BOT/smartapi-python-main/runs/20250406_165347/data/nifty_indicators_3min.csv",  # <--- ADJUST PATH
        "5min": "/Users/shekhar/Desktop/BOT/smartapi-python-main/runs/20250406_165347/data/nifty_indicators_5min.csv",  # <--- ADJUST PATH
        "15min": "/Users/shekhar/Desktop/BOT/smartapi-python-main/runs/20250406_165347/data/nifty_indicators_15min.csv", # <--- ADJUST PATH
    }

    # --- Check if files exist ---
    # (Keep existing file check code)
    valid_data_files = {tf: path_str for tf, path_str in data_files_by_timeframe.items() if Path(path_str).is_file()}
    missing_files = set(data_files_by_timeframe.keys()) - set(valid_data_files.keys())
    if missing_files: logger.warning(f"Data files missing for timeframes: {missing_files}. They will be skipped.")
    if not valid_data_files: logger.error("No valid data files found. Exiting."); sys.exit(1)


    # --- Execution ---
    run_full_backtest_workflow(
        data_paths=valid_data_files,
        output_base_dir='multi_timeframe_analysis_results' # Main output folder
    )

    logger.info("All backtesting workflows finished.")
    # --- Configuration ---
    # IMPORTANT: Define the paths to your different timeframe datasets
    # Make sure these CSV files contain all necessary base columns (OHLCV)
    # AND the required indicator columns calculated beforehand (EMAs, RSI, MACD, BBands, ADX, ATR, VolMA etc.)
    data_files_by_timeframe = {
        "3min": "/Users/shekhar/Desktop/BOT/smartapi-python-main/runs/20250406_165347/data/nifty_indicators_3min.csv",  # <--- ADJUST PATH
        "5min": "/Users/shekhar/Desktop/BOT/smartapi-python-main/runs/20250406_165347/data/nifty_indicators_5min.csv",  # <--- ADJUST PATH
        "15min": "/Users/shekhar/Desktop/BOT/smartapi-python-main/runs/20250406_165347/data/nifty_indicators_15min.csv", # <--- ADJUST PATH
    }

    # --- Check if files exist ---
    valid_data_files = {tf: path_str for tf, path_str in data_files_by_timeframe.items() if Path(path_str).is_file()}
    missing_files = set(data_files_by_timeframe.keys()) - set(valid_data_files.keys())
    if missing_files: logger.warning(f"Data files missing for timeframes: {missing_files}. They will be skipped.")
    if not valid_data_files: logger.error("No valid data files found. Exiting."); sys.exit(1)


    # --- Execution ---
    run_full_backtest_workflow(
        data_paths=valid_data_files,
        output_base_dir='multi_timeframe_analysis_results' # Main output folder
    )

    logger.info("All backtesting workflows finished.")