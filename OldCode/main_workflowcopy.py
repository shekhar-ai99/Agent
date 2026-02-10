from datetime import datetime
from typing import Dict, List, Optional
import math
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from backtester_engine import EnhancedMultiStrategyBacktester
from backtester_analyzer import BacktestAnalyzerReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main_workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enhanced Strategy Functions
def strategy_ema_crossover(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    params = params or {}
    adx_threshold = params.get('adx_threshold', 25)
    volume_multiplier = params.get('volume_multiplier', 1.5)
    required_indicators = ['ema_9', 'ema_21', 'adx', 'volume', 'vol_ma']
    if current_row[required_indicators].isna().any():
        return 'hold'
    idx = data.index.get_loc(current_row.name)
    if idx == 0:
        return 'hold'
    prev_row = data.iloc[idx-1]
    if current_row['adx'] < adx_threshold or current_row['volume'] < current_row['vol_ma'] * volume_multiplier:
        return 'hold'
    if current_row['ema_9'] > current_row['ema_21'] and prev_row['ema_9'] <= prev_row['ema_21']:
        return 'buy'
    elif current_row['ema_9'] < current_row['ema_21'] and prev_row['ema_9'] >= prev_row['ema_21']:
        return 'sell'
    return 'hold'

def strategy_rsi_threshold(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    params = params or {}
    rsi_oversold = params.get('rsi_oversold', 35)
    rsi_overbought = params.get('rsi_overbought', 65)
    adx_threshold = params.get('adx_threshold', 20)
    required_indicators = ['rsi', 'adx']
    if current_row[required_indicators].isna().any():
        return 'hold'
    idx = data.index.get_loc(current_row.name)
    if idx == 0:
        return 'hold'
    prev_row = data.iloc[idx-1]
    if current_row['adx'] < adx_threshold:
        return 'hold'
    if current_row['rsi'] > rsi_oversold and prev_row['rsi'] <= rsi_oversold:
        return 'buy'
    elif current_row['rsi'] < rsi_overbought and prev_row['rsi'] >= rsi_overbought:
        return 'sell'
    return 'hold'

def strategy_macd_cross(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    params = params or {}
    ema_trend_period = params.get('ema_trend_period', 50)
    required_indicators = ['macd', 'macd_signal', 'macd_hist', f'ema_{ema_trend_period}']
    if current_row[required_indicators].isna().any():
        return 'hold'
    idx = data.index.get_loc(current_row.name)
    if idx == 0:
        return 'hold'
    prev_row = data.iloc[idx-1]
    if current_row['close'] < current_row[f'ema_{ema_trend_period}'] and current_row['macd'] > current_row['macd_signal']:
        return 'hold'
    if current_row['macd'] > current_row['macd_signal'] and prev_row['macd'] <= prev_row['macd_signal'] and current_row['macd_hist'] > 0:
        return 'buy'
    elif current_row['macd'] < current_row['macd_signal'] and prev_row['macd'] >= prev_row['macd_signal'] and current_row['macd_hist'] < 0:
        return 'sell'
    return 'hold'

def strategy_bb_squeeze_breakout(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    params = params or {}
    adx_threshold = params.get('adx_threshold', 25)
    volume_multiplier = params.get('volume_multiplier', 1.5)
    required_indicators = ['bollinger_bandwidth', 'close', 'bollinger_upper', 'bollinger_lower', 'adx', 'volume', 'vol_ma']
    if current_row[required_indicators].isna().any():
        return 'hold'
    idx = data.index.get_loc(current_row.name)
    squeeze_lookback = params.get('squeeze_lookback', 20)
    if idx < squeeze_lookback:
        return 'hold'
    prev_rows = data.iloc[idx-squeeze_lookback:idx]
    bandwidth_threshold = prev_rows['bollinger_bandwidth'].quantile(0.15)
    prev_bandwidth = data.iloc[idx-1]['bollinger_bandwidth']
    if pd.isna(prev_bandwidth):
        return 'hold'
    in_squeeze_prev = prev_bandwidth < bandwidth_threshold
    breaking_out_up = current_row['close'] > current_row['bollinger_upper']
    breaking_out_down = current_row['close'] < current_row['bollinger_lower']
    if in_squeeze_prev and breaking_out_up and current_row['adx'] > adx_threshold and current_row['volume'] > current_row['vol_ma'] * volume_multiplier:
        return 'buy'
    elif in_squeeze_prev and breaking_out_down and current_row['adx'] > adx_threshold and current_row['volume'] > current_row['vol_ma'] * volume_multiplier:
        return 'sell'
    return 'hold'

def strategy_combined_momentum(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    params = params or {}
    rsi_confirm_buy = params.get('rsi_confirm_buy', 55)
    rsi_confirm_sell = params.get('rsi_confirm_sell', 45)
    required_indicators = ['ema_9', 'ema_21', 'rsi', 'bollinger_bandwidth']
    if current_row[required_indicators].isna().any():
        return 'hold'
    idx = data.index.get_loc(current_row.name)
    if idx == 0:
        return 'hold'
    prev_row = data.iloc[idx-1]
    bandwidth_threshold = data.iloc[idx-20:idx]['bollinger_bandwidth'].quantile(0.25) if idx >= 20 else np.inf
    if current_row['bollinger_bandwidth'] < bandwidth_threshold:
        return 'hold'
    ema_crossed_up = current_row['ema_9'] > current_row['ema_21'] and prev_row['ema_9'] <= prev_row['ema_21']
    ema_crossed_down = current_row['ema_9'] < current_row['ema_21'] and prev_row['ema_9'] >= prev_row['ema_21']
    if ema_crossed_up and current_row['rsi'] > rsi_confirm_buy:
        return 'buy'
    elif ema_crossed_down and current_row['rsi'] < rsi_confirm_sell:
        return 'sell'
    return 'hold'

def strategy_ai_powered_v6(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    params = params or {}
    ema9_length = params.get('ema9_length', 9)
    ema50_length = params.get('ema50_length', 50)
    adx_threshold = params.get('adx_threshold', 20)
    ai_buy_threshold = params.get('ai_buy_threshold', 0.2)
    ai_sell_threshold = params.get('ai_sell_threshold', -0.2)
    required_indicators = [f'ema_{ema9_length}', f'ema_{ema50_length}', 'macd', 'macd_signal', 'macd_hist', 'rsi', 'plus_di', 'minus_di', 'adx', 'close']
    if current_row[required_indicators].isna().any():
        return 'hold'
    idx = data.index.get_loc(current_row.name)
    if idx == 0:
        return 'hold'
    prev_row = data.iloc[idx - 1]
    if pd.isna(prev_row['macd']) or pd.isna(prev_row['macd_signal']):
        return 'hold'
    macd_line = current_row['macd']
    macd_signal_line = current_row['macd_signal']
    macd_hist = current_row['macd_hist']
    rsi = current_row['rsi']
    adx = current_row['adx']
    close = current_row['close']
    ema9 = current_row[f'ema_{ema9_length}']
    ema50 = current_row[f'ema_{ema50_length}']
    hist_lookback = params.get('hist_lookback', 5)
    if idx >= hist_lookback:
        trend_factor = data['macd_hist'].iloc[idx-hist_lookback+1:idx+1].mean()
        if pd.isna(trend_factor):
            trend_factor = macd_hist
    else:
        trend_factor = macd_hist
    w_trend = params.get('w_trend', 1.0)
    w_rsi = params.get('w_rsi', 1.0)
    rsi_factor = (rsi - 50) / 10
    ai_confidence = (w_trend * trend_factor + w_rsi * rsi_factor) / max((w_trend + w_rsi), 1e-6)
    ai_buy_signal = ai_confidence > ai_buy_threshold
    ai_sell_signal = ai_confidence < ai_sell_threshold
    is_trending = adx > adx_threshold
    macd_crossover = macd_line > macd_signal_line and prev_row['macd'] <= prev_row['macd_signal']
    macd_crossunder = macd_line < macd_signal_line and prev_row['macd'] >= prev_row['macd_signal']
    close_gt_ema9 = close > ema9
    close_gt_ema50 = close > ema50
    close_lt_ema9 = close < ema9
    close_lt_ema50 = close < ema50
    if macd_crossover and ai_buy_signal and close_gt_ema9 and close_gt_ema50 and is_trending:
        return 'buy'
    elif macd_crossunder and ai_sell_signal and close_lt_ema9 and close_lt_ema50 and is_trending:
        return 'sell'
    return 'hold'

def strategy_bb_adx_trend(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    params = params or {}
    adx_trend_threshold = params.get('adx_trend_threshold', 30)
    ema_period = params.get('ema_period', 50)
    required_indicators = ['bollinger_upper', 'bollinger_lower', 'adx', 'plus_di', 'minus_di', 'close', f'ema_{ema_period}']
    if current_row[required_indicators].isna().any():
        return 'hold'
    idx = data.index.get_loc(current_row.name)
    if idx == 0:
        return 'hold'
    prev_row = data.iloc[idx - 1]
    if pd.isna(prev_row['close']) or pd.isna(prev_row['bollinger_upper']) or pd.isna(prev_row['bollinger_lower']):
        return 'hold'
    bb_upper = current_row['bollinger_upper']
    bb_lower = current_row['bollinger_lower']
    adx = current_row['adx']
    plus_di = current_row['plus_di']
    minus_di = current_row['minus_di']
    close = current_row['close']
    is_trending = adx > adx_trend_threshold
    is_uptrend_dmi = plus_di > minus_di
    is_downtrend_dmi = minus_di > plus_di
    prev_close_inside_bands = (prev_row['close'] < prev_row['bollinger_upper']) and (prev_row['close'] > prev_row['bollinger_lower'])
    long_condition = (close > bb_upper) and prev_close_inside_bands and is_trending and is_uptrend_dmi and close > current_row[f'ema_{ema_period}']
    short_condition = (close < bb_lower) and prev_close_inside_bands and is_trending and is_downtrend_dmi and close < current_row[f'ema_{ema_period}']
    if long_condition:
        return 'buy'
    elif short_condition:
        return 'sell'
    return 'hold'

def strategy_rsi_confirmed(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    params = params or {}
    rsi_oversold = params.get('rsi_oversold', 35)
    rsi_overbought = params.get('rsi_overbought', 65)
    required_indicators = ['rsi', 'macd', 'macd_signal']
    if current_row[required_indicators].isna().any():
        return 'hold'
    idx = data.index.get_loc(current_row.name)
    if idx == 0:
        return 'hold'
    prev_row = data.iloc[idx-1]
    if pd.isna(prev_row['rsi']):
        return 'hold'
    rsi_crossed_above_os = current_row['rsi'] > rsi_oversold and prev_row['rsi'] <= rsi_oversold
    rsi_crossed_below_ob = current_row['rsi'] < rsi_overbought and prev_row['rsi'] >= rsi_overbought
    macd_bullish = current_row['macd'] > current_row['macd_signal'] and current_row['macd'] > 0
    macd_bearish = current_row['macd'] < current_row['macd_signal'] and current_row['macd'] < 0
    if rsi_crossed_above_os and macd_bullish:
        return 'buy'
    elif rsi_crossed_below_ob and macd_bearish:
        return 'sell'
    return 'hold'

def strategy_ema_crossover_filtered(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    params = params or {}
    ema_fast = params.get('ema_fast', 9)
    ema_slow = params.get('ema_slow', 21)
    ema_trend = params.get('ema_trend', 50)
    required_indicators = [f'ema_{ema_fast}', f'ema_{ema_slow}', f'ema_{ema_trend}', 'close', 'atr']
    if current_row[required_indicators].isna().any():
        return 'hold'
    idx = data.index.get_loc(current_row.name)
    if idx == 0:
        return 'hold'
    prev_row = data.iloc[idx-1]
    if pd.isna(prev_row[f'ema_{ema_fast}']) or pd.isna(prev_row[f'ema_{ema_slow}']):
        return 'hold'
    crossed_up = current_row[f'ema_{ema_fast}'] > current_row[f'ema_{ema_slow}'] and prev_row[f'ema_{ema_fast}'] <= prev_row[f'ema_{ema_slow}']
    crossed_down = current_row[f'ema_{ema_fast}'] < current_row[f'ema_{ema_slow}'] and prev_row[f'ema_{ema_fast}'] >= prev_row[f'ema_{ema_slow}']
    is_uptrend = current_row['close'] > current_row[f'ema_{ema_trend}']
    is_downtrend = current_row['close'] < current_row[f'ema_{ema_trend}']
    atr_filter = current_row['atr'] > data['atr'].rolling(20).mean().iloc[idx] if idx >= 20 else True
    if crossed_up and is_uptrend and atr_filter:
        return 'buy'
    elif crossed_down and is_downtrend and atr_filter:
        return 'sell'
    return 'hold'

def strategy_consolidated_simplified(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    params = params or {}
    ema_fast_len = params.get('ema_fast_len', 9)
    ema_med_len = params.get('ema_med_len', 14)
    ema_slow_len = params.get('ema_slow_len', 21)
    rsi_buy_level = params.get('rsi_buy_level', 55.0)
    rsi_sell_level = params.get('rsi_sell_level', 45.0)
    adx_threshold = params.get('adx_threshold', 25.0)
    required_indicators = [f'ema_{ema_fast_len}', f'ema_{ema_med_len}', f'ema_{ema_slow_len}', 'macd', 'macd_signal', 'rsi', 'volume', 'vol_ma', 'plus_di', 'minus_di', 'adx', 'close', 'open']
    if current_row[required_indicators].isna().any():
        return 'hold'
    idx = data.index.get_loc(current_row.name)
    if idx == 0:
        return 'hold'
    prev_row = data.iloc[idx - 1]
    ema_fast = current_row[f'ema_{ema_fast_len}']
    ema_med = current_row[f'ema_{ema_med_len}']
    ema_slow = current_row[f'ema_{ema_slow_len}']
    macd_line = current_row['macd']
    signal_line = current_row['macd_signal']
    price_rsi = current_row['rsi']
    di_pos = current_row['plus_di']
    di_neg = current_row['minus_di']
    adx_val = current_row['adx']
    close = current_row['close']
    cond_ema_fast_slow_cross_buy = macd_line > signal_line and prev_row['macd'] <= prev_row['macd_signal']
    cond_ema_fast_slow_cross_sell = macd_line < signal_line and prev_row['macd'] >= prev_row['macd_signal']
    cond_rsi_ok_buy = price_rsi > rsi_buy_level
    cond_rsi_ok_sell = price_rsi < rsi_sell_level
    cond_ema_trend_ok_buy = ema_med > ema_slow
    cond_ema_trend_ok_sell = ema_med < ema_slow
    cond_adx_strength_ok = adx_val > adx_threshold
    cond_adx_direction_ok_buy = di_pos > di_neg
    cond_adx_direction_ok_sell = di_neg > di_pos
    cond_adx_filter_ok_buy = cond_adx_strength_ok and cond_adx_direction_ok_buy
    cond_adx_filter_ok_sell = cond_adx_strength_ok and cond_adx_direction_ok_sell
    if (cond_ema_fast_slow_cross_buy and cond_rsi_ok_buy and cond_ema_trend_ok_buy and cond_adx_filter_ok_buy):
        return 'buy'
    elif (cond_ema_fast_slow_cross_sell and cond_rsi_ok_sell and cond_ema_trend_ok_sell and cond_adx_filter_ok_sell):
        return 'sell'
    return 'hold'

def strategy_zlema_kalman_cross(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    params = params or {}
    period1 = params.get('period1', 8)
    period2 = params.get('period2', 21)
    show_cross = params.get('show_cross', True)
    adx_threshold = params.get('adx_threshold', 20)
    required_cols = [f'zlema_{period1}', f'zlema_{period2}', 'adx']
    if not all(col in current_row.index for col in required_cols) or current_row[required_cols].isna().any():
        return 'hold'
    if not show_cross:
        return 'hold'
    idx = data.index.get_loc(current_row.name)
    if idx == 0:
        return 'hold'
    prev_row = data.iloc[idx - 1]
    if prev_row[required_cols].isna().any():
        return 'hold'
    zlema1_curr = current_row[f'zlema_{period1}']
    zlema2_curr = current_row[f'zlema_{period2}']
    zlema1_prev = prev_row[f'zlema_{period1}']
    zlema2_prev = prev_row[f'zlema_{period2}']
    crossed_above = zlema1_curr > zlema2_curr and zlema1_prev <= zlema2_prev
    crossed_below = zlema1_curr < zlema2_curr and zlema1_prev >= zlema2_prev
    if crossed_above and current_row['adx'] > adx_threshold:
        return 'buy'
    elif crossed_below and current_row['adx'] > adx_threshold:
        return 'sell'
    return 'hold'

def strategy_ict_turtle_soup(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    params = params or {}
    bar_length = params.get('bar_length', 20)
    mss_offset = params.get('mss_offset', 10)
    breakout_method = params.get('breakout_method', 'Wick')
    volume_multiplier = params.get('volume_multiplier', 1.5)
    required_columns = ['high', 'low', 'close', 'open', 'htf_high_shifted', 'htf_low_shifted', 'mss_high_shifted', 'mss_low_shifted', 'volume', 'vol_ma']
    if current_row[required_columns].isna().any():
        return 'hold'
    idx_loc = data.index.get_loc(current_row.name)
    if idx_loc < bar_length:
        return 'hold'
    current_high = current_row['high']
    current_low = current_row['low']
    current_close = current_row['close']
    prev_mss_high = current_row['mss_high_shifted']
    prev_mss_low = current_row['mss_low_shifted']
    break_high_price = current_high if breakout_method == 'Wick' else current_close
    break_low_price = current_low if breakout_method == 'Wick' else current_close
    lookback_start_index = idx_loc - bar_length
    if lookback_start_index < 0:
        lookback_start_index = 0
    recent_df_slice = data.iloc[lookback_start_index:idx_loc]
    recent_high_liq_grab = (recent_df_slice['high'].values > recent_df_slice['htf_high_shifted'].values).any()
    recent_low_liq_grab = (recent_df_slice['low'].values < recent_df_slice['htf_low_shifted'].values).any()
    high_volume = current_row['volume'] > current_row['vol_ma'] * volume_multiplier
    if recent_high_liq_grab and break_low_price < prev_mss_low and high_volume:
        return 'sell'
    elif recent_low_liq_grab and break_high_price > prev_mss_high and high_volume:
        return 'buy'
    return 'hold'

def strategy_channel_breakout(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    params = params or {}
    length = params.get('length', 10)
    adx_threshold = params.get('adx_threshold', 25)
    highest_col = f'highest_{length}_shifted'
    lowest_col = f'lowest_{length}_shifted'
    required_cols = [highest_col, lowest_col, 'high', 'low', 'adx']
    if not all(col in current_row.index for col in required_cols) or current_row[required_cols].isna().any():
        return 'hold'
    prev_upBound = current_row[highest_col]
    prev_downBound = current_row[lowest_col]
    current_high = current_row['high']
    current_low = current_row['low']
    if current_high > prev_upBound and current_row['adx'] > adx_threshold:
        return 'buy'
    elif current_low < prev_downBound and current_row['adx'] > adx_threshold:
        return 'sell'
    return 'hold'

def strategy_cs_alpha(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    params = params or {}
    ema_period = params.get('ema_period', 50)
    rsi_period = params.get('rsi_period', 14)
    volume_multiplier = params.get('volume_multiplier', 2.0)
    adx_threshold = params.get('adx_threshold', 25)
    required_indicators = ['close', f'ema_{ema_period}', 'rsi', 'adx', 'volume', 'vol_ma']
    if current_row[required_indicators].isna().any():
        return 'hold'
    idx = data.index.get_loc(current_row.name)
    if idx < 20:
        return 'hold'
    prev_row = data.iloc[idx-1]
    ema_slope = (current_row[f'ema_{ema_period}'] - prev_row[f'ema_{ema_period}']) / prev_row[f'ema_{ema_period}']
    rsi_divergence = current_row['rsi'] - data['rsi'].rolling(20).mean().iloc[idx]
    volume_spike = current_row['volume'] > current_row['vol_ma'] * volume_multiplier
    is_trending = current_row['adx'] > adx_threshold
    if ema_slope > 0 and rsi_divergence > 0 and volume_spike and is_trending:
        return 'buy'
    elif ema_slope < 0 and rsi_divergence < 0 and volume_spike and is_trending:
        return 'sell'
    return 'hold'

def strategy_ai_adaptive(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None) -> str:
    params = params or {}
    base_ema_fast = params.get('base_ema_fast', 9)
    base_ema_slow = params.get('base_ema_slow', 21)
    rsi_buy_level = params.get('rsi_buy_level', 55)
    rsi_sell_level = params.get('rsi_sell_level', 45)
    adx_threshold = params.get('adx_threshold', 25)
    required_indicators = ['close', 'high', 'low', 'atr', 'adx', 'rsi', 'volume', 'vol_ma']
    if current_row[required_indicators].isna().any():
        return 'hold'
    idx = data.index.get_loc(current_row.name)
    if idx < 20:
        return 'hold'
    prev_row = data.iloc[idx-1]
    atr_ratio = current_row['atr'] / data['atr'].rolling(20).mean().iloc[idx]
    ema_fast = int(base_ema_fast / atr_ratio) if atr_ratio > 0 else base_ema_fast
    ema_slow = int(base_ema_slow / atr_ratio) if atr_ratio > 0 else base_ema_slow
    ema_fast = max(3, min(50, ema_fast))
    ema_slow = max(10, min(100, ema_slow))
    if f'ema_{ema_fast}' not in data.columns or f'ema_{ema_slow}' not in data.columns:
        return 'hold'
    crossed_up = current_row[f'ema_{ema_fast}'] > current_row[f'ema_{ema_slow}'] and prev_row[f'ema_{ema_fast}'] <= prev_row[f'ema_{ema_slow}']
    crossed_down = current_row[f'ema_{ema_fast}'] < current_row[f'ema_{ema_slow}'] and prev_row[f'ema_{ema_fast}'] >= prev_row[f'ema_{ema_slow}']
    is_trending = current_row['adx'] > adx_threshold
    volume_confirmed = current_row['volume'] > current_row['vol_ma']
    if crossed_up and current_row['rsi'] > rsi_buy_level and is_trending and volume_confirmed:
        return 'buy'
    elif crossed_down and current_row['rsi'] < rsi_sell_level and is_trending and volume_confirmed:
        return 'sell'
    return 'hold'

def strategy_enhanced_trading(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
    params = params or {}
    state = state or {'regime': 'wait', 'prev_signal': 0, 'prev_sl': None, 'prev_tp': None}
    required_indicators = [
        'ema_9', 'ema_21', 'rsi', 'bollinger_lower', 'bollinger_mid', 'bollinger_upper',
        'atr', 'adx', 'SUPERT_10_3.0', 'macd', 'macd_signal', 'stochastic_k', 'close'
    ]
    if not all(col in current_row.index for col in required_indicators) or current_row[required_indicators].isna().any():
        return 'hold'

    idx = data.index.get_loc(current_row.name)
    if idx < 20:
        state['regime'] = 'wait'
        return 'hold'

    prev_row = data.iloc[idx-1]

    # Detect Regime
    bb_squeeze = (current_row['bollinger_upper'] - current_row['bollinger_lower']) < 1.5 * current_row['atr']
    strong_trend = (current_row['adx'] > 25) and (
        abs(current_row['ema_9'] - current_row['ema_21']) > 0.005 * current_row['close']
    )
    macd_bullish = (
        current_row['macd'] > current_row['macd_signal'] and
        prev_row['macd'] <= prev_row['macd_signal']
    )
    macd_bearish = (
        current_row['macd'] < current_row['macd_signal'] and
        prev_row['macd'] >= prev_row['macd_signal']
    )

    if strong_trend and (macd_bullish or macd_bearish):
        state['regime'] = 'trend'
    elif bb_squeeze and current_row['adx'] < 20:
        state['regime'] = 'range'
    else:
        state['regime'] = 'wait'

    signal = 0
    sl = np.nan
    tp = np.nan

    # Generate Signals
    if state['regime'] == 'trend':
        # Bullish Entry
        if (
            current_row['ema_9'] > current_row['ema_21'] and
            current_row['SUPERT_10_3.0'] == 1 and
            current_row['macd'] > 0
        ):
            signal = 1
            sl = current_row['close'] - 1.5 * current_row['atr']
            tp = current_row['close'] + 2.5 * current_row['atr']
        # Bearish Entry
        elif (
            current_row['ema_9'] < current_row['ema_21'] and
            current_row['SUPERT_10_3.0'] == -1 and
            current_row['macd'] < 0
        ):
            signal = -1
            sl = current_row['close'] + 1.5 * current_row['atr']
            tp = current_row['close'] - 2.5 * current_row['atr']
    elif state['regime'] == 'range':
        # Long Entry
        if (
            current_row['close'] <= current_row['bollinger_lower'] and
            current_row['rsi'] < 35 and
            current_row['stochastic_k'] < 30
        ):
            signal = 1
            sl = current_row['close'] - 0.8 * current_row['atr']
            tp = current_row['bollinger_mid']
        # Short Entry
        elif (
            current_row['close'] >= current_row['bollinger_upper'] and
            current_row['rsi'] > 65 and
            current_row['stochastic_k'] > 70
        ):
            signal = -1
            sl = current_row['close'] + 0.8 * current_row['atr']
            tp = current_row['bollinger_mid']

    # Apply Exit Rules
    if state['prev_signal'] != 0:
        if state['regime'] == 'trend' and signal == 0:
            # Trailing Stop Update
            if state['prev_signal'] == 1:  # Long
                sl = max(state['prev_sl'] or 0, current_row['close'] - 1.2 * current_row['atr'])
            elif state['prev_signal'] == -1:  # Short
                sl = min(state['prev_sl'] or float('inf'), current_row['close'] + 1.2 * current_row['atr'])
        elif state['regime'] == 'range' and signal == 0:
            # RSI-based Early Exit
            if (state['prev_signal'] == 1 and current_row['rsi'] > 55) or (
                state['prev_signal'] == -1 and current_row['rsi'] < 45
            ):
                signal = -state['prev_signal']  # Exit signal
                tp = current_row['close']
                sl = state['prev_sl']
        else:
            sl = state['prev_sl']
            tp = state['prev_tp']

    # Update State
    state['prev_signal'] = signal
    state['prev_sl'] = sl
    state['prev_tp'] = tp

    # Store SL/TP in DataFrame for Backtester
    if signal != 0:
        data.at[current_row.name, 'enhanced_trading_sl'] = sl
        data.at[current_row.name, 'enhanced_trading_tp'] = tp

    return 'buy' if signal == 1 else 'sell' if signal == -1 else 'hold'

def strategy_voting_ensemble(current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
    params = params or {}
    state = state or {'active_trade': False, 'position': ''}
    min_votes_entry = params.get('min_votes_entry', 2)
    constituent_strategies = params.get('constituent_strategies', [])
    exit_vote_percentage = params.get('exit_vote_percentage', 0.40)
    if not constituent_strategies:
        logger.warning("Voting_Ensemble: No constituent strategies configured.")
        return 'hold'
    total_strategies = len(constituent_strategies)
    if total_strategies == 0:
        return 'hold'
    min_opposing_votes_exit = math.ceil(total_strategies * exit_vote_percentage)
    buy_votes = 0
    sell_votes = 0
    for name in constituent_strategies:
        signal_col = f'{name}_signal'.lower()
        if signal_col in current_row.index:
            signal = current_row[signal_col]
            if signal == 'buy':
                buy_votes += 1
            elif signal == 'sell':
                sell_votes += 1
        else:
            logger.debug(f"Voting_Ensemble: Signal column '{signal_col}' not found at {current_row.name}")
    current_position = state.get('position', '')
    if current_position == 'long' and sell_votes >= min_opposing_votes_exit:
        logger.debug(f"Voting_Ensemble EXIT LONG @ {current_row.name}: Sells={sell_votes}/{min_opposing_votes_exit}")
        return 'sell'
    elif current_position == 'short' and buy_votes >= min_opposing_votes_exit:
        logger.debug(f"Voting_Ensemble EXIT SHORT @ {current_row.name}: Buys={buy_votes}/{min_opposing_votes_exit}")
        return 'buy'
    signal_to_return = 'hold'
    if not state.get('active_trade', False):
        if buy_votes >= min_votes_entry:
            signal_to_return = 'buy'
            logger.debug(f"Voting_Ensemble ENTRY BUY @ {current_row.name}: Buys={buy_votes}/{min_votes_entry}")
        elif sell_votes >= min_votes_entry:
            signal_to_return = 'sell'
            logger.debug(f"Voting_Ensemble ENTRY SELL @ {current_row.name}: Sells={sell_votes}/{min_votes_entry}")
    return signal_to_return

def calculate_tr(df: pd.DataFrame) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr

def calculate_kalman_filter_iterative(src_series: pd.Series, tr_series: pd.Series) -> pd.Series:
    value1 = pd.Series(index=src_series.index, dtype=float)
    value2 = pd.Series(index=src_series.index, dtype=float)
    filtered_series = pd.Series(index=src_series.index, dtype=float)
    first_valid_src_index = src_series.first_valid_index()
    first_valid_tr_index = tr_series.first_valid_index()
    if first_valid_src_index is None or first_valid_tr_index is None:
        logger.warning("Kalman Filter: Not enough valid data in source or TR series.")
        return filtered_series
    start_index = max(first_valid_src_index, first_valid_tr_index)
    start_loc = src_series.index.get_loc(start_index)
    if start_loc == 0:
        logger.warning("Kalman Filter: Cannot calculate on the first data point. Shifting start.")
        if len(src_series.index) > 1:
            start_loc = 1
            start_index = src_series.index[start_loc]
        else:
            return filtered_series
    init_loc = start_loc - 1
    init_index = src_series.index[init_loc]
    value1.iloc[init_loc] = 0.0
    value2.iloc[init_loc] = tr_series.iloc[init_loc] if pd.notna(tr_series.iloc[init_loc]) else 0.0
    filtered_series.iloc[init_loc] = src_series.iloc[init_loc] if pd.notna(src_series.iloc[init_loc]) else 0.0
    for i in range(start_loc, len(src_series.index)):
        current_index = src_series.index[i]
        prev_index = src_series.index[i-1]
        src_curr = src_series.loc[current_index]
        src_prev = src_series.loc[prev_index]
        tr_curr = tr_series.loc[current_index]
        prev_value1 = value1.loc[prev_index] if pd.notna(value1.loc[prev_index]) else 0.0
        prev_value2 = value2.loc[prev_index] if pd.notna(value2.loc[prev_index]) else 0.0
        prev_filtered = filtered_series.loc[prev_index] if pd.notna(filtered_series.loc[prev_index]) else src_prev
        if pd.isna(src_curr) or pd.isna(src_prev) or pd.isna(tr_curr):
            value1.loc[current_index] = prev_value1
            value2.loc[current_index] = prev_value2
            filtered_series.loc[current_index] = prev_filtered
            continue
        v1 = 0.2 * (src_curr - src_prev) + 0.8 * prev_value1
        v2 = 0.1 * tr_curr + 0.8 * prev_value2
        value1.loc[current_index] = v1
        value2.loc[current_index] = v2
        lambda_val = abs(v1 / v2) if v2 != 0 else 0
        lambda_pow4 = lambda_val**4
        lambda_pow2 = lambda_val**2
        sqrt_term = lambda_pow4 + 16 * lambda_pow2
        alpha = (-lambda_pow2 + np.sqrt(max(0, sqrt_term))) / 8.0
        alpha = max(0, min(1, alpha))
        v3 = alpha * src_curr + (1 - alpha) * prev_filtered
        filtered_series.loc[current_index] = v3
    return filtered_series

def calculate_zlema(series: pd.Series, period: int) -> pd.Series:
    if period < 1:
        return pd.Series(index=series.index, dtype=float)
    lag = int(round((period - 1) / 2))
    shifted_series = series.shift(lag)
    ema_data = series + (series - shifted_series)
    zl = ema_data.ewm(span=period, adjust=False).mean()
    return zl

def generate_consolidated_html_report(
    all_metrics: Dict[str, Optional[pd.DataFrame]],
    all_plot_uris: Dict[str, Dict[str, Optional[str]]],
    all_best_strategies: Dict[str, Optional[Dict]],
    output_file: Path):
    logger.info(f"Generating consolidated HTML report at {output_file}...")
    timeframes = list(all_metrics.keys())
    if not timeframes:
        logger.warning("No metrics data found to generate consolidated report.")
        return
    html_start = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Consolidated Backtest Report</title><style>body{{font-family:'Segoe UI',sans-serif;margin:20px;background-color:#f4f4f4;color:#333;line-height:1.6;}}.container{{max-width:1400px;margin:auto;background-color:#fff;padding:30px;box-shadow:0 2px 15px rgba(0,0,0,0.1);border-radius:8px}}h1,h2,h3{{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:8px;margin-top:35px;margin-bottom:20px}}h1{{text-align:center;margin-bottom:25px;border-bottom:none;}}.tab-buttons button {{ background-color: #eee; border: 1px solid #ccc; padding: 10px 15px; cursor: pointer; transition: background-color 0.3s; border-radius: 4px 4px 0 0; margin-right: 2px; font-size: 1em; }}.tab-buttons button.active {{ background-color: #3498db; color: white; border-bottom: 1px solid #3498db; }}.tab-content {{ display: none; padding: 20px; border: 1px solid #ccc; border-top: none; border-radius: 0 0 4px 4px; background-color: #fff; animation: fadeIn 0.5s; }}.tab-content.active {{ display: block; }}@keyframes fadeIn {{ from {{opacity: 0;}} to {{opacity: 1;}} }}table.performance-table{{border-collapse:collapse;width:100%;margin-bottom:25px;font-size:0.9em;}}th,td{{border:1px solid #ddd;padding:10px 12px;text-align:right;}}th{{background-color:#3498db;color:white;text-align:center;font-weight:600;}}tr:nth-child(even){{background-color:#f9f9f9;}} tr:hover {{background-color: #f1f1f1;}}.metric-card{{border:1px solid #e0e0e0;border-radius:6px;padding:20px;margin-bottom:25px;background-color:#fff;box-shadow:0 1px 4px rgba(0,0,0,0.06);}}.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:25px;}}img{{max-width:100%;height:auto;border-radius:4px;margin-top:15px;border:1px solid #eee;display:block;margin-left:auto;margin-right:auto;background-color:#fafafa;}}.positive{{color:#27ae60;font-weight:bold;}} .negative{{color:#c0392b;font-weight:bold;}} .neutral{{color:#7f8c8d;}}.timestamp{{text-align:center;color:#7f8c8d;margin-bottom:35px;font-size:0.9em;}}details{{margin-top:15px;border:1px solid #eee;padding:10px;border-radius:4px;background-color:#f9f9f9;}}details>summary{{cursor:pointer;font-weight:bold;color:#3498db;padding:5px;display:inline-block;}}details[open]>summary {{ border-bottom: 1px solid #ddd; margin-bottom: 10px; }}</style></head><body><div class="container"><h1>Consolidated Multi-Timeframe Backtest Report</h1><p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p><div class="tab-buttons">"""
    for i, tf in enumerate(timeframes):
        active_class = 'active' if i == 0 else ''
        html_start += f'<button class="tab-button {active_class}" onclick="openTab(event, \'{tf}\')">{tf}</button>'
    html_start += "</div>"
    html_content = ""
    for i, tf in enumerate(timeframes):
        active_class = 'active' if i == 0 else ''
        html_content += f'<div id="{tf}" class="tab-content {active_class}">\n<h2>Results for Timeframe: {tf}</h2>\n'
        metrics_df = all_metrics.get(tf)
        plot_uris = all_plot_uris.get(tf, {})
        best_strat_info = all_best_strategies.get(tf)
        if best_strat_info:
            bs_name = best_strat_info['name']
            bs_metric = best_strat_info['metric'].replace('_', ' ').title()
            bs_score = best_strat_info['score']
            score_format = {
                'total_pnl_points': '{:,.2f}', 'avg_win_points': '{:,.2f}', 'avg_loss_points': '{:,.2f}',
                'win_rate': '{:.2f}%', 'profit_factor': '{:.2f}', 'expectancy_points': '{:.2f}',
                'max_drawdown_points': '{:,.2f}'
            }.get(best_strat_info['metric'], '{:.2f}')
            bs_score_str = score_format.format(bs_score) if pd.notna(bs_score) else 'N/A'
            html_content += f'<div class="metric-card" style="border-left: 5px solid #27ae60;"><p>Best Strategy ({tf}, based on max. \'{bs_metric}\'): <strong>{bs_name}</strong> (Score: {bs_score_str})</p></div>'
        else:
            html_content += '<div class="metric-card"><p>Could not determine best strategy (requires >= 5 trades).</p></div>'
        html_content += "<h3>Performance Summary</h3><div class=\"metric-card\">"
        if metrics_df is not None and not metrics_df.empty:
            try:
                metrics_to_format = {
                    'total_pnl_points': '{:,.2f}', 'avg_win_points': '{:,.2f}', 'avg_loss_points': '{:,.2f}',
                    'win_rate': '{:.2f}%', 'profit_factor': '{:.2f}', 'expectancy_points': '{:.2f}',
                    'max_drawdown_points': '{:,.2f}'
                }
                metrics_display_df = metrics_df.copy()
                for col, fmt in metrics_to_format.items():
                    if col in metrics_display_df.columns:
                        metrics_display_df[col] = metrics_display_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')
                if 'total_trades' in metrics_display_df.columns:
                    metrics_display_df['total_trades'] = metrics_display_df['total_trades'].astype(int)
                best_strategy_name = best_strat_info['name'] if best_strat_info else None
                def highlight_best(s):
                    return ['background-color: #e0ffe0' if s.name == best_strategy_name else '' for v in s]
                metrics_table_html = metrics_display_df.style.apply(highlight_best, axis=1).to_html(classes='performance-table', border=1, justify='right')
                html_content += metrics_table_html
            except Exception as e:
                logger.error(f"HTML metrics table error for {tf}: {e}")
                html_content += "<p>Error displaying metrics table.</p>"
        else:
            html_content += "<p>No metrics data available.</p>"
        html_content += '<p style="font-size: 0.8em; color: #555;text-align:center;">Note: PnL & Drawdown in points.</p></div>'
        html_content += "<h3>Equity Curves (Cumulative PnL Points)</h3><div class=\"metric-card\">"
        equity_uri = plot_uris.get('equity_curves', '')
        html_content += f'<img src="{equity_uri}" alt="Equity Curves {tf}">' if equity_uri else "<p>Equity curve plot unavailable.</p>"
        html_content += "</div>"
        html_content += "<h3>Individual Strategy Analysis</h3><div class=\"metric-grid\">"
        strategies_in_metrics = metrics_df.index if metrics_df is not None else []
        for name in strategies_in_metrics:
            if metrics_df is None or name not in metrics_df.index or pd.isna(metrics_df.loc[name, 'total_trades']) or metrics_df.loc[name, 'total_trades'] == 0:
                html_content += f'<div class="metric-card"><h4>{name}</h4><p>No trades or error.</p></div>'
                continue
            metrics = metrics_df.loc[name]
            win_rate_str = f"{metrics.get('win_rate', 0):.2f}%"
            pf_str = f"{metrics.get('profit_factor', 0):.2f}"
            pnl_str = f"{metrics.get('total_pnl_points', 0):,.2f}"
            pnl_dist_uri = plot_uris.get(f'{name}_pnl_distribution', '')
            cum_pnl_uri = plot_uris.get(f'{name}_cumulative_pnl', '')
            best_style = ' style="border-left: 5px solid #27ae60;"' if name == best_strategy_name else ''
            html_content += f"""<div class="metric-card"{best_style}><h4>{name}{' (Best)' if name == best_strategy_name else ''}</h4><p><strong>Total Trades:</strong> {int(metrics.get('total_trades', 0))}</p><p><strong>Win Rate:</strong> {win_rate_str}</p><p><strong>Profit Factor:</strong> {pf_str}</p><p><strong>Total PnL (Points):</strong> {pnl_str}</p><details><summary>Show Plots ({name})</summary>{'<img src="' + pnl_dist_uri + '" alt="' + name + ' PnL Dist">' if pnl_dist_uri else '<p>PnL Distribution plot unavailable.</p>'}{'<img src="' + cum_pnl_uri + '" alt="' + name + ' Cum PnL">' if cum_pnl_uri else '<p>Cumulative PnL plot unavailable.</p>'}</details></div>"""
        html_content += "</div>"
        html_content += "</div>\n"
    html_end = """</div> <script>function openTab(evt, timeframeName) { var i, tabcontent, tablinks; tabcontent = document.getElementsByClassName("tab-content"); for (i = 0; i < tabcontent.length; i++) { tabcontent[i].style.display = "none"; tabcontent[i].classList.remove("active"); } tablinks = document.getElementsByClassName("tab-button"); for (i = 0; i < tablinks.length; i++) { tablinks[i].classList.remove("active"); } document.getElementById(timeframeName).style.display = "block"; document.getElementById(timeframeName).classList.add("active"); evt.currentTarget.classList.add("active"); }</script></body></html>"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_start + html_content + html_end)
        logger.info(f"Consolidated HTML report saved successfully to {output_file}")
    except Exception as e:
        logger.error(f"Failed to write consolidated HTML report: {e}", exc_info=True)

def run_full_backtest_workflow(data_paths: Dict[str, str], output_base_dir: str = 'backtest_runs'):
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_path = Path(output_base_dir) / f'run_{run_timestamp}'
    base_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output base directory: {base_path}")
    all_metrics_dfs: Dict[str, Optional[pd.DataFrame]] = {}
    all_plot_uris: Dict[str, Dict[str, Optional[str]]] = {}
    all_best_strategies: Dict[str, Optional[Dict]] = {}
    all_raw_results: Dict[str, Optional[pd.DataFrame]] = {}
    default_sl_tp = {"sl_atr_mult": 2.0, "tp_atr_mult": 3.0}
    default_tsl = {"use_trailing_sl": True, "trailing_sl_atr_mult": 1.5}
    all_strategy_names = [
        'EMA_Crossover', 'RSI_Threshold', 'MACD_Cross', 'BB_Breakout',
        'Combined_Mom', 'AI_Powered_v6', 'BB_ADX_Trend', 'Rsi_confirmed',
        'Ema_crossover_filtered', 'Consolidated_Simp', 'ZLEMA_Kalman_Cross',
        'ICT_Turtle_Soup', 'Channel_Breakout', 'CS_Alpha', 'AI_Adaptive',
        'Enhanced_Trading'
    ]
    strategies_config = {
        'EMA_Crossover': {"function": strategy_ema_crossover, "params": {**default_sl_tp, **default_tsl, "adx_threshold": 25, "volume_multiplier": 1.5}},
        'RSI_Threshold': {"function": strategy_rsi_threshold, "params": {**default_sl_tp, **default_tsl, "rsi_oversold": 35, "rsi_overbought": 65, "adx_threshold": 20}},
        'MACD_Cross': {"function": strategy_macd_cross, "params": {**default_sl_tp, **default_tsl, "ema_trend_period": 50}},
        'BB_Breakout': {"function": strategy_bb_squeeze_breakout, "params": {**default_sl_tp, **default_tsl, "adx_threshold": 25, "volume_multiplier": 1.5, "squeeze_lookback": 20}},
        'Combined_Mom': {"function": strategy_combined_momentum, "params": {**default_sl_tp, **default_tsl, "rsi_confirm_buy": 55, "rsi_confirm_sell": 45}},
        'AI_Powered_v6': {"function": strategy_ai_powered_v6, "params": {**default_sl_tp, **default_tsl, "adx_threshold": 20, "ai_buy_threshold": 0.2, "ai_sell_threshold": -0.2}},
        'BB_ADX_Trend': {"function": strategy_bb_adx_trend, "params": {**default_sl_tp, **default_tsl, "adx_trend_threshold": 30, "ema_period": 50}},
        'Rsi_confirmed': {"function": strategy_rsi_confirmed, "params": {**default_sl_tp, **default_tsl, "rsi_oversold": 35, "rsi_overbought": 65}},
        'Ema_crossover_filtered': {"function": strategy_ema_crossover_filtered, "params": {**default_sl_tp, **default_tsl, "ema_fast": 9, "ema_slow": 21, "ema_trend": 50}},
        'Consolidated_Simp': {"function": strategy_consolidated_simplified, "params": {**default_sl_tp, **default_tsl, "adx_threshold": 25}},
        'ZLEMA_Kalman_Cross': {"function": strategy_zlema_kalman_cross, "params": {"period1": 8, "period2": 21, "period3": 55, "enable_kalman": "ON", "show_cross": True, **default_sl_tp, **default_tsl, "adx_threshold": 20}},
        'ICT_Turtle_Soup': {"function": strategy_ict_turtle_soup, "params": {"mss_offset": 10, "breakout_method": "Wick", "volume_multiplier": 1.5, **default_sl_tp, **default_tsl}},
        'Channel_Breakout': {"function": strategy_channel_breakout, "params": {"length": 10, **default_sl_tp, **default_tsl, "adx_threshold": 25}},
        'CS_Alpha': {"function": strategy_cs_alpha, "params": {**default_sl_tp, **default_tsl, "ema_period": 50, "rsi_period": 14, "volume_multiplier": 2.0, "adx_threshold": 25}},
        'AI_Adaptive': {"function": strategy_ai_adaptive, "params": {**default_sl_tp, **default_tsl, "base_ema_fast": 9, "base_ema_slow": 21, "rsi_buy_level": 55, "rsi_sell_level": 45, "adx_threshold": 25}},
        'Enhanced_Trading': {"function": strategy_enhanced_trading, "params": {**default_sl_tp, **default_tsl}},
        'Voting_Ensemble': {"function": strategy_voting_ensemble, "params": {"min_votes_entry": 2, "exit_vote_percentage": 0.40, "constituent_strategies": all_strategy_names, **default_sl_tp, **default_tsl}}
    }
    strategy_names = list(strategies_config.keys())
    for timeframe, data_path_str in data_paths.items():
        logger.info(f"\n===== Processing Timeframe: {timeframe} =====")
        data_path = Path(data_path_str)
        timeframe_output_dir = base_path / timeframe
        timeframe_output_dir.mkdir(parents=True, exist_ok=True)
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}. Skipping.")
            continue
        try:
            logger.info(f"Loading data from {data_path}")
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            if not isinstance(data.index, pd.DatetimeIndex):
                logger.warning(f"Index for {timeframe} is not DatetimeIndex. Attempting conversion.")
                data.index = pd.to_datetime(data.index)
            data.index.name = 'datetime'
            logger.info(f"Preprocessing data for {timeframe} strategies...")
            try:
                current_tf_minutes = int(timeframe.replace('min', ''))
            except ValueError:
                logger.error(f"Could not determine minutes from timeframe string: {timeframe}. Using default 1.")
                current_tf_minutes = 1
            htf_minutes = 60
            mss_offset = strategies_config.get('ICT_Turtle_Soup', {}).get('params', {}).get('mss_offset', 10)
            if current_tf_minutes > 0 and htf_minutes >= current_tf_minutes and htf_minutes % current_tf_minutes == 0:
                bar_length = htf_minutes // current_tf_minutes
                logger.info(f"Calculated bar_length for {timeframe}: {bar_length} (HTF={htf_minutes}min)")
            else:
                bar_length = 20
                logger.warning(f"Timeframes incompatible. Using default bar_length: {bar_length} for Turtle Soup.")
            if 'ICT_Turtle_Soup' in strategies_config:
                strategies_config['ICT_Turtle_Soup']['params']['bar_length'] = bar_length
            data['htf_high_shifted'] = data['high'].rolling(window=bar_length, min_periods=bar_length).max().shift(1)
            data['htf_low_shifted'] = data['low'].rolling(window=bar_length, min_periods=bar_length).min().shift(1)
            data['mss_high_shifted'] = data['high'].rolling(window=mss_offset, min_periods=mss_offset).max().shift(1)
            data['mss_low_shifted'] = data['low'].rolling(window=mss_offset, min_periods=mss_offset).min().shift(1)
            if 'ZLEMA_Kalman_Cross' in strategies_config:
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
                    data['tr'] = data['tr'].bfill()
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
            if 'Channel_Breakout' in strategies_config:
                logger.info(f"Calculating Channel Breakout prerequisites for {timeframe}...")
                cb_params = strategies_config['Channel_Breakout']['params']
                cb_length = cb_params.get('length', 10)
                highest_col_temp = f'highest_{cb_length}'
                lowest_col_temp = f'lowest_{cb_length}'
                data[highest_col_temp] = data['high'].rolling(window=cb_length, min_periods=cb_length).max()
                data[lowest_col_temp] = data['low'].rolling(window=cb_length, min_periods=cb_length).min()
                data[f'highest_{cb_length}_shifted'] = data[highest_col_temp].shift(1)
                data[f'lowest_{cb_length}_shifted'] = data[lowest_col_temp].shift(1)
                logger.info(f"Calculated Channel Breakout lines (Length={cb_length}) for {timeframe}.")
            if 'AI_Adaptive' in strategies_config:
                logger.info(f"Calculating adaptive EMA prerequisites for {timeframe}...")
                ai_params = strategies_config['AI_Adaptive']['params']
                base_ema_fast = ai_params.get('base_ema_fast', 9)
                base_ema_slow = ai_params.get('base_ema_slow', 21)
                for period in range(3, 101):
                    data[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
            if 'atr' not in data.columns:
                logger.error(f"'atr' column not found in data for {timeframe}. Adding dummy ATR.")
                data['atr'] = 1.0
            all_required_columns = set(['open', 'high', 'low', 'close', 'volume', 'atr'])
            if 'ICT_Turtle_Soup' in strategies_config:
                all_required_columns.update(['htf_high_shifted', 'htf_low_shifted', 'mss_high_shifted', 'mss_low_shifted'])
            if 'ZLEMA_Kalman_Cross' in strategies_config:
                z_params = strategies_config['ZLEMA_Kalman_Cross']['params']
                z_p1 = z_params.get('period1', 8)
                z_p2 = z_params.get('period2', 21)
                z_p3 = z_params.get('period3', 55)
                all_required_columns.add('hlc3')
                if z_params.get('enable_kalman', "ON") == "ON":
                    all_required_columns.update(['tr', 'kalman_src'])
                all_required_columns.update([f"zlema_{z_p1}", f"zlema_{z_p2}", f"zlema_{z_p3}"])
            if 'Channel_Breakout' in strategies_config:
                cb_params = strategies_config['Channel_Breakout']['params']
                cb_len = cb_params.get('length', 10)
                all_required_columns.update([f"highest_{cb_len}_shifted", f"lowest_{cb_len}_shifted"])
            if 'AI_Adaptive' in strategies_config:
                all_required_columns.update([f'ema_{p}' for p in range(3, 101)])
            if 'Enhanced_Trading' in strategies_config:
                all_required_columns.update(['SUPERT_10_3.0', 'stochastic_k', 'bollinger_mid'])
            all_required_columns.update([
                'ema_9', 'ema_14', 'ema_21', 'ema_50', 'macd', 'macd_signal', 'macd_hist',
                'rsi', 'plus_di', 'minus_di', 'adx', 'bollinger_upper', 'bollinger_lower',
                'bollinger_bandwidth', 'vol_ma'
            ])
            initial_rows = len(data)
            present_required_columns = [col for col in all_required_columns if col in data.columns]
            missing_indicator_cols = all_required_columns - set(data.columns)
            if missing_indicator_cols:
                logger.warning(f"Missing expected indicator columns for {timeframe}: {missing_indicator_cols}. NaN drop might be incomplete.")
            logger.info(f"Dropping rows with NaN in columns: {present_required_columns}")
            data.dropna(subset=present_required_columns, inplace=True)
            logger.info(f"Preprocessing: Dropped {initial_rows - len(data)} total rows with NaN values for {timeframe}.")
            if data.empty:
                logger.error(f"No data remaining after NaN drop for {timeframe}. Skipping backtest.")
                continue
            logger.info(f"Initializing & Running backtester engine for {timeframe}...")
            backtester = EnhancedMultiStrategyBacktester(strategies_config)
            results_df = backtester.run_backtest(data)
            if results_df is not None and not results_df.empty:
                all_raw_results[timeframe] = results_df
                detailed_csv_path = timeframe_output_dir / 'backtest_results_detailed.csv'
                results_df.to_csv(detailed_csv_path)
                logger.info(f"Saved detailed results for {timeframe} to {detailed_csv_path}")
                logger.info(f"Initializing analyzer for {timeframe}...")
                analyzer = BacktestAnalyzerReporter(results_df, strategy_names)
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
            all_metrics_dfs[timeframe] = None
            all_plot_uris[timeframe] = {}
            all_best_strategies[timeframe] = None
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
    logger.info("\n===== Overall Timeframe Comparison (Console Summary) =====")
    for timeframe, metrics_data in all_metrics_dfs.items():
        print(f"\n--- {timeframe} Summary ---")
        if metrics_data is not None and not metrics_data.empty:
            metrics_data_sorted = metrics_data.sort_index()
            for strategy_name, strategy_metrics in metrics_data_sorted.iterrows():
                pnl = strategy_metrics.get('total_pnl_points', 'N/A')
                trades = strategy_metrics.get('total_trades', 'N/A')
                win_rate = strategy_metrics.get('win_rate', 'N/A')
                pf = strategy_metrics.get('profit_factor', 'N/A')
                wr_str = f"{win_rate:.2f}%" if isinstance(win_rate, (int, float)) and pd.notna(win_rate) else 'N/A'
                pf_str = f"{pf:.2f}" if isinstance(pf, (int, float)) and pd.notna(pf) else 'N/A'
                pnl_str = f"{pnl:.2f}" if isinstance(pnl, (int, float)) and pd.notna(pnl) else 'N/A'
                trade_str = str(int(trades)) if isinstance(trades, (int, float)) and pd.notna(trades) else 'N/A'
                print(f"  Strategy: {strategy_name:<25} | Trades: {trade_str:<5} | Win Rate: {wr_str:<8} | Profit Factor: {pf_str:<5} | PnL Points: {pnl_str}")
        else:
            print(f"  No metrics calculated for {timeframe}.")

if __name__ == "__main__":
    data_files_by_timeframe = {
        "1min": "/Users/shekhar/Desktop/BOT/trading_bot_final/runs/20250406_165347/data/nifty_indicators_1min.csv",
        "3min": "/Users/shekhar/Desktop/BOT/trading_bot_final/runs/20250406_165347/data/nifty_indicators_3min.csv",
        "5min": "/Users/shekhar/Desktop/BOT/trading_bot_final/runs/20250406_165347/data/nifty_indicators_5min.csv",
        "15min": "/Users/shekhar/Desktop/BOT/trading_bot_final/runs/20250406_165347/data/nifty_indicators_15min.csv",
    }
    valid_data_files = {tf: path_str for tf, path_str in data_files_by_timeframe.items() if Path(path_str).is_file()}
    missing_files = set(data_files_by_timeframe.keys()) - set(valid_data_files.keys())
    if missing_files:
        logger.warning(f"Data files missing for timeframes: {missing_files}. They will be skipped.")
    if not valid_data_files:
        logger.error("No valid data files found. Exiting.")
        sys.exit(1)
    run_full_backtest_workflow(
        data_paths=valid_data_files,
        output_base_dir='multi_timeframe_analysis_results'
    )
    logger.info("All backtesting workflows finished.")