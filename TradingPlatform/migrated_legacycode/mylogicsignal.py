import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import argparse
from datetime import timedelta
from collections import defaultdict # Needed for EnhancedSignalAnalyzer
import sys # Needed for exit code in main

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('signals_analyzer_merged.log'), # New log file name
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATA_FOLDER = Path("/Users/shekhar/Desktop/BOT/smartapi-python-main/runs/20250404_010442/data")
# Default file names (can be overridden by command line args)
DEFAULT_INPUT_FILE = DATA_FOLDER / "nifty_signals.csv"
DEFAULT_OUTPUT_FILE = DATA_FOLDER / "nifty_signals_final_analyzer.csv"
DEFAULT_PLOT_FILE = DATA_FOLDER / "signals_plot_analyzer.png"

# Parameter Organization - Using nested dictionaries (from Script 1)
PARAMS: Dict[str, Dict[str, Any]] = {
    # --- Backtesting Parameters ---
    'backtest': {
        # 'initial_capital': 100000, # Analyzer calculates P/L in points/price, not capital based yet
        'commission_pct': 0.0005, # Example commission per trade (0.05%)
        'slippage_pct': 0.0002,  # Example slippage per trade (0.02%)
        'filter_last_month': True # Flag to filter data for last month only
    },
    'core': {
        "max_trade_duration": 50,
        'exit_score_drop_threshold': 1.5, # Exit if score drops below (5.0 - Threshold) or above (5.0 + Threshold)
        'use_score_drop_exit': True,     # Flag to enable/disable score drop exit
        'entry_score_threshold': 6.0,     # Minimum scaled score required for any entry signal
        'use_fib_bounce_entry': True,
        'use_fib_bounce_sell': True,
        'fib_bounce_lookback': 3,        # Bars back to check for Fib zone touch
        'fib_bounce_long_zone': (0.5, 0.618), # Tuple for low and high fib level for long bounce zone
        'fib_bounce_short_zone': (0.382, 0.5),# Tuple for low and high fib level for short bounce zone
        'fib_bounce_confirmation_level': 0.5, # Level price needs to cross after bounce
        'use_ema_bounce_buy': True,
        'use_ema_bounce_sell': True,
        'ema_bounce_lookback': 2,
        'ema_bounce_source_str': "Fast EMA", # "Fast EMA" or "Medium EMA"
        'use_bb_mid_bounce_buy': True,
        'use_bb_mid_bounce_sell': True,
        'bb_bounce_lookback': 2,
        'use_vol_breakout_buy': True,
        'use_vol_breakout_sell': True,
        'trailing_stop_type': "atr",        # "percentage" or "atr"
        'trailing_stop_pct': 0.02,          # Percentage for trailing stop
        'trailing_stop_atr_multiplier': 1.5,# ATR multiplier for trailing stop
        'profit_protection_levels': {       # Tighten stops after reaching profit levels (based on ATR multiplier)
            'level1': {'profit_pct': 0.05, 'new_atr_mult': 1.0},
            'level2': {'profit_pct': 0.10, 'new_atr_mult': 0.5}
        }
    },
  'ema': {
        'fast_len': 9,
        'med_len': 21,
        'slow_len': 50
    },
    'bollinger': {
        'bb_len': 20,
        'bb_std_dev': 2
    },
    'rsi': {
        'rsi_len': 14,
        'rsi_buy_level': 30,
        'rsi_sell_level': 70,
        'rsi_confirm_fib': True,
        'rsi_confirm_ema': True,
        'rsi_confirm_bb': True,
        'rsi_confirm_level_buy': 40,
        'rsi_confirm_level_sell': 60
    },
    'macd': {
        'macd_fast_len': 12,
        'macd_slow_len': 26,
        'macd_signal_len': 9
    },
    'volume': {
        'vol_ma_len': 20,
        'vol_multiplier': 1.5
    },
    'atr': {
        'atr_len': 14,
        'atr_mult': 2.0
    },
    'trend': {
        'use_adx_filter': True,
        'use_adx_direction_filter': True,
        'use_ema_trend_filter': True,
        'adx_len': 14,
        'adx_threshold': 25
    },
    'fibonacci': {
        'fib_pivot_lookback': 5,
        'fib_max_bars': 100,
        'fib_lookback_exit': 10,
        'fib_extension_level': 1.618,
        'use_fib_exit': True
    },
    'score_weights': {
        'w_ema_trend': 1.5,
        'w_ema_signal': 2.0,
        'w_rsi_thresh': 1.0,
        'w_macd_signal': 1.5,
        'w_macd_zero': 1.0,
        'w_vol_break': 1.0,
        'w_adx_strength': 1.0,
        'w_adx_direction': 0.5,
        'w_fib_bounce': 2.0,
        'w_ema_bounce': 1.5,
        'w_bb_bounce': 1.0
    },
    'backtest': {
        'slippage_pct': 0.0005,
        'commission_pct': 0.0005
    }
}

# --- TradeState Class (from Script 1) ---
class TradeState:
    """Keeps track of the current trade status."""
    def __init__(self):
        self.position = None
        self.entry_price = None
        self.entry_index = None
        self.trailing_stop = None
        self.highest_high_in_trade = None
        self.lowest_low_in_trade = None
        self.time_in_trade = 0

    def reset(self):
        self.__init__()

    def update_trailing_stop(self, current_low, current_high, current_close, atr):
        if self.position is None or atr is None or pd.isna(atr) or self.entry_price is None:
            return

        stop_type = PARAMS['core']['trailing_stop_type']
        initial_atr_mult = PARAMS['core']['trailing_stop_atr_multiplier']
        stop_pct = PARAMS['core']['trailing_stop_pct']
        current_atr_mult = initial_atr_mult

        if self.position == 'Long':
            if self.highest_high_in_trade is None or current_high > self.highest_high_in_trade:
                self.highest_high_in_trade = current_high

            profit_pct = (current_close - self.entry_price) / self.entry_price if self.entry_price else 0
            for level_name, level_info in sorted(PARAMS['core']['profit_protection_levels'].items()):
                if profit_pct >= level_info['profit_pct']:
                    current_atr_mult = level_info['new_atr_mult']

            if stop_type == "percentage":
                potential_stop = self.highest_high_in_trade * (1 - stop_pct) if self.highest_high_in_trade else None
            else:
                potential_stop = self.highest_high_in_trade - (atr * current_atr_mult) if self.highest_high_in_trade else None

            if potential_stop is not None and (self.trailing_stop is None or potential_stop > self.trailing_stop):
                self.trailing_stop = potential_stop

        elif self.position == 'Short':
            if self.lowest_low_in_trade is None or current_low < self.lowest_low_in_trade:
                self.lowest_low_in_trade = current_low

            profit_pct = (self.entry_price - current_close) / self.entry_price if self.entry_price else 0
            for level_name, level_info in sorted(PARAMS['core']['profit_protection_levels'].items()):
                 if profit_pct >= level_info['profit_pct']:
                    current_atr_mult = level_info['new_atr_mult']

            if stop_type == "percentage":
                potential_stop = self.lowest_low_in_trade * (1 + stop_pct) if self.lowest_low_in_trade else None
            else:
                potential_stop = self.lowest_low_in_trade + (atr * current_atr_mult) if self.lowest_low_in_trade else None

            if potential_stop is not None and (self.trailing_stop is None or potential_stop < self.trailing_stop):
                self.trailing_stop = potential_stop

# --- Indicator/Feature Calculation Functions (from Script 1) ---
def calculate_fibonacci_levels(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating Fibonacci levels")
    try:
        lookback = PARAMS['fibonacci']['fib_pivot_lookback']
        max_bars = PARAMS['fibonacci']['fib_max_bars']
        df['rolling_high'] = df['high'].rolling(window=2 * lookback + 1, center=True, min_periods=lookback//2+1).max()
        df['rolling_low'] = df['low'].rolling(window=2 * lookback + 1, center=True, min_periods=lookback//2+1).min()
        df['is_pivot_high'] = df['high'] == df['rolling_high']
        df['is_pivot_low'] = df['low'] == df['rolling_low']
        df['last_pivot_high_price'] = df.loc[df['is_pivot_high'], 'high'].ffill().shift()
        df['last_pivot_high_idx'] = df.loc[df['is_pivot_high']].index.to_series().ffill().shift()
        df['last_pivot_low_price'] = df.loc[df['is_pivot_low'], 'low'].ffill().shift()
        df['last_pivot_low_idx'] = df.loc[df['is_pivot_low']].index.to_series().ffill().shift()
        current_bar_indices = np.arange(len(df))
        df['last_pivot_high_idx_num'] = df['last_pivot_high_idx'].apply(lambda x: df.index.get_loc(x) if pd.notna(x) and x in df.index else np.nan)
        df['last_pivot_low_idx_num'] = df['last_pivot_low_idx'].apply(lambda x: df.index.get_loc(x) if pd.notna(x) and x in df.index else np.nan)
        high_too_old = (current_bar_indices - df['last_pivot_high_idx_num']) > max_bars
        low_too_old = (current_bar_indices - df['last_pivot_low_idx_num']) > max_bars
        df.loc[high_too_old, ['last_pivot_high_price', 'last_pivot_high_idx', 'last_pivot_high_idx_num']] = np.nan
        df.loc[low_too_old, ['last_pivot_low_price', 'last_pivot_low_idx', 'last_pivot_low_idx_num']] = np.nan
        df['last_pivot_high_idx_num'].fillna(-1, inplace=True)
        df['last_pivot_low_idx_num'].fillna(-1, inplace=True)
        is_uptrend_fib = (df['last_pivot_high_idx_num'] > df['last_pivot_low_idx_num'])
        swing_high = np.where(is_uptrend_fib, df['last_pivot_high_price'], df['last_pivot_low_price'])
        swing_low = np.where(is_uptrend_fib, df['last_pivot_low_price'], df['last_pivot_high_price'])
        fib_range = swing_high - swing_low
        fib_range[fib_range <= 0] = np.nan
        level_0 = np.where(is_uptrend_fib, swing_low, swing_high)
        level_100 = np.where(is_uptrend_fib, swing_high, swing_low)
        df['fib_0'] = level_0
        df['fib_236'] = level_0 + fib_range * 0.236
        df['fib_382'] = level_0 + fib_range * 0.382
        df['fib_500'] = level_0 + fib_range * 0.500
        df['fib_618'] = level_0 + fib_range * 0.618
        df['fib_786'] = level_0 + fib_range * 0.786
        df['fib_100'] = level_100
        df['is_uptrend_fib'] = is_uptrend_fib
        df.drop(columns=[ 'rolling_high', 'rolling_low', 'is_pivot_high', 'is_pivot_low', 'last_pivot_high_price', 'last_pivot_high_idx', 'last_pivot_low_price', 'last_pivot_low_idx', 'last_pivot_high_idx_num', 'last_pivot_low_idx_num'], inplace=True, errors='ignore')
        logger.info("Finished calculating Fibonacci levels")
        return df
    except Exception as e:
        logger.error(f"Error in calculate_fibonacci_levels: {str(e)}", exc_info=True)
        raise

def calculate_score(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating weighted trading score")
    try:
        weights = PARAMS['score_weights']
        total_possible_score = sum(w for k, w in weights.items() if PARAMS.get(k.split('_')[1], {}).get(f'use_{k.split("_")[1]}', True) or 'use_' not in k) # Rough dynamic total
        total_possible_score = max(1, total_possible_score)
        df['cond_ema_trend_buy'] = df['ema_med'] > df['ema_slow']
        df['cond_ema_trend_sell'] = df['ema_med'] < df['ema_slow']
        df['cond_ema_signal_buy'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift() <= df['ema_slow'].shift())
        df['cond_ema_signal_sell'] = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift() >= df['ema_slow'].shift())
        df['cond_rsi_buy'] = df['rsi'] > PARAMS['rsi']['rsi_buy_level']
        df['cond_rsi_sell'] = df['rsi'] < PARAMS['rsi']['rsi_sell_level']
        df['cond_macd_signal_buy'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift())
        df['cond_macd_signal_sell'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift() >= df['macd_signal'].shift())
        df['cond_macd_zero_buy'] = (df['macd'] > 0) & (df['macd'].shift() <= 0)
        df['cond_macd_zero_sell'] = (df['macd'] < 0) & (df['macd'].shift() >= 0)
        df['cond_vol_break_buy'] = (df['volume'] > df['vol_ma'] * PARAMS['volume']['vol_multiplier']) & (df['close'] > df['open'])
        df['cond_vol_break_sell'] = (df['volume'] > df['vol_ma'] * PARAMS['volume']['vol_multiplier']) & (df['close'] < df['open'])
        df['cond_adx_strength'] = df['adx'] > PARAMS['trend']['adx_threshold']
        df['cond_adx_dir_buy'] = df['plus_di'] > df['minus_di']
        df['cond_adx_dir_sell'] = df['minus_di'] > df['plus_di']
        df['cond_fib_bounce_buy'] = False; df['cond_fib_bounce_sell'] = False
        df['cond_ema_bounce_buy'] = False; df['cond_ema_bounce_sell'] = False
        df['cond_bb_bounce_buy'] = False; df['cond_bb_bounce_sell'] = False
        buy_score = pd.Series(0.0, index=df.index); sell_score = pd.Series(0.0, index=df.index)
        if PARAMS['trend']['use_ema_trend_filter']: buy_score += df['cond_ema_trend_buy'] * weights.get('w_ema_trend', 0); sell_score += df['cond_ema_trend_sell'] * weights.get('w_ema_trend', 0)
        buy_score += df['cond_ema_signal_buy'] * weights.get('w_ema_signal', 0); sell_score += df['cond_ema_signal_sell'] * weights.get('w_ema_signal', 0)
        buy_score += df['cond_rsi_buy'] * weights.get('w_rsi_thresh', 0); sell_score += df['cond_rsi_sell'] * weights.get('w_rsi_thresh', 0)
        buy_score += df['cond_macd_signal_buy'] * weights.get('w_macd_signal', 0); sell_score += df['cond_macd_signal_sell'] * weights.get('w_macd_signal', 0)
        buy_score += df['cond_macd_zero_buy'] * weights.get('w_macd_zero', 0); sell_score += df['cond_macd_zero_sell'] * weights.get('w_macd_zero', 0)
        if PARAMS['core']['use_vol_breakout_buy'] or PARAMS['core']['use_vol_breakout_sell']: buy_score += df['cond_vol_break_buy'] * weights.get('w_vol_break', 0); sell_score += df['cond_vol_break_sell'] * weights.get('w_vol_break', 0)
        if PARAMS['trend']['use_adx_filter']: buy_score += (df['cond_adx_strength'] & df['cond_adx_dir_buy']) * weights.get('w_adx_strength', 0); sell_score += (df['cond_adx_strength'] & df['cond_adx_dir_sell']) * weights.get('w_adx_strength', 0)
        if PARAMS['trend']['use_adx_direction_filter']: buy_score += df['cond_adx_dir_buy'] * weights.get('w_adx_direction', 0); sell_score += df['cond_adx_dir_sell'] * weights.get('w_adx_direction', 0)
        df['buy_score_raw'] = buy_score; df['sell_score_raw'] = sell_score
        df['scaled_score'] = 5.0
        logger.info("Finished initial score calculation")
        return df
    except Exception as e:
        logger.error(f"Error in calculate_score: {str(e)}", exc_info=True)
        raise

def calculate_rsi_divergence(df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
    logger.info(f"Calculating RSI divergence with lookback {lookback}")
    try:
        df['bullish_rsi_div'] = False; df['bearish_rsi_div'] = False
        df['price_low_roll'] = df['low'].rolling(window=lookback, closed='left').min()
        df['rsi_low_roll'] = df['rsi'].rolling(window=lookback, closed='left').min()
        df['price_high_roll'] = df['high'].rolling(window=lookback, closed='left').max()
        df['rsi_high_roll'] = df['rsi'].rolling(window=lookback, closed='left').max()
        df['bullish_rsi_div'] = (df['low'] < df['price_low_roll']) & (df['rsi'] > df['rsi_low_roll'])
        df['bearish_rsi_div'] = (df['high'] > df['price_high_roll']) & (df['rsi'] < df['rsi_high_roll'])
        df.drop(columns=['price_low_roll', 'rsi_low_roll', 'price_high_roll', 'rsi_high_roll'], inplace=True, errors='ignore')
        logger.info("Finished calculating RSI divergence")
        return df
    except Exception as e:
        logger.error(f"Error in calculate_rsi_divergence: {str(e)}", exc_info=True)
        raise

def calculate_targets_for_trade(df: pd.DataFrame, entry_idx: int, position: str, entry_price: float, initial_stop: float) -> pd.DataFrame:
    try:
        current_idx = df.index[entry_idx]
        atr = df.loc[current_idx, 'atr']
        fib_lookback = PARAMS['fibonacci']['fib_lookback_exit']
        fib_ext_level = PARAMS['fibonacci']['fib_extension_level']
        use_fib_target = PARAMS['fibonacci']['use_fib_exit']
        atr_mult_target = PARAMS['atr'].get('atr_mult', 2.0) # Use .get with default

        if pd.isna(atr) or pd.isna(entry_price) or pd.isna(initial_stop):
            logger.warning(f"Cannot calculate targets for trade at index {entry_idx} due to NaN inputs.")
            return df

        if position == 'Long':
            atr_target = entry_price + (atr * atr_mult_target)
            risk = entry_price - initial_stop
            rr_target = entry_price + (risk * 2) if risk > 0 else np.nan
            fib_target = np.nan
            if use_fib_target:
                lookback_start = max(0, entry_idx - fib_lookback)
                swing_low_series = df['low'].iloc[lookback_start : entry_idx]
                if not swing_low_series.empty:
                    swing_low_price = swing_low_series.min()
                    swing_range = entry_price - swing_low_price
                    if swing_range > 0: fib_target = entry_price + (swing_range * fib_ext_level)
            df.loc[current_idx:, 'fib_target_long'] = fib_target
            valid_targets = [t for t in [atr_target, rr_target, fib_target] if pd.notna(t)]
            if valid_targets: df.loc[current_idx:, 'target_price_long'] = min(valid_targets)

        elif position == 'Short':
            atr_target = entry_price - (atr * atr_mult_target)
            risk = initial_stop - entry_price
            rr_target = entry_price - (risk * 2) if risk > 0 else np.nan
            fib_target = np.nan
            if use_fib_target:
                lookback_start = max(0, entry_idx - fib_lookback)
                swing_high_series = df['high'].iloc[lookback_start : entry_idx]
                if not swing_high_series.empty:
                    swing_high_price = swing_high_series.max()
                    swing_range = swing_high_price - entry_price
                    if swing_range > 0: fib_target = entry_price - (swing_range * fib_ext_level)
            df.loc[current_idx:, 'fib_target_short'] = fib_target
            valid_targets = [t for t in [atr_target, rr_target, fib_target] if pd.notna(t)]
            if valid_targets: df.loc[current_idx:, 'target_price_short'] = max(valid_targets)

        return df
    except Exception as e:
        logger.error(f"Error calculating targets for trade at index {entry_idx}: {str(e)}", exc_info=True)
        return df

# --- Main Signal Processing Function (from Script 1) ---
def process_signals(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting enhanced signal processing")
    try:
        # --- Indicator Calculations ---
        logger.info("Calculating base indicators")
        df['ema_fast'] = df['close'].ewm(span=PARAMS['ema']['fast_len'], adjust=False).mean()
        df['ema_med'] = df['close'].ewm(span=PARAMS['ema']['med_len'], adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=PARAMS['ema']['slow_len'], adjust=False).mean()
        df['bb_middle'] = df['close'].rolling(PARAMS['bollinger']['bb_len']).mean()
        rolling_std = df['close'].rolling(PARAMS['bollinger']['bb_len']).std()
        df['bb_upper'] = df['bb_middle'] + rolling_std * PARAMS['bollinger']['bb_std_dev']
        df['bb_lower'] = df['bb_middle'] - rolling_std * PARAMS['bollinger']['bb_std_dev']
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/PARAMS['rsi']['rsi_len'], adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/PARAMS['rsi']['rsi_len'], adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-6)
        df['rsi'] = 100.0 - (100.0 / (1.0 + rs)); df['rsi'].fillna(50, inplace=True)
        ema_fast_macd = df['close'].ewm(span=PARAMS['macd']['macd_fast_len'], adjust=False).mean()
        ema_slow_macd = df['close'].ewm(span=PARAMS['macd']['macd_slow_len'], adjust=False).mean()
        df['macd'] = ema_fast_macd - ema_slow_macd
        df['macd_signal'] = df['macd'].ewm(span=PARAMS['macd']['macd_signal_len'], adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['vol_ma'] = df['volume'].rolling(PARAMS['volume']['vol_ma_len']).mean()
        high_low = df['high'] - df['low']; high_close_prev = abs(df['high'] - df['close'].shift()); low_close_prev = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1/PARAMS['atr']['atr_len'], adjust=False).mean()
        if PARAMS['trend']['use_adx_filter'] or PARAMS['trend']['use_adx_direction_filter']:
            adx_len = PARAMS['trend']['adx_len']; up_move = df['high'].diff(); down_move = -df['low'].diff()
            plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move; minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
            plus_dm.fillna(0, inplace=True); minus_dm.fillna(0, inplace=True)
            tr_smoothed = tr.ewm(alpha=1/adx_len, adjust=False).mean().replace(0, 1e-6)
            smooth_plus_dm = plus_dm.ewm(alpha=1/adx_len, adjust=False).mean(); smooth_minus_dm = minus_dm.ewm(alpha=1/adx_len, adjust=False).mean()
            df['plus_di'] = 100 * (smooth_plus_dm / tr_smoothed); df['minus_di'] = 100 * (smooth_minus_dm / tr_smoothed)
            di_sum = (df['plus_di'] + df['minus_di']).replace(0, 1e-6); dx = 100 * (abs(df['plus_di'] - df['minus_di']) / di_sum)
            df['adx'] = dx.ewm(alpha=1/adx_len, adjust=False).mean(); df.fillna({'adx': 0, 'plus_di': 0, 'minus_di': 0}, inplace=True)
        else:
            df['plus_di'] = 0.0; df['minus_di'] = 0.0; df['adx'] = 0.0

        df = calculate_fibonacci_levels(df)
        df = calculate_score(df)
        df = calculate_rsi_divergence(df, lookback=PARAMS['rsi']['rsi_len'])

        # --- Initialize Signal Columns ---
        df['signal'] = 'Hold'; df['position'] = ''; df['entry_price'] = np.nan
        df['stop_loss'] = np.nan; df['trailing_stop'] = np.nan; df['exit_price'] = np.nan
        df['exit_reason'] = ''; df['entry_signal_type'] = ''
        df['target_price_long'] = np.nan; df['target_price_short'] = np.nan # Initialize target columns
        df['fib_target_long'] = np.nan; df['fib_target_short'] = np.nan

        start_index = max( PARAMS['ema']['slow_len'], PARAMS['bollinger']['bb_len'], PARAMS['rsi']['rsi_len'], PARAMS['macd']['macd_slow_len'], PARAMS['volume']['vol_ma_len'], PARAMS['atr']['atr_len'], (PARAMS['trend']['adx_len'] * 2) if (PARAMS['trend']['use_adx_filter'] or PARAMS['trend']['use_adx_direction_filter']) else 0, (PARAMS['fibonacci']['fib_pivot_lookback'] * 2 + 1), PARAMS['fibonacci']['fib_lookback_exit'], PARAMS['core']['fib_bounce_lookback'], PARAMS['core']['ema_bounce_lookback'], PARAMS['core']['bb_bounce_lookback'] ) + 1

        if start_index >= len(df):
             logger.warning(f"Not enough data. Required: {start_index}, Available: {len(df)}")
             return df # Return early if not enough data

        logger.info(f"Starting main processing loop from index {start_index}")
        trade_state = TradeState()
        weights = PARAMS['score_weights']
        total_possible_score = max(1, sum(weights.values())) # Simplification

        for i in range(start_index, len(df)):
            current_idx = df.index[i]; prev_idx = df.index[i-1]
            current = df.iloc[i]; prev = df.iloc[i-1]
            exit_triggered_this_bar = False; exit_price_this_bar = np.nan

            # --- I. Check for Exits ---
            if trade_state.position is not None:
                trade_state.time_in_trade += 1
                trade_state.update_trailing_stop(current['low'], current['high'], current['close'], current['atr'])
                df.loc[current_idx, 'trailing_stop'] = trade_state.trailing_stop
                potential_exit_price = current['close']

                # Exit Priority: SL > Div > Target > Score > EMA > BB > Vol > Time
                if PARAMS['atr']['use_atr_stop'] and trade_state.trailing_stop is not None:
                    if trade_state.position == 'Long' and current['low'] <= trade_state.trailing_stop: exit_reason='Trailing Stop'; exit_price_this_bar = min(current['open'], trade_state.trailing_stop); exit_triggered_this_bar = True
                    elif trade_state.position == 'Short' and current['high'] >= trade_state.trailing_stop: exit_reason='Trailing Stop'; exit_price_this_bar = max(current['open'], trade_state.trailing_stop); exit_triggered_this_bar = True
                if not exit_triggered_this_bar and PARAMS['rsi']['use_rsi_div_exit']:
                    if trade_state.position == 'Long' and current['bearish_rsi_div']: exit_reason='RSI Div'; exit_price_this_bar = potential_exit_price; exit_triggered_this_bar = True
                    elif trade_state.position == 'Short' and current['bullish_rsi_div']: exit_reason='RSI Div'; exit_price_this_bar = potential_exit_price; exit_triggered_this_bar = True
                if not exit_triggered_this_bar and PARAMS['fibonacci']['use_fib_exit']:
                    fib_target_long = df.loc[current_idx, 'fib_target_long']; fib_target_short = df.loc[current_idx, 'fib_target_short']
                    if trade_state.position == 'Long' and pd.notna(fib_target_long) and current['high'] >= fib_target_long: exit_reason='Fib Target'; exit_price_this_bar = max(current['open'], fib_target_long); exit_triggered_this_bar = True
                    elif trade_state.position == 'Short' and pd.notna(fib_target_short) and current['low'] <= fib_target_short: exit_reason='Fib Target'; exit_price_this_bar = min(current['open'], fib_target_short); exit_triggered_this_bar = True
                # Recalculate score for exit check
                buy_score_bar_exit = df.loc[current_idx, 'buy_score_raw']; sell_score_bar_exit = df.loc[current_idx, 'sell_score_raw']
                if df.loc[current_idx, 'cond_fib_bounce_buy']: buy_score_bar_exit += weights.get('w_fib_bounce', 0); # Add bounce scores if flags were set (in entry logic)
                if df.loc[current_idx, 'cond_fib_bounce_sell']: sell_score_bar_exit += weights.get('w_fib_bounce', 0); # etc...
                if df.loc[current_idx, 'cond_ema_bounce_buy']: buy_score_bar_exit += weights.get('w_ema_bounce', 0)
                if df.loc[current_idx, 'cond_ema_bounce_sell']: sell_score_bar_exit += weights.get('w_ema_bounce', 0)
                if df.loc[current_idx, 'cond_bb_bounce_buy']: buy_score_bar_exit += weights.get('w_bb_bounce', 0)
                if df.loc[current_idx, 'cond_bb_bounce_sell']: sell_score_bar_exit += weights.get('w_bb_bounce', 0)
                net_score_bar_exit = buy_score_bar_exit - sell_score_bar_exit; scaled_score_bar_exit = max(0.0, min(10.0, ((net_score_bar_exit / total_possible_score) * 5.0) + 5.0))
                df.loc[current_idx, 'scaled_score'] = scaled_score_bar_exit # Update final score regardless of exit
                if not exit_triggered_this_bar and PARAMS['core']['use_score_drop_exit']:
                    score_threshold = PARAMS['core']['exit_score_drop_threshold']
                    if trade_state.position == 'Long' and scaled_score_bar_exit < (5.0 - score_threshold): exit_reason=f'Score Drop ({scaled_score_bar_exit:.1f})'; exit_price_this_bar = potential_exit_price; exit_triggered_this_bar = True
                    elif trade_state.position == 'Short' and scaled_score_bar_exit > (5.0 + score_threshold): exit_reason=f'Score Drop ({scaled_score_bar_exit:.1f})'; exit_price_this_bar = potential_exit_price; exit_triggered_this_bar = True
                if not exit_triggered_this_bar and PARAMS['ema']['use_ema_exit']:
                    fast_ema_cross_med_sell = current['ema_fast'] < current['ema_med'] and prev['ema_fast'] >= prev['ema_med']; fast_ema_cross_med_buy = current['ema_fast'] > current['ema_med'] and prev['ema_fast'] <= prev['ema_med']
                    if trade_state.position == 'Long' and fast_ema_cross_med_sell: exit_reason='EMA Cross'; exit_price_this_bar = potential_exit_price; exit_triggered_this_bar = True
                    elif trade_state.position == 'Short' and fast_ema_cross_med_buy: exit_reason='EMA Cross'; exit_price_this_bar = potential_exit_price; exit_triggered_this_bar = True
                if not exit_triggered_this_bar and PARAMS['bollinger']['use_bb_return_exit']:
                    cross_under_bb_mid = current['close'] < current['bb_middle'] and prev['close'] >= prev['bb_middle']; cross_over_bb_mid = current['close'] > current['bb_middle'] and prev['close'] <= prev['bb_middle']
                    if trade_state.position == 'Long' and cross_under_bb_mid: exit_reason='BB Mid Exit'; exit_price_this_bar = potential_exit_price; exit_triggered_this_bar = True
                    elif trade_state.position == 'Short' and cross_over_bb_mid: exit_reason='BB Mid Exit'; exit_price_this_bar = potential_exit_price; exit_triggered_this_bar = True
                if not exit_triggered_this_bar and PARAMS['volume']['use_vol_fade_exit']:
                    low_vol = current['volume'] < df.loc[prev_idx, 'vol_ma']; pullback_long = current['close'] < current['ema_fast']; pullback_short = current['close'] > current['ema_fast']
                    if trade_state.position == 'Long' and low_vol and pullback_long: exit_reason='Vol Fade'; exit_price_this_bar = potential_exit_price; exit_triggered_this_bar = True
                    elif trade_state.position == 'Short' and low_vol and pullback_short: exit_reason='Vol Fade'; exit_price_this_bar = potential_exit_price; exit_triggered_this_bar = True
                if not exit_triggered_this_bar and trade_state.time_in_trade >= PARAMS['core']['max_trade_duration']: exit_reason='Time Exit'; exit_price_this_bar = potential_exit_price; exit_triggered_this_bar = True

                if exit_triggered_this_bar:
                    signal_prefix = 'Exit Long' if trade_state.position == 'Long' else 'Exit Short'
                    df.loc[current_idx, 'signal'] = signal_prefix
                    df.loc[current_idx, 'exit_reason'] = exit_reason
                    df.loc[current_idx, 'exit_price'] = exit_price_this_bar
                    trade_state.reset()
                    df.loc[current_idx:, ['target_price_long', 'target_price_short', 'fib_target_long', 'fib_target_short']] = np.nan

            # --- II. Check for Entries ---
            if trade_state.position is None and not exit_triggered_this_bar:
                entry_signal = None; buy_score_bar = df.loc[current_idx, 'buy_score_raw']; sell_score_bar = df.loc[current_idx, 'sell_score_raw']
                df.loc[current_idx, ['cond_fib_bounce_buy', 'cond_fib_bounce_sell','cond_ema_bounce_buy', 'cond_ema_bounce_sell','cond_bb_bounce_buy', 'cond_bb_bounce_sell']] = False # Reset flags

                # Calculate bounce/breakout conditions & update score
                fib_bounce_buy_cond = False; fib_bounce_sell_cond = False
                if PARAMS['core']['use_fib_bounce_entry'] or PARAMS['core']['use_fib_bounce_sell']:
                    lookback=PARAMS['core']['fib_bounce_lookback']; low_zone_long, high_zone_long = PARAMS['core']['fib_bounce_long_zone']; low_zone_short, high_zone_short = PARAMS['core']['fib_bounce_short_zone']; conf_level = PARAMS['core']['fib_bounce_confirmation_level']; rsi_conf = PARAMS['rsi']['rsi_confirm_fib']; rsi_buy_lvl, rsi_sell_lvl = PARAMS['rsi']['rsi_confirm_level_buy'], PARAMS['rsi']['rsi_confirm_level_sell']
                    if PARAMS['core']['use_fib_bounce_entry'] and current['is_uptrend_fib']:
                        fib_low_name = f'fib_{int(low_zone_long*1000)}'; fib_high_name = f'fib_{int(high_zone_long*1000)}'; fib_conf_name = f'fib_{int(conf_level*1000)}'
                        if fib_low_name in df.columns and fib_high_name in df.columns and fib_conf_name in df.columns:
                            fib_low = current[fib_low_name]; fib_high = current[fib_high_name]; fib_conf = current[fib_conf_name]
                            if pd.notna(fib_low) and pd.notna(fib_high) and pd.notna(fib_conf):
                                touched_zone = (df['low'].iloc[i-lookback:i] <= fib_high).any(); bounced_above = current['close'] > fib_conf; rsi_ok = not rsi_conf or (current['rsi'] > rsi_buy_lvl and current['rsi'] > prev['rsi'])
                                if touched_zone and bounced_above and rsi_ok: fib_bounce_buy_cond = True; df.loc[current_idx, 'cond_fib_bounce_buy'] = True; buy_score_bar += weights.get('w_fib_bounce', 0)
                    if PARAMS['core']['use_fib_bounce_sell'] and not current['is_uptrend_fib']:
                        fib_low_name = f'fib_{int(low_zone_short*1000)}'; fib_high_name = f'fib_{int(high_zone_short*1000)}'; fib_conf_name = f'fib_{int(conf_level*1000)}'
                        if fib_low_name in df.columns and fib_high_name in df.columns and fib_conf_name in df.columns:
                            fib_low = current[fib_low_name]; fib_high = current[fib_high_name]; fib_conf = current[fib_conf_name]
                            if pd.notna(fib_low) and pd.notna(fib_high) and pd.notna(fib_conf):
                                touched_zone = (df['high'].iloc[i-lookback:i] >= fib_low).any(); bounced_below = current['close'] < fib_conf; rsi_ok = not rsi_conf or (current['rsi'] < rsi_sell_lvl and current['rsi'] < prev['rsi'])
                                if touched_zone and bounced_below and rsi_ok: fib_bounce_sell_cond = True; df.loc[current_idx, 'cond_fib_bounce_sell'] = True; sell_score_bar += weights.get('w_fib_bounce', 0)
                ema_bounce_buy_cond = False; ema_bounce_sell_cond = False
                if PARAMS['core']['use_ema_bounce_buy'] or PARAMS['core']['use_ema_bounce_sell']:
                    lookback=PARAMS['core']['ema_bounce_lookback']; ema_source_col = 'ema_fast' if PARAMS['core']['ema_bounce_source_str'] == "Fast EMA" else 'ema_med'; ema_source = df[ema_source_col]; rsi_conf = PARAMS['rsi']['rsi_confirm_ema']; rsi_buy_lvl, rsi_sell_lvl = PARAMS['rsi']['rsi_confirm_level_buy'], PARAMS['rsi']['rsi_confirm_level_sell']
                    if PARAMS['core']['use_ema_bounce_buy']:
                         touched_ema = (df['low'].iloc[i-lookback:i] <= ema_source.iloc[i-lookback:i]).any(); bounced_above = current['close'] > current[ema_source_col] and current['close'] > current['open']; rsi_ok = not rsi_conf or (current['rsi'] > rsi_buy_lvl and current['rsi'] > prev['rsi'])
                         if touched_ema and bounced_above and rsi_ok: ema_bounce_buy_cond = True; df.loc[current_idx, 'cond_ema_bounce_buy'] = True; buy_score_bar += weights.get('w_ema_bounce', 0)
                    if PARAMS['core']['use_ema_bounce_sell']:
                        touched_ema = (df['high'].iloc[i-lookback:i] >= ema_source.iloc[i-lookback:i]).any(); bounced_below = current['close'] < current[ema_source_col] and current['close'] < current['open']; rsi_ok = not rsi_conf or (current['rsi'] < rsi_sell_lvl and current['rsi'] < prev['rsi'])
                        if touched_ema and bounced_below and rsi_ok: ema_bounce_sell_cond = True; df.loc[current_idx, 'cond_ema_bounce_sell'] = True; sell_score_bar += weights.get('w_ema_bounce', 0)
                bb_bounce_buy_cond = False; bb_bounce_sell_cond = False
                if PARAMS['core']['use_bb_mid_bounce_buy'] or PARAMS['core']['use_bb_mid_bounce_sell']:
                    lookback=PARAMS['core']['bb_bounce_lookback']; bb_mid = df['bb_middle']; rsi_conf = PARAMS['rsi']['rsi_confirm_bb']; rsi_buy_lvl, rsi_sell_lvl = PARAMS['rsi']['rsi_confirm_level_buy'], PARAMS['rsi']['rsi_confirm_level_sell']
                    if PARAMS['core']['use_bb_mid_bounce_buy']:
                        touched_bb = (df['low'].iloc[i-lookback:i] <= bb_mid.iloc[i-lookback:i]).any(); bounced_above = current['close'] > current['bb_middle'] and current['close'] > current['open']; rsi_ok = not rsi_conf or (current['rsi'] > rsi_buy_lvl and current['rsi'] > prev['rsi'])
                        if touched_bb and bounced_above and rsi_ok: bb_bounce_buy_cond = True; df.loc[current_idx, 'cond_bb_bounce_buy'] = True; buy_score_bar += weights.get('w_bb_bounce', 0)
                    if PARAMS['core']['use_bb_mid_bounce_sell']:
                        touched_bb = (df['high'].iloc[i-lookback:i] >= bb_mid.iloc[i-lookback:i]).any(); bounced_below = current['close'] < current['bb_middle'] and current['close'] < current['open']; rsi_ok = not rsi_conf or (current['rsi'] < rsi_sell_lvl and current['rsi'] < prev['rsi'])
                        if touched_bb and bounced_below and rsi_ok: bb_bounce_sell_cond = True; df.loc[current_idx, 'cond_bb_bounce_sell'] = True; sell_score_bar += weights.get('w_bb_bounce', 0)
                vol_breakout_buy_cond = False; vol_breakout_sell_cond = False
                if PARAMS['core']['use_vol_breakout_buy'] or PARAMS['core']['use_vol_breakout_sell']:
                    high_vol = current['volume'] > current['vol_ma'] * PARAMS['volume']['vol_multiplier']; lookback_pa = 5
                    if i >= lookback_pa: recent_high = df['high'].iloc[i-lookback_pa:i].max(); recent_low = df['low'].iloc[i-lookback_pa:i].min()
                    else: recent_high = np.nan; recent_low = np.nan
                    if pd.notna(recent_high) and PARAMS['core']['use_vol_breakout_buy'] and high_vol and current['close'] > current['open'] and current['close'] > recent_high: vol_breakout_buy_cond = True
                    if pd.notna(recent_low) and PARAMS['core']['use_vol_breakout_sell'] and high_vol and current['close'] < current['open'] and current['close'] < recent_low: vol_breakout_sell_cond = True

                # Final score calculation for the bar
                net_score_bar = buy_score_bar - sell_score_bar
                current_total_possible = sum(w for k, w in weights.items() if df.loc[current_idx, f'cond_{k.split("_")[1]}_{k.split("_")[-1]}'] or 'bounce' not in k) # Dynamic total based on active conditions
                safe_total_score = max(1, current_total_possible if current_total_possible > 0 else total_possible_score) # Use dynamic total if available
                scaled_score_bar = max(0.0, min(10.0, ((net_score_bar / safe_total_score) * 5.0) + 5.0))
                df.loc[current_idx, 'scaled_score'] = scaled_score_bar

                # Trend Filters
                trend_ok_buy = True; trend_ok_sell = True
                if PARAMS['trend']['use_ema_trend_filter']: trend_ok_buy &= current['cond_ema_trend_buy']; trend_ok_sell &= current['cond_ema_trend_sell']
                if PARAMS['trend']['use_adx_filter']: trend_ok_buy &= current['cond_adx_strength']; trend_ok_sell &= current['cond_adx_strength']
                if PARAMS['trend']['use_adx_direction_filter']: trend_ok_buy &= current['cond_adx_dir_buy']; trend_ok_sell &= current['cond_adx_dir_sell']

                # Entry Decision
                entry_score_threshold = PARAMS['core']['entry_score_threshold']
                if trend_ok_buy and scaled_score_bar >= entry_score_threshold:
                    if fib_bounce_buy_cond: entry_signal = "Fib Bounce Long"
                    elif ema_bounce_buy_cond: entry_signal = "EMA Bounce Long"
                    elif bb_bounce_buy_cond: entry_signal = "BB Bounce Long"
                    elif vol_breakout_buy_cond: entry_signal = "Vol Breakout Long"
                    elif df.loc[current_idx,'cond_ema_signal_buy'] and df.loc[current_idx,'cond_rsi_buy']: entry_signal = "Basic Long"
                elif trend_ok_sell and scaled_score_bar <= (10.0 - entry_score_threshold):
                    if fib_bounce_sell_cond: entry_signal = "Fib Bounce Short"
                    elif ema_bounce_sell_cond: entry_signal = "EMA Bounce Short"
                    elif bb_bounce_sell_cond: entry_signal = "BB Bounce Short"
                    elif vol_breakout_sell_cond: entry_signal = "Vol Breakout Short"
                    elif df.loc[current_idx,'cond_ema_signal_sell'] and df.loc[current_idx,'cond_rsi_sell']: entry_signal = "Basic Short"

                # Process Entry
                if entry_signal:
                    entry_price_adj = current['close'] * (1 + PARAMS['backtest']['slippage_pct']) if "Long" in entry_signal else current['close'] * (1 - PARAMS['backtest']['slippage_pct'])
                    trade_state.entry_price = entry_price_adj; trade_state.entry_index = i; trade_state.time_in_trade = 0
                    if "Long" in entry_signal:
                        trade_state.position = 'Long'; trade_state.highest_high_in_trade = current['high']
                        initial_stop = current['low'] - current['atr'] * PARAMS['core']['trailing_stop_atr_multiplier']
                        trade_state.trailing_stop = initial_stop; df.loc[current_idx, 'signal'] = 'Long'; df.loc[current_idx, 'stop_loss'] = initial_stop
                    elif "Short" in entry_signal:
                        trade_state.position = 'Short'; trade_state.lowest_low_in_trade = current['low']
                        initial_stop = current['high'] + current['atr'] * PARAMS['core']['trailing_stop_atr_multiplier']
                        trade_state.trailing_stop = initial_stop; df.loc[current_idx, 'signal'] = 'Short'; df.loc[current_idx, 'stop_loss'] = initial_stop
                    df.loc[current_idx, 'position'] = trade_state.position; df.loc[current_idx, 'entry_price'] = trade_state.entry_price
                    df.loc[current_idx, 'entry_signal_type'] = entry_signal; df.loc[current_idx, 'trailing_stop'] = trade_state.trailing_stop
                    df = calculate_targets_for_trade(df, i, trade_state.position, trade_state.entry_price, initial_stop)

            # --- III. Update Position State ---
            if trade_state.position is not None and df.loc[current_idx, 'signal'] == 'Hold':
                df.loc[current_idx, 'position'] = trade_state.position
                df.loc[current_idx, 'entry_price'] = trade_state.entry_price

        logger.info("Finished main processing loop")
        return df
    except Exception as e:
        logger.error(f"Error in process_signals: {str(e)}", exc_info=True)
        raise

# --- EnhancedSignalAnalyzer Class (from Script 2 - Adapted) ---
class EnhancedSignalAnalyzer:
    def __init__(self):
        self.trade_history = []
        self.summary_stats = defaultdict(float) # Use defaultdict for easier summing
        self.entry_stats = defaultdict(lambda: defaultdict(float))
        self.exit_stats = defaultdict(lambda: defaultdict(float))
        self.signal_stats = defaultdict(lambda: defaultdict(float))

    def analyze_trades(self, df: pd.DataFrame):
        """Analyzes trades from a DataFrame processed by process_signals."""
        logger.info("Analyzing generated trades for detailed statistics")
        try:
            self.trade_history = [] # Reset history
            current_trade = None
            running_pnl = 0.0
            peak_pnl = 0.0
            max_drawdown = 0.0
            win_pnls = []; loss_pnls = []
            win_durations = []; loss_durations = []
            risk_rewards = []
            current_streak = 0; max_consec_wins = 0; max_consec_losses = 0
            current_streak_type = None

            # Iterate through DataFrame to identify trades
            for i in range(len(df)):
                current_idx = df.index[i]
                signal = df.loc[current_idx, 'signal']

                # Entry Signal Found
                if signal in ['Long', 'Short'] and current_trade is None:
                    entry_price = df.loc[current_idx, 'entry_price']
                    stop_loss = df.loc[current_idx, 'stop_loss']
                    target_price_col = 'target_price_long' if signal == 'Long' else 'target_price_short'
                    target_price = df.loc[current_idx, target_price_col] # Get target set at entry

                    if pd.notna(entry_price) and pd.notna(stop_loss):
                        current_trade = {
                            'entry_index': i,
                            'entry_idx_time': current_idx,
                            'entry_price': entry_price,
                            'position': signal,
                            'stop_loss': stop_loss,
                            'target_price': target_price, # Store potential target
                            'signal_type': df.loc[current_idx, 'entry_signal_type']
                        }
                        # Calculate risk/reward ratio at entry
                        if current_trade['position'] == 'Long':
                            risk = current_trade['entry_price'] - current_trade['stop_loss']
                            reward = current_trade['target_price'] - current_trade['entry_price'] if pd.notna(current_trade['target_price']) else np.nan
                        else: # Short
                            risk = current_trade['stop_loss'] - current_trade['entry_price']
                            reward = current_trade['entry_price'] - current_trade['target_price'] if pd.notna(current_trade['target_price']) else np.nan

                        current_trade['risk_at_entry'] = risk
                        current_trade['reward_at_entry'] = reward
                        current_trade['risk_reward_ratio'] = reward / risk if risk > 0 and pd.notna(reward) else np.nan

                # Exit Signal Found
                elif current_trade and 'Exit' in signal:
                    exit_price = df.loc[current_idx, 'exit_price']
                    exit_idx_time = current_idx
                    exit_reason = df.loc[current_idx, 'exit_reason']

                    if pd.isna(exit_price): # Fallback if exit price missing
                        exit_price = df.loc[current_idx, 'close']
                        logger.warning(f"Exit price NaN for trade entered on {current_trade['entry_idx_time'].date()}, using close price {exit_price:.2f}")

                    # Calculate P/L (Points) - Costs applied later if needed
                    if current_trade['position'] == 'Long':
                        pnl_points = exit_price - current_trade['entry_price']
                    else: # Short
                        pnl_points = current_trade['entry_price'] - exit_price

                    # Apply simple commission/slippage (as points approx)
                    # Note: This is simplified. Real calculation needs position size.
                    commission = (current_trade['entry_price'] + exit_price) * PARAMS['backtest']['commission_pct']
                    # Slippage already applied in process_signals entry/exit price setting
                    net_pnl_points = pnl_points - commission

                    pct_change = (net_pnl_points / current_trade['entry_price']) * 100 if current_trade['entry_price'] else 0
                    duration_delta = exit_idx_time - current_trade['entry_idx_time']
                    duration_bars = i - current_trade['entry_index'] # Duration in bars

                    outcome = 'Win' if net_pnl_points > 0 else 'Loss'

                    trade_record = {
                        **current_trade,
                        'exit_idx_time': exit_idx_time,
                        'exit_price': exit_price,
                        'pnl_points': net_pnl_points,
                        'pct_change': pct_change,
                        'duration_delta': duration_delta,
                        'duration_bars': duration_bars,
                        'outcome': outcome,
                        'exit_reason': exit_reason
                    }
                    self.trade_history.append(trade_record)

                    # --- Update Summary Stats ---
                    self.summary_stats['total_trades'] += 1
                    self.summary_stats['total_pnl'] += net_pnl_points
                    # total_pnl_percent requires capital tracking, skip for now

                    if outcome == 'Win':
                        self.summary_stats['winning_trades'] += 1
                        win_pnls.append(net_pnl_points)
                        win_durations.append(duration_bars)
                    else:
                        self.summary_stats['losing_trades'] += 1
                        loss_pnls.append(abs(net_pnl_points)) # Use absolute loss
                        loss_durations.append(duration_bars)

                    if pd.notna(current_trade['risk_reward_ratio']):
                         risk_rewards.append(current_trade['risk_reward_ratio'])

                    # Update streaks
                    if outcome == current_streak_type: current_streak += 1
                    else: current_streak = 1; current_streak_type = outcome
                    if outcome == 'Win': max_consec_wins = max(max_consec_wins, current_streak)
                    else: max_consec_losses = max(max_consec_losses, current_streak)

                    # Update running PnL for drawdown
                    running_pnl += net_pnl_points
                    peak_pnl = max(peak_pnl, running_pnl)
                    drawdown = peak_pnl - running_pnl
                    self.summary_stats['max_drawdown_points'] = max(self.summary_stats.get('max_drawdown_points', 0.0), drawdown)

                    # --- Update Detailed Stats ---
                    pos = current_trade['position']
                    sig_type = current_trade['signal_type']
                    exit_type = exit_reason

                    self.entry_stats[pos]['total'] += 1
                    self.entry_stats[pos]['total_pnl'] += net_pnl_points
                    self.entry_stats[pos]['total_duration_bars'] += duration_bars
                    if outcome == 'Win': self.entry_stats[pos]['success'] += 1
                    # Check SL/Target Hit (Approximate check)
                    if exit_type == 'Trailing Stop' or exit_type == 'ATR Stop': self.entry_stats[pos]['sl_hit'] += 1
                    if exit_type == 'Fib Target' or exit_type == 'Target': self.entry_stats[pos]['target_hit'] += 1

                    self.exit_stats[exit_type]['total'] += 1
                    self.exit_stats[exit_type]['total_pnl'] += net_pnl_points
                    self.exit_stats[exit_type]['total_duration_bars'] += duration_bars
                    if outcome == 'Win': self.exit_stats[exit_type]['success'] += 1

                    self.signal_stats[sig_type]['total'] += 1
                    self.signal_stats[sig_type]['total_pnl'] += net_pnl_points
                    if outcome == 'Win': self.signal_stats[sig_type]['success'] += 1

                    current_trade = None # Reset current trade

            # --- Finalize Summary Metrics ---
            if self.summary_stats['total_trades'] > 0:
                avg_win = np.mean(win_pnls) if win_pnls else 0
                avg_loss = np.mean(loss_pnls) if loss_pnls else 0
                self.summary_stats['win_rate'] = (self.summary_stats['winning_trades'] / self.summary_stats['total_trades']) * 100
                self.summary_stats['avg_win_points'] = avg_win
                self.summary_stats['avg_loss_points'] = avg_loss
                self.summary_stats['profit_factor'] = abs(sum(win_pnls) / sum(loss_pnls)) if sum(loss_pnls) != 0 else np.inf
                self.summary_stats['expectancy_points'] = (avg_win * (self.summary_stats['win_rate']/100)) - (avg_loss * (1 - self.summary_stats['win_rate']/100))
                self.summary_stats['max_consec_wins'] = max_consec_wins
                self.summary_stats['max_consec_losses'] = max_consec_losses
                self.summary_stats['avg_win_duration_bars'] = np.mean(win_durations) if win_durations else 0
                self.summary_stats['avg_loss_duration_bars'] = np.mean(loss_durations) if loss_durations else 0
                self.summary_stats['avg_risk_reward_ratio'] = np.nanmean(risk_rewards) if risk_rewards else np.nan # Use nanmean

                # Health Score (Example)
                health_score = min(100, max(0,
                    (self.summary_stats['win_rate'] * 0.4) +
                    (min(self.summary_stats['profit_factor'], 5) * 10 if pd.notna(self.summary_stats['profit_factor']) else 0) +
                    (1 - (self.summary_stats['avg_loss_duration_bars']/self.summary_stats['avg_win_duration_bars'] if self.summary_stats['avg_win_duration_bars'] > 0 else 1) * 20) +
                    (self.summary_stats['avg_risk_reward_ratio'] * 10 if pd.notna(self.summary_stats['avg_risk_reward_ratio']) else 0)
                ))
                self.summary_stats['health_score'] = health_score

            # --- Finalize Detailed Stats (Averages) ---
            for pos in list(self.entry_stats.keys()): # Use list to avoid dict size change error
                if self.entry_stats[pos]['total'] > 0:
                    self.entry_stats[pos]['avg_pnl'] = self.entry_stats[pos]['total_pnl'] / self.entry_stats[pos]['total']
                    self.entry_stats[pos]['avg_duration_bars'] = self.entry_stats[pos]['total_duration_bars'] / self.entry_stats[pos]['total']
                    self.entry_stats[pos]['win_rate'] = (self.entry_stats[pos]['success'] / self.entry_stats[pos]['total']) * 100
                    self.entry_stats[pos]['sl_rate'] = (self.entry_stats[pos]['sl_hit'] / self.entry_stats[pos]['total']) * 100
                    self.entry_stats[pos]['target_rate'] = (self.entry_stats[pos]['target_hit'] / self.entry_stats[pos]['total']) * 100

            for exit_type in list(self.exit_stats.keys()):
                 if self.exit_stats[exit_type]['total'] > 0:
                    self.exit_stats[exit_type]['avg_pnl'] = self.exit_stats[exit_type]['total_pnl'] / self.exit_stats[exit_type]['total']
                    self.exit_stats[exit_type]['avg_duration_bars'] = self.exit_stats[exit_type]['total_duration_bars'] / self.exit_stats[exit_type]['total']
                    self.exit_stats[exit_type]['win_rate'] = (self.exit_stats[exit_type]['success'] / self.exit_stats[exit_type]['total']) * 100

            for sig_type in list(self.signal_stats.keys()):
                 if self.signal_stats[sig_type]['total'] > 0:
                    self.signal_stats[sig_type]['avg_pnl'] = self.signal_stats[sig_type]['total_pnl'] / self.signal_stats[sig_type]['total']
                    self.signal_stats[sig_type]['win_rate'] = (self.signal_stats[sig_type]['success'] / self.signal_stats[sig_type]['total']) * 100

            logger.info("Finished analyzing trades.")
            # Return original df, analysis is stored in the object
            return df

        except Exception as e:
            logger.error(f"Error in analyze_trades: {str(e)}", exc_info=True)
            raise

    def print_summary(self):
        """Print comprehensive performance statistics."""
        if not self.summary_stats or self.summary_stats['total_trades'] == 0:
             print("\nNo trades to summarize.")
             logger.info("No trades to summarize.")
             return
        try:
            summary = [
                "\n=== ENHANCED TRADE SUMMARY ===",
                f"Total Trades: {int(self.summary_stats['total_trades'])}",
                f"Winning Trades: {int(self.summary_stats['winning_trades'])} ({self.summary_stats['win_rate']:.1f}%)",
                f"Losing Trades: {int(self.summary_stats['losing_trades'])}",
                f"Total P&L (Points): {self.summary_stats['total_pnl']:.2f}",
                # f"Total P&L %: {self.summary_stats['total_pnl_percent']:.2f}%", # Needs capital
                f"Profit Factor: {self.summary_stats['profit_factor']:.2f}",
                f"Expectancy (Points): {self.summary_stats['expectancy_points']:.2f}",
                f"Max Drawdown (Points): {self.summary_stats['max_drawdown_points']:.2f}",
                f"Avg Win / Avg Loss (Points): {self.summary_stats['avg_win_points']:.2f} / {self.summary_stats['avg_loss_points']:.2f}",
                f"Max Consecutive Wins: {int(self.summary_stats['max_consec_wins'])}",
                f"Max Consecutive Losses: {int(self.summary_stats['max_consec_losses'])}",
                f"Avg Win Duration (Bars): {self.summary_stats['avg_win_duration_bars']:.1f}",
                f"Avg Loss Duration (Bars): {self.summary_stats['avg_loss_duration_bars']:.1f}",
                f"Avg Risk/Reward Ratio (at Entry): {self.summary_stats['avg_risk_reward_ratio']:.2f}:1" if pd.notna(self.summary_stats['avg_risk_reward_ratio']) else "N/A",
                f"Strategy Health Score: {self.summary_stats['health_score']:.1f}/100",
                "\n=== ENTRY STATISTICS ===",
                "Position | Total | Win % | Avg P&L | Avg Dur | SL % | Target %",
                "-------------------------------------------------------------"
            ]
            for position, stats in self.entry_stats.items():
                summary.append(
                    f"{position:8} | {int(stats['total']):5} | {stats['win_rate']:5.1f}% | {stats['avg_pnl']:7.2f} | {stats['avg_duration_bars']:7.1f} | {stats['sl_rate']:4.1f}% | {stats['target_rate']:6.1f}%"
                )
            summary.extend([
                "\n=== EXIT STATISTICS ===",
                "Exit Type           | Total | Win % | Avg P&L | Avg Dur",
                "------------------------------------------------------"
            ])
            for exit_type, stats in sorted(self.exit_stats.items()): # Sort for consistency
                summary.append(
                    f"{str(exit_type):19} | {int(stats['total']):5} | {stats['win_rate']:5.1f}% | {stats['avg_pnl']:7.2f} | {stats['avg_duration_bars']:7.1f}"
                )
            summary.extend([
                "\n=== SIGNAL TYPE STATISTICS ===",
                "Signal Type          | Total | Win % | Avg P&L",
                "---------------------------------------------"
            ])
            for signal_type, stats in sorted(self.signal_stats.items()): # Sort for consistency
                 if not signal_type: continue # Skip empty signal types if any
                 summary.append(
                    f"{str(signal_type):20} | {int(stats['total']):5} | {stats['win_rate']:5.1f}% | {stats['avg_pnl']:7.2f}"
                )

            print("\n".join(summary))
            logger.info("\n".join(summary))

            health = self.summary_stats['health_score']
            if health >= 70: health_indicator = "🟢 STRONG"
            elif health >= 50: health_indicator = "🟡 MODERATE"
            else: health_indicator = "🔴 WEAK"
            print(f"\nStrategy Health: {health_indicator}")
            if self.summary_stats['total_trades'] < 30: print("⚠️  Warning: Low sample size (<30 trades)")

        except Exception as e:
            logger.error(f"Error printing summary: {str(e)}", exc_info=True)
            raise

# --- Plotting Function (Simplified - No Equity Curve) ---
def plot_signals(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Visualize the signals with price and indicators."""
    logger.info("Generating signal plot")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(20, 15)) # 4 subplots

    # --- Price Chart ---
    ax1 = axes[0]
    ax1.plot(df['close'], label='Price', color='black', alpha=0.9, linewidth=1.0)
    ax1.plot(df['ema_fast'], label=f"EMA {PARAMS['ema']['fast_len']}", alpha=0.6, linewidth=0.8)
    ax1.plot(df['ema_med'], label=f"EMA {PARAMS['ema']['med_len']}", alpha=0.6, linewidth=0.8)
    ax1.plot(df['ema_slow'], label=f"EMA {PARAMS['ema']['slow_len']}", alpha=0.6, linewidth=0.8)
    long_entries = df[df['signal'] == 'Long']; short_entries = df[df['signal'] == 'Short']
    long_exits = df[df['signal'].str.contains('Exit Long', na=False)]; short_exits = df[df['signal'].str.contains('Exit Short', na=False)]
    ax1.scatter(long_entries.index, long_entries['entry_price'], marker='^', color='lime', s=100, label='Long Entry', zorder=5, edgecolors='black')
    ax1.scatter(short_entries.index, short_entries['entry_price'], marker='v', color='red', s=100, label='Short Entry', zorder=5, edgecolors='black')
    ax1.scatter(long_exits.index, long_exits['exit_price'], marker='x', color='fuchsia', s=80, label='Exit', zorder=5) # Combined exit label
    ax1.scatter(short_exits.index, short_exits['exit_price'], marker='x', color='fuchsia', s=80, zorder=5)
    ax1.plot(df['trailing_stop'], label='Trailing Stop', linestyle='--', color='purple', alpha=0.7, linewidth=1.0)
    ax1.scatter(df.index, df['target_price_long'], marker='_', color='blue', alpha=0.5, s=50, label='Long Target')
    ax1.scatter(df.index, df['target_price_short'], marker='_', color='orange', alpha=0.5, s=50, label='Short Target')
    ax1.legend(loc='upper left'); ax1.set_title('Price, EMAs, Signals, Stops & Targets'); ax1.set_ylabel('Price'); ax1.grid(True, which='both', linestyle=':', linewidth=0.5)

    # --- Score Chart ---
    ax_score = axes[1]
    ax_score.plot(df['scaled_score'], label='Scaled Score (0-10)', color='teal', linewidth=1.5)
    ax_score.axhline(PARAMS['core']['entry_score_threshold'], color='green', linestyle='--', alpha=0.5, label=f'Long Entry Thresh')
    ax_score.axhline(10.0 - PARAMS['core']['entry_score_threshold'], color='red', linestyle='--', alpha=0.5, label=f'Short Entry Thresh')
    if PARAMS['core']['use_score_drop_exit']:
        ax_score.axhline(5.0 - PARAMS['core']['exit_score_drop_threshold'], color='red', linestyle=':', alpha=0.4, label=f'Long Exit Drop')
        ax_score.axhline(5.0 + PARAMS['core']['exit_score_drop_threshold'], color='green', linestyle=':', alpha=0.4, label=f'Short Exit Drop')
    ax_score.set_ylim(0, 10); ax_score.legend(loc='upper left'); ax_score.set_title('Confidence Score'); ax_score.set_ylabel('Score'); ax_score.grid(True, which='both', linestyle=':', linewidth=0.5)

    # --- RSI Chart ---
    ax_rsi = axes[2]
    ax_rsi.plot(df['rsi'], label='RSI', color='orange', linewidth=1.0)
    ax_rsi.axhline(50, linestyle='--', color='gray', alpha=0.5); ax_rsi.axhline(PARAMS['rsi']['rsi_buy_level'], linestyle=':', color='green', alpha=0.4); ax_rsi.axhline(PARAMS['rsi']['rsi_sell_level'], linestyle=':', color='red', alpha=0.4)
    bull_div = df[df['bullish_rsi_div']]; bear_div = df[df['bearish_rsi_div']]
    ax_rsi.scatter(bull_div.index, bull_div['rsi'] * 0.98, marker='^', color='cyan', s=50, label='Bullish Div', zorder=5)
    ax_rsi.scatter(bear_div.index, bear_div['rsi'] * 1.02, marker='v', color='magenta', s=50, label='Bearish Div', zorder=5)
    ax_rsi.legend(loc='upper left'); ax_rsi.set_title('RSI & Divergence'); ax_rsi.set_ylabel('RSI'); ax_rsi.grid(True, which='both', linestyle=':', linewidth=0.5)

    # --- MACD Chart ---
    ax_macd = axes[3]
    ax_macd.plot(df['macd'], label='MACD', color='blue', linewidth=1.0); ax_macd.plot(df['macd_signal'], label='Signal', color='red', alpha=0.8, linewidth=1.0)
    colors = ['g' if v >= 0 else 'r' for v in df['macd_hist']]; ax_macd.bar(df.index, df['macd_hist'], label='Histogram', color=colors, alpha=0.5)
    ax_macd.axhline(0, linestyle='--', color='gray', alpha=0.5); ax_macd.legend(loc='upper left'); ax_macd.set_title('MACD'); ax_macd.set_ylabel('MACD'); ax_macd.grid(True, which='both', linestyle=':', linewidth=0.5)

    # Common x-axis label and formatting
    axes[-1].tick_params(axis='x', rotation=45)
    plt.xlabel('Date')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.suptitle('Trading Strategy Signals and Indicators', fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved signal plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)

# --- Command Line Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='Generate enhanced trading signals and calculate detailed stats.')
    parser.add_argument('--input', type=str, default=str(DEFAULT_INPUT_FILE), help=f'Input CSV (default: {DEFAULT_INPUT_FILE})')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT_FILE), help=f'Output CSV (default: {DEFAULT_OUTPUT_FILE})')
    parser.add_argument('--plot', type=str, default=str(DEFAULT_PLOT_FILE), help=f'Plot image path (default: {DEFAULT_PLOT_FILE})')
    parser.add_argument('--no-plot', action='store_true', help='Disable generating plot')
    parser.add_argument('--full-history', action='store_true', help='Run on full history')
    return parser.parse_args()

# --- Main Execution Block ---
if __name__ == "__main__":
    args = parse_args()
    input_file = Path(args.input); output_file = Path(args.output); plot_file = Path(args.plot)

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if not args.no_plot: plot_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading data from {input_file}")
        if not input_file.exists(): raise FileNotFoundError(f"Input file not found: {input_file}")
        df_full = pd.read_csv(input_file, parse_dates=['datetime'], index_col='datetime')
        if df_full.empty: raise pd.errors.EmptyDataError("Input file is empty.")
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df_full.columns for col in required_cols): raise ValueError(f"Input CSV must contain: {', '.join(required_cols)}")

        # --- Filter Data ---
        if PARAMS['backtest']['filter_last_month'] and not args.full_history:
            if not df_full.empty:
                last_date = df_full.index.max(); one_month_prior = last_date - pd.DateOffset(months=1)
                df = df_full.loc[one_month_prior:].copy()
                logger.info(f"Filtered data from {df.index.min().date()} to {df.index.max().date()}.")
            else: df = df_full
        else:
            df = df_full.copy()
            logger.info(f"Using full data history from {df.index.min().date()} to {df.index.max().date()}.")

        if df.empty:
             logger.warning("DataFrame empty after filtering. Skipping processing.")
             df_processed = df
             analyzer = None # No analyzer needed
        else:
            logger.info("Processing signals...")
            df_processed = process_signals(df) # Generate signals

            logger.info("Analyzing trades...")
            analyzer = EnhancedSignalAnalyzer() # Instantiate analyzer
            analyzer.analyze_trades(df_processed) # Analyze the processed df
            analyzer.print_summary() # Print detailed stats

        logger.info(f"Saving results to {output_file}")
        df_processed.to_csv(output_file) # Save df with signals

        # --- Plotting ---
        if not args.no_plot and not df_processed.empty:
            plot_signals(df_processed, plot_file) # Use simplified plot function
        elif not args.no_plot and df_processed.empty:
             logger.warning("Plotting skipped as no data.")

        logger.info("Processing completed successfully")
        sys.exit(0) # Success exit code

    except FileNotFoundError as e: logger.error(str(e)); sys.exit(1)
    except pd.errors.EmptyDataError as e: logger.error(f"Input file '{input_file}' empty/corrupt."); sys.exit(1)
    except ValueError as e: logger.error(f"Data validation error: {str(e)}"); sys.exit(1)
    except Exception as e: logger.error(f"Unexpected error: {str(e)}", exc_info=True); sys.exit(1)