import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def process_signals(df):
    # --- Initialize Columns ---
    # Original signals
    df['signal_type'] = 'Hold'
    df['signal_strength'] = 0.0
    df['signal_reason'] = ''
    df['position'] = 'Hold'
    df['entry_price'] = np.nan
    df['stop_loss'] = np.nan
    df['target_price'] = np.nan
    df['fib_retracement'] = np.nan
    # P/L and Exit Tracking
    df['exit_reason'] = ''
    df['trade_outcome'] = ''
    df['profit_loss'] = np.nan
    # Trade Numbering
    df['trade_id'] = '' # Will store e.g., 'L1', 'S1', 'L2'
    # Individual Score Components
    score_components = [
        'score_ema_trend', 'score_ema_signal', 'score_rsi_thresh', 'score_macd_signal',
        'score_macd_zero', 'score_vol_break', 'score_adx_strength', 'score_adx_direction',
        'score_htf_trend', 'score_fib_bounce', 'score_ema_bounce', 'score_bb_bounce'
    ]
    for col in score_components:
        df[col] = 0.0

    # --- Initialize Tracking Dictionaries (for stats) ---
    # (Stats dictionaries remain the same as before)
    entry_signal_stats = {
        'Fib': {'total': 0, 'success': 0, 'sl': 0, 'target': 0},
        'EMA': {'total': 0, 'success': 0, 'sl': 0, 'target': 0},
        'BB': {'total': 0, 'success': 0, 'sl': 0, 'target': 0},
        'Vol': {'total': 0, 'success': 0, 'sl': 0, 'target': 0}
    }
    exit_signal_stats = {
        'ATR SL': {'total': 0, 'success': 0},
        'Score Drop': {'total': 0, 'success': 0},
        'EMA Cross': {'total': 0, 'success': 0},
        'BB Mid': {'total': 0, 'success': 0},
        'Fib Target': {'total': 0, 'success': 0},
        'RSI Div': {'total': 0, 'success': 0},
        'Vol Fade': {'total': 0, 'success': 0} # Added Vol Fade stat
    }


    # --- Initialize State Variables ---
    in_long = False
    in_short = False
    entry_price = np.nan
    stop_loss = np.nan
    target_price = np.nan
    entry_signal_type = '' # Type of signal that triggered entry (Fib, EMA etc)
    last_pivot_high = np.nan
    last_pivot_low = np.nan
    last_pivot_high_idx = -1
    last_pivot_low_idx = -1
    trade_count_long = 0
    trade_count_short = 0
    current_trade_id = ''

    # --- Parameters (Match Pine Script inputs & your settings) ---
    # Core
    exit_score_drop_threshold = 1.5
    use_fib_bounce_entry = True # Pine: useFibBounceEntry
    use_fib_bounce_sell = True # Pine: useFibBounceSell
    fib_bounce_lookback = 3    # Pine: fibBounceLookback
    use_ema_bounce_buy = True  # Pine: useEmaBounceBuy
    use_ema_bounce_sell = True # Pine: useEmaBounceSell
    ema_bounce_lookback = 2    # Pine: emaBounceLookback
    ema_bounce_source_str = "Fast EMA" # Pine: emaBounceSource
    use_bb_mid_bounce_buy = True # Pine: useBbMidBounceBuy
    use_bb_mid_bounce_sell = True# Pine: useBbMidBounceSell
    bb_bounce_lookback = 2     # Pine: bbBounceLookback
    use_vol_breakout_buy = True# Pine: useVolBreakoutBuy
    use_vol_breakout_sell = True# Pine: useVolBreakoutSell (Default)

    # EMAs
    ema_fast_len = 9
    ema_med_len = 14
    ema_slow_len = 21
    use_ema_exit = True

    # Bollinger Bands
    bb_len = 20
    bb_std_dev = 2.0
    use_bb_return_exit = True # Pine: useBBReturnExit

    # RSI
    rsi_len = 14
    rsi_buy_level = 55.0
    rsi_sell_level = 45.0
    use_rsi_div_exit = False # Pine default is False
    rsi_confirm_fib = True   # Pine: rsiConfirmFibBounce
    rsi_confirm_ema = False  # Pine: rsiConfirmEmaBounce
    rsi_confirm_bb = False   # Pine: rsiConfirmBbBounce

    # MACD
    macd_fast_len = 12
    macd_slow_len = 26
    macd_signal_len = 9

    # Volume
    vol_ma_len = 50
    vol_multiplier = 1.5
    use_vol_fade_exit = True # Pine: useVolFadeExit

    # ATR
    atr_len = 14
    atr_mult = 2.0
    use_atr_stop = True # Pine: useAtrStop

    # Fibonacci Exit Target
    use_fib_exit = True # Pine: useFibExit
    fib_lookback_exit = 30
    fib_extension_level = 1.618

    # Fibonacci Retracement
    fib_pivot_lookback = 15
    fib_max_bars = 200

    # Trend Filters
    use_ema_trend_filter = True
    use_adx_filter = True
    adx_len = 14
    adx_threshold = 20.0
    use_adx_direction_filter = True
    # HTF Filter not implemented

    # Score Weights
    weights = {
        'ema_trend': 2, 'ema_signal': 1, 'rsi_thresh': 1, 'macd_signal': 1,
        'macd_zero': 1, 'vol_break': 1, 'adx_strength': 1, 'adx_direction': 1,
        'htf_trend': 0, 'fib_bounce': 2, 'ema_bounce': 1, 'bb_bounce': 1
    }
    total_possible_score = sum(w for key, w in weights.items() if key != 'htf_trend')

    # --- Calculate Indicators ---
    # Ensure column names match exactly what's used later
    ema_fast_col = 'ema_fast'
    ema_med_col = 'ema_med'
    ema_slow_col = 'ema_slow'
    bb_middle_col = 'bb_middle'
    bb_upper_col = 'bb_upper'
    bb_lower_col = 'bb_lower'
    rsi_col = 'rsi'
    macd_col = 'macd'
    macd_signal_col = 'macd_signal'
    macd_hist_col = 'macd_hist'
    vol_ma_col = 'vol_ma'
    atr_col = 'atr'
    plus_di_col = 'plus_di'
    minus_di_col = 'minus_di'
    adx_col = 'adx'


    df[ema_fast_col] = df['close'].ewm(span=ema_fast_len, adjust=False).mean()
    df[ema_med_col] = df['close'].ewm(span=ema_med_len, adjust=False).mean()
    df[ema_slow_col] = df['close'].ewm(span=ema_slow_len, adjust=False).mean()

    df[bb_middle_col] = df['close'].rolling(bb_len).mean()
    rolling_std = df['close'].rolling(bb_len).std()
    df[bb_upper_col] = df[bb_middle_col] + rolling_std * bb_std_dev
    df[bb_lower_col] = df[bb_middle_col] - rolling_std * bb_std_dev

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/rsi_len, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_len, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1)
    df[rsi_col] = 100.0 - (100.0 / (1.0 + rs))
    df[rsi_col].fillna(50, inplace=True)

    if macd_col not in df.columns or macd_signal_col not in df.columns or macd_hist_col not in df.columns:
        ema_fast_macd = df['close'].ewm(span=macd_fast_len, adjust=False).mean()
        ema_slow_macd = df['close'].ewm(span=macd_slow_len, adjust=False).mean()
        df[macd_col] = ema_fast_macd - ema_slow_macd
        df[macd_signal_col] = df[macd_col].ewm(span=macd_signal_len, adjust=False).mean()
        df[macd_hist_col] = df[macd_col] - df[macd_signal_col]

    df[vol_ma_col] = df['volume'].rolling(vol_ma_len).mean()

    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift())
    low_close_prev = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df[atr_col] = tr.ewm(alpha=1/atr_len, adjust=False).mean()

    if use_adx_filter or use_adx_direction_filter:
        up_move = df['high'].diff()
        down_move = -df['low'].diff()
        plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
        plus_dm.fillna(0, inplace=True)
        minus_dm.fillna(0, inplace=True)
        atr_adx = df[atr_col].replace(0, 1) # Avoid division by zero
        smooth_plus_dm = plus_dm.ewm(alpha=1/adx_len, adjust=False).mean()
        smooth_minus_dm = minus_dm.ewm(alpha=1/adx_len, adjust=False).mean()
        df[plus_di_col] = 100 * (smooth_plus_dm / atr_adx)
        df[minus_di_col] = 100 * (smooth_minus_dm / atr_adx)
        di_sum = (df[plus_di_col] + df[minus_di_col]).replace(0, 1)
        dx = 100 * (abs(df[plus_di_col] - df[minus_di_col]) / di_sum)
        df[adx_col] = dx.ewm(alpha=1/adx_len, adjust=False).mean()
        df.fillna({adx_col: 0, plus_di_col: 0, minus_di_col: 0}, inplace=True)
    else: # Ensure columns exist even if filter is off
        df[plus_di_col] = 0.0
        df[minus_di_col] = 0.0
        df[adx_col] = 0.0


    # --- Main Processing Loop ---
    start_index = max(bb_len, ema_slow_len, macd_slow_len, vol_ma_len, atr_len, adx_len if (use_adx_filter or use_adx_direction_filter) else 0, fib_pivot_lookback, fib_lookback_exit) + 1

    for i in range(start_index, len(df)):
        # Use .loc for potentially better performance and clearer setting of values
        current_index = df.index[i]
        prev_index = df.index[i-1]

        current = df.loc[current_index]
        prev = df.loc[prev_index]

        # --- Fibonacci Pivot Detection & Level Calculation ---
        # (Keeping simplified pivot logic from previous version)
        lookback_data = df.iloc[max(0, i - fib_pivot_lookback*2):i]
        pivot_high_price = lookback_data['high'].max()
        pivot_low_price = lookback_data['low'].min()
        is_uptrend_fib = current[ema_slow_col] > prev[ema_slow_col]

        level_0, level_236, level_382, level_500, level_618, level_786, level_100 = (np.nan,) * 7
        valid_fib_range = False
        fib_range = np.nan

        if not np.isnan(pivot_high_price) and not np.isnan(pivot_low_price) and pivot_high_price > pivot_low_price:
            valid_fib_range = True
            fib_range = pivot_high_price - pivot_low_price
            if is_uptrend_fib: level_0, level_100 = pivot_low_price, pivot_high_price
            else: level_0, level_100 = pivot_high_price, pivot_low_price
            level_236 = level_0 + (level_100 - level_0) * 0.236
            level_382 = level_0 + (level_100 - level_0) * 0.382
            level_500 = level_0 + (level_100 - level_0) * 0.500
            level_618 = level_0 + (level_100 - level_0) * 0.618
            level_786 = level_0 + (level_100 - level_0) * 0.786
            df.loc[current_index, 'fib_retracement'] = abs(current['close'] - level_0) / fib_range if fib_range > 0 else np.nan

        # --- Basic Conditions ---
        cond_ema_fast_slow_cross_buy = current[ema_fast_col] > current[ema_slow_col] and prev[ema_fast_col] <= prev[ema_slow_col]
        cond_ema_fast_slow_cross_sell = current[ema_fast_col] < current[ema_slow_col] and prev[ema_fast_col] >= prev[ema_slow_col]
        cond_ema_fast_med_cross_buy = current[ema_fast_col] > current[ema_med_col] and prev[ema_fast_col] <= prev[ema_med_col]
        cond_ema_fast_med_cross_sell = current[ema_fast_col] < current[ema_med_col] and prev[ema_fast_col] >= prev[ema_med_col]
        cond_bb_return_mean_buy = current['close'] > current[bb_middle_col] and prev['close'] <= prev[bb_middle_col]
        cond_bb_return_mean_sell = current['close'] < current[bb_middle_col] and prev['close'] >= prev[bb_middle_col]
        cond_rsi_buy = current[rsi_col] > rsi_buy_level
        cond_rsi_sell = current[rsi_col] < rsi_sell_level
        cond_macd_signal_cross_buy = current[macd_col] > current[macd_signal_col] and prev[macd_col] <= prev[macd_signal_col]
        cond_macd_signal_cross_sell = current[macd_col] < current[macd_signal_col] and prev[macd_col] >= prev[macd_signal_col]
        cond_macd_zero_cross_buy = current[macd_col] > 0 and prev[macd_col] <= 0
        cond_macd_zero_cross_sell = current[macd_col] < 0 and prev[macd_col] >= 0
        cond_high_vol = current['volume'] > current[vol_ma_col] * vol_multiplier
        cond_vol_breakout_buy = cond_high_vol and current['close'] > current['open'] and current['close'] > current[ema_slow_col]
        cond_vol_breakout_sell = cond_high_vol and current['close'] < current['open'] and current['close'] < current[ema_slow_col]
        cond_vol_fade_long = current['close'] < current[ema_fast_col] and current['volume'] < current[vol_ma_col]
        cond_vol_fade_short = current['close'] > current[ema_fast_col] and current['volume'] < current[vol_ma_col]

        # --- Filter Conditions ---
        cond_ema_trend_ok_buy = not use_ema_trend_filter or current[ema_med_col] > current[ema_slow_col]
        cond_ema_trend_ok_sell = not use_ema_trend_filter or current[ema_med_col] < current[ema_slow_col]
        cond_adx_strength_ok = not use_adx_filter or current[adx_col] > adx_threshold
        cond_adx_direction_ok_buy = not use_adx_direction_filter or current[plus_di_col] > current[minus_di_col]
        cond_adx_direction_ok_sell = not use_adx_direction_filter or current[minus_di_col] > current[plus_di_col]
        cond_adx_filter_ok_buy = cond_adx_strength_ok and cond_adx_direction_ok_buy
        cond_adx_filter_ok_sell = cond_adx_strength_ok and cond_adx_direction_ok_sell
        # HTF Filter not implemented

        all_filters_ok_buy = cond_ema_trend_ok_buy and cond_adx_filter_ok_buy
        all_filters_ok_sell = cond_ema_trend_ok_sell and cond_adx_filter_ok_sell

        # --- Bounce Conditions ---
        # Fib Bounce
        touched_fib_zone_buy = False
        if use_fib_bounce_entry and is_uptrend_fib and valid_fib_range:
            for j in range(1, fib_bounce_lookback + 1):
                if i - j >= 0 and df.iloc[i - j]['low'] <= level_618:
                    touched_fib_zone_buy = True; break
        bounced_above_50_buy = current['close'] > level_500 if valid_fib_range else False
        rsi_confirms_fib_buy = not rsi_confirm_fib or (current[rsi_col] > 40 and current[rsi_col] > prev[rsi_col])
        cond_fib_bounce_buy = use_fib_bounce_entry and is_uptrend_fib and touched_fib_zone_buy and bounced_above_50_buy and rsi_confirms_fib_buy

        touched_fib_zone_sell = False
        if use_fib_bounce_sell and not is_uptrend_fib and valid_fib_range:
             for j in range(1, fib_bounce_lookback + 1):
                if i - j >= 0 and df.iloc[i - j]['high'] >= level_382:
                    touched_fib_zone_sell = True; break
        rejected_below_50_sell = current['close'] < level_500 if valid_fib_range else False
        rsi_confirms_fib_sell = not rsi_confirm_fib or (current[rsi_col] < 60 and current[rsi_col] < prev[rsi_col])
        cond_fib_bounce_sell = use_fib_bounce_sell and not is_uptrend_fib and valid_fib_range and touched_fib_zone_sell and rejected_below_50_sell and rsi_confirms_fib_sell

        # --- FIX FOR 'fast_ema' ERROR: Use explicit column names ---
        if ema_bounce_source_str == "Fast EMA":
            ema_bounce_col = ema_fast_col
        elif ema_bounce_source_str == "Medium EMA":
            ema_bounce_col = ema_med_col
        else: # Default or error case
            ema_bounce_col = ema_fast_col # Defaulting to fast ema

        ema_source_val = current[ema_bounce_col]

        # EMA Bounce Buy
        touched_ema_buy = False
        if use_ema_bounce_buy:
            for j in range(1, ema_bounce_lookback + 1):
                if i - j >= 0 and df.iloc[i - j]['low'] <= df.iloc[i - j][ema_bounce_col]:
                    touched_ema_buy = True; break
        above_ema_before_buy = False
        if i > ema_bounce_lookback :
             # Check if the index exists before accessing
             prev_lookback_index = i - (ema_bounce_lookback + 1)
             if prev_lookback_index >= 0:
                  above_ema_before_buy = df.iloc[prev_lookback_index][ema_bounce_col] < df.iloc[prev_lookback_index]['close']

        rsi_confirms_ema_buy = not rsi_confirm_ema or (current[rsi_col] > 40 and current[rsi_col] > prev[rsi_col])
        cond_ema_bounce_buy = use_ema_bounce_buy and above_ema_before_buy and touched_ema_buy and current['close'] > ema_source_val and current['close'] > current['open'] and rsi_confirms_ema_buy

        # EMA Bounce Sell
        touched_ema_sell = False
        if use_ema_bounce_sell:
             for j in range(1, ema_bounce_lookback + 1):
                 if i - j >= 0 and df.iloc[i - j]['high'] >= df.iloc[i - j][ema_bounce_col]:
                    touched_ema_sell = True; break
        below_ema_before_sell = False
        if i > ema_bounce_lookback :
             prev_lookback_index = i - (ema_bounce_lookback + 1)
             if prev_lookback_index >= 0:
                  below_ema_before_sell = df.iloc[prev_lookback_index][ema_bounce_col] > df.iloc[prev_lookback_index]['close']

        rsi_confirms_ema_sell = not rsi_confirm_ema or (current[rsi_col] < 60 and current[rsi_col] < prev[rsi_col])
        cond_ema_bounce_sell = use_ema_bounce_sell and below_ema_before_sell and touched_ema_sell and current['close'] < ema_source_val and current['close'] < current['open'] and rsi_confirms_ema_sell


        # BB Mid Bounce
        touched_bb_mid_buy = False
        if use_bb_mid_bounce_buy:
             for j in range(1, bb_bounce_lookback + 1):
                 if i - j >= 0 and df.iloc[i - j]['low'] <= df.iloc[i - j][bb_middle_col]:
                    touched_bb_mid_buy = True; break
        rsi_confirms_bb_buy = not rsi_confirm_bb or (current[rsi_col] > 40 and current[rsi_col] > prev[rsi_col])
        cond_bb_mid_bounce_buy = use_bb_mid_bounce_buy and touched_bb_mid_buy and current['close'] > current[bb_middle_col] and current['close'] > current['open'] and rsi_confirms_bb_buy

        touched_bb_mid_sell = False
        if use_bb_mid_bounce_sell:
             for j in range(1, bb_bounce_lookback + 1):
                 if i - j >= 0 and df.iloc[i - j]['high'] >= df.iloc[i - j][bb_middle_col]:
                    touched_bb_mid_sell = True; break
        rsi_confirms_bb_sell = not rsi_confirm_bb or (current[rsi_col] < 60 and current[rsi_col] < prev[rsi_col])
        cond_bb_mid_bounce_sell = use_bb_mid_bounce_sell and touched_bb_mid_sell and current['close'] < current[bb_middle_col] and current['close'] < current['open'] and rsi_confirms_bb_sell

        # --- Score Calculation & Individual Components ---
        # (Score calculation logic remains the same as previous version,
        #  but uses explicit column names like current[rsi_col] etc.)
        buy_score_bar = 0.0
        sell_score_bar = 0.0
        bull_reason_bar = ""
        bear_reason_bar = ""
        current_scores = {col: 0.0 for col in score_components}

        comp_score = weights['ema_trend'] if cond_ema_trend_ok_buy else (-weights['ema_trend'] if cond_ema_trend_ok_sell else 0)
        current_scores['score_ema_trend'] = comp_score
        if comp_score > 0: buy_score_bar += comp_score; bull_reason_bar += "ET+"
        elif comp_score < 0: sell_score_bar -= comp_score; bear_reason_bar += "ET-"

        comp_score = weights['ema_signal'] if cond_ema_fast_slow_cross_buy else (-weights['ema_signal'] if cond_ema_fast_slow_cross_sell else 0)
        current_scores['score_ema_signal'] = comp_score
        if comp_score > 0: buy_score_bar += comp_score; bull_reason_bar += "ES+"
        elif comp_score < 0: sell_score_bar -= comp_score; bear_reason_bar += "ES-"

        comp_score = weights['rsi_thresh'] if cond_rsi_buy else (-weights['rsi_thresh'] if cond_rsi_sell else 0)
        current_scores['score_rsi_thresh'] = comp_score
        if comp_score > 0: buy_score_bar += comp_score; bull_reason_bar += "R+"
        elif comp_score < 0: sell_score_bar -= comp_score; bear_reason_bar += "R-"

        comp_score = weights['macd_signal'] if cond_macd_signal_cross_buy else (-weights['macd_signal'] if cond_macd_signal_cross_sell else 0)
        current_scores['score_macd_signal'] = comp_score
        if comp_score > 0: buy_score_bar += comp_score; bull_reason_bar += "MS+"
        elif comp_score < 0: sell_score_bar -= comp_score; bear_reason_bar += "MS-"

        comp_score = weights['macd_zero'] if cond_macd_zero_cross_buy else (-weights['macd_zero'] if cond_macd_zero_cross_sell else 0)
        current_scores['score_macd_zero'] = comp_score
        if comp_score > 0: buy_score_bar += comp_score; bull_reason_bar += "MZ+"
        elif comp_score < 0: sell_score_bar -= comp_score; bear_reason_bar += "MZ-"

        comp_score = weights['vol_break'] if cond_vol_breakout_buy else (-weights['vol_break'] if cond_vol_breakout_sell else 0)
        current_scores['score_vol_break'] = comp_score
        if comp_score > 0: buy_score_bar += comp_score; bull_reason_bar += "V+"
        elif comp_score < 0: sell_score_bar -= comp_score; bear_reason_bar += "V-"

        comp_score_strength = 0.0
        comp_score_direction = 0.0
        if use_adx_filter: # Only add score if filter is used
            if cond_adx_strength_ok:
                if cond_adx_direction_ok_buy:
                    comp_score_strength = weights['adx_strength']
                    bull_reason_bar += "AS+"
                elif cond_adx_direction_ok_sell:
                    comp_score_strength = -weights['adx_strength']
                    bear_reason_bar += "AS-"
        current_scores['score_adx_strength'] = comp_score_strength
        if comp_score_strength > 0: buy_score_bar += comp_score_strength
        elif comp_score_strength < 0: sell_score_bar -= comp_score_strength

        if use_adx_direction_filter: # Only add score if filter is used
            if cond_adx_direction_ok_buy:
                 comp_score_direction = weights['adx_direction']
                 bull_reason_bar += "AD+"
            elif cond_adx_direction_ok_sell:
                 comp_score_direction = -weights['adx_direction']
                 bear_reason_bar += "AD-"
        current_scores['score_adx_direction'] = comp_score_direction
        if comp_score_direction > 0: buy_score_bar += comp_score_direction
        elif comp_score_direction < 0: sell_score_bar -= comp_score_direction

        current_scores['score_htf_trend'] = 0.0 # HTF not implemented

        comp_score = weights['fib_bounce'] if cond_fib_bounce_buy else (-weights['fib_bounce'] if cond_fib_bounce_sell else 0)
        current_scores['score_fib_bounce'] = comp_score
        if comp_score > 0: buy_score_bar += comp_score; bull_reason_bar += "FB+"
        elif comp_score < 0: sell_score_bar -= comp_score; bear_reason_bar += "FB-"

        comp_score = weights['ema_bounce'] if cond_ema_bounce_buy else (-weights['ema_bounce'] if cond_ema_bounce_sell else 0)
        current_scores['score_ema_bounce'] = comp_score
        if comp_score > 0: buy_score_bar += comp_score; bull_reason_bar += "EB+"
        elif comp_score < 0: sell_score_bar -= comp_score; bear_reason_bar += "EB-"

        comp_score = weights['bb_bounce'] if cond_bb_mid_bounce_buy else (-weights['bb_bounce'] if cond_bb_mid_bounce_sell else 0)
        current_scores['score_bb_bounce'] = comp_score
        if comp_score > 0: buy_score_bar += comp_score; bull_reason_bar += "BB+"
        elif comp_score < 0: sell_score_bar -= comp_score; bear_reason_bar += "BB-"

        # Assign individual scores
        for col, score_val in current_scores.items():
            df.loc[current_index, col] = score_val

        net_score = buy_score_bar - sell_score_bar
        scaled_score = (net_score / total_possible_score) * 5.0 + 5.0 if total_possible_score > 0 else 5.0
        scaled_score = max(0.0, min(10.0, scaled_score))
        df.loc[current_index, 'signal_strength'] = scaled_score
        df.loc[current_index, 'signal_reason'] = bull_reason_bar + bear_reason_bar

        # --- RSI Divergence ---
        bearish_rsi_div = False
        bullish_rsi_div = False
        # (Keeping simplified divergence logic)
        if use_rsi_div_exit and i >= 5:
             rsi_lookback = df[rsi_col].iloc[max(0, i-5):i]
             high_lookback = df['high'].iloc[max(0, i-5):i]
             low_lookback = df['low'].iloc[max(0, i-5):i]
             if not rsi_lookback.empty:
                 if not high_lookback.empty:
                     rsi_peak_idx = rsi_lookback.idxmax()
                     if rsi_peak_idx in df.index: # Check if index exists
                        rsi_peak = df.loc[rsi_peak_idx, rsi_col]
                        price_at_rsi_peak = df.loc[rsi_peak_idx, 'high']
                        if current[rsi_col] < rsi_peak and current['high'] > price_at_rsi_peak: bearish_rsi_div = True
                 if not low_lookback.empty:
                     rsi_trough_idx = rsi_lookback.idxmin()
                     if rsi_trough_idx in df.index: # Check if index exists
                        rsi_trough = df.loc[rsi_trough_idx, rsi_col]
                        price_at_rsi_trough = df.loc[rsi_trough_idx, 'low']
                        if current[rsi_col] > rsi_trough and current['low'] < price_at_rsi_trough: bullish_rsi_div = True

        cond_rsi_bull_div_exit = bearish_rsi_div # Bearish div exits long
        cond_rsi_bear_div_exit = bullish_rsi_div # Bullish div exits short

        # --- Determine Entry Signal Type ---
        is_potential_buy = False
        is_potential_sell = False
        # (Entry condition checks remain the same)
        is_potential_ema_buy = cond_ema_fast_slow_cross_buy and all_filters_ok_buy
        is_potential_fib_buy = cond_fib_bounce_buy and all_filters_ok_buy
        is_potential_ema_bounce_buy = cond_ema_bounce_buy and all_filters_ok_buy
        is_potential_bb_bounce_buy = cond_bb_mid_bounce_buy and all_filters_ok_buy
        is_potential_vol_buy = use_vol_breakout_buy and cond_vol_breakout_buy and all_filters_ok_buy
        is_potential_buy = is_potential_ema_buy or is_potential_fib_buy or is_potential_ema_bounce_buy or is_potential_bb_bounce_buy or is_potential_vol_buy

        is_potential_ema_sell = cond_ema_fast_slow_cross_sell and all_filters_ok_sell
        is_potential_fib_sell = cond_fib_bounce_sell and all_filters_ok_sell
        is_potential_ema_bounce_sell = cond_ema_bounce_sell and all_filters_ok_sell
        is_potential_bb_bounce_sell = cond_bb_mid_bounce_sell and all_filters_ok_sell
        is_potential_vol_sell = use_vol_breakout_sell and cond_vol_breakout_sell and all_filters_ok_sell
        is_potential_sell = is_potential_ema_sell or is_potential_fib_sell or is_potential_ema_bounce_sell or is_potential_bb_bounce_sell or is_potential_vol_sell

        current_entry_signal_type = 'Hold'
        if is_potential_buy:
            # Determine type based on Pine priority (adjust if needed)
            if is_potential_bb_bounce_buy: current_entry_signal_type = "BB"
            elif is_potential_ema_bounce_buy: current_entry_signal_type = "EMA"
            elif is_potential_vol_buy: current_entry_signal_type = "Vol"
            elif is_potential_fib_buy: current_entry_signal_type = "Fib"
            elif is_potential_ema_buy: current_entry_signal_type = "EMA"
        elif is_potential_sell:
            if is_potential_bb_bounce_sell: current_entry_signal_type = "BB"
            elif is_potential_ema_bounce_sell: current_entry_signal_type = "EMA"
            elif is_potential_vol_sell: current_entry_signal_type = "Vol"
            elif is_potential_fib_sell: current_entry_signal_type = "Fib"
            elif is_potential_ema_sell: current_entry_signal_type = "EMA"

        # --- State Management & Entry/Exit Logic ---
        was_in_long = in_long
        was_in_short = in_short
        exit_signal = False
        exit_reason = ''
        exit_price = np.nan
        profit_loss = np.nan

        # --- Exit Logic ---
        if was_in_long:
            if use_atr_stop:
                new_stop_long = current['low'] - current[atr_col] * atr_mult
                stop_loss = max(stop_loss, new_stop_long) if not np.isnan(stop_loss) else new_stop_long
                df.loc[current_index, 'stop_loss'] = stop_loss

            atr_stop_hit_long = use_atr_stop and not np.isnan(stop_loss) and current['low'] <= stop_loss
            fib_hit_long = use_fib_exit and not np.isnan(target_price) and current['high'] >= target_price
            score_drop_exit_long = scaled_score < (5.0 - exit_score_drop_threshold)
            ema_exit_long = use_ema_exit and cond_ema_fast_med_cross_sell
            bb_exit_long = use_bb_return_exit and cond_bb_return_mean_sell
            vol_exit_long = use_vol_fade_exit and cond_vol_fade_long
            rsi_div_exit_long = use_rsi_div_exit and cond_rsi_bull_div_exit

            # Determine exit reason based on priority
            if atr_stop_hit_long:       exit_signal=True; exit_reason='ATR SL'; exit_price = stop_loss
            elif fib_hit_long:          exit_signal=True; exit_reason='Fib Target'; exit_price = target_price
            elif score_drop_exit_long:  exit_signal=True; exit_reason='Score Drop'; exit_price = current['close']
            elif ema_exit_long:         exit_signal=True; exit_reason='EMA Cross'; exit_price = current['close']
            elif bb_exit_long:          exit_signal=True; exit_reason='BB Mid'; exit_price = current['close']
            elif vol_exit_long:         exit_signal=True; exit_reason='Vol Fade'; exit_price = current['close']
            elif rsi_div_exit_long:     exit_signal=True; exit_reason='RSI Div'; exit_price = current['close']

            if exit_signal:
                in_long = False
                profit_loss = exit_price - entry_price
                df.loc[current_index, 'signal_type'] = f'Exit Long {trade_count_long} ({exit_reason})'
                df.loc[current_index, 'position'] = 'Hold'
                df.loc[current_index, 'trade_id'] = ''
                df.loc[current_index, 'exit_reason'] = exit_reason
                df.loc[current_index, 'trade_outcome'] = 'Profit' if profit_loss > 0 else ('Loss' if profit_loss < 0 else 'BreakEven')
                df.loc[current_index, 'profit_loss'] = profit_loss

                if entry_signal_type in entry_signal_stats: # Check if key exists
                    stat_exit_type = 'sl' if exit_reason == 'ATR SL' else 'target'
                    entry_signal_stats[entry_signal_type][stat_exit_type] += 1
                    if profit_loss > 0: entry_signal_stats[entry_signal_type]['success'] += 1
                if exit_reason in exit_signal_stats: # Check if key exists
                    exit_signal_stats[exit_reason]['total'] += 1
                    if profit_loss > 0: exit_signal_stats[exit_reason]['success'] += 1

                entry_price, stop_loss, target_price, entry_signal_type, current_trade_id = np.nan, np.nan, np.nan, '', ''

        elif was_in_short:
            if use_atr_stop:
                new_stop_short = current['high'] + current[atr_col] * atr_mult
                stop_loss = min(stop_loss, new_stop_short) if not np.isnan(stop_loss) else new_stop_short
                df.loc[current_index, 'stop_loss'] = stop_loss

            atr_stop_hit_short = use_atr_stop and not np.isnan(stop_loss) and current['high'] >= stop_loss
            fib_hit_short = use_fib_exit and not np.isnan(target_price) and current['low'] <= target_price
            score_drop_exit_short = scaled_score > (5.0 + exit_score_drop_threshold)
            ema_exit_short = use_ema_exit and cond_ema_fast_med_cross_buy
            bb_exit_short = use_bb_return_exit and cond_bb_return_mean_buy
            vol_exit_short = use_vol_fade_exit and cond_vol_fade_short
            rsi_div_exit_short = use_rsi_div_exit and cond_rsi_bear_div_exit

            if atr_stop_hit_short:      exit_signal=True; exit_reason='ATR SL'; exit_price = stop_loss
            elif fib_hit_short:         exit_signal=True; exit_reason='Fib Target'; exit_price = target_price
            elif score_drop_exit_short: exit_signal=True; exit_reason='Score Drop'; exit_price = current['close']
            elif ema_exit_short:        exit_signal=True; exit_reason='EMA Cross'; exit_price = current['close']
            elif bb_exit_short:         exit_signal=True; exit_reason='BB Mid'; exit_price = current['close']
            elif vol_exit_short:        exit_signal=True; exit_reason='Vol Fade'; exit_price = current['close']
            elif rsi_div_exit_short:    exit_signal=True; exit_reason='RSI Div'; exit_price = current['close']

            if exit_signal:
                in_short = False
                profit_loss = entry_price - exit_price # P/L for short
                df.loc[current_index, 'signal_type'] = f'Exit Short {trade_count_short} ({exit_reason})'
                df.loc[current_index, 'position'] = 'Hold'
                df.loc[current_index, 'trade_id'] = ''
                df.loc[current_index, 'exit_reason'] = exit_reason
                df.loc[current_index, 'trade_outcome'] = 'Profit' if profit_loss > 0 else ('Loss' if profit_loss < 0 else 'BreakEven')
                df.loc[current_index, 'profit_loss'] = profit_loss

                if entry_signal_type in entry_signal_stats:
                    stat_exit_type = 'sl' if exit_reason == 'ATR SL' else 'target'
                    entry_signal_stats[entry_signal_type][stat_exit_type] += 1
                    if profit_loss > 0: entry_signal_stats[entry_signal_type]['success'] += 1
                if exit_reason in exit_signal_stats:
                    exit_signal_stats[exit_reason]['total'] += 1
                    if profit_loss > 0: exit_signal_stats[exit_reason]['success'] += 1

                entry_price, stop_loss, target_price, entry_signal_type, current_trade_id = np.nan, np.nan, np.nan, '', ''

        # --- Entry Logic ---
        # Check if not currently in a trade and no exit happened on this bar
        if not in_long and not in_short and not exit_signal:
            if is_potential_buy:
                in_long = True
                trade_count_long += 1
                current_trade_id = f'L{trade_count_long}'
                entry_price = current['close']
                stop_loss = current['low'] - current[atr_col] * atr_mult if use_atr_stop else np.nan
                entry_signal_type = current_entry_signal_type

                if use_fib_exit:
                    lookback_range = df['low'].iloc[max(0, i - fib_lookback_exit):i]
                    if not lookback_range.empty:
                        swing_low = lookback_range.min()
                        swing_range = entry_price - swing_low
                        target_price = entry_price + swing_range * fib_extension_level if swing_range > 0 else np.nan
                    else: target_price = np.nan
                else: target_price = np.nan

                df.loc[current_index, 'signal_type'] = f'Enter Long {trade_count_long} ({entry_signal_type})'
                df.loc[current_index, 'position'] = 'Long'
                df.loc[current_index, 'trade_id'] = current_trade_id
                df.loc[current_index, 'entry_price'] = entry_price
                df.loc[current_index, 'stop_loss'] = stop_loss
                df.loc[current_index, 'target_price'] = target_price
                if entry_signal_type in entry_signal_stats: entry_signal_stats[entry_signal_type]['total'] += 1

            elif is_potential_sell:
                in_short = True
                trade_count_short += 1
                current_trade_id = f'S{trade_count_short}'
                entry_price = current['close']
                stop_loss = current['high'] + current[atr_col] * atr_mult if use_atr_stop else np.nan
                entry_signal_type = current_entry_signal_type

                if use_fib_exit:
                    lookback_range = df['high'].iloc[max(0, i - fib_lookback_exit):i]
                    if not lookback_range.empty:
                        swing_high = lookback_range.max()
                        swing_range = swing_high - entry_price
                        target_price = entry_price - swing_range * fib_extension_level if swing_range > 0 else np.nan
                    else: target_price = np.nan
                else: target_price = np.nan

                df.loc[current_index, 'signal_type'] = f'Enter Short {trade_count_short} ({entry_signal_type})'
                df.loc[current_index, 'position'] = 'Short'
                df.loc[current_index, 'trade_id'] = current_trade_id
                df.loc[current_index, 'entry_price'] = entry_price
                df.loc[current_index, 'stop_loss'] = stop_loss
                df.loc[current_index, 'target_price'] = target_price
                if entry_signal_type in entry_signal_stats: entry_signal_stats[entry_signal_type]['total'] += 1

        # Update 'Running' status if still in trade
        if not exit_signal:
            if in_long:
                df.loc[current_index, 'position'] = 'Long'
                df.loc[current_index, 'trade_id'] = current_trade_id
                df.loc[current_index, 'signal_type'] = f'Running Long {trade_count_long}'
                df.loc[current_index, 'entry_price'] = entry_price # Carry forward entry price
                df.loc[current_index, 'stop_loss'] = stop_loss # Carry forward updated SL
                df.loc[current_index, 'target_price'] = target_price # Carry forward target
            elif in_short:
                df.loc[current_index, 'position'] = 'Short'
                df.loc[current_index, 'trade_id'] = current_trade_id
                df.loc[current_index, 'signal_type'] = f'Running Short {trade_count_short}'
                df.loc[current_index, 'entry_price'] = entry_price
                df.loc[current_index, 'stop_loss'] = stop_loss
                df.loc[current_index, 'target_price'] = target_price

    # --- Calculate Final Statistics ---
    # (Stats calculation logic remains the same)
    for signal in entry_signal_stats:
        total = entry_signal_stats[signal]['total']
        if total > 0:
            entry_signal_stats[signal]['success_rate'] = entry_signal_stats[signal]['success'] / total * 100
            entry_signal_stats[signal]['sl_rate'] = entry_signal_stats[signal]['sl'] / total * 100
            entry_signal_stats[signal]['target_rate'] = entry_signal_stats[signal]['target'] / total * 100
        else:
            entry_signal_stats[signal]['success_rate'], entry_signal_stats[signal]['sl_rate'], entry_signal_stats[signal]['target_rate'] = 0, 0, 0

    for exit_type in exit_signal_stats:
        total = exit_signal_stats[exit_type]['total']
        if total > 0:
            exit_signal_stats[exit_type]['success_rate'] = exit_signal_stats[exit_type]['success'] / total * 100
        else:
            exit_signal_stats[exit_type]['success_rate'] = 0

    return df, entry_signal_stats, exit_signal_stats

# --- Example Usage ---
# (Example Usage block remains the same)
try:
    df = pd.read_csv('NIFTY_signals_20250402_2229.csv', parse_dates=['datetime'])
    required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain columns: {required_cols}")

    df_enhanced, entry_stats, exit_stats = process_signals(df.copy())
    output_filename = 'enhanced_signals_v3_fixed.csv' # Changed output filename
    df_enhanced.to_csv(output_filename, index=False)

    print("\nEntry Signal Statistics:")
    print("Signal | Total | Success % | SL Hit % | Target/Other Exit %")
    print("--------------------------------------------------------------")
    for signal, stats in entry_stats.items():
        print(f"{signal:6} | {stats['total']:5} | {stats['success_rate']:8.1f}% | {stats['sl_rate']:8.1f}% | {stats['target_rate']:17.1f}%")

    print("\nExit Signal Statistics (Trade P/L when this exit triggered):")
    print("Exit Type     | Total | Profitable Exit %")
    print("-------------------------------------------")
    for exit_type, stats in exit_stats.items():
        print(f"{exit_type:12} | {stats['total']:5} | {stats['success_rate']:16.1f}%")

    print(f"\nSignal processing complete. Enhanced data saved to {output_filename}")

except FileNotFoundError:
    print("Error: Input CSV file 'NIFTY_signals_20250402_2229.csv' not found.")
except ValueError as ve:
    print(f"Error: {ve}")
except KeyError as ke:
    print(f"An unexpected KeyError occurred: {ke}. This might indicate a mismatch in expected column names.")
    # Add more specific debug info if possible, e.g., print df.columns before the error
except Exception as e:
    import traceback
    print(f"An unexpected error occurred: {e}")
    print("Traceback:")
    traceback.print_exc()