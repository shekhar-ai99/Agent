import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def process_signals(df):
    # Initialize signal columns
    df['signal_type'] = 'Hold'
    df['signal_strength'] = 0.0
    df['signal_reason'] = ''
    df['position'] = 'Hold'
    df['entry_price'] = np.nan
    df['stop_loss'] = np.nan
    df['target_price'] = np.nan
    df['fib_retracement'] = np.nan
    df['exit_reason'] = ''
    df['trade_outcome'] = ''
    df['profit_loss'] = np.nan
    
    # Initialize tracking dictionaries for success rates
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
        'RSI Div': {'total': 0, 'success': 0}
    }
    
    # Initialize state variables
    in_long = False
    in_short = False
    entry_price = np.nan
    stop_loss = np.nan
    target_price = np.nan
    entry_signal = ''
    last_pivot_high = np.nan
    last_pivot_low = np.nan
    
    # Parameters from indicator settings
    exit_score_drop_threshold = 1.5
    fib_bounce_lookback = 3
    ema_bounce_lookback = 2
    bb_bounce_lookback = 2
    use_fib_bounce = True
    use_ema_bounce = True
    use_bb_bounce = True
    use_vol_breakout = True
    
    # EMA settings
    ema_fast_len = 9
    ema_med_len = 14
    ema_slow_len = 21
    use_ema_exit = True
    
    # Bollinger Bands settings
    bb_len = 20
    bb_std_dev = 2.0
    use_bb_return_exit = False
    
    # RSI settings
    rsi_len = 14
    rsi_buy_level = 55
    rsi_sell_level = 45
    use_rsi_div_exit = True
    rsi_confirm_fib = True
    rsi_confirm_ema = False
    rsi_confirm_bb = False
    
    # MACD settings
    macd_fast_len = 12
    macd_slow_len = 26
    macd_signal_len = 9
    
    # Volume settings
    vol_ma_len = 50
    vol_multiplier = 1.5
    
    # ATR settings
    atr_len = 14
    atr_mult = 2.0
    
    # Fibonacci settings
    fib_lookback_exit = 30
    fib_extension_level = 1.618
    fib_pivot_lookback = 15
    fib_max_bars = 200
    
    # Trend filters
    use_ema_trend_filter = False
    use_adx_filter = False
    adx_len = 14
    adx_threshold = 20
    use_adx_direction_filter = False
    
    # Score weights
    weights = {
        'ema_trend': 2,
        'ema_signal': 1,
        'rsi_thresh': 1,
        'macd_signal': 1,
        'macd_zero': 1,
        'vol_break': 1,
        'adx_strength': 1,
        'adx_direction': 1,
        'htf_trend': 2,
        'fib_bounce': 2,
        'ema_bounce': 1,
        'bb_bounce': 1
    }
    
    # Calculate indicators
    df['ema_fast'] = df['close'].ewm(span=ema_fast_len, adjust=False).mean()
    df['ema_med'] = df['close'].ewm(span=ema_med_len, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow_len, adjust=False).mean()
    
    # Calculate Bollinger Bands
    df['bb_middle'] = df['close'].rolling(bb_len).mean()
    df['bb_upper'] = df['bb_middle'] + df['close'].rolling(bb_len).std() * bb_std_dev
    df['bb_lower'] = df['bb_middle'] - df['close'].rolling(bb_len).std() * bb_std_dev
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_len).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_len).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD if not present
    if 'macd' not in df.columns or 'macd_signal' not in df.columns:
        df['macd'] = df['close'].ewm(span=macd_fast_len, adjust=False).mean() - df['close'].ewm(span=macd_slow_len, adjust=False).mean()
        df['macd_signal'] = df['macd'].ewm(span=macd_signal_len, adjust=False).mean()
    
    # Calculate Volume MA
    df['vol_ma'] = df['volume'].rolling(vol_ma_len).mean()
    
    # Calculate ATR
    df['tr'] = np.maximum.reduce([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ])
    df['atr'] = df['tr'].rolling(atr_len).mean()
    
    # Calculate ADX if needed
    if use_adx_filter or use_adx_direction_filter:
        # Simplified ADX calculation
        df['up_move'] = df['high'] - df['high'].shift()
        df['down_move'] = df['low'].shift() - df['low']
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        df['plus_di'] = 100 * (df['plus_dm'].ewm(span=adx_len, adjust=False).mean() / df['atr'])
        df['minus_di'] = 100 * (df['minus_dm'].ewm(span=adx_len, adjust=False).mean() / df['atr'])
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].ewm(span=adx_len, adjust=False).mean()
    
    for i in range(max(fib_pivot_lookback, fib_lookback_exit), len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Fibonacci pivot detection
        lookback = df.iloc[i-fib_pivot_lookback:i]
        pivot_high = lookback['high'].max()
        pivot_low = lookback['low'].min()
        
        if not np.isnan(pivot_high) and (np.isnan(last_pivot_high) or pivot_high > last_pivot_high):
            last_pivot_high = pivot_high
            last_pivot_high_idx = lookback['high'].idxmax()
        
        if not np.isnan(pivot_low) and (np.isnan(last_pivot_low) or pivot_low < last_pivot_low):
            last_pivot_low = pivot_low
            last_pivot_low_idx = lookback['low'].idxmin()
        
        # Determine trend direction based on pivots
        uptrend = False
        if not np.isnan(last_pivot_high) and not np.isnan(last_pivot_low):
            uptrend = last_pivot_low_idx < last_pivot_high_idx
        
        # Calculate Fibonacci levels
        if uptrend and not np.isnan(last_pivot_low) and not np.isnan(last_pivot_high):
            fib_range = last_pivot_high - last_pivot_low
            level_0 = last_pivot_low
            level_236 = level_0 + fib_range * 0.236
            level_382 = level_0 + fib_range * 0.382
            level_500 = level_0 + fib_range * 0.5
            level_618 = level_0 + fib_range * 0.618
            level_786 = level_0 + fib_range * 0.786
            level_100 = last_pivot_high
            df.at[i, 'fib_retracement'] = (current['close'] - level_0) / fib_range
        elif not np.isnan(last_pivot_high) and not np.isnan(last_pivot_low):
            fib_range = last_pivot_high - last_pivot_low
            level_0 = last_pivot_high
            level_100 = last_pivot_low
            level_236 = level_0 - fib_range * 0.236
            level_382 = level_0 - fib_range * 0.382
            level_500 = level_0 - fib_range * 0.5
            level_618 = level_0 - fib_range * 0.618
            level_786 = level_0 - fib_range * 0.786
            df.at[i, 'fib_retracement'] = (level_0 - current['close']) / fib_range
        
        # Calculate basic conditions
        ema_fast_slow_cross_buy = current['ema_fast'] > current['ema_slow'] and prev['ema_fast'] <= prev['ema_slow']
        ema_fast_slow_cross_sell = current['ema_fast'] < current['ema_slow'] and prev['ema_fast'] >= prev['ema_slow']
        
        ema_fast_med_cross_buy = current['ema_fast'] > current['ema_med'] and prev['ema_fast'] <= prev['ema_med']
        ema_fast_med_cross_sell = current['ema_fast'] < current['ema_med'] and prev['ema_fast'] >= prev['ema_med']
        
        rsi_buy = current['rsi'] > rsi_buy_level
        rsi_sell = current['rsi'] < rsi_sell_level
        
        macd_signal_cross_buy = current['macd'] > current['macd_signal'] and prev['macd'] <= prev['macd_signal']
        macd_signal_cross_sell = current['macd'] < current['macd_signal'] and prev['macd'] >= prev['macd_signal']
        
        macd_zero_cross_buy = current['macd'] > 0 and prev['macd'] <= 0
        macd_zero_cross_sell = current['macd'] < 0 and prev['macd'] >= 0
        
        high_vol = current['volume'] > current['vol_ma'] * vol_multiplier
        vol_breakout_buy = high_vol and current['close'] > current['open'] and current['close'] > current['ema_slow']
        vol_breakout_sell = high_vol and current['close'] < current['open'] and current['close'] < current['ema_slow']
        
        # Trend filter conditions
        ema_trend_ok_buy = not use_ema_trend_filter or current['ema_med'] > current['ema_slow']
        ema_trend_ok_sell = not use_ema_trend_filter or current['ema_med'] < current['ema_slow']
        
        adx_strength_ok = not use_adx_filter or current['adx'] > adx_threshold if 'adx' in df.columns else True
        adx_direction_ok_buy = not use_adx_direction_filter or current['plus_di'] > current['minus_di'] if 'plus_di' in df.columns else True
        adx_direction_ok_sell = not use_adx_direction_filter or current['minus_di'] > current['plus_di'] if 'minus_di' in df.columns else True
        
        all_filters_ok_buy = ema_trend_ok_buy and adx_strength_ok and adx_direction_ok_buy
        all_filters_ok_sell = ema_trend_ok_sell and adx_strength_ok and adx_direction_ok_sell
        
        # Bounce conditions
        fib_bounce_buy = False
        fib_bounce_sell = False
        if use_fib_bounce and not np.isnan(level_618) and not np.isnan(level_500):
            if uptrend:
                # Check if price touched fib zone recently
                touched_fib_zone = False
                for j in range(1, fib_bounce_lookback + 1):
                    if df.iloc[i-j]['low'] <= level_618:
                        touched_fib_zone = True
                        break
                
                bounced_above_50 = current['close'] > level_500
                rsi_confirms = not rsi_confirm_fib or (current['rsi'] > 40 and current['rsi'] > prev['rsi'])
                
                fib_bounce_buy = touched_fib_zone and bounced_above_50 and rsi_confirms and all_filters_ok_buy
            else:
                # Check if price touched fib zone recently
                touched_fib_zone = False
                for j in range(1, fib_bounce_lookback + 1):
                    if df.iloc[i-j]['high'] >= level_382:
                        touched_fib_zone = True
                        break
                
                rejected_below_50 = current['close'] < level_500
                rsi_confirms = not rsi_confirm_fib or (current['rsi'] < 60 and current['rsi'] < prev['rsi'])
                
                fib_bounce_sell = touched_fib_zone and rejected_below_50 and rsi_confirms and all_filters_ok_sell
        
        ema_bounce_buy = False
        ema_bounce_sell = False
        if use_ema_bounce:
            ema_source = current['ema_fast']  # Using Fast EMA as per settings
            
            # Check EMA bounce for buy
            touched_ema = False
            above_ema_before = True
            for j in range(1, ema_bounce_lookback + 1):
                if df.iloc[i-j]['low'] <= df.iloc[i-j]['ema_fast']:
                    touched_ema = True
                if df.iloc[i-j]['close'] <= df.iloc[i-j]['ema_fast']:
                    above_ema_before = False
            
            rsi_confirms = not rsi_confirm_ema or (current['rsi'] > 40 and current['rsi'] > prev['rsi'])
            ema_bounce_buy = touched_ema and above_ema_before and current['close'] > ema_source and current['close'] > current['open'] and rsi_confirms and all_filters_ok_buy
            
            # Check EMA bounce for sell
            touched_ema = False
            below_ema_before = True
            for j in range(1, ema_bounce_lookback + 1):
                if df.iloc[i-j]['high'] >= df.iloc[i-j]['ema_fast']:
                    touched_ema = True
                if df.iloc[i-j]['close'] >= df.iloc[i-j]['ema_fast']:
                    below_ema_before = False
            
            rsi_confirms = not rsi_confirm_ema or (current['rsi'] < 60 and current['rsi'] < prev['rsi'])
            ema_bounce_sell = touched_ema and below_ema_before and current['close'] < ema_source and current['close'] < current['open'] and rsi_confirms and all_filters_ok_sell
        
        bb_bounce_buy = False
        bb_bounce_sell = False
        if use_bb_bounce:
            # Check BB bounce for buy
            touched_bb = False
            for j in range(1, bb_bounce_lookback + 1):
                if df.iloc[i-j]['low'] <= df.iloc[i-j]['bb_middle']:
                    touched_bb = True
                    break
            
            rsi_confirms = not rsi_confirm_bb or (current['rsi'] > 40 and current['rsi'] > prev['rsi'])
            bb_bounce_buy = touched_bb and current['close'] > current['bb_middle'] and current['close'] > current['open'] and rsi_confirms and all_filters_ok_buy
            
            # Check BB bounce for sell
            touched_bb = False
            for j in range(1, bb_bounce_lookback + 1):
                if df.iloc[i-j]['high'] >= df.iloc[i-j]['bb_middle']:
                    touched_bb = True
                    break
            
            rsi_confirms = not rsi_confirm_bb or (current['rsi'] < 60 and current['rsi'] < prev['rsi'])
            bb_bounce_sell = touched_bb and current['close'] < current['bb_middle'] and current['close'] < current['open'] and rsi_confirms and all_filters_ok_sell
        
        # Confidence score calculation
        buy_score = 0
        sell_score = 0
        reason = ""

        if ema_trend_ok_buy:
            buy_score += weights['ema_trend']
            reason += "ET+"
        elif ema_trend_ok_sell:
            sell_score += weights['ema_trend']
            reason += "ET-"
            
        if ema_fast_slow_cross_buy:
            buy_score += weights['ema_signal']
            reason += "ES+"
        if ema_fast_slow_cross_sell:
            sell_score += weights['ema_signal']
            reason += "ES-"
            
        if rsi_buy:
            buy_score += weights['rsi_thresh']
            reason += "R+"
        if rsi_sell:
            sell_score += weights['rsi_thresh']
            reason += "R-"
            
        if macd_signal_cross_buy:
            buy_score += weights['macd_signal']
            reason += "MS+"
        if macd_signal_cross_sell:
            sell_score += weights['macd_signal']
            reason += "MS-"
            
        if macd_zero_cross_buy:
            buy_score += weights['macd_zero']
            reason += "MZ+"
        if macd_zero_cross_sell:
            sell_score += weights['macd_zero']
            reason += "MZ-"
            
        if vol_breakout_buy:
            buy_score += weights['vol_break']
            reason += "V+"
        if vol_breakout_sell:
            sell_score += weights['vol_break']
            reason += "V-"
            
        if adx_strength_ok:
            if adx_direction_ok_buy:
                buy_score += weights['adx_strength']
                reason += "AS+"
            if adx_direction_ok_sell:
                sell_score += weights['adx_strength']
                reason += "AS-"

        if adx_direction_ok_buy:
            buy_score += weights['adx_direction']
            reason += "AD+"
        if adx_direction_ok_sell:
            sell_score += weights['adx_direction']
            reason += "AD-"
            
        if fib_bounce_buy:
            buy_score += weights['fib_bounce']
            reason += "FB+"
        if fib_bounce_sell:
            sell_score += weights['fib_bounce']
            reason += "FB-"
            
        if ema_bounce_buy:
            buy_score += weights['ema_bounce']
            reason += "EB+"
        if ema_bounce_sell:
            sell_score += weights['ema_bounce']
            reason += "EB-"
            
        if bb_bounce_buy:
            buy_score += weights['bb_bounce']
            reason += "BB+"
        if bb_bounce_sell:
            sell_score += weights['bb_bounce']
            reason += "BB-"

        total_possible_score = sum(weights.values())
        net_score = buy_score - sell_score
        scaled_score = (net_score / total_possible_score) * 5.0 + 5.0
        scaled_score = max(0.0, min(10.0, scaled_score))
        
        # Determine signal type
        signal_type = 'Hold'
        if fib_bounce_buy:
            signal_type = 'Fib'
        elif ema_bounce_buy:
            signal_type = 'EMA'
        elif bb_bounce_buy:
            signal_type = 'BB'
        elif vol_breakout_buy:
            signal_type = 'Vol'
        elif ema_fast_slow_cross_buy and ema_trend_ok_buy:
            signal_type = 'EMA'
        elif fib_bounce_sell:
            signal_type = 'Fib'
        elif ema_bounce_sell:
            signal_type = 'EMA'
        elif bb_bounce_sell:
            signal_type = 'BB'
        elif vol_breakout_sell:
            signal_type = 'Vol'
        elif ema_fast_slow_cross_sell and ema_trend_ok_sell:
            signal_type = 'EMA'
        
        # Entry logic
        if not in_long and not in_short:
            if (fib_bounce_buy or ema_bounce_buy or bb_bounce_buy or 
                vol_breakout_buy or (ema_fast_slow_cross_buy and ema_trend_ok_buy)):
                in_long = True
                entry_price = current['close']
                stop_loss = current['low'] - current['atr'] * atr_mult
                entry_signal = signal_type
                
                # Calculate fib target
                lookback = df.iloc[max(0, i-fib_lookback_exit):i]
                swing_low = lookback['low'].min()
                swing_range = entry_price - swing_low
                target_price = entry_price + swing_range * fib_extension_level
                
                df.at[i, 'signal_type'] = f'Enter Long ({signal_type})'
                df.at[i, 'position'] = 'Long'
                df.at[i, 'entry_price'] = entry_price
                df.at[i, 'stop_loss'] = stop_loss
                df.at[i, 'target_price'] = target_price
                
                # Update entry signal stats
                entry_signal_stats[signal_type]['total'] += 1
                
            elif (fib_bounce_sell or ema_bounce_sell or bb_bounce_sell or 
                  vol_breakout_sell or (ema_fast_slow_cross_sell and ema_trend_ok_sell)):
                in_short = True
                entry_price = current['close']
                stop_loss = current['high'] + current['atr'] * atr_mult
                entry_signal = signal_type
                
                # Calculate fib target
                lookback = df.iloc[max(0, i-fib_lookback_exit):i]
                swing_high = lookback['high'].max()
                swing_range = swing_high - entry_price
                target_price = entry_price - swing_range * fib_extension_level
                
                df.at[i, 'signal_type'] = f'Enter Short ({signal_type})'
                df.at[i, 'position'] = 'Short'
                df.at[i, 'entry_price'] = entry_price
                df.at[i, 'stop_loss'] = stop_loss
                df.at[i, 'target_price'] = target_price
                
                # Update entry signal stats
                entry_signal_stats[signal_type]['total'] += 1
        
        # Exit logic
        elif in_long:
            # Update trailing stop
            new_stop = current['low'] - current['atr'] * atr_mult
            stop_loss = max(stop_loss, new_stop)
            df.at[i, 'stop_loss'] = stop_loss
            
            # Check exit conditions
            exit_reason = ''
            exit_signal = False
            profit_loss = 0
            
            # ATR stop hit
            if current['close'] < stop_loss:
                exit_reason = 'ATR SL'
                exit_signal = True
                profit_loss = current['close'] - entry_price
                entry_signal_stats[entry_signal]['sl'] += 1
                if profit_loss > 0:
                    entry_signal_stats[entry_signal]['success'] += 1
                exit_signal_stats[exit_reason]['total'] += 1
                exit_signal_stats[exit_reason]['success'] += 1 if profit_loss > 0 else 0
            # Score drop exit
            elif scaled_score < (5.0 - exit_score_drop_threshold):
                exit_reason = 'Score Drop'
                exit_signal = True
                profit_loss = current['close'] - entry_price
                if profit_loss > 0:
                    entry_signal_stats[entry_signal]['success'] += 1
                    entry_signal_stats[entry_signal]['target'] += 1
                exit_signal_stats[exit_reason]['total'] += 1
                exit_signal_stats[exit_reason]['success'] += 1 if profit_loss > 0 else 0
            # EMA cross exit
            elif use_ema_exit and ema_fast_med_cross_sell:
                exit_reason = 'EMA Cross'
                exit_signal = True
                profit_loss = current['close'] - entry_price
                if profit_loss > 0:
                    entry_signal_stats[entry_signal]['success'] += 1
                    entry_signal_stats[entry_signal]['target'] += 1
                exit_signal_stats[exit_reason]['total'] += 1
                exit_signal_stats[exit_reason]['success'] += 1 if profit_loss > 0 else 0
            # BB return to mean exit
            elif use_bb_return_exit and current['close'] < current['bb_middle']:
                exit_reason = 'BB Mid'
                exit_signal = True
                profit_loss = current['close'] - entry_price
                if profit_loss > 0:
                    entry_signal_stats[entry_signal]['success'] += 1
                    entry_signal_stats[entry_signal]['target'] += 1
                exit_signal_stats[exit_reason]['total'] += 1
                exit_signal_stats[exit_reason]['success'] += 1 if profit_loss > 0 else 0
            # Fib target hit
            elif not np.isnan(target_price) and current['high'] >= target_price:
                exit_reason = 'Fib Target'
                exit_signal = True
                profit_loss = target_price - entry_price
                entry_signal_stats[entry_signal]['success'] += 1
                entry_signal_stats[entry_signal]['target'] += 1
                exit_signal_stats[exit_reason]['total'] += 1
                exit_signal_stats[exit_reason]['success'] += 1
            # RSI divergence exit
            elif use_rsi_div_exit:
                # Simplified RSI divergence detection
                if i >= 5:
                    rsi_peak = df.iloc[i-5:i]['rsi'].max()
                    price_peak = df.iloc[i-5:i]['high'].max()
                    if current['rsi'] < rsi_peak and current['high'] > price_peak:
                        exit_reason = 'RSI Div'
                        exit_signal = True
                        profit_loss = current['close'] - entry_price
                        if profit_loss > 0:
                            entry_signal_stats[entry_signal]['success'] += 1
                            entry_signal_stats[entry_signal]['target'] += 1
                        exit_signal_stats[exit_reason]['total'] += 1
                        exit_signal_stats[exit_reason]['success'] += 1 if profit_loss > 0 else 0
            
            if exit_signal:
                in_long = False
                df.at[i, 'signal_type'] = f'Exit Long ({exit_reason})'
                df.at[i, 'position'] = 'Hold'
                df.at[i, 'exit_reason'] = exit_reason
                df.at[i, 'trade_outcome'] = 'Profit' if profit_loss > 0 else 'Loss'
                df.at[i, 'profit_loss'] = profit_loss
                entry_price = np.nan
                stop_loss = np.nan
                target_price = np.nan
                entry_signal = ''
        
        elif in_short:
            # Update trailing stop
            new_stop = current['high'] + current['atr'] * atr_mult
            stop_loss = min(stop_loss, new_stop)
            df.at[i, 'stop_loss'] = stop_loss
            
            # Check exit conditions
            exit_reason = ''
            exit_signal = False
            profit_loss = 0
            
            # ATR stop hit
            if current['close'] > stop_loss:
                exit_reason = 'ATR SL'
                exit_signal = True
                profit_loss = entry_price - current['close']
                entry_signal_stats[entry_signal]['sl'] += 1
                if profit_loss > 0:
                    entry_signal_stats[entry_signal]['success'] += 1
                exit_signal_stats[exit_reason]['total'] += 1
                exit_signal_stats[exit_reason]['success'] += 1 if profit_loss > 0 else 0
            # Score drop exit
            elif scaled_score > (5.0 + exit_score_drop_threshold):
                exit_reason = 'Score Drop'
                exit_signal = True
                profit_loss = entry_price - current['close']
                if profit_loss > 0:
                    entry_signal_stats[entry_signal]['success'] += 1
                    entry_signal_stats[entry_signal]['target'] += 1
                exit_signal_stats[exit_reason]['total'] += 1
                exit_signal_stats[exit_reason]['success'] += 1 if profit_loss > 0 else 0
            # EMA cross exit
            elif use_ema_exit and ema_fast_med_cross_buy:
                exit_reason = 'EMA Cross'
                exit_signal = True
                profit_loss = entry_price - current['close']
                if profit_loss > 0:
                    entry_signal_stats[entry_signal]['success'] += 1
                    entry_signal_stats[entry_signal]['target'] += 1
                exit_signal_stats[exit_reason]['total'] += 1
                exit_signal_stats[exit_reason]['success'] += 1 if profit_loss > 0 else 0
            # BB return to mean exit
            elif use_bb_return_exit and current['close'] > current['bb_middle']:
                exit_reason = 'BB Mid'
                exit_signal = True
                profit_loss = entry_price - current['close']
                if profit_loss > 0:
                    entry_signal_stats[entry_signal]['success'] += 1
                    entry_signal_stats[entry_signal]['target'] += 1
                exit_signal_stats[exit_reason]['total'] += 1
                exit_signal_stats[exit_reason]['success'] += 1 if profit_loss > 0 else 0
            # Fib target hit
            elif not np.isnan(target_price) and current['low'] <= target_price:
                exit_reason = 'Fib Target'
                exit_signal = True
                profit_loss = entry_price - target_price
                entry_signal_stats[entry_signal]['success'] += 1
                entry_signal_stats[entry_signal]['target'] += 1
                exit_signal_stats[exit_reason]['total'] += 1
                exit_signal_stats[exit_reason]['success'] += 1
            # RSI divergence exit
            elif use_rsi_div_exit:
                # Simplified RSI divergence detection
                if i >= 5:
                    rsi_trough = df.iloc[i-5:i]['rsi'].min()
                    price_trough = df.iloc[i-5:i]['low'].min()
                    if current['rsi'] > rsi_trough and current['low'] < price_trough:
                        exit_reason = 'RSI Div'
                        exit_signal = True
                        profit_loss = entry_price - current['close']
                        if profit_loss > 0:
                            entry_signal_stats[entry_signal]['success'] += 1
                            entry_signal_stats[entry_signal]['target'] += 1
                        exit_signal_stats[exit_reason]['total'] += 1
                        exit_signal_stats[exit_reason]['success'] += 1 if profit_loss > 0 else 0
            
            if exit_signal:
                in_short = False
                df.at[i, 'signal_type'] = f'Exit Short ({exit_reason})'
                df.at[i, 'position'] = 'Hold'
                df.at[i, 'exit_reason'] = exit_reason
                df.at[i, 'trade_outcome'] = 'Profit' if profit_loss > 0 else 'Loss'
                df.at[i, 'profit_loss'] = profit_loss
                entry_price = np.nan
                stop_loss = np.nan
                target_price = np.nan
                entry_signal = ''
    
    # Calculate success rates
    for signal in entry_signal_stats:
        if entry_signal_stats[signal]['total'] > 0:
            entry_signal_stats[signal]['success_rate'] = entry_signal_stats[signal]['success'] / entry_signal_stats[signal]['total'] * 100
            entry_signal_stats[signal]['sl_rate'] = entry_signal_stats[signal]['sl'] / entry_signal_stats[signal]['total'] * 100
            entry_signal_stats[signal]['target_rate'] = entry_signal_stats[signal]['target'] / entry_signal_stats[signal]['total'] * 100
        else:
            entry_signal_stats[signal]['success_rate'] = 0
            entry_signal_stats[signal]['sl_rate'] = 0
            entry_signal_stats[signal]['target_rate'] = 0
    
    for exit_type in exit_signal_stats:
        if exit_signal_stats[exit_type]['total'] > 0:
            exit_signal_stats[exit_type]['success_rate'] = exit_signal_stats[exit_type]['success'] / exit_signal_stats[exit_type]['total'] * 100
        else:
            exit_signal_stats[exit_type]['success_rate'] = 0
    
    return df, entry_signal_stats, exit_signal_stats

# Load your CSV data
df = pd.read_csv('NIFTY_signals_20250402_2229.csv', parse_dates=['datetime'])

# Process the signals and get statistics
df, entry_stats, exit_stats = process_signals(df)

# Save the enhanced data
df.to_csv('enhanced_signals_with_stats.csv', index=False)

# Print statistics
print("\nEntry Signal Statistics:")
print("Signal | Total | Success % | SL % | Target %")
print("--------------------------------------------")
for signal, stats in entry_stats.items():
    print(f"{signal:6} | {stats['total']:5} | {stats['success_rate']:8.1f}% | {stats['sl_rate']:4.1f}% | {stats['target_rate']:6.1f}%")

print("\nExit Signal Statistics:")
print("Exit Type     | Total | Success %")
print("--------------------------------")
for exit_type, stats in exit_stats.items():
    print(f"{exit_type:12} | {stats['total']:5} | {stats['success_rate']:8.1f}%")

print("\nSignal processing complete. Enhanced data saved to enhanced_signals_with_stats.csv")