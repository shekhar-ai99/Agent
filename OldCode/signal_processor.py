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
    
    # Initialize state variables
    in_long = False
    in_short = False
    entry_price = np.nan
    stop_loss = np.nan
    target_price = np.nan
    
    # Parameters (from PineScript)
    exit_score_drop_threshold = 1.5
    atr_mult = 2.0
    atr_len = 14
    fib_extension_level = 1.618
    fib_lookback_exit = 30
    
    # Calculate additional indicators needed
    df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_med'] = df['close'].ewm(span=14, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
    df['atr'] = df['high'].rolling(atr_len).max() - df['low'].rolling(atr_len).min()
    
    # Calculate RSI if not present
    if 'rsi' not in df.columns:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Calculate basic conditions
        ema_fast_slow_cross_buy = current['ema_fast'] > current['ema_slow'] and prev['ema_fast'] <= prev['ema_slow']
        ema_fast_slow_cross_sell = current['ema_fast'] < current['ema_slow'] and prev['ema_fast'] >= prev['ema_slow']
        
        ema_fast_med_cross_buy = current['ema_fast'] > current['ema_med'] and prev['ema_fast'] <= prev['ema_med']
        ema_fast_med_cross_sell = current['ema_fast'] < current['ema_med'] and prev['ema_fast'] >= prev['ema_med']
        
        rsi_buy = current['rsi'] > 55
        rsi_sell = current['rsi'] < 45
        
        # Calculate MACD signals if MACD columns exist
        macd_signal_cross_buy = False
        macd_signal_cross_sell = False
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_signal_cross_buy = current['macd'] > current['macd_signal'] and prev['macd'] <= prev['macd_signal']
            macd_signal_cross_sell = current['macd'] < current['macd_signal'] and prev['macd'] >= prev['macd_signal']
        
        # Volume breakout
        vol_ma = df['volume'].rolling(50).mean()
        high_vol = current['volume'] > vol_ma.iloc[i] * 1.5
        vol_breakout_buy = high_vol and current['close'] > current['open'] and current['close'] > current['ema_slow']
        vol_breakout_sell = high_vol and current['close'] < current['open'] and current['close'] < current['ema_slow']
        
        # Trend filters
        ema_trend_ok_buy = current['ema_med'] > current['ema_slow']
        ema_trend_ok_sell = current['ema_med'] < current['ema_slow']
        
        # Confidence score calculation (simplified)
        buy_score = 0
        sell_score = 0
        reason = ""
        
        if ema_trend_ok_buy:
            buy_score += 2
            reason += "ET+"
        elif ema_trend_ok_sell:
            sell_score += 2
            reason += "ET-"
            
        if ema_fast_slow_cross_buy:
            buy_score += 1
            reason += "ES+"
        if ema_fast_slow_cross_sell:
            sell_score += 1
            reason += "ES-"
            
        if rsi_buy:
            buy_score += 1
            reason += "R+"
        if rsi_sell:
            sell_score += 1
            reason += "R-"
            
        if macd_signal_cross_buy:
            buy_score += 1
            reason += "MS+"
        if macd_signal_cross_sell:
            sell_score += 1
            reason += "MS-"
            
        if vol_breakout_buy:
            buy_score += 1
            reason += "V+"
        if vol_breakout_sell:
            sell_score += 1
            reason += "V-"
        
        total_possible_score = 7  # Sum of weights used
        net_score = buy_score - sell_score
        scaled_score = (net_score / total_possible_score) * 5.0 + 5.0
        scaled_score = max(0.0, min(10.0, scaled_score))
        
        # Determine signal type
        signal_type = 'Hold'
        if ema_fast_slow_cross_buy and ema_trend_ok_buy:
            signal_type = 'EMA'
        elif ema_fast_slow_cross_sell and ema_trend_ok_sell:
            signal_type = 'EMA'
        elif vol_breakout_buy and ema_trend_ok_buy:
            signal_type = 'Vol'
        elif vol_breakout_sell and ema_trend_ok_sell:
            signal_type = 'Vol'
        
        # Entry logic
        if not in_long and not in_short:
            if (ema_fast_slow_cross_buy and ema_trend_ok_buy) or vol_breakout_buy:
                in_long = True
                entry_price = current['close']
                stop_loss = current['low'] - current['atr'] * atr_mult
                
                # Calculate fib target
                lookback = df.iloc[max(0, i-fib_lookback_exit):i]
                swing_low = lookback['low'].min()
                swing_range = entry_price - swing_low
                target_price = entry_price + swing_range * fib_extension_level
                
                df.at[i, 'signal_type'] = 'Enter Long'
                df.at[i, 'position'] = 'Long'
                
            elif (ema_fast_slow_cross_sell and ema_trend_ok_sell) or vol_breakout_sell:
                in_short = True
                entry_price = current['close']
                stop_loss = current['high'] + current['atr'] * atr_mult
                
                # Calculate fib target
                lookback = df.iloc[max(0, i-fib_lookback_exit):i]
                swing_high = lookback['high'].max()
                swing_range = swing_high - entry_price
                target_price = entry_price - swing_range * fib_extension_level
                
                df.at[i, 'signal_type'] = 'Enter Short'
                df.at[i, 'position'] = 'Short'
        
        # Exit logic
        elif in_long:
            # Update trailing stop
            new_stop = current['low'] - current['atr'] * atr_mult
            stop_loss = max(stop_loss, new_stop)
            
            # Check exit conditions
            exit_reason = ''
            exit_signal = False
            
            # ATR stop hit
            if current['close'] < stop_loss:
                exit_reason = 'ATR SL'
                exit_signal = True
            # Score drop exit
            elif scaled_score < (5.0 - exit_score_drop_threshold):
                exit_reason = f'Score Drop ({scaled_score:.1f})'
                exit_signal = True
            # EMA cross exit
            elif ema_fast_med_cross_sell:
                exit_reason = 'EMA Cross'
                exit_signal = True
            # Fib target hit
            elif not np.isnan(target_price) and current['high'] >= target_price:
                exit_reason = 'Fib Target'
                exit_signal = True
            
            if exit_signal:
                in_long = False
                df.at[i, 'signal_type'] = f'Exit Long ({exit_reason})'
                df.at[i, 'position'] = 'Hold'
                entry_price = np.nan
                stop_loss = np.nan
                target_price = np.nan
        
        elif in_short:
            # Update trailing stop
            new_stop = current['high'] + current['atr'] * atr_mult
            stop_loss = min(stop_loss, new_stop)
            
            # Check exit conditions
            exit_reason = ''
            exit_signal = False
            
            # ATR stop hit
            if current['close'] > stop_loss:
                exit_reason = 'ATR SL'
                exit_signal = True
            # Score drop exit
            elif scaled_score > (5.0 + exit_score_drop_threshold):
                exit_reason = f'Score Drop ({scaled_score:.1f})'
                exit_signal = True
            # EMA cross exit
            elif ema_fast_med_cross_buy:
                exit_reason = 'EMA Cross'
                exit_signal = True
            # Fib target hit
            elif not np.isnan(target_price) and current['low'] <= target_price:
                exit_reason = 'Fib Target'
                exit_signal = True
            
            if exit_signal:
                in_short = False
                df.at[i, 'signal_type'] = f'Exit Short ({exit_reason})'
                df.at[i, 'position'] = 'Hold'
                entry_price = np.nan
                stop_loss = np.nan
                target_price = np.nan
        
        # Update signal strength and reason
        df.at[i, 'signal_strength'] = scaled_score
        df.at[i, 'signal_reason'] = reason
        
        # Update position tracking
        if in_long:
            df.at[i, 'position'] = 'Long'
        elif in_short:
            df.at[i, 'position'] = 'Short'
        
        # Update entry price, stop loss, and target
        df.at[i, 'entry_price'] = entry_price
        df.at[i, 'stop_loss'] = stop_loss
        df.at[i, 'target_price'] = target_price
    
    return df

# Load your CSV data
df = pd.read_csv('NIFTY_signals_20250402_2229.csv', parse_dates=['datetime'])

# Process the signals
df = process_signals(df)

# Save the enhanced data
df.to_csv('enhanced_signals.csv', index=False)

print("Signal processing complete. Enhanced data saved to enhanced_signals.csv")