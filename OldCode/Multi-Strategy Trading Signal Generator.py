import pandas as pd
import pandas_ta as ta
from datetime import time

def generate_signal(row, prev_row, df):
    # Calculate indicators
    atr = row['ATR']
    alphatrend = 0.65 * row['close'] + 0.25 * prev_row['AlphaTrend'] + 0.1 * row['RSI'] * (atr / prev_row['close'])
    ema7, ema14, ema21, ema50 = row['EMA7'], row['EMA14'], row['EMA21'], row['EMA50']
    supertrend = row['SuperTrend']
    rsi = row['RSI']
    adx = row['ADX']
    macd, macd_signal = row['MACD'], row['MACD_signal']
    pivot_s1, pivot_r1 = row['Pivot_S1'], row['Pivot_R1']
    
    # Market phase detection
    current_time = pd.to_datetime(row['datetime']).time()
    is_trending = adx > 30
    is_ranging = adx < 20
    is_volatile = atr > 1.8 * df['ATR'].mean()  # Proxy for VIX >25
    
    # Entry score calculation
    entry_score = (
        20 * int(row['close'] > alphatrend) +  # AlphaTrend
        15 * int(row['close'] > supertrend) +  # SuperTrend
        15 * int(ema7 > ema14 > ema21 > ema50) +  # EMA stack
        12 * int(row['close'] > pivot_r1 or row['close'] < pivot_s1) +  # Pivot breakout
        10 * int(rsi < 40 and rsi > prev_row['RSI']) +  # RSI divergence
        10 * int(row['volume'] > 2 * df['volume'].rolling(20).mean()) +  # Volume Surge
        8 * int(row['close'] > ema21 and row['close'] < prev_row['close']) +  # Pullback
        5 * int(macd > macd_signal) +  # MACD
        5 * int(atr > 12)  # Volatility filter
    )
    
    # Strategy-specific conditions
    momentum_ok = is_trending and adx > 25 and ema7 > ema14
    reversal_ok = is_ranging and rsi < 40 and row['close'] < pivot_r1
    breakout_ok = is_volatile and row['close'] > pivot_r1
    scalping_ok = is_ranging and abs(ema7 - ema14) < 0.3 * atr
    pullback_ok = is_trending and row['close'] > ema21
    gap_and_go_ok = current_time < time(9, 30) and row['open'] > prev_row['close'] * 1.015
    bull_flag_ok = is_trending and row['volume'] > df['volume'].rolling(20).mean()
    
    # Session handling
    if current_time < time(9, 45):
        atr_mult = 3.0 if atr > 18 else 2.2
    elif current_time < time(14, 0):
        atr_mult = 2.5
    else:
        atr_mult = 1.8
    
    # Final signal
    if entry_score >= 65 and any([momentum_ok, reversal_ok, breakout_ok, scalping_ok, pullback_ok, gap_and_go_ok, bull_flag_ok]):
        return {
            'signal': 'buy' if row['close'] > alphatrend else 'sell',
            'strategy': 'momentum' if momentum_ok else 'reversal' if reversal_ok else 'breakout' if breakout_ok else 'scalping' if scalping_ok else 'pullback' if pullback_ok else 'gap_and_go' if gap_and_go_ok else 'bull_flag',
            'stop_loss': row['close'] - (1.2 * (ema7 - ema14) if scalping_ok else 2.5 * atr if momentum_ok else 0.5 * atr)
        }
    return {'signal': None, 'strategy': None, 'stop_loss': None}

# Example usage
def backtest(df):
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['EMA7'] = ta.ema(df['close'], length=7)
    df['EMA14'] = ta.ema(df['close'], length=14)
    df['EMA21'] = ta.ema(df['close'], length=21)
    df['EMA50'] = ta.ema(df['close'], length=50)
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx['ADX_14']
    # Add SuperTrend and Pivot Points via custom functions
    df['AlphaTrend'] = df['close']  # Placeholder, replace with actual calculation
    df['SuperTrend'] = df['close']  # Placeholder
    df['Pivot_S1'] = df['close']  # Placeholder
    df['Pivot_R1'] = df['close']  # Placeholder
    
    trades = []
    position = None
    for i in range(1, len(df)):
        signal = generate_signal(df.iloc[i], df.iloc[i-1], df)
        if signal['signal'] == 'buy' and position is None:
            position = {'entry_price': df.iloc[i]['open'], 'entry_time': df.iloc[i]['datetime'], 'strategy': signal['strategy'], 'stop_loss': signal['stop_loss']}
        elif signal['signal'] == 'sell' and position:
            profit = df.iloc[i]['open'] - position['entry_price']
            trades.append({'strategy': position['strategy'], 'profit': profit, 'entry_time': position['entry_time'], 'exit_time': df.iloc[i]['datetime']})
            position = None
        # Close at session end (15:25)
        if pd.to_datetime(df.iloc[i]['datetime']).time() >= time(15, 25) and position:
            profit = df.iloc[i]['close'] - position['entry_price']
            trades.append({'strategy': position['strategy'], 'profit': profit, 'entry_time': position['entry_time'], 'exit_time': df.iloc[i]['datetime']})
            position = None
    
    return pd.DataFrame(trades)