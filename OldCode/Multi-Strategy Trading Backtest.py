import pandas as pd
import pandas_ta as ta
from datetime import time

# Load data
df = pd.read_csv("test 1 week.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Calculate Indicators
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

# SuperTrend Calculation (simplified)
def supertrend(df, period=10, multiplier=3):
    atr = df['ATR']
    hl2 = (df['high'] + df['low']) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(1, len(df)):
        if df['close'].iloc[i-1] > supertrend.iloc[i-1]:
            supertrend.iloc[i] = lowerband.iloc[i]
            direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = -1
        if direction.iloc[i] == 1 and supertrend.iloc[i] < supertrend.iloc[i-1]:
            supertrend.iloc[i] = supertrend.iloc[i-1]
        elif direction.iloc[i] == -1 and supertrend.iloc[i] > supertrend.iloc[i-1]:
            supertrend.iloc[i] = supertrend.iloc[i-1]
    return supertrend, direction

df['SuperTrend'], df['SuperTrend_dir'] = supertrend(df)

# Pivot Points (Daily)
df['Date'] = df.index.date
pivot_points = df.groupby('Date').agg({
    'high': 'max',
    'low': 'min',
    'close': 'last'
}).shift(1)  # Use previous day's data
pivot_points['Pivot'] = (pivot_points['high'] + pivot_points['low'] + pivot_points['close']) / 3
pivot_points['R1'] = 2 * pivot_points['Pivot'] - pivot_points['low']
pivot_points['S1'] = 2 * pivot_points['Pivot'] - pivot_points['high']
pivot_points['R2'] = pivot_points['Pivot'] + (pivot_points['high'] - pivot_points['low'])
pivot_points['S2'] = pivot_points['Pivot'] - (pivot_points['high'] - pivot_points['low'])
pivot_points['R3'] = pivot_points['R1'] + (pivot_points['high'] - pivot_points['low'])
pivot_points['S3'] = pivot_points['S1'] - (pivot_points['high'] - pivot_points['low'])

# Map Pivot Points back to main dataframe
df = df.merge(pivot_points[['Pivot', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3']], left_on=df['Date'], right_index=True)

# Approximate AlphaTrend (based on your system logic)
df['AlphaTrend'] = 0.65 * df['close'] + 0.25 * df['close'].shift(1) + 0.1 * df['RSI'] * (df['ATR'] / df['close'].shift(1))
df['AlphaTrend'].fillna(df['close'], inplace=True)

# Signal Generation
def generate_signal(row, prev_row, df):
    atr = row['ATR']
    ema7, ema14, ema21, ema50 = row['EMA7'], row['EMA14'], row['EMA21'], row['EMA50']
    supertrend = row['SuperTrend']
    supertrend_dir = row['SuperTrend_dir']
    rsi = row['RSI']
    adx = row['ADX']
    macd, macd_signal = row['MACD'], row['MACD_signal']
    pivot_s1, pivot_r1, pivot_r2, pivot_s2 = row['S1'], row['R1'], row['R2'], row['S2']
    
    # Market phase detection
    current_time = row.name.time()
    is_trending = adx > 30
    is_ranging = adx < 20
    is_volatile = atr > 1.8 * df['ATR'].mean()  # Proxy for VIX >25
    
    # Entry score calculation (adjusted for missing volume)
    entry_score = (
        20 * int(row['close'] > row['AlphaTrend']) +  # AlphaTrend
        15 * int(supertrend_dir == 1) +  # SuperTrend
        15 * int(ema7 > ema14 > ema21 > ema50) +  # EMA stack
        12 * int(row['close'] > pivot_r1 or row['close'] < pivot_s1) +  # Pivot breakout
        10 * int(rsi < 40 and rsi > prev_row['RSI']) +  # RSI divergence
        8 * int(row['close'] > ema21 and row['close'] < prev_row['close']) +  # Pullback
        5 * int(macd > macd_signal) +  # MACD
        5 * int(atr > df['ATR'].mean())  # Volatility filter
    )
    
    # Strategy-specific conditions
    momentum_ok = is_trending and adx > 25 and ema7 > ema14
    reversal_ok = is_ranging and rsi < 40 and row['close'] < pivot_r1
    breakout_ok = is_volatile and row['close'] > pivot_r1
    scalping_ok = is_ranging and abs(ema7 - ema14) < 0.3 * atr
    pullback_ok = is_trending and row['close'] > ema21
    gap_and_go_ok = current_time < time(9, 30) and row['open'] > prev_row['close'] * 1.015
    bull_flag_ok = is_trending  # Volume condition removed
    
    # Session handling for ATR multiplier
    if current_time < time(9, 45):
        atr_mult = 3.0 if atr > 18 else 2.2
    elif current_time < time(14, 0):
        atr_mult = 2.5
    else:
        atr_mult = 1.8
    
    # Trailing Stop Logic (bar-by-bar adjustment)
    stop_loss = row['close'] - (1.2 * (ema7 - ema14) if scalping_ok else 2.5 * atr if momentum_ok else 0.5 * atr)
    
    # Final signal
    if entry_score >= 65 and any([momentum_ok, reversal_ok, breakout_ok, scalping_ok, pullback_ok, gap_and_go_ok, bull_flag_ok]):
        return {
            'signal': 'buy' if row['close'] > row['AlphaTrend'] else 'sell',
            'strategy': 'momentum' if momentum_ok else 'reversal' if reversal_ok else 'breakout' if breakout_ok else 'scalping' if scalping_ok else 'pullback' if pullback_ok else 'gap_and_go' if gap_and_go_ok else 'bull_flag',
            'stop_loss': stop_loss,
            'atr_mult': atr_mult
        }
    return {'signal': None, 'strategy': None, 'stop_loss': None, 'atr_mult': atr_mult}

# Backtest
trades = []
position = None
highest_price = None

for i in range(1, len(df)):
    row = df.iloc[i]
    prev_row = df.iloc[i-1]
    signal = generate_signal(row, prev_row, df)
    
    # Update trailing stop if in a position
    if position and position['signal'] == 'buy':
        highest_price = max(highest_price, row['high'])
        trailing_stop = highest_price - 3.5 * row['ATR'] * position['atr_mult']
        if row['low'] <= trailing_stop:
            profit = trailing_stop - position['entry_price']
            trades.append({
                'strategy': position['strategy'],
                'profit': profit,
                'entry_time': position['entry_time'],
                'exit_time': row.name,
                'entry_price': position['entry_price'],
                'exit_price': trailing_stop
            })
            position = None
            highest_price = None
    
    # Enter new position
    if signal['signal'] == 'buy' and position is None:
        position = {
            'signal': 'buy',
            'entry_price': row['open'],
            'entry_time': row.name,
            'strategy': signal['strategy'],
            'stop_loss': signal['stop_loss'],
            'atr_mult': signal['atr_mult']
        }
        highest_price = row['high']
    
    # Exit at session end (15:25)
    if row.name.time() >= time(15, 25) and position:
        profit = row['close'] - position['entry_price']
        trades.append({
            'strategy': position['strategy'],
            'profit': profit,
            'entry_time': position['entry_time'],
            'exit_time': row.name,
            'entry_price': position['entry_price'],
            'exit_price': row['close']
        })
        position = None
        highest_price = None

# Results
trades_df = pd.DataFrame(trades)
total_profit = trades_df['profit'].sum()
win_rate = len(trades_df[trades_df['profit'] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0
profit_factor = trades_df[trades_df['profit'] > 0]['profit'].sum() / abs(trades_df[trades_df['profit'] < 0]['profit'].sum()) if len(trades_df[trades_df['profit'] < 0]) > 0 else float('inf')
max_drawdown = (trades_df['profit'].cumsum().cummax() - trades_df['profit'].cumsum()).max()

print(f"Total Profit: {total_profit:.2f} points")
print(f"Number of Trades: {len(trades_df)}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Max Drawdown: {max_drawdown:.2f} points")
print("\nTrades:")
print(trades_df[['entry_time', 'exit_time', 'strategy', 'entry_price', 'exit_price', 'profit']])