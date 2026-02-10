import pandas as pd
import pandas_ta as ta
from datetime import time
import uuid

# Helper function to calculate SuperTrend
def supertrend(df, period=10, mult=3.0):
    atr = df['ATR_10']
    hl2 = (df['high'] + df['low']) / 2
    ub = hl2 + mult * atr
    lb = hl2 - mult * atr
    st = pd.Series(index=df.index, dtype=float)
    dir = pd.Series(index=df.index, dtype=int)
    for i in range(len(df)):
        if i == 0:
            st.iloc[i], dir.iloc[i] = lb.iloc[i], 1
        else:
            if df['close'].iloc[i-1] > st.iloc[i-1]:
                st.iloc[i], dir.iloc[i] = lb.iloc[i], 1
            else:
                st.iloc[i], dir.iloc[i] = ub.iloc[i], -1
            if dir.iloc[i] == 1 and st.iloc[i] < st.iloc[i-1]:
                st.iloc[i] = st.iloc[i-1]
            if dir.iloc[i] == -1 and st.iloc[i] > st.iloc[i-1]:
                st.iloc[i] = st.iloc[i-1]
    return st, dir

# Helper function to calculate AlphaTrend
def compute_alpha_trend(df):
    alpha_trend = pd.Series(index=df.index, dtype=float)
    atr_weight = df['ATR_14'] / df['close'].shift(1).ffill()
    for i in range(len(df)):
        if i == 0:
            alpha_trend.iloc[i] = df['close'].iloc[i]
        else:
            alpha_trend.iloc[i] = (
                0.65 * df['close'].iloc[i]
                + 0.25 * alpha_trend.iloc[i-1]
                + 0.1 * df['RSI_10'].iloc[i] * atr_weight.iloc[i]
            )
    df['AlphaTrend'] = alpha_trend.ffill().fillna(df['close'])
    return df

# Strategy 1: Original Mean-Reversion
def original_mean_reversion(df):
    df = df.copy()
    df['support'] = df['low'].rolling(window=20).min()
    df['resistance'] = df['high'].rolling(window=20).max()
    df['buy_signal'] = (df['close'] < df['support'] * 1.01) & (df['close'].shift(1) > df['support'].shift(1) * 1.01)
    df['sell_signal'] = (df['close'] > df['resistance'] * 0.99) & (df['close'].shift(1) < df['resistance'].shift(1) * 0.99)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    trades = []
    position = None
    for i in range(20, len(df)):
        if position is None:
            if df['buy_signal'].iloc[i]:
                entry = df['close'].iloc[i]
                sl = entry - 1 * df['ATR'].iloc[i]
                tp = entry + 1.5 * df['ATR'].iloc[i]
                position = {'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
            elif df['sell_signal'].iloc[i]:
                entry = df['close'].iloc[i]
                sl = entry + 1 * df['ATR'].iloc[i]
                tp = entry - 1.5 * df['ATR'].iloc[i]
                position = {'type': 'sell', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
        else:
            current_price = df['close'].iloc[i]
            if (position['type'] == 'buy' and current_price <= position['sl']) or (position['type'] == 'sell' and current_price >= position['sl']):
                position['exit'] = position['sl']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
            elif (position['type'] == 'buy' and current_price >= position['tp']) or (position['type'] == 'sell' and current_price <= position['tp']):
                position['exit'] = position['tp']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
    
    total_pnl = sum((trade['exit'] - trade['entry']) if trade['type'] == 'buy' else (trade['entry'] - trade['exit']) for trade in trades)
    return total_pnl, trades

# Strategy 2: Refined Mean-Reversion
def refined_mean_reversion(df):
    df = df.copy()
    df['support'] = df['low'].rolling(window=20).min()
    df['resistance'] = df['high'].rolling(window=20).max()
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['buy_signal'] = (df['close'] < df['support'] * 1.01) & (df['RSI'] < 35)
    df['sell_signal'] = (df['close'] > df['resistance'] * 0.99) & (df['RSI'] > 65)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    trades = []
    position = None
    for i in range(20, len(df)):
        if position is None:
            if df['buy_signal'].iloc[i]:
                entry = df['close'].iloc[i]
                sl = entry - 1 * df['ATR'].iloc[i]
                tp = entry + 1.5 * df['ATR'].iloc[i]
                position = {'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
            elif df['sell_signal'].iloc[i]:
                entry = df['close'].iloc[i]
                sl = entry + 1 * df['ATR'].iloc[i]
                tp = entry - 1.5 * df['ATR'].iloc[i]
                position = {'type': 'sell', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
        else:
            current_price = df['close'].iloc[i]
            if (position['type'] == 'buy' and current_price <= position['sl']) or (position['type'] == 'sell' and current_price >= position['sl']):
                position['exit'] = position['sl']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
            elif (position['type'] == 'buy' and current_price >= position['tp']) or (position['type'] == 'sell' and current_price <= position['tp']):
                position['exit'] = position['tp']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
    
    total_pnl = sum((trade['exit'] - trade['entry']) if trade['type'] == 'buy' else (trade['entry'] - trade['exit']) for trade in trades)
    return total_pnl, trades

# Strategy 3: Momentum Breakout
def momentum_breakout(df):
    df = df.copy()
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ATR_mean'] = df['ATR'].rolling(window=5).mean()
    df['ATR_std'] = df['ATR'].rolling(window=5).std()
    df['breakout_level'] = df['high'].rolling(window=20).max() + 1 * df['ATR']
    df['buy_signal'] = (df['close'] > df['breakout_level']) & (df['ATR'] > df['ATR_mean'] + 1.8 * df['ATR_std'])
    
    trades = []
    position = None
    for i in range(20, len(df)):
        if position is None and df['buy_signal'].iloc[i]:
            entry = df['close'].iloc[i]
            sl = entry - 0.7 * df['ATR'].iloc[i]
            tp = entry + 2 * df['ATR'].iloc[i]
            position = {'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
        elif position is not None:
            current_price = df['close'].iloc[i]
            if current_price <= position['sl']:
                position['exit'] = position['sl']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
            elif current_price >= position['tp']:
                position['exit'] = position['tp']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
    
    total_pnl = sum((trade['exit'] - trade['entry']) for trade in trades)
    return total_pnl, trades

# Strategy 4: SMA Crossover
def sma_crossover(df):
    df = df.copy()
    df['SMA10'] = df['close'].rolling(window=10).mean()
    df['SMA50'] = df['close'].rolling(window=50).mean()
    df['buy_signal'] = (df['SMA10'] > df['SMA50']) & (df['SMA10'].shift(1) < df['SMA50'].shift(1))
    df['sell_signal'] = (df['SMA10'] < df['SMA50']) & (df['SMA10'].shift(1) > df['SMA50'].shift(1))
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    trades = []
    position = None
    for i in range(50, len(df)):
        if position is None and df['buy_signal'].iloc[i]:
            entry = df['close'].iloc[i]
            sl = entry - 0.7 * df['ATR'].iloc[i]
            tp = entry + 1.5 * df['ATR'].iloc[i]
            position = {'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
        elif position is not None:
            current_price = df['close'].iloc[i]
            if df['sell_signal'].iloc[i] or current_price <= position['sl'] or current_price >= position['tp']:
                exit_price = position['sl'] if current_price <= position['sl'] else (position['tp'] if current_price >= position['tp'] else current_price)
                position['exit'] = exit_price
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
    
    total_pnl = sum((trade['exit'] - trade['entry']) for trade in trades)
    return total_pnl, trades

# Strategy 5: Bollinger Scalping
def bollinger_scalping(df):
    df = df.copy()
    bb = ta.bbands(df['close'], length=20, std=2)
    df['BB_upper'] = bb['BBU_20_2.0']
    df['BB_lower'] = bb['BBL_20_2.0']
    df['buy_signal'] = df['close'] < df['BB_lower']
    df['sell_signal'] = df['close'] > df['BB_upper']
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    trades = []
    position = None
    for i in range(20, len(df)):
        if position is None:
            if df['buy_signal'].iloc[i]:
                entry = df['close'].iloc[i]
                sl = entry - 0.7 * df['ATR'].iloc[i]
                tp = entry + 1.8 * df['ATR'].iloc[i]
                position = {'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
            elif df['sell_signal'].iloc[i]:
                entry = df['close'].iloc[i]
                sl = entry + 0.7 * df['ATR'].iloc[i]
                tp = entry - 1.8 * df['ATR'].iloc[i]
                position = {'type': 'sell', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
        else:
            current_price = df['close'].iloc[i]
            if (position['type'] == 'buy' and current_price <= position['sl']) or (position['type'] == 'sell' and current_price >= position['sl']):
                position['exit'] = position['sl']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
            elif (position['type'] == 'buy' and current_price >= position['tp']) or (position['type'] == 'sell' and current_price <= position['tp']):
                position['exit'] = position['tp']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
    
    total_pnl = sum((trade['exit'] - trade['entry']) if trade['type'] == 'buy' else (trade['entry'] - trade['exit']) for trade in trades)
    return total_pnl, trades

# Strategy 6: Adjusted AlphaTrend
def adjusted_alphatrend(df):
    df = df.copy()
    df['RSI_10'] = ta.rsi(df['close'], length=10).ffill()
    df['ATR_10'] = ta.atr(df['high'], df['low'], df['close'], length=10).ffill()
    macd = ta.macd(df['close']).ffill()
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['RSI_slope'] = df['RSI_10'].pct_change().ffill()
    df['SuperTrend'], df['SuperTrend_dir'] = supertrend(df)
    
    df['crossover'] = (df['close'] > df['close'].shift(2)) & (df['close'].shift(1) <= df['close'].shift(2))
    df['momentum'] = abs(df['close'] - df['close'].shift(2)) > 1
    df['rsi_buy'] = (df['RSI_10'] > 50) & (df['RSI_10'] < 70)
    df['rsi_sell'] = (df['RSI_10'] > 30) & (df['RSI_10'] < 50)
    df['supertrend_buy'] = df['SuperTrend_dir'] == 1
    df['supertrend_sell'] = df['SuperTrend_dir'] == -1
    df['rsi_slope_pos'] = df['RSI_slope'] > 0
    df['rsi_slope_neg'] = df['RSI_slope'] < 0
    df['macd_buy'] = df['MACD'] > df['MACD_signal']
    df['macd_sell'] = df['MACD'] < df['MACD_signal']
    
    df['buy_score'] = (
        df['crossover'].astype(int) + df['momentum'].astype(int) + df['rsi_buy'].astype(int) +
        df['supertrend_buy'].astype(int) + df['rsi_slope_pos'].astype(int) + df['macd_buy'].astype(int)
    )
    df['sell_score'] = (
        (~df['crossover']).astype(int) + (~df['momentum']).astype(int) + df['rsi_sell'].astype(int) +
        df['supertrend_sell'].astype(int) + df['rsi_slope_neg'].astype(int) + df['macd_sell'].astype(int)
    )
    
    df['buy_signal'] = df['buy_score'] >= 5
    df['sell_signal'] = df['sell_score'] >= 5
    
    trades = []
    position = None
    for i in range(20, len(df)):
        current_time = df.index[i].time()
        is_morning = current_time < time(9, 30)
        if position is None:
            if df['buy_signal'].iloc[i]:
                entry = df['close'].iloc[i]
                sl = entry - (1 if is_morning else 2.5) * df['ATR_10'].iloc[i]
                tp = entry + 2.5 * df['ATR_10'].iloc[i]
                position = {'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i], 'trailing_sl': sl}
            elif df['sell_signal'].iloc[i]:
                entry = df['close'].iloc[i]
                sl = entry + (1 if is_morning else 2.5) * df['ATR_10'].iloc[i]
                tp = entry - 2.5 * df['ATR_10'].iloc[i]
                position = {'type': 'sell', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i], 'trailing_sl': sl}
        else:
            current_price = df['close'].iloc[i]
            rsi = df['RSI_10'].iloc[i]
            atr = df['ATR_10'].iloc[i]
            if position['type'] == 'buy' and (rsi > 75 or not is_morning):
                position['trailing_sl'] = max(position['trailing_sl'], current_price - 2 * atr)
            elif position['type'] == 'sell' and (rsi < 25 or not is_morning):
                position['trailing_sl'] = min(position['trailing_sl'], current_price + 2 * atr)
            
            if (position['type'] == 'buy' and current_price <= position['trailing_sl']) or (position['type'] == 'sell' and current_price >= position['trailing_sl']):
                position['exit'] = position['trailing_sl']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
            elif (position['type'] == 'buy' and current_price >= position['tp']) or (position['type'] == 'sell' and current_price <= position['tp']):
                position['exit'] = position['tp']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
    
    total_pnl = sum((trade['exit'] - trade['entry']) if trade['type'] == 'buy' else (trade['entry'] - trade['exit']) for trade in trades)
    return total_pnl, trades

# Strategy 7: Momentum
def momentum(df):
    df = df.copy()
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14'].ffill()
    df['EMA7'] = ta.ema(df['close'], length=7)
    df['EMA14'] = ta.ema(df['close'], length=14)
    df['RSI_10'] = ta.rsi(df['close'], length=10).ffill()
    df = compute_alpha_trend(df)
    df['buy_signal'] = (
        (df['ADX'] > 20) & (df['EMA7'] > df['EMA14']) & (df['close'] > df['AlphaTrend']) & (df['RSI_10'] > 50) &
        (df.index.map(lambda x: x.time()) < time(14, 0))
    )
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    trades = []
    position = None
    for i in range(20, len(df)):
        if position is None and df['buy_signal'].iloc[i]:
            entry = df['close'].iloc[i]
            sl = entry - 0.7 * df['ATR'].iloc[i]
            tp = entry + 1.8 * df['ATR'].iloc[i]
            position = {'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
        elif position is not None:
            current_price = df['close'].iloc[i]
            if current_price <= position['sl']:
                position['exit'] = position['sl']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
            elif current_price >= position['tp']:
                position['exit'] = position['tp']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
    
    total_pnl = sum((trade['exit'] - trade['entry']) for trade in trades)
    return total_pnl, trades

# Strategy 8: Reversal
def reversal(df):
    df = df.copy()
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14'].ffill()
    df['RSI_10'] = ta.rsi(df['close'], length=10).ffill()
    df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
    df['RSI_slope'] = df['RSI_10'].pct_change().ffill()
    df['buy_signal'] = (df['ADX'] < 20) & (df['RSI_10'] < 35) & (df['close'] < df['pivot']) & (df['RSI_slope'] < -0.01)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    trades = []
    position = None
    for i in range(20, len(df)):
        if position is None and df['buy_signal'].iloc[i]:
            entry = df['close'].iloc[i]
            sl = entry - 0.7 * df['ATR'].iloc[i]
            tp = entry + 2 * df['ATR'].iloc[i]
            position = {'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
        elif position is not None:
            current_price = df['close'].iloc[i]
            if current_price <= position['sl']:
                position['exit'] = position['sl']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
            elif current_price >= position['tp']:
                position['exit'] = position['tp']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
    
    total_pnl = sum((trade['exit'] - trade['entry']) for trade in trades)
    return total_pnl, trades

# Strategy 9: Breakout
def breakout(df):
    df = df.copy()
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ATR_mean'] = df['ATR'].rolling(window=5).mean()
    df['ATR_std'] = df['ATR'].rolling(window=5).std()
    df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
    df['R1'] = df['pivot'] + 0.5 * df['ATR']
    macd = ta.macd(df['close']).ffill()
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['buy_signal'] = (df['ATR'] > df['ATR_mean'] + 1.8 * df['ATR_std']) & (df['close'] > df['R1']) & (df['MACD'] > df['MACD_signal'])
    
    trades = []
    position = None
    for i in range(20, len(df)):
        if position is None and df['buy_signal'].iloc[i]:
            entry = df['close'].iloc[i]
            sl = entry - 0.7 * df['ATR'].iloc[i]
            tp = entry + 2 * df['ATR'].iloc[i]
            position = {'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
        elif position is not None:
            current_price = df['close'].iloc[i]
            if current_price <= position['sl']:
                position['exit'] = position['sl']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
            elif current_price >= position['tp']:
                position['exit'] = position['tp']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
    
    total_pnl = sum((trade['exit'] - trade['entry']) for trade in trades)
    return total_pnl, trades

# Strategy 10: Scalping
def scalping(df):
    df = df.copy()
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14'].ffill()
    df['EMA7'] = ta.ema(df['close'], length=7)
    df['EMA14'] = ta.ema(df['close'], length=14)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ATR_mean'] = df['ATR'].rolling(window=5).mean()
    df['RSI_10'] = ta.rsi(df['close'], length=10).ffill()
    df['buy_signal'] = (
        (df['ADX'] < 20) & (abs(df['EMA7'] - df['EMA14']) < 0.2 * df['ATR']) &
        (df['ATR'] > 1.5 * df['ATR_mean']) & (df['RSI_10'] < 50)
    )
    
    trades = []
    position = None
    for i in range(20, len(df)):
        if position is None and df['buy_signal'].iloc[i]:
            entry = df['close'].iloc[i]
            sl = entry - 0.7 * df['ATR'].iloc[i]
            tp = entry + 1.5 * df['ATR'].iloc[i]
            position = {'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
        elif position is not None:
            current_price = df['close'].iloc[i]
            if current_price <= position['sl']:
                position['exit'] = position['sl']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
            elif current_price >= position['tp']:
                position['exit'] = position['tp']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
    
    total_pnl = sum((trade['exit'] - trade['entry']) for trade in trades)
    return total_pnl, trades

# Strategy 11: MA_Crossover
def ma_crossover(df):
    df = df.copy()
    df['EMA7'] = ta.ema(df['close'], length=7)
    df['EMA14'] = ta.ema(df['close'], length=14)
    df['EMA21'] = ta.ema(df['close'], length=21)
    df['EMA50'] = ta.ema(df['close'], length=50)
    df['RSI_14'] = ta.rsi(df['close'], length=14).ffill()
    df['ATR_10'] = ta.atr(df['high'], df['low'], df['close'], length=10).ffill()
    df['SuperTrend'], df['SuperTrend_dir'] = supertrend(df)
    df['buy_signal'] = (
        (df['EMA7'] > df['EMA14']) & (df['EMA14'] > df['EMA21']) & (df['EMA21'] > df['EMA50']) &
        (df['RSI_14'] > 50) & (df['SuperTrend_dir'] == 1)
    )
    
    trades = []
    position = None
    for i in range(50, len(df)):
        if position is None and df['buy_signal'].iloc[i]:
            entry = df['close'].iloc[i]
            sl = entry - 0.7 * df['ATR_10'].iloc[i]
            tp = entry + 1.5 * df['ATR_10'].iloc[i]
            position = {'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
        elif position is not None:
            current_price = df['close'].iloc[i]
            if current_price <= position['sl']:
                position['exit'] = position['sl']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
            elif current_price >= position['tp']:
                position['exit'] = position['tp']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
    
    total_pnl = sum((trade['exit'] - trade['entry']) for trade in trades)
    return total_pnl, trades

# Strategy 12: Pivot_Point
def pivot_point(df):
    df = df.copy()
    df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['R1'] = df['pivot'] + 0.5 * df['ATR']
    df['S1'] = df['pivot'] - 0.5 * df['ATR']
    df['buy_signal'] = (df['close'] > df['R1']) | ((df['close'] < df['S1']) & (df['close'].shift(1) > df['S1'].shift(1)))
    df['sell_signal'] = (df['close'] < df['S1']) | ((df['close'] > df['R1']) & (df['close'].shift(1) < df['R1'].shift(1)))
    
    trades = []
    position = None
    for i in range(20, len(df)):
        if position is None:
            if df['buy_signal'].iloc[i]:
                entry = df['close'].iloc[i]
                sl = entry - 0.7 * df['ATR'].iloc[i]
                tp = entry + 1.5 * df['ATR'].iloc[i]
                position = {'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
            elif df['sell_signal'].iloc[i]:
                entry = df['close'].iloc[i]
                sl = entry + 0.7 * df['ATR'].iloc[i]
                tp = entry - 1.5 * df['ATR'].iloc[i]
                position = {'type': 'sell', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
        else:
            current_price = df['close'].iloc[i]
            if (position['type'] == 'buy' and current_price <= position['sl']) or (position['type'] == 'sell' and current_price >= position['sl']):
                position['exit'] = position['sl']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
            elif (position['type'] == 'buy' and current_price >= position['tp']) or (position['type'] == 'sell' and current_price <= position['tp']):
                position['exit'] = position['tp']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
    
    total_pnl = sum((trade['exit'] - trade['entry']) if trade['type'] == 'buy' else (trade['entry'] - trade['exit']) for trade in trades)
    return total_pnl, trades

# Strategy 13: Pullback
def pullback(df):
    df = df.copy()
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14'].ffill()
    df['EMA21'] = ta.ema(df['close'], length=21)
    df['RSI_10'] = ta.rsi(df['close'], length=10).ffill()
    df['buy_signal'] = (
        (df['ADX'] > 20) & (df['close'] > df['EMA21']) & (df['close'] < df['close'].shift(1)) & (df['RSI_10'] > 45)
    )
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    trades = []
    position = None
    for i in range(21, len(df)):
        if position is None and df['buy_signal'].iloc[i]:
            entry = df['close'].iloc[i]
            sl = entry - 0.7 * df['ATR'].iloc[i]
            tp = entry + 1.5 * df['ATR'].iloc[i]
            position = {'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
        elif position is not None:
            current_price = df['close'].iloc[i]
            if current_price <= position['sl']:
                position['exit'] = position['sl']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
            elif current_price >= position['tp']:
                position['exit'] = position['tp']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
    
    total_pnl = sum((trade['exit'] - trade['entry']) for trade in trades)
    return total_pnl, trades

# Strategy 14: Gap_and_Go
def gap_and_go(df):
    df = df.copy()
    df['RSI_10'] = ta.rsi(df['close'], length=10).ffill()
    df['prev_close'] = df['close'].shift(1)
    df['buy_signal'] = (
        (df.index.map(lambda x: x.time()) == time(9, 15)) &
        (df['open'] > df['prev_close'] * 1.01) & (df['RSI_10'] < 70)
    )
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    trades = []
    position = None
    for i in range(20, len(df)):
        if position is None and df['buy_signal'].iloc[i]:
            entry = df['close'].iloc[i]
            sl = entry - 0.7 * df['ATR'].iloc[i]
            tp = entry + 2 * df['ATR'].iloc[i]
            position = {'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
        elif position is not None:
            current_price = df['close'].iloc[i]
            if current_price <= position['sl']:
                position['exit'] = position['sl']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
            elif current_price >= position['tp']:
                position['exit'] = position['tp']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
    
    total_pnl = sum((trade['exit'] - trade['entry']) for trade in trades)
    return total_pnl, trades

# Strategy 15: Bull_Flag
def bull_flag(df):
    df = df.copy()
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14'].ffill()
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ATR_mean'] = df['ATR'].rolling(window=5).mean()
    df['RSI_10'] = ta.rsi(df['close'], length=10).ffill()
    macd = ta.macd(df['close']).ffill()
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['buy_signal'] = (
        (df['ADX'] > 20) & (df['ATR'] > 1.5 * df['ATR_mean']) & (df['RSI_10'] > 50) & (df['MACD'] > df['MACD_signal'])
    )
    
    trades = []
    position = None
    for i in range(20, len(df)):
        if position is None and df['buy_signal'].iloc[i]:
            entry = df['close'].iloc[i]
            sl = entry - 0.7 * df['ATR'].iloc[i]
            tp = entry + 1.8 * df['ATR'].iloc[i]
            position = {'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
        elif position is not None:
            current_price = df['close'].iloc[i]
            if current_price <= position['sl']:
                position['exit'] = position['sl']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
            elif current_price >= position['tp']:
                position['exit'] = position['tp']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
    
    total_pnl = sum((trade['exit'] - trade['entry']) for trade in trades)
    return total_pnl, trades

# Strategy 16: Ensemble Voting
def ensemble_voting(df):
    df = df.copy()
    
    # Run individual strategies to get signals
    strategies = [
        original_mean_reversion, refined_mean_reversion, momentum_breakout, sma_crossover,
        bollinger_scalping, adjusted_alphatrend, momentum, reversal, breakout, scalping,
        ma_crossover, pivot_point, pullback, gap_and_go, bull_flag
    ]
    
    buy_signals = pd.DataFrame(index=df.index, columns=[f'strategy_{i}' for i in range(len(strategies))])
    sell_signals = pd.DataFrame(index=df.index, columns=[f'strategy_{i}' for i in range(len(strategies))])
    
    for idx, strategy in enumerate(strategies):
        temp_df = df.copy()
        if strategy.__name__ == 'adjusted_alphatrend':
            temp_df['RSI_10'] = ta.rsi(temp_df['close'], length=10).ffill()
            temp_df['ATR_10'] = ta.atr(temp_df['high'], temp_df['low'], temp_df['close'], length=10).ffill()
            macd = ta.macd(temp_df['close']).ffill()
            temp_df['MACD'] = macd['MACD_12_26_9']
            temp_df['MACD_signal'] = macd['MACDs_12_26_9']
            temp_df['RSI_slope'] = temp_df['RSI_10'].pct_change().ffill()
            temp_df['SuperTrend'], temp_df['SuperTrend_dir'] = supertrend(temp_df)
        elif strategy.__name__ in ['momentum', 'reversal', 'scalping', 'pullback', 'bull_flag']:
            temp_df['ADX'] = ta.adx(temp_df['high'], temp_df['low'], temp_df['close'], length=14)['ADX_14'].ffill()
            temp_df['RSI_10'] = ta.rsi(temp_df['close'], length=10).ffill()
            if strategy.__name__ == 'momentum':
                temp_df = compute_alpha_trend(temp_df)
        
        _, trades = strategy(temp_df)
        buy_times = [trade['entry_time'] for trade in trades if trade['type'] == 'buy']
        sell_times = [trade['entry_time'] for trade in trades if trade['type'] == 'sell']
        buy_signals[f'strategy_{idx}'] = df.index.isin(buy_times)
        sell_signals[f'strategy_{idx}'] = df.index.isin(sell_times)
    
    df['buy_vote'] = buy_signals.sum(axis=1)
    df['sell_vote'] = sell_signals.sum(axis=1)
    df['buy_signal'] = df['buy_vote'] >= 5
    df['sell_signal'] = df['sell_vote'] >= 5
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    trades = []
    position = None
    for i in range(20, len(df)):
        if position is None:
            if df['buy_signal'].iloc[i]:
                entry = df['close'].iloc[i]
                sl = entry - 0.7 * df['ATR'].iloc[i]
                tp = entry + 1.8 * df['ATR'].iloc[i]
                position = {'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
            elif df['sell_signal'].iloc[i]:
                entry = df['close'].iloc[i]
                sl = entry + 0.7 * df['ATR'].iloc[i]
                tp = entry - 1.8 * df['ATR'].iloc[i]
                position = {'type': 'sell', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i]}
        else:
            current_price = df['close'].iloc[i]
            if (position['type'] == 'buy' and current_price <= position['sl']) or (position['type'] == 'sell' and current_price >= position['sl']):
                position['exit'] = position['sl']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
            elif (position['type'] == 'buy' and current_price >= position['tp']) or (position['type'] == 'sell' and current_price <= position['tp']):
                position['exit'] = position['tp']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
    
    total_pnl = sum((trade['exit'] - trade['entry']) if trade['type'] == 'buy' else (trade['entry'] - trade['exit']) for trade in trades)
    return total_pnl, trades

# Strategy 17: CS AlphaTrend (Original)
def cs_alphatrend_original(df):
    return adjusted_alphatrend(df)  # Same logic as Adjusted AlphaTrend

# Strategy 18: CS AlphaTrend (Modified)
def cs_alphatrend_modified(df):
    df = df.copy()
    df['RSI_10'] = ta.rsi(df['close'], length=10).ffill()
    df['ATR_10'] = ta.atr(df['high'], df['low'], df['close'], length=10).ffill()
    df['EMA3'] = ta.ema(df['close'], length=3)
    df['EMA6'] = ta.ema(df['close'], length=6)
    macd = ta.macd(df['close']).ffill()
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['RSI_slope'] = df['RSI_10'].pct_change().ffill()
    df['SuperTrend'], df['SuperTrend_dir'] = supertrend(df)
    
    df['crossover'] = (df['close'] > df['close'].shift(2)) & (df['close'].shift(1) <= df['close'].shift(2))
    df['momentum'] = abs(df['close'] - df['close'].shift(2)) > 1
    df['rsi_buy'] = (df['RSI_10'] > 50) & (df['RSI_10'] < 70)
    df['rsi_sell'] = (df['RSI_10'] > 30) & (df['RSI_10'] < 50)
    df['supertrend_buy'] = df['SuperTrend_dir'] == 1
    df['supertrend_sell'] = df['SuperTrend_dir'] == -1
    df['rsi_slope_pos'] = df['RSI_slope'] > 0
    df['rsi_slope_neg'] = df['RSI_slope'] < 0
    df['macd_buy'] = df['MACD'] > df['MACD_signal']
    df['macd_sell'] = df['MACD'] < df['MACD_signal']
    df['ema_crossover'] = (df['EMA3'] > df['EMA6']) & (df['EMA3'].shift(1) <= df['EMA6'].shift(1))
    df['ema_crossunder'] = (df['EMA3'] < df['EMA6']) & (df['EMA3'].shift(1) >= df['EMA6'].shift(1))
    
    df['buy_score'] = (
        df['crossover'].astype(int) + df['momentum'].astype(int) + df['rsi_buy'].astype(int) +
        df['supertrend_buy'].astype(int) + df['rsi_slope_pos'].astype(int) + df['macd_buy'].astype(int) +
        df['ema_crossover'].astype(int)
    )
    df['sell_score'] = (
        (~df['crossover']).astype(int) + (~df['momentum']).astype(int) + df['rsi_sell'].astype(int) +
        df['supertrend_sell'].astype(int) + df['rsi_slope_neg'].astype(int) + df['macd_sell'].astype(int) +
        df['ema_crossunder'].astype(int)
    )
    
    df['buy_signal'] = df['buy_score'] >= 4
    df['sell_signal'] = df['sell_score'] >= 4
    df['exit_signal'] = (df['buy_score'] >= 3.5) | (df['sell_score'] >= 3.5)
    
    trades = []
    position = None
    for i in range(20, len(df)):
        current_time = df.index[i].time()
        is_morning = current_time < time(9, 30)
        if position is None:
            if df['buy_signal'].iloc[i]:
                entry = df['close'].iloc[i]
                sl = entry - 1 * df['ATR_10'].iloc[i]
                tp = entry + (2.5 if is_morning else 2) * df['ATR_10'].iloc[i]
                position = {'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i], 'trailing_sl': sl}
            elif df['sell_signal'].iloc[i]:
                entry = df['close'].iloc[i]
                sl = entry + 1 * df['ATR_10'].iloc[i]
                tp = entry - (2.5 if is_morning else 2) * df['ATR_10'].iloc[i]
                position = {'type': 'sell', 'entry': entry, 'sl': sl, 'tp': tp, 'entry_time': df.index[i], 'trailing_sl': sl}
        else:
            current_price = df['close'].iloc[i]
            rsi = df['RSI_10'].iloc[i]
            atr = df['ATR_10'].iloc[i]
            if not is_morning or (position['type'] == 'buy' and rsi > 75) or (position['type'] == 'sell' and rsi < 25):
                position['trailing_sl'] = (
                    max(position['trailing_sl'], current_price - 1 * atr) if position['type'] == 'buy' else
                    min(position['trailing_sl'], current_price + 1 * atr)
                )
            
            if (position['type'] == 'buy' and (current_price <= position['trailing_sl'] or df['exit_signal'].iloc[i])) or \
               (position['type'] == 'sell' and (current_price >= position['trailing_sl'] or df['exit_signal'].iloc[i])):
                position['exit'] = position['trailing_sl'] if not df['exit_signal'].iloc[i] else current_price
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
            elif (position['type'] == 'buy' and current_price >= position['tp']) or (position['type'] == 'sell' and current_price <= position['tp']):
                position['exit'] = position['tp']
                position['exit_time'] = df.index[i]
                trades.append(position)
                position = None
    
    total_pnl = sum((trade['exit'] - trade['entry']) if trade['type'] == 'buy' else (trade['entry'] - trade['exit']) for trade in trades)
    return total_pnl, trades

# Example usage
if __name__ == "__main__":
    # Sample data loading (replace with actual data)
    # df = pd.read_csv("nifty50_futures_5min.csv", parse_dates=['datetime'], index_col='datetime')
    # df.columns = ['open', 'high', 'low', 'close']
    
    # Run all strategies
    strategies = [
        original_mean_reversion, refined_mean_reversion, momentum_breakout, sma_crossover,
        bollinger_scalping, adjusted_alphatrend, momentum, reversal, breakout, scalping,
        ma_crossover, pivot_point, pullback, gap_and_go, bull_flag, ensemble_voting,
        cs_alphatrend_original, cs_alphatrend_modified
    ]
    
    # for strategy in strategies:
    #     total_pnl, trades = strategy(df)
    #     print(f"{strategy.__name__}: Total PNL = {total_pnl}, Trades = {len(trades)}")