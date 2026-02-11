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

# AI-Enhanced Adaptive Intraday Trading Strategy
def ai_adaptive_strategy(df):
    df = df.copy()
    
    # Calculate Indicators
    df['EMA9'] = ta.ema(df['close'], length=9)
    df['EMA21'] = ta.ema(df['close'], length=21)
    df['RSI_10'] = ta.rsi(df['close'], length=10).ffill()
    df['RSI_slope'] = df['RSI_10'].diff().ffill()
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14).ffill()
    df['ATR_SMA5'] = df['ATR_14'].rolling(window=5).mean()
    df['ATR_10'] = ta.atr(df['high'], df['low'], df['close'], length=10).ffill()
    df['SuperTrend'], df['SuperTrend_dir'] = supertrend(df)
    bb = ta.bbands(df['close'], length=20, std=2)
    df['BB_mid'] = bb['BBM_20_2.0']
    df['BB_upper'] = bb['BBU_20_2.0']
    df['BB_lower'] = bb['BBL_20_2.0']
    df['ADX_14'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14'].ffill()
    
    # Market Regime Detection
    df['volatility_regime'] = 'moderate'
    df.loc[df['ATR_14'] < df['ATR_SMA5'], 'volatility_regime'] = 'low'
    df.loc[df['ATR_14'] > 1.5 * df['ATR_SMA5'], 'volatility_regime'] = 'high'
    
    df['market_regime'] = 'range'
    df.loc[(df['ADX_14'] > 20) | (abs(df['EMA9'] - df['EMA21']) > df['ATR_14']), 'market_regime'] = 'trending'
    df.loc[(df['ADX_14'] < 15) & (df['close'].between(df['BB_lower'], df['BB_upper'])), 'market_regime'] = 'range'
    df.loc[df['ATR_14'] > 1.5 * df['ATR_SMA5'], 'market_regime'] = 'volatile'
    
    # Entry Signals
    df['buy_signal'] = (
        (df['EMA9'] > df['EMA21']) & (df['EMA9'].shift(1) <= df['EMA21'].shift(1)) &
        (df['RSI_slope'] > 0) & (df['close'] > df['SuperTrend']) &
        (df['close'] > df['BB_mid']) & (df['ATR_14'] >= df['ATR_SMA5'])
    )
    df['sell_signal'] = (
        (df['EMA9'] < df['EMA21']) & (df['EMA9'].shift(1) >= df['EMA21'].shift(1)) &
        (df['RSI_slope'] < 0) & (df['close'] < df['SuperTrend']) &
        (df['close'] < df['BB_mid']) & (df['ATR_14'] >= df['ATR_SMA5'])
    )
    
    # Exit Signals (Opposite EMA Crossover)
    df['buy_exit'] = (df['EMA9'] < df['EMA21']) & (df['EMA9'].shift(1) >= df['EMA21'].shift(1))
    df['sell_exit'] = (df['EMA9'] > df['EMA21']) & (df['EMA9'].shift(1) <= df['EMA21'].shift(1))
    
    trades = []
    position = None
    trailing_active = False
    
    for i in range(21, len(df)):
        current_price = df['close'].iloc[i]
        atr = df['ATR_14'].iloc[i]
        volatility = df['volatility_regime'].iloc[i]
        market_regime = df['market_regime'].iloc[i]
        
        # Determine SL and TP multipliers based on volatility and market regime
        sl_multiplier = 0.5 if volatility == 'low' else (1.5 if volatility == 'high' else 1.0)
        if market_regime == 'range':
            tp_multiplier = 1.5
        elif market_regime == 'volatile':
            tp_multiplier = 2.5 if volatility == 'high' else 2.0
        else:  # trending
            tp_multiplier = 3.0 if volatility == 'high' else 2.0
        
        if position is None:
            if df['buy_signal'].iloc[i]:
                entry = current_price
                sl = entry - sl_multiplier * atr
                tp = entry + tp_multiplier * atr
                position = {
                    'type': 'buy', 'entry': entry, 'sl': sl, 'tp': tp,
                    'entry_time': df.index[i], 'trailing_sl': sl
                }
                trailing_active = False
            elif df['sell_signal'].iloc[i]:
                entry = current_price
                sl = entry + sl_multiplier * atr
                tp = entry - tp_multiplier * atr
                position = {
                    'type': 'sell', 'entry': entry, 'sl': sl, 'tp': tp,
                    'entry_time': df.index[i], 'trailing_sl': sl
                }
                trailing_active = False
        else:
            # Activate trailing SL after 1*ATR move
            if not trailing_active:
                if position['type'] == 'buy' and current_price >= position['entry'] + atr:
                    trailing_active = True
                    position['trailing_sl'] = current_price - 0.7 * atr
                elif position['type'] == 'sell' and current_price <= position['entry'] - atr:
                    trailing_active = True
                    position['trailing_sl'] = current_price + 0.7 * atr
            
            # Update trailing SL
            if trailing_active:
                if position['type'] == 'buy':
                    position['trailing_sl'] = max(position['trailing_sl'], current_price - 0.7 * atr)
                else:
                    position['trailing_sl'] = min(position['trailing_sl'], current_price + 0.7 * atr)
            
            # Check for exit conditions
            if position['type'] == 'buy':
                if current_price <= position['trailing_sl'] or current_price >= position['tp'] or df['buy_exit'].iloc[i]:
                    position['exit'] = (
                        position['trailing_sl'] if current_price <= position['trailing_sl'] else
                        position['tp'] if current_price >= position['tp'] else current_price
                    )
                    position['exit_time'] = df.index[i]
                    trades.append(position)
                    position = None
            else:  # sell
                if current_price >= position['trailing_sl'] or current_price <= position['tp'] or df['sell_exit'].iloc[i]:
                    position['exit'] = (
                        position['trailing_sl'] if current_price >= position['trailing_sl'] else
                        position['tp'] if current_price <= position['tp'] else current_price
                    )
                    position['exit_time'] = df.index[i]
                    trades.append(position)
                    position = None
    
    total_pnl = sum(
        (trade['exit'] - trade['entry']) if trade['type'] == 'buy' else
        (trade['entry'] - trade['exit']) for trade in trades
    )
    return total_pnl, trades

# Example usage
if __name__ == "__main__":
    # Sample data loading (replace with actual BankNifty/Nifty50 5-min data)
    # df = pd.read_csv("banknifty_5min.csv", parse_dates=['datetime'], index_col='datetime')
    # df.columns = ['open', 'high', 'low', 'close']
    
    # Run strategy
    # total_pnl, trades = ai_adaptive_strategy(df)
    # print(f"AI-Enhanced Adaptive Strategy: Total PNL = {total_pnl}, Trades = {len(trades)}")
    # for trade in trades:
    #     print(f"Trade: {trade['type'].upper()} | Entry: {trade['entry']} | Exit: {trade['exit']} | "
    #           f"PNL: {trade['exit'] - trade['entry'] if trade['type'] == 'buy' else trade['entry'] - trade['exit']} | "
    #           f"Time: {trade['entry_time']} to {trade['exit_time']}")