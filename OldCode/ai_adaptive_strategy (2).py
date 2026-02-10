import pandas as pd
import pandas_ta as ta
from datetime import time
import uuid

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
    try:
        df['EMA9'] = ta.ema(df['close'], length=9)
        df['EMA21'] = ta.ema(df['close'], length=21)
        df['RSI_10'] = ta.rsi(df['close'], length=10).ffill()
        df['RSI_slope'] = df['RSI_10'].diff().ffill()
        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14).ffill()
        df['ATR_SMA5'] = df['ATR_14'].rolling(window=5).mean()
        df['ATR_10'] = ta.atr(df['high'], df['low'], df['close'], length=10).ffill()
        df['SuperTrend'], df['SuperTrend_dir'] = supertrend(df)
        
        # Calculate Bollinger Bands and dynamically detect column names
        bb = ta.bbands(df['close'], length=20, std=2)
        bb_columns = bb.columns.tolist()
        print(f"Bollinger Bands columns: {bb_columns}")
        
        bb_mid_col = next((col for col in bb_columns if 'BBM' in col), None)
        bb_upper_col = next((col for col in bb_columns if 'BBU' in col), None)
        bb_lower_col = next((col for col in bb_columns if 'BBL' in col), None)
        
        if not all([bb_mid_col, bb_upper_col, bb_lower_col]):
            raise ValueError(f"Bollinger Bands columns not found in {bb_columns}")
        
        df['BB_mid'] = bb[bb_mid_col]
        df['BB_upper'] = bb[bb_upper_col]
        df['BB_lower'] = bb[bb_lower_col]
        
        df['ADX_14'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14'].ffill()
        
    except Exception as e:
        raise ValueError(f"Error calculating indicators: {str(e)}")
    
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
    
    # Filter for market hours (9:15 AM to 3:30 PM IST)
    market_open = time(9, 15)
    market_close = time(15, 30)
    
    # Analyze April 29 data
    april_29 = df[df.index.date == pd.to_datetime('2025-04-29').date()]
    if not april_29.empty:
        print(f"\nApril 29 Data Summary:")
        print(f"Price Range: {april_29['close'].min():.2f} - {april_29['close'].max():.2f}")
        print(f"Rows: {len(april_29)}")
    else:
        print("\nNo data for April 29 found in the CSV.")
    
    for i in range(21, len(df)):
        current_time = df.index[i].time()
        if not (market_open <= current_time <= market_close):
            continue  # Skip non-trading hours
        
        current_price = df['close'].iloc[i]
        atr = df['ATR_14'].iloc[i]
        volatility = df['volatility_regime'].iloc[i]
        market_regime = df['market_regime'].iloc[i]
        
        # Debug buy signal conditions on April 29
        if df.index[i].date() == pd.to_datetime('2025-04-29').date() and current_price >= 24340:
            print(f"\nChecking Buy Signal at {df.index[i]} (Price: {current_price:.2f}):")
            print(f"EMA9 ({df['EMA9'].iloc[i]:.2f}) > EMA21 ({df['EMA21'].iloc[i]:.2f}): {df['EMA9'].iloc[i] > df['EMA21'].iloc[i]}")
            print(f"EMA9 Crossover: {df['EMA9'].iloc[i] > df['EMA21'].iloc[i] and df['EMA9'].shift(1).iloc[i] <= df['EMA21'].shift(1).iloc[i]}")
            print(f"RSI_slope ({df['RSI_slope'].iloc[i]:.2f}) > 0: {df['RSI_slope'].iloc[i] > 0}")
            print(f"Close ({current_price:.2f}) > SuperTrend ({df['SuperTrend'].iloc[i]:.2f}): {current_price > df['SuperTrend'].iloc[i]}")
            print(f"Close ({current_price:.2f}) > BB_mid ({df['BB_mid'].iloc[i]:.2f}): {current_price > df['BB_mid'].iloc[i]}")
            print(f"ATR_14 ({df['ATR_14'].iloc[i]:.2f}) >= ATR_SMA5 ({df['ATR_SMA5'].iloc[i]:.2f}): {df['ATR_14'].iloc[i] >= df['ATR_SMA5'].iloc[i]}")
        
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
# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    try:
        df = pd.read_csv("results/nifty_historical_data_5min.csv", parse_dates=['datetime'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('Asia/Kolkata')
        df.set_index('datetime', inplace=True)
        df = df[['open', 'high', 'low', 'close']].copy()
        df = df.dropna().ffill()  # Handle missing values
        print(f"Data loaded successfully. Rows: {len(df)}, Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
    
    # Run strategy
    try:
        total_pnl, trades = ai_adaptive_strategy(df)
        print(f"\nAI-Enhanced Adaptive Strategy Results:")
        print(f"Total PNL: {total_pnl:.2f} points")
        print(f"Number of Trades: {len(trades)}")
        print("\nTrade Log:")
        for i, trade in enumerate(trades, 1):
            pnl = (trade['exit'] - trade['entry']) if trade['type'] == 'buy' else (trade['entry'] - trade['exit'])
            print(f"Trade {i}: {trade['type'].upper()} | "
                  f"Entry: {trade['entry']:.2f} at {trade['entry_time']} | "
                  f"Exit: {trade['exit']:.2f} at {trade['exit_time']} | "
                  f"PNL: {pnl:.2f} points")
    except Exception as e:
        print(f"Error running strategy: {e}")