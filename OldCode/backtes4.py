import pandas as pd
import pandas_ta as ta
from datetime import time

# 1) Load & filter your real 5-min file for 2025-04-29
df = pd.read_csv('results/nifty_historical_data_5min.csv')
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime']).set_index('datetime')
df = df[df.index.date == pd.to_datetime('2025-04-29').date()]

# 2) Compute Indicators
df['RSI_10']   = ta.rsi(df['close'], length=10).fillna(0)
df['ATR_10']   = ta.atr(df['high'], df['low'], df['close'], length=10).fillna(0)
for L in [7,14,21,50]:
    df[f'EMA{L}'] = ta.ema(df['close'], length=L).fillna(0)
macd = ta.macd(df['close']).fillna(0)
df['MACD']        = macd['MACD_12_26_9']
df['MACD_signal'] = macd['MACDs_12_26_9']
df['RSI_slope']   = df['RSI_10'].pct_change().fillna(0)

# 3) SuperTrend helper
def supertrend(df, period=10, mult=3):
    atr = df['ATR_10']
    hl2 = (df['high'] + df['low'])/2
    ub  = hl2 + mult*atr
    lb  = hl2 - mult*atr
    st  = pd.Series(index=df.index, dtype=float)
    dir = pd.Series(index=df.index, dtype=int)
    for i in range(len(df)):
        if i == 0:
            st.iloc[i], dir.iloc[i] = lb.iloc[i], 1
        else:
            if df['close'].iloc[i-1] > st.iloc[i-1]:
                st.iloc[i], dir.iloc[i] = lb.iloc[i], 1
            else:
                st.iloc[i], dir.iloc[i] = ub.iloc[i], -1
            # prevent back-step
            if dir.iloc[i]==1 and st.iloc[i]<st.iloc[i-1]:
                st.iloc[i]=st.iloc[i-1]
            if dir.iloc[i]==-1 and st.iloc[i]>st.iloc[i-1]:
                st.iloc[i]=st.iloc[i-1]
    return st, dir

df['SuperTrend'], df['SuperTrend_dir'] = supertrend(df)

# 4) AlphaTrend (seed = prev close)
df['AlphaTrend'] = (
      0.65 * df['close']
    + 0.25 * df['close'].shift(1).fillna(method='ffill')
    + 0.1  * df['RSI_10'] * (df['ATR_10']/df['close'].shift(1))
)
df['AlphaTrend'] = df['AlphaTrend'].ffill().fillna(df['close'])

# 5) Full backtest with morning + scoring + logging
def backtest_cs_alpha(df):
    trades       = []
    position     = None
    sl_price     = None
    morning_done = False

    for i in range(len(df)):
        row = df.iloc[i]
        t   = row.name.time()

        # --- Morning rule on first bar (i==0) ---
        if not morning_done and position is None and i == 0:
            side     = 'buy'   # or detect your gap-up/gap-down here
            entry    = row['open']
            first_rng= df.iloc[0]['high'] - df.iloc[0]['low']
            sl_price = entry - 0.5 * first_rng
            print(f"ENTER {t} | {side.upper():<4} ENTRY={entry:.2f} | SL={sl_price:.2f} (½ of 1st bar range)")
            position     = {'side': side, 'entry': entry, 'time': row.name}
            morning_done = True
            continue

        # --- Mid-day scoring entries after 1st bar ---
        if morning_done and position is None and i >= 2:
            prev2 = df.iloc[i-2]
            cross_buy  = int(row['close'] > prev2['close'])
            cross_sell = int(row['close'] < prev2['close'])
            abs_diff   = int(abs(row['close']-prev2['close']) > 1)
            rsi_buy    = int(row['RSI_10'] > 50)
            rsi_sell   = int(row['RSI_10'] < 50)
            macd_buy   = int(row['MACD'] > row['MACD_signal'])
            macd_sell  = int(row['MACD'] < row['MACD_signal'])
            st_buy     = int(row['SuperTrend_dir'] == 1)
            st_sell    = int(row['SuperTrend_dir'] == -1)
            slope_buy  = 0.5 if row['RSI_slope'] > 0 else 0
            slope_sell = 0.5 if row['RSI_slope'] < 0 else 0

            buy_score  = cross_buy + abs_diff + rsi_buy + macd_buy + st_buy + slope_buy
            sell_score = cross_sell + abs_diff + rsi_sell + macd_sell + st_sell + slope_sell

            if buy_score  >= 3.5:
                entry = row['open']
                print(f"ENTER {t} | BUY  ENTRY={entry:.2f} | SCORE={buy_score:.1f}")
                position = {'side':'buy', 'entry':entry, 'time':row.name}
                continue
            if sell_score >= 3.5:
                entry = row['open']
                print(f"ENTER {t} | SELL ENTRY={entry:.2f} | SCORE={sell_score:.1f}")
                position = {'side':'sell','entry':entry,'time':row.name}
                continue

        # --- Exit logic ---
        if position:
            side  = position['side']
            entry = position['entry']
            etime = position['time']

            # 1) Morning tight TSL
            if etime == df.index[0]:
                exit_price = sl_price
                profit     = (exit_price - entry) if side=='buy' else (entry - exit_price)
                print(f"EXIT  {t} | PROFIT={profit:.2f} | EXIT={exit_price:.2f} | TSL (morning)")
                trades.append((etime, side, entry, row.name, exit_price, profit))
                position = None
                continue

            # 2) Fixed-slot exits
            if t in [
                time(9,30), time(9,50), time(10,00), time(10,45),
                time(11,45), time(12,30), time(13,20), time(14,15)
            ]:
                exit_price = row['close']
                profit     = (exit_price - entry) if side=='buy' else (entry - exit_price)
                print(f"EXIT  {t} | PROFIT={profit:.2f} | EXIT={exit_price:.2f} | Slot exit")
                trades.append((etime, side, entry, row.name, exit_price, profit))
                position = None
                continue

    return pd.DataFrame(trades, columns=[
        'Entry_Time','Side','Entry','Exit_Time','Exit','Profit'
    ])

# 6) Run & show
print("\n>>> CS_Alpha trades for 2025-04-29:")
df_trades = backtest_cs_alpha(df)
print(df_trades.to_string(index=False))
