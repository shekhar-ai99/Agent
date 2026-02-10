import pandas as pd
import pandas_ta as ta
from datetime import datetime, time


import pandas as pd
import pandas_ta as ta
from datetime import datetime, time

# —————————————————————————————————————————————————————————————————————
# 1) Hardcode CSV data for 2025-04-29
data = '''datetime,open,high,low,close,volume
2025-04-29 09:15:00+05:30,24370.7,24442.25,24364.35,24439.25,0
2025-04-29 09:20:00+05:30,24438.5,24455.05,24424.5,24453.7,0
2025-04-29 09:25:00+05:30,24453.6,24457.65,24413.3,24417.1,0
2025-04-29 09:30:00+05:30,24418.25,24452.9,24393.55,24440.2,0
2025-04-29 09:35:00+05:30,24440.85,24442.5,24359.2,24369.45,0
2025-04-29 09:40:00+05:30,24370.1,24387.05,24303.9,24316.6,0
2025-04-29 09:45:00+05:30,24317.85,24364.5,24308.1,24315.55,0
2025-04-29 09:50:00+05:30,24316.75,24340.05,24290.75,24329.1,0
2025-04-29 09:55:00+05:30,24330.15,24395,24326.5,24390.25,0
2025-04-29 10:00:00+05:30,24390,24396.15,24362.2,24378.3,0
2025-04-29 10:05:00+05:30,24379.15,24379.75,24333.8,24344.65,0
2025-04-29 10:10:00+05:30,24345.2,24365.1,24338.55,24351.55,0
2025-04-29 10:15:00+05:30,24351.3,24354.1,24319,24319,0
2025-04-29 10:20:00+05:30,24318.1,24347.1,24317.4,24346.4,0
2025-04-29 10:25:00+05:30,24346.45,24348.15,24311.3,24317.8,0
2025-04-29 10:30:00+05:30,24318.85,24329.7,24305.5,24314.95,0
2025-04-29 10:35:00+05:30,24316.2,24335.25,24314.55,24331.5,0
2025-04-29 10:40:00+05:30,24331.55,24333.9,24312.8,24321.5,0
2025-04-29 10:45:00+05:30,24321.85,24352.35,24317.95,24350.7,0
2025-04-29 10:50:00+05:30,24349.85,24353.7,24328.45,24330.85,0
2025-04-29 10:55:00+05:30,24330.25,24349.95,24330.25,24333.8,0
2025-04-29 11:00:00+05:30,24334.7,24351.9,24332.8,24351.9,0
2025-04-29 11:05:00+05:30,24353.05,24369.85,24351.9,24367.1,0
2025-04-29 11:10:00+05:30,24367.25,24369.7,24350.6,24356.05,0
2025-04-29 11:15:00+05:30,24356.55,24358.5,24338.95,24348.8,0
2025-04-29 11:20:00+05:30,24348.3,24360.35,24347.05,24348.95,0
2025-04-29 11:25:00+05:30,24349.05,24358.75,24347.6,24349.25,0
2025-04-29 11:30:00+05:30,24348.65,24351.6,24334.95,24347.65,0
2025-04-29 11:35:00+05:30,24347.95,24353.6,24341.6,24346.45,0
2025-04-29 11:40:00+05:30,24346.7,24356.8,24342.2,24354.4,0
2025-04-29 11:45:00+05:30,24354.65,24355.8,24335.3,24340.15,0
2025-04-29 11:50:00+05:30,24340.8,24340.85,24321,24322.95,0
2025-04-29 11:55:00+05:30,24323.4,24329.1,24311.95,24313.25,0
2025-04-29 12:00:00+05:30,24313.4,24322.5,24302.45,24322.05,0
2025-04-29 12:05:00+05:30,24322.15,24323.75,24310.65,24315.5,0
2025-04-29 12:10:00+05:30,24315.3,24322.3,24309.2,24316.6,0
2025-04-29 12:15:00+05:30,24316.4,24321.6,24310.65,24313.1,0
2025-04-29 12:20:00+05:30,24315.1,24315.85,24304.1,24313.65,0
2025-04-29 12:25:00+05:30,24312.95,24317.25,24306.6,24309.75,0
2025-04-29 12:30:00+05:30,24309.25,24330.8,24309.15,24329.95,0
2025-04-29 12:35:00+05:30,24328.75,24347.95,24325.4,24341.35,0
2025-04-29 12:40:00+05:30,24340.6,24346.8,24336.4,24341.85,0
2025-04-29 12:45:00+05:30,24338.8,24344.85,24330.6,24339.55,0
2025-04-29 12:50:00+05:30,24339.15,24358.35,24336.45,24357.9,0
2025-04-29 12:55:00+05:30,24357.75,24363.55,24353,24358.95,0
2025-04-29 13:00:00+05:30,24358.25,24369.5,24352.2,24368.4,0
2025-04-29 13:05:00+05:30,24368.05,24368.5,24350.6,24360.1,0
2025-04-29 13:10:00+05:30,24359.4,24363.55,24353.5,24356.85,0
2025-04-29 13:15:00+05:30,24355.45,24359.35,24341.35,24354.1,0
2025-04-29 13:20:00+05:30,24352.2,24363.15,24346.55,24362.9,0
2025-04-29 13:25:00+05:30,24362.1,24376.05,24356,24366.05,0
2025-04-29 13:30:00+05:30,24366.65,24371.7,24361.45,24366.8,0
2025-04-29 13:35:00+05:30,24366.5,24368.8,24357.4,24366.15,0
2025-04-29 13:40:00+05:30,24367.15,24372.7,24351,24353.9,0
2025-04-29 13:45:00+05:30,24352.7,24366.95,24351.75,24363.45,0
2025-04-29 13:50:00+05:30,24364,24368.75,24355.1,24355.65,0
2025-04-29 13:55:00+05:30,24356.65,24363.4,24347.65,24358.25,0
2025-04-29 14:00:00+05:30,24360.15,24368.3,24356.85,24363.55,0
2025-04-29 14:05:00+05:30,24363.8,24375.5,24360.6,24365.2,0
2025-04-29 14:10:00+05:30,24365.35,24374.8,24360.25,24365.95,0
2025-04-29 14:15:00+05:30,24366.9,24368.3,24355.05,24361.65,0
2025-04-29 14:20:00+05:30,24361.9,24367.55,24356.25,24357.15,0
2025-04-29 14:25:00+05:30,24357.75,24359.7,24342.4,24346,0
2025-04-29 14:30:00+05:30,24346.45,24350.5,24339,24343.45,0
2025-04-29 14:35:00+05:30,24343.75,24361.95,24343.75,24348.1,0
2025-04-29 14:40:00+05:30,24348.8,24360.7,24347.2,24356.45,0
2025-04-29 14:45:00+05:30,24356.2,24357.95,24347.85,24352.95,0
2025-04-29 14:50:00+05:30,24353.35,24358,24344.85,24346.85,0
2025-04-29 14:55:00+05:30,24346.8,24355.7,24339.45,24340.2,0
2025-04-29 15:00:00+05:30,24340.55,24347.8,24325.2,24331.4,0
2025-04-29 15:05:00+05:30,24331.55,24336.5,24324.9,24332.7,0
2025-04-29 15:10:00+05:30,24332.6,24335.6,24324.45,24326.45,0
2025-04-29 15:15:00+05:30,24326.2,24341.45,24310.6,24339.65,0
2025-04-29 15:20:00+05:30,24338.85,24348.5,24325.85,24333.1,0
2025-04-29 15:25:00+05:30,24331.55,24340.85,24317.6,24325.45,0'''
# load into DataFrame
from io import StringIO
df = pd.read_csv(StringIO(data), parse_dates=['datetime'])
df.set_index('datetime', inplace=True)


# —————————————————————————————————————————————————————————————————————
# # 1) Load & filter your 5-min file for 2025-04-29
# df = pd.read_csv(
#     'results/nifty_historical_data_5min.csv',
#     parse_dates=['datetime'], index_col='datetime'
# )
# df = df[df.index.date == pd.to_datetime('2025-04-29').date()]

# —————————————————————————————————————————————————————————————————————
# 2) Compute indicators
df['RSI_10'] = ta.rsi(df['close'], length=10).fillna(0)
df['ATR_10'] = ta.atr(df['high'], df['low'], df['close'], length=10).fillna(method='ffill').fillna(df['close'].diff().abs())
macd = ta.macd(df['close']).fillna(0)
df['MACD']        = macd['MACD_12_26_9']
df['MACD_signal'] = macd['MACDs_12_26_9']
df['RSI_slope']   = df['RSI_10'].pct_change().fillna(0)

# SuperTrend
def supertrend(df, mult=3):
    atr = df['ATR_10']
    hl2 = (df['high'] + df['low'])/2
    ub  = hl2 + mult*atr
    lb  = hl2 - mult*atr
    st, direction = pd.Series(index=df.index), pd.Series(index=df.index)
    for i in range(len(df)):
        if i==0:
            st.iat[i], direction.iat[i] = lb.iat[i], 1
        else:
            prev_st = st.iat[i-1]
            if df['close'].iat[i-1] > prev_st:
                st.iat[i], direction.iat[i] = lb.iat[i], 1
            else:
                st.iat[i], direction.iat[i] = ub.iat[i], -1
            # prevent back-step
            if direction.iat[i]==1 and st.iat[i]<prev_st:
                st.iat[i] = prev_st
            if direction.iat[i]==-1 and st.iat[i]>prev_st:
                st.iat[i] = prev_st
    return st, direction

df['SuperTrend'], df['SuperTrend_dir'] = supertrend(df)

# AlphaTrend (seed = prev close)
df['AlphaTrend'] = (
      0.65 * df['close']
    + 0.25 * df['close'].shift(1).ffill()
    + 0.1  * df['RSI_10'] * (df['ATR_10']/df['close'].shift(1))
)
df['AlphaTrend'] = df['AlphaTrend'].ffill().fillna(df['close'])

# —————————————————————————————————————————————————————————————————————
# 3) Backtest with enhanced logic
def backtest_cs_alpha(df):
    trades, log = [], []
    position = None
    morning_done = False

    # Pre-compute first bar range for morning SL/TP
    first_rng = (df.iloc[0]['high'] - df.iloc[0]['low']) if len(df)>0 else 0

    # Volatility benchmark
    atr_mean = df['ATR_10'].mean()

    for i, (ts, row) in enumerate(df.iterrows()):
        t = ts.time()
        # scores only once we have at least 2 prior bars
        buy_score = sell_score = 0.0
        if i>=2:
            prev2 = df.iloc[i-2]
            buy_score = (
                int(row['close']>prev2['close'])
              + int(abs(row['close']-prev2['close'])>1)
              + int(row['RSI_10']>50)
              + int(row['MACD']>row['MACD_signal'])
              + int(row['SuperTrend_dir']==1)
              + (0.5 if row['RSI_slope']>0 else 0)
            )
            sell_score = (
                int(row['close']<prev2['close'])
              + int(abs(row['close']-prev2['close'])>1)
              + int(row['RSI_10']<50)
              + int(row['MACD']<row['MACD_signal'])
              + int(row['SuperTrend_dir']==-1)
              + (0.5 if row['RSI_slope']<0 else 0)
            )
        log.append(f"{t} ROW    O={row['open']:.2f} C={row['close']:.2f} "
                   f"BUY={buy_score:.1f} SELL={sell_score:.1f} ATR={row['ATR_10']:.2f}")

        # 1) If in a trade, update TSL & check exits
        if position:
            side   = position['side']
            entry  = position['entry']
            stop   = position['stop']
            target = position['target']

            pnl = (row['close']-entry) if side=='buy' else (entry-row['close'])
            # tighten stop by max(½·P&L, ATR)
            if pnl>0:
                trail_amt = max(0.5*pnl, row['ATR_10'])
                new_stop = entry + trail_amt if side=='buy' else entry - trail_amt
                stop = max(stop, new_stop) if side=='buy' else min(stop, new_stop)
            log.append(f"{t} RUNNING {side.upper():<4} | Entry={entry:.2f} C={row['close']:.2f} "
                       f"STOP={stop:.2f} TGT={target:.2f}")

            # 1a) target-hit at close
            if (side=='buy' and row['close']>=target) or (side=='sell' and row['close']<=target):
                exit_price = row['close']
                profit = (exit_price-entry) if side=='buy' else (entry-exit_price)
                log.append(f"{t} EXIT    {side.upper():<4} | EXIT={exit_price:.2f} "
                           f"P&L={profit:.2f} TARGET-HIT")
                trades.append((position['id'], position['time'], side, entry, ts, exit_price, profit))
                position = None
                continue

            # 1b) stop-hit at open
            if (side=='buy' and row['open']<=stop) or (side=='sell' and row['open']>=stop):
                exit_price = row['open']
                profit = (exit_price-entry) if side=='buy' else (entry-exit_price)
                log.append(f"{t} EXIT    {side.upper():<4} | EXIT={exit_price:.2f} "
                           f"P&L={profit:.2f} STOP-HIT")
                trades.append((position['id'], position['time'], side, entry, ts, exit_price, profit))
                position = None
                continue

            # save updated stop
            position['stop'] = stop
            continue

        # 2) Morning gap entry at 09:15
        if not morning_done and t==time(9,15):
            entry = row['open']
            side  = 'buy'  # always long morning
            sl    = entry - 0.5*first_rng
            tgt   = entry +     1.0*first_rng
            tid   = f"MORNING_{t.hour:02d}{t.minute:02d}"
            position = dict(side=side, entry=entry, stop=sl, target=tgt, id=tid, time=ts)
            log.append(f"{t} ENTER   {side.upper():<4} | ENT={entry:.2f} "
                       f"STOP={sl:.2f} TGT={tgt:.2f} | {tid}")
            morning_done = True
            continue

        # 3) Mid-day score entries (once morning done)
        if morning_done and not position and i>=2:
            # dynamic threshold
            thresh = 4.0 if row['ATR_10']>atr_mean else 3.0
            # momentum filter
            if buy_score>=thresh and row['close']>row['open']:
                entry = row['open']
                sl    = entry - 2*row['ATR_10']
                tgt   = entry +    80
                tid   = f"LONG_{t.hour:02d}{t.minute:02d}"
                position = dict(side='buy', entry=entry, stop=sl, target=tgt, id=tid, time=ts)
                log.append(f"{t} ENTER   BUY  | ENT={entry:.2f} STOP={sl:.2f} TGT={tgt:.2f} "
                           f"THR={thresh:.1f} SCORE={buy_score:.1f}")
                continue
            if sell_score>=thresh and row['close']<row['open']:
                entry = row['open']
                sl    = entry + 2*row['ATR_10']
                tgt   = entry -    80
                tid   = f"SHORT_{t.hour:02d}{t.minute:02d}"
                position = dict(side='sell', entry=entry, stop=sl, target=tgt, id=tid, time=ts)
                log.append(f"{t} ENTER   SELL | ENT={entry:.2f} STOP={sl:.2f} TGT={tgt:.2f} "
                           f"THR={thresh:.1f} SCORE={sell_score:.1f}")
                continue

    # write full bar-by-bar log to file
    with open('cs_alpha_log_2025-04-29.txt','w') as f:
        for line in log:
            f.write(line+"\n")

    # build trades DataFrame
    df_trades = pd.DataFrame(trades, columns=[
        'Trade_ID','Entry_Time','Side','Entry','Exit_Time','Exit','Profit'
    ])

    # console summary
    print("\n=== BAR-BY-BAR LOG WRITTEN TO cs_alpha_log_2025-04-29.txt ===\n")
    if not df_trades.empty:
        print("=== EXECUTED TRADES ===")
        print(df_trades[['Trade_ID','Side','Entry','Exit','Profit']].to_string(index=False))
        print(f"\nTOTAL P&L: {df_trades['Profit'].sum():.2f}")
    else:
        print("No trades executed.")

    return df_trades

# —————————————————————————————————————————————————————————————————————
# 4) Run the backtest
if __name__ == "__main__":
    print("\n>>> CS_Alpha backtest for 2025-04-29:")
    backtest_cs_alpha(df)
