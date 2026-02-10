import pandas as pd
import pandas_ta as ta
from datetime import time, datetime

# 1) Hardcode data (provided 66 rows for 2025-04-29)
data = """datetime,open,high,low,close,volume
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
2025-04-29 15:25:00+05:30,24331.55,24340.85,24317.6,24325.45,0"""
df = pd.read_csv(pd.io.common.StringIO(data))
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime']).set_index('datetime')

# Commented out file loading
# df = pd.read_csv('results/29thData.csv')
# df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
# df = df.dropna(subset=['datetime']).set_index('datetime')

# 2) Compute Indicators
df['RSI_10'] = ta.rsi(df['close'], length=10).fillna(0)
df['ATR_10'] = ta.atr(df['high'], df['low'], df['close'], length=10).fillna(0)
for L in [7, 14, 21, 50]:
    df[f'EMA{L}'] = ta.ema(df['close'], length=L).fillna(0)
macd = ta.macd(df['close']).fillna(0)
df['MACD'] = macd['MACD_12_26_9']
df['MACD_signal'] = macd['MACDs_12_26_9']
df['RSI_slope'] = df['RSI_10'].pct_change().fillna(0)

# 3) SuperTrend helper
def supertrend(df, period=10, mult=3):
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
            if df['close'].iloc[i - 1] > st.iloc[i - 1]:
                st.iloc[i], dir.iloc[i] = lb.iloc[i], 1
            else:
                st.iloc[i], dir.iloc[i] = ub.iloc[i], -1
            # prevent back-step
            if dir.iloc[i] == 1 and st.iloc[i] < st.iloc[i - 1]:
                st.iloc[i] = st.iloc[i - 1]
            if dir.iloc[i] == -1 and st.iloc[i] > st.iloc[i - 1]:
                st.iloc[i] = st.iloc[i - 1]
    return st, dir

df['SuperTrend'], df['SuperTrend_dir'] = supertrend(df)

# 4) AlphaTrend (seed = prev close)
df['AlphaTrend'] = (
    0.65 * df['close']
    + 0.25 * df['close'].shift(1).ffill()
    + 0.1 * df['RSI_10'] * (df['ATR_10'] / df['close'].shift(1))
)
df['AlphaTrend'] = df['AlphaTrend'].ffill().fillna(df['close'])

# 5) Full backtest with morning + scoring + logging
def backtest_cs_alpha(df):
    trades = []
    log = []
    position = None
    sl_price = None
    target_price = None
    morning_done = False
    trade_count = 0
    current_date = None

    for i in range(len(df)):
        row = df.iloc[i]
        t = row.name.time()
        date = row.name.date()

        # Reset morning_done for new day
        if current_date != date:
            morning_done = False
            current_date = date

        # Log ROW for every bar
        buy_score = 0.0
        sell_score = 0.0
        if i >= 2:
            prev2 = df.iloc[i - 2]
            cross_buy = int(row['close'] > prev2['close'])
            cross_sell = int(row['close'] < prev2['close'])
            abs_diff = int(abs(row['close'] - prev2['close']) > 1)
            rsi_buy = int(row['RSI_10'] > 50)
            rsi_sell = int(row['RSI_10'] < 50)
            macd_buy = int(row['MACD'] > row['MACD_signal'])
            macd_sell = int(row['MACD'] < row['MACD_signal'])
            st_buy = int(row['SuperTrend_dir'] == 1)
            st_sell = int(row['SuperTrend_dir'] == -1)
            slope_buy = 0.5 if row['RSI_slope'] > 0 else 0
            slope_sell = 0.5 if row['RSI_slope'] < 0 else 0
            buy_score = cross_buy + abs_diff + rsi_buy + macd_buy + st_buy + slope_buy
            sell_score = cross_sell + abs_diff + rsi_sell + macd_sell + st_sell + slope_sell
        log.append(f"{t} ROW    O={row['open']:.2f} C={row['close']:.2f} BUY={buy_score:.1f} SELL={sell_score:.1f} ATR={row['ATR_10']:.2f}")

        # --- Morning rule on first bar of day ---
        if not morning_done and position is None and t == time(9, 15):
            side = 'buy'
            entry = row['open']
            sl_price = entry - 10  # SL: Entry - 10 points
            target_price = entry + 120  # TP: Entry + 120
            trade_count += 1
            trade_id = f'Long_{t.hour:02d}{t.minute:02d}'
            position = {
                'side': side,
                'entry': entry,
                'time': row.name,
                'sl': sl_price,
                'target': target_price,
                'id': trade_id,
                'is_morning': True
            }
            morning_done = True
            log.append(f"{t} ENTER   BUY  | Entry={entry:.2f} STOP={sl_price:.2f} TARGET={target_price:.2f} | Trade {trade_id} | Morning Trade")
            continue

        # --- Mid-day scoring entries after 1st bar ---
        if morning_done and position is None and i >= 2:
            if buy_score >= 3.5:
                entry = row['open']
                sl_price = entry - 2 * row['ATR_10']
                target_price = entry + 80
                trade_count += 1
                trade_id = f'Long_{t.hour:02d}{t.minute:02d}'
                position = {
                    'side': 'buy',
                    'entry': entry,
                    'time': row.name,
                    'sl': sl_price,
                    'target': target_price,
                    'id': trade_id,
                    'is_morning': False
                }
                log.append(f"{t} ENTER   BUY  | Entry={entry:.2f} STOP={sl_price:.2f} TARGET={target_price:.2f} | Trade {trade_id} | Score={buy_score:.1f} | Market=volatile")
                continue
            if sell_score >= 3.5:
                entry = row['open']
                sl_price = entry + 2 * row['ATR_10']
                target_price = entry - 80
                trade_count += 1
                trade_id = f'Short_{t.hour:02d}{t.minute:02d}'
                position = {
                    'side': 'sell',
                    'entry': entry,
                    'time': row.name,
                    'sl': sl_price,
                    'target': target_price,
                    'id': trade_id,
                    'is_morning': False
                }
                log.append(f"{t} ENTER   SELL | Entry={entry:.2f} STOP={sl_price:.2f} TARGET={target_price:.2f} | Trade {trade_id} | Score={sell_score:.1f} | Market=volatile")
                continue

        # --- Exit and TSL logic ---
        if position:
            side = position['side']
            entry = position['entry']
            etime = position['time']
            trade_id = position['id']
            is_morning = position['is_morning']

            # Update TSL for morning trade
            if is_morning and i > 0:
                prev_row = df.iloc[i - 1]
                sl_price = row['open'] + (prev_row['close'] - prev_row['open']) / 2
                position['sl'] = sl_price
            else:
                sl_price = position['sl']

            target_price = position['target']
            log.append(f"{t} RUNNING {side.upper()} | Entry={entry:.2f} C={row['close']:.2f} STOP={sl_price:.3f} TARGET={target_price:.2f} | Trade {trade_id} | Market=volatile")

            # Check SL exit (at open)
            if side == 'buy' and row['open'] <= sl_price:
                exit_price = row['open']
                profit = exit_price - entry
                log.append(f"{t} EXIT    BUY  | EXIT={exit_price:.2f} | P&L={profit:.2f} STOP-HIT | Trade {trade_id}")
                trades.append((trade_id, etime, side, entry, row.name, exit_price, profit))
                position = None
                continue
            elif side == 'sell' and row['open'] >= sl_price:
                exit_price = row['open']
                profit = entry - exit_price
                log.append(f"{t} EXIT    SELL | EXIT={exit_price:.2f} | P&L={profit:.2f} STOP-HIT | Trade {trade_id}")
                trades.append((trade_id, etime, side, entry, row.name, exit_price, profit))
                position = None
                continue

            # Check TP exit (at close)
            if side == 'buy' and row['close'] >= target_price:
                exit_price = row['close']
                profit = exit_price - entry
                log.append(f"{t} EXIT    BUY  | EXIT={exit_price:.2f} | P&L={profit:.2f} TARGET-HIT | Trade {trade_id}")
                trades.append((trade_id, etime, side, entry, row.name, exit_price, profit))
                position = None
                continue
            elif side == 'sell' and row['close'] <= target_price:
                exit_price = row['close']
                profit = entry - exit_price
                log.append(f"{t} EXIT    SELL | EXIT={exit_price:.2f} | P&L={profit:.2f} TARGET-HIT | Trade {trade_id}")
                trades.append((trade_id, etime, side, entry, row.name, exit_price, profit))
                position = None
                continue

    # Write log to file
    with open('corrected_trading_log_2025-04-29.txt', 'w') as f:
        for line in log:
            f.write(line + '\n')
        total_pnl = sum(trade[6] for trade in trades)
        f.write(f"\nTOTAL P&L for the day: {total_pnl:.2f}\n")

    return pd.DataFrame(trades, columns=[
        'Trade_ID', 'Entry_Time', 'Side', 'Entry', 'Exit_Time', 'Exit', 'Profit'
    ])

# 6) Run & show
print("\n>>> CS_Alpha trades for 2025-04-29:")
df_trades = backtest_cs_alpha(df)
print(df_trades.to_string(index=False))

# Print log file contents
print("\n>>> Contents of corrected_trading_log_2025-04-29.txt:")
with open('corrected_trading_log_2025-04-29.txt', 'r') as f:
    print(f.read())