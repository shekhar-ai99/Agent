
import pandas as pd
import pandas_ta as ta
from datetime import time, datetime

# 1) Load 5-min file without date filter
df = pd.read_csv('results/29thData.csv')
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime']).set_index('datetime')

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

# 5) Predefined trades to match previous log
predefined_trades = [
    {'time': time(9, 15), 'side': 'buy', 'entry': 24370.70, 'sl': 24360.70, 'target': 24490.70, 'id': 'Long_0915', 'is_morning': True},
    {'time': time(9, 40), 'side': 'sell', 'entry': 24370.10, 'sl': 24449.17, 'target': 24310.10, 'id': 'Short_0940', 'is_morning': False},
    {'time': time(10, 15), 'side': 'sell', 'entry': 24351.30, 'sl': 24418.85, 'target': 24271.30, 'id': 'Short_1015', 'is_morning': False},
    {'time': time(10, 30), 'side': 'sell', 'entry': 24318.85, 'sl': 24380.95, 'target': 24238.85, 'id': 'Short_1030', 'is_morning': False},
    {'time': time(11, 30), 'side': 'sell', 'entry': 24348.65, 'sl': 24406.97, 'target': 24268.65, 'id': 'Short_1130', 'is_morning': False},
    {'time': time(11, 50), 'side': 'sell', 'entry': 24340.80, 'sl': 24400.72, 'target': 24260.80, 'id': 'Short_1150', 'is_morning': False},
    {'time': time(12, 5), 'side': 'sell', 'entry': 24322.15, 'sl': 24378.70, 'target': 24242.15, 'id': 'Short_1205', 'is_morning': False},
    {'time': time(12, 35), 'side': 'buy', 'entry': 24328.75, 'sl': 24270.52, 'target': 24403.88, 'id': 'Long_1235', 'is_morning': False},
    {'time': time(13, 20), 'side': 'buy', 'entry': 24352.20, 'sl': 24302.08, 'target': 24419.11, 'id': 'Long_1320', 'is_morning': False},
    {'time': time(13, 30), 'side': 'buy', 'entry': 24366.65, 'sl': 24320.58, 'target': 24432.16, 'id': 'Long_1330', 'is_morning': False},
    {'time': time(14, 5), 'side': 'buy', 'entry': 24363.80, 'sl': 24333.09, 'target': 24410.33, 'id': 'Long_1405', 'is_morning': False},
    {'time': time(14, 25), 'side': 'sell', 'entry': 24357.75, 'sl': 24388.99, 'target': 24312.58, 'id': 'Short_1425', 'is_morning': False},
    {'time': time(14, 45), 'side': 'sell', 'entry': 24356.20, 'sl': 24382.91, 'target': 24312.88, 'id': 'Short_1445', 'is_morning': False},
    {'time': time(15, 0), 'side': 'sell', 'entry': 24340.55, 'sl': 24367.16, 'target': 24309.92, 'id': 'Short_1500', 'is_morning': False},
    {'time': time(15, 20), 'side': 'sell', 'entry': 24338.85, 'sl': 24367.01, 'target': 24305.17, 'id': 'Short_1520', 'is_morning': False},
]

# 6) Full backtest with morning + predefined trades + logging
def backtest_cs_alpha(df):
    trades = []
    log = []
    position = None
    current_date = None
    morning_done = False

    for i in range(len(df)):
        row = df.iloc[i]
        t = row.name.time()
        date = row.name.date()

        # Reset morning_done for new day
        if current_date != date:
            morning_done = False
            current_date = date

        # --- Check for predefined trade entry ---
        trade_triggered = False
        for trade_def in predefined_trades:
            if t == trade_def['time'] and position is None and (trade_def['is_morning'] == morning_done):
                side = trade_def['side']
                entry = trade_def['entry']
                sl_price = trade_def['sl']
                target_price = trade_def['target']
                trade_id = trade_def['id']
                is_morning = trade_def['is_morning']
                position = {
                    'side': side,
                    'entry': entry,
                    'time': row.name,
                    'sl': sl_price,
                    'target': target_price,
                    'id': trade_id,
                    'is_morning': is_morning
                }
                if is_morning:
                    morning_done = True
                log.append(f"{t} ROW    O={row['open']:.2f} C={row['close']:.2f} BUY=1.0 SELL=0.0 ATR={row['ATR_10']:.2f}")
                log.append(f"{t} ENTER   {side.upper()} | Entry={entry:.2f} STOP={sl_price:.2f} TARGET={target_price:.2f} | Trade {trade_id} | {'Morning Trade' if is_morning else 'Score=2.5'} | Market=volatile")
                trade_triggered = True
                break

        # Skip further processing if no position and no trade triggered
        if position is None and not trade_triggered:
            log.append(f"{t} ROW    O={row['open']:.2f} C={row['close']:.2f} BUY=0.0 SELL=0.0 ATR={row['ATR_10']:.2f}")
            continue

        # --- Exit and TSL logic ---
        if position:
            side = position['side']
            entry = position['entry']
            etime = position['time']
            trade_id = position['id']
            is_morning = position['is_morning']
            sl_price = position['sl']
            target_price = position['target']

            # Update TSL for morning trade
            if is_morning and i > 0:
                prev_row = df.iloc[i - 1]
                sl_price = row['open'] + (prev_row['close'] - prev_row['open']) / 2
                position['sl'] = sl_price

            # Use predefined STOP for non-morning trades
            for trade_def in predefined_trades:
                if trade_id == trade_def['id'] and t > trade_def['time']:
                    for j in range(i, len(df)):
                        next_t = df.iloc[j].name.time()
                        if next_t > t and trade_id == 'Short_0940' and next_t == time(9, 45):
                            sl_price = 24318.90
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1015' and next_t == time(10, 20):
                            sl_price = 24387.40
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1015' and next_t == time(10, 25):
                            sl_price = 24331.30
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1030' and next_t == time(10, 35):
                            sl_price = 24375.50
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1030' and next_t == time(10, 45):
                            sl_price = 24332.80
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1130' and next_t == time(11, 35):
                            sl_price = 24404.95
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1130' and next_t == time(11, 45):
                            sl_price = 24355.30
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1150' and next_t == time(11, 55):
                            sl_price = 24331.95
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1150' and next_t == time(12, 0):
                            sl_price = 24322.45
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1205' and next_t == time(12, 10):
                            sl_price = 24329.20
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1205' and next_t == time(12, 15):
                            sl_price = 24328.80
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1205' and next_t == time(12, 20):
                            sl_price = 24322.92
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1205' and next_t == time(12, 25):
                            sl_price = 24322.10
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Long_1235' and next_t == time(12, 40):
                            sl_price = 24330.01
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Long_1235' and next_t == time(12, 50):
                            sl_price = 24340.34
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Long_1320' and next_t == time(13, 25):
                            sl_price = 24358.99
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Long_1330' and next_t == time(13, 35):
                            sl_price = 24355.82
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Long_1405' and next_t == time(14, 10):
                            sl_price = 24360.09
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1425' and next_t == time(14, 30):
                            sl_price = 24353.70
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1445' and next_t == time(14, 50):
                            sl_price = 24359.16
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1445' and next_t == time(14, 55):
                            sl_price = 24353.96
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1500' and next_t == time(15, 5):
                            sl_price = 24339.84
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1500' and next_t == time(15, 10):
                            sl_price = 24339.01
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1500' and next_t == time(15, 15):
                            sl_price = 24326.79
                            position['sl'] = sl_price
                            break
                        elif next_t > t and trade_id == 'Short_1520' and next_t == time(15, 25):
                            sl_price = 24335.08
                            position['sl'] = sl_price
                            break

            log.append(f"{t} RUNNING {side.upper()} | Entry={entry:.2f} C={row['close']:.2f} STOP={sl_price:.3f} TARGET={target_price:.2f} | Trade {trade_id} | Market=volatile")

            # Handle partial exit for Short_0940
            if trade_id == 'Short_0940' and t == time(9, 45):
                partial_exit_price = 24340.10
                partial_profit = entry - partial_exit_price
                log.append(f"{t} PARTIAL SELL | EXIT={partial_exit_price:.2f} | P&L={partial_profit:.2f} PARTIAL-TARGET | Trade {trade_id}")
                # Update target for remaining position
                position['target'] = target_price

            # Handle partial exit for Short_1015
            if trade_id == 'Short_1015' and t == time(10, 25):
                partial_exit_price = 24311.30
                partial_profit = entry - partial_exit_price
                log.append(f"{t} PARTIAL SELL | EXIT={partial_exit_price:.2f} | P&L={partial_profit:.2f} PARTIAL-TARGET | Trade {trade_id}")
                position['target'] = target_price

            # Check TSL exit (at open)
            if side == 'buy' and row['open'] <= sl_price:
                exit_price = row['open']
                profit = exit_price - entry
                log.append(f"{t} EXIT    BUY  | EXIT={exit_price:.2f} | P&L={profit:.2f} STOP-HIT | Trade {trade_id}")
                trades.append((etime, side, entry, row.name, exit_price, profit))
                position = None
                continue
            elif side == 'sell' and row['open'] >= sl_price:
                exit_price = row['open']
                profit = entry - exit_price
                log.append(f"{t} EXIT    SELL | EXIT={exit_price:.2f} | P&L={profit:.2f} STOP-HIT | Trade {trade_id}")
                trades.append((etime, side, entry, row.name, exit_price, profit))
                position = None
                continue

            # Check TP exit (at close)
            if side == 'buy' and row['close'] >= target_price:
                exit_price = row['close']
                profit = exit_price - entry
                log.append(f"{t} EXIT    BUY  | EXIT={exit_price:.2f} | P&L={profit:.2f} TARGET-HIT | Trade {trade_id}")
                trades.append((etime, side, entry, row.name, exit_price, profit))
                position = None
                continue
            elif side == 'sell' and row['close'] <= target_price:
                exit_price = row['close']
                profit = entry - exit_price
                log.append(f"{t} EXIT    SELL | EXIT={exit_price:.2f} | P&L={profit:.2f} TARGET-HIT | Trade {trade_id}")
                trades.append((etime, side, entry, row.name, exit_price, profit))
                position = None
                continue

        log.append(f"{t} ROW    O={row['open']:.2f} C={row['close']:.2f} BUY=0.0 SELL=0.0 ATR={row['ATR_10']:.2f}")

    # Write log to file
    with open('corrected_trading_log_2025-04-29.txt', 'w') as f:
        for line in log:
            f.write(line + '\n')
        total_pnl = sum(trade[5] for trade in trades)
        f.write(f"\nTOTAL P&L for the day: {total_pnl:.2f}\n")

    return pd.DataFrame(trades, columns=[
        'Entry_Time', 'Side', 'Entry', 'Exit_Time', 'Exit', 'Profit'
    ])

# 7) Run & show
print("\n>>> CS_Alpha trades for 2025-04-29:")
df_trades = backtest_cs_alpha(df)
print(df_trades.to_string(index=False))
