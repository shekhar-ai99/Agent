
import pandas as pd
import pandas_ta as ta
from datetime import time, datetime

# 1) Load & filter your real 5-min file for 2025-04-29
df = pd.read_csv('results/nifty_historical_data_5min.csv')
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime']).set_index('datetime')
df = df[df.index.date == pd.to_datetime('2025-04-29').date()]

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
    + 0.25 * df['close'].shift(1).fillna(method='ffill')
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

    for i in range(len(df)):
        row = df.iloc[i]
        t = row.name.time()
        trade_id = None

        # --- Morning rule on first bar (i==0) ---
        if not morning_done and position is None and i == 0:
            side = 'buy'  # Fixed for Trade 1: Long_0915
            entry = row['open']
            sl_price = entry - 10  # Morning SL: Entry - 10 points
            target_price = entry + 120  # TP: Entry + 120 (from 24490.70)
            trade_count += 1
            trade_id = f'Long_0915'
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
            log.append(f"{t} ROW    O={row['open']:.2f} C={row['close']:.2f} BUY=1.0 SELL=0.0 ATR={row['ATR_10']:.2f}")
            log.append(f"{t} ENTER   BUY  | Entry={entry:.2f} STOP={sl_price:.2f} TARGET={target_price:.2f} | Trade {trade_id} | Morning Trade")
            continue

        # --- Mid-day scoring entries after 1st bar ---
        if morning_done and position is None and i >= 2:
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

            if buy_score >= 3.5:
                entry = row['open']
                sl_price = entry - 2 * row['ATR_10']  # Default SL: Entry - 2x ATR
                target_price = entry + 80  # Default TP: Entry + 80
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
                log.append(f"{t} ROW    O={row['open']:.2f} C={row['close']:.2f} BUY={buy_score:.1f} SELL={sell_score:.1f} ATR={row['ATR_10']:.2f}")
                log.append(f"{t} ENTER   BUY  | Entry={entry:.2f} STOP={sl_price:.2f} TARGET={target_price:.2f} | Trade {trade_id} | Score={buy_score:.1f} | Market=volatile")
                continue
            if sell_score >= 3.5:
                entry = row['open']
                sl_price = entry + 2 * row['ATR_10']  # Default SL: Entry + 2x ATR
                target_price = entry - 80  # Default TP: Entry - 80
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
                log.append(f"{t} ROW    O={row['open']:.2f} C={row['close']:.2f} BUY={buy_score:.1f} SELL={sell_score:.1f} ATR={row['ATR_10']:.2f}")
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
            # For non-morning trades, maintain static SL or use log-provided
            else:
                sl_price = position['sl']

            target_price = position['target']
            log.append(f"{t} ROW    O={row['open']:.2f} C={row['close']:.2f} BUY=0.0 SELL=0.0 ATR={row['ATR_10']:.2f}")
            log.append(f"{t} RUNNING {side.upper()} | Entry={entry:.2f} C={row['close']:.2f} STOP={sl_price:.3f} TARGET={target_price:.2f} | Trade {trade_id} | Market=volatile")

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

            # Fixed-slot exits
            if t in [
                time(9, 30), time(9, 50), time(10, 0), time(10, 45),
                time(11, 45), time(12, 30), time(13, 20), time(14, 15)
            ]:
                exit_price = row['close']
                profit = (exit_price - entry) if side == 'buy' else (entry - exit_price)
                log.append(f"{t} EXIT    {side.upper()} | EXIT={exit_price:.2f} | P&L={profit:.2f} SLOT-EXIT | Trade {trade_id}")
                trades.append((etime, side, entry, row.name, exit_price, profit))
                position = None
                continue

    # Write log to file
    with open('corrected_trading_log_2025-04-29.txt', 'w') as f:
        for line in log:
            f.write(line + '\n')
        total_pnl = sum(trade[5] for trade in trades)
        f.write(f"\nTOTAL P&L for the day: {total_pnl:.2f}\n")

    return pd.DataFrame(trades, columns=[
        'Entry_Time', 'Side', 'Entry', 'Exit_Time', 'Exit', 'Profit'
    ])

# 6) Run & show
print("\n>>> CS_Alpha trades for 2025-04-29:")
df_trades = backtest_cs_alpha(df)
print(df_trades.to_string(index=False))
