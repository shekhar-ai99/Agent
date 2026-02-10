import pandas as pd
import pandas_ta as ta
from datetime import time

# —————————————————————————————————————————————————————————————————————
# 1) load & filter function
def load_and_filter(filename: str, date_str: str) -> pd.DataFrame:
    df = pd.read_csv(
        filename,
        parse_dates=['datetime'],
        index_col='datetime'
    )
    df = df[df.index.date == pd.to_datetime(date_str).date()]
    return df

# —————————————————————————————————————————————————————————————————————
# 2) compute indicators
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['RSI_10'] = ta.rsi(df['close'], length=10).fillna(0)
    df['ATR_10'] = ta.atr(df['high'], df['low'], df['close'], length=10).fillna(0)
    for L in [7, 14, 21, 50]:
        df[f'EMA{L}'] = ta.ema(df['close'], length=L).fillna(0)
    macd = ta.macd(df['close']).fillna(0)
    df['MACD']        = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['RSI_slope']   = df['RSI_10'].pct_change().fillna(0)
    return df

# —————————————————————————————————————————————————————————————————————
# 3) SuperTrend helper
def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0):
    atr = df['ATR_10']
    hl2 = (df['high'] + df['low']) / 2
    ub  = hl2 + mult * atr
    lb  = hl2 - mult * atr
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
            if dir.iloc[i] == 1 and st.iloc[i] < st.iloc[i-1]:
                st.iloc[i] = st.iloc[i-1]
            if dir.iloc[i] == -1 and st.iloc[i] > st.iloc[i-1]:
                st.iloc[i] = st.iloc[i-1]
    return st, dir

# —————————————————————————————————————————————————————————————————————
# 4) AlphaTrend calculation
def compute_alpha_trend(df: pd.DataFrame) -> pd.DataFrame:
    df['AlphaTrend'] = (
          0.65 * df['close']
        + 0.25 * df['close'].shift(1).ffill()
        + 0.1  * df['RSI_10'] * (df['ATR_10'] / df['close'].shift(1))
    )
    df['AlphaTrend'] = df['AlphaTrend'].ffill().fillna(df['close'])
    return df

# —————————————————————————————————————————————————————————————————————
# 5) backtest function

def backtest_cs_alpha(df: pd.DataFrame, prev_day_close: float) -> pd.DataFrame:
    trades       = []
    position     = None    # dict with keys: side, entry, stop, time
    morning_done = False

    for i in range(1, len(df)):
        r = df.iloc[i]
        p = df.iloc[i-1]
        t = r.name.time()

        # compute scores
        buy_score = (
            int(r['close'] > p['close']) +
            int(abs(r['close'] - p['close']) > 1) +
            int(r['RSI_10'] > 50) +
            int(r['MACD'] > r['MACD_signal']) +
            int(df['SuperTrend_dir'].iloc[i] == 1) +
            (0.5 if r['RSI_slope'] > 0 else 0)
        )
        sell_score = (
            int(r['close'] < p['close']) +
            int(abs(r['close'] - p['close']) > 1) +
            int(r['RSI_10'] < 50) +
            int(r['MACD'] < r['MACD_signal']) +
            int(df['SuperTrend_dir'].iloc[i] == -1) +
            (0.5 if r['RSI_slope'] < 0 else 0)
        )

        # log row
        print(f"{t} ROW    O={r['open']:.2f} C={r['close']:.2f} "
              f"BUY={buy_score:.1f} SELL={sell_score:.1f}")

        # 1) morning gap entry (first >=09:15)
        if not morning_done and position is None and t >= time(9,15):
            entry = r['open']
            side  = 'buy' if entry > prev_day_close else 'sell'
            gap   = abs(entry - prev_day_close)
            stop  = entry - 0.5 * gap if side=='buy' else entry + 0.5 * gap
            print(f"{t} ENTER   {side.upper():<4} | Entry={entry:.2f}" \
                  f" PrevClose={prev_day_close:.2f} | INIT-STOP={stop:.2f}")
            position     = {'side':side, 'entry':entry, 'stop':stop, 'time':r.name}
            morning_done = True
            continue

        # 2) if in trade, update stop & check exits
        if position:
            side  = position['side']
            entry = position['entry']
            stop  = position['stop']

            # running profit
            pnl = (r['close'] - entry) if side=='buy' else (entry - r['close'])
            if pnl > 0:
                # trail-stop = half of running profit
                new_stop = entry + 0.5 * pnl if side=='buy' else entry - 0.5 * pnl
                stop     = max(stop, new_stop) if side=='buy' else min(stop, new_stop)

            print(f"{t} RUNNING {side.upper():<4} | Entry={entry:.2f} " \
                  f"C={r['close']:.2f} STOP={stop:.2f}")

            # 2a) stop hit
            if (side=='buy'  and r['low']  <= stop) or \
               (side=='sell' and r['high'] >= stop):
                exit_price = stop
                final_pnl  = (exit_price-entry) if side=='buy' else (entry-exit_price)
                print(f"{t} EXIT    {side.upper():<4} | EXIT={exit_price:.2f} " \
                      f"| P&L={final_pnl:.2f} STOP-HIT")
                trades.append((position['time'], side, entry, r.name, exit_price, final_pnl))
                position = None
                continue

            # 2b) score-flip exit
            flip = (side=='buy'  and sell_score >= 3.5) or \
                   (side=='sell' and buy_score  >= 3.5)
            if flip:
                exit_price = r['close'] if pnl>0 else stop
                final_pnl  = (exit_price-entry) if side=='buy' else (entry-exit_price)
                print(f"{t} EXIT    {side.upper():<4} | EXIT={exit_price:.2f} " \
                      f"| P&L={final_pnl:.2f} SCORE-FLIP")
                trades.append((position['time'], side, entry, r.name, exit_price, final_pnl))
                position = None
                continue

            # update stop
            position['stop'] = stop
            continue

        # 3) mid-day score entries (no time filter)
        if buy_score >= 4:
            entry = r['open']
            stop  = entry - 0.5 * (p['high'] - p['low'])
            position = {'side':'buy','entry':entry,'stop':stop,'time':r.name}
            print(f"{t} ENTER   BUY  | Entry={entry:.2f} STOP={stop:.2f} "
                  f"| Score={buy_score:.1f}")
            continue

        if sell_score >= 4.5:
            entry = r['open']
            stop  = entry + 0.5 * (p['high'] - p['low'])
            position = {'side':'sell','entry':entry,'stop':stop,'time':r.name}
            print(f"{t} ENTER   SELL | Entry={entry:.2f} STOP={stop:.2f} "
                  f"| Score={sell_score:.1f}")
            continue

    # build results
    trades_df = pd.DataFrame(trades, columns=[
        'Entry_Time','Side','Entry','Exit_Time','Exit','Profit'
    ])

    # print PnL summary
    total = trades_df['Profit'].sum() if not trades_df.empty else 0.0
    print(f"\nTOTAL P&L for the day: {total:.2f}\n")

    return trades_df

# —————————————————————————————————————————————————————————————————————
# main execution
if __name__ == '__main__':
    filename       = 'results/nifty_historical_data_5min.csv'
    date_str       = '2025-04-29'
    prev_day_close = 24360.0

    df = load_and_filter(filename, date_str)
    df = compute_indicators(df)
    df['SuperTrend'], df['SuperTrend_dir'] = supertrend(df)
    df = compute_alpha_trend(df)

    print(f"\n>>> CS_Alpha trades for {date_str}:")
    trades = backtest_cs_alpha(df, prev_day_close)
    print(trades.to_string(index=False))
