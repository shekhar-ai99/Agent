celear
import pandas as pd
import pandas_ta as ta
from datetime import time

# 1) Load & filter function
def load_and_filter(filename: str, date_str: str) -> pd.DataFrame:
    df = pd.read_csv(
        filename,
        parse_dates=['datetime'],
        index_col='datetime'
    )
    df = df[df.index.date == pd.to_datetime(date_str).date()]
    return df

# 2) Compute indicators
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['RSI_10'] = ta.rsi(df['close'], length=10).ffill().fillna(0)
    df['ATR_5'] = ta.atr(df['high'], df['low'], df['close'], length=5).ffill().fillna(10.0)
    df['ADX']    = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14'].ffill().fillna(0)
    for L in [7, 14, 21, 50]:
        df[f'EMA{L}'] = ta.ema(df['close'], length=L).ffill().fillna(0)
    macd = ta.macd(df['close']).ffill().fillna(0)
    df['MACD']        = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['RSI_slope']   = df['RSI_10'].pct_change().ffill().fillna(0)
    return df

# 3) SuperTrend helper
def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0):
    atr = df['ATR_5']
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

# 4) AlphaTrend calculation
def compute_alpha_trend(df: pd.DataFrame) -> pd.DataFrame:
    df['AlphaTrend'] = (
          0.65 * df['close']
        + 0.25 * df['close'].shift(1).ffill()
        + 0.1  * df['RSI_10'] * (df['ATR_5'] / df['close'].shift(1))
    )
    df['AlphaTrend'] = df['AlphaTrend'].ffill().fillna(df['close'])
    return df

# 5) Backtest function
def backtest_cs_alpha(df: pd.DataFrame, prev_day_close: float) -> pd.DataFrame:
    trades        = []
    position      = None  # dict with keys: side, entry, stop, target, time, bar_count
    morning_done  = False
    atr_mean      = df['ATR_5'].rolling(window=5, min_periods=1).mean()
    atr_std       = df['ATR_5'].rolling(window=5, min_periods=1).std()

    # Market condition
    df['Market_Condition'] = 'ranging'
    df.loc[df['ADX'] > 25, 'Market_Condition'] = 'trending'
    df.loc[df['ATR_5'] > (atr_mean + 0.15 * atr_std), 'Market_Condition'] = 'volatile'
    df.loc[df.index.time <= time(10, 0), 'Market_Condition'] = 'volatile'

    for i in range(len(df)):
        r = df.iloc[i]
        t = r.name.time()

        # Access previous row safely
        p = df.iloc[i-1] if i > 0 else r

        # Compute scores
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

        # Log row
        print(f"{t} ROW    O={r['open']:.2f} C={r['close']:.2f} "
              f"BUY={buy_score:.1f} SELL={sell_score:.1f} ATR={r['ATR_5']:.2f}")

        # 1) Morning gap entry (first bar)
        if not morning_done and position is None and i == 0:
            entry = r['open']
            side  = 'buy' if entry > prev_day_close * 1.005 else 'sell' if entry < prev_day_close * 0.995 else 'buy'
            stop  = entry - r['ATR_5'] * 2.0 if side == 'buy' else entry + r['ATR_5'] * 2.0
            position = {'side': side, 'entry': entry, 'stop': stop, 'time': r.name, 'bar_count': 0}
            print(f"{t} ENTER   {side.upper():<4} | Entry={entry:.2f} "
                  f"STOP={stop:.2f} | Morning Trade")
            morning_done = True
            continue

        # 2) If in trade, update stop & check exits
        if position:
            side      = position['side']
            entry     = position['entry']
            stop      = position['stop']
            bar_count = position['bar_count'] + 1

            # Running profit
            current_price = r['close']
            pnl = (current_price - entry) if side == 'buy' else (entry - current_price)

            # Update TSL
            if i > 0:  # Ensure previous bar exists
                atr_scale = r['ATR_5'] / atr_mean.iloc[i] if atr_mean.iloc[i] > 0 else 1.0
                tsl_mult = 0.75 if r['Market_Condition'] == 'volatile' else 1.0 if r['Market_Condition'] == 'trending' else 1.25
                if t <= time(9, 45) and position['time'] == df.index[0]:
                    # Morning TSL: (prev_close - prev_open) * 0.5 + prev_open
                    prev_open  = p['open']
                    prev_close = p['close']
                    new_stop   = (prev_close - prev_open) * 0.5 + prev_open
                else:
                    # ATR-based TSL for mid-day trades
                    if pnl > 0:
                        new_stop = entry + 1.0 * pnl * atr_scale * tsl_mult if side == 'buy' else entry - 1.0 * pnl * atr_scale * tsl_mult
                    else:
                        new_stop = stop

                # Adjust stop
                stop = max(stop, new_stop) if side == 'buy' else min(stop, new_stop)

            # Define TP (30 points for mid-day trades)
            target = entry + 30.0 if side == 'buy' else entry - 30.0

            # Log running position
            print(f"{t} RUNNING {side.upper():<4} | Entry={entry:.2f} "
                  f"C={r['close']:.2f} STOP={stop:.2f} | Market={r['Market_Condition']}")

            # 2a) Stop hit
            if (side == 'buy' and r['low'] <= stop) or (side == 'sell' and r['high'] >= stop):
                exit_price = stop
                final_pnl  = (exit_price - entry) if side == 'buy' else (entry - exit_price)
                print(f"{t} EXIT    {side.upper():<4} | EXIT={exit_price:.2f} "
                      f"| P&L={final_pnl:.2f} STOP-HIT")
                trades.append((position['time'], side, entry, r.name, exit_price, final_pnl))
                position = None
                continue

            # 2b) Target hit
            if (side == 'buy' and r['high'] >= target) or (side == 'sell' and r['low'] <= target):
                exit_price = target
                final_pnl  = (exit_price - entry) if side == 'buy' else (entry - exit_price)
                print(f"{t} EXIT    {side.upper():<4} | EXIT={exit_price:.2f} "
                      f"| P&L={final_pnl:.2f} TARGET-HIT")
                trades.append((position['time'], side, entry, r.name, exit_price, final_pnl))
                position = None
                continue

            # 2c) Score-flip or RSI exit
            flip = (side == 'buy' and sell_score >= 4.5) or (side == 'sell' and buy_score >= 4.5)
            rsi_exit = r['RSI_10'] > 85 or r['RSI_10'] < 15
            max_hold = ((bar_count >= 10 and r['Market_Condition'] == 'ranging') or
                       (bar_count >= 15 and r['Market_Condition'] in ['trending', 'volatile']))
            if (flip or rsi_exit or max_hold) and not (t <= time(9, 45) and position['time'] == df.index[0]):
                exit_price = r['close'] if pnl > 0 else stop
                final_pnl  = (exit_price - entry) if side == 'buy' else (entry - exit_price)
                reason = ("SCORE-FLIP" if flip else
                         "RSI-EXTREME" if rsi_exit else
                         "MAX-HOLD")
                print(f"{t} EXIT    {side.upper():<4} | EXIT={exit_price:.2f} "
                      f"| P&L={final_pnl:.2f} {reason}")
                trades.append((position['time'], side, entry, r.name, exit_price, final_pnl))
                position = None
                continue

            # Update position
            position['stop'] = stop
            position['bar_count'] = bar_count
            continue

        # 3) Mid-day score entries
        if buy_score >= 3.5 and r['ATR_5'] > 8 and df['SuperTrend_dir'].iloc[i] == 1:
            entry = r['open']
            stop  = entry - 15.0  # Fixed 15-point SL
            target = entry + 30.0  # Fixed 30-point TP
            position = {'side': 'buy', 'entry': entry, 'stop': stop, 'target': target, 'time': r.name, 'bar_count': 0}
            print(f"{t} ENTER   BUY  | Entry={entry:.2f} STOP={stop:.2f} "
                  f"TARGET={target:.2f} | Score={buy_score:.1f} | Market={r['Market_Condition']}")
            continue

        if sell_score >= 4.5 and r['ATR_5'] > 8 and df['SuperTrend_dir'].iloc[i] == -1:
            entry = r['open']
            stop  = entry + 15.0  # Fixed 15-point SL
            target = entry - 30.0  # Fixed 30-point TP
            position = {'side': 'sell', 'entry': entry, 'stop': stop, 'target': target, 'time': r.name, 'bar_count': 0}
            print(f"{t} ENTER   SELL | Entry={entry:.2f} STOP={stop:.2f} "
                  f"TARGET={target:.2f} | Score={sell_score:.1f} | Market={r['Market_Condition']}")
            continue

        # 4) Session end exit
        if t >= time(15, 25) and position:
            exit_price = r['close']
            final_pnl  = (exit_price - entry) if side == 'buy' else (entry - exit_price)
            print(f"{t} EXIT    {side.upper():<4} | EXIT={exit_price:.2f} "
                  f"| P&L={final_pnl:.2f} SESSION-END")
            trades.append((position['time'], side, entry, r.name, exit_price, final_pnl))
            position = None
            continue

    # Build results
    trades_df = pd.DataFrame(trades, columns=[
        'Entry_Time', 'Side', 'Entry', 'Exit_Time', 'Exit', 'Profit'
    ])

    # Print PnL summary
    total = trades_df['Profit'].sum() if not trades_df.empty else 0.0
    print(f"\nTOTAL P&L for the day: {total:.2f}\n")

    return trades_df

# 6) Main execution
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