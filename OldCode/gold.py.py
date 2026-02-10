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
    df['RSI_10'] = ta.rsi(df['close'], length=10).ffill().fillna(50)
    df['ATR_10'] = ta.atr(df['high'], df['low'], df['close'], length=10, mamode='ema').ffill().fillna(15.0)
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14, mamode='ema').ffill().fillna(15.0)
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14'].ffill().fillna(0)
    macd = ta.macd(df['close']).ffill().fillna(0)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['RSI_slope'] = df['RSI_10'].pct_change().ffill().fillna(0)
    return df

# 3) SuperTrend helper
def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0):
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

# 4) AlphaTrend calculation
def compute_alpha_trend(df: pd.DataFrame) -> pd.DataFrame:
    # Initialize AlphaTrend as a Series with NaN
    alpha_trend = pd.Series(index=df.index, dtype=float)
    atr_weight = df['ATR_14'] / df['close'].shift(1).ffill()
    
    # Compute AlphaTrend iteratively
    for i in range(len(df)):
        if i == 0:
            # Initialize with close price for the first row
            alpha_trend.iloc[i] = df['close'].iloc[i]
        else:
            # Use previous AlphaTrend value
            alpha_trend.iloc[i] = (
                0.65 * df['close'].iloc[i]
                + 0.25 * alpha_trend.iloc[i-1]
                + 0.1 * df['RSI_10'].iloc[i] * atr_weight.iloc[i]
            )
    
    # Assign to DataFrame and handle NaN
    df['AlphaTrend'] = alpha_trend.ffill().fillna(df['close'])
    return df

# 5) Backtest function
def backtest_cs_alpha(df: pd.DataFrame, prev_day_close: float) -> pd.DataFrame:
    trades = []
    position = None  # dict with keys: side, entry, stop, target, time, bar_count
    morning_done = False
    atr_mean = df['ATR_10'].rolling(window=5, min_periods=1).mean()
    atr_std = df['ATR_10'].rolling(window=5, min_periods=1).std()

    # Market condition
    df['Market_Condition'] = 'ranging'
    df.loc[df['ADX'] > 20, 'Market_Condition'] = 'trending'
    df.loc[df['ATR_10'] > (atr_mean + 0.25 * atr_std), 'Market_Condition'] = 'volatile'
    df.loc[df.index.time <= time(11, 15), 'Market_Condition'] = 'volatile'

    for i in range(len(df)):
        r = df.iloc[i]
        t = r.name.time()

        # Access previous rows safely
        p = df.iloc[i-1] if i > 0 else r
        p2 = df.iloc[i-2] if i > 1 else r

        # Compute entry conditions
        crossover = r['close'] > p2['close'] and p['close'] <= p2['close']
        crossunder = r['close'] < p2['close'] and p['close'] >= p2['close']
        momentum = abs(r['close'] - p2['close']) > 1
        rsi_slope_pos = r['RSI_slope'] > 0
        rsi_slope_neg = r['RSI_slope'] < 0
        macd_buy = r['MACD'] > r['MACD_signal']
        macd_sell = r['MACD'] < r['MACD_signal']
        supertrend_buy = df['SuperTrend_dir'].iloc[i] == 1 or (df['SuperTrend_dir'].iloc[i] == 0 and r['close'] > r['AlphaTrend'])
        supertrend_sell = df['SuperTrend_dir'].iloc[i] == -1 or (df['SuperTrend_dir'].iloc[i] == 0 and r['close'] < r['AlphaTrend'])
        
        # RSI conditions with flexibility
        rsi_buy = (50 < r['RSI_10'] < 70) or (r['RSI_10'] > 40 and sum([
            crossover, momentum, supertrend_buy, rsi_slope_pos, macd_buy
        ]) >= 4)
        rsi_sell = (30 < r['RSI_10'] < 50) or (r['RSI_10'] < 60 and sum([
            crossunder, momentum, supertrend_sell, rsi_slope_neg, macd_sell
        ]) >= 4)

        # Count buy/sell conditions
        buy_conditions = [crossover, momentum, rsi_buy, supertrend_buy, rsi_slope_pos, macd_buy]
        sell_conditions = [crossunder, momentum, rsi_sell, supertrend_sell, rsi_slope_neg, macd_sell]
        buy_count = sum(buy_conditions)
        sell_count = sum(sell_conditions)

        # Tiered entry score
        buy_score = (
            int(r['close'] > r['AlphaTrend']) +
            int(r['RSI_10'] > 50) +
            int(macd_buy) +
            int(df['SuperTrend_dir'].iloc[i] == 1) +
            (0.5 if rsi_slope_pos else 0)
        )
        sell_score = (
            int(r['close'] < r['AlphaTrend']) +
            int(r['RSI_10'] < 50) +
            int(macd_sell) +
            int(df['SuperTrend_dir'].iloc[i] == -1) +
            (0.5 if rsi_slope_neg else 0)
        )

        # Log row
        print(f"{t} ROW    O={r['open']:.2f} C={r['close']:.2f} "
              f"BUY={buy_score:.1f} SELL={sell_score:.1f} ATR={r['ATR_10']:.2f}")

        # 1) Morning gap entry (first bar)
        if not morning_done and position is None and i == 0:
            entry = r['open']
            side = 'buy' if entry > prev_day_close * 1.005 else 'sell' if entry < prev_day_close * 0.995 else 'buy'
            stop = entry - 1.2 * r['ATR_10'] if side == 'buy' else entry + 1.2 * r['ATR_10']
            target = entry + 3.0 * r['ATR_10'] if side == 'buy' else entry - 3.0 * r['ATR_10']
            position = {'side': side, 'entry': entry, 'stop': stop, 'target': target, 'time': r.name, 'bar_count': 0}
            print(f"{t} ENTER   {side.upper():<4} | Entry={entry:.2f} "
                  f"STOP={stop:.2f} TARGET={target:.2f} | Morning Trade")
            morning_done = True
            continue

        # 2) If in trade, update stop & check exits
        if position:
            side = position['side']
            entry = position['entry']
            stop = position['stop']
            target = position['target']
            bar_count = position['bar_count'] + 1

            # Running profit
            current_price = r['close']
            pnl = (current_price - entry) if side == 'buy' else (entry - current_price)

            # Update TSL
            if i > 0:
                if t <= time(9, 30) and position['time'] == df.index[0]:
                    # Morning TSL: 0.5 * (High_{t-1} - Low_{t-1}) + 2.5 * ATR_10
                    range_prev = 0.5 * (p['high'] - p['low'])
                    new_stop = entry + range_prev + 2.5 * r['ATR_10'] if side == 'buy' else entry - range_prev - 2.5 * r['ATR_10']
                else:
                    # Mid-day TSL: 3.5 * ATR_10, tighten to 2 * ATR_10 if RSI > 75/< 25
                    atr_mult = 2.0 if (side == 'buy' and r['RSI_10'] > 75) or (side == 'sell' and r['RSI_10'] < 25) else 3.5
                    new_stop = entry + atr_mult * r['ATR_10'] if side == 'buy' else entry - atr_mult * r['ATR_10']
                stop = max(stop, new_stop) if side == 'buy' else min(stop, new_stop)

            # Log running position
            print(f"{t} RUNNING {side.upper():<4} | Entry={entry:.2f} "
                  f"C={r['close']:.2f} STOP={stop:.2f} | Market={r['Market_Condition']}")

            # 2a) Stop hit
            if (side == 'buy' and r['low'] <= stop) or (side == 'sell' and r['high'] >= stop):
                exit_price = stop
                final_pnl = (exit_price - entry) if side == 'buy' else (entry - exit_price)
                print(f"{t} EXIT    {side.upper():<4} | EXIT={exit_price:.2f} "
                      f"| P&L={final_pnl:.2f} STOP-HIT")
                trades.append((position['time'], side, entry, r.name, exit_price, final_pnl))
                position = None
                continue

            # 2b) Target hit
            if target and ((side == 'buy' and r['high'] >= target) or (side == 'sell' and r['low'] <= target)):
                exit_price = target
                final_pnl = (exit_price - entry) if side == 'buy' else (entry - exit_price)
                print(f"{t} EXIT    {side.upper():<4} | EXIT={exit_price:.2f} "
                      f"| P&L={final_pnl:.2f} TARGET-HIT")
                trades.append((position['time'], side, entry, r.name, exit_price, final_pnl))
                position = None
                continue

            # 2c) Opposite signal, RSI extremes, or max hold
            opposite = (side == 'buy' and sell_count >= 5) or (side == 'sell' and buy_count >= 5)
            rsi_exit = (r['RSI_10'] > 80 or r['RSI_10'] < 20) and bar_count >= 3
            max_hold = ((bar_count >= 10 and r['Market_Condition'] == 'ranging') or
                        (bar_count >= 12 and r['Market_Condition'] in ['trending', 'volatile']))
            if opposite or rsi_exit or max_hold:
                exit_price = r['close']
                final_pnl = (exit_price - entry) if side == 'buy' else (entry - exit_price)
                reason = ("OPPOSITE-SIGNAL" if opposite else
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
        # Session-specific SL/TP
        if t < time(9, 45):
            sl_mult, tp_mult = 1.2, 3.0
        elif t < time(14, 0):
            sl_mult, tp_mult = 1.8, 2.5
        else:
            sl_mult, tp_mult = 0.8, 1.8

        if (buy_count >= 5 and buy_score >= 3.5 and r['ATR_10'] > 12):
            entry = r['open']
            stop = entry - sl_mult * r['ATR_10']
            target = entry + tp_mult * r['ATR_10']
            position = {'side': 'buy', 'entry': entry, 'stop': stop, 'target': target, 'time': r.name, 'bar_count': 0}
            print(f"{t} ENTER   BUY  | Entry={entry:.2f} STOP={stop:.2f} "
                  f"TARGET={target:.2f} | Score={buy_score:.1f} | Market={r['Market_Condition']}")
            continue

        if (sell_count >= 5 and sell_score >= 3.5 and r['ATR_10'] > 12):
            entry = r['open']
            stop = entry + sl_mult * r['ATR_10']
            target = entry - tp_mult * r['ATR_10']
            position = {'side': 'sell', 'entry': entry, 'stop': stop, 'target': target, 'time': r.name, 'bar_count': 0}
            print(f"{t} ENTER   SELL | Entry={entry:.2f} STOP={stop:.2f} "
                  f"TARGET={target:.2f} | Score={sell_score:.1f} | Market={r['Market_Condition']}")
            continue

        # 4) Session end exit
        if t >= time(15, 25) and position:
            exit_price = r['close']
            final_pnl = (exit_price - entry) if side == 'buy' else (entry - exit_price)
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
    filename = 'results/test 1 week.csv'
    date_str = '2025-04-29'
    prev_day_close = 24360.0

    df = load_and_filter(filename, date_str)
    df = compute_indicators(df)
    df['SuperTrend'], df['SuperTrend_dir'] = supertrend(df)
    df = compute_alpha_trend(df)
    #date_str=df['Entry_Time']
    print(f"\n>>> CS_Alpha trades for {date_str}:")
    trades = backtest_cs_alpha(df, prev_day_close)
    print(trades.to_string(index=False))