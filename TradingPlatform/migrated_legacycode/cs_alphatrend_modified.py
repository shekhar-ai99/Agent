import pandas as pd
import pandas_ta as ta
from datetime import time
from io import StringIO

# Load & filter function
def load_and_filter(data_string: str, date_str: str) -> pd.DataFrame:
    df = pd.read_csv(
        StringIO(data_string),
        parse_dates=['datetime'],
        index_col='datetime'
    )
    df = df[df.index.date == pd.to_datetime(date_str).date()]
    return df

# Compute indicators
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['RSI_10'] = ta.rsi(df['close'], length=10).ffill().fillna(50)
    df['ATR_10'] = ta.atr(df['high'], df['low'], df['close'], length=10, mamode='ema').ffill().fillna(15.0)
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14, mamode='ema').ffill().fillna(15.0)
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14'].ffill().fillna(0)
    macd = ta.macd(df['close']).ffill().fillna(0)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['RSI_slope'] = df['RSI_10'].pct_change().ffill().fillna(0)
    df['EMA3'] = df['close'].ewm(span=3, adjust=False).mean()
    df['EMA6'] = df['close'].ewm(span=6, adjust=False).mean()
    return df

# SuperTrend helper
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

# AlphaTrend calculation
def compute_alpha_trend(df: pd.DataFrame) -> pd.DataFrame:
    alpha_trend = pd.Series(index=df.index, dtype=float)
    atr_weight = df['ATR_14'] / df['close'].shift(1).ffill()
    for i in range(len(df)):
        if i == 0:
            alpha_trend.iloc[i] = df['close'].iloc[i]
        else:
            alpha_trend.iloc[i] = (
                0.65 * df['close'].iloc[i]
                + 0.25 * alpha_trend.iloc[i-1]
                + 0.1 * df['RSI_10'].iloc[i] * atr_weight.iloc[i]
            )
    df['AlphaTrend'] = alpha_trend.ffill().fillna(df['close'])
    return df

# Backtest function
def backtest_cs_alpha(df: pd.DataFrame, prev_day_close: float) -> pd.DataFrame:
    trades = []
    position = None
    morning_done = False
    atr_mean = df['ATR_10'].rolling(window=5, min_periods=1).mean()
    atr_std = df['ATR_10'].rolling(window=5, min_periods=1).std()

    df['Market_Condition'] = 'ranging'
    df.loc[df['ADX'] > 20, 'Market_Condition'] = 'trending'
    df.loc[df['ATR_10'] > (atr_mean + 0.25 * atr_std), 'Market_Condition'] = 'volatile'
    df.loc[df.index.time <= time(11, 15), 'Market_Condition'] = 'volatile'

    for i in range(len(df)):
        r = df.iloc[i]
        t = r.name.time()

        p = df.iloc[i-1] if i > 0 else r
        p2 = df.iloc[i-2] if i > 1 else r

        crossover = r['close'] > p2['close'] and p['close'] <= p2['close']
        crossunder = r['close'] < p2['close'] and p['close'] >= p2['close']
        momentum = abs(r['close'] - p2['close']) > 1
        rsi_slope_pos = r['RSI_slope'] > 0
        rsi_slope_neg = r['RSI_slope'] < 0
        macd_buy = r['MACD'] > r['MACD_signal']
        macd_sell = r['MACD'] < r['MACD_signal']
        supertrend_buy = df['SuperTrend_dir'].iloc[i] == 1 or (df['SuperTrend_dir'].iloc[i] == 0 and r['close'] > r['AlphaTrend'])
        supertrend_sell = df['SuperTrend_dir'].iloc[i] == -1 or (df['SuperTrend_dir'].iloc[i] == 0 and r['close'] < r['AlphaTrend'])
        ema_crossover_buy = r['EMA3'] > r['EMA6'] and p['EMA3'] <= p['EMA6']
        ema_crossover_sell = r['EMA3'] < r['EMA6'] and p['EMA3'] >= p['EMA6']
        rsi_buy = r['RSI_10'] < 35 or (r['RSI_10'] > 40 and sum([crossover, momentum, supertrend_buy, rsi_slope_pos, macd_buy]) >= 3)
        rsi_sell = r['RSI_10'] > 65 or (r['RSI_10'] < 60 and sum([crossunder, momentum, supertrend_sell, rsi_slope_neg, macd_sell]) >= 3)

        buy_conditions = [crossover, momentum, rsi_buy, supertrend_buy, rsi_slope_pos, macd_buy]
        sell_conditions = [crossunder, momentum, rsi_sell, supertrend_sell, rsi_slope_neg, macd_sell]
        buy_count = sum(buy_conditions)
        sell_count = sum(sell_conditions)

        buy_score = (
            int(r['close'] > r['AlphaTrend']) +
            int(r['RSI_10'] > 50) +
            int(macd_buy) +
            int(df['SuperTrend_dir'].iloc[i] == 1) +
            (0.5 if rsi_slope_pos else 0) +
            int(ema_crossover_buy)
        )
        sell_score = (
            int(r['close'] < r['AlphaTrend']) +
            int(r['RSI_10'] < 50) +
            int(macd_sell) +
            int(df['SuperTrend_dir'].iloc[i] == -1) +
            (0.5 if rsi_slope_neg else 0) +
            int(ema_crossover_sell)
        )

        if not morning_done and position is None and i == 0:
            entry = r['open']
            side = 'buy' if entry > prev_day_close else 'sell'
            stop = entry - 1.0 * r['ATR_10'] if side == 'buy' else entry + 1.0 * r['ATR_10']
            target = entry + 2.5 * r['ATR_10'] if side == 'buy' else entry - 2.5 * r['ATR_10']
            position = {'side': side, 'entry': entry, 'stop': stop, 'target': target, 'time': r.name, 'bar_count': 0}
            morning_done = True
            continue

        if position:
            side = position['side']
            entry = position['entry']
            stop = position['stop']
            target = position['target']
            bar_count = position['bar_count'] + 1

            current_price = r['close']
            pnl = (current_price - entry) if side == 'buy' else (entry - current_price)

            if i > 0:
                if t <= time(9, 30) and position['time'] == df.index[0]:
                    range_prev = 0.5 * (p['high'] - p['low'])
                    new_stop = entry - (range_prev + 1.0 * r['ATR_10']) if side == 'buy' else entry + (range_prev + 1.0 * r['ATR_10'])
                else:
                    atr_mult = 1.0 if (side == 'buy' and r['RSI_10'] > 75) or (side == 'sell' and r['RSI_10'] < 25) else 1.0
                    new_stop = entry - atr_mult * r['ATR_10'] if side == 'buy' else entry + atr_mult * r['ATR_10']
                stop = max(stop, new_stop) if side == 'buy' else min(stop, new_stop)

            if (side == 'buy' and r['low'] <= stop) or (side == 'sell' and r['high'] >= stop):
                exit_price = stop
                final_pnl = (exit_price - entry) if side == 'buy' else (entry - exit_price)
                trades.append((position['time'], side, entry, r.name, exit_price, final_pnl))
                position = None
                continue

            if target and ((side == 'buy' and r['high'] >= target) or (side == 'sell' and r['low'] <= target)):
                exit_price = target
                final_pnl = (exit_price - entry) if side == 'buy' else (entry - exit_price)
                trades.append((position['time'], side, entry, r.name, exit_price, final_pnl))
                position = None
                continue

            opposite = (side == 'buy' and sell_count >= 4 and sell_score >= 3.5) or (side == 'sell' and buy_count >= 4 and buy_score >= 3.5)
            rsi_exit = (r['RSI_10'] > 80 or r['RSI_10'] < 20) and bar_count >= 3
            max_hold = ((bar_count >= 8 and r['Market_Condition'] == 'ranging') or
                        (bar_count >= 10 and r['Market_Condition'] in ['trending', 'volatile']))
            if opposite or rsi_exit or max_hold:
                exit_price = r['close']
                final_pnl = (exit_price - entry) if side == 'buy' else (entry - exit_price)
                trades.append((position['time'], side, entry, r.name, exit_price, final_pnl))
                position = None
                continue

            position['stop'] = stop
            position['bar_count'] = bar_count
            continue

        if t < time(9, 45):
            sl_mult, tp_mult = 1.0, 2.5
        elif t < time(14, 0):
            sl_mult, tp_mult = 1.0, 2.0
        else:
            sl_mult, tp_mult = 0.7, 1.5

        if (buy_count >= 4 and buy_score >= 3.0 and r['ATR_10'] > 10):
            entry = r['close']
            stop = entry - sl_mult * r['ATR_10']
            target = entry + tp_mult * r['ATR_10']
            position = {'side': 'buy', 'entry': entry, 'stop': stop, 'target': target, 'time': r.name, 'bar_count': 0}
            continue

        if (sell_count >= 4 and sell_score >= 3.0 and r['ATR_10'] > 10):
            entry = r['close']
            stop = entry + sl_mult * r['ATR_10']
            target = entry - tp_mult * r['ATR_10']
            position = {'side': 'sell', 'entry': entry, 'stop': stop, 'target': target, 'time': r.name, 'bar_count': 0}
            continue

        if t >= time(15, 25) and position:
            exit_price = r['close']
            final_pnl = (exit_price - entry) if side == 'buy' else (entry - exit_price)
            trades.append((position['time'], side, entry, r.name, exit_price, final_pnl))
            position = None
            continue

    trades_df = pd.DataFrame(trades, columns=[
        'Entry_Time', 'Side', 'Entry', 'Exit_Time', 'Exit', 'Profit'
    ])
    return trades_df

# Main execution
if __name__ == '__main__':
    data_string = """datetime,open,high,low,close,volume
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
2025-04-29 13:40:00+05:30,24366.9,24368.25,24361.65,24361.65,0
2025-04-29 13:45:00+05:30,24361.75,24362.9,24357.65,24358.15,0
2025-04-29 13:50:00+05:30,24358.05,24360.95,24355.55,24359.75,0
2025-04-29 13:55:00+05:30,24359.75,24361.25,24356.35,24357.65,0
2025-04-29 14:00:00+05:30,24357.65,24359.55,24354.05,24355.65,0
2025-04-29 14:05:00+05:30,24355.55,24357.75,24353.65,24355.55,0
2025-04-29 14:10:00+05:30,24355.65,24358.05,24353.75,24356.05,0
2025-04-29 14:15:00+05:30,24356.15,24358.65,24353.85,24356.65,0
2025-04-29 14:20:00+05:30,24356.75,24359.05,24354.25,24356.75,0
2025-04-29 14:25:00+05:30,24356.85,24359.15,24354.35,24356.85,0
2025-04-29 14:30:00+05:30,24356.95,24359.25,24354.45,24356.95,0
2025-04-29 14:35:00+05:30,24357.05,24359.35,24354.55,24357.05,0
2025-04-29 14:40:00+05:30,24357.15,24359.45,24354.65,24357.15,0
2025-04-29 14:45:00+05:30,24357.25,24359.55,24354.75,24357.25,0
2025-04-29 14:50:00+05:30,24357.35,24359.65,24354.85,24357.35,0
2025-04-29 14:55:00+05:30,24357.45,24359.75,24354.95,24357.45,0
2025-04-29 15:00:00+05:30,24357.55,24359.85,24355.05,24357.55,0
2025-04-29 15:05:00+05:30,24357.65,24359.95,24355.15,24357.65,0
2025-04-29 15:10:00+05:30,24357.75,24360.05,24355.25,24357.75,0
2025-04-29 15:15:00+05:30,24357.85,24360.15,24355.35,24357.85,0
2025-04-29 15:20:00+05:30,24357.95,24360.25,24333.1,24338.85,0
2025-04-29 15:25:00+05:30,24338.95,24360.35,24325.45,24325.45,0"""
    date_str = '2025-04-29'
    prev_day_close = 24360.0

    df = load_and_filter(data_string, date_str)
    df = compute_indicators(df)
    df['SuperTrend'], df['SuperTrend_dir'] = supertrend(df)
    df = compute_alpha_trend(df)
    trades = backtest_cs_alpha(df, prev_day_close)
    print(f"\n>>> CS_Alpha trades for {date_str}:")
    print(trades.to_string(index=False))
    print(f"\nTotal P&L: {trades['Profit'].sum():.2f} points")