import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import warnings
warnings.filterwarnings("ignore")

def backtest_strategy(df, rr=2.0, atr_mult=1.2):
    df = df.copy()

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        print("Missing columns:", df.columns)
        return None

    try:
        df[required_cols] = df[required_cols].apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=required_cols, inplace=True)
    except Exception as e:
        print("Data type conversion error:", e)
        return None

    if df['Volume'].isnull().all() or (df['Volume'] == 0).all():
        print("Invalid Volume data")
        return None

    try:
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['EMA9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
        df = df.dropna().reset_index()
        if 'index' in df.columns:
            df.rename(columns={"index": "Datetime"}, inplace=True)
    except Exception as e:
        print("Indicator calculation error:", e)
        return None

    df['Position'] = 0
    df['EntryPrice'] = np.nan
    df['ExitPrice'] = np.nan
    df['PnL'] = 0.0

    trade_log = []
    in_position = False
    position_type = 0  # 1 = long, -1 = short
    entry_price = sl = tp = highest_price = lowest_price = 0

    for i in range(1, len(df)):
        row_prev = df.loc[i - 1]
        row = df.loc[i]

        if not in_position:
            # Long entry
            if (
                row['EMA9'] > row_prev['EMA9'] and
                row['RSI'] > 45 and
                row['Close'] > row['VWAP']
            ):
                in_position = True
                position_type = 1
                entry_price = row['Close']
                sl = entry_price - row['ATR'] * atr_mult
                tp = entry_price + row['ATR'] * atr_mult * rr
                highest_price = entry_price
                df.at[i, 'Position'] = 1
                df.at[i, 'EntryPrice'] = entry_price
                print(f"🚀 LONG ENTRY at {row['Datetime']} | Entry: {entry_price:.2f}, SL: {sl:.2f}, TP: {tp:.2f}")

            # Short entry
            elif (
                row['EMA9'] < row_prev['EMA9'] and
                row['RSI'] < 55 and
                row['Close'] < row['VWAP']
            ):
                in_position = True
                position_type = -1
                entry_price = row['Close']
                sl = entry_price + row['ATR'] * atr_mult
                tp = entry_price - row['ATR'] * atr_mult * rr
                lowest_price = entry_price
                df.at[i, 'Position'] = -1
                df.at[i, 'EntryPrice'] = entry_price
                print(f"🔻 SHORT ENTRY at {row['Datetime']} | Entry: {entry_price:.2f}, SL: {sl:.2f}, TP: {tp:.2f}")

        else:
            # Trailing SL for long
            if position_type == 1:
                highest_price = max(highest_price, row['High'])
                new_trailing_sl = highest_price - row['ATR'] * atr_mult
                if new_trailing_sl > sl:
                    sl = new_trailing_sl
                    print(f"🔄 LONG Trailing SL updated at {row['Datetime']} | New SL: {sl:.2f}")

                if row['Low'] <= sl:
                    exit_price = sl
                    pnl = exit_price - entry_price
                    in_position = False
                    df.at[i, 'ExitPrice'] = exit_price
                    df.at[i, 'PnL'] = pnl
                    print(f"🏁 LONG SL HIT at {row['Datetime']} | Exit: {exit_price:.2f}, PnL: {pnl:.2f}")

                elif row['High'] >= tp:
                    exit_price = tp
                    pnl = exit_price - entry_price
                    in_position = False
                    df.at[i, 'ExitPrice'] = exit_price
                    df.at[i, 'PnL'] = pnl
                    print(f"🎯 LONG TP HIT at {row['Datetime']} | Exit: {exit_price:.2f}, PnL: {pnl:.2f}")

            # Trailing SL for short
            elif position_type == -1:
                lowest_price = min(lowest_price, row['Low'])
                new_trailing_sl = lowest_price + row['ATR'] * atr_mult
                if new_trailing_sl < sl:
                    sl = new_trailing_sl
                    print(f"🔄 SHORT Trailing SL updated at {row['Datetime']} | New SL: {sl:.2f}")

                if row['High'] >= sl:
                    exit_price = sl
                    pnl = entry_price - exit_price
                    in_position = False
                    df.at[i, 'ExitPrice'] = exit_price
                    df.at[i, 'PnL'] = pnl
                    print(f"🏁 SHORT SL HIT at {row['Datetime']} | Exit: {exit_price:.2f}, PnL: {pnl:.2f}")

                elif row['Low'] <= tp:
                    exit_price = tp
                    pnl = entry_price - exit_price
                    in_position = False
                    df.at[i, 'ExitPrice'] = exit_price
                    df.at[i, 'PnL'] = pnl
                    print(f"🎯 SHORT TP HIT at {row['Datetime']} | Exit: {exit_price:.2f}, PnL: {pnl:.2f}")

            if not in_position:
                trade_log.append({
                    'Datetime': row['Datetime'],
                    'Position': position_type,
                    'EntryPrice': entry_price,
                    'ExitPrice': exit_price,
                    'PnL': pnl
                })

    trades_df = pd.DataFrame(trade_log)
    return df, trades_df

if __name__ == '__main__':
    try:
        print("Downloading data...")
        ticker = "NIFTYBEES.NS"
        interval = "15m"
        period = "30d"
        data = yf.download(ticker, interval=interval, period=period)

        if data.empty:
            print("Error: No data downloaded.")
        else:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(col).strip() for col in data.columns.values]

            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col not in data.columns:
                    for c in data.columns:
                        if col.lower() in c.lower():
                            data.rename(columns={c: col}, inplace=True)

            print("Columns after cleaning:", list(data.columns))
            print("Running backtest...")
            result, trades = backtest_strategy(data)

            if trades.empty:
                print("No trades were executed.")
            else:
                total_trades = len(trades)
                winning_trades = trades[trades['PnL'] > 0]
                losing_trades = trades[trades['PnL'] <= 0]
                win_rate = len(winning_trades) / total_trades * 100
                avg_pnl = trades['PnL'].mean()
                total_pnl = trades['PnL'].sum()
                max_drawdown = trades['PnL'].cumsum().min()
                profit_factor = winning_trades['PnL'].sum() / abs(losing_trades['PnL'].sum()) if not losing_trades.empty else np.inf
                expectancy = avg_pnl

                print("\n📊 PERFORMANCE SUMMARY:")
                print(f"Total Trades       : {total_trades}")
                print(f"Winning Trades     : {len(winning_trades)}")
                print(f"Losing Trades      : {len(losing_trades)}")
                print(f"Win Rate           : {win_rate:.2f}%")
                print(f"Total PnL          : {total_pnl:.2f}")
                print(f"Average PnL/Trade  : {avg_pnl:.2f}")
                print(f"Profit Factor      : {profit_factor:.2f}")
                print(f"Max Drawdown       : {max_drawdown:.2f}")
                print(f"Expectancy         : {expectancy:.2f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
