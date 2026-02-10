import yfinance as yf
import pandas as pd
import numpy as np
import os

# -------------------- Custom Indicator Calculations --------------------
def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def atr(high, low, close, window=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=window).mean()

# -------------------- Strategy Logic --------------------
def backtest_strategy(df, rr=1.5, atr_mult=1.5):
    df = df.copy()
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[required_cols] = df[required_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=required_cols, inplace=True)

    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['EMA9'] = ema(df['Close'], 9)
    df['EMA21'] = ema(df['Close'], 21)
    df['RSI'] = rsi(df['Close'])
    df['ATR'] = atr(df['High'], df['Low'], df['Close'])
    df = df.dropna().reset_index()
    print("🔎 Columns in downloaded data:", data.columns.tolist())

    df['Position'] = 0
    df['EntryPrice'] = np.nan
    df['ExitPrice'] = np.nan
    df['PnL'] = 0.0

    in_position = False
    position_type = 0
    entry_price = sl = tp = trailing_sl = 0

    for i in range(1, len(df)):
        row_prev = df.loc[i - 1]
        row = df.loc[i]

        current_time = row['Datetime'].time() if 'Datetime' in row else pd.to_datetime(row['Date']).time()
        if not (pd.to_datetime("09:30").time() <= current_time <= pd.to_datetime("15:15").time()):
            continue

        if not in_position:
            atr_avg = df['ATR'].rolling(20).mean()[i]

            if (
                row['EMA9'] > row['EMA21'] and
                row_prev['EMA9'] <= row_prev['EMA21'] and
                row['RSI'] > 45 and
                row['ATR'] > atr_avg and
                row['Close'] > row['VWAP']
            ):
                in_position = True
                position_type = 1
                entry_price = row['Close']
                sl = entry_price - row['ATR'] * atr_mult
                tp = entry_price + row['ATR'] * atr_mult * rr
                trailing_sl = sl
                df.at[i, 'Position'] = 1
                df.at[i, 'EntryPrice'] = entry_price
                print(f"🚀 LONG ENTRY at {row['Datetime']} | Entry: {entry_price:.2f}, SL: {sl:.2f}, TP: {tp:.2f}")
        else:
            if position_type == 1:
                if row['High'] > entry_price + row['ATR'] * 0.5:
                    new_trailing_sl = row['High'] - row['ATR'] * atr_mult
                    if new_trailing_sl > trailing_sl:
                        trailing_sl = new_trailing_sl
                        print(f"🔁 Trailing SL updated at {row['Datetime']} | New SL: {trailing_sl:.2f}")

                if row['Low'] <= trailing_sl:
                    df.at[i, 'ExitPrice'] = trailing_sl
                    df.at[i, 'PnL'] = trailing_sl - entry_price
                    in_position = False
                    print(f"🏁 TRAILING STOP LOSS HIT at {row['Datetime']} | Exit: {trailing_sl:.2f}, PnL: {df.at[i, 'PnL']:.2f}")

                elif row['High'] >= tp:
                    df.at[i, 'ExitPrice'] = tp
                    df.at[i, 'PnL'] = tp - entry_price
                    in_position = False
                    print(f"🎯 TAKE PROFIT HIT at {row['Datetime']} | Exit: {tp:.2f}, PnL: {df.at[i, 'PnL']:.2f}")

    return df

# -------------------- Data Download or Fallback --------------------
def load_data(ticker="NIFTYBEES.NS", interval="15m", period="30d", fallback_csv="data.csv"):
    try:
        print("📡 Downloading data...")
        data = yf.download(ticker, interval=interval, period=period, auto_adjust=False)
        if data.empty:
            raise ValueError("Downloaded data is empty.")
        data.reset_index(inplace=True)
        print("🔎 Columns in downloaded data:", data.columns.tolist())

        return data
    except Exception as e:
        print(f"⚠️ Yahoo Finance failed: {e}")
        if os.path.exists(fallback_csv):
            print(f"📁 Loading fallback data from {fallback_csv}")
            return pd.read_csv(fallback_csv, parse_dates=True)
        else:
            print("❌ No fallback CSV found.")
            return pd.DataFrame()

# -------------------- Main Execution --------------------
if __name__ == '__main__':
    data = load_data()
    if data.empty:
        print("💥 Exiting: No data to run backtest.")
    else:
        print("🚀 Running backtest...")
        result = backtest_strategy(data)
        if result is not None:
            trades = result[result['ExitPrice'].notna()]
            if not trades.empty:
                total_trades = len(trades)
                winning_trades = sum(trades['PnL'] > 0)
                total_pnl = trades['PnL'].sum()
                avg_pnl = trades['PnL'].mean()
                max_dd = result['PnL'].cumsum().min()
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                profit_factor = trades[trades['PnL'] > 0]['PnL'].sum() / abs(trades[trades['PnL'] < 0]['PnL'].sum()) if trades[trades['PnL'] < 0].sum().any() else np.inf

                print("\n📊 PERFORMANCE SUMMARY:")
                print(f"Total Trades       : {total_trades}")
                print(f"Winning Trades     : {winning_trades}")
                print(f"Losing Trades      : {total_trades - winning_trades}")
                print(f"Win Rate           : {win_rate:.2f}%")
                print(f"Total PnL          : {total_pnl:.2f}")
                print(f"Average PnL/Trade  : {avg_pnl:.2f}")
                print(f"Profit Factor      : {profit_factor:.2f}")
                print(f"Max Drawdown       : {max_dd:.2f}")
                print(f"Expectancy         : {avg_pnl:.2f}")
            else:
                print("🤷 No trades executed during the backtest period.")
