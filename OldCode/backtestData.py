import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

def backtest_strategy(df, rr=2.0, atr_mult=1.2, use_rsi=True):
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
    except Exception as e:
        print("Indicator calculation error:", e)
        return None

    df['Position'] = 0
    df['EntryPrice'] = np.nan
    df['ExitPrice'] = np.nan
    df['PnL'] = 0.0

    in_position=False
    position_type = 0  # 1 = long, -1 = short
    entry_price = sl = tp = 0

    for i in range(1, len(df)):
        row_prev = df.loc[i - 1]
        row = df.loc[i]
        entry_conditions_met = (
        row['EMA9'] > row_prev['EMA9'] and
        row['RSI'] > 45 and
        row['Close'] > row['VWAP']
    )

        if entry_conditions_met and not in_position:
            print("🚀 LONG ENTRY TRIGGERED")
            in_position = True
            position_type = 1
            entry_price = row['Close']
            sl = entry_price - row['ATR'] * atr_mult
            tp = entry_price + row['ATR'] * atr_mult * rr
            df.at[i, 'Position'] = 1
            df.at[i, 'EntryPrice'] = entry_price
            continue


        try:
            rsi_val = float(row['RSI']) if not pd.isnull(row['RSI']) else None
        except:
            rsi_val = None

        atr_value = row['ATR']  

        #print(f"Checking Entry on {row['Datetime']}: EMA9={row['EMA9']}, VWAP={row['VWAP']}, RSI={row['RSI']}, ATR={atr_value}")


        if not in_position:
            condition_ema = row['EMA9'] > row_prev['EMA9']
            condition_rsi = row['RSI'] > 45
            condition_vwap = row['Close'] > row['VWAP']

            print(f"[{row['Datetime']}] Conditions - EMA_up: {condition_ema}, RSI>45: {condition_rsi}, Close>VWAP: {condition_vwap}")

            if condition_ema and condition_rsi and condition_vwap:
                print("🚀 LONG ENTRY TRIGGERED")
                in_position = True
                position_type = 1
                entry_price = row['Close']
                sl = entry_price - row['ATR'] * atr_mult
                tp = entry_price + row['ATR'] * atr_mult * rr
                df.at[i, 'Position'] = 1
                df.at[i, 'EntryPrice'] = entry_price
                continue

            exit_conditions_met = (
            (position_type == 1 and (row['Low'] <= sl or row['High'] >= tp)) or
            (position_type == -1 and (row['High'] >= sl or row['Low'] <= tp))
        )

            if exit_conditions_met and in_position:
                print("🏁 EXIT TRIGGERED")
                df.at[i, 'ExitPrice'] = sl if row['Low'] <= sl else tp
                df.at[i, 'PnL'] = df.at[i, 'ExitPrice'] - entry_price if position_type == 1 else entry_price - df.at[i, 'ExitPrice']
                in_position = False
                continue
                # print("🔻 SHORT ENTRY TRIGGERED")
                # in_position = True
                # position_type = -1
                # entry_price = row['Close']
                # sl = entry_price + row['ATR'] * atr_mult
                # tp = entry_price - row['ATR'] * atr_mult * rr
                # df.at[i, 'Position'] = -1
                # df.at[i, 'EntryPrice'] = entry_price
                # continue

        else:
            df.at[i, 'Position'] = position_type
            df.at[i, 'EntryPrice'] = entry_price

            if position_type == 1:
                if row['Low'] <= sl:
                    df.at[i, 'ExitPrice'] = sl
                    df.at[i, 'PnL'] = sl - entry_price
                    in_position = False
                elif row['High'] >= tp:
                    df.at[i, 'ExitPrice'] = tp
                    df.at[i, 'PnL'] = tp - entry_price
                    in_position = False

            elif position_type == -1:
                if row['High'] >= sl:
                    df.at[i, 'ExitPrice'] = sl
                    df.at[i, 'PnL'] = entry_price - sl
                    in_position = False
                elif row['Low'] <= tp:
                    df.at[i, 'ExitPrice'] = tp
                    df.at[i, 'PnL'] = entry_price - tp
                    in_position = False
            if df.at[i - 1, 'Position'] == 1:
                if row['Low'] <= sl:
                    df.at[i, 'ExitPrice'] = sl
                    df.at[i, 'PnL'] = sl - entry_price
                    in_position = False
                elif row['High'] >= tp:
                    df.at[i, 'ExitPrice'] = tp
                    df.at[i, 'PnL'] = tp - entry_price
                    in_position = False
            elif df.at[i - 1, 'Position'] == -1:
                if row['High'] >= sl:
                    df.at[i, 'ExitPrice'] = sl
                    df.at[i, 'PnL'] = entry_price - sl
                    in_position = False
                elif row['Low'] <= tp:
                    df.at[i, 'ExitPrice'] = tp
                    df.at[i, 'PnL'] = entry_price - tp
                    in_position = False

    return df

if __name__ == '__main__':
    try:
        print("Downloading data...")
        ticker = "NIFTYBEES.NS"  # Nifty 50 index
        interval = "5m"
        period = "5d"
        data = yf.download(ticker, interval=interval, period=period)

        if data.empty:
            print("Error: No data downloaded.")
        else:
            # FLATTEN COLUMNS if MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(col).strip() for col in data.columns.values]

    # ENSURE required columns are named as expected
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col not in data.columns:
                    print(f"Renaming missing column for {col}...")
                    for c in data.columns:
                        if col.lower() in c.lower():
                            data.rename(columns={c: col}, inplace=True)

            print("Columns after cleaning:", list(data.columns))
            print("Running backtest...")
            result = backtest_strategy(data)
            if result is None:
                print("Backtest failed due to errors.")
            else:
                #print(result.tail())  # 👈 print last few rows to confirm strategy ran
                trades = result[result['ExitPrice'].notna()]
                if not trades.empty:
                    print("\nTRADES EXECUTED:")
                    print(trades[['Position', 'EntryPrice', 'ExitPrice', 'PnL']])

                    total_trades = len(trades)
                    winning_trades = sum(trades['PnL'] > 0)
                    total_pnl = trades['PnL'].sum()
                    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

                    print("\nPERFORMANCE SUMMARY:")
                    print(f"Total Trades: {total_trades}")
                    print(f"Winning Trades: {winning_trades}")
                    print(f"Losing Trades: {total_trades - winning_trades}")
                    print(f"Total PnL: {total_pnl:.2f}")
                    print(f"Win Rate: {win_rate:.2f}%")
                else:
                    print("No trades were executed during the backtest period.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
