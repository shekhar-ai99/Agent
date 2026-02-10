import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

def backtest_strategy(df, rr=2.0, atr_mult=1.2, use_rsi=True):
    df = df.copy()

    # VWAP
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    # Technical Indicators
    df['EMA9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()

    # Initialize trade tracking
    df['Position'] = 0
    df['EntryPrice'] = np.nan
    df['ExitPrice'] = np.nan
    df['PnL'] = 0.0

    in_position = False
    long = False
    entry_price = sl = tp = 0

    for i in range(30, len(df)):
        if not in_position:
            # LONG setup
            if (
                df['Close'].iloc[i - 1] < df['VWAP'].iloc[i - 1] and
                df['Close'].iloc[i] > df['VWAP'].iloc[i] and
                df['Low'].iloc[i] <= df['EMA9'].iloc[i] and
                (not use_rsi or (pd.notna(df['RSI'].iloc[i]) and df['RSI'].iloc[i] < 35))
            ):
                in_position = True
                long = True
                entry_price = df['Close'].iloc[i]
                sl = entry_price - df['ATR'].iloc[i] * atr_mult
                tp = entry_price + df['ATR'].iloc[i] * atr_mult * rr
                df.at[df.index[i], 'Position'] = 1
                df.at[df.index[i], 'EntryPrice'] = entry_price

            # SHORT setup
            elif (
                df['Close'].iloc[i - 1] > df['VWAP'].iloc[i - 1] and
                df['Close'].iloc[i] < df['VWAP'].iloc[i] and
                df['High'].iloc[i] >= df['EMA9'].iloc[i] and
                (not use_rsi or (pd.notna(df['RSI'].iloc[i]) and df['RSI'].iloc[i] > 65))
            ):
                in_position = True
                long = False
                entry_price = df['Close'].iloc[i]
                sl = entry_price + df['ATR'].iloc[i] * atr_mult
                tp = entry_price - df['ATR'].iloc[i] * atr_mult * rr
                df.at[df.index[i], 'Position'] = -1
                df.at[df.index[i], 'EntryPrice'] = entry_price

        else:
            # Manage trade exit
            if long:
                if df['Low'].iloc[i] <= sl:
                    df.at[df.index[i], 'ExitPrice'] = sl
                    df.at[df.index[i], 'PnL'] = sl - entry_price
                    in_position = False
                elif df['High'].iloc[i] >= tp:
                    df.at[df.index[i], 'ExitPrice'] = tp
                    df.at[df.index[i], 'PnL'] = tp - entry_price
                    in_position = False
            else:
                if df['High'].iloc[i] >= sl:
                    df.at[df.index[i], 'ExitPrice'] = sl
                    df.at[df.index[i], 'PnL'] = entry_price - sl
                    in_position = False
                elif df['Low'].iloc[i] <= tp:
                    df.at[df.index[i], 'ExitPrice'] = tp
                    df.at[df.index[i], 'PnL'] = entry_price - tp
                    in_position = False

    return df

# ======================
# RUN BACKTEST
# ======================
if __name__ == '__main__':
    # Download historical data (change ticker if needed)
    ticker = "^NSEI"  # NIFTY 50 index
    data = yf.download(ticker, interval="1d", period="3mo")
    data = data.dropna()

    # Run backtest
    result = backtest_strategy(data)

    # Show trades only
    trades = result[result['ExitPrice'].notna()]
    print(trades[['Position', 'EntryPrice', 'ExitPrice', 'PnL']])

    # Summary
    total_trades = len(trades)
    total_pnl = trades['PnL'].sum()
    win_rate = len(trades[trades['PnL'] > 0]) / total_trades * 100 if total_trades > 0 else 0

    print("\n=== BACKTEST SUMMARY ===")
    print(f"Total Trades: {total_trades}")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Win Rate: {win_rate:.2f}%")
