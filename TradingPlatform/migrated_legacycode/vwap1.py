import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

def backtest_strategy(df, rr=2.0, atr_mult=1.2, use_rsi=True):
    df = df.copy()

    # VWAP
    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        print("Warning: 'Volume' is missing or zero, can't compute VWAP.")
        return df
    df = df.dropna(subset=['EMA9', 'RSI', 'ATR'])

    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    # === DEBUG (optional) ===
    print("Close shape:", df['Close'].shape, type(df['Close']))

    # Technical Indicators
    ema_raw = EMAIndicator(close=df['Close'], window=9).ema_indicator()
    df['EMA9'] = pd.Series(ema_raw.values.flatten(), index=df.index)

    # === DEBUG (optional) ===
    print("EMA9 shape:", df['EMA9'].shape)

    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()

    df['Position'] = 0
    df['EntryPrice'] = np.nan
    df['ExitPrice'] = np.nan
    df['PnL'] = 0.0

    in_position = False
    long = False
    entry_price = 0
    sl = 0
    tp = 0

    for i in range(30, len(df)):
        if not in_position:
            # LONG condition
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
                print(f"{df.index[i]}: Entry {'LONG' if long else 'SHORT'} at {entry_price}, SL={sl}, TP={tp}")
                df['CumulativePnL'] = df['PnL'].cumsum()

            # SHORT condition
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
            # EXIT CONDITIONS
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
