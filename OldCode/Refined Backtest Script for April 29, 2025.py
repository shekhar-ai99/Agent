import pandas as pd
import pandas_ta as ta
from datetime import time, datetime

try:
    # Load data
    df = pd.read_csv("results/test 1 week.csv")
    if df.empty:
        print("Error: CSV file is empty.")
        exit(1)
    required_columns = ['datetime', 'open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing required columns. Expected {required_columns}, found {df.columns.tolist()}")
        exit(1)

    # Convert 'datetime' column to datetime and set it as the index
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    if df['datetime'].isnull().any():
        print("Error: Some datetime values could not be converted.")
        exit(1)
    df.set_index('datetime', inplace=True)

    # Filter for April 29, 2025
    df = df[df.index.date == pd.to_datetime('2025-04-29').date()]

    # Calculate Indicators
    for col in ['RSI', 'ATR', 'EMA7', 'EMA14', 'EMA21', 'EMA50', 'MACD', 'MACD_signal', 'ADX', 'RSI_slope']:
        if col == 'RSI':
            df[col] = ta.rsi(df['close'], length=14).fillna(0).astype(float)
        elif col == 'ATR':
            df[col] = ta.atr(df['high'], df['low'], df['close'], length=14).fillna(0).astype(float)
        elif col in ['EMA7', 'EMA14', 'EMA21', 'EMA50']:
            df[col] = ta.ema(df['close'], length=int(col.replace('EMA', ''))).fillna(0).astype(float)
        elif col in ['MACD', 'MACD_signal']:
            macd = ta.macd(df['close']).fillna(0)
            df[col] = macd[f'MACD_12_26_9'] if col == 'MACD' else macd['MACDs_12_26_9']
        elif col == 'ADX':
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            df[col] = adx['ADX_14'].fillna(0).astype(float)
        elif col == 'RSI_slope':
            df[col] = df['RSI'].pct_change().rolling(5).mean().fillna(0).astype(float)
        df[col] = df[col].ffill().fillna(0)

    # SuperTrend Calculation
    def supertrend(df, period=10, multiplier=3):
        atr = df['ATR']
        hl2 = (df['high'] + df['low']) / 2
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        supertrend = pd.Series(index=df.index, dtype=float, data=0)
        direction = pd.Series(index=df.index, dtype=int, data=0)
        for i in range(1, len(df)):
            if df['close'].iloc[i-1] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lowerband.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upperband.iloc[i]
                direction.iloc[i] = -1
            if direction.iloc[i] == 1 and supertrend.iloc[i] < supertrend.iloc[i-1]:
                supertrend.iloc[i] = supertrend.iloc[i-1]
            elif direction.iloc[i] == -1 and supertrend.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = supertrend.iloc[i-1]
        return supertrend, direction

    df['SuperTrend'], df['SuperTrend_dir'] = supertrend(df)

    # Pivot Points
    df['Date'] = df.index.date
    pivot_points = df.groupby('Date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).shift(1)
    pivot_points['Pivot'] = (pivot_points['high'] + pivot_points['low'] + pivot_points['close']) / 3
    pivot_points['R1'] = 2 * pivot_points['Pivot'] - pivot_points['low']
    pivot_points['S1'] = 2 * pivot_points['Pivot'] - pivot_points['high']
    df = df.merge(pivot_points[['Pivot', 'R1', 'S1']], left_on=df['Date'], right_index=True, how='left').ffill().fillna(0)

    # Custom AlphaTrend
    df['AlphaTrend'] = df['close'].copy()
    df['AlphaTrend'] = 0.65 * df['close'] + 0.25 * df['AlphaTrend'].shift(1) + 0.1 * df['RSI'] * (df['ATR'] / df['close'].shift(1))
    df['AlphaTrend'] = df['AlphaTrend'].ffill().fillna(df['close']).astype(float)

    # Market Condition
    df['EMA_slope'] = df['EMA7'].pct_change().rolling(5).mean().fillna(0).astype(float)
    atr_mean, atr_std = df['ATR'].mean(), df['ATR'].std()
    df['Market_Condition'] = 'ranging'
    df.loc[df['ADX'] > 20, 'Market_Condition'] = 'trending'
    df.loc[df['ATR'] > (atr_mean + 0.5 * atr_std), 'Market_Condition'] = 'volatile'

    # Strategy Definitions
    strategies = {
        'Momentum': lambda row, prev_row: row['ADX'] > 15 and row['EMA7'] > row['EMA14'] and row['close'] > row['AlphaTrend'] and (row['RSI'] > 40 and row['RSI'] < 70),
        'Reversal': lambda row, prev_row: row['ADX'] < 15 and row['RSI'] < 40 and row['close'] < row['Pivot'],
        'Breakout': lambda row, prev_row: row['ATR'] > (atr_mean + 0.5 * atr_std) and row['close'] > row['R1'] and row['MACD'] > row['MACD_signal'],
        'Scalping': lambda row, prev_row: row['ADX'] < 15 and abs(row['EMA7'] - row['EMA14']) < 0.1 * row['ATR'] and (row['RSI'] > 40 and row['RSI'] < 70),
        'MA_Crossover': lambda row, prev_row: row['EMA7'] > row['EMA14'] > row['EMA21'] > row['EMA50'] and row['RSI'] > 50 and row['EMA_slope'] > 0,
        'Pivot_Point': lambda row, prev_row: (row['close'] > row['R1'] and row['RSI'] > 40) or (row['close'] < row['S1'] and row['RSI'] < 60),
        'Pullback': lambda row, prev_row: row['ADX'] > 15 and row['close'] > row['EMA21'] and row['close'] < prev_row['close'] and (row['RSI'] > 40 and row['RSI'] < 70),
        'Gap_and_Go': lambda row, prev_row: row.name.time() < time(9, 30) and row['open'] > prev_row['close'] * 1.005,
        'Bull_Flag': lambda row, prev_row: row['ADX'] > 15 and row['ATR'] > 1.0 * atr_mean and row['RSI'] > 40,
        'AlphaTrend_ST_RSI_v1': lambda row, prev_row: (row['close'] > row['AlphaTrend'] and row['RSI'] > 40 and row['SuperTrend_dir'] == 1) or (row['close'] < row['AlphaTrend'] and row['RSI'] < 60 and row['SuperTrend_dir'] == -1),
        'AlphaTrend_ST_RSI_v2': lambda row, prev_row: (row['close'] > row['AlphaTrend'] and row['RSI'] > 40 and row['SuperTrend_dir'] == 1) or (row['close'] < row['AlphaTrend'] and row['RSI'] < 60 and row['SuperTrend_dir'] == -1),
        'CS_Alpha': lambda row, prev_row: ((row.name.time() == time(9, 15) and row['close'] > row['AlphaTrend'] and row['RSI'] > 40) or (row['close'] > row['AlphaTrend'] and row['RSI'] > 40 and row['SuperTrend_dir'] == 1)) or (row['close'] < row['AlphaTrend'] and row['RSI'] < 60 and row['SuperTrend_dir'] == -1)
    }

    # Backtest Function with Trade Details
    def backtest_strategy(df, strategy_name, condition):
        trades = []
        position = None
        highest_price = None

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            current_time = row.name.time()
            atr_mult = 1.0 if row['Market_Condition'] == 'volatile' else 1.5 if row['Market_Condition'] == 'trending' else 2.0

            # Entry Logic
            if condition(row, prev_row) and position is None:
                signal = 'buy' if row['close'] > row['AlphaTrend'] else 'sell'
                position = {'signal': signal, 'entry_price': row['open'], 'entry_time': row.name, 'strategy': strategy_name}
                highest_price = row['high']
                sl = position['entry_price'] - (1.0 * atr_mult * row['ATR']) if position['signal'] == 'buy' else position['entry_price'] + (1.0 * atr_mult * row['ATR'])
                tsl = None  # Removed TSL for CS_Alpha to avoid premature exits
                print(f"{strategy_name} - Trade Entered: {row.name} | Signal: {position['signal']} | Entry Price: {position['entry_price']:.2f} | SL: {sl:.2f} | TSL: {tsl} | Market: {row['Market_Condition']}")

            # Exit Logic
            if position:
                highest_price = max(highest_price, row['high'])
                sl = highest_price - (1.0 * atr_mult * row['ATR']) if position['signal'] == 'buy' else highest_price + (1.0 * atr_mult * row['ATR'])
                take_profit = position['entry_price'] + (2.0 * atr_mult * row['ATR']) if position['signal'] == 'buy' else position['entry_price'] - (2.0 * atr_mult * row['ATR'])
                if (row['low'] <= sl or row['close'] >= take_profit or (strategy_name == 'CS_Alpha' and (row['RSI'] > 70 or row['RSI'] < 30))):
                    exit_price = sl if row['low'] <= sl else take_profit if row['close'] >= take_profit else row['close']
                    profit = (exit_price - position['entry_price']) if position['signal'] == 'buy' else (position['entry_price'] - exit_price)
                    trades.append({
                        'strategy': strategy_name,
                        'market_condition': row['Market_Condition'],
                        'profit': profit,
                        'entry_time': position['entry_time'],
                        'exit_time': row.name,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'sl': sl,
                        'tsl': tsl
                    })
                    print(f"{strategy_name} - Trade Exited: {row.name} | Profit: {profit:.2f} | Exit Price: {exit_price:.2f} | SL: {sl:.2f} | TSL: {tsl}")
                    position = None
                    highest_price = None

            # Session End Exit
            if current_time >= time(15, 25) and position:
                profit = row['close'] - position['entry_price'] if position['signal'] == 'buy' else position['entry_price'] - row['close']
                trades.append({
                    'strategy': strategy_name,
                    'market_condition': row['Market_Condition'],
                    'profit': profit,
                    'entry_time': position['entry_time'],
                    'exit_time': row.name,
                    'entry_price': position['entry_price'],
                    'exit_price': row['close'],
                    'sl': sl,
                    'tsl': tsl
                })
                print(f"{strategy_name} - Trade Exited: {row.name} | Profit: {profit:.2f} | Exit Price: {row['close']:.2f} | SL: {sl:.2f} | TSL: {tsl}")
                position = None
                highest_price = None

        return pd.DataFrame(trades)

    # Run Backtest
    results = {}
    for strategy, condition in strategies.items():
        results[strategy] = backtest_strategy(df, strategy, condition)

    # Summary Table
    summary = pd.DataFrame({
        'Strategy': list(results.keys()),
        'Total_Profit': [results[s]['profit'].sum() if not results[s].empty else 0 for s in results.keys()],
        'Trade_Count': [len(results[s]) if not results[s].empty else 0 for s in results.keys()],
        'Avg_Profit': [results[s]['profit'].mean() if not results[s].empty else 0 for s in results.keys()],
        'Win_Rate': [len(results[s][results[s]['profit'] > 0]) / len(results[s]) * 100 if not results[s].empty else 0 for s in results.keys()]
    })
    print("\nSummary Table for April 29, 2025:")
    print(summary)

except Exception as e:
    print(f"Error occurred: {str(e)}")
    exit(1)