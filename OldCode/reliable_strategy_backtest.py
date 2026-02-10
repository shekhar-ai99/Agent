import pandas as pd
import pandas_ta as ta
from datetime import time

try:
    # Load data with error handling
    df = pd.read_csv("test 1 week.csv")
    if df.empty:
        print("Error: CSV file is empty.")
        exit(1)
    required_columns = ['datetime', 'open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing required columns. Expected {required_columns}, found {df.columns.tolist()}")
        exit(1)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df.set_index('datetime', inplace=True)

    # Calculate Indicators with NaN handling
    for col in ['RSI', 'ATR', 'EMA7', 'EMA14', 'EMA21', 'EMA50', 'MACD', 'MACD_signal', 'ADX', 'RSI_slope']:
        if col == 'RSI':
            df[col] = ta.rsi(df['close'], length=14, fillna=True)
        elif col == 'ATR':
            df[col] = ta.atr(df['high'], df['low'], df['close'], length=14, fillna=True)
        elif col in ['EMA7', 'EMA14', 'EMA21', 'EMA50']:
            df[col] = ta.ema(df['close'], length=int(col.replace('EMA', '')))
        elif col in ['MACD', 'MACD_signal']:
            macd = ta.macd(df['close'], fillna=True)
            df[col] = macd[col]
        elif col == 'ADX':
            adx = ta.adx(df['high'], df['low'], df['close'], length=14, fillna=True)
            df[col] = adx['ADX_14']
        elif col == 'RSI_slope':
            df[col] = df['RSI'].pct_change().fillna(0)
        df[col] = df[col].fillna(method='ffill').fillna(0)

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

    # Pivot Points (Daily)
    df['Date'] = df.index.date
    pivot_points = df.groupby('Date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).shift(1)
    pivot_points['Pivot'] = (pivot_points['high'] + pivot_points['low'] + pivot_points['close']) / 3
    pivot_points['R1'] = 2 * pivot_points['Pivot'] - pivot_points['low']
    pivot_points['S1'] = 2 * pivot_points['Pivot'] - pivot_points['high']
    pivot_points['R2'] = pivot_points['Pivot'] + (pivot_points['high'] - pivot_points['low'])
    pivot_points['S2'] = pivot_points['Pivot'] - (pivot_points['high'] - pivot_points['low'])
    df = df.merge(pivot_points[['Pivot', 'R1', 'S1', 'R2', 'S2']], left_on=df['Date'], right_index=True, how='left').fillna(method='ffill').fillna(0)

    # Custom AlphaTrend (v2)
    df['AlphaTrend'] = 0.65 * df['close'] + 0.25 * df['AlphaTrend'].shift(1) + 0.1 * df['RSI'] * (df['ATR'] / df['close'].shift(1))
    df['AlphaTrend'] = df['AlphaTrend'].fillna(df['close'])

    # Market Condition Detection
    df['EMA_slope'] = df['EMA7'].pct_change().rolling(5).mean().fillna(0)
    atr_mean, atr_std = df['ATR'].mean(), df['ATR'].std()
    df['Market_Condition'] = 'ranging'
    df.loc[df['ADX'] > 30, 'Market_Condition'] = 'trending'
    df.loc[df['ATR'] > (atr_mean + 1.8 * atr_std), 'Market_Condition'] = 'volatile'
    df.loc[(df['EMA_slope'] > 0).all() or (df['EMA_slope'] < 0).all(), 'Market_Condition'] = 'trending'

    # Strategy Definitions
    strategies = {
        'Momentum': lambda row, prev_row: row['ADX'] > 25 and row['EMA7'] > row['EMA14'] and row['close'] > row['AlphaTrend'] and row['RSI'] > 50 and row.name.time() < time(14, 0),
        'Reversal': lambda row, prev_row: row['ADX'] < 20 and row['RSI'] < 35 and row['close'] < row['Pivot'] and row['RSI_slope'] < -0.01,
        'Breakout': lambda row, prev_row: row['ATR'] > (atr_mean + 1.8 * atr_std) and row['close'] > row['R1'] and row['MACD'] > row['MACD_signal'],
        'Scalping': lambda row, prev_row: row['ADX'] < 20 and abs(row['EMA7'] - row['EMA14']) < 0.2 * row['ATR'] and row['ATR'] > 1.5 * atr_mean and row['RSI'] < 50,
        'MA_Crossover': lambda row, prev_row: row['EMA7'] > row['EMA14'] > row['EMA21'] > row['EMA50'] and row['RSI'] > 50 and row['SuperTrend_dir'] == 1,
        'Pivot_Point': lambda row, prev_row: (row['close'] > row['R1'] and row['RSI'] > 50) or (row['close'] < row['S1'] and row['RSI'] < 50),
        'Pullback': lambda row, prev_row: row['ADX'] > 20 and row['close'] > row['EMA21'] and row['close'] < prev_row['close'] and row['RSI'] > 45,
        'Gap_and_Go': lambda row, prev_row: row.name.time() < time(9, 30) and row['open'] > prev_row['close'] * 1.015 and row['RSI'] < 70,
        'Bull_Flag': lambda row, prev_row: row['ADX'] > 20 and row['ATR'] > 1.5 * atr_mean and row['RSI'] > 50 and row['MACD'] > row['MACD_signal'],
        'AlphaTrend_ST_RSI': lambda row, prev_row: True  # Special handling below
    }

    # Backtest Function
    def backtest_strategy(df, strategy_name, condition):
        trades = []
        position = None
        highest_price = None
        atr_mult = 2.0 if df['Market_Condition'].iloc[-1] == 'trending' else 3.0
        
        for i in range(1, len(df)):
            row, prev_row = df.iloc[i], df.iloc[i-1]
            current_time = row.name.time()
            
            # AlphaTrend_ST_RSI_v2 Logic
            if strategy_name == 'AlphaTrend_ST_RSI':
                # Morning Rule
                if current_time == time(9, 15) and position is None:
                    position = {'signal': 'buy', 'entry_price': row['open'], 'entry_time': row.name, 'strategy': strategy_name}
                    highest_price = row['high']
                    continue
                if current_time == time(9, 20) and position and position['signal'] == 'buy':
                    profit = row['close'] - position['entry_price']
                    trades.append({
                        'strategy': strategy_name,
                        'market_condition': row['Market_Condition'],
                        'profit': profit,
                        'entry_time': position['entry_time'],
                        'exit_time': row.name,
                        'entry_price': position['entry_price'],
                        'exit_price': row['close']
                    })
                    position = None
                    highest_price = None
                    continue
                
                # Entry Score for Buy
                entry_score = (
                    1 if row['close'] > row['AlphaTrend'] else 0 +
                    1 if row['RSI'] > 50 else 0 +
                    1 if row['MACD'] > row['MACD_signal'] else 0 +
                    1 if row['SuperTrend_dir'] == 1 else 0 +
                    0.5 if row['RSI_slope'] > 0 else 0
                )
                # Sell Score
                sell_score = (
                    1 if row['close'] < row['AlphaTrend'] else 0 +
                    1 if row['RSI'] < 50 else 0 +
                    1 if row['MACD'] < row['MACD_signal'] else 0 +
                    1 if row['SuperTrend_dir'] == -1 else 0 +
                    0.5 if row['RSI_slope'] < 0 else 0
                )
                
                if position is None and entry_score >= 3.5 and row['ATR'] > 12 and current_time < time(15, 0) and row['close'] > row['Pivot']:
                    position = {'signal': 'buy', 'entry_price': row['open'], 'entry_time': row.name, 'strategy': strategy_name}
                    highest_price = row['high']
                elif position and position['signal'] == 'buy' and sell_score >= 3.5:
                    profit = row['close'] - position['entry_price']
                    trades.append({
                        'strategy': strategy_name,
                        'market_condition': row['Market_Condition'],
                        'profit': profit,
                        'entry_time': position['entry_time'],
                        'exit_time': row.name,
                        'entry_price': position['entry_price'],
                        'exit_price': row['close']
                    })
                    position = None
                    highest_price = None
                elif position and position['signal'] == 'buy':
                    highest_price = max(highest_price, row['high'])
                    if current_time < time(9, 45):
                        stop_loss = highest_price - 1.2 * row['ATR']
                        take_profit = position['entry_price'] + 3.0 * row['ATR']
                    elif current_time < time(14, 0):
                        stop_loss = highest_price - 1.8 * row['ATR']
                        take_profit = position['entry_price'] + 2.5 * row['ATR']
                    else:
                        stop_loss = highest_price - 0.8 * row['ATR']
                        take_profit = position['entry_price'] + 1.8 * row['ATR']
                    if row['low'] <= stop_loss or row['close'] >= take_profit or row['RSI'] > 80 or row['RSI'] < 20:
                        profit = (stop_loss if row['low'] <= stop_loss else take_profit if row['close'] >= take_profit else row['close']) - position['entry_price']
                        trades.append({
                            'strategy': strategy_name,
                            'market_condition': row['Market_Condition'],
                            'profit': profit,
                            'entry_time': position['entry_time'],
                            'exit_time': row.name,
                            'entry_price': position['entry_price'],
                            'exit_price': stop_loss if row['low'] <= stop_loss else take_profit if row['close'] >= take_profit else row['close']
                        })
                        position = None
                        highest_price = None
            else:
                if condition(row, prev_row) and position is None:
                    position = {'signal': 'buy', 'entry_price': row['open'], 'entry_time': row.name, 'strategy': strategy_name}
                    highest_price = row['high']
            
            # Trailing Stop and Exit Logic
            if position and strategy_name != 'AlphaTrend_ST_RSI':
                highest_price = max(highest_price, row['high'])
                if strategy_name == 'Gap_and_Go' and current_time >= time(9, 30):
                    profit = row['close'] - position['entry_price']
                    trades.append({
                        'strategy': strategy_name,
                        'market_condition': row['Market_Condition'],
                        'profit': profit,
                        'entry_time': position['entry_time'],
                        'exit_time': row.name,
                        'entry_price': position['entry_price'],
                        'exit_price': row['close']
                    })
                    position = None
                    highest_price = None
                else:
                    stop_loss = highest_price - (1.5 * atr_mult * row['ATR'] if strategy_name in ['Momentum', 'Pullback'] else atr_mult * row['ATR'])
                    if strategy_name == 'Breakout':
                        stop_loss = highest_price - 2.5 * row['ATR']
                    elif strategy_name == 'Gap_and_Go':
                        stop_loss = highest_price - 2.0 * row['ATR']
                    if row['low'] <= stop_loss or (strategy_name in ['Momentum', 'Breakout'] and row['RSI'] > 75) or (strategy_name in ['Reversal', 'Scalping'] and row['RSI'] < 25):
                        profit = (stop_loss - position['entry_price']) * (1.5 if strategy_name == 'Breakout' else 1.0)
                        trades.append({
                            'strategy': strategy_name,
                            'market_condition': row['Market_Condition'],
                            'profit': profit,
                            'entry_time': position['entry_time'],
                            'exit_time': row.name,
                            'entry_price': position['entry_price'],
                            'exit_price': stop_loss
                        })
                        position = None
                        highest_price = None
            
            # Session End Exit
            if current_time >= time(15, 25) and position:
                profit = row['close'] - position['entry_price']
                trades.append({
                    'strategy': strategy_name,
                    'market_condition': row['Market_Condition'],
                    'profit': profit,
                    'entry_time': position['entry_time'],
                    'exit_time': row.name,
                    'entry_price': position['entry_price'],
                    'exit_price': row['close']
                })
                position = None
                highest_price = None
        
        return pd.DataFrame(trades)

    # Run Backtest
    results = {}
    for strategy, condition in strategies.items():
        results[strategy] = backtest_strategy(df, strategy, condition)

    # Output Results
    for strategy in strategies.keys():
        strategy_df = results[strategy]
        if not strategy_df.empty:
            total_profit = strategy_df['profit'].sum()
            win_rate = len(strategy_df[strategy_df['profit'] > 0]) / len(strategy_df) * 100 if len(strategy_df) > 0 else 0
            print(f"\n{strategy} Backtest Results:")
            print(f"Total Profit: {total_profit:.2f} points")
            print(f"Number of Trades: {len(strategy_df)}")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Average Profit per Trade: {strategy_df['profit'].mean():.2f}")
            print(f"Market Condition Distribution: {strategy_df['market_condition'].value_counts().to_dict()}")
            print(f"Detailed Trades:")
            for _, trade in strategy_df.iterrows():
                print(f"  Entry: {trade['entry_time']}, Exit: {trade['exit_time']}, Condition: {trade['market_condition']}, "
                      f"Entry Price: {trade['entry_price']:.2f}, Exit Price: {trade['exit_price']:.2f}, Profit: {trade['profit']:.2f}")

    # Summary Table
    print("\nSummary Table:")
    summary = pd.DataFrame({
        'Strategy': list(results.keys()),
        'Total_Profit': [results[s]['profit'].sum() if not results[s].empty else 0 for s in results.keys()],
        'Trade_Count': [len(results[s]) if not results[s].empty else 0 for s in results.keys()],
        'Avg_Profit': [results[s]['profit'].mean() if not results[s].empty else 0 for s in results.keys()],
        'Win_Rate': [len(results[s][results[s]['profit'] > 0]) / len(results[s]) * 100 if not results[s].empty else 0 for s in results.keys()],
        'Market_Condition_Dist': [results[s]['market_condition'].value_counts().to_dict() if not results[s].empty else {} for s in results.keys()]
    })
    print(summary)

except Exception as e:
    print(f"Error occurred: {str(e)}")
    exit(1)