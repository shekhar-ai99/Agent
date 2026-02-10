import pandas as pd
import pandas_ta as ta
from datetime import time

# Load data
df = pd.read_csv("test 1 week.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Calculate Indicators
df['RSI'] = ta.rsi(df['close'], length=14)
df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
df['EMA7'] = ta.ema(df['close'], length=7)
df['EMA14'] = ta.ema(df['close'], length=14)
df['EMA21'] = ta.ema(df['close'], length=21)
df['EMA50'] = ta.ema(df['close'], length=50)
macd = ta.macd(df['close'])
df['MACD'] = macd['MACD_12_26_9']
df['MACD_signal'] = macd['MACDs_12_26_9']
adx = ta.adx(df['high'], df['low'], df['close'], length=14)
df['ADX'] = adx['ADX_14']

# SuperTrend Calculation
def supertrend(df, period=10, multiplier=3):
    atr = df['ATR']
    hl2 = (df['high'] + df['low']) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
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
df = df.merge(pivot_points[['Pivot', 'R1', 'S1', 'R2', 'S2']], left_on=df['Date'], right_index=True)

# Approximate AlphaTrend
df['AlphaTrend'] = 0.65 * df['close'] + 0.25 * df['close'].shift(1) + 0.1 * df['RSI'] * (df['ATR'] / df['close'].shift(1))
df['AlphaTrend'].fillna(df['close'], inplace=True)

# Market Condition Detection
df['EMA_slope'] = df['EMA7'].pct_change().rolling(5).mean()
atr_mean, atr_std = df['ATR'].mean(), df['ATR'].std()
df['Market_Condition'] = 'ranging'
df.loc[df['ADX'] > 30, 'Market_Condition'] = 'trending'
df.loc[df['ATR'] > (atr_mean + 1.8 * atr_std), 'Market_Condition'] = 'volatile'
df.loc[(df['EMA_slope'] > 0).all() or (df['EMA_slope'] < 0).all(), 'Market_Condition'] = 'trending'

# Strategy Definitions
strategies = {
    'Momentum': lambda row, prev_row: row['ADX'] > 25 and row['EMA7'] > row['EMA14'] and row['close'] > row['AlphaTrend'],
    'Reversal': lambda row, prev_row: row['ADX'] < 20 and row['RSI'] < 40 and row['close'] < row['Pivot'],
    'Breakout': lambda row, prev_row: row['ATR'] > (atr_mean + 1.8 * atr_std) and row['close'] > row['R1'],
    'Scalping': lambda row, prev_row: row['ADX'] < 20 and abs(row['EMA7'] - row['EMA14']) < 0.3 * row['ATR'],
    'MA_Crossover': lambda row, prev_row: row['EMA7'] > row['EMA14'] > row['EMA21'] > row['EMA50'],
    'Pivot_Point': lambda row, prev_row: row['close'] > row['R1'] or row['close'] < row['S1'],
    'Pullback': lambda row, prev_row: row['ADX'] > 20 and row['close'] > row['EMA21'] and row['close'] < prev_row['close'],
    'Gap_and_Go': lambda row, prev_row: row.name.time() < time(9, 30) and row['open'] > prev_row['close'] * 1.015,
    'Bull_Flag': lambda row, prev_row: row['ADX'] > 20  # Volume condition omitted
}

# Backtest Function
def backtest_strategy(df, strategy_name, condition):
    trades = []
    position = None
    for i in range(1, len(df)):
        row, prev_row = df.iloc[i], df.iloc[i-1]
        if condition(row, prev_row) and position is None:
            position = {'entry_price': row['open'], 'entry_time': row.name, 'condition': strategy_name}
        elif position and row['low'] <= row['open'] - 2.5 * row['ATR']:
            profit = row['close'] - position['entry_price'] if row['close'] > position['entry_price'] else row['open'] - 2.5 * row['ATR'] - position['entry_price']
            trades.append({
                'strategy': strategy_name,
                'market_condition': row['Market_Condition'],
                'profit': profit,
                'entry_time': position['entry_time'],
                'exit_time': row.name,
                'entry_price': position['entry_price'],
                'exit_price': row['close'] if row['close'] > position['entry_price'] else row['open'] - 2.5 * row['ATR']
            })
            position = None
        elif position and row.name.time() >= time(15, 25):
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
    return pd.DataFrame(trades)

# Run Backtest for Each Strategy
results = {}
for strategy, condition in strategies.items():
    results[strategy] = backtest_strategy(df, strategy, condition)

# Aggregate Results
all_trades = pd.concat(results.values())
summary = all_trades.groupby(['strategy', 'market_condition']).agg({
    'profit': ['sum', 'count', 'mean'],
    'entry_time': 'count'
}).reset_index()
summary.columns = ['Strategy', 'Market_Condition', 'Total_Profit', 'Trade_Count', 'Avg_Profit', 'Entry_Count']

# Print Results
for strategy in strategies.keys():
    strategy_df = results[strategy]
    if not strategy_df.empty:
        total_profit = strategy_df['profit'].sum()
        win_rate = len(strategy_df[strategy_df['profit'] > 0]) / len(strategy_df) * 100 if len(strategy_df) > 0 else 0
        print(f"\n{strategy} Results:")
        print(f"Total Profit: {total_profit:.2f} points")
        print(f"Number of Trades: {len(strategy_df)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Trades by Market Condition:")
        for cond in strategy_df['market_condition'].unique():
            cond_df = strategy_df[strategy_df['market_condition'] == cond]
            if not cond_df.empty:
                print(f"  {cond}: {len(cond_df)} trades, Profit: {cond_df['profit'].sum():.2f}, Avg Profit: {cond_df['profit'].mean():.2f}")
        print(f"Detailed Trades:\n{strategy_df[['entry_time', 'exit_time', 'market_condition', 'entry_price', 'exit_price', 'profit']]}")

print("\nSummary Table:")
print(summary)