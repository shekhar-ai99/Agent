import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def fetch_data(symbol, start_date, end_date, interval='5m'):
    """Fetch historical stock data from Yahoo Finance."""
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    return data

def calculate_indicators(data, ema_period=20, rsi_period=14):
    """Calculate EMA, RSI, and VWAP indicators."""
    # EMA
    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # VWAP
    data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['Cum_Volume'] = data['Volume'].cumsum()
    data['Cum_Volume_TP'] = (data['TP'] * data['Volume']).cumsum()
    data['VWAP'] = data['Cum_Volume_TP'] / data['Cum_Volume']
    
    return data

def define_strategy(data):
    """Define trading strategy based on EMA, RSI, and VWAP conditions."""
    data['Signal'] = 0
    data.loc[(data['Close'] > data['VWAP']) & (data['RSI'] > 45) & (data['Close'] > data['EMA']), 'Signal'] = 1
    data.loc[(data['Close'] < data['VWAP']) & (data['RSI'] < 55) & (data['Close'] < data['EMA']), 'Signal'] = -1
    return data

def backtest_strategy(data, initial_balance=10000):
    """Backtest the trading strategy."""
    balance = initial_balance
    position = 0
    for index, row in data.iterrows():
        if row['Signal'] == 1 and position == 0:
            position = balance / row['Close']
            balance = 0
            print(f"BUY at {row['Close']} on {index}")
        elif row['Signal'] == -1 and position > 0:
            balance = position * row['Close']
            position = 0
            print(f"SELL at {row['Close']} on {index}")
    # Final value
    balance += position * data.iloc[-1]['Close']
    print(f"Final Balance: {balance}")
    return balance

def plot_results(data):
    """Plot the stock price and indicators."""
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(data['EMA'], label='EMA', color='red')
    plt.plot(data['VWAP'], label='VWAP', color='green')
    plt.title('Stock Price with EMA and VWAP')
    plt.legend()
    plt.show()

# Parameters
symbol = 'AAPL'
start_date = '2025-03-01'
end_date = '2025-04-01'

# Fetch data
data = fetch_data(symbol, start_date, end_date)

# Calculate indicators
data = calculate_indicators(data)

# Define strategy
data = define_strategy(data)

# Backtest strategy
final_balance = backtest_strategy(data)

# Plot results
plot_results(data)
