import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt

def load_data(filepath):
    print("\nLoading data...")
    try:
        df = pd.read_csv(filepath, parse_dates=['datetime'])
        df = df.set_index('datetime')
        
        required = ['open', 'high', 'low', 'close', 'SUPERTd_10_3.0', 'atr', 'rsi', 'macd', 'macd_signal']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        df['SUPERTd_10_3.0'] = df['SUPERTd_10_3.0'].fillna(0)
        df['atr'] = df['atr'].ffill()
        
        print("Data loaded successfully.")
        print(f"Time range: {df.index.min()} to {df.index.max()}")
        print(f"SuperTrend signals:\n{df['SUPERTd_10_3.0'].value_counts()}")
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

class AlphaTrendStrategy(Strategy):
    def init(self):
        # Initialize RSI, MACD, SuperTrend for strategy
        self.rsi = self.I(lambda: self.data.df['rsi'])
        self.macd = self.I(lambda: self.data.df['macd'])
        self.macd_signal = self.I(lambda: self.data.df['macd_signal'])
        self.supertrend = self.I(lambda: self.data.df['SUPERTd_10_3.0'])
        
    def calculate_atr(self):
        # Calculate True Range (TR) manually without shift()
        high_low = self.data.High - self.data.Low
        high_close = np.abs(self.data.High - self.data.Close)
        low_close = np.abs(self.data.Low - self.data.Close)
        
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        
        # Calculate ATR (14-period rolling average)
        atr = np.zeros(len(tr))
        for i in range(14, len(tr)):
            atr[i] = np.mean(tr[i-14:i])
        
        return atr
    
    def calculate_alpha_trend(self, atr):
        # AlphaTrend calculation with ATR weighting
        atr_weight = atr[-1] / self.data.Close[-1]  # Normalize volatility
        return 0.65 * self.data.Close[-1] + 0.25 * self.data.Close[-2] + 0.1 * self.rsi[-1] * atr_weight
    
    def next(self):
        if len(self.data) < 2:
            return
        
        # Calculate ATR for the current bar
        atr = self.calculate_atr()

        # Calculate AlphaTrend for the current bar
        alpha_trend = self.calculate_alpha_trend(atr)

        # Debugging: Print key variables for each row
        print(f"\nTime: {self.data.index[-1]}")
        print(f"Close: {self.data.Close[-1]}")
        print(f"ATR: {atr[-1]}")
        print(f"AlphaTrend: {alpha_trend}")
        print(f"RSI: {self.rsi[-1]}")
        print(f"MACD: {self.macd[-1]}, MACD Signal: {self.macd_signal[-1]}")
        print(f"SuperTrend: {self.supertrend[-1]}")
        
        # Simplified entry condition for debugging
        entry_score = (
            int(self.data.Close[-1] > alpha_trend) +  # If Close > AlphaTrend
            int(self.rsi[-1] > 50) +  # If RSI > 50
            int(self.macd[-1] > self.macd_signal[-1]) +  # If MACD > MACD Signal
            int(self.supertrend[-1] == 1)  # If SuperTrend is 1 (uptrend)
        )

        # Debugging: Print entry score
        print(f"Entry Score: {entry_score}")

        # ATR-based session-specific scaling
        if self.data.index[-1].time() < pd.to_datetime("09:45").time():
            atr_multiplier = 3.0 if atr[-1] > 18 else 2.2
        else:
            atr_multiplier = 2.5
        
        # Simplify entry condition to trigger more trades
        if entry_score >= 2 and atr[-1] > 12:  # Reduce threshold for entry score
            if self.supertrend[-1] == 1 and self.supertrend[-2] != 1:
                stop_loss = self.data.Close[-1] - atr_multiplier * atr[-1]
                take_profit = self.data.Close[-1] + atr_multiplier * 2.5 * atr[-1]
                print(f"Buying: Stop Loss: {stop_loss}, Take Profit: {take_profit}")
                self.buy(sl=stop_loss, tp=take_profit)
        
        # Sell conditions
        if entry_score >= 2 and atr[-1] > 12:  # Reduce threshold for entry score
            if self.supertrend[-1] == -1 and self.supertrend[-2] != -1:
                stop_loss = self.data.Close[-1] + atr_multiplier * atr[-1]
                take_profit = self.data.Close[-1] - atr_multiplier * 2.5 * atr[-1]
                print(f"Selling: Stop Loss: {stop_loss}, Take Profit: {take_profit}")
                self.sell(sl=stop_loss, tp=take_profit)

def run_backtest(df):
    print("\nRunning backtest...")
    bt = Backtest(
        df,
        AlphaTrendStrategy,
        commission=0.0002,
        margin=0.05,
        trade_on_close=True,
        exclusive_orders=True,
        cash=1_000_000
    )
    
    stats = bt.run()
    return bt, stats

def analyze_results(bt, stats, original_df):
    print("\nAnalyzing results...")
    
    print("\n=== PERFORMANCE SUMMARY ===")
    print(f"Start Date: {stats['Start']}")
    print(f"End Date: {stats['End']}")
    print(f"Duration: {stats['Duration']}")
    print(f"Return: {stats['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    
    if '_trades' in stats and not stats['_trades'].empty:
        trades = stats['_trades']
        print("\n=== TRADE DETAILS ===")
        print(f"Total Trades: {len(trades)}")
        print(trades[['EntryTime', 'ExitTime', 'EntryPrice', 'ExitPrice', 'PnL']])
        
        # Plot trades using original_df since we can't access bt.data
        buys = trades[trades['Size'] > 0]
        sells = trades[trades['Size'] < 0]
        
        plt.figure(figsize=(15, 7))
        plt.plot(original_df.index, original_df['Close'], label='Price', alpha=0.5)
        plt.scatter(buys['EntryTime'], buys['EntryPrice'], marker='^', color='g', label='Buy')
        plt.scatter(sells['EntryTime'], sells['EntryPrice'], marker='v', color='r', label='Sell')
        plt.title('Trade Entries')
        plt.legend()
        plt.show()
    else:
        print("\nNo trades were executed during this period")

if __name__ == "__main__":
    try:
        data_file = 'results/result_nifty50_nse_alphatrend_5minute_20250430_143507.csv'
        df = load_data(data_file)
        
        bt, stats = run_backtest(df)
        analyze_results(bt, stats, df)  # Pass original df for plotting
        
        # Plot using the library's built-in method
        try:
            bt.plot()
        except Exception as e:
            print(f"\nWarning: Could not generate backtest plot - {str(e)}")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
