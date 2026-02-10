# File: trade_management.py
import pandas as pd
import os

data_folder = "/mnt/data/"
input_file = os.path.join(data_folder, "nifty_sentiment.csv")
output_file = os.path.join(data_folder, "nifty_trade_management.csv")

def manage_trades(df):
    """Update trade management data"""
    df['entry_price'] = df['close']
    df['target_price'] = df['close'] * 1.02  # 2% profit target
    df['stop_loss'] = df['close'] * 0.98  # 2% stop loss
    df['exit_price'] = None  # To be updated dynamically
    df['status'] = 'Open'
    
    return df

if __name__ == "__main__":
    df = pd.read_csv(input_file, parse_dates=['datetime'], index_col='datetime')
    df = manage_trades(df)
    df.to_csv(output_file)
    print(f"Trade management data saved to {output_file}")
