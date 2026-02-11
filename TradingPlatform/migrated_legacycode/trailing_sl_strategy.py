import pandas as pd
import numpy as np
import logging
import uuid
from datetime import datetime

# Setup logging
logging.basicConfig(filename='trades.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

def calculate_sig(row, prev_row):
    """Define sig based on AlphaTrend and supertrend_dir."""
    try:
        if (row['AlphaTrend'] > row['AT_shifted_2'] and 
            row['supertrend_dir'] == 1.0):
            return 1
        elif (row['AlphaTrend'] < row['AT_shifted_2'] and 
              row['supertrend_dir'] == 0.0):
            return -1
    except KeyError:
        # Fallback to supertrend_dir if AlphaTrend unavailable
        return row['supertrend_dir']
    return 0

def validate_exit(row, exit_price):
    """Validate exit price against close, low, high."""
    if np.isclose(exit_price, row['close'], rtol=0.01) or \
       (row['low'] <= exit_price <= row['high']):
        return True
    logging.warning(f"Invalid exit price: {exit_price} not in "
                    f"[close={row['close']}, low={row['low']}, "
                    f"high={row['high']}]")
    return False

def backtest_strategy(df):
    """Backtest Trailing SL Strategy with tweaks."""
    trades = []
    position = None
    entry_price = 0
    trailing_sl = 0
    prev_sig = 0

    for i, row in df.iterrows():
        # Calculate sig
        sig = calculate_sig(row, df.iloc[i-1] if i > 0 else row)
        
        # ATR filter
        if row['atr'] < 15:
            sig = 0  # Skip low-volatility entries

        # Morning trade (09:15-09:30)
        if row['is_0915'] and position is None:
            position = 'Long'
            entry_price = row['open']
            trailing_sl = entry_price - 0.5 * row['atr']
            entry_time = row['datetime']
            logging.info(f"Entry: {position} at {entry_price}, "
                        f"Sig: {sig}, ATR: {row['atr']}, SL: {trailing_sl}")
            continue

        # Morning trade updates
        if position == 'Long' and row['datetime'].time() <= pd.Timestamp('09:30').time():
            # Update trailing SL
            trailing_sl = max(trailing_sl, row['close'] - 0.5 * row['atr'])
            
            # Check SL hit
            if row['low'] < trailing_sl:
                exit_price = trailing_sl
                if validate_exit(row, exit_price):
                    gross_pnl = (exit_price - entry_price) * 25
                    cost = 0.0002 * (entry_price + exit_price) * 25
                    trades.append({
                        'datetime': entry_time,
                        'action': 'Buy',
                        'price': entry_price,
                        'position': position,
                        'exit_datetime': row['datetime'],
                        'exit_price': exit_price,
                        'pnl_points': exit_price - entry_price,
                        'gross_pnl_inr': gross_pnl,
                        'transaction_cost_inr': cost,
                        'net_pnl_inr': gross_pnl - cost
                    })
                    logging.info(f"Exit: {position} at {exit_price}, "
                                f"PnL: {gross_pnl}, Cost: {cost}")
                    position = None
                continue
            
            # Check sig < 0 or 09:30
            if sig < 0 or row['datetime'].time() == pd.Timestamp('09:30').time():
                exit_price = row['close']
                if validate_exit(row, exit_price):
                    gross_pnl = (exit_price - entry_price) * 25
                    cost = 0.0002 * (entry_price + exit_price) * 25
                    trades.append({
                        'datetime': entry_time,
                        'action': 'Buy',
                        'price': entry_price,
                        'position': position,
                        'exit_datetime': row['datetime'],
                        'exit_price': exit_price,
                        'pnl_points': exit_price - entry_price,
                        'gross_pnl_inr': gross_pnl,
                        'transaction_cost_inr': cost,
                        'net_pnl_inr': gross_pnl - cost
                    })
                    logging.info(f"Exit: {position} at {exit_price}, "
                                f"PnL: {gross_pnl}, Cost: {cost}")
                    position = None
                continue

        # Post-09:30 trades
        if position is None:
            if sig > 0 and prev_sig <= 0 and row['atr'] >= 15:
                position = 'Long'
                entry_price = row['open']
                entry_time = row['datetime']
                logging.info(f"Entry: {position} at {entry_price}, "
                            f"Sig: {sig}, ATR: {row['atr']}")
            elif sig < 0 and prev_sig >= 0 and row['atr'] >= 15:
                position = 'Short'
                entry_price = row['open']
                entry_time = row['datetime']
                logging.info(f"Entry: {position} at {entry_price}, "
                            f"Sig: {sig}, ATR: {row['atr']}")

        # Exit conditions
        if position == 'Long' and (sig <= 0 or row['is_1525']):
            exit_price = row['close']
            if validate_exit(row, exit_price):
                gross_pnl = (exit_price - entry_price) * 25
                cost = 0.0002 * (entry_price + exit_price) * 25
                trades.append({
                    'datetime': entry_time,
                    'action': 'Buy',
                    'price': entry_price,
                    'position': position,
                    'exit_datetime': row['datetime'],
                    'exit_price': exit_price,
                    'pnl_points': exit_price - entry_price,
                    'gross_pnl_inr': gross_pnl,
                    'transaction_cost_inr': cost,
                    'net_pnl_inr': gross_pnl - cost
                })
                logging.info(f"Exit: {position} at {exit_price}, "
                            f"PnL: {gross_pnl}, Cost: {cost}")
                position = None

        if position == 'Short' and (sig >= 0 or row['is_1525']):
            exit_price = row['close']
            if validate_exit(row, exit_price):
                gross_pnl = (entry_price - exit_price) * 25
                cost = 0.0002 * (entry_price + exit_price) * 25
                trades.append({
                    'datetime': entry_time,
                    'action': 'Sell',
                    'price': entry_price,
                    'position': position,
                    'exit_datetime': row['datetime'],
                    'exit_price': exit_price,
                    'pnl_points': entry_price - exit_price,
                    'gross_pnl_inr': gross_pnl,
                    'transaction_cost_inr': cost,
                    'net_pnl_inr': gross_pnl - cost
                })
                logging.info(f"Exit: {position} at {exit_price}, "
                            f"PnL: {gross_pnl}, Cost: {cost}")
                position = None

        prev_sig = sig

    return trades

def main():
    # Load dataset
    df = pd.read_csv('result_nifty50_nse_alphatrend_5minute_20250430_143507.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['is_0915'] = df['is_0915'].astype(bool)
    df['is_1525'] = df['is_1525'].astype(bool)

    # Run backtest
    trades = backtest_strategy(df)

    # Save trades to CSV
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv('Trailing_SL_Strategy_Trades.csv', index=False)

    # Calculate performance
    total_points = trades_df['pnl_points'].sum()
    gross_pnl = trades_df['gross_pnl_inr'].sum()
    costs = trades_df['transaction_cost_inr'].sum()
    net_pnl = trades_df['net_pnl_inr'].sum()
    win_rate = len(trades_df[trades_df['net_pnl_inr'] > 0]) / len(trades_df) if trades_df.shape[0] > 0 else 0

    logging.info(f"Performance: {len(trades)} trades, {total_points:.2f} points, "
                 f"Gross PnL: {gross_pnl:.2f} INR, Costs: {costs:.2f} INR, "
                 f"Net PnL: {net_pnl:.2f} INR, Win Rate: {win_rate:.2%}")

if __name__ == "__main__":
    main()