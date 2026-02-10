import pandas as pd
import numpy as np
import talib

# Load 5-min OHLCV data
df = pd.read_csv('nifty_5min.csv', parse_dates=['datetime'])
df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
df['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)


# Ichimoku Components
df['tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
df['kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
df['senkou_b'] = (df['high'].rolling(52).max() + df['low'].rolling(52).min()).shift(26)
df['cloud_thickness'] = ((df['senkou_a'] - df['senkou_b']) / df['close']) * 100


def orb_strategy(df):
    # Get 09:15-09:30 range
    morning = df.between_time('09:15', '09:30')
    high, low = morning['high'].max(), morning['low'].min()
    
    # Conditions
    long_cond = (df['close'] > high) & (df['volume'] > 1.2 * df['volume'].rolling(50).mean())
    short_cond = (df['close'] < low) & (df['volume'] > 1.2 * df['volume'].rolling(50).mean())
    
    return long_cond, short_cond

def ema_strategy(df):
        df['ema9'] = talib.EMA(df['close'], timeperiod=9)
        df['ema21'] = talib.EMA(df['close'], timeperiod=21)
        
        long_cond = (df['ema9'] > df['ema21']) & (df['ema9'].shift(1) <= df['ema21'].shift(1))
        short_cond = (df['ema9'] < df['ema21']) & (df['ema9'].shift(1) >= df['ema21'].shift(1))
        
        return long_cond, short_cond
        def bollinger_strategy(df):
            df['upper_bb'], df['middle_bb'], df['lower_bb'] = talib.BBANDS(df['close'], timeperiod=20)
            df['bandwidth'] = (df['upper_bb'] - df['lower_bb']) / df['middle_bb']
            
            long_cond = (df['close'] > df['upper_bb']) & (df['bandwidth'].rolling(5).min() < 0.5)
            short_cond = (df['close'] < df['lower_bb']) & (df['bandwidth'].rolling(5).min() < 0.5)
            
            return long_cond, short_cond

            def vwap_strategy(df):
                df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
                df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
                
                long_cond = (df['close'] < df['vwap']) & (df['rsi_14'] < 30) & (df['close'] >= df['senkou_b'])
                short_cond = (df['close'] > df['vwap']) & (df['rsi_14'] > 70) & (df['close'] <= df['senkou_a'])
                
                return long_cond, short_cond
                def market_condition(df):
                    # Trend Direction
                    trend = "Neutral"
                    if (df['close'] > df['senkou_a']) & (df['tenkan'] > df['kijun']):
                        trend = "Bullish"
                    elif (df['close'] < df['senkou_b']) & (df['tenkan'] < df['kijun']):
                        trend = "Bearish"
                    
                    # Volatility Regime
                    volatility = "Low" if df['atr_14'] < df['atr_14'].rolling(50).mean() else "High"
                    
                    return trend, volatility

                    def calculate_position_size(df, account_risk=0.01, account_size=100000):
                        risk_per_trade = account_size * account_risk
                        base_size = risk_per_trade / (df['atr_14'] * 1.5)
                        
                        # Adjust for cloud thickness
                        if df['cloud_thickness'] > 1.0:
                            return base_size * 2
                        elif df['cloud_thickness'] < 0.5:
                            return base_size * 0.5
                        else:
                            return base_size

                            def run_strategy(df):
                                trades = []
                                
                                for i, row in df.iterrows():
                                    trend, vol = market_condition(row)
                                    size = calculate_position_size(row)
                                    
                                    # Get all strategy signals
                                    orb_long, orb_short = orb_strategy(row)
                                    ema_long, ema_short = ema_strategy(row)
                                    bb_long, bb_short = bollinger_strategy(row)
                                    vwap_long, vwap_short = vwap_strategy(row)
                                    
                                    # Strategy Activation Rules
                                    if trend == "Bullish":
                                        if orb_long: trades.append(('ORB Long', i, row['close'], size))
                                        if ema_long: trades.append(('EMA Long', i, row['close'], size))
                                    elif trend == "Bearish":
                                        if orb_short: trades.append(('ORB Short', i, row['close'], size))
                                        if ema_short: trades.append(('EMA Short', i, row['close'], size))
                                    
                                    if vol == "High" and bb_long: trades.append(('BB Long', i, row['close'], size*1.5))
                                    if trend == "Neutral" and vwap_short: trades.append(('VWAP Short', i, row['close'], size))
                                
                                return pd.DataFrame(trades, columns=['Strategy', 'Time', 'Entry', 'Size'])





                                def backtest(df):
                                    trades = run_strategy(df)
                                    results = []
                                    
                                    for trade in trades.itertuples():
                                        exit_cond = False
                                        exit_price = 0
                                        
                                        # Dynamic exits based on strategy
                                        if 'ORB' in trade.Strategy:
                                            exit_price = trade.Entry + (2 * df.at[trade.Time, 'atr_14'])
                                        elif 'EMA' in trade.Strategy:
                                            exit_price = df.loc[trade.Time:, 'close'].iloc[1:].max() if 'Long' in trade.Strategy else df.loc[trade.Time:, 'close'].iloc[1:].min()
                                        
                                        # Calculate PnL
                                        pnl = (exit_price - trade.Entry) * trade.Size if 'Long' in trade.Strategy else (trade.Entry - exit_price) * trade.Size
                                        results.append((trade.Strategy, trade.Time, trade.Entry, exit_price, pnl))
                                    
                                    return pd.DataFrame(results, columns=['Strategy', 'Time', 'Entry', 'Exit', 'PnL'])
                                

                                # Generate signals
signals = run_strategy(df)

# Backtest
results = backtest(df)

# Save results
results.to_csv('backtest_results.csv', index=False)