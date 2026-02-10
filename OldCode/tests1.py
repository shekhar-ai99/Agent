import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import timedelta

class BacktestingEngine:
    def __init__(self, data_path='/Users/shekhar/Desktop/BOT/trading_bot_final/tests/dataset/nifty5minApr2025.csv', initial_capital=100000, risk_per_trade=0.01):
        self.df = pd.read_csv(data_path, parse_dates=['datetime'],index_col='datetime')
        #self.df.set_index('datetime', inplace=True)
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.df.index = self.df.index.tz_localize(None)
        self.prepare_indicators()
        
    def prepare_indicators(self):
        """Calculate all technical indicators needed for strategies"""
        df = self.df
        
        # Basic indicators
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], timeperiod=14)
        adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx_14'] = adx_data['ADX_14']
        df['rsi_14'] = ta.rsi(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = ta.macd(df['close'])
        
        # Ichimoku Cloud
        df['tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
        df['kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
        df['senkou_b'] = (df['high'].rolling(52).max() + df['low'].rolling(52).min()).shift(26)
        df['cloud_thickness'] = ((df['senkou_a'] - df['senkou_b']) / df['close']) * 100
        
        # EMA
        df['ema9'] = ta.ema(df['close'], timeperiod=9)
        df['ema21'] = ta.ema(df['close'], timeperiod=21)
        df['ema50'] = ta.ema(df['close'], timeperiod=50)
        df['ema200'] = ta.ema(df['close'], timeperiod=200)
        
        # Bollinger Bands
        bbands_data = ta.bbands(df['close'], length=20)
        df['upper_bb'] = bbands_data['BBU_20_2.0']  # Upper Band
        df['middle_bb'] = bbands_data['BBM_20_2.0']  # Middle Band
        df['lower_bb'] = bbands_data['BBL_20_2.0']   # Lower Band  df['bandwidth'] = (df['upper_bb'] - df['lower_bb']) / df['middle_bb']
        df['bandwidth'] = (df['upper_bb'] - df['lower_bb']) / df['middle_bb']        
        # VWAP
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # SuperTrend (for AlphaTrend)
        df['basic_ub'] = (df['high'] + df['low']) / 2 + 2 * df['atr_14']
        df['basic_lb'] = (df['high'] + df['low']) / 2 - 2 * df['atr_14']
        df['supertrend'] = self.calculate_supertrend(df)
        
        # RSI Divergence
        df['rsi_divergence'] = self.calculate_rsi_divergence(df)
        
    def calculate_supertrend(self, df, period=14, multiplier=2):
        """Calculate SuperTrend indicator"""
        hl2 = (df['high'] + df['low']) / 2
        matr = multiplier * df['atr_14']
        
        upperband = hl2 + matr
        lowerband = hl2 - matr
        
        supertrend = pd.Series(index=df.index)
        direction = pd.Series(index=df.index)
        
        supertrend.iloc[0] = upperband.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(df)):
            if df['close'].iloc[i-1] <= supertrend.iloc[i-1]:
                supertrend.iloc[i] = min(upperband.iloc[i], supertrend.iloc[i-1])
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = max(lowerband.iloc[i], supertrend.iloc[i-1])
                direction.iloc[i] = -1
        
        return supertrend * direction
    
    def calculate_rsi_divergence(self, df, period=14):
        """Calculate RSI divergence patterns"""
        rsi = df['rsi_14']
        close = df['close']
        
        # Find peaks and troughs in price and RSI
        price_peaks = (close.shift(1) < close) & (close.shift(-1) < close)
        price_troughs = (close.shift(1) > close) & (close.shift(-1) > close)
        rsi_peaks = (rsi.shift(1) < rsi) & (rsi.shift(-1) < rsi)
        rsi_troughs = (rsi.shift(1) > rsi) & (rsi.shift(-1) > rsi)
        
        # Regular bullish divergence (price makes lower low, RSI makes higher low)
        reg_bullish = (price_troughs & price_troughs.shift(1) & 
                      (close < close.shift(1)) & (rsi_troughs & rsi_troughs.shift(1) & 
                      (rsi > rsi.shift(1))))
        
        # Regular bearish divergence (price makes higher high, RSI makes lower high)
        reg_bearish = (price_peaks & price_peaks.shift(1) & 
                      (close > close.shift(1)) & (rsi_peaks & rsi_peaks.shift(1) & 
                      (rsi < rsi.shift(1))))
        
        divergence = pd.Series(0, index=df.index)
        divergence[reg_bullish] = 1
        divergence[reg_bearish] = -1
        
        return divergence
    
    def orb_strategy(self, df):
        """Opening Range Breakout strategy"""
        morning = df.between_time('09:15', '09:30')
        if len(morning) == 0:
            return pd.Series(False, index=df.index), pd.Series(False, index=df.index)
        
        high, low = morning['high'].max(), morning['low'].min()
        
        long_cond = (df['close'] > high) & (df['volume'] > 1.2 * df['volume'].rolling(50).mean())
        short_cond = (df['close'] < low) & (df['volume'] > 1.2 * df['volume'].rolling(50).mean())
        
        return long_cond, short_cond
    
    def ema_strategy(self, df):
        """EMA crossover strategy"""
        long_cond = (df['ema9'] > df['ema21']) & (df['ema9'].shift(1) <= df['ema21'].shift(1))
        short_cond = (df['ema9'] < df['ema21']) & (df['ema9'].shift(1) >= df['ema21'].shift(1))
        
        return long_cond, short_cond
    
    def bollinger_strategy(self, df):
        """Improved Bollinger Bands strategy"""
        # Add confirmation filters
        long_cond = (
            (df['close'] < df['lower_bb']) & 
            (df['bandwidth'].rolling(5).min() < 0.5) &
            (df['rsi_14'] < 30) &  # Oversold condition
            (df['volume'] > df['volume'].rolling(20).mean())  # Volume confirmation
        )
        
        short_cond = (
            (df['close'] > df['upper_bb']) & 
            (df['bandwidth'].rolling(5).min() < 0.5) &
            (df['rsi_14'] > 70) &  # Overbought condition
            (df['volume'] > df['volume'].rolling(20).mean())
        )
        
        return long_cond, short_cond 
    def vwap_strategy(self, df):
        """VWAP strategy"""
        long_cond = (df['close'] < df['vwap']) & (df['rsi_14'] < 30) & (df['close'] >= df['senkou_b'])
        short_cond = (df['close'] > df['vwap']) & (df['rsi_14'] > 70) & (df['close'] <= df['senkou_a'])
        
        return long_cond, short_cond
    
    def alphatrend_strategy(self, df):
        """Custom AlphaTrend strategy combining multiple indicators"""
        # SuperTrend component
        supertrend_long = df['supertrend'] > 0
        supertrend_short = df['supertrend'] < 0
        
        # RSI Divergence component
        rsi_div_long = df['rsi_divergence'] > 0
        rsi_div_short = df['rsi_divergence'] < 0
        
        # MACD component
        macd_long = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        macd_short = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # VWAP component
        vwap_long = df['close'] > df['vwap']
        vwap_short = df['close'] < df['vwap']
        
        # EMA component
        ema_long = (df['close'] > df['ema50']) & (df['ema50'] > df['ema200'])
        ema_short = (df['close'] < df['ema50']) & (df['ema50'] < df['ema200'])
        
        # Bollinger Bands component
        bb_long = (df['close'] < df['lower_bb']) & (df['bandwidth'] > 0.5)
        bb_short = (df['close'] > df['upper_bb']) & (df['bandwidth'] > 0.5)
        
        # Combined conditions
        long_cond = (supertrend_long & (rsi_div_long | macd_long) & 
                    (vwap_long | ema_long) & (bb_long | (df['rsi_14'] < 40)))
        
        short_cond = (supertrend_short & (rsi_div_short | macd_short) & 
                    (vwap_short | ema_short) & (bb_short | (df['rsi_14'] > 60)))
        
        return long_cond, short_cond
    
  
 
    def market_condition(self, row):
        # Initialize default values
        trend = "Neutral"
        volatility = "Medium"
        
        # For DataFrame-wide operations
        if isinstance(row, pd.DataFrame):
            atr_ma = row['atr_14'].rolling(50).mean()
            volatility = np.where(row['atr_14'] < atr_ma, "Low", "High")
            trend = np.where(
                (row['close'] > row['senkou_a']) & (row['tenkan'] > row['kijun']),
                "Bullish",
                np.where(
                    (row['close'] < row['senkou_b']) & (row['tenkan'] < row['kijun']),
                    "Bearish",
                    "Neutral"
                )
            )
            return trend, volatility
        
        # For single row operations
        current_idx = self.df.index.get_loc(row.name)
        
        # Volatility calculation
        if current_idx >= 50:
            atr_ma = self.df['atr_14'].iloc[current_idx-50:current_idx].mean()
            volatility = "Low" if row['atr_14'] < atr_ma else "High"
        
        # Trend determination
        if (row['close'] > row['senkou_a']) and (row['tenkan'] > row['kijun']):
            trend = "Bullish"
        elif (row['close'] < row['senkou_b']) and (row['tenkan'] < row['kijun']):
            trend = "Bearish"
        
        return trend, volatility 
    def calculate_position_size(self, row):
  
        risk_amount = self.initial_capital * self.risk_per_trade
        
        # Dynamic ATR-based stop loss
        atr_multiplier = 1.5 if row['volatility'] == "High" else 2.0
        atr_adjusted = max(row['atr_14'], row['close']*0.01) * atr_multiplier
        
        size = risk_amount / atr_adjusted
        max_size = self.initial_capital / row['close'] * 0.1  # Max 10% of capital
        
        # Reduce size in high volatility
        if row['volatility'] == "High":
            size = size * 0.7
            
        return min(size, max_size)
    def run_backtest(self):
        """Run complete backtest with all strategies"""
        trades = []
        df = self.df.copy()
        
        # Calculate all strategy signals
        orb_long, orb_short = self.orb_strategy(df)
        ema_long, ema_short = self.ema_strategy(df)
        bb_long, bb_short = self.bollinger_strategy(df)
        vwap_long, vwap_short = self.vwap_strategy(df)
        at_long, at_short = self.alphatrend_strategy(df)
        
        for i, row in df.iterrows():
            trend, vol = self.market_condition(row)  # This must come first
            size = self.calculate_position_size(row)
            
            debug_info = {
                'Time': i,
                'Close': row['close'],
                'Trend': trend,
                'Volatility': vol,
                'BB_Long': bb_long.loc[i],
                'EMA_Cross': ema_long.loc[i] | ema_short.loc[i]
            }
            print(", ".join(f"{k}: {v}" for k,v in debug_info.items()))
            # Strategy Activation Rules
            if trend == "Bullish":
                if orb_long.loc[i]: trades.append(('ORB Long', i, row['close'], size))
                if ema_long.loc[i]: trades.append(('EMA Long', i, row['close'], size))
                if at_long.loc[i]: trades.append(('AlphaTrend Long', i, row['close'], size * 1.5))
            elif trend == "Bearish":
                if orb_short.loc[i]: trades.append(('ORB Short', i, row['close'], size))
                if ema_short.loc[i]: trades.append(('EMA Short', i, row['close'], size))
                if at_short.loc[i]: trades.append(('AlphaTrend Short', i, row['close'], size * 1.5))
            
            if vol == "High":
                if bb_long.loc[i]: trades.append(('BB Long', i, row['close'], size * 1.5))
                if bb_short.loc[i]: trades.append(('BB Short', i, row['close'], size * 1.5))
            
            if trend == "Neutral":
                if vwap_long.loc[i]: trades.append(('VWAP Long', i, row['close'], size * 0.75))
                if vwap_short.loc[i]: trades.append(('VWAP Short', i, row['close'], size * 0.75))
        
        # Create trades dataframe
        # In run_backtest():
        trades_df = pd.DataFrame(trades, columns=['Strategy', 'Time', 'Entry', 'Size'])
        trades_df['Time'] = pd.to_datetime(trades_df['Time'])
        trades_df['Exit'] = 0.0
        trades_df['Exit_Time'] = pd.NaT
        trades_df['Exit_Time'] = trades_df['Exit_Time'].astype('datetime64[ns]')
        trades_df['PnL'] = 0.0
        trades_df['Duration'] = pd.Timedelta(0)
        trades_df['Status'] = 'Open'
        
        # Calculate exits and P&L
        trades_df = self.calculate_exits(trades_df, df)
        
        return trades_df
    def generate_report(self, trades_df):
        """Generate performance report"""
        if trades_df.empty:
            return "No trades were generated during this period."
        
        report = {
            'total_trades': len(trades_df),
            'winning_trades': len(trades_df[trades_df['PnL'] > 0]),
            'losing_trades': len(trades_df[trades_df['PnL'] <= 0]),
            'win_rate': len(trades_df[trades_df['PnL'] > 0]) / len(trades_df),
            'total_pnl': trades_df['PnL'].sum(),
            'avg_pnl': trades_df['PnL'].mean(),
            'max_pnl': trades_df['PnL'].max(),
            'min_pnl': trades_df['PnL'].min(),
            'avg_trade_duration': trades_df['Duration'].mean(),
            'strategy_breakdown': trades_df.groupby('Strategy')['PnL'].sum().to_dict()
        }
        
        return report
    def calculate_exits(self, trades_df, price_df):
        """Improved exit calculation with trailing stops and dynamic risk management"""
        if trades_df.empty:
            return pd.DataFrame()
        
        # Calculate ATR moving average for volatility assessment
        price_df['atr_ma_50'] = price_df['atr_14'].rolling(50).mean()
        
        # Initialize columns with proper dtypes
        trades_df['Exit'] = 0.0
        trades_df['Exit_Time'] = pd.NaT
        trades_df['Exit_Time'] = trades_df['Exit_Time'].astype('datetime64[ns]')
        trades_df['PnL'] = 0.0
        trades_df['Duration'] = pd.Timedelta(0)
        trades_df['Status'] = 'Open'
        trades_df['Exit_Reason'] = 'No Exit'
        
        for i, trade in trades_df.iterrows():
            entry_time = trade['Time']
            entry_price = trade['Entry']
            strategy = trade['Strategy']
            position_size = trade['Size']
            
            # Get current row data
            row = price_df.loc[entry_time]
            atr = row['atr_14']
            
            # Determine volatility level based on ATR vs its moving average
            is_high_vol = row['atr_14'] > row.get('atr_ma_50', row['atr_14'] * 1.5)
            
            # Dynamic ATR multipliers
            sl_multiplier = 1.5 if is_high_vol else 2.0
            tp_multiplier = 2.5 if is_high_vol else 3.0
            
            # Find subsequent prices
            subsequent_prices = price_df.loc[entry_time + timedelta(minutes=5):]
            
            exit_price = 0
            exit_time = pd.NaT
            reason = 'No Exit'
            
            if 'Long' in strategy:
                initial_sl = entry_price - (sl_multiplier * atr)
                initial_tp = entry_price + (tp_multiplier * atr)
                trailing_sl = initial_sl
                highest_price = entry_price
                
                for time, row in subsequent_prices.iterrows():
                    current_low = row['low']
                    current_high = row['high']
                    current_close = row['close']
                    
                    # Update trailing stop
                    if current_high > highest_price:
                        highest_price = current_high
                        if highest_price > entry_price + atr:
                            trailing_sl = max(trailing_sl, highest_price - atr)
                    
                    # Check exit conditions
                    if current_low <= trailing_sl:
                        exit_price = max(trailing_sl, current_low)
                        exit_time = time
                        reason = 'Trailing SL Hit'
                        break
                    elif current_high >= initial_tp:
                        exit_price = initial_tp
                        exit_time = time
                        reason = 'TP Hit'
                        break
                    elif time == subsequent_prices.index[-1]:
                        exit_price = current_close
                        exit_time = time
                        reason = 'EOD Exit'
                        break
                        
            else:  # Short position
                initial_sl = entry_price + (sl_multiplier * atr)
                initial_tp = entry_price - (tp_multiplier * atr)
                trailing_sl = initial_sl
                lowest_price = entry_price
                
                for time, row in subsequent_prices.iterrows():
                    current_low = row['low']
                    current_high = row['high']
                    current_close = row['close']
                    
                    # Update trailing stop
                    if current_low < lowest_price:
                        lowest_price = current_low
                        if lowest_price < entry_price - atr:
                            trailing_sl = min(trailing_sl, lowest_price + atr)
                    
                    # Check exit conditions
                    if current_high >= trailing_sl:
                        exit_price = min(trailing_sl, current_high)
                        exit_time = time
                        reason = 'Trailing SL Hit'
                        break
                    elif current_low <= initial_tp:
                        exit_price = initial_tp
                        exit_time = time
                        reason = 'TP Hit'
                        break
                    elif time == subsequent_prices.index[-1]:
                        exit_price = current_close
                        exit_time = time
                        reason = 'EOD Exit'
                        break
            
            # Update trade info
            trades_df.at[i, 'Exit'] = exit_price
            trades_df.at[i, 'Exit_Time'] = exit_time
            trades_df.at[i, 'Duration'] = exit_time - entry_time if pd.notna(exit_time) else pd.Timedelta(0)
            trades_df.at[i, 'Exit_Reason'] = reason
            trades_df.at[i, 'Status'] = 'Closed'
            
            # Calculate PnL
            if 'Long' in strategy:
                trades_df.at[i, 'PnL'] = (exit_price - entry_price) * position_size
            else:
                trades_df.at[i, 'PnL'] = (entry_price - exit_price) * position_size
        
        return trades_df
if __name__ == "__main__":
    backtester = BacktestingEngine(data_path='/Users/shekhar/Desktop/BOT/trading_bot_final/tests/dataset/nifty5minApr2025.csv')
    trades = backtester.run_backtest()
    report = backtester.generate_report(trades)
    
    print("Backtesting Results:")
    print(f"Total Trades: {report['total_trades']}")
    print(f"Win Rate: {report['win_rate']:.2%}")
    print(f"Total P&L: {report['total_pnl']:.2f}")
    print(f"Average P&L per trade: {report['avg_pnl']:.2f}")
    print("\nStrategy Breakdown:")
    for strategy, pnl in report['strategy_breakdown'].items():
        print(f"{strategy}: {pnl:.2f}")