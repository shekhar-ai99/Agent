from collections import defaultdict
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Parameter Organization
PARAMS: Dict[str, Dict[str, Any]] = {
    'core': {
        "max_trade_duration": 50,
        'exit_score_drop_threshold': 1.5,
        'use_score_drop_exit': True,
        'entry_score_threshold': 6.0,
        'use_fib_bounce_entry': True,
        'use_fib_bounce_sell': True,
        'fib_bounce_lookback': 3,
        'fib_bounce_long_zone': (0.5, 0.618),
        'fib_bounce_short_zone': (0.382, 0.5),
        'fib_bounce_confirmation_level': 0.5,
        'use_ema_bounce_buy': True,
        'use_ema_bounce_sell': True,
        'ema_bounce_lookback': 2,
        'ema_bounce_source_str': "Fast EMA",
        'use_bb_mid_bounce_buy': True,
        'use_bb_mid_bounce_sell': True,
        'bb_bounce_lookback': 2,
        'use_vol_breakout_buy': True,
        'use_vol_breakout_sell': True,
        'trailing_stop_type': "atr",
        'trailing_stop_pct': 0.02,
        'trailing_stop_atr_multiplier': 1.5,
        'profit_protection_levels': {
            'level1': {'profit_pct': 0.05, 'new_atr_mult': 1.0},
            'level2': {'profit_pct': 0.10, 'new_atr_mult': 0.5}
        }
    },
    'ema': {
        'fast_len': 9,
        'med_len': 21,
        'slow_len': 50
    },
    'bollinger': {
        'bb_len': 20,
        'bb_std_dev': 2
    },
    'rsi': {
        'rsi_len': 14,
        'rsi_buy_level': 30,
        'rsi_sell_level': 70,
        'rsi_confirm_fib': True,
        'rsi_confirm_ema': True,
        'rsi_confirm_bb': True,
        'rsi_confirm_level_buy': 40,
        'rsi_confirm_level_sell': 60
    },
    'macd': {
        'macd_fast_len': 12,
        'macd_slow_len': 26,
        'macd_signal_len': 9
    },
    'volume': {
        'vol_ma_len': 20,
        'vol_multiplier': 1.5
    },
    'atr': {
        'atr_len': 14,
        'atr_mult': 2.0
    },
    'trend': {
        'use_adx_filter': True,
        'use_adx_direction_filter': True,
        'use_ema_trend_filter': True,
        'adx_len': 14,
        'adx_threshold': 25
    },
    'fibonacci': {
        'fib_pivot_lookback': 5,
        'fib_max_bars': 100,
        'fib_lookback_exit': 10,
        'fib_extension_level': 1.618,
        'use_fib_exit': True
    },
    'score_weights': {
        'w_ema_trend': 1.5,
        'w_ema_signal': 2.0,
        'w_rsi_thresh': 1.0,
        'w_macd_signal': 1.5,
        'w_macd_zero': 1.0,
        'w_vol_break': 1.0,
        'w_adx_strength': 1.0,
        'w_adx_direction': 0.5,
        'w_fib_bounce': 2.0,
        'w_ema_bounce': 1.5,
        'w_bb_bounce': 1.0
    },
    'backtest': {
        'slippage_pct': 0.0005,
        'commission_pct': 0.0005
    }
}



class StatsGenerator:
    """Generates performance statistics from trading signals"""
    
    def __init__(self):
        self.trade_history = []
        self.summary_stats = defaultdict(float)
        self.entry_stats = defaultdict(lambda: defaultdict(float))
        self.exit_stats = defaultdict(lambda: defaultdict(float))
    
    def analyze_trades(self, df: pd.DataFrame) -> Dict:
        """Analyze trades and generate performance statistics"""
        try:
            self._reset_stats()
            current_trade = None
            
            for i in range(len(df)):
                signal = df.iloc[i]['signal']
                
                # Entry signal found
                if signal in ['enter_long', 'enter_short'] and current_trade is None:
                    current_trade = {
                        'entry_index': i,
                        'entry_time': df.index[i],
                        'entry_price': df.iloc[i]['entry_price'],
                        'position': 'long' if signal == 'enter_long' else 'short',
                        'stop_loss': df.iloc[i]['stop_loss'],
                        'take_profit': df.iloc[i]['take_profit'],
                        'entry_condition': self._get_entry_condition(df, i)
                    }
                
                # Exit signal found
                elif current_trade and signal in ['exit_long', 'exit_short']:
                    if (current_trade['position'] == 'long' and signal == 'exit_long') or \
                       (current_trade['position'] == 'short' and signal == 'exit_short'):
                        
                        exit_price = df.iloc[i]['exit_price']
                        exit_reason = df.iloc[i]['exit_reason']
                        
                        # Calculate P&L
                        if current_trade['position'] == 'long':
                            pnl = exit_price - current_trade['entry_price']
                            pnl_pct = (pnl / current_trade['entry_price']) * 100
                        else:
                            pnl = current_trade['entry_price'] - exit_price
                            pnl_pct = (pnl / current_trade['entry_price']) * 100
                        
                        # Calculate trade duration
                        duration = (df.index[i] - current_trade['entry_time']).total_seconds() / (60 * 60 * 24)
                        
                        # Calculate risk/reward
                        if current_trade['position'] == 'long':
                            risk = current_trade['entry_price'] - current_trade['stop_loss']
                            reward = current_trade['take_profit'] - current_trade['entry_price']
                        else:
                            risk = current_trade['stop_loss'] - current_trade['entry_price']
                            reward = current_trade['entry_price'] - current_trade['take_profit']
                        
                        risk_reward = reward / risk if risk != 0 else np.nan
                        
                        trade_record = {
                            **current_trade,
                            'exit_time': df.index[i],
                            'exit_price': exit_price,
                            'exit_reason': exit_reason,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'duration': duration,
                            'risk': risk,
                            'reward': reward,
                            'risk_reward': risk_reward,
                            'outcome': 'win' if pnl > 0 else 'loss'
                        }
                        
                        self._update_stats(trade_record)
                        self.trade_history.append(trade_record)
                        current_trade = None
            
            self._calculate_summary_stats()
            return self._get_results()
            
        except Exception as e:
            logging.error(f"Error analyzing trades: {str(e)}", exc_info=True)
            raise
    
    def _get_entry_condition(self, df: pd.DataFrame, idx: int) -> str:
        """Determine what conditions triggered the entry"""
        conditions = []
        row = df.iloc[idx]
        
        if row['macd_crossover'] and row['position'] == 'long':
            conditions.append("MACD crossover")
        elif row['macd_crossunder'] and row['position'] == 'short':
            conditions.append("MACD crossunder")
        
        if row['ai_buy_signal'] and row['position'] == 'long':
            conditions.append("AI buy signal")
        elif row['ai_sell_signal'] and row['position'] == 'short':
            conditions.append("AI sell signal")
        
        if row['close'] > row['ema9'] and row['position'] == 'long':
            conditions.append("Price > EMA9")
        elif row['close'] < row['ema9'] and row['position'] == 'short':
            conditions.append("Price < EMA9")
        
        if row['close'] > row['ema50'] and row['position'] == 'long':
            conditions.append("Price > EMA50")
        elif row['close'] < row['ema50'] and row['position'] == 'short':
            conditions.append("Price < EMA50")
        
        if row['is_trending']:
            conditions.append("Trending market")
        
        return ", ".join(conditions)
    
    def _update_stats(self, trade: Dict) -> None:
        """Update statistics with trade data"""
        # Update summary stats
        self.summary_stats['total_trades'] += 1
        self.summary_stats['total_pnl'] += trade['pnl']
        
        if trade['outcome'] == 'win':
            self.summary_stats['winning_trades'] += 1
            self.summary_stats['total_win_pnl'] += trade['pnl']
        else:
            self.summary_stats['losing_trades'] += 1
            self.summary_stats['total_loss_pnl'] += trade['pnl']
        
        # Update entry stats
        pos = trade['position']
        self.entry_stats[pos]['total'] += 1
        self.entry_stats[pos]['total_pnl'] += trade['pnl']
        self.entry_stats[pos]['total_duration'] += trade['duration']
        
        if trade['outcome'] == 'win':
            self.entry_stats[pos]['wins'] += 1
        else:
            self.entry_stats[pos]['losses'] += 1
        
        # Update exit stats
        exit_reason = trade['exit_reason']
        self.exit_stats[exit_reason]['total'] += 1
        self.exit_stats[exit_reason]['total_pnl'] += trade['pnl']
        
        if trade['outcome'] == 'win':
            self.exit_stats[exit_reason]['wins'] += 1
        else:
            self.exit_stats[exit_reason]['losses'] += 1
    
    def _calculate_summary_stats(self) -> None:
        """Calculate derived statistics"""
        if self.summary_stats['total_trades'] > 0:
            # Win rate
            self.summary_stats['win_rate'] = (
                self.summary_stats['winning_trades'] / self.summary_stats['total_trades'] * 100
            )
            
            # Average P&L
            self.summary_stats['avg_pnl'] = (
                self.summary_stats['total_pnl'] / self.summary_stats['total_trades']
            )
            
            # Average win/loss
            if self.summary_stats['winning_trades'] > 0:
                self.summary_stats['avg_win'] = (
                    self.summary_stats['total_win_pnl'] / self.summary_stats['winning_trades']
                )
            if self.summary_stats['losing_trades'] > 0:
                self.summary_stats['avg_loss'] = (
                    self.summary_stats['total_loss_pnl'] / self.summary_stats['losing_trades']
                )
            
            # Profit factor
            if self.summary_stats['total_loss_pnl'] != 0:
                self.summary_stats['profit_factor'] = (
                    abs(self.summary_stats['total_win_pnl'] / self.summary_stats['total_loss_pnl'])
                )
            else:
                self.summary_stats['profit_factor'] = float('inf')
            
            # Expectancy
            self.summary_stats['expectancy'] = (
                (self.summary_stats['win_rate'] / 100 * self.summary_stats['avg_win']) +
                ((1 - self.summary_stats['win_rate'] / 100) * -self.summary_stats['avg_loss'])
            )
        
        # Calculate entry stats
        for pos in self.entry_stats:
            if self.entry_stats[pos]['total'] > 0:
                self.entry_stats[pos]['win_rate'] = (
                    self.entry_stats[pos]['wins'] / self.entry_stats[pos]['total'] * 100
                )
                self.entry_stats[pos]['avg_pnl'] = (
                    self.entry_stats[pos]['total_pnl'] / self.entry_stats[pos]['total']
                )
                self.entry_stats[pos]['avg_duration'] = (
                    self.entry_stats[pos]['total_duration'] / self.entry_stats[pos]['total']
                )
        
        # Calculate exit stats
        for reason in self.exit_stats:
            if self.exit_stats[reason]['total'] > 0:
                self.exit_stats[reason]['win_rate'] = (
                    self.exit_stats[reason]['wins'] / self.exit_stats[reason]['total'] * 100
                )
                self.exit_stats[reason]['avg_pnl'] = (
                    self.exit_stats[reason]['total_pnl'] / self.exit_stats[reason]['total']
                )
    
    def _reset_stats(self) -> None:
        """Reset all statistics"""
        self.trade_history = []
        self.summary_stats = defaultdict(float)
        self.entry_stats = defaultdict(lambda: defaultdict(float))
        self.exit_stats = defaultdict(lambda: defaultdict(float))
    
    def _get_results(self) -> Dict:
        """Return all results in a structured format"""
        return {
            'trade_history': self.trade_history,
            'summary_stats': dict(self.summary_stats),
            'entry_stats': {k: dict(v) for k, v in self.entry_stats.items()},
            'exit_stats': {k: dict(v) for k, v in self.exit_stats.items()}
        }
    
    def print_summary(self) -> None:
        """Print formatted summary of statistics"""
        if not self.summary_stats or self.summary_stats['total_trades'] == 0:
            print("No trades to summarize")
            return
        
        print("\n=== TRADING STRATEGY PERFORMANCE SUMMARY ===")
        print(f"Total Trades: {self.summary_stats['total_trades']}")
        print(f"Winning Trades: {self.summary_stats['winning_trades']} ({self.summary_stats['win_rate']:.1f}%)")
        print(f"Losing Trades: {self.summary_stats['losing_trades']}")
        print(f"Total P&L: {self.summary_stats['total_pnl']:.2f}")
        print(f"Average P&L per Trade: {self.summary_stats['avg_pnl']:.2f}")
        print(f"Average Winning Trade: {self.summary_stats['avg_win']:.2f}")
        print(f"Average Losing Trade: {self.summary_stats['avg_loss']:.2f}")
        print(f"Profit Factor: {self.summary_stats['profit_factor']:.2f}")
        print(f"Expectancy: {self.summary_stats['expectancy']:.2f}")
        
        print("\n=== ENTRY STATISTICS ===")
        for pos, stats in self.entry_stats.items():
            print(f"\n{pos.upper()} Positions:")
            print(f"  Total: {stats['total']}")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  Avg P&L: {stats['avg_pnl']:.2f}")
            print(f"  Avg Duration: {stats['avg_duration']:.2f} days")
        
        print("\n=== EXIT STATISTICS ===")
        for reason, stats in self.exit_stats.items():
            print(f"\nExit Reason: {reason}")
            print(f"  Total: {stats['total']}")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  Avg P&L: {stats['avg_pnl']:.2f}")

# Example usage:
# stats = StatsGenerator()
# results = stats.analyze_trades(df_with_signals)
# stats.print_summary()


# Configuration
DATA_FOLDER = Path("/Users/shekhar/Desktop/BOT/smartapi-python-main/runs/20250404_021909/data")
DEFAULT_INPUT_FILE = DATA_FOLDER / "nifty_indicators.csv"
DEFAULT_OUTPUT_FILE = DATA_FOLDER / "nifty_signals_final_enhanced23.csv"
DEFAULT_PLOT_FILE = DATA_FOLDER / "signals_plot_enhanced23.png"

class TradingStrategy:
    """Implements the PineScript trading strategy logic"""
    
    DEFAULT_PARAMS = {
        # Indicator Lengths
        'fast_ema_length': 12,
        'slow_ema_length': 26,
        'signal_length': 9,
        'rsi_period': 14,
        'ema9_length': 9,
        'ema14_length': 14,
        'ema50_length': 50,
        'dmi_length': 14,
        'adx_smoothing': 14,
        'atr_length': 14,
        
        # AI Confidence Parameters
        'w_trend': 1.0,
        'w_rsi': 1.0,
        'ai_buy_threshold': 0.2,
        'ai_sell_threshold': -0.2,
        
        # Exit Condition Parameters
        'rsi_exit_long': 45,
        'rsi_exit_short': 60,
        
        # Trade Management
        'sl_multiplier': 1.5,
        'tp_multiplier': 2.0,
        'adx_threshold': 20
    }
    
    def __init__(self, params: Optional[Dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
    
    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate required columns exist in the dataframe"""
        column_mapping = {
        'fast_ema': 'ema_9',       # Using ema_9 as fast EMA
        'slow_ema': 'ema_21',      # Using ema_21 as slow EMA
        'ema9': 'ema_9',
        'ema14': 'ema_21',         # Using ema_21 as ema14 since you don't have ema14
        'ema50': 'ema_50',
        'plus_di': None,           # These DMI indicators aren't in your CSV
        'minus_di': None,
        'adx': None
        }
    
    # Check for required columns that we can map
        missing = []
        for expected_col, actual_col in column_mapping.items():
            if actual_col is None:
                missing.append(expected_col)
            elif actual_col not in df.columns:
                missing.append(expected_col)
        
        if missing:
            logger.warning(f"Missing required columns: {missing}. Some strategy features may be disabled.")
            
    # Check for absolutely essential columns
        essential_columns = ['open', 'high', 'low', 'close', 'volume', 
                            'macd', 'macd_signal', 'macd_hist', 'rsi']
        essential_missing = [col for col in essential_columns if col not in df.columns]
        if essential_missing:
            raise ValueError(f"Missing essential columns: {essential_missing}")
        required_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'fast_ema', 'slow_ema', 'macd', 'macd_signal', 'macd_hist',
            'rsi', 'ema9', 'ema14', 'ema50', 'plus_di', 'minus_di', 'adx', 'atr'
        ]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def calculate_ai_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate AI confidence score"""
        df['trend_factor'] = df['macd_hist'].rolling(5).mean()
        df['rsi_factor'] = (df['rsi'] - 50) / 10
        df['ai_confidence'] = (
            (self.params['w_trend'] * df['trend_factor'] + 
            self.params['w_rsi'] * df['rsi_factor']
        ) / (self.params['w_trend'] + self.params['w_rsi']))
        
        df['ai_buy_signal'] = df['ai_confidence'] > self.params['ai_buy_threshold']
        df['ai_sell_signal'] = df['ai_confidence'] < self.params['ai_sell_threshold']
        
        return df
    
    def calculate_macd_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD crossover/under signals"""
        df['macd_crossover'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift())
        df['macd_crossunder'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift() >= df['macd_signal'].shift())
        return df
    
    def calculate_trend_condition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate if market is trending based on ADX"""
        df['is_trending'] = df['adx'] > self.params['adx_threshold']
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on strategy rules"""
        try:
            # Validate input data first
            self.validate_data(df)
            
            # Make a copy to avoid SettingWithCopyWarning
            df = df.copy()
            if 'plus_di' not in df.columns:
                df['plus_di'] = 0  # Default value that won't trigger signals
                if 'minus_di' not in df.columns:
                        df['minus_di'] = 0
                if 'adx' not in df.columns:
                        df['adx'] = 0
                        df['is_trending'] = True  # Assume trending if we don't have ADX
                else:
                        df['is_trending'] = df['adx'] > self.params['adx_threshold']
            # Calculate additional required signals
            df['fast_ema'] = df['ema_9']
            df['slow_ema'] = df['ema_21']
            df['ema9'] = df['ema_9']
            df['ema14'] = df['ema_21']  # Using ema_21 as proxy for ema14
            df['ema50'] = df['ema_50']
            
            df = self.calculate_ai_confidence(df)
            df = self.calculate_macd_signals(df)
            df = self.calculate_trend_condition(df)
            
            # Initialize signal columns
            df['signal'] = 'hold'
            df['position'] = ''
            df['entry_price'] = np.nan
            df['stop_loss'] = np.nan
            df['take_profit'] = np.nan
            df['exit_price'] = np.nan
            df['exit_reason'] = ''
            
            # Track current position
            current_position = None
            entry_price = None
            stop_loss = None
            take_profit = None
            
            for i in range(1, len(df)):
                # Exit conditions
                if current_position == 'long':
                    # Check for stop loss/take profit
                    if df.at[df.index[i], 'low'] <= stop_loss:
                        df.at[df.index[i], 'signal'] = 'exit_long'
                        df.at[df.index[i], 'exit_price'] = stop_loss
                        df.at[df.index[i], 'exit_reason'] = 'stop_loss'
                        current_position = None
                    elif df.at[df.index[i], 'high'] >= take_profit:
                        df.at[df.index[i], 'signal'] = 'exit_long'
                        df.at[df.index[i], 'exit_price'] = take_profit
                        df.at[df.index[i], 'exit_reason'] = 'take_profit'
                        current_position = None
                    # Check for conditional exits
                    elif ((df.at[df.index[i], 'rsi'] < self.params['rsi_exit_long'] and 
                          df.at[df.index[i], 'close'] < df.at[df.index[i], 'ema9']) or
                          df.at[df.index[i], 'macd_crossunder']):
                        df.at[df.index[i], 'signal'] = 'exit_long'
                        df.at[df.index[i], 'exit_price'] = df.at[df.index[i], 'close']
                        df.at[df.index[i], 'exit_reason'] = 'conditional'
                        current_position = None
                
                elif current_position == 'short':
                    # Check for stop loss/take profit
                    if df.at[df.index[i], 'high'] >= stop_loss:
                        df.at[df.index[i], 'signal'] = 'exit_short'
                        df.at[df.index[i], 'exit_price'] = stop_loss
                        df.at[df.index[i], 'exit_reason'] = 'stop_loss'
                        current_position = None
                    elif df.at[df.index[i], 'low'] <= take_profit:
                        df.at[df.index[i], 'signal'] = 'exit_short'
                        df.at[df.index[i], 'exit_price'] = take_profit
                        df.at[df.index[i], 'exit_reason'] = 'take_profit'
                        current_position = None
                    # Check for conditional exits
                    elif ((df.at[df.index[i], 'rsi'] > self.params['rsi_exit_short'] and 
                          df.at[df.index[i], 'close'] > df.at[df.index[i], 'ema9']) or
                          df.at[df.index[i], 'macd_crossover']):
                        df.at[df.index[i], 'signal'] = 'exit_short'
                        df.at[df.index[i], 'exit_price'] = df.at[df.index[i], 'close']
                        df.at[df.index[i], 'exit_reason'] = 'conditional'
                        current_position = None
                
                # Entry conditions (only if no current position)
                if current_position is None:
                    # Long entry
                    if (df.at[df.index[i], 'macd_crossover'] and 
                        df.at[df.index[i], 'ai_buy_signal'] and 
                        df.at[df.index[i], 'close'] > df.at[df.index[i], 'ema9'] and 
                        df.at[df.index[i], 'close'] > df.at[df.index[i], 'ema50'] and 
                        df.at[df.index[i], 'is_trending']):
                        
                        df.at[df.index[i], 'signal'] = 'enter_long'
                        entry_price = df.at[df.index[i], 'close']
                        stop_loss = entry_price - df.at[df.index[i], 'atr'] * self.params['sl_multiplier']
                        take_profit = entry_price + df.at[df.index[i], 'atr'] * self.params['tp_multiplier']
                        current_position = 'long'
                    
                    # Short entry
                    elif (df.at[df.index[i], 'macd_crossunder'] and 
                          df.at[df.index[i], 'ai_sell_signal'] and 
                          df.at[df.index[i], 'close'] < df.at[df.index[i], 'ema9'] and 
                          df.at[df.index[i], 'close'] < df.at[df.index[i], 'ema50'] and 
                          df.at[df.index[i], 'is_trending']):
                        
                        df.at[df.index[i], 'signal'] = 'enter_short'
                        entry_price = df.at[df.index[i], 'close']
                        stop_loss = entry_price + df.at[df.index[i], 'atr'] * self.params['sl_multiplier']
                        take_profit = entry_price - df.at[df.index[i], 'atr'] * self.params['tp_multiplier']
                        current_position = 'short'
                
                # Update position tracking columns
                if current_position:
                    df.at[df.index[i], 'position'] = current_position
                    df.at[df.index[i], 'entry_price'] = entry_price
                    df.at[df.index[i], 'stop_loss'] = stop_loss
                    df.at[df.index[i], 'take_profit'] = take_profit
            
            logger.info("Signal generation completed")
            return df
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}", exc_info=True)
            raise

def main(input_file: Path, output_file: Path) -> bool:
    """Process input data and generate trading signals"""
    try:
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file, parse_dates=['datetime'], index_col='datetime')
        
        if df.empty:
            logger.error("Input file is empty")
            return False
        
        logger.info(f"Processing {len(df)} records from {df.index.min()} to {df.index.max()}")
        
        # Generate trading signals
        strategy = TradingStrategy()
        df_with_signals = strategy.generate_signals(df)
        # After generating signals with TradingStrategy
        stats = StatsGenerator()
        results = stats.analyze_trades(df_with_signals)

        # Print summary to console
        stats.print_summary()

        # Or access the raw results
        trade_history = results['trade_history']
        summary_stats = results['summary_stats']
        entry_stats = results['entry_stats']
        exit_stats = results['exit_stats']
        # Save results
        logger.info(f"Saving signals to {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file)
        
        logger.info("Signal generation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}", exc_info=True)
        return False
def analyze_trades(self, df: pd.DataFrame):
        """Analyzes trades from a DataFrame processed by process_signals."""
        logger.info("Analyzing generated trades for detailed statistics")
        try:
            self.trade_history = []
            current_trade = None
            running_pnl = 0.0
            peak_pnl = 0.0
            max_drawdown = 0.0
            win_pnls = []
            loss_pnls = []
            win_durations = []
            loss_durations = []
            risk_rewards = []
            current_streak = 0
            max_consec_wins = 0
            max_consec_losses = 0
            current_streak_type = None

            for i in range(len(df)):
                current_idx = df.index[i]
                signal = df.loc[current_idx, 'signal']

                # Entry Signal Found
                if signal in ['Long', 'Short'] and current_trade is None:
                    entry_price = df.loc[current_idx, 'entry_price']
                    stop_loss = df.loc[current_idx, 'stop_loss']
                    target_price_col = 'target_price_long' if signal == 'Long' else 'target_price_short'
                    target_price = df.loc[current_idx, target_price_col]

                    if pd.notna(entry_price) and pd.notna(stop_loss):
                        current_trade = {
                            'entry_index': i,
                            'entry_idx_time': current_idx,
                            'entry_price': entry_price,
                            'position': signal,
                            'stop_loss': stop_loss,
                            'target_price': target_price,
                            'signal_type': df.loc[current_idx, 'entry_signal_type']
                        }
                        
                        if current_trade['position'] == 'Long':
                            risk = current_trade['entry_price'] - current_trade['stop_loss']
                            reward = current_trade['target_price'] - current_trade['entry_price'] if pd.notna(current_trade['target_price']) else np.nan
                        else:
                            risk = current_trade['stop_loss'] - current_trade['entry_price']
                            reward = current_trade['entry_price'] - current_trade['target_price'] if pd.notna(current_trade['target_price']) else np.nan

                        current_trade['risk_at_entry'] = risk
                        current_trade['reward_at_entry'] = reward
                        current_trade['risk_reward_ratio'] = reward / risk if risk > 0 and pd.notna(reward) else np.nan

                # Exit Signal Found
                elif current_trade and 'Exit' in signal:
                    exit_price = df.loc[current_idx, 'exit_price']
                    exit_idx_time = current_idx
                    exit_reason = df.loc[current_idx, 'exit_reason']

                    if pd.isna(exit_price):
                        exit_price = df.loc[current_idx, 'close']
                        logger.warning(f"Exit price NaN for trade entered on {current_trade['entry_idx_time'].date()}, using close price {exit_price:.2f}")

                    if current_trade['position'] == 'Long':
                        pnl_points = exit_price - current_trade['entry_price']
                    else:
                        pnl_points = current_trade['entry_price'] - exit_price

                    commission = (current_trade['entry_price'] + exit_price) * PARAMS['backtest']['commission_pct']
                    net_pnl_points = pnl_points - commission

                    pct_change = (net_pnl_points / current_trade['entry_price']) * 100 if current_trade['entry_price'] else 0
                    duration_delta = exit_idx_time - current_trade['entry_idx_time']
                    duration_bars = i - current_trade['entry_index']

                    outcome = 'Win' if net_pnl_points > 0 else 'Loss'

                    trade_record = {
                        **current_trade,
                        'exit_idx_time': exit_idx_time,
                        'exit_price': exit_price,
                        'pnl_points': net_pnl_points,
                        'pct_change': pct_change,
                        'duration_delta': duration_delta,
                        'duration_bars': duration_bars,
                        'outcome': outcome,
                        'exit_reason': exit_reason
                    }
                    self.trade_history.append(trade_record)

                    self.summary_stats['total_trades'] += 1
                    self.summary_stats['total_pnl'] += net_pnl_points

                    if outcome == 'Win':
                        self.summary_stats['winning_trades'] += 1
                        win_pnls.append(net_pnl_points)
                        win_durations.append(duration_bars)
                    else:
                        self.summary_stats['losing_trades'] += 1
                        loss_pnls.append(abs(net_pnl_points))
                        loss_durations.append(duration_bars)

                    if pd.notna(current_trade['risk_reward_ratio']):
                        risk_rewards.append(current_trade['risk_reward_ratio'])

                    if outcome == current_streak_type:
                        current_streak += 1
                    else:
                        current_streak = 1
                        current_streak_type = outcome
                    
                    if outcome == 'Win':
                        max_consec_wins = max(max_consec_wins, current_streak)
                    else:
                        max_consec_losses = max(max_consec_losses, current_streak)

                    running_pnl += net_pnl_points
                    peak_pnl = max(peak_pnl, running_pnl)
                    drawdown = peak_pnl - running_pnl
                    self.summary_stats['max_drawdown_points'] = max(self.summary_stats.get('max_drawdown_points', 0.0), drawdown)

                    pos = current_trade['position']
                    sig_type = current_trade['signal_type']
                    exit_type = exit_reason

                    self.entry_stats[pos]['total'] += 1
                    self.entry_stats[pos]['total_pnl'] += net_pnl_points
                    self.entry_stats[pos]['total_duration_bars'] += duration_bars
                    if outcome == 'Win':
                        self.entry_stats[pos]['success'] += 1
                    
                    if exit_type == 'Trailing Stop' or exit_type == 'ATR Stop':
                        self.entry_stats[pos]['sl_hit'] += 1
                    if exit_type == 'Fib Target' or exit_type == 'Target':
                        self.entry_stats[pos]['target_hit'] += 1

                    self.exit_stats[exit_type]['total'] += 1
                    self.exit_stats[exit_type]['total_pnl'] += net_pnl_points
                    self.exit_stats[exit_type]['total_duration_bars'] += duration_bars
                    if outcome == 'Win':
                        self.exit_stats[exit_type]['success'] += 1

                    self.signal_stats[sig_type]['total'] += 1
                    self.signal_stats[sig_type]['total_pnl'] += net_pnl_points
                    if outcome == 'Win':
                        self.signal_stats[sig_type]['success'] += 1

                    current_trade = None

            if self.summary_stats['total_trades'] > 0:
                avg_win = np.mean(win_pnls) if win_pnls else 0
                avg_loss = np.mean(loss_pnls) if loss_pnls else 0
                self.summary_stats['win_rate'] = (self.summary_stats['winning_trades'] / self.summary_stats['total_trades']) * 100
                self.summary_stats['avg_win_points'] = avg_win
                self.summary_stats['avg_loss_points'] = avg_loss
                self.summary_stats['profit_factor'] = abs(sum(win_pnls) / sum(loss_pnls)) if sum(loss_pnls) != 0 else np.inf
                self.summary_stats['expectancy_points'] = (avg_win * (self.summary_stats['win_rate']/100)) - (avg_loss * (1 - self.summary_stats['win_rate']/100))
                self.summary_stats['max_consec_wins'] = max_consec_wins
                self.summary_stats['max_consec_losses'] = max_consec_losses
                self.summary_stats['avg_win_duration_bars'] = np.mean(win_durations) if win_durations else 0
                self.summary_stats['avg_loss_duration_bars'] = np.mean(loss_durations) if loss_durations else 0
                self.summary_stats['avg_risk_reward_ratio'] = np.nanmean(risk_rewards) if risk_rewards else np.nan

                # Calculate health score components safely
                win_rate_component = self.summary_stats['win_rate'] * 0.4

                profit_factor = self.summary_stats.get('profit_factor', 0)
                profit_factor_component = (min(profit_factor, 5) * 10 if pd.notna(profit_factor) else 0)

                # Safely calculate duration ratio component
                avg_win_duration = self.summary_stats.get('avg_win_duration_bars', 1)
                avg_loss_duration = self.summary_stats.get('avg_loss_duration_bars', 1)
                duration_ratio = (avg_loss_duration / avg_win_duration) if avg_win_duration > 0 else 1
                duration_component = (1 - duration_ratio) * 20

                # Safely calculate risk/reward component
                risk_reward = self.summary_stats.get('avg_risk_reward_ratio', 0)
                risk_reward_component = (risk_reward * 10 if pd.notna(risk_reward) else 0)

                # Calculate final health score
                health_score = min(100, max(0,
                    win_rate_component +
                    profit_factor_component +
                    duration_component +
                    risk_reward_component
                ))

                self.summary_stats['health_score'] = round(health_score, 2)  # Store with 2 decimal places

            for pos in list(self.entry_stats.keys()):
                if self.entry_stats[pos]['total'] > 0:
                    self.entry_stats[pos]['avg_pnl'] = self.entry_stats[pos]['total_pnl'] / self.entry_stats[pos]['total']
                    self.entry_stats[pos]['avg_duration_bars'] = self.entry_stats[pos]['total_duration_bars'] / self.entry_stats[pos]['total']
                    self.entry_stats[pos]['win_rate'] = (self.entry_stats[pos]['success'] / self.entry_stats[pos]['total']) * 100
                    self.entry_stats[pos]['sl_rate'] = (self.entry_stats[pos]['sl_hit'] / self.entry_stats[pos]['total']) * 100
                    self.entry_stats[pos]['target_rate'] = (self.entry_stats[pos]['target_hit'] / self.entry_stats[pos]['total']) * 100

            for exit_type in list(self.exit_stats.keys()):
                if self.exit_stats[exit_type]['total'] > 0:
                    self.exit_stats[exit_type]['avg_pnl'] = self.exit_stats[exit_type]['total_pnl'] / self.exit_stats[exit_type]['total']
                    self.exit_stats[exit_type]['avg_duration_bars'] = self.exit_stats[exit_type]['total_duration_bars'] / self.exit_stats[exit_type]['total']
                    self.exit_stats[exit_type]['win_rate'] = (self.exit_stats[exit_type]['success'] / self.exit_stats[exit_type]['total']) * 100

            for sig_type in list(self.signal_stats.keys()):
                if self.signal_stats[sig_type]['total'] > 0:
                    self.signal_stats[sig_type]['avg_pnl'] = self.signal_stats[sig_type]['total_pnl'] / self.signal_stats[sig_type]['total']
                    self.signal_stats[sig_type]['win_rate'] = (self.signal_stats[sig_type]['success'] / self.signal_stats[sig_type]['total']) * 100

            logger.info("Finished analyzing trades.")
            return df

        except Exception as e:
            logger.error(f"Error in analyze_trades: {str(e)}", exc_info=True)
            raise

    # def print_summary(self):
    #     """Print comprehensive performance statistics."""
    #     if not self.summary_stats or self.summary_stats['total_trades'] == 0:
    #         print("\nNo trades to summarize.")
    #         logger.info("No trades to summarize.")
    #         return
    #     try:
    #         summary = [
    #             "\n=== ENHANCED TRADE SUMMARY ===",
    #             f"Total Trades: {int(self.summary_stats['total_trades'])}",
    #             f"Winning Trades: {int(self.summary_stats['winning_trades'])} ({self.summary_stats['win_rate']:.1f}%)",
    #             f"Losing Trades: {int(self.summary_stats['losing_trades'])}",
    #             f"Total P&L (Points): {self.summary_stats['total_pnl']:.2f}",
    #             f"Profit Factor: {self.summary_stats['profit_factor']:.2f}",
    #             f"Expectancy (Points): {self.summary_stats['expectancy_points']:.2f}",
    #             f"Max Drawdown (Points): {self.summary_stats['max_drawdown_points']:.2f}",
    #             f"Avg Win / Avg Loss (Points): {self.summary_stats['avg_win_points']:.2f} / {self.summary_stats['avg_loss_points']:.2f}",
    #             f"Max Consecutive Wins: {int(self.summary_stats['max_consec_wins'])}",
    #             f"Max Consecutive Losses: {int(self.summary_stats['max_consec_losses'])}",
    #             f"Avg Win Duration (Bars): {self.summary_stats['avg_win_duration_bars']:.1f}",
    #             f"Avg Loss Duration (Bars): {self.summary_stats['avg_loss_duration_bars']:.1f}",
    #             f"Avg Risk/Reward Ratio (at Entry): {self.summary_stats['avg_risk_reward_ratio']:.2f}:1" if pd.notna(self.summary_stats['avg_risk_reward_ratio']) else "N/A",
    #             f"Strategy Health Score: {self.summary_stats['health_score']:.1f}/100",
    #             "\n=== ENTRY STATISTICS ===",
    #             "Position | Total | Win % | Avg P&L | Avg Dur | SL % | Target %",
    #             "-------------------------------------------------------------"
    #         ]
    #         for position, stats in self.entry_stats.items():
    #             summary.append(
    #                 f"{position:8} | {int(stats['total']):5} | {stats['win_rate']:5.1f}% | {stats['avg_pnl']:7.2f} | {stats['avg_duration_bars']:7.1f} | {stats['sl_rate']:4.1f}% | {stats['target_rate']:6.1f}%"
    #             )
    #         summary.extend([
    #             "\n=== EXIT STATISTICS ===",
    #             "Exit Type           | Total | Win % | Avg P&L | Avg Dur",
    #             "------------------------------------------------------"
    #         ])
    #         for exit_type, stats in sorted(self.exit_stats.items()):
    #             summary.append(
    #                 f"{str(exit_type):19} | {int(stats['total']):5} | {stats['win_rate']:5.1f}% | {stats['avg_pnl']:7.2f} | {stats['avg_duration_bars']:7.1f}"
    #             )
    #         summary.extend([
    #             "\n=== SIGNAL TYPE STATISTICS ===",
    #             "Signal Type          | Total | Win % | Avg P&L",
    #             "---------------------------------------------"
    #         ])
    #         for signal_type, stats in sorted(self.signal_stats.items()):
    #             if not signal_type:
    #                 continue
    #             summary.append(
    #                 f"{str(signal_type):20} | {int(stats['total']):5} | {stats['win_rate']:5.1f}% | {stats['avg_pnl']:7.2f}"
    #             )

    #         print("\n".join(summary))
    #         logger.info("\n".join(summary))

    #         health = self.summary_stats['health_score']
    #         if health >= 70:
    #             health_indicator = "🟢 STRONG"
    #         elif health >= 50:
    #             health_indicator = "🟡 MODERATE"
    #         else:
    #             health_indicator = "🔴 WEAK"
    #         print(f"\nStrategy Health: {health_indicator}")
    #         if self.summary_stats['total_trades'] < 30:
    #             print("⚠️  Warning: Low sample size (<30 trades)")

    #     except Exception as e:
    #         logger.error(f"Error printing summary: {str(e)}", exc_info=True)
    #         raise
def print_summary(self):
    
        if not self.summary_stats or self.summary_stats['total_trades'] == 0:
            print("\nNo trades to summarize.")
            logger.info("No trades to summarize.")
            return
        
        try:
            # Calculate additional metrics
            total_profit = sum(trade['pnl_points'] for trade in self.trade_history if trade['pnl_points'] > 0)
            total_loss = abs(sum(trade['pnl_points'] for trade in self.trade_history if trade['pnl_points'] < 0))
            profit_percentage = (total_profit / (total_profit + total_loss)) * 100 if (total_profit + total_loss) > 0 else 0
            loss_percentage = 100 - profit_percentage
            
            avg_profit_per_trade = total_profit / self.summary_stats['winning_trades'] if self.summary_stats['winning_trades'] > 0 else 0
            avg_loss_per_trade = total_loss / self.summary_stats['losing_trades'] if self.summary_stats['losing_trades'] > 0 else 0
            
            # Calculate profit/loss ratio
            profit_loss_ratio = avg_profit_per_trade / avg_loss_per_trade if avg_loss_per_trade != 0 else float('inf')
            
            # Enhanced summary with new metrics
            summary = [
                "\n=== ENHANCED TRADE SUMMARY ===",
                f"Total Trades: {int(self.summary_stats['total_trades'])}",
                f"Winning Trades: {int(self.summary_stats['winning_trades'])} ({self.summary_stats['win_rate']:.1f}%)",
                f"Losing Trades: {int(self.summary_stats['losing_trades'])}",
                f"Total Profit: {total_profit:.2f} points ({profit_percentage:.1f}%)",
                f"Total Loss: {total_loss:.2f} points ({loss_percentage:.1f}%)",
                f"Net P&L: {self.summary_stats['total_pnl']:.2f} points",
                f"Profit Factor: {self.summary_stats['profit_factor']:.2f}",
                f"Profit/Loss Ratio: {profit_loss_ratio:.2f}:1",
                f"Avg Profit per Win: {avg_profit_per_trade:.2f} points",
                f"Avg Loss per Loss: {avg_loss_per_trade:.2f} points",
                f"Expectancy (Points): {self.summary_stats['expectancy_points']:.2f}",
                f"Max Drawdown (Points): {self.summary_stats['max_drawdown_points']:.2f}",
                f"Max Consecutive Wins: {int(self.summary_stats['max_consec_wins'])}",
                f"Max Consecutive Losses: {int(self.summary_stats['max_consec_losses'])}",
                f"Avg Win Duration: {self.summary_stats['avg_win_duration_bars']:.1f} bars",
                f"Avg Loss Duration: {self.summary_stats['avg_loss_duration_bars']:.1f} bars",
                f"Risk/Reward Ratio: {self.summary_stats['avg_risk_reward_ratio']:.2f}:1" if pd.notna(self.summary_stats['avg_risk_reward_ratio']) else "Risk/Reward: N/A",
                f"Strategy Health: {self.summary_stats['health_score']:.1f}/100",
                "\n=== ENTRY STATISTICS ===",
                "Position | Total | Win % | Avg P&L | Profit% | Loss% | Avg Dur",
                "-------------------------------------------------------------"
            ]
            
            # Enhanced entry stats with profit/loss percentages
            for position, stats in self.entry_stats.items():
                pos_trades = [t for t in self.trade_history if t['position'] == position]
                pos_profit = sum(t['pnl_points'] for t in pos_trades if t['pnl_points'] > 0)
                pos_loss = abs(sum(t['pnl_points'] for t in pos_trades if t['pnl_points'] < 0))
                pos_total = pos_profit + pos_loss
                pos_profit_pct = (pos_profit / pos_total * 100) if pos_total > 0 else 0
                pos_loss_pct = 100 - pos_profit_pct
                
                summary.append(
                    f"{position:8} | {int(stats['total']):5} | {stats['win_rate']:5.1f}% | {stats['avg_pnl']:7.2f} | "
                    f"{pos_profit_pct:6.1f}% | {pos_loss_pct:5.1f}% | {stats['avg_duration_bars']:7.1f}"
                )
            
            # Rest of your existing summary code...
            summary.extend([
                "\n=== EXIT STATISTICS ===",
                "Exit Type           | Total | Win % | Avg P&L | Profit% | Loss% | Avg Dur",
                "----------------------------------------------------------------------"
            ])
            
            for exit_type, stats in sorted(self.exit_stats.items()):
                exit_trades = [t for t in self.trade_history if t['exit_reason'] == exit_type]
                exit_profit = sum(t['pnl_points'] for t in exit_trades if t['pnl_points'] > 0)
                exit_loss = abs(sum(t['pnl_points'] for t in exit_trades if t['pnl_points'] < 0))
                exit_total = exit_profit + exit_loss
                exit_profit_pct = (exit_profit / exit_total * 100) if exit_total > 0 else 0
                exit_loss_pct = 100 - exit_profit_pct
                
                summary.append(
                    f"{str(exit_type):19} | {int(stats['total']):5} | {stats['win_rate']:5.1f}% | {stats['avg_pnl']:7.2f} | "
                    f"{exit_profit_pct:6.1f}% | {exit_loss_pct:5.1f}% | {stats['avg_duration_bars']:7.1f}"
                )
            
            print("\n".join(summary))
            logger.info("\n".join(summary))
            
            # Health indicator remains the same
            health = self.summary_stats['health_score']
            if health >= 70:
                health_indicator = "🟢 STRONG"
            elif health >= 50:
                health_indicator = "🟡 MODERATE"
            else:
                health_indicator = "🔴 WEAK"
            print(f"\nStrategy Health: {health_indicator}")
            
            if self.summary_stats['total_trades'] < 30:
                print("⚠️  Warning: Low sample size (<30 trades)")
        
        except Exception as e:
            logger.error(f"Error printing summary: {str(e)}", exc_info=True)
            raise


def parse_args():
    parser = argparse.ArgumentParser(description='Generate enhanced trading signals and calculate detailed stats.')
    parser.add_argument('--input', type=str, default=str(DEFAULT_INPUT_FILE), help=f'Input CSV (default: {DEFAULT_INPUT_FILE})')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT_FILE), help=f'Output CSV (default: {DEFAULT_OUTPUT_FILE})')
    parser.add_argument('--plot', type=str, default=str(DEFAULT_PLOT_FILE), help=f'Plot image path (default: {DEFAULT_PLOT_FILE})')
    parser.add_argument('--no-plot', action='store_true', help='Disable generating plot')
    parser.add_argument('--full-history', action='store_true', help='Run on full history')
    return parser.parse_args()
if __name__ == "__main__":
    import argparse
    args = parse_args()
    input_file = Path(args.input)
    output_file = Path(args.output)
    plot_file = Path(args.plot)
    
    
    if not main(input_file, output_file):
        exit(1)

