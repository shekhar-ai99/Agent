# File: signals.py
import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, Any

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('signals.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
data_folder = "data/"
input_file = os.path.join(data_folder, "nifty_signals.csv")
output_file = os.path.join(data_folder, "nifty_signals_final.csv")

# Parameter Organization - Using nested dictionaries
PARAMS: Dict[str, Dict[str, Any]] = {
    'core': {
        'exit_score_drop_threshold': 1.5,
        'use_fib_bounce_entry': True,
        'use_fib_bounce_sell': True,
        'fib_bounce_lookback': 3,
        'use_ema_bounce_buy': True,
        'use_ema_bounce_sell': True,
        'ema_bounce_lookback': 2,
        'ema_bounce_source_str': "Fast EMA",
        'use_bb_mid_bounce_buy': True,
        'use_bb_mid_bounce_sell': True,
        'bb_bounce_lookback': 2,
        'use_vol_breakout_buy': True,
        'use_vol_breakout_sell': True
    },
    'ema': {
        'fast_len': 9,
        'med_len': 14,
        'slow_len': 21,
        'use_ema_exit': True
    },
    'bollinger': {
        'bb_len': 20,
        'bb_std_dev': 2.0,
        'use_bb_return_exit': True
    },
    'rsi': {
        'rsi_len': 14,
        'rsi_buy_level': 55.0,
        'rsi_sell_level': 45.0,
        'use_rsi_div_exit': False,
        'rsi_confirm_fib': True,
        'rsi_confirm_ema': False,
        'rsi_confirm_bb': False
    },
    'macd': {
        'macd_fast_len': 12,
        'macd_slow_len': 26,
        'macd_signal_len': 9
    },
    'volume': {
        'vol_ma_len': 50,
        'vol_multiplier': 1.5,
        'use_vol_fade_exit': True
    },
    'atr': {
        'atr_len': 14,
        'atr_mult': 2.0,
        'use_atr_stop': True
    },
    'fibonacci': {
        'use_fib_exit': True,
        'fib_lookback_exit': 30,
        'fib_extension_level': 1.618,
        'fib_pivot_lookback': 15,
        'fib_max_bars': 200
    },
    'trend': {
        'use_ema_trend_filter': True,
        'use_adx_filter': True,
        'adx_len': 14,
        'adx_threshold': 20.0,
        'use_adx_direction_filter': True
    }
}

# --- Helper Functions ---
def calculate_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trading score based on multiple factors"""
    logger.debug("Calculating trading score")
    try:
        df['raw_score'] = 0
        
        # Trend Score (0-2 points)
        df['raw_score'] += (df['ema_fast'] > df['ema_slow']).astype(int)
        df['raw_score'] += (df['ema_med'] > df['ema_slow']).astype(int)
        
        # Momentum Score (0-2 points)
        df['raw_score'] += (df['rsi'] > 50).astype(int)
        df['raw_score'] += (df['macd'] > df['macd_signal']).astype(int)
        
        # Volatility Score (0-1 point)
        df['raw_score'] += (df['close'] > df['bb_middle']).astype(int)
        
        # Volume Score (0-1 point)
        df['raw_score'] += (df['volume'] > df['vol_ma']).astype(int)
        
        # Scale to 0-10 range
        df['raw_score'] = (df['raw_score'] / 5) * 10
        df['scaled_score'] = df['raw_score'].rolling(5).mean()
        return df
    except Exception as e:
        logger.error(f"Error in calculate_score: {str(e)}")
        raise

def calculate_rsi_divergence(df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
    """Calculate bullish and bearish RSI divergences"""
    logger.debug(f"Calculating RSI divergence with lookback {lookback}")
    try:
        df['bullish_rsi_div'] = False
        df['bearish_rsi_div'] = False
        
        for i in range(lookback, len(df)):
            # Price lows and RSI lows
            recent_lows = df['low'].iloc[i-lookback:i]
            recent_rsi_lows = df['rsi'].iloc[i-lookback:i]
            
            # Bullish divergence
            if (df['low'].iloc[i] < recent_lows.min() and 
                df['rsi'].iloc[i] > recent_rsi_lows.min()):
                df.at[df.index[i], 'bullish_rsi_div'] = True
            
            # Bearish divergence
            recent_highs = df['high'].iloc[i-lookback:i]
            recent_rsi_highs = df['rsi'].iloc[i-lookback:i]
            if (df['high'].iloc[i] > recent_highs.max() and 
                df['rsi'].iloc[i] < recent_rsi_highs.max()):
                df.at[df.index[i], 'bearish_rsi_div'] = True
        
        return df
    except Exception as e:
        logger.error(f"Error in calculate_rsi_divergence: {str(e)}")
        raise

def process_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Main signal processing function"""
    logger.info("Starting signal processing")
    
    # Initialize trading state
    trade_state = {
        'in_long': False,
        'in_short': False,
        'exit_signal': False,
        'trade_count_long': 0,
        'trade_count_short': 0,
        'entry_price': np.nan,
        'stop_loss': np.nan,
        'target_price': np.nan,
        'entry_signal_type': '',
        'current_trade_id': ''
    }
    
    try:
        # --- Indicator Calculations ---
        logger.info("Calculating indicators")
        # EMAs
        df['ema_fast'] = df['close'].ewm(span=PARAMS['ema']['fast_len'], adjust=False).mean()
        df['ema_med'] = df['close'].ewm(span=PARAMS['ema']['med_len'], adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=PARAMS['ema']['slow_len'], adjust=False).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(PARAMS['bollinger']['bb_len']).mean()
        rolling_std = df['close'].rolling(PARAMS['bollinger']['bb_len']).std()
        df['bb_upper'] = df['bb_middle'] + rolling_std * PARAMS['bollinger']['bb_std_dev']
        df['bb_lower'] = df['bb_middle'] - rolling_std * PARAMS['bollinger']['bb_std_dev']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/PARAMS['rsi']['rsi_len'], adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/PARAMS['rsi']['rsi_len'], adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1)
        df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
        df['rsi'].fillna(50, inplace=True)
        
        # MACD
        ema_fast_macd = df['close'].ewm(span=PARAMS['macd']['macd_fast_len'], adjust=False).mean()
        ema_slow_macd = df['close'].ewm(span=PARAMS['macd']['macd_slow_len'], adjust=False).mean()
        df['macd'] = ema_fast_macd - ema_slow_macd
        df['macd_signal'] = df['macd'].ewm(span=PARAMS['macd']['macd_signal_len'], adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Volume
        df['vol_ma'] = df['volume'].rolling(PARAMS['volume']['vol_ma_len']).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift())
        low_close_prev = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1/PARAMS['atr']['atr_len'], adjust=False).mean()
        
        # ADX
        if PARAMS['trend']['use_adx_filter'] or PARAMS['trend']['use_adx_direction_filter']:
            up_move = df['high'].diff()
            down_move = -df['low'].diff()
            plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
            minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
            plus_dm.fillna(0, inplace=True)
            minus_dm.fillna(0, inplace=True)
            atr_adx = df['atr'].replace(0, 1)
            smooth_plus_dm = plus_dm.ewm(alpha=1/PARAMS['trend']['adx_len'], adjust=False).mean()
            smooth_minus_dm = minus_dm.ewm(alpha=1/PARAMS['trend']['adx_len'], adjust=False).mean()
            df['plus_di'] = 100 * (smooth_plus_dm / atr_adx)
            df['minus_di'] = 100 * (smooth_minus_dm / atr_adx)
            di_sum = (df['plus_di'] + df['minus_di']).replace(0, 1)
            dx = 100 * (abs(df['plus_di'] - df['minus_di']) / di_sum)
            df['adx'] = dx.ewm(alpha=1/PARAMS['trend']['adx_len'], adjust=False).mean()
            df.fillna({'adx': 0, 'plus_di': 0, 'minus_di': 0}, inplace=True)
        else:
            df['plus_di'] = 0.0
            df['minus_di'] = 0.0
            df['adx'] = 0.0

        # Calculate score and divergence
        df = calculate_score(df)
        df = calculate_rsi_divergence(df, lookback=PARAMS['rsi']['rsi_len'])
        
        # --- Main Processing Loop ---
        start_index = max(
            PARAMS['bollinger']['bb_len'],
            PARAMS['ema']['slow_len'],
            PARAMS['macd']['macd_slow_len'],
            PARAMS['volume']['vol_ma_len'],
            PARAMS['atr']['atr_len'],
            PARAMS['trend']['adx_len'] if (PARAMS['trend']['use_adx_filter'] or PARAMS['trend']['use_adx_direction_filter']) else 0,
            PARAMS['fibonacci']['fib_pivot_lookback'],
            PARAMS['fibonacci']['fib_lookback_exit']
        ) + 1

        logger.info(f"Starting main processing loop from index {start_index}")
        for i in range(start_index, len(df)):
            current_index = df.index[i]
            prev_index = df.index[i-1]
            current = df.loc[current_index]
            prev = df.loc[prev_index]

            # [Rest of your existing trading logic...]
            # Replace all parameter references with PARAMS dictionary
            # Example: use PARAMS['core']['use_fib_bounce_entry'] instead of use_fib_bounce_entry
            
            # [Maintain all your existing trading logic, just replace parameter references]
            
        return df

    except Exception as e:
        logger.error(f"Error in process_signals: {str(e)}", exc_info=True)
        raise

def calculate_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate target prices using multiple methods"""
    logger.info("Calculating targets")
    try:
        df['target_price'] = np.nan
        df['target_type'] = ''
        
        # Parameters for target calculation
        atr_multiplier = 1.5
        rr_ratio = 2.0
        recent_swing_lookback = 20
        
        for i in range(1, len(df)):
            if df.loc[df.index[i], 'position'] == 'Long' and pd.isna(df.loc[df.index[i], 'target_price']):
                entry_price = df.loc[df.index[i], 'entry_price']
                stop_loss = df.loc[df.index[i], 'stop_loss']
                
                # Method 1: ATR-based target
                atr = df.loc[df.index[i], 'atr']
                atr_target = entry_price + (atr * atr_multiplier)
                
                # Method 2: Recent swing high
                recent_high = df['high'].iloc[max(0,i-recent_swing_lookback):i].max()
                swing_target = entry_price + (recent_high - entry_price) * 0.8
                
                # Method 3: Risk-reward based
                risk = entry_price - stop_loss
                rr_target = entry_price + (risk * rr_ratio)
                
                # Method 4: Fibonacci extension
                fib_target = df.loc[df.index[i], 'target_price']
                
                # Choose target
                valid_targets = [t for t in [atr_target, swing_target, rr_target, fib_target] 
                               if not pd.isna(t)]
                if valid_targets:
                    final_target = min(valid_targets)
                    df.loc[df.index[i], 'target_price'] = final_target
                    df.loc[df.index[i], 'target_type'] = 'Multi-Method'
                    
            elif df.loc[df.index[i], 'position'] == 'Short' and pd.isna(df.loc[df.index[i], 'target_price']):
                # Similar logic for short positions
                pass
                
        return df
    except Exception as e:
        logger.error(f"Error in calculate_targets: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file, parse_dates=['datetime'], index_col='datetime')
        
        logger.info("Processing signals")
        df = process_signals(df)
        
        logger.info("Calculating targets")
        df = calculate_targets(df)
        
        logger.info(f"Saving results to {output_file}")
        df.to_csv(output_file)
        logger.info("Processing completed successfully")
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
    except pd.errors.EmptyDataError:
        logger.error("Input file is empty or corrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)