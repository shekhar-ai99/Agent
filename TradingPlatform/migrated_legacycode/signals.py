import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SignalGenerator:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Signal parameters
        self.ema_fast = 9
        self.ema_slow = 21
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.rsi_neutral = 50

    def load_data(self) -> pd.DataFrame:
        """Load data with validation"""
        logger.info(f"Loading data from {self.input_path}")
        try:
            df = pd.read_csv(self.input_path, parse_dates=['datetime'], index_col='datetime')
            
            required_cols = ['close', 'ema_9', 'ema_21', 'rsi', 'macd', 'macd_signal']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"Missing required columns: {missing}")
                
            # Ensure numeric columns are properly formatted
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals with proper null checks"""
        try:
            # Initialize all signal columns
            df['signal'] = 'Hold'
            df['signal_type'] = 'Hold'  # Added for signal_analyzer compatibility
            df['position'] = None
            df['entry_price'] = None
            df['stop_loss'] = None
            df['target_price'] = None
            df['exit_price'] = None
            df['trade_pnl'] = None
            df['pct_change'] = None

            current_position = None
            entry_price = None
            stop_loss = None
            
            for i in range(1, len(df)):
                row = df.iloc[i]
                prev_row = df.iloc[i-1]
                
                # Skip if we have missing values
                if any(pd.isna(row[col]) for col in ['close', 'ema_9', 'ema_21', 'rsi', 'macd', 'macd_signal']):
                    continue
                
                # Entry Signals
                if current_position is None:
                    long_cond = (
                        row['close'] > row['ema_9'] > row['ema_21'] and
                        row['rsi'] > self.rsi_neutral and
                        row['macd'] > row['macd_signal'] > 0
                    )
                    
                    short_cond = (
                        row['close'] < row['ema_9'] < row['ema_21'] and
                        row['rsi'] < self.rsi_neutral and
                        row['macd'] < row['macd_signal'] < 0
                    )
                    
                    if long_cond:
                        current_position = 'Long'
                        entry_price = row['close']
                        stop_loss = entry_price * 0.99
                        signal = 'Enter Long'
                        df.at[row.name, 'signal'] = signal
                        df.at[row.name, 'signal_type'] = signal  # Set both for compatibility
                        df.at[row.name, 'position'] = current_position
                        df.at[row.name, 'entry_price'] = entry_price
                        df.at[row.name, 'stop_loss'] = stop_loss
                        df.at[row.name, 'target_price'] = entry_price * 1.02
                        logger.info(f"Entered Long at {entry_price}")
                        
                    elif short_cond:
                        current_position = 'Short'
                        entry_price = row['close']
                        stop_loss = entry_price * 1.01
                        signal = 'Enter Short'
                        df.at[row.name, 'signal'] = signal
                        df.at[row.name, 'signal_type'] = signal  # Set both for compatibility
                        df.at[row.name, 'position'] = current_position
                        df.at[row.name, 'entry_price'] = entry_price
                        df.at[row.name, 'stop_loss'] = stop_loss
                        df.at[row.name, 'target_price'] = entry_price * 0.98
                        logger.info(f"Entered Short at {entry_price}")
                
                # Exit Conditions - only proceed if we have valid values
                elif current_position and not pd.isna(entry_price) and not pd.isna(stop_loss):
                    exit_cond = False
                    exit_reason = ''
                    
                    if current_position == 'Long':
                        if not pd.isna(row['close']) and row['close'] <= stop_loss:
                            exit_cond = True
                            exit_reason = 'Stop Loss'
                        elif not pd.isna(row['rsi']) and row['rsi'] > self.rsi_overbought:
                            exit_cond = True
                            exit_reason = 'RSI Overbought'
                        elif not pd.isna(row['close']) and not pd.isna(row['ema_9']) and row['close'] < row['ema_9']:
                            exit_cond = True
                            exit_reason = 'EMA Cross'
                            
                    elif current_position == 'Short':
                        if not pd.isna(row['close']) and row['close'] >= stop_loss:
                            exit_cond = True
                            exit_reason = 'Stop Loss'
                        elif not pd.isna(row['rsi']) and row['rsi'] < self.rsi_oversold:
                            exit_cond = True
                            exit_reason = 'RSI Oversold'
                        elif not pd.isna(row['close']) and not pd.isna(row['ema_9']) and row['close'] > row['ema_9']:
                            exit_cond = True
                            exit_reason = 'EMA Cross'
                    
                    if exit_cond:
                        exit_price = row['close']
                        pnl = exit_price - entry_price if current_position == 'Long' else entry_price - exit_price
                        pct_change = (pnl / entry_price) * 100
                        
                        signal = f'Exit {current_position} ({exit_reason})'
                        df.at[row.name, 'signal'] = signal
                        df.at[row.name, 'signal_type'] = signal  # Set both for compatibility
                        df.at[row.name, 'exit_price'] = exit_price
                        df.at[row.name, 'trade_pnl'] = pnl
                        df.at[row.name, 'pct_change'] = pct_change
                        
                        logger.info(f"Exited {current_position} at {exit_price} (P&L: {pct_change:.2f}%)")
                        
                        current_position = None
                        entry_price = None
                        stop_loss = None
            
            return df

        except Exception as e:
            logger.error(f"Signal generation failed: {str(e)}")
            raise

    def save_signals(self, df: pd.DataFrame) -> None:
        """Save signals to CSV"""
        try:
            # Ensure all required columns exist before saving
            required_columns = ['signal', 'signal_type', 'position', 'entry_price', 
                              'stop_loss', 'target_price', 'exit_price', 'trade_pnl', 'pct_change']
            
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None
                    
            df.to_csv(self.output_path)
            logger.info(f"Signals saved to {self.output_path}")
        except Exception as e:
            logger.error(f"Error saving signals: {str(e)}")
            raise

def main(input_path: str, output_path: str):
    """Main execution function"""
    try:
        logger.info("Starting signal generation process")
        logger.info(f"Input path set to: {input_path}")
        logger.info(f"Output path set to: {output_path}")
        
        generator = SignalGenerator(input_path, output_path)
        df = generator.load_data()
        logger.info(f"Successfully loaded {len(df)} records")
        logger.info("Generating trading signals")
        df = generator.generate_signals(df)
        generator.save_signals(df)
        logger.info("Signal generation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to generate signals: {str(e)}")
        # Create empty output file with error message
        error_df = pd.DataFrame({'error': [str(e)]})
        error_df.to_csv(output_path)
        logger.info(f"Created error output file at {output_path}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    args = parser.parse_args()
    
    success = main(args.input, args.output)
    exit(0 if success else 1)