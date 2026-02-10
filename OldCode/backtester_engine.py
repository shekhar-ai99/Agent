# backtester_engine.py (Corrected for TypeError)

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import sys
# Ensure typing imports are complete
from typing import Dict, Callable, Optional, Tuple, Any 

# Keep your logging setup
log_file = 'backtest_engine.log'
# Configure logging properly (avoid basicConfig if used elsewhere/in main)
engine_logger = logging.getLogger(__name__) 
if not engine_logger.hasHandlers(): # Avoid adding handlers multiple times
    engine_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    engine_logger.addHandler(fh)
    engine_logger.addHandler(sh)
    engine_logger.propagate = False # Prevent duplication if root logger also configured

class EnhancedMultiStrategyBacktester:
    """
    Core backtesting engine. Runs strategies, generates signals, PnL points,
    and manages trade state. Assumes input data already has indicators.
    """
    # MODIFIED __init__ to accept the config dictionary
    def __init__(self, strategies_config: Dict[str, Dict], 
                 commission_pct: float = 0.0005, slippage_pct: float = 0.0002):
        if not strategies_config: 
            raise ValueError("Strategies config dictionary cannot be empty.")
        
        # Validate the structure and callable functions within the config
        for name, config in strategies_config.items():
            if not isinstance(config, dict):
                 raise ValueError(f"Configuration for strategy '{name}' must be a dictionary.")
            if 'function' not in config or not callable(config['function']):
                raise ValueError(f"Strategy '{name}' configuration missing callable 'function' key.")
            if 'params' not in config or not isinstance(config['params'], dict):
                 raise ValueError(f"Strategy '{name}' configuration missing 'params' dictionary key.")

        # Store the entire config dictionary
        self.strategies_config = strategies_config 
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.results_df = None
        engine_logger.info("Backtester Engine Initialized with %d strategies", len(strategies_config))


    # preprocess_data remains the same as your last version
    def preprocess_data(self, data_with_indicators: pd.DataFrame) -> pd.DataFrame:
        """
        Validates data that already includes indicators.
        No recalculation needed here.
        """
        engine_logger.info("Preprocessing data (validation only)...")
        data_with_indicators.columns = [col.lower() for col in data_with_indicators.columns] # Ensure lowercase

        # --- Basic Validation ---
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'atr'] # Ensure ATR is present
        missing = [col for col in required_cols if col not in data_with_indicators.columns]
        if missing: raise ValueError(f"Input indicator data missing required columns: {missing}")

        if not isinstance(data_with_indicators.index, pd.DatetimeIndex):
            engine_logger.warning("Input data index is not DatetimeIndex. Attempting conversion.")
            data_with_indicators.index = pd.to_datetime(data_with_indicators.index)

        # Optional: Add calculation for simple returns if needed by analyzer later
        if 'returns' not in data_with_indicators.columns:
             data_with_indicators['returns'] = data_with_indicators['close'].pct_change()

        # Drop rows where essential OHLCV or ATR is NaN (usually affects start)
        initial_rows = len(data_with_indicators)
        # Make sure to check the original required_cols list here
        data_with_indicators = data_with_indicators.dropna(subset=['open', 'high', 'low', 'close', 'volume', 'atr']) 
        rows_dropped = initial_rows - len(data_with_indicators)
        if rows_dropped > 0:
            engine_logger.info(f"Dropped {rows_dropped} rows with NaN in essential columns (OHLCV, ATR).")

        engine_logger.info(f"Preprocessing/validation complete. Shape: {data_with_indicators.shape}")
        return data_with_indicators.copy() # Return a copy


    def run_backtest(self, data_with_indicators: pd.DataFrame, 
                     # Removed default SL/TP/TSL args here, should come from config now
                     ) -> Optional[pd.DataFrame]: 
        """
        Run backtest across all strategies using pre-calculated indicator data.
        Uses parameters defined within the strategies_config.
        """
        engine_logger.info("Starting backtest run...")
        try:
            results = self.preprocess_data(data_with_indicators)
            if results.empty:
                engine_logger.error("No valid data after preprocessing/validation. Aborting backtest.")
                return None

            strategy_states = self._initialize_strategy_columns(results)

            engine_logger.info(f"Starting backtest loop ({len(results)} bars)...")
            if len(results) < 2:
                 engine_logger.error("Not enough data rows to run backtest loop after NaN drop.")
                 return None

            for i in range(1, len(results)): # Start from 1
                current_idx = results.index[i]
                current_row = results.iloc[i]
                data_slice = results.iloc[:i+1] # Historical slice up to current row

                # MODIFIED Loop: Iterate through the config dictionary
                for name, config in self.strategies_config.items(): 
                    strategy_func = config['function'] # Extract function
                    params = config['params']          # Extract params
                    
                    # Get necessary SL/TP/TSL parameters from the strategy's config
                    atr_stop_multiplier = params.get('sl_atr_mult', 1.5) # Default if not specified
                    atr_target_multiplier = params.get('tp_atr_mult', 2.0) # Default if not specified
                    use_trailing_stop = params.get('use_trailing_sl', True) # Default if not specified
                    
                    self._process_strategy(
                        results=results, 
                        strategy_states=strategy_states,
                        name=name,
                        strategy_func=strategy_func, # Pass the function
                        params=params,              # Pass the params dict
                        current_idx=current_idx,
                        current_row=current_row,
                        data_slice=data_slice,       # Pass historical slice
                        atr_stop_multiplier=atr_stop_multiplier, # Pass specific SL/TP/TSL params
                        atr_target_multiplier=atr_target_multiplier,
                        use_trailing_stop=use_trailing_stop
                    )

            engine_logger.info("Backtest loop finished.")
            self.results_df = results
            return results

        except Exception as e:
            engine_logger.error(f"Backtest run failed: {str(e)}", exc_info=True)
            self.results_df = None
            return None

    # MODIFIED to use self.strategies_config
    def _initialize_strategy_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Initializes DataFrame columns and state dictionary for each strategy."""
        strategy_states = {}
        # Use keys from the config dictionary passed during init
        for name in self.strategies_config.keys(): 
            df[f'{name}_signal'] = 'hold'; df[f'{name}_position'] = ''; df[f'{name}_entry_price'] = np.nan
            df[f'{name}_exit_price'] = np.nan; df[f'{name}_pnl_points'] = 0.0; df[f'{name}_cumulative_pnl_points'] = 0.0
            df[f'{name}_sl'] = np.nan; df[f'{name}_tp'] = np.nan; df[f'{name}_atr_at_entry'] = np.nan
            df[f'{name}_trailing_sl'] = np.nan; df[f'{name}_trade_id'] = 0; df[f'{name}_active_trade'] = False
            # Initialize state dictionary
            strategy_states[name] = { 
                'active_trade': False, 'position': '', 'entry_price': np.nan,
                'initial_sl': np.nan, 'take_profit': np.nan, 'trailing_sl': np.nan,
                'highest_high_since_entry': np.nan, 'lowest_low_since_entry': np.nan,
                'trade_id': 0, 'cumulative_pnl_points': 0.0, 'atr_at_entry': np.nan 
            }
        return strategy_states

    # MODIFIED _process_strategy signature and call
    def _process_strategy(self, results: pd.DataFrame, strategy_states: Dict[str, Dict[str, Any]],
                         name: str, strategy_func: Callable, params: Dict, # Added params here
                         current_idx, current_row, data_slice: pd.DataFrame,
                         atr_stop_multiplier: float, atr_target_multiplier: float, use_trailing_stop: bool) -> None:
        
        state = strategy_states[name]
        
        try:
            # CORRECTED CALL: Pass the params dictionary to the strategy function
            # Also pass data_slice using a consistent name like 'data' if strategies expect it
            signal = strategy_func(current_row=current_row, 
                                 data=data_slice, # Renamed arg to match typical strategy def
                                 params=params)   # Pass the params dict
            
            # Ensure signal is valid
            if signal not in ['buy', 'sell', 'hold']:
                engine_logger.warning(f"Invalid signal '{signal}' from strategy '{name}'. Defaulting to 'hold'")
                signal = 'hold'
                
        except Exception as e:
            engine_logger.error(f"Strategy '{name}' failed at {current_idx}: {str(e)}", exc_info=True) # Log traceback
            signal = 'hold'
            
        results.loc[current_idx, f'{name}_signal'] = signal
        
        # --- Handle Active Trade ---
        if state['active_trade']:
            # Pass correct TSL multiplier from params
            trailing_sl_atr_mult = params.get('trailing_sl_atr_mult', 1.5) # Get TSL mult specific to this strategy
            if use_trailing_stop: 
                 self._update_trailing_stop(state, current_row, name, current_idx, trailing_sl_atr_mult) # Pass TSL multiplier

            exit_triggered, exit_price, exit_reason = self._check_exits(state, current_row, signal, name, current_idx)
            if exit_triggered:
                pnl_points = self._calculate_pnl_points(state['entry_price'], exit_price, state['position'])
                results.loc[current_idx, f'{name}_exit_price'] = exit_price; results.loc[current_idx, f'{name}_pnl_points'] = pnl_points
                state['cumulative_pnl_points'] += pnl_points; results.loc[current_idx, f'{name}_cumulative_pnl_points'] = state['cumulative_pnl_points']
                results.loc[current_idx, f'{name}_trade_id'] = state['trade_id']; results.loc[current_idx, f'{name}_active_trade'] = False; results.loc[current_idx, f'{name}_position'] = ''
                # Reset state (careful with shallow vs deep copies if state gets complex)
                strategy_states[name] = { # Re-initialize state cleanly
                     'active_trade': False, 'position': '', 'entry_price': np.nan,
                     'initial_sl': np.nan, 'take_profit': np.nan, 'trailing_sl': np.nan,
                     'highest_high_since_entry': np.nan, 'lowest_low_since_entry': np.nan,
                     'trade_id': state['trade_id'], # Keep last trade ID for exit row?
                     'cumulative_pnl_points': state['cumulative_pnl_points'], # Carry forward PnL
                     'atr_at_entry': np.nan
                 }
            else: # Carry forward active trade state in results DF
                results.loc[current_idx, f'{name}_active_trade'] = True; results.loc[current_idx, f'{name}_position'] = state['position']
                results.loc[current_idx, f'{name}_entry_price'] = state['entry_price']; results.loc[current_idx, f'{name}_sl'] = state['initial_sl']
                results.loc[current_idx, f'{name}_tp'] = state['take_profit']; results.loc[current_idx, f'{name}_trailing_sl'] = state['trailing_sl'] # Write current TSL to DF
                results.loc[current_idx, f'{name}_trade_id'] = state['trade_id']; results.loc[current_idx, f'{name}_cumulative_pnl_points'] = state['cumulative_pnl_points']
                results.loc[current_idx, f'{name}_atr_at_entry'] = state['atr_at_entry'] # Carry forward ATR at entry

        # --- Check for New Entries ---
        # Use elif to prevent entry on the same bar as an exit for the same strategy
        elif signal in ['buy', 'sell'] and not state['active_trade']: 
            entry_price_slippage, sl, tp, atr_at_entry = self._calculate_entry_sl_tp(current_row, signal, atr_stop_multiplier, atr_target_multiplier)
            if pd.notna(entry_price_slippage) and pd.notna(sl) and pd.notna(tp): # Ensure SL/TP are valid
                position_type = 'long' if signal == 'buy' else 'short'; results.loc[current_idx, f'{name}_position'] = position_type
                results.loc[current_idx, f'{name}_entry_price'] = entry_price_slippage; results.loc[current_idx, f'{name}_sl'] = sl
                results.loc[current_idx, f'{name}_tp'] = tp; results.loc[current_idx, f'{name}_atr_at_entry'] = atr_at_entry
                results.loc[current_idx, f'{name}_active_trade'] = True; state['trade_id'] += 1; results.loc[current_idx, f'{name}_trade_id'] = state['trade_id']
                results.loc[current_idx, f'{name}_cumulative_pnl_points'] = state['cumulative_pnl_points'] # PnL doesn't change on entry bar
                results.loc[current_idx, f'{name}_trailing_sl'] = sl # Initialize TSL

                # Update runtime state dictionary
                state['active_trade'] = True; state['position'] = position_type; state['entry_price'] = entry_price_slippage
                state['initial_sl'] = sl; state['take_profit'] = tp; state['trailing_sl'] = sl # Start TSL at initial SL
                state['highest_high_since_entry'] = current_row['high'] if signal == 'buy' else np.nan # Reset high/low tracking
                state['lowest_low_since_entry'] = current_row['low'] if signal == 'short' else np.nan
                state['atr_at_entry'] = atr_at_entry # Store ATR at time of entry

                engine_logger.info(f"🚀 ENTRY [{name}]: {state['position'].upper()} at {entry_price_slippage:.2f} (SL: {sl:.2f}, TP: {tp:.2f}) on {current_idx.strftime('%Y-%m-%d %H:%M')}")
            
            else: # Entry failed (e.g., NaN ATR)
                 results.loc[current_idx, f'{name}_cumulative_pnl_points'] = state['cumulative_pnl_points'] # Carry forward pnl if entry fails
                 results.loc[current_idx, f'{name}_active_trade'] = False
                 results.loc[current_idx, f'{name}_position'] = ''


        # --- No Active Trade, No Entry Signal ---
        else: # signal == 'hold' or (signal in ['buy','sell'] but already active_trade)
            # Ensure state columns are correctly propagated or reset
            results.loc[current_idx, f'{name}_signal'] = 'hold' 
            results.loc[current_idx, f'{name}_cumulative_pnl_points'] = state['cumulative_pnl_points']
            # Ensure non-active state is reflected if not already exited on this bar
            if not state['active_trade']: 
                 results.loc[current_idx, f'{name}_active_trade'] = False
                 results.loc[current_idx, f'{name}_position'] = ''
                 results.loc[current_idx, f'{name}_trade_id'] = state['trade_id'] # Keep last ID until new trade

    # MODIFIED signature for _update_trailing_stop
    def _update_trailing_stop(self, state: Dict[str, Any], current_row: pd.Series, name: str, current_idx, trailing_sl_atr_mult: float) -> None:
        """Updates the trailing stop loss based on price movement and ATR."""
        # Basic checks
        if not state['active_trade'] or pd.isna(state['trailing_sl']): return
        
        # Use ATR from current row if available, else fallback (though preprocess should ensure it)
        # Using current ATR makes the trail more adaptive to recent volatility
        current_atr = current_row.get('atr', state.get('atr_at_entry', np.nan)) 
        if pd.isna(current_atr) or current_atr <= 0: 
             # engine_logger.warning(f"TSL Update skipped for [{name}] at {current_idx}: Invalid ATR ({current_atr})")
             return # Cannot trail without valid ATR

        # Use the ATR multiplier specific to this strategy for trailing
        atr_multiplier = trailing_sl_atr_mult 
        
        new_trailing_sl = state['trailing_sl'] # Start with the current TSL

        if state['position'] == 'long':
            # Update highest high since entry
            if pd.isna(state['highest_high_since_entry']) or current_row['high'] > state['highest_high_since_entry']: 
                 state['highest_high_since_entry'] = current_row['high']
            
            # Calculate potential new TSL based on highest high and current ATR
            potential_stop = state['highest_high_since_entry'] - current_atr * atr_multiplier
            
            # TSL should only move up (increase) and should not go below initial SL? (Optional constraint)
            # Also common: TSL shouldn't move below entry price once breached (Breakeven+)
            # Simple version: Only move TSL up
            if potential_stop > new_trailing_sl: 
                new_trailing_sl = potential_stop
                # Optional: Log TSL update
                # engine_logger.debug(f"🔁 TSL Update [{name}]: LONG stop to {new_trailing_sl:.2f} on {current_idx.strftime('%Y-%m-%d %H:%M')}")
        
        elif state['position'] == 'short':
            # Update lowest low since entry
            if pd.isna(state['lowest_low_since_entry']) or current_row['low'] < state['lowest_low_since_entry']: 
                 state['lowest_low_since_entry'] = current_row['low']

            # Calculate potential new TSL based on lowest low and current ATR
            potential_stop = state['lowest_low_since_entry'] + current_atr * atr_multiplier
            
            # TSL should only move down (decrease)
            if potential_stop < new_trailing_sl: 
                new_trailing_sl = potential_stop
                # Optional: Log TSL update
                # engine_logger.debug(f"🔁 TSL Update [{name}]: SHORT stop to {new_trailing_sl:.2f} on {current_idx.strftime('%Y-%m-%d %H:%M')}")

        # Update the state dictionary with the new TSL
        state['trailing_sl'] = new_trailing_sl

    # _check_exits remains largely the same, ensures it uses state['trailing_sl']
    def _check_exits(self, state: Dict[str, Any], current_row: pd.Series, signal: str, name: str, current_idx) -> Tuple[bool, float, str]:
        """Checks for Stop Loss, Take Profit, Trailing SL, or Signal Reversal."""
        if not state['active_trade']: return False, np.nan, ""
        
        position_type = state['position']
        initial_stop_loss = state['initial_sl'] 
        trailing_stop_loss = state['trailing_sl'] # Use the potentially updated TSL from state
        take_profit = state['take_profit']
        
        exit_triggered = False; exit_price = np.nan; exit_reason = ""

        # 1. Check Trailing Stop Loss (use the value updated in _update_trailing_stop)
        if pd.notna(trailing_stop_loss): 
             if position_type == 'long' and current_row['low'] <= trailing_stop_loss: 
                 exit_triggered=True; exit_price=trailing_stop_loss; exit_reason="Trailing SL"
             elif position_type == 'short' and current_row['high'] >= trailing_stop_loss: 
                 exit_triggered=True; exit_price=trailing_stop_loss; exit_reason="Trailing SL"

        # 2. Check Initial Stop Loss (only if TSL wasn't hit)
        if not exit_triggered and pd.notna(initial_stop_loss): 
            if position_type == 'long' and current_row['low'] <= initial_stop_loss: 
                exit_triggered=True; exit_price=initial_stop_loss; exit_reason="Initial SL"
            elif position_type == 'short' and current_row['high'] >= initial_stop_loss: 
                exit_triggered=True; exit_price=initial_stop_loss; exit_reason="Initial SL"
        
        # 3. Check Take Profit (only if SL/TSL wasn't hit)
        if not exit_triggered and pd.notna(take_profit): 
             if position_type == 'long' and current_row['high'] >= take_profit: 
                 exit_triggered=True; exit_price=take_profit; exit_reason="Take Profit"
             elif position_type == 'short' and current_row['low'] <= take_profit: 
                 exit_triggered=True; exit_price=take_profit; exit_reason="Take Profit"

        # 4. Check Signal Reversal (only if SL/TSL/TP wasn't hit)
        # Note: Consider if reversal should override TP (e.g., if signal reverses before TP hit)
        # Current logic prioritizes SL/TSL/TP over reversal on the same bar.
        if not exit_triggered and signal in ['buy', 'sell']: 
            if (signal == 'buy' and position_type == 'short') or \
               (signal == 'sell' and position_type == 'long'): 
                 exit_triggered = True
                 exit_price = self._apply_slippage(current_row['close'], 'sell' if position_type == 'long' else 'buy') 
                 exit_reason = "Signal Reversal"
        
        # Log the exit if triggered
        if exit_triggered:
             # Ensure valid exit price (e.g., if SL/TP was NaN somehow) - use close as fallback?
             if pd.isna(exit_price): exit_price = current_row['close'] # Fallback needed? Should not happen if checks are right.
             log_symbol = "🛑" if "SL" in exit_reason else "❌" # Simplified symbol logic
             engine_logger.info(f"{log_symbol} EXIT [{name}]: {position_type.upper()} at {exit_price:.2f} due to {exit_reason} on {current_idx.strftime('%Y-%m-%d %H:%M')}")
        
        return exit_triggered, exit_price, exit_reason

    # _calculate_entry_sl_tp requires ATR
    def _calculate_entry_sl_tp(self, current_row: pd.Series, signal: str, 
                               atr_stop_multiplier: float, atr_target_multiplier: float
                               ) -> Tuple[float, float, float, float]:
        """Calculates entry price with slippage, SL, TP based on ATR."""
        atr = current_row.get('atr', np.nan) # Check lowercase 'atr'
        if pd.isna(atr) or atr <= 0: 
            engine_logger.warning(f"ATR is NaN or zero at {current_row.name}, cannot calculate SL/TP.")
            return np.nan, np.nan, np.nan, np.nan
            
        entry_price_slippage = self._apply_slippage(current_row['close'], signal)
        if pd.isna(entry_price_slippage): return np.nan, np.nan, np.nan, np.nan # If slippage calc failed
        
        atr_value = atr # Use the ATR value from the current row for calculation
        
        if signal == 'buy': 
            sl = entry_price_slippage - atr_value * atr_stop_multiplier
            tp = entry_price_slippage + atr_value * atr_target_multiplier
        else: # signal == 'sell'
            sl = entry_price_slippage + atr_value * atr_stop_multiplier
            tp = entry_price_slippage - atr_value * atr_target_multiplier
            
        return entry_price_slippage, sl, tp, atr_value # Return ATR used for state['atr_at_entry']

    # _calculate_pnl_points remains the same
    def _calculate_pnl_points(self, entry_price: float, exit_price: float, position_type: str) -> float:
        if pd.isna(entry_price) or pd.isna(exit_price): return 0.0
        if position_type == 'long': gross_pnl_points = exit_price - entry_price
        elif position_type == 'short': gross_pnl_points = entry_price - exit_price
        else: return 0.0
        commission_points_entry = entry_price * self.commission_pct; commission_points_exit = exit_price * self.commission_pct
        net_pnl_points = gross_pnl_points - (commission_points_entry + commission_points_exit)
        return net_pnl_points

    # _apply_slippage remains the same
    def _apply_slippage(self, price: float, signal: str) -> float:
        if pd.isna(price): return np.nan
        slippage_amount = price * self.slippage_pct
        if signal == 'buy': return price + slippage_amount
        elif signal == 'sell': return price - slippage_amount
        return price