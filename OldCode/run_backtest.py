import pandas as pd
import numpy as np
from datetime import time, datetime
import logging
from io import StringIO # To read string data
from typing import Optional, List, Dict, Any # Added for type hinting

# --- Import the Strategy ---
# Assumes 'my_alphatrend_strategy.py' is in the current directory
# or in a directory included in the Python path.
try:
    # If strategy file is in the same directory:
    from csalphav2 import AlphaTrendStrategy
    # If strategy file is in a 'strategy' subdirectory:
    # from strategy.my_alphatrend_strategy import AlphaTrendStrategy
except ImportError as e:
    print(f"Error: Could not import AlphaTrendStrategy: {e}")
    print("Make sure 'my_alphatrend_strategy.py' exists and is accessible.")
    exit()


# --- Configure Logging ---
# Configure basic logging for the backtest script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Use a specific logger name for this script
logger = logging.getLogger('BacktestRunner')


# --- Hardcoded Data ---
data_string = """datetime,open,high,low,close,volume
2025-04-29 09:15:00+05:30,24370.7,24442.25,24364.35,24439.25,0
2025-04-29 09:20:00+05:30,24438.5,24455.05,24424.5,24453.7,0
2025-04-29 09:25:00+05:30,24453.6,24457.65,24413.3,24417.1,0
2025-04-29 09:30:00+05:30,24418.25,24452.9,24393.55,24440.2,0
2025-04-29 09:35:00+05:30,24440.85,24442.5,24359.2,24369.45,0
2025-04-29 09:40:00+05:30,24370.1,24387.05,24303.9,24316.6,0
2025-04-29 09:45:00+05:30,24317.85,24364.5,24308.1,24315.55,0
2025-04-29 09:50:00+05:30,24316.75,24340.05,24290.75,24329.1,0
2025-04-29 09:55:00+05:30,24330.15,24395,24326.5,24390.25,0
2025-04-29 10:00:00+05:30,24390,24396.15,24362.2,24378.3,0
2025-04-29 10:05:00+05:30,24379.15,24379.75,24333.8,24344.65,0
2025-04-29 10:10:00+05:30,24345.2,24365.1,24338.55,24351.55,0
2025-04-29 10:15:00+05:30,24351.3,24354.1,24319,24319,0
2025-04-29 10:20:00+05:30,24318.1,24347.1,24317.4,24346.4,0
2025-04-29 10:25:00+05:30,24346.45,24348.15,24311.3,24317.8,0
2025-04-29 10:30:00+05:30,24318.85,24329.7,24305.5,24314.95,0
2025-04-29 10:35:00+05:30,24316.2,24335.25,24314.55,24331.5,0
2025-04-29 10:40:00+05:30,24331.55,24333.9,24312.8,24321.5,0
2025-04-29 10:45:00+05:30,24321.85,24352.35,24317.95,24350.7,0
2025-04-29 10:50:00+05:30,24349.85,24353.7,24328.45,24330.85,0
2025-04-29 10:55:00+05:30,24330.25,24349.95,24330.25,24333.8,0
2025-04-29 11:00:00+05:30,24334.7,24351.9,24332.8,24351.9,0
2025-04-29 11:05:00+05:30,24353.05,24369.85,24351.9,24367.1,0
2025-04-29 11:10:00+05:30,24367.25,24369.7,24350.6,24356.05,0
2025-04-29 11:15:00+05:30,24356.55,24358.5,24338.95,24348.8,0
2025-04-29 11:20:00+05:30,24348.3,24360.35,24347.05,24348.95,0
2025-04-29 11:25:00+05:30,24349.05,24358.75,24347.6,24349.25,0
2025-04-29 11:30:00+05:30,24348.65,24351.6,24334.95,24347.65,0
2025-04-29 11:35:00+05:30,24347.95,24353.6,24341.6,24346.45,0
2025-04-29 11:40:00+05:30,24346.7,24356.8,24342.2,24354.4,0
2025-04-29 11:45:00+05:30,24354.65,24355.8,24335.3,24340.15,0
2025-04-29 11:50:00+05:30,24340.8,24340.85,24321,24322.95,0
2025-04-29 11:55:00+05:30,24323.4,24329.1,24311.95,24313.25,0
2025-04-29 12:00:00+05:30,24313.4,24322.5,24302.45,24322.05,0
2025-04-29 12:05:00+05:30,24322.15,24323.75,24310.65,24315.5,0
2025-04-29 12:10:00+05:30,24315.3,24322.3,24309.2,24316.6,0
2025-04-29 12:15:00+05:30,24316.4,24321.6,24310.65,24313.1,0
2025-04-29 12:20:00+05:30,24315.1,24315.85,24304.1,24313.65,0
2025-04-29 12:25:00+05:30,24312.95,24317.25,24306.6,24309.75,0
2025-04-29 12:30:00+05:30,24309.25,24330.8,24309.15,24329.95,0
2025-04-29 12:35:00+05:30,24328.75,24347.95,24325.4,24341.35,0
2025-04-29 12:40:00+05:30,24340.6,24346.8,24336.4,24341.85,0
2025-04-29 12:45:00+05:30,24338.8,24344.85,24330.6,24339.55,0
2025-04-29 12:50:00+05:30,24339.15,24358.35,24336.45,24357.9,0
2025-04-29 12:55:00+05:30,24357.75,24363.55,24353,24358.95,0
2025-04-29 13:00:00+05:30,24358.25,24369.5,24352.2,24368.4,0
2025-04-29 13:05:00+05:30,24368.05,24368.5,24350.6,24360.1,0
2025-04-29 13:10:00+05:30,24359.4,24363.55,24353.5,24356.85,0
2025-04-29 13:15:00+05:30,24355.45,24359.35,24341.35,24354.1,0
2025-04-29 13:20:00+05:30,24352.2,24363.15,24346.55,24362.9,0
2025-04-29 13:25:00+05:30,24362.1,24376.05,24356,24366.05,0
2025-04-29 13:30:00+05:30,24366.65,24371.7,24361.45,24366.8,0
2025-04-29 13:35:00+05:30,24366.5,24368.8,24357.4,24366.15,0
2025-04-29 13:40:00+05:30,24367.15,24372.7,24351,24353.9,0
2025-04-29 13:45:00+05:30,24352.7,24366.95,24351.75,24363.45,0
2025-04-29 13:50:00+05:30,24364,24368.75,24355.1,24355.65,0
2025-04-29 13:55:00+05:30,24356.65,24363.4,24347.65,24358.25,0
2025-04-29 14:00:00+05:30,24360.15,24368.3,24356.85,24363.55,0
2025-04-29 14:05:00+05:30,24363.8,24375.5,24360.6,24365.2,0
2025-04-29 14:10:00+05:30,24365.35,24374.8,24360.25,24365.95,0
2025-04-29 14:15:00+05:30,24366.9,24368.3,24355.05,24361.65,0
2025-04-29 14:20:00+05:30,24361.9,24367.55,24356.25,24357.15,0
2025-04-29 14:25:00+05:30,24357.75,24359.7,24342.4,24346,0
2025-04-29 14:30:00+05:30,24346.45,24350.5,24339,24343.45,0
2025-04-29 14:35:00+05:30,24343.75,24361.95,24343.75,24348.1,0
2025-04-29 14:40:00+05:30,24348.8,24360.7,24347.2,24356.45,0
2025-04-29 14:45:00+05:30,24356.2,24357.95,24347.85,24352.95,0
2025-04-29 14:50:00+05:30,24353.35,24358,24344.85,24346.85,0
2025-04-29 14:55:00+05:30,24346.8,24355.7,24339.45,24340.2,0
2025-04-29 15:00:00+05:30,24340.55,24347.8,24325.2,24331.4,0
2025-04-29 15:05:00+05:30,24331.55,24336.5,24324.9,24332.7,0
2025-04-29 15:10:00+05:30,24332.6,24335.6,24324.45,24326.45,0
2025-04-29 15:15:00+05:30,24326.2,24341.45,24310.6,24339.65,0
2025-04-29 15:20:00+05:30,24338.85,24348.5,24325.85,24333.1,0
2025-04-29 15:25:00+05:30,24331.55,24340.85,24317.6,24325.45,0"""


# --- Detailed Backtesting Function ---
def run_detailed_backtest(df_signals: pd.DataFrame, strategy: AlphaTrendStrategy) -> None:
    """
    Runs a detailed backtest simulation based on signals from AlphaTrendStrategy.
    Includes logic to tighten TSL based on opposite condition scores.

    Args:
        df_signals (pd.DataFrame): DataFrame with OHLC, indicators, position, signal, exit_reason.
        strategy (AlphaTrendStrategy): The instantiated strategy object to access parameters.
    """
    trades = []
    backtest_log = []
    active_trade: Optional[Dict[str, Any]] = None # Holds current open trade details
    trade_counter = 0
    score_flip_threshold = 3 # Number of opposite conditions met to trigger TSL tightening

    # Get necessary parameters from strategy object
    atr_strategy_col = f'atr_sma_{strategy.ap}'
    rsi_col = f'rsi_{strategy.ap}'
    sl_pct = 1.0 # Example SL %, could be made dynamic or passed if needed

    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'signal', atr_strategy_col, rsi_col,
                     'buy_conditions_met', 'sell_conditions_met', 'high_shift1', 'low_shift1']
    if not all(col in df_signals.columns for col in required_cols):
        logger.error(f"Backtest input missing columns: {[c for c in required_cols if c not in df_signals.columns]}")
        return

    logger.info("Starting detailed backtest with Score Flip TSL...")
    print("\n--- Detailed Backtest Log ---")

    for idx, row in df_signals.iterrows():
        current_time = idx.time()
        current_price = row['close']
        signal = row['signal']
        current_atr = row[atr_strategy_col]
        current_rsi = row[rsi_col]
        buy_conditions_met = row['buy_conditions_met']
        sell_conditions_met = row['sell_conditions_met']

        log_entry = f"[{idx}] Px:{current_price:.2f} | "
        trade_info = ""
        tsl_tightened = False # Flag to indicate if TSL was tightened this bar due to score flip

        # --- TSL Update ---
        if active_trade:
            side = active_trade['side']
            # Update highest/lowest price since entry
            if side == 'long':
                active_trade['high_since_entry'] = max(active_trade['high_since_entry'], row['high'])
            elif side == 'short':
                active_trade['low_since_entry'] = min(active_trade['low_since_entry'], row['low'])

            # Calculate potential new TSL only after 09:30
            if current_time >= time(9, 30) and pd.notna(current_atr) and current_atr > 0:
                atr_mult = np.nan
                # Determine base ATR multiplier (RSI or default)
                if side == 'long':
                    atr_mult = strategy.atr_stop_mult_tight if current_rsi > strategy.rsi_tighten_long else strategy.atr_stop_mult_late
                    # *** Score Flip Check for Long ***
                    if sell_conditions_met >= score_flip_threshold:
                        atr_mult = strategy.atr_stop_mult_tight # Use tight multiplier if score flips
                        tsl_tightened = True
                        # logger.debug(f"TSL Tightened (Long) at {idx} due to Sell Score {sell_conditions_met}>={score_flip_threshold}")

                    potential_tsl = active_trade['high_since_entry'] - (atr_mult * current_atr)
                    active_trade['tsl'] = np.nanmax([active_trade['tsl'], potential_tsl]) # Trail up

                elif side == 'short':
                    atr_mult = strategy.atr_stop_mult_tight if current_rsi < strategy.rsi_tighten_short else strategy.atr_stop_mult_late
                    # *** Score Flip Check for Short ***
                    if buy_conditions_met >= score_flip_threshold:
                         atr_mult = strategy.atr_stop_mult_tight # Use tight multiplier if score flips
                         tsl_tightened = True
                         # logger.debug(f"TSL Tightened (Short) at {idx} due to Buy Score {buy_conditions_met}>={score_flip_threshold}")

                    potential_tsl = active_trade['low_since_entry'] + (atr_mult * current_atr)
                    active_trade['tsl'] = np.nanmin([active_trade['tsl'], potential_tsl]) # Trail down

            # Format trade info string for logging
            sl_str = f"{active_trade['initial_sl']:.2f}" if pd.notna(active_trade['initial_sl']) else "N/A"
            tsl_val_str = f"{active_trade['tsl']:.2f}" if pd.notna(active_trade['tsl']) else "N/A"
            tsl_info_str = f"| TSL:{tsl_val_str}{'*' if tsl_tightened else ''}" if pd.notna(active_trade['tsl']) and active_trade['tsl'] != active_trade['initial_sl'] else ""

            trade_info = (f"Running {active_trade['id']} ({side.upper()}) | "
                          f"Entry:{active_trade['entry_price']:.2f} | "
                          f"SL:{sl_str} "
                          f"{tsl_info_str}") # Append TSL info


        # --- Exit Check ---
        if active_trade:
            side = active_trade['side']
            initial_sl = active_trade['initial_sl']
            tsl = active_trade['tsl']
            exit_reason = None
            exit_price = None

            # Check SL/TSL based on low/high of the current bar
            if side == 'long':
                if pd.notna(initial_sl) and row['low'] <= initial_sl: exit_reason, exit_price = "SL Hit", initial_sl
                elif pd.notna(tsl) and row['low'] <= tsl: exit_reason, exit_price = "TSL Hit", tsl
            elif side == 'short':
                if pd.notna(initial_sl) and row['high'] >= initial_sl: exit_reason, exit_price = "SL Hit", initial_sl
                elif pd.notna(tsl) and row['high'] >= tsl: exit_reason, exit_price = "TSL Hit", tsl

            # Check strategy exit signal (exit happens at close of the signal bar)
            if not exit_reason:
                if side == 'long' and signal == 2: exit_reason, exit_price = "Strategy Exit Signal", current_price
                elif side == 'short' and signal == -2: exit_reason, exit_price = "Strategy Exit Signal", current_price

            # Check other exit reasons generated by the strategy
            if not exit_reason and row['exit_reason'] and row['exit_reason'] != '':
                exit_reason, exit_price = row['exit_reason'], current_price

            if exit_reason:
                # Clip exit price
                exit_price_calc = exit_price if pd.notna(exit_price) else current_price
                exit_price_final = np.clip(exit_price_calc, row['low'], row['high'])

                profit = (exit_price_final - active_trade['entry_price']) if side == 'long' else (active_trade['entry_price'] - exit_price_final)
                active_trade['exit_time'] = idx
                active_trade['exit_price'] = exit_price_final
                active_trade['exit_reason'] = exit_reason
                active_trade['profit'] = profit
                trades.append(active_trade.copy())
                log_entry += f"EXIT {active_trade['id']} ({side.upper()}) @ {exit_price_final:.2f} ({exit_reason}). PnL: {profit:.2f}"
                active_trade = None
            else:
                 log_entry += trade_info

        # --- Entry Check ---
        if not active_trade:
            entry_price = row['close']
            side = None
            conditions_met = 0
            initial_sl = np.nan

            if signal == 1: # Enter Long
                side = 'long'
                conditions_met = row['buy_conditions_met']
                if pd.notna(current_atr) and current_atr > 0:
                    time_check = current_time < time(9, 30); atr_mult = strategy.atr_stop_mult_early if time_check else strategy.atr_stop_mult_late
                    prev_high = row['high_shift1']; prev_low = row['low_shift1']
                    if time_check and pd.notna(prev_high) and pd.notna(prev_low): prev_range = prev_high - prev_low; initial_sl_offset = (0.5 * prev_range + atr_mult * current_atr) if prev_range >= 0 else (atr_mult * current_atr)
                    else: initial_sl_offset = atr_mult * current_atr
                    initial_sl = (entry_price - initial_sl_offset) if pd.notna(initial_sl_offset) and initial_sl_offset > 0 else np.nan

            elif signal == -1: # Enter Short
                side = 'short'
                conditions_met = row['sell_conditions_met']
                if pd.notna(current_atr) and current_atr > 0:
                    time_check = current_time < time(9, 30); atr_mult = strategy.atr_stop_mult_early if time_check else strategy.atr_stop_mult_late
                    prev_high = row['high_shift1']; prev_low = row['low_shift1']
                    if time_check and pd.notna(prev_high) and pd.notna(prev_low): prev_range = prev_high - prev_low; initial_sl_offset = (0.5 * prev_range + atr_mult * current_atr) if prev_range >= 0 else (atr_mult * current_atr)
                    else: initial_sl_offset = atr_mult * current_atr
                    initial_sl = (entry_price + initial_sl_offset) if pd.notna(initial_sl_offset) and initial_sl_offset > 0 else np.nan

            if side and pd.notna(initial_sl):
                trade_counter += 1
                trade_id = f"{side.capitalize()}_{trade_counter}_{idx.strftime('%H%M')}"
                active_trade = {
                    'id': trade_id, 'side': side, 'entry_time': idx, 'entry_price': entry_price,
                    'initial_sl': initial_sl, 'tsl': initial_sl,
                    'high_since_entry': entry_price if side == 'long' else -np.inf,
                    'low_since_entry': entry_price if side == 'short' else np.inf,
                    'conditions_met': conditions_met,
                    'exit_time': None, 'exit_price': None, 'exit_reason': None, 'profit': None
                }
                sl_log_str = f"{active_trade['initial_sl']:.2f}" if pd.notna(active_trade['initial_sl']) else "N/A"
                log_entry += (f"ENTER {side.upper()} | ID:{trade_id} | Entry:{entry_price:.2f} | "
                              f"SL:{sl_log_str} | CondMet:{conditions_met:.1f}")
            elif signal in [1, -1] and not pd.notna(initial_sl):
                 log_entry += f"Hold (Entry Signal {signal} Skipped - Invalid SL Calc)"
            elif trade_info == "":
                log_entry += "Hold"

        # Append log entry for the current bar
        print(log_entry) # Print log entry to console
        backtest_log.append(log_entry)

    # --- Post-Backtest Summary ---
    print("\n--- Backtest Summary ---")
    total_trades = len(trades)
    if total_trades > 0:
        valid_profits = [t['profit'] for t in trades if pd.notna(t['profit'])]
        total_pnl = sum(valid_profits)
        winning_trades = sum(1 for pnl in valid_profits if pnl > 0)
        losing_trades = total_trades - winning_trades
        accuracy = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Total PnL: {total_pnl:.2f}")

        # --- Print Trade Details to Console ---
        print("\n--- Trade Details ---")
        trades_df = pd.DataFrame(trades)
        trade_cols_order = ['id', 'side', 'entry_time', 'entry_price', 'initial_sl',
                           'exit_time', 'exit_price', 'exit_reason', 'profit', 'conditions_met']
        trade_cols_order = [col for col in trade_cols_order if col in trades_df.columns]
        if not trades_df.empty:
            print(trades_df[trade_cols_order].to_string(index=False, float_format='%.2f'))
        else: print("No completed trade details to display.")

        # Save detailed trades to CSV
        trades_filename = f"detailed_trades_output_{datetime.now():%Y%m%d_%H%M%S}.csv"
        try: trades_df[trade_cols_order].to_csv(trades_filename, index=False, float_format='%.2f'); logger.info(f"Detailed trade list saved to {trades_filename}")
        except Exception as e: logger.error(f"Failed to save detailed trades: {e}")
    else: print("No trades were executed.")

    # Save detailed log to file
    log_filename = f"detailed_backtest_log_{datetime.now():%Y%m%d_%H%M%S}.txt"
    try:
        with open(log_filename, 'w') as f: f.write('\n'.join(backtest_log))
        logger.info(f"Detailed backtest log saved to {log_filename}")
    except Exception as e: logger.error(f"Failed to save backtest log: {e}")


# --- Script Execution ---
# Load Data
df = pd.read_csv(StringIO(data_string))
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime'])
df = df.set_index('datetime')

# Ensure Timezone
target_tz = 'Asia/Kolkata'
if df.index.tz is None:
    try: df.index = df.index.tz_localize(target_tz, ambiguous='infer'); logger.info(f"Localized to {target_tz}")
    except Exception as e: logger.warning(f"TZ localize failed: {e}")
elif str(df.index.tz) != target_tz:
    try: df.index = df.index.tz_convert(target_tz); logger.info(f"Converted to {target_tz}")
    except Exception as e: logger.warning(f"TZ convert failed: {e}")

# Instantiate and Run Strategy to get signals
try:
    logger.info("Instantiating MORE Relaxed AlphaTrendStrategy...")
    strategy = AlphaTrendStrategy() # Uses the MORE relaxed __init__ defaults

    logger.info("Running MORE Relaxed AlphaTrendStrategy.generate_signals...")
    df_with_signals = strategy.generate_signals(df.copy()) # Pass a copy
    logger.info("Signal generation complete.")

    # --- Run Detailed Backtest ---
    if not df_with_signals.empty:
        run_detailed_backtest(df_with_signals, strategy)
    else:
        print("\nStrategy did not generate signals. Cannot run backtest.")

except Exception as e:
    logger.error(f"Error during MORE Relaxed strategy execution or backtest: {e}", exc_info=True)
