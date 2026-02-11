import pandas as pd
import numpy as np
import logging
from datetime import time

logger = logging.getLogger(__name__)

def run_simulation(
    data: pd.DataFrame,
    strategy, # Strategy object is passed, contains parameters like atr multipliers
    sl_pct: float = 1.0,
    trail_pct: float = 0.5, # Note: This parameter is defined but not used in the logic below
    commission_per_trade: float = 0.0, # Applied once on entry in this logic
    slippage_per_trade: float = 0.0 # Applied once on entry and once on exit
) -> dict:
    """
    Runs a backtesting simulation based on strategy signals.

    Args:
        data (pd.DataFrame): DataFrame containing OHLC data, indicators, and a 'position' column
                             (-1 for exit signal, 1 for entry signal, 0 for hold).
                             Must have a DatetimeIndex or columns convertible to it.
                             Requires specific columns like 'close', 'high', 'low', 'rsi_10',
                             'rsi_slope', 'macd', 'macd_signal', 'atr_14'.
        strategy: The strategy object (used to access strategy-specific parameters like ATR multipliers).
        sl_pct (float): Stop loss percentage.
        trail_pct (float): Trailing stop percentage (parameter exists but logic uses ATR).
        commission_per_trade (float): Commission cost added to entry price.
        slippage_per_trade (float): Slippage cost added to entry and subtracted from exit.

    Returns:
        dict: Simulation results including PnL, trade count, status, and trade details.
    """
    # --- Input Data Validation and Setup ---
    df = data.copy()
    logger.info(f"Starting simulation. Input data shape: {df.shape}")

    # Define columns required by the simulation logic itself
    required_cols = ['close', 'position', 'high', 'low', 'rsi_10', 'rsi_slope', 'macd', 'macd_signal', 'atr_14']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if df.empty or missing_cols:
        logger.error(f"Input DataFrame empty or missing required columns: {missing_cols}")
        return {'pnl': 0.0, 'trades': 0, 'status': 'Input Error - Missing Columns', 'trades_details': [], 'parameters': {}}

    # Ensure index is datetime
    has_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    if not has_datetime_index:
        logger.warning("Input DataFrame does not have a DatetimeIndex. Time-based logic (morning ATR, session end) will be skipped.")

    # --- Initialize Simulation State ---
    pnl = 0.0 # Tracks realized PnL
    position_active = 0 # 0 = Flat, 1 = Long (only long implemented here)
    entry_price = 0.0 # Tracks effective entry price (incl. commission/slippage)
    raw_entry_price = 0.0 # Tracks actual entry price before costs
    stop_loss_level = 0.0
    trailing_stop_level = 0.0
    highest_price_since_entry = 0.0
    trades = 0
    trades_details = [] # List to store details of each trade

    # --- Get Strategy Parameters (Safely using getattr) ---
    # Get ATR multipliers from the strategy object, provide defaults if not found
    atr_stop_mult_early = getattr(strategy, 'atr_stop_mult_early', 2.5) # Default based on AlphaTrend
    atr_stop_mult_late = getattr(strategy, 'atr_stop_mult_late', 3.5)   # Default based on AlphaTrend
    atr_stop_mult_tight = getattr(strategy, 'atr_stop_mult_tight', 2.0) # Default based on AlphaTrend
    rsi_tighten_long = getattr(strategy, 'rsi_tighten_long', 75)         # Default based on AlphaTrend
    rsi_tighten_short = getattr(strategy, 'rsi_tighten_short', 25)       # Default based on AlphaTrend (though only long side implemented here)
    rsi_extreme_long = getattr(strategy, 'rsi_extreme_long', 80)         # Default based on AlphaTrend
    rsi_extreme_short = getattr(strategy, 'rsi_extreme_short', 20)        # Default based on AlphaTrend

    logger.info(f"Using ATR Multipliers: Early={atr_stop_mult_early}, Late={atr_stop_mult_late}, Tight={atr_stop_mult_tight}")
    logger.info(f"Using RSI Thresholds: Tighten=({rsi_tighten_short}/{rsi_tighten_long}), Extreme=({rsi_extreme_short}/{rsi_extreme_long})")


    # --- Pre-calculations ---
    # Calculate morning ATR (09:15–09:30) if possible
    atr_morning = np.nan # Default to NaN
    if has_datetime_index:
        try:
            # Filter data for the morning session
            morning_data = df[(df.index.time >= time(9, 15)) & (df.index.time <= time(9, 30))]
            if not morning_data.empty and 'atr_14' in morning_data.columns:
                # Calculate mean ATR during that period, handle potential NaNs in ATR column
                atr_morning = morning_data['atr_14'].mean()
                if pd.isna(atr_morning):
                     logger.warning("Morning ATR calculation resulted in NaN (likely NaNs in atr_14 during morning).")
                else:
                     logger.info(f"Calculated Morning ATR (9:15-9:30): {atr_morning:.2f}")
            else:
                logger.warning("Could not calculate morning ATR: 'atr_14' missing or no data between 9:15-9:30.")
        except Exception as e:
            logger.error(f"Error calculating morning ATR: {e}", exc_info=True)


    # Define base columns needed for iteration + indicators to log
    log_indicators = ['rsi_10', 'macd', 'macd_signal', 'atr_14'] # Add 'AlphaTrend' if it exists and you want to log it
    base_iter_cols = ['close', 'position', 'high', 'low', 'rsi_10', 'rsi_slope', 'macd', 'macd_signal', 'atr_14']
    iter_cols = list(set(base_iter_cols + [ind for ind in log_indicators if ind in df.columns])) # Only include existing log indicators

    # Pre-calculate previous MACD values robustly
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        df['macd_prev'] = df['macd'].shift(1)
        df['macd_signal_prev'] = df['macd_signal'].shift(1)
        # Add the new columns to the list if they were created successfully
        iter_cols.extend(['macd_prev', 'macd_signal_prev'])
        logger.info("Pre-calculated macd_prev and macd_signal_prev.")
    else:
        logger.warning("MACD columns ('macd', 'macd_signal') not found. Skipping MACD crossover logic and prev value calculation.")
        # Ensure these columns exist as NaN if logic below relies on them via getattr
        if 'macd_prev' not in df.columns: df['macd_prev'] = np.nan
        if 'macd_signal_prev' not in df.columns: df['macd_signal_prev'] = np.nan

    # Ensure only columns that actually exist in the DataFrame are requested by itertuples
    iter_cols = [col for col in iter_cols if col in df.columns]
    logger.info(f"Columns used in simulation loop: {iter_cols}")

    # --- Simulation Loop ---
    logger.info("Starting simulation loop...")
    print("\n--- Simulation Row-by-Row ---") # Header for console output

    # Use itertuples for efficient row iteration
    for row_data in df[iter_cols].itertuples():
        # Get index (time)
        idx = row_data.Index
        time_key = idx.isoformat() if has_datetime_index and hasattr(idx, 'isoformat') else str(idx)

        # Get essential data for the current row using getattr for safety
        price = getattr(row_data, 'close', np.nan)
        position_signal = getattr(row_data, 'position', 0) # Signal from strategy (1=Buy, -1=Sell/Exit)
        row_high = getattr(row_data, 'high', price) # Use price if high is missing
        row_low = getattr(row_data, 'low', price)   # Use price if low is missing

        # Get indicator values using getattr
        rsi = getattr(row_data, 'rsi_10', np.nan)
        rsi_slope = getattr(row_data, 'rsi_slope', np.nan)
        macd = getattr(row_data, 'macd', np.nan)
        macd_signal = getattr(row_data, 'macd_signal', np.nan)
        atr = getattr(row_data, 'atr_14', np.nan)
        macd_prev = getattr(row_data, 'macd_prev', np.nan) # Will be NaN if not calculated
        macd_signal_prev = getattr(row_data, 'macd_signal_prev', np.nan) # Will be NaN if not calculated

        # Basic check: If price is NaN, cannot proceed for this row
        if pd.isna(price):
            # logger.debug(f"Skipping row {idx}: Price is NaN.")
            continue

        # Determine strategy signals (more descriptive names)
        signal_enter_long = position_signal == 1
        signal_exit_long = position_signal == -1 # Assuming -1 signal means exit long

        # --- Logging Current State ---
        print(f"\n[{time_key}] Px: {price:.2f} | Pos Chg: {position_signal:2d} | InPos: {'Y' if position_active==1 else 'N'}", end="")
        indicator_vals_str = []
        for ind in log_indicators:
             if hasattr(row_data, ind): # Check if indicator exists in the row tuple
                 val = getattr(row_data, ind)
                 indicator_vals_str.append(f"{ind.split('_')[0][:4]}: {val:.2f}" if pd.notna(val) else f"{ind.split('_')[0][:4]}: NaN")
        if indicator_vals_str: print(" | " + " | ".join(indicator_vals_str), end="")
        if position_active == 1: print(f" | Entry: {raw_entry_price:.2f} | SL: {stop_loss_level:.2f} | TSL: {trailing_stop_level:.2f} | HighSince: {highest_price_since_entry:.2f}")
        else: print() # Newline if not in position

        action_taken = "Hold" # Default action

        # --- Trailing Stop Loss Calculation (only when in a long position) ---
        if position_active == 1:
            # Update highest price since entry
            highest_price_since_entry = max(highest_price_since_entry, row_high) if pd.notna(row_high) else highest_price_since_entry

            # Calculate ATR multiplier for trailing stop
            current_trail_multiplier = np.nan
            if pd.notna(atr): # Need current ATR
                idx_time = idx.time() if has_datetime_index else None
                # Determine base ATR: use morning ATR if available and within morning, else use current ATR
                base_atr_for_trail = atr_morning if (idx_time and idx_time <= time(9, 30) and pd.notna(atr_morning)) else atr

                # Determine multiplier based on time and RSI
                if idx_time and idx_time <= time(9, 30):
                    # Morning session: Use fixed multiplier * base ATR
                    # *** CORRECTED: Access attribute via strategy object ***
                    current_trail_multiplier = atr_stop_mult_early * base_atr_for_trail
                else:
                    # Afternoon session: Use base multiplier * current ATR, tighten if RSI extreme
                    # *** CORRECTED: Access attributes via strategy object ***
                    atr_mult = atr_stop_mult_tight if (pd.notna(rsi) and (rsi > rsi_tighten_long)) else atr_stop_mult_late
                    current_trail_multiplier = atr_mult * atr # Use current ATR for afternoon

                # Update trailing stop level if calculation is valid
                if pd.notna(current_trail_multiplier) and pd.notna(highest_price_since_entry) and current_trail_multiplier > 0:
                    potential_tsl = highest_price_since_entry - current_trail_multiplier
                    # Trail the stop: only move it up (max of current TSL and potential new TSL)
                    # Use np.nanmax to handle initialization where trailing_stop_level might be NaN or 0
                    trailing_stop_level = np.nanmax([trailing_stop_level, potential_tsl])
                # else: logger.debug(f"TSL calc skipped at {idx}: Multiplier={current_trail_multiplier}, High={highest_price_since_entry}")
            # else: logger.debug(f"TSL calc skipped at {idx}: ATR is NaN")


        # --- Exit Conditions Check (only when in a long position) ---
        if position_active == 1:
            exit_trade = False
            exit_reason = None
            chosen_exit_price = price # Default exit price is current close

            # Check MACD Crossover (using pre-calculated previous values)
            is_macd_crossover = (pd.notna(macd) and pd.notna(macd_signal) and
                                 pd.notna(macd_prev) and pd.notna(macd_signal_prev) and
                                 macd < macd_signal and macd_prev >= macd_signal_prev)

            # Evaluate exit conditions in order of priority (e.g., SL before TSL)
            if pd.notna(row_low) and pd.notna(stop_loss_level) and row_low <= stop_loss_level:
                exit_trade = True; exit_reason = "Stop Loss"; chosen_exit_price = stop_loss_level
            elif pd.notna(row_low) and pd.notna(trailing_stop_level) and row_low <= trailing_stop_level:
                exit_trade = True; exit_reason = "Trailing Stop"; chosen_exit_price = trailing_stop_level
            elif signal_exit_long: # Check strategy's explicit exit signal
                exit_trade = True; exit_reason = "Strategy Exit Signal"
            # *** CORRECTED: Use RSI thresholds from strategy object ***
            elif pd.notna(rsi) and (rsi > rsi_extreme_long): # RSI Overbought
                exit_trade = True; exit_reason = f"RSI Extreme (> {rsi_extreme_long})"
            elif pd.notna(rsi) and (rsi < rsi_extreme_short): # RSI Oversold
                 exit_trade = True; exit_reason = f"RSI Extreme (< {rsi_extreme_short})"
            # elif pd.notna(rsi_slope) and rsi_slope < -2: # Optional: Exit on sharp RSI drop
            #     exit_trade = True; exit_reason = f"RSI Slope Drop ({rsi_slope:.2f})"
            elif is_macd_crossover: # MACD Bearish Crossover
                exit_trade = True; exit_reason = "MACD Crossover"

            # Session end exits (only if index is datetime)
            if not exit_trade and has_datetime_index: # Check only if not already exiting
                 idx_time = idx.time()
                 # Force exit at 15:25 regardless of conditions
                 if idx_time >= time(15, 25):
                     exit_trade = True; exit_reason = "Session End Time (15:25)"

            # --- Process Exit ---
            if exit_trade:
                # Ensure exit price isn't higher than the row's high or lower than low
                chosen_exit_price = np.clip(chosen_exit_price, row_low, row_high) if pd.notna(row_low) and pd.notna(row_high) else chosen_exit_price

                action_taken = f"Exit Long ({exit_reason} @ {chosen_exit_price:.2f})"
                # Calculate effective exit price after slippage
                effective_exit_price = chosen_exit_price - slippage_per_trade
                # Calculate profit (commission was included in entry_price)
                trade_profit = effective_exit_price - entry_price
                logger.debug(f"Exit Triggered: {exit_reason}. Raw Exit: {chosen_exit_price:.2f}, Eff Exit: {effective_exit_price:.2f}, Eff Entry: {entry_price:.2f}, Profit: {trade_profit:.2f}")

                # Update last trade details safely
                if trades_details:
                     trades_details[-1].update({
                        'exit_time': time_key,
                        'exit_price': round(chosen_exit_price, 2), # Log raw exit price
                        'effective_exit_price': round(effective_exit_price, 2),
                        'exit_reason': exit_reason,
                        'profit': round(trade_profit, 2) if pd.notna(trade_profit) else 0.0,
                        'profit_pct': round((trade_profit / entry_price) * 100, 2) if entry_price > 0 and pd.notna(trade_profit) else 0.0
                    })
                else:
                    logger.error("Attempted to update non-existent trade details on exit.")

                pnl += trade_profit if pd.notna(trade_profit) else 0 # Add realized PnL
                position_active = 0 # Reset position status
                trades += 1 # Increment trade counter
                # Reset state variables for next trade
                entry_price = 0.0; raw_entry_price = 0.0; stop_loss_level = 0.0; trailing_stop_level = 0.0; highest_price_since_entry = 0.0;


        # --- Entry Conditions Check (only when flat) ---
        # Check basic conditions first
        can_enter = pd.notna(price) and pd.notna(sl_pct)
        # Add strategy-specific entry condition checks if needed
        # e.g., if strategy requires RSI < 70 for entry:
        # can_enter = can_enter and pd.notna(rsi) and rsi < 70

        if position_active == 0 and signal_enter_long: # Check strategy's entry signal
            if can_enter:
                action_taken = f"Enter Long @ {price:.2f}"
                position_active = 1
                raw_entry_price = price
                # Calculate effective entry price including costs
                effective_entry_price = raw_entry_price + slippage_per_trade + commission_per_trade
                entry_price = effective_entry_price # Store effective price for PnL calc
                # Calculate initial stop loss based on effective entry price
                stop_loss_level = effective_entry_price * (1 - sl_pct / 100)
                # Initialize trailing stop loss at the initial stop loss level
                trailing_stop_level = stop_loss_level
                # Initialize highest price tracker
                highest_price_since_entry = raw_entry_price

                # Log trade details
                trades_details.append({
                    'direction': 'BUY',
                    'entry_time': time_key,
                    'entry_price': round(raw_entry_price, 2), # Log raw price
                    'effective_entry_price': round(effective_entry_price, 2), # Log effective price
                    'sl_level': round(stop_loss_level, 2),
                    'initial_trail_level': round(trailing_stop_level, 2),
                    'exit_time': None, 'exit_price': None, 'effective_exit_price': None,
                    'exit_reason': None, 'profit': None, 'profit_pct': None
                })
            else:
                 action_taken = "Hold (Entry Signal Skipped - Conditions not met)"
                 logger.debug(f"Entry skipped at {idx}: Price={price}, SL%={sl_pct}, RSI={rsi}")


        # --- Log Action Taken ---
        print(f"  -> Action: {action_taken}")


    # --- Post-Loop Summary ---
    print("\n--- Simulation Loop Finished ---")
    final_status = 'Flat'
    # Handle position still open at the end of data
    if position_active == 1 and trades_details:
        final_status = 'Holding Long'
        last_price = df['close'].iloc[-1]
        last_time_key = df.index[-1].isoformat() if has_datetime_index and hasattr(df.index[-1], 'isoformat') else str(df.index[-1])

        # Calculate unrealized PnL based on last closing price vs effective entry
        # Account for slippage on this hypothetical exit
        hypothetical_exit_price = last_price - slippage_per_trade
        unreal_pnl = hypothetical_exit_price - entry_price

        trades_details[-1].update({
            'exit_time': 'End of Data',
            'exit_price': round(last_price, 2), # Mark exit at last close
            'effective_exit_price': round(hypothetical_exit_price, 2),
            'exit_reason': 'Still Holding',
            'profit': round(unreal_pnl, 2) if pd.notna(unreal_pnl) else 0.0,
            'profit_pct': round((unreal_pnl / entry_price) * 100, 2) if entry_price > 0 and pd.notna(unreal_pnl) else 0.0
        })
        # Decide whether to include unrealized PnL in the final returned 'pnl' value
        # pnl += unreal_pnl if pd.notna(unreal_pnl) else 0 # Uncomment to add unrealized PnL


    logger.info(f"Simulation finished. Final Realized PnL: {pnl:.2f}, Total Trades Completed: {trades}, Final Status: {final_status}")
    # Return results dictionary
    return {
        'pnl': round(pnl, 2), # Realized PnL from closed trades
        'trades': trades, # Number of closed trades
        'status': final_status, # 'Flat' or 'Holding Long'
        'trades_details': trades_details, # List of dictionaries for each trade
        'parameters': { # Log simulation parameters used
            'sl_pct': sl_pct,
            'trail_pct_param': trail_pct, # Parameter value (logic uses ATR)
            'commission_per_trade': commission_per_trade,
            'slippage_per_trade': slippage_per_trade
        }
    }
