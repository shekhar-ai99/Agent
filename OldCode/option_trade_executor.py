

from datetime import datetime
import glob
import logging
import pandas as pd
import os
# from datetime import datetime # Not strictly used in this version, but often useful

logger = logging.getLogger(__name__)

# MODIFICATION: timestamp_to_key and add_datetime_key_column are removed as they are
# no longer needed for indexing ce_premiums_df and pe_premiums_df when using DatetimeIndex.
# If you use them for other purposes elsewhere in your project, you can keep them defined
# in a utility module, but they are not used by this class for its primary indexing.
def format_expiry(expiry_str):
    """Convert 'yyyy-mm-dd' to 'DDMMMYY' (e.g. '2025-05-29' -> '29MAY25')."""
    dt = datetime.strptime(expiry_str, "%Y-%m-%d")
    return dt.strftime("%d%b%y").upper()
class OptionTradeExecutor:
 
    def __init__(self, lot_size, entry_premium, premium_change_rate, points_per_change,
                 pece_csv_path, symbol, expiry, strike, timeframe, run_id="default_run"):
        self.lot_size = lot_size
        self.entry_premium = entry_premium
        self.premium_change_rate = premium_change_rate
        self.points_per_change = points_per_change
        self.run_id = run_id
        self.option_trades = []
        self.timeframe = timeframe
        self.symbol = symbol.upper()
        self.expiry = format_expiry(expiry.upper())  # "29MAY25"
        self.strike = str(int(strike))  # "24000"
        self.folder = pece_csv_path
        self.ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']

        # Find and load CE/PE files dynamically
        self.ce_premiums_df = self._load_option_file(option_type="CE",timeframe=timeframe)
        self.pe_premiums_df = self._load_option_file(option_type="PE",timeframe=timeframe)
    def _find_file(self, option_type,timeframe):
        """
        Dynamically find the best matching file for the given option_type (PE or CE)
        """
        pattern = f"{self.symbol}{self.strike}{option_type}{self.expiry}_{timeframe}_filled.csv"
        search_path = os.path.join(self.folder, pattern)
        matches = glob.glob(search_path)
        if matches:
            logger.info(f"Found {option_type} file: {matches[0]}")
            return matches[0]
        else:
            logger.warning(f"No {option_type} premium file found with pattern {search_path}")
            return None
    def _load_option_file(self, option_type,timeframe):
        file_path = self._find_file(option_type,timeframe)
        if file_path and os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, parse_dates=['datetime'])
                if not df.empty and 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                    df.dropna(subset=['datetime'], inplace=True)
                    if not df.empty:
                        df = df.set_index('datetime')
                        df = df.sort_index()
                        for col in self.ohlcv_cols:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        logger.info(f"Loaded {option_type} premiums: Shape: {df.shape} from {file_path}")
                        return df
                    else:
                        logger.warning(f"{option_type} DataFrame empty after datetime parsing: {file_path}")
                else:
                    logger.warning(f"{option_type} file missing 'datetime' or is empty: {file_path}")
            except Exception as e:
                logger.error(f"Error loading {option_type} premiums from {file_path}: {e}", exc_info=True)
        else:
            logger.warning(f"{option_type} premiums file not found: {file_path}")
        return None
    # MODIFICATION: New helper method for fetching premium from a DatetimeIndexed DataFrame
    def _fetch_premium_from_datetime_indexed_df(self, df, actual_timestamp_in_df, column_name, option_type, context_msg):
        """
        Safely fetches and validates a premium value from the DataFrame at a specific timestamp.
        """
        try:
            premium_val = df.loc[actual_timestamp_in_df, column_name]
            
            # Handle if .loc returns a Series (e.g., duplicate index, though unlikely with sorted unique DatetimeIndex)
            if isinstance(premium_val, pd.Series):
                if not premium_val.empty:
                    premium_val = premium_val.iloc[0] # Take the first value
                    logger.warning(f"Multiple values found for {option_type} {context_msg} at {actual_timestamp_in_df} for column '{column_name}'. Using first: {premium_val}")
                else:
                    logger.warning(f"Premium lookup for {option_type} {context_msg} at {actual_timestamp_in_df} for column '{column_name}' returned an empty Series.")
                    return None

            if pd.isna(premium_val):
                logger.warning(f"Premium for {option_type} {context_msg} at {actual_timestamp_in_df} (column '{column_name}') is NaN.")
                return None
            return float(premium_val)
        except KeyError: # Should not happen if actual_timestamp_in_df is from df.index
             logger.warning(f"KeyError fetching premium for {option_type} {context_msg} at {actual_timestamp_in_df} (column '{column_name}'). This indicates an issue with the provided timestamp.")
             return None
        except (ValueError, TypeError) as e:
            logger.error(f"Premium for {option_type} {context_msg} at {actual_timestamp_in_df} (value: {premium_val if 'premium_val' in locals() else 'N/A'}, column '{column_name}') could not be converted to float: {e}")
            return None

    # MODIFICATION: _get_entry_premium now uses DatetimeIndex and pd.Timestamp for lookups
    def _get_entry_premium(self, df, timestamp, option_type): 
        if df is None or df.empty:
            logger.warning(f"No {option_type} premium data available (entry).")
            return None
        
        if not self.validate_df_is_datetime_indexed(df, option_type): 
            logger.warning(f"DataFrame validation failed for {option_type} (entry).")
            return None
            
        try:
            requested_timestamp = pd.to_datetime(timestamp, errors='coerce')
            if pd.isna(requested_timestamp):
                logger.error(f"Invalid input timestamp for {option_type} entry: {timestamp}. Cannot process.")
                return None
            
            # Align timezone of requested_timestamp with df.index if necessary
            if df.index.tz is not None and requested_timestamp.tz is None:
                requested_timestamp = requested_timestamp.tz_localize(df.index.tz, ambiguous='infer', nonexistent='shift_forward')
            elif df.index.tz is None and requested_timestamp.tz is not None:
                requested_timestamp = requested_timestamp.tz_localize(None)
            elif df.index.tz is not None and requested_timestamp.tz is not None and df.index.tz != requested_timestamp.tz:
                 requested_timestamp = requested_timestamp.tz_convert(df.index.tz)


            if requested_timestamp in df.index:
                # print(f"Found exact {option_type} ENTRY premium at timestamp: {requested_timestamp}") # Your debug
                return self._fetch_premium_from_datetime_indexed_df(df, requested_timestamp, 'open', option_type, "ENTRY")
            else:
                # print(f"Checking nearby {option_type} premium for ENTRY timestamp: {requested_timestamp}.") # Your debug
                nearest_idx_pos_arr = df.index.get_indexer([requested_timestamp], method='nearest')
                
                if not nearest_idx_pos_arr.size or nearest_idx_pos_arr[0] == -1:
                    logger.warning(f"No nearby {option_type} premium found for timestamp {requested_timestamp} (ENTRY).")
                    return None
                
                closest_timestamp_from_index = df.index[nearest_idx_pos_arr[0]]
                premium = self._fetch_premium_from_datetime_indexed_df(df, closest_timestamp_from_index, 'open', option_type, "ENTRY (Nearest)")
                
                if premium is not None: # Only log time_diff if premium was found
                    time_diff = abs((requested_timestamp - closest_timestamp_from_index).total_seconds() / 60)
                    logger.warning(f"Entry timestamp {requested_timestamp} not found. Nearest for {option_type} was {closest_timestamp_from_index} ({time_diff:.0f} min away), using its 'open' premium: {premium:.2f}")
                
                return premium

        except Exception as e: 
            logger.error(f"Error finding {option_type} entry premium for timestamp {timestamp}: {e}", exc_info=True)
            return None

    # MODIFICATION: _get_exit_premium now uses DatetimeIndex and pd.Timestamp for lookups
    def _get_exit_premium(self, df, timestamp, option_type): 
        if df is None or df.empty:
            logger.warning(f"No {option_type} premium data available (exit).")
            return None
        
        if not self.validate_df_is_datetime_indexed(df, option_type):
            logger.warning(f"DataFrame validation failed for {option_type} (exit).")
            return None

        try:
            requested_timestamp = pd.to_datetime(timestamp, errors='coerce')
            if pd.isna(requested_timestamp):
                logger.error(f"Invalid input timestamp for {option_type} exit: {timestamp}. Cannot process.")
                return None

            # Align timezone of requested_timestamp with df.index if necessary
            if df.index.tz is not None and requested_timestamp.tz is None:
                requested_timestamp = requested_timestamp.tz_localize(df.index.tz, ambiguous='infer', nonexistent='shift_forward')
            elif df.index.tz is None and requested_timestamp.tz is not None:
                requested_timestamp = requested_timestamp.tz_localize(None)
            elif df.index.tz is not None and requested_timestamp.tz is not None and df.index.tz != requested_timestamp.tz:
                 requested_timestamp = requested_timestamp.tz_convert(df.index.tz)

            # User requested to use 'open' price for exit as well
            price_column_for_exit = 'open' 

            if requested_timestamp in df.index:
                # print(f"Found exact {option_type} EXIT premium at timestamp: {requested_timestamp} using column '{price_column_for_exit}'") # Your debug
                return self._fetch_premium_from_datetime_indexed_df(df, requested_timestamp, price_column_for_exit, option_type, "EXIT")
            else:
                # print(f"Checking nearby {option_type} premium for EXIT timestamp: {requested_timestamp}.") # Your debug
                nearest_idx_pos_arr = df.index.get_indexer([requested_timestamp], method='nearest')
                if not nearest_idx_pos_arr.size or nearest_idx_pos_arr[0] == -1:
                    logger.warning(f"No nearby {option_type} premium found for timestamp {requested_timestamp} (EXIT).")
                    return None
                
                closest_timestamp_from_index = df.index[nearest_idx_pos_arr[0]]
                premium = self._fetch_premium_from_datetime_indexed_df(df, closest_timestamp_from_index, price_column_for_exit, option_type, f"EXIT (Nearest, using '{price_column_for_exit}')")
                
                if premium is not None:
                    time_diff = abs((requested_timestamp - closest_timestamp_from_index).total_seconds() / 60)
                    logger.warning(f"Exit timestamp {requested_timestamp} not found. Nearest for {option_type} was {closest_timestamp_from_index} ({time_diff:.0f} min away), using its '{price_column_for_exit}' premium: {premium:.2f}")
                
                return premium

        except Exception as e: 
            logger.error(f"Error finding {option_type} exit premium for timestamp {timestamp}: {e}", exc_info=True)
            return None

    def enter_option_trade(self, signal, timestamp, index_price, trade_id):
        option_type = 'CE' if signal == 'buy_potential' else 'PE'
        strike_price = 25000  # Or pass dynamically
        premiums_df = self.ce_premiums_df if option_type == 'CE' else self.pe_premiums_df
        
        premium = self._get_entry_premium(premiums_df, timestamp, option_type) 
        
        if premium is None:
            premium = self.entry_premium # Fallback to default
            logger.warning(f"No valid {option_type} premium found for entry at {timestamp}. Using default entry premium: {premium}")

        trade = {
            'trade_id': trade_id, 'option_type': option_type, 'entry_time': timestamp,
            'index_entry_price': index_price, 'strike_price': strike_price,
            'entry_premium': premium, 'lot_size': self.lot_size, 'strategy': None,
            'timeframe': None, 'exit_time': None, 'index_exit_price': None,
            'exit_premium': None, 'exit_reason': None, 'pl': None
        }
        self.option_trades.append(trade)
        entry_prem_display = f"{premium:.2f}" if isinstance(premium, (int, float)) else "N/A"
        logger.info(f"ENTERED {option_type} (ID: {trade_id}) @ {timestamp} | Index={index_price} | Strike={strike_price} | EntryPrem={entry_prem_display}")

    def exit_option_trade(self, trade_id, timestamp, index_price, exit_reason, strategy_name, timeframe):
        trade_found_and_exited = False
        for trade_idx, trade in enumerate(self.option_trades):
            if trade['trade_id'] == trade_id and trade['exit_time'] is None:
                option_type = trade['option_type']
                premiums_df = self.ce_premiums_df if option_type == 'CE' else self.pe_premiums_df
                
                exit_premium = self._get_exit_premium(premiums_df, timestamp, option_type)
                
                calculated_pl = None
                current_entry_premium = trade.get('entry_premium')

                if exit_premium is None:
                    logger.warning(f"No valid {option_type} premium found for exit (ID: {trade_id}) at {timestamp}.")
                    if current_entry_premium is not None and isinstance(current_entry_premium, (int, float)):
                        logger.warning("Using P&L calculation based on index change as premium fallback.")
                        index_change = index_price - trade['index_entry_price']
                        premium_change_estimate = (index_change / self.points_per_change) * self.premium_change_rate
                        exit_premium = current_entry_premium + premium_change_estimate 
                        calculated_pl = (exit_premium - current_entry_premium) * self.lot_size
                        logger.warning(f"Estimated exit_premium: {exit_premium:.2f} for TradeID {trade_id}.")
                    else:
                        logger.error(f"Cannot calculate fallback P&L for TradeID {trade_id} as entry_premium is invalid ({current_entry_premium}). Setting P&L to 0.")
                        exit_premium = None 
                        calculated_pl = 0.0
                elif current_entry_premium is not None and isinstance(current_entry_premium, (int, float)):
                     calculated_pl = (exit_premium - current_entry_premium) * self.lot_size
                else:
                    logger.error(f"Cannot calculate P&L for TradeID {trade_id}: entry_premium ({current_entry_premium}) missing/invalid, even though exit_premium {exit_premium} was found. Setting P&L to 0.")
                    calculated_pl = 0.0
                
                updated_trade_details = {
                    'exit_time': timestamp, 'index_exit_price': index_price,
                    'exit_premium': exit_premium, 'exit_reason': exit_reason,
                    'strategy': strategy_name, 'timeframe': timeframe,
                    'pl': calculated_pl if calculated_pl is not None else 0.0
                }
                # MODIFICATION: Update the dictionary in the list directly
                self.option_trades[trade_idx].update(updated_trade_details)

                entry_prem_display = f"{current_entry_premium:.2f}" if isinstance(current_entry_premium, (int,float)) else "N/A"
                exit_prem_display = f"{exit_premium:.2f}" if isinstance(exit_premium, (int,float)) else "N/A"
                pl_display = f"{trade['pl']:.2f}" if isinstance(trade['pl'], (int,float)) else "N/A"

                logger.info(f"EXITED {option_type} (ID: {trade_id}) @ {timestamp} | Reason={exit_reason} | IdxEntry={trade['index_entry_price']} IdxExit={index_price} | PremEntry={entry_prem_display} PremExit={exit_prem_display} | PnL={pl_display}")
                self.save_option_trades(strategy_name,timeframe)
                trade_found_and_exited = True
                break 
        
        if not trade_found_and_exited:
            logger.warning(f"No open trade found for trade_id={trade_id} to process exit.")
        
        # MODIFICATION: Removed save_option_trades from here. It should be called less frequently.
        # self.save_option_trades(timeframe,strategy_name) 

    # MODIFICATION: Renamed and adapted validate_df for DatetimeIndex
    def validate_df_is_datetime_indexed(self, df, option_type): 
        if df is None or df.empty:
            logger.error(f"DataFrame for {option_type} is None or empty in validate_df.")
            return False
            
        # MODIFICATION: Check for 'open', 'close', 'high', 'low', 'volume'
        required_value_cols = ['open', 'high', 'low', 'close', 'volume'] 
        for col_to_check in required_value_cols: 
            if col_to_check not in df.columns:
                 logger.error(f"DataFrame for {option_type} missing required value column: '{col_to_check}'. Available: {df.columns.tolist()}")
                 return False
            if not pd.api.types.is_numeric_dtype(df[col_to_check]):
                logger.warning(f"Column '{col_to_check}' in {option_type} DataFrame is not numeric. Attempting coercion.")
                try:
                    df[col_to_check] = pd.to_numeric(df[col_to_check], errors='coerce')
                except Exception as e: 
                    logger.error(f"Error coercing column '{col_to_check}' to numeric for {option_type}: {e}")
                    return False 
                if not pd.api.types.is_numeric_dtype(df[col_to_check]):
                    logger.error(f"Column '{col_to_check}' for {option_type} is STILL not numeric after coercion.")
                    return False
        
        if not isinstance(df.index, pd.DatetimeIndex): 
            logger.error(f"DataFrame for {option_type} does not have a DatetimeIndex. Current index type: {type(df.index)}, name: {df.index.name}")
            return False
        
        if not df.index.is_monotonic_increasing and not df.index.is_monotonic_decreasing :
            logger.warning(f"Performance Warning: DatetimeIndex for {option_type} is not monotonic (sorted). Sorting it now. This should ideally be done once in __init__.")
            try:
                df.sort_index(inplace=True) 
            except Exception as e:
                logger.error(f"Error sorting index for {option_type}: {e}")
                return False 
        return True

    # MODIFICATION: save_option_trades now creates one consolidated file per run.
    def save_option_trades(self, timeframe=None, strategyname=None): # Parameters kept for compatibility if called from old code path
        output_dir = os.path.join('reports', f'run_{self.run_id}', 'option_trade_logs')
        output_path = os.path.join(output_dir, f"option_trades_{self.run_id}_{timeframe}.csv") 

        if not self.option_trades:
            logger.info(f"No option trades recorded for run {self.run_id} to save (list is empty).")
            return
        try:
            os.makedirs(output_dir, exist_ok=True)
            df_trades = pd.DataFrame(list(self.option_trades)) 
            df_trades.to_csv(output_path, index=False, float_format='%.2f')
            logger.info(f"Saved/Updated {len(df_trades)} option trades for run {self.run_id} to CSV: {output_path}")
        except Exception as e:
            logger.error(f"Error saving option trades for run {self.run_id} to {output_path}: {e}")

    def close_open_trades_at_session_end(self, session_end_time, index_price, strategy_name, timeframe):
        session_end_time = pd.to_datetime(session_end_time) 
        logger.debug(f"Checking for open option trades at session end: {session_end_time} (Context: {strategy_name}, {timeframe})")
        
        trade_ids_to_close = [
            trade['trade_id'] for trade in self.option_trades if trade['exit_time'] is None
        ]

        if not trade_ids_to_close:
            logger.debug(f"No open trades for current context ({strategy_name} {timeframe}) to close at session end.")
        else:
            logger.info(f"Found {len(trade_ids_to_close)} open trades to close at session end for {strategy_name} {timeframe}.")

        for trade_id_to_close in trade_ids_to_close:
            # exit_option_trade will attempt to find the premium for session_end_time
            self.exit_option_trade(trade_id_to_close, session_end_time, index_price, "session_end", strategy_name, timeframe)
        logger.debug(f"Performing final save after processing session end for {strategy_name} {timeframe}.")
        self.save_option_trades(strategy_name,timeframe)
        # MODIFICATION: Call the consolidated save_option_trades once after all session-end closures.
        #self.save_option_trades() 
