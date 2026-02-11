# # live_trading_engine.py

# import sys
# import pandas as pd
# import logging
# import time
# from datetime import datetime, timedelta
# import os
# from termcolor import cprint

# # Assuming 'angel_one_integration.py' is in the same directory
# from angel_one import AngelOneAPI

# # Assuming 'strategy_tester' directory is in the parent directory
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from app.compute_indicators import compute_indicators, compute_atr
# from app.strategies import strategy_factories
# from app.backtest_engine import (
#     SimpleBacktester,
#     initialize_trade_log,
#     get_trade_session,
#     get_trade_regime,
#     get_trade_volatility,
#     save_trades_to_csv
# )
# from config import Config, setup_logging, get_data_dir, get_reports_dir, load_config
# from angel_one.angel_one_instrument_manager import InstrumentManager # Assuming this file exists

# # Ensure logs directory exists
# os.makedirs("logs", exist_ok=True)

# # Initialize logger
# logger = logging.getLogger(__name__)

# def get_live_data_save_paths(symbol, timeframe, run_id):
#     data_dir = get_data_dir()
#     raw_live_dir = os.path.join(data_dir, "raw", "live")
#     indicator_live_dir = os.path.join(data_dir, "datawithindicator", "live")
#     os.makedirs(raw_live_dir, exist_ok=True)
#     os.makedirs(indicator_live_dir, exist_ok=True)

#     raw_data_filename = f"live_{symbol.lower().replace('-eq', '').replace('&', '_')}_{timeframe}_raw_{run_id}.csv"
#     indicator_data_filename = f"live_{symbol.lower().replace('-eq', '').replace('&', '_')}_{timeframe}_indicators_{run_id}.csv"

#     raw_data_path = os.path.join(raw_live_dir, raw_data_filename)
#     indicator_data_path = os.path.join(indicator_live_dir, indicator_data_filename)

#     return raw_data_path, indicator_data_path

# def get_live_trade_log_paths(run_id, timeframe, symbol):
#     reports_dir = get_reports_dir()
#     run_dir_name = f"live_run_{symbol.lower().replace('-eq', '').replace('&', '_')}_{timeframe}_{run_id}"
#     run_dir = os.path.join(reports_dir, run_dir_name)
#     os.makedirs(run_dir, exist_ok=True)

#     base_filename = f"trade_log_live_{symbol.lower().replace('-eq', '').replace('&', '_')}_{timeframe}_{run_id}"
#     csv_path = os.path.join(run_dir, f"{base_filename}.csv")
#     json_path = os.path.join(run_dir, f"{base_filename}.json")
#     return csv_path, json_path

# def save_data_row(df_row, file_path):
#     header = not os.path.exists(file_path)
#     try:
#         df_row.to_csv(file_path, mode='a', header=header, index=False)
#     except Exception as e:
#         logger.error(f"Error saving data row to {file_path}: {e}")

# def save_trades_live(trades_df, csv_path, json_path):
#     if trades_df is None or trades_df.empty:
#         return
#     try:
#         trades_df.to_csv(csv_path, index=False)
#         df_for_json = trades_df.copy()
#         for col in df_for_json.select_dtypes(include=['datetime64']).columns:
#             df_for_json[col] = df_for_json[col].astype(str)
#         df_for_json.to_json(json_path, orient='records', indent=4)
#         logger.debug(f"Trades saved to {csv_path} and {json_path}")
#     except Exception as e:
#         logger.error(f"Error saving trades to {csv_path} or {json_path}: {e}")

# def initialize_live_session_data(api_client, symbol, timeframe, exchange="NFO"):
#     logger.info(f"Fetching initial historical data for {symbol} ({timeframe})...")
#     days_history = config_instance.get('live_initial_history_days', 90)

#     from_dt = datetime.now() - timedelta(days=days_history)
#     from_date_str = from_dt.strftime('%Y-%m-%d %H:%M')
#     to_date_str = datetime.now().strftime('%Y-%m-%d %H:%M')

#     broker_tf = api_client.timeframe_to_broker_format(timeframe, config_instance.get('broker_name', 'angelone'))
#     if not broker_tf:
#         logger.error(f"Could not convert timeframe {timeframe} to broker format. Exiting.")
#         return None

#     historical_df = api_client.fetch_historical_data(
#         symbol_name=symbol,
#         timeframe_val=broker_tf,
#         exchange=exchange,
#         from_date=from_date_str,
#         to_date=to_date_str
#     )

#     if historical_df is None or historical_df.empty:
#         logger.error("Failed to fetch initial historical data. Exiting.")
#         return None

#     if 'date' in historical_df.columns and 'datetime' not in historical_df.columns:
#         historical_df.rename(columns={'date': 'datetime'}, inplace=True)
#     if not pd.api.types.is_datetime64_any_dtype(historical_df['datetime']):
#         historical_df['datetime'] = pd.to_datetime(historical_df['datetime'], errors='coerce', utc=True).dt.tz_convert('Asia/Kolkata')
#     historical_df = historical_df.dropna(subset=['datetime'])

#     historical_df.sort_values(by='datetime', inplace=True)
#     historical_df.drop_duplicates(subset=['datetime'], keep='last', inplace=True)
#     logger.info(f"Fetched {len(historical_df)} initial historical candles for {symbol} {timeframe}.")
#     return historical_df

# def display_live_status(symbol, timeframe, active_positions, all_trades_session_df, current_price=None, last_candle_time=None):
#     os.system('cls' if os.name == 'nt' else 'clear')
#     header = f"--- Live Dry Run: {symbol} ({timeframe}) --- Status at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---"
#     if last_candle_time:
#         header += f"\nLast Candle Processed: {last_candle_time.strftime('%Y-%m-%d %H:%M:%S')}, Current Market Price (approx): {current_price if current_price else 'N/A'}"
#     print(header)

#     total_trades_taken = len(all_trades_session_df[all_trades_session_df['status'] != 'OPEN'])
#     open_trades_count = len(all_trades_session_df[all_trades_session_df['status'] == 'OPEN'])
#     print(f"Total Trades Taken in Session: {total_trades_taken}, Open Positions: {open_trades_count}")

#     if not active_positions:
#         print("No Active Positions currently.")
#     else:
#         print("\nCurrently Active Positions:")
#         for strategy_name, pos_list in active_positions.items():
#             if not pos_list: continue
#             pos = pos_list[0]
#             pnl_str = ""
#             if current_price and pos.get('entry_price'):
#                 pnl_points = (current_price - pos['entry_price']) if pos['position_type'] == 'LONG' else (pos['entry_price'] - current_price)
#                 pnl_str = f", Approx P&L (pts): {pnl_points:.2f}"
#             print(f"  Strategy: {strategy_name}, Type: {pos.get('position_type')}, Entry: {pos.get('entry_price'):.2f} at {str(pos.get('entry_time'))} {pnl_str}")

#     if not all_trades_session_df.empty:
#         print("\nLast 5 Logged Trade Actions (Open/Closed):")
#         print(all_trades_session_df.tail().to_string(index=False, columns=['strategy', 'position_type', 'entry_price', 'exit_price', 'pnl', 'status', 'entry_time', 'exit_time']))
#     print("-" * len(header.split('\n')[0]))

# def process_trade_signal_live(current_signal_info, current_candle_data, active_position_list, strategy_name, symbol, timeframe, trade_id_counter_ref, api_client, instrument_manager, master_data_df):
#     newly_executed_trades_log = []
#     trade_actioned = False
#     active_position = active_position_list[0] if active_position_list else None

#     signal = current_signal_info.get('signal')
#     price = current_candle_data['close']
#     timestamp = current_candle_data['datetime']
#     sl_signal = current_signal_info.get('sl')
#     tp_signal = current_signal_info.get('tp')

#     # Handle Exits
#     if active_position:
#         exit_triggered = False
#         exit_reason = None
#         exit_price_actual = price

#         if signal == 0 and active_position['position_type'] in ('LONG', 'SHORT'):
#             exit_triggered = True
#             exit_reason = "Strategy Exit Signal (0)"
#         elif signal == -1 and active_position['position_type'] == 'LONG':
#             exit_triggered = True
#             exit_reason = "Opposite Signal (Short)"
#         elif signal == 1 and active_position['position_type'] == 'SHORT':
#             exit_triggered = True
#             exit_reason = "Opposite Signal (Long)"

#         pos_sl = active_position.get('stop_loss')
#         pos_tp = active_position.get('take_profit')
#         if active_position['position_type'] == 'LONG':
#             if pos_sl and current_candle_data['low'] <= pos_sl:
#                 exit_triggered = True
#                 exit_price_actual = pos_sl
#                 exit_reason = f"Stop Loss Hit at {pos_sl}"
#             elif pos_tp and current_candle_data['high'] >= pos_tp:
#                 exit_triggered = True
#                 exit_price_actual = pos_tp
#                 exit_reason = f"Take Profit Hit at {pos_tp}"
#         elif active_position['position_type'] == 'SHORT':
#             if pos_sl and current_candle_data['high'] >= pos_sl:
#                 exit_triggered = True
#                 exit_price_actual = pos_sl
#                 exit_reason = f"Stop Loss Hit at {pos_sl}"
#             elif pos_tp and current_candle_data['low'] <= pos_tp:
#                 exit_triggered = True
#                 exit_price_actual = pos_tp
#                 exit_reason = f"Take Profit Hit at {pos_tp}"

#         if exit_triggered:
#             pnl = (exit_price_actual - active_position['entry_price']) if active_position['position_type'] == 'LONG' else (active_position['entry_price'] - exit_price_actual)
#             trade_log_entry = active_position.copy()
#             trade_log_entry.update({
#                 'exit_time': timestamp,
#                 'exit_price': exit_price_actual,
#                 'pnl': pnl,
#                 'status': 'CLOSED',
#                 'exit_reason': exit_reason
#             })
#             newly_executed_trades_log.append(trade_log_entry)
#             logger.info(f"CLOSED {active_position['position_type']} for {strategy_name} at {exit_price_actual:.2f}. P&L: {pnl:.2f}. Reason: {exit_reason}")
#             active_position_list.clear()
#             trade_actioned = True

#     # Handle Entries
#     if not active_position:
#         entry_signal = None
#         if signal == 1:
#             entry_signal = 'LONG'
#         elif signal == -1:
#             entry_signal = 'SHORT'

#         if entry_signal:
#             valid_expiries = instrument_manager.get_valid_expiries(symbol)
#             if not valid_expiries:
#                 logger.error(f"No valid expiries found for {symbol}. Please check instrument list.")
#                 return trade_actioned, pd.DataFrame(newly_executed_trades_log)

#             current_date = pd.Timestamp.now(tz='Asia/Kolkata')
#             valid_expiries_dt = [datetime.strptime(exp, "%d%b%Y") for exp in valid_expiries]
#             future_expiries = [exp for exp in valid_expiries_dt if exp >= current_date]
#             if not future_expiries:
#                 logger.error(f"No future expiries found for {symbol}. Available: {valid_expiries}")
#                 return trade_actioned, pd.DataFrame(newly_executed_trades_log)
#             expiry_dt = min(future_expiries)
#             expiry_date = expiry_dt.strftime("%d%b%y").upper()

#             current_price = current_candle_data['close']
#             atm_strike = instrument_manager.get_atm_strike(symbol, expiry_date, current_price)
#             if not atm_strike:
#                 logger.error(f"Failed to get ATM strike for {symbol} with expiry {expiry_date}")
#                 return trade_actioned, pd.DataFrame(newly_executed_trades_log)

#             option_type = "CE" if entry_signal == "LONG" else "PE"
#             option_ticker = f"{symbol.upper()}{expiry_date}{int(atm_strike)}{option_type}"
#             token = instrument_manager.get_instrument_token(option_ticker, "NFO")
#             if not token:
#                 logger.error(f"No token found for {option_ticker}. Available tickers: {instrument_manager.instrument_df[instrument_manager.instrument_df['exchange'] == 'NFO']['tradingsymbol'].tolist()}")
#                 return trade_actioned, pd.DataFrame(newly_executed_trades_log)

#             ltp = api_client.fetch_websocket_ltp(option_ticker, "NFO") or \
#                   api_client._fetch_current_price(option_ticker, "NFO")
#             if ltp <= 0:
#                 logger.error(f"Invalid LTP for {option_ticker}: {ltp}")
#                 return trade_actioned, pd.DataFrame(newly_executed_trades_log)

#             # Calculate ATR-based SL/TP
#             recent_data = master_data_df.tail(14)
#             atr = compute_atr(recent_data, period=14).iloc[-1]
#             sl = sl_signal if sl_signal else (ltp - 2 * atr if entry_signal == 'LONG' else ltp + 2 * atr)
#             tp = tp_signal if tp_signal else (ltp + 4 * atr if entry_signal == 'LONG' else ltp - 4 * atr)

#             trade_id_counter_ref[0] += 1
#             trade_id = f"{strategy_name}_{symbol}_{timeframe}_{trade_id_counter_ref[0]}"

#             active_position_data = {
#                 'trade_id': trade_id,
#                 'strategy': strategy_name,
#                 'symbol': option_ticker,
#                 'timeframe': timeframe,
#                 'entry_time': timestamp,
#                 'entry_price': ltp,
#                 'exit_time': pd.NaT,
#                 'exit_price': float('nan'),
#                 'position_type': entry_signal,
#                 'quantity': 1,
#                 'pnl': 0.0,
#                 'stop_loss': sl,
#                 'take_profit': tp,
#                 'status': 'OPEN',
#                 'exit_reason': None
#             }

#             active_position_list.append(active_position_data)
#             newly_executed_trades_log.append(active_position_data.copy())
#             logger.info(f"OPENED {entry_signal} for {strategy_name} | Option: {option_ticker} | LTP: ₹{ltp:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
#             print(f"\n🟢 [ENTRY] {entry_signal} | {option_ticker} | LTP: ₹{ltp:.2f} | Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} | SL: {sl:.2f} | TP: {tp:.2f}\n")
#             trade_actioned = True

#     return trade_actioned, pd.DataFrame(newly_executed_trades_log)

# def _is_nse_holiday(date: datetime) -> bool:
#     holidays = [
#         datetime(2025, 5, 1),  # Labour Day
#         # Add more from https://www.nseindia.com/products-services/equity-market-timings-holidays
#     ]
#     return date.date() in [h.date() for h in holidays]

# def run_live_dry_mode(symbol, timeframe, strategies_config, run_id, instrument_mgr_instance, config_instance):
#     logger.info(f"Starting live dry run for {symbol} on {timeframe} with run_id {run_id}")

#     api_client = AngelOneAPI()
#     smart_connect = api_client.get_smart_connect_object()
#     if not hasattr(smart_connect, 'generateSession'):
#         logger.error("Broker client doesn't appear initialized properly. Exiting live mode.")
#         return

#     master_data_df = initialize_live_session_data(api_client, symbol, timeframe)
#     if master_data_df is None or master_data_df.empty:
#         return

#     raw_data_path, indicator_data_path = get_live_data_save_paths(symbol, timeframe, run_id)
#     live_trade_log_csv_path, live_trade_log_json_path = get_live_trade_log_paths(run_id, timeframe, symbol)

#     all_trades_session_df = initialize_trade_log()
#     active_positions = {strategy_name: [] for strategy_name in strategies_config.keys()}
#     trade_id_counter = [0]

#     try:
#         last_processed_candle_time = master_data_df.iloc[-1]['datetime'] if not master_data_df.empty else pd.Timestamp.min.tz_localize('Asia/Kolkata')

#         while True:
#             current_loop_time = pd.Timestamp.now(tz='Asia/Kolkata')

#             market_open_hour = 9
#             market_open_minute = 15
#             market_close_hour = 15
#             market_close_minute = 30
#             if not ((current_loop_time.hour > market_open_hour or (current_loop_time.hour == market_open_hour and current_loop_time.minute >= market_open_minute)) and \
#                     (current_loop_time.hour < market_close_hour or (current_loop_time.hour == market_close_hour and current_loop_time.minute <= market_close_minute)) and \
#                     current_loop_time.weekday() < 5 and not _is_nse_holiday(current_loop_time)):
#                 logger.info(f"Market is closed ({current_loop_time.strftime('%H:%M')}). Pausing live dry run. Will check again in 60s.")
#                 display_live_status(symbol, timeframe, active_positions, all_trades_session_df, master_data_df.iloc[-1]['close'] if not master_data_df.empty else None, last_processed_candle_time)
#                 time.sleep(60)
#                 continue

#             fetch_from_dt = last_processed_candle_time + timedelta(seconds=1)
#             fetch_from_date_str = fetch_from_dt.strftime('%Y-%m-%d %H:%M:%S')
#             fetch_to_date_str = current_loop_time.strftime('%Y-%m-%d %H:%M:%S')

#             broker_tf = api_client.timeframe_to_broker_format(timeframe, config_instance.get('broker_name', 'angelone'))
#             latest_candles_df = api_client.fetch_live_data(
#                 symbol_name=symbol,
#                 timeframe_val=broker_tf,
#                 exchange=config_instance.get('exchange', 'NFO'),
#                 from_date=fetch_from_date_str,
#                 to_date=fetch_to_date_str
#             )

#             new_data_processed_this_cycle = False
#             if latest_candles_df is not None and not latest_candles_df.empty:
#                 if 'date' in latest_candles_df.columns and 'datetime' not in latest_candles_df.columns:
#                     latest_candles_df.rename(columns={'date': 'datetime'}, inplace=True)
#                 if not pd.api.types.is_datetime64_any_dtype(latest_candles_df['datetime']):
#                     latest_candles_df['datetime'] = pd.to_datetime(latest_candles_df['datetime'], errors='coerce', utc=True).dt.tz_convert('Asia/Kolkata')

#                 latest_candles_df = latest_candles_df.dropna(subset=['datetime'])
#                 latest_candles_df.sort_values(by='datetime', inplace=True)
#                 latest_candles_df.drop_duplicates(subset=['datetime'], keep='last', inplace=True)

#                 for _, new_row_series in latest_candles_df.iterrows():
#                     new_candle_time = new_row_series['datetime']

#                     if new_candle_time <= last_processed_candle_time:
#                         continue

#                     logger.info(f"New Candle: T={new_candle_time}, O={new_row_series['open']:.2f}, H={new_row_series['high']:.2f}, L={new_row_series['low']:.2f}, C={new_row_series['close']:.2f}, V={new_row_series['volume']}")

#                     master_data_df = pd.concat([master_data_df, pd.DataFrame([new_row_series])], ignore_index=True)
#                     master_data_df.drop_duplicates(subset=['datetime'], keep='last', inplace=True)

#                     save_data_row(pd.DataFrame([new_row_series]), raw_data_path)

#                     data_with_indicators = compute_indicators(master_data_df.copy(), timeframe)

#                     if data_with_indicators is None or data_with_indicators.empty:
#                         logger.warning(f"Indicator computation for candle {new_candle_time} resulted in empty DataFrame. Skipping strategies.")
#                         last_processed_candle_time = new_candle_time
#                         continue

#                     save_data_row(data_with_indicators.iloc[[-1]], indicator_data_path)
#                     current_candle_for_strategy = data_with_indicators.iloc[-1]

#                     for strat_name, strat_config in strategies_config.items():
#                         if strat_name not in strategy_factories:
#                             logger.warning(f"Strategy factory for '{strat_name}' not found. Skipping.")
#                             continue

#                         strategy_instance = strategy_factories[strat_name](strat_config.get('params', {}))
#                         signal_df_strat = strategy_instance.generate_signals(data_with_indicators.copy())

#                         if signal_df_strat.empty:
#                             continue
#                         current_signal_info_strat = signal_df_strat.iloc[-1]

#                         trade_actioned, trade_log_entries_df = process_trade_signal_live(
#                             current_signal_info=current_signal_info_strat,
#                             current_candle_data=current_candle_for_strategy,
#                             active_position_list=active_positions[strat_name],
#                             strategy_name=strat_name,
#                             symbol=symbol,
#                             timeframe=timeframe,
#                             trade_id_counter_ref=trade_id_counter,
#                             api_client=api_client,
#                             instrument_manager=instrument_mgr_instance,
#                             master_data_df=master_data_df
#                         )

#                         if trade_actioned and not trade_log_entries_df.empty:
#                             for _, new_trade_action in trade_log_entries_df.iterrows():
#                                 existing_trade_idx = all_trades_session_df[all_trades_session_df['trade_id'] == new_trade_action['trade_id']].index
#                                 if not existing_trade_idx.empty:
#                                     all_trades_session_df.loc[existing_trade_idx, new_trade_action.index] = new_trade_action
#                                 else:
#                                     all_trades_session_df = pd.concat([all_trades_session_df, pd.DataFrame([new_trade_action])], ignore_index=True)

#                             logger.info(f"Trade action for {strat_name}. Total logged actions: {len(all_trades_session_df)}")
#                             save_trades_live(all_trades_session_df, live_trade_log_csv_path, live_trade_log_json_path)

#                     last_processed_candle_time = new_candle_time
#                     new_data_processed_this_cycle = True

#             current_market_price = master_data_df.iloc[-1]['close'] if not master_data_df.empty else None
#             display_live_status(symbol, timeframe, active_positions, all_trades_session_df, current_market_price, last_processed_candle_time)

#             sleep_duration = config_instance.get('live_data_fetch_interval_seconds', 10)
#             if timeframe.endswith("min"):
#                 try:
#                     tf_val_min = int(timeframe[:-3])
#                     now = pd.Timestamp.now(tz='Asia/Kolkata')
#                     time_to_next_candle = (tf_val_min - (now.minute % tf_val_min)) * 60 - now.second
#                     if time_to_next_candle < 5:
#                         sleep_duration = 3
#                     elif time_to_next_candle < sleep_duration + 5:
#                         sleep_duration = max(3, time_to_next_candle - 5)
#                 except ValueError:
#                     pass

#             logger.debug(f"Next check in {sleep_duration} seconds...")
#             time.sleep(sleep_duration)

#     except KeyboardInterrupt:
#         logger.info("Live dry run stopped by user (KeyboardInterrupt).")
#     except Exception as e:
#         logger.error(f"Critical error during live dry run: {e}", exc_info=True)
#     finally:
#         logger.info("Live dry run session ending. Saving final data.")
#         save_trades_live(all_trades_session_df, live_trade_log_csv_path, live_trade_log_json_path)
#         logger.info(f"Final data saved. Raw: {raw_data_path}, Indicators: {indicator_data_path}")
#         logger.info("Live dry run completed.")

# if __name__ == '__main__':
#     config_instance = load_config()
#     setup_logging(
#         config_instance.get('log_level', 'INFO'),
#         config_instance.get('log_file_template', 'app.log'),
#         config_instance.run_id
#     )

#     # Initialize InstrumentManager
#     try:
#         api_client = AngelOneAPI()
#         instrument_mgr_instance = InstrumentManager(config=config_instance, broker_client=api_client.get_smart_connect_object())
#         logger.info("InstrumentManager initialized successfully")
#     except Exception as e:
#         logger.error(f"Failed to initialize InstrumentManager: {e}", exc_info=True)
#         sys.exit(1)

#     RUN_ID = datetime.now().strftime("%Y%m%d%H%M%S")

#     # Strategy Configuration
#     if not strategy_factories:
#         logger.warning("`strategy_factories` is empty. Using dummy strategy for testing.")

#         class DummyStrategy:
#             def __init__(self, params):
#                 self.length = params.get('length', 14)
#                 self.threshold = params.get('threshold', 1)

#             def generate_signals(self, data):
#                 signals = pd.DataFrame(index=data.index)
#                 signals['signal'] = 0
#                 if len(data) < self.length: return signals
#                 if data['close'].iloc[-1] > data['close'].iloc[-2] + self.threshold:
#                     signals['signal'].iloc[-1] = 1
#                 elif data['close'].iloc[-1] < data['close'].iloc[-2] - self.threshold:
#                     signals['signal'].iloc[-1] = -1
#                 signals['sl'] = data['low'].iloc[-1] - 10 if signals['signal'].iloc[-1] == 1 else \
#                                 (data['high'].iloc[-1] + 10 if signals['signal'].iloc[-1] == -1 else None)
#                 signals['tp'] = data['high'].iloc[-1] + 20 if signals['signal'].iloc[-1] == 1 else \
#                                 (data['low'].iloc[-1] - 20 if signals['signal'].iloc[-1] == -1 else None)
#                 return signals

#         strategy_factories['DummyLiveStrategy'] = DummyStrategy

#         STRATEGIES_TO_RUN_CONFIG = {
#             "DummyLiveStrategy": {"params": {"length": 5, "threshold": 0.5}}
#         }
#     else:
#         STRATEGIES_TO_RUN_CONFIG = config_instance.get('strategies_to_run', {
#             "DummyLiveStrategy": {"params": {"length": 5, "threshold": 0.5}}
#         })

#     # Override symbol and exchange for options trading
#     SYMBOL_TO_RUN = "NIFTY"  # Use "NIFTY" instead of "NIFTY 50" for NFO
#     TIMEFRAME_TO_RUN = config_instance.get('live_timeframe_to_run', '5min')
#     config_instance.set('exchange', 'NFO')  # Ensure NFO for options

#     if not SYMBOL_TO_RUN or not TIMEFRAME_TO_RUN or not STRATEGIES_TO_RUN_CONFIG:
#         logger.error("Live Dry Run: Symbol, timeframe, or strategies not configured correctly. Exiting.")
#     else:
#         logger.info(f"Starting live dry run for {SYMBOL_TO_RUN}, {TIMEFRAME_TO_RUN} with strategies: {list(STRATEGIES_TO_RUN_CONFIG.keys())}")
#         run_live_dry_mode(SYMBOL_TO_RUN, TIMEFRAME_TO_RUN, STRATEGIES_TO_RUN_CONFIG, RUN_ID, instrument_mgr_instance, config_instance)

import pandas as pd
from termcolor import cprint
from datetime import datetime

class LiveTradingEngine:
    def __init__(self, symbol, ohlcv_history, config, fetcher=None, strategy_func=None):
        self.symbol = symbol
        self.ohlcv_history = ohlcv_history  # {tf: DataFrame}
        self.config = config
        self.fetcher = fetcher  # Optional, for live queries if needed
        self.strategy_func = strategy_func or self.dummy_strategy
        self.trades = {tf: [] for tf in ohlcv_history.keys()}
        self.open_trades = {tf: None for tf in ohlcv_history.keys()}

    def on_new_bar(self, tf, ohlcv_row):
        tf_str = f"{tf}min"
        df = self.ohlcv_history[tf_str]
        df = pd.concat([df, pd.DataFrame([ohlcv_row])], ignore_index=True)
        self.ohlcv_history[tf_str] = df

        # Run dummy strategy (BUY if close>open, SELL if close<open, EXIT otherwise)
        signal = self.strategy_func(df)
        self.handle_signal(tf, ohlcv_row, signal)
        self.print_trade_status(tf, ohlcv_row, signal)

        # --- OPTIONAL: Use fetcher to get LTP/Greeks/etc. ---
        # if self.fetcher:
        #     ltp = self.fetcher.fetch_websocket_ltp(self.symbol)
        #     print(f"[INFO] Live LTP for {self.symbol}: {ltp}")

    def dummy_strategy(self, df):
        if len(df) < 2: return 0
        last = df.iloc[-1]
        if last['close'] > last['open']:
            return 1  # BUY
        elif last['close'] < last['open']:
            return -1  # SELL
        else:
            return 0  # HOLD/EXIT

    def handle_signal(self, tf, ohlcv_row, signal):
        if self.open_trades[tf] is None and signal != 0:
            trade = {
                'entry_time': ohlcv_row['datetime'],
                'entry_price': ohlcv_row['close'],
                'side': 'LONG' if signal == 1 else 'SHORT'
            }
            self.open_trades[tf] = trade
            self.trades[tf].append({**trade, 'exit_time': None, 'exit_price': None, 'pnl': None})
        elif self.open_trades[tf] is not None and signal == 0:
            open_trade = self.open_trades[tf]
            pnl = (ohlcv_row['close'] - open_trade['entry_price']) if open_trade['side'] == 'LONG' else (open_trade['entry_price'] - ohlcv_row['close'])
            open_trade.update({'exit_time': ohlcv_row['datetime'], 'exit_price': ohlcv_row['close'], 'pnl': pnl})
            self.trades[tf][-1].update(open_trade)
            self.open_trades[tf] = None

    def print_trade_status(self, tf, ohlcv_row, signal):
        tf_str = f"{tf}min"
        dt_str = pd.to_datetime(ohlcv_row['datetime']).strftime('%Y-%m-%d %H:%M')
        if self.open_trades[tf] and signal != 0:
            msg = f"[{dt_str}] {tf_str}: {'🟩 LONG' if signal == 1 else '🟥 SHORT'} ENTRY at {ohlcv_row['close']:.2f}"
            cprint(msg, 'green' if signal == 1 else 'red', attrs=['bold'])
        elif signal == 0 and self.trades[tf] and self.trades[tf][-1]['exit_time']:
            pnl = self.trades[tf][-1]['pnl']
            side = self.trades[tf][-1]['side']
            msg = f"[{dt_str}] {tf_str}: EXIT {side} at {ohlcv_row['close']:.2f} | PnL: {pnl:.2f} {'✔️' if pnl >= 0 else '❌'}"
            cprint(msg, 'cyan', attrs=['bold'])

    def save_trades(self, outdir="results/live_trades"):
        import os
        os.makedirs(outdir, exist_ok=True)
        for tf, trade_list in self.trades.items():
            if trade_list:
                df = pd.DataFrame(trade_list)
                tf_str = f"{tf}min"
                df.to_csv(f"{outdir}/{self.symbol}_{tf_str}_trades.csv", index=False)
