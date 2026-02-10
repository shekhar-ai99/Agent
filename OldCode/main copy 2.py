
# # # import argparse
# # # import os
# # # import pandas as pd
# # # import numpy as np
# # # import random
# # # import uuid
# # # import time
# # # from datetime import datetime, timedelta, time as dttime
# # # import pytz
# # # from termcolor import cprint, colored
# # # from tabulate import tabulate
# # # import threading
# # # import logging
# # # import signal
# # # import sys
# # # from concurrent.futures import ThreadPoolExecutor
# # # import backoff

# # # from app.compute_indicators import compute_indicators
# # # from app.strategies import strategy_factories
# # # from angel_one.angel_data_fetcher import AngelDataFetcher
# # # from app.tick_aggregator import TickAggregator
# # # from config import load_config, default_params

# # # try:
# # #     from app.LiveDataFeeder import LiveDataFeeder
# # # except ImportError as e:
# # #     print("INFO: Could not import LiveDataFeeder. Live mode will not be available.:", e)
# # #     LiveDataFeeder = None

# # # INDEX_SYMBOL_MAP = {
# # #     "NIFTY": {
# # #         "hist_symbol": "Nifty 50",
# # #         "hist_token": "99926000",
# # #         "tick_symbol": "NIFTY",
# # #         "tick_token": "26000"
# # #     },
# # #     "BANKNIFTY": {
# # #         "hist_symbol": "Bank Nifty",
# # #         "hist_token": "99926009",
# # #         "tick_symbol": "BANKNIFTY",
# # #         "tick_token": "26009"
# # #     },
# # # }

# # # TIMEFRAMES = ["1min", "3min", "5min", "15min"]
# # # TIMEFRAME_MINUTES = {"1min": 1, "3min": 3, "5min": 5, "15min": 15}
# # # all_trades_g = []
# # # all_trades_lock = threading.Lock()
# # # stop_event_g = threading.Event()
# # # live_feeders_g = []
# # # base_run_dir_g = ""
# # # run_id_g = ""
# # # app_logger_g = None
# # # fetcher_g = None  # Global fetcher instance
# # # fetcher_lock = threading.Lock()

# # # def setup_logging(run_id, base_log_dir):
# # #     global app_logger_g
# # #     os.makedirs(base_log_dir, exist_ok=True)
# # #     log_file_app = os.path.join(base_log_dir, f"app_{run_id}.log")
# # #     logger = logging.getLogger("AlgoTradingApp")
# # #     logger.setLevel(logging.INFO)
# # #     if not logger.handlers:
# # #         fh_app = logging.FileHandler(log_file_app)
# # #         fh_app.setLevel(logging.INFO)
# # #         ch_app = logging.StreamHandler(sys.stdout)
# # #         ch_app.setLevel(logging.INFO)
# # #         formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
# # #         fh_app.setFormatter(formatter)
# # #         ch_app.setFormatter(formatter)
# # #         logger.addHandler(fh_app)
# # #         logger.addHandler(ch_app)
# # #     app_logger_g = logger
# # #     return logger

# # # def get_timeframe_tick_logger(timeframe_str, run_id, base_log_dir):
# # #     os.makedirs(base_log_dir, exist_ok=True)
# # #     log_file_tf = os.path.join(base_log_dir, f"ticks_{timeframe_str}_{run_id}.log")
# # #     tf_logger = logging.getLogger(f"TickLogger_{timeframe_str}")
# # #     tf_logger.setLevel(logging.INFO)
# # #     if not tf_logger.handlers:
# # #         fh_tf = logging.FileHandler(log_file_tf)
# # #         fh_tf.setLevel(logging.INFO)
# # #         formatter = logging.Formatter('%(asctime)s - %(message)s')
# # #         fh_tf.setFormatter(formatter)
# # #         tf_logger.addHandler(fh_tf)
# # #         tf_logger.propagate = False
# # #     return tf_logger

# # # @backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=60, base=2)
# # # def fetch_historical_with_retry(fetcher, symbol, timeframe, days, exchange):
# # #     if fetcher is None:
# # #         raise ValueError("AngelDataFetcher is not initialized")
# # #     return fetcher.fetch_historical_candles(symbol, timeframe, days=days, exchange=exchange)

# # # def initialize_fetcher():
# # #     global fetcher_g
# # #     with fetcher_lock:
# # #         if fetcher_g is None:
# # #             config = load_config()
# # #             try:
# # #                 fetcher_g = AngelDataFetcher(config)
# # #                 app_logger_g.info("Initialized global AngelDataFetcher instance")
# # #             except Exception as e:
# # #                 app_logger_g.error(f"Failed to initialize AngelDataFetcher: {e}")
# # #                 fetcher_g = None
# # #     return fetcher_g

# # # def load_and_prepare_data_tf(symbol_hist_name, exchange, timeframe, tf_paths):
# # #     raw_csv_path = tf_paths["raw_csv"]
# # #     hist_ind_csv_path = tf_paths["hist_ind_csv"]

# # #     if not os.path.exists(raw_csv_path):
# # #         app_logger_g.info(f"[{timeframe}] Fetching historical data for {symbol_hist_name}...")
# # #         fetcher = initialize_fetcher()
# # #         if fetcher is None:
# # #             app_logger_g.error(f"[{timeframe}] AngelDataFetcher initialization failed. Cannot fetch data.")
# # #             return pd.DataFrame()
# # #         try:
# # #             df_raw = fetch_historical_with_retry(fetcher, symbol_hist_name, timeframe, 90, exchange)
# # #             if df_raw.empty:
# # #                 app_logger_g.error(f"[{timeframe}] No data fetched for {symbol_hist_name}.")
# # #                 return pd.DataFrame()
# # #             df_raw.to_csv(raw_csv_path, index=False)
# # #         except Exception as e:
# # #             app_logger_g.error(f"[{timeframe}] Failed to fetch historical data: {e}")
# # #             return pd.DataFrame()
# # #     else:
# # #         app_logger_g.info(f"[{timeframe}] Loading raw data from {raw_csv_path}")
# # #         df_raw = pd.read_csv(raw_csv_path, parse_dates=['datetime'])

# # #     if df_raw.empty:
# # #         return pd.DataFrame()

# # #     if 'datetime' not in df_raw.columns:
# # #         if 'date' in df_raw.columns:
# # #             df_raw.rename(columns={'date': 'datetime'}, inplace=True)
# # #         elif df_raw.index.name == 'datetime':
# # #             df_raw.reset_index(inplace=True)
# # #         else:
# # #             app_logger_g.error(f"[{timeframe}] 'datetime' column not found in {raw_csv_path}.")
# # #             return pd.DataFrame()

# # #     df_raw = ensure_datetime_tz_aware(df_raw.copy(), 'datetime', 'Asia/Kolkata')
# # #     df_raw.set_index('datetime', inplace=True)
# # #     df_raw.sort_index(inplace=True)
# # #     df_raw = df_raw[~df_raw.index.duplicated(keep='last')]

# # #     if not os.path.exists(hist_ind_csv_path):
# # #         app_logger_g.info(f"[{timeframe}] Computing indicators for historical data and saving to {hist_ind_csv_path}")
# # #         df_hist_with_ind = compute_indicators(df_raw.copy(), hist_ind_csv_path)
# # #         if df_hist_with_ind is None or df_hist_with_ind.empty:
# # #             app_logger_g.error(f"[{timeframe}] Indicator computation failed.")
# # #             return pd.DataFrame()
# # #         df_hist_with_ind.to_csv(hist_ind_csv_path)
# # #     else:
# # #         app_logger_g.info(f"[{timeframe}] Loading historical indicators from {hist_ind_csv_path}")
# # #         df_hist_with_ind = pd.read_csv(hist_ind_csv_path, parse_dates=['datetime'])
# # #         df_hist_with_ind = ensure_datetime_tz_aware(df_hist_with_ind.copy(), 'datetime', 'Asia/Kolkata')
# # #         df_hist_with_ind.set_index('datetime', inplace=True)
# # #         df_hist_with_ind.sort_index(inplace=True)
# # #         df_hist_with_ind = df_hist_with_ind[~df_hist_with_ind.index.duplicated(keep='last')]

# # #     return df_hist_with_ind

# # # def ensure_datetime_tz_aware(df, col='datetime', tz_str='Asia/Kolkata'):
# # #     target_tz = pytz.timezone(tz_str)
# # #     if col not in df.columns:
# # #         if df.index.name == col:
# # #             df = df.reset_index()
# # #         else:
# # #             app_logger_g.error(f"'{col}' column or index not found in DataFrame.")
# # #             raise ValueError(f"'{col}' column or index not found in DataFrame.")

# # #     df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.tz_convert(target_tz)
# # #     if df[col].isnull().any():
# # #         app_logger_g.warning(f"[{timeframe}] NaNs introduced in '{col}' during conversion.")
# # #     return df

# # # def get_timeframe_specific_paths(main_run_dir, timeframe, symbol_name_safe):
# # #     tf_dir = os.path.join(main_run_dir, timeframe)
# # #     data_dir = os.path.join(tf_dir, "data")
# # #     raw_dir = os.path.join(data_dir, "raw")
# # #     ind_dir = os.path.join(data_dir, "datawithindicators")
# # #     trades_dir = os.path.join(tf_dir, "trades")

# # #     for d in [raw_dir, ind_dir, trades_dir]:
# # #         os.makedirs(d, exist_ok=True)

# # #     return {
# # #         "tf_base": tf_dir, "data_dir": data_dir, "raw_dir": raw_dir, "ind_dir": ind_dir,
# # #         "trades_dir": trades_dir,
# # #         "raw_csv": os.path.join(raw_dir, f"{symbol_name_safe}_{timeframe}_historical.csv"),
# # #         "hist_ind_csv": os.path.join(ind_dir, f"{symbol_name_safe}_{timeframe}_historical_with_indicators.csv")
# # #     }

# # # def print_trade_action_tf(trade, action_type, timeframe):
# # #     strat = trade['strategy']
# # #     side = trade['side']
# # #     price = trade['entry_price'] if action_type == "OPEN" else trade.get('exit_price', 'N/A')
# # #     dt = trade['entry_time'] if action_type == "OPEN" else trade.get('exit_time', 'N/A')
# # #     log_msg = ""
# # #     console_color = "white"
# # #     attrs = ['bold']

# # #     if action_type == "OPEN":
# # #         arrow = "🟩" if side == 'LONG' else "🟥"
# # #         console_color = 'green' if side == 'LONG' else 'red'
# # #         log_msg = (f"[{timeframe}] {arrow} [{strat}] {action_type} {side} at {price:.2f} "
# # #                    f"(SL={trade.get('sl', 'N/A')}, TP={trade.get('tp', 'N/A')}) @ {dt}")
# # #     else:
# # #         pnl = trade.get('pnl', 0)
# # #         emoji = '✔️' if pnl > 0 else ('❌' if pnl < 0 else '➖')
# # #         console_color = 'green' if pnl > 0 else ('red' if pnl < 0 else 'white')
# # #         log_msg = (f"[{timeframe}] 🔴 [{strat}] EXIT {side} at {price:.2f} | PnL: {pnl:.2f} {emoji} "
# # #                    f"({trade.get('exit_reason', 'N/A')}) @ {dt}")

# # #     app_logger_g.info(log_msg)
# # #     cprint(log_msg, console_color, attrs=attrs)

# # # def save_trade_to_timeframe_strategy_csv(trade_details, trades_dir_tf, strategy_name, timeframe):
# # #     filename = os.path.join(trades_dir_tf, f"trades_{strategy_name}_{timeframe}.csv")
# # #     df_trade = pd.DataFrame([trade_details])
# # #     for col in ['entry_time', 'exit_time']:
# # #         if col in df_trade.columns and pd.api.types.is_datetime64_any_dtype(df_trade[col]):
# # #             df_trade[col] = df_trade[col].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
# # #     header = not os.path.exists(filename)
# # #     try:
# # #         df_trade.to_csv(filename, mode='a', header=header, index=False)
# # #     except Exception as e:
# # #         app_logger_g.error(f"[{timeframe}] Failed to save trade to {filename}: {e}")

# # # def generate_consolidated_report(all_trades_list_global, main_run_dir, report_name_prefix="consolidated"):
# # #     if not all_trades_list_global:
# # #         msg = "No trades were executed across any timeframe for this session."
# # #         app_logger_g.info(msg)
# # #         cprint(f"\n{msg}", "red", attrs=['bold'])
# # #         return

# # #     df_all_trades = pd.DataFrame(all_trades_list_global)
# # #     df_all_trades['pnl'] = pd.to_numeric(df_all_trades['pnl'], errors='coerce').fillna(0)
# # #     df_all_trades['entry_price'] = pd.to_numeric(df_all_trades['entry_price'], errors='coerce')
# # #     df_all_trades['exit_price'] = pd.to_numeric(df_all_trades['exit_price'], errors='coerce')

# # #     for col_name in ['entry_time', 'exit_time']:
# # #         if col_name in df_all_trades.columns and not pd.api.types.is_string_dtype(df_all_trades[col_name]):
# # #             if pd.api.types.is_datetime64_any_dtype(df_all_trades[col_name]):
# # #                 df_all_trades[col_name] = df_all_trades[col_name].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
# # #             else:
# # #                 df_all_trades[col_name] = df_all_trades[col_name].astype(str)

# # #     report_header = f"\n==== CONSOLIDATED TRADE REPORT ({report_name_prefix}) ====\nRun ID: {run_id_g}\n"
# # #     app_logger_g.info(report_header)
# # #     cprint(report_header, 'yellow', attrs=['bold', 'underline'])

# # #     total_trades_overall = len(df_all_trades)
# # #     total_pnl_overall = df_all_trades['pnl'].sum()
# # #     wins_overall = (df_all_trades['pnl'] > 0).sum()
# # #     losses_overall = (df_all_trades['pnl'] <= 0).sum()

# # #     overall_summary_text = (
# # #         f"OVERALL SUMMARY:\n"
# # #         f"Total Trades: {total_trades_overall}\n"
# # #         f"Profitable Trades: {wins_overall}\n"
# # #         f"Losing/BE Trades: {losses_overall}\n"
# # #         f"Total PnL: {total_pnl_overall:.2f}\n"
# # #     )
# # #     app_logger_g.info(overall_summary_text)
# # #     cprint("OVERALL SUMMARY:", 'cyan', attrs=['bold'])
# # #     cprint(f"Total Trades: {total_trades_overall}", 'cyan')
# # #     cprint(f"Profitable Trades: {wins_overall}", 'cyan')
# # #     cprint(f"Losing/BE Trades: {losses_overall}", 'cyan')
# # #     cprint(f"Total PnL: {total_pnl_overall:.2f}\n", 'magenta', attrs=['bold'])

# # #     summary_by_tf = df_all_trades.groupby('timeframe')['pnl'].agg(
# # #         TotalTrades='count',
# # #         Wins=lambda x: (x > 0).sum(),
# # #         LossesBE=lambda x: (x <= 0).sum(),
# # #         TotalPnL='sum'
# # #     )
# # #     tf_summary_text = f"SUMMARY BY TIMEFRAME:\n{summary_by_tf.to_string()}\n"
# # #     app_logger_g.info(tf_summary_text)
# # #     cprint("SUMMARY BY TIMEFRAME:", 'cyan', attrs=['bold'])
# # #     print(tabulate(summary_by_tf, headers='keys', tablefmt='fancy_grid', floatfmt=".2f"))
# # #     print("\n")

# # #     summary_by_strat = df_all_trades.groupby('strategy')['pnl'].agg(
# # #         TotalTrades='count',
# # #         Wins=lambda x: (x > 0).sum(),
# # #         LossesBE=lambda x: (x <= 0).sum(),
# # #         TotalPnL='sum'
# # #     )
# # #     strat_summary_text = f"SUMMARY BY STRATEGY (Overall):\n{summary_by_strat.to_string()}\n"
# # #     app_logger_g.info(strat_summary_text)
# # #     cprint("SUMMARY BY STRATEGY (Overall):", 'cyan', attrs=['bold'])
# # #     print(tabulate(summary_by_strat, headers='keys', tablefmt='fancy_grid', floatfmt=".2f"))
# # #     print("\n")

# # #     cols_to_display = ['timeframe', 'strategy', 'side', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'pnl', 'exit_reason']
# # #     cols_to_display = [col for col in cols_to_display if col in df_all_trades.columns]

# # #     detailed_trades_text = f"DETAILED TRADES:\n{df_all_trades[cols_to_display].to_string(index=False)}\n"
# # #     app_logger_g.info(detailed_trades_text)
# # #     cprint("DETAILED TRADES:", 'yellow', attrs=['bold'])
# # #     print(tabulate(df_all_trades[cols_to_display], headers='keys', tablefmt='fancy_grid', showindex=False, floatfmt=".2f"))

# # #     report_csv_path = os.path.join(main_run_dir, f"{report_name_prefix}_trades_consolidated_report.csv")
# # #     df_all_trades.to_csv(report_csv_path, index=False)
# # #     msg_saved = f"Consolidated trade report saved to: {report_csv_path}"
# # #     app_logger_g.info(msg_saved)
# # #     cprint(f"\n{msg_saved}", "green")

# # # def simulate_future_candles_tf(df_raw_hist, sim_date_dt, timeframe, tz, token, tf_tick_logger):
# # #     if df_raw_hist.empty or 'close' not in df_raw_hist.columns:
# # #         app_logger_g.warning(f"[{timeframe}] Cannot simulate future candles. Historical data is empty or missing 'close'.")
# # #         return pd.DataFrame()

# # #     last_close = int(df_raw_hist['close'].iloc[-1] * 100)
# # #     minutes = int(timeframe.replace("min", ""))
# # #     total_market_minutes = int((dttime(15, 30).hour * 60 + dttime(15, 30).minute) -
# # #                               (dttime(9, 15).hour * 60 + dttime(9, 15).minute))

# # #     sim_candles_list = simulate_and_aggregate_candles(
# # #         start_price=last_close,
# # #         total_minutes=total_market_minutes,
# # #         interval_minutes=minutes,
# # #         token=token,
# # #         tick_logger=tf_tick_logger
# # #     )

# # #     market_open_time = tz.localize(datetime.combine(sim_date_dt, dttime(9, 15)))
# # #     for i, candle_dict in enumerate(sim_candles_list):
# # #         candle_dict['datetime'] = market_open_time + timedelta(minutes=i * minutes)

# # #     app_logger_g.info(f"[{timeframe}] Simulated {len(sim_candles_list)} future candles for {sim_date_dt.date()}.")
# # #     return pd.DataFrame(sim_candles_list)

# # # def simulate_and_aggregate_candles(start_price, total_minutes, interval_minutes, token, tick_logger=None):
# # #     aggregator = TickAggregator(interval_minutes=interval_minutes)
# # #     candles = []
# # #     current_sim_price = start_price
# # #     num_candles_to_sim = total_minutes // interval_minutes

# # #     for _ in range(num_candles_to_sim):
# # #         ticks_this_interval = simulate_ticks_for_interval(current_sim_price, interval_minutes * 60, num_ticks=60, token=token, tick_logger=tick_logger)
# # #         for tick in ticks_this_interval:
# # #             aggregated_candle = aggregator.process_tick(tick)
# # #             if aggregated_candle:
# # #                 candles.append(aggregated_candle)
# # #                 if 'close' in aggregated_candle:
# # #                     current_sim_price = aggregated_candle['close']
# # #                 break
# # #         last_candle = aggregator.force_close()
# # #         if last_candle:
# # #             candles.append(last_candle)
# # #             if 'close' in last_candle:
# # #                 current_sim_price = last_candle['close']

# # #     for c in candles:
# # #         for key_price in ['open', 'high', 'low', 'close']:
# # #             if key_price in c:
# # #                 c[key_price] /= 100.0
# # #     return candles

# # # def simulate_ticks_for_interval(start_price, interval_seconds, num_ticks, token, tick_logger=None):
# # #     ticks = []
# # #     current_price = start_price
# # #     base_sim_timestamp_ms = int(time.time() * 1000)
# # #     for i in range(num_ticks):
# # #         price_change = random.randint(-5, 5)
# # #         current_price = max(1, current_price + price_change)
# # #         tick = {
# # #             'subscription_mode': 1,
# # #             'exchange_type': 1,
# # #             'token': token,
# # #             'sequence_number': i,
# # #             'exchange_timestamp': base_sim_timestamp_ms + (i * (interval_seconds * 1000 // num_ticks)),
# # #             'last_traded_price': current_price,
# # #             'subscription_mode_val': 'LTP'
# # #         }
# # #         ticks.append(tick)
# # #         if tick_logger:
# # #             tick_logger.info(f"SIM_TICK: {tick}")
# # #     return ticks

# # # def merge_and_compute_indicators_sim_tf(df_raw_hist_indexed, df_sim_future, tf_paths, symbol_name_safe, timeframe, sim_date_str):
# # #     if 'datetime' in df_sim_future.columns:
# # #         df_sim_future_indexed = df_sim_future.set_index('datetime')
# # #     else:
# # #         df_sim_future_indexed = df_sim_future

# # #     if not df_sim_future_indexed.empty and df_sim_future_indexed.index.tzinfo is None:
# # #         df_sim_future_indexed = df_sim_future_indexed.tz_localize('Asia/Kolkata', ambiguous='infer', nonexistent='shift_forward')

# # #     df_full = pd.concat([df_raw_hist_indexed, df_sim_future_indexed], axis=0)
# # #     df_full.sort_index(inplace=True)
# # #     df_full = df_full[~df_full.index.duplicated(keep='last')]

# # #     out_file_path = os.path.join(tf_paths["ind_dir"], f"{symbol_name_safe}_{timeframe}_sim_{sim_date_str}_full_with_indicators.csv")
# # #     df_all_with_ind = compute_indicators(df_full.copy(), out_file_path)
# # #     if df_all_with_ind is None or df_all_with_ind.empty:
# # #         app_logger_g.error(f"[{timeframe}] Indicator computation failed for simulated data.")
# # #         return pd.DataFrame()
# # #     df_all_with_ind.to_csv(out_file_path)
# # #     app_logger_g.info(f"[{timeframe}] Saved SIM merged data with indicators to: {out_file_path}")
# # #     return df_all_with_ind

# # # def run_strategies_on_simulated_day_tf(simulated_day_with_ind, full_history_with_ind, strategies_config, timeframe, tf_paths, sim_date_str):
# # #     trades_for_this_tf_and_strat = {}
# # #     active_trades_sim = {}

# # #     if simulated_day_with_ind.empty:
# # #         app_logger_g.warning(f"[{timeframe}] No simulated day data for {sim_date_str} to run strategies.")
# # #         return

# # #     for idx, (dt_index, candle_row) in enumerate(simulated_day_with_ind.iterrows()):
# # #         current_history_for_signal = full_history_with_ind.loc[:dt_index]

# # #         for strat_name, strat_params in strategies_config.items():
# # #             if strat_name not in strategy_factories:
# # #                 app_logger_g.error(f"[{timeframe}] Strategy {strat_name} not found!")
# # #                 continue

# # #             strat_func = strategy_factories[strat_name]
# # #             out = strat_func(candle_row, current_history_for_signal, strat_params)
# # #             active_trade = active_trades_sim.get(strat_name)

# # #             if out['signal'] in ('buy_potential', 'sell_potential'):
# # #                 if not active_trade or active_trade['status'] == 'closed':
# # #                     trade = {
# # #                         'run_id': run_id_g,
# # #                         'timeframe': timeframe,
# # #                         'strategy': strat_name,
# # #                         'entry_time': dt_index,
# # #                         'entry_price': candle_row['close'],
# # #                         'side': 'LONG' if out['signal'] == 'buy_potential' else 'SHORT',
# # #                         'sl': out.get('sl'),
# # #                         'tp': out.get('tp'),
# # #                         'status': 'open',
# # #                         'exit_time': pd.NaT,
# # #                         'exit_price': np.nan,
# # #                         'pnl': np.nan,
# # #                         'exit_reason': None
# # #                     }
# # #                     active_trades_sim[strat_name] = trade
# # #                     print_trade_action_tf(trade, "OPEN", timeframe)

# # #             elif active_trade and active_trade['status'] == 'open':
# # #                 exit_trade_flag = False
# # #                 exit_price_val = np.nan
# # #                 exit_reason_str = ""

# # #                 sl = active_trade.get('sl')
# # #                 tp = active_trade.get('tp')

# # #                 if sl is not None and pd.notna(sl):
# # #                     if (active_trade['side'] == 'LONG' and candle_row['low'] <= sl) or \
# # #                        (active_trade['side'] == 'SHORT' and candle_row['high'] >= sl):
# # #                         exit_trade_flag = True
# # #                         exit_price_val = sl
# # #                         exit_reason_str = "Stop Loss"

# # #                 if not exit_trade_flag and tp is not None and pd.notna(tp):
# # #                     if (active_trade['side'] == 'LONG' and candle_row['high'] >= tp) or \
# # #                        (active_trade['side'] == 'SHORT' and candle_row['low'] <= tp):
# # #                         exit_trade_flag = True
# # #                         exit_price_val = tp
# # #                         exit_reason_str = "Target Profit"

# # #                 if not exit_trade_flag and idx == len(simulated_day_with_ind) - 1:
# # #                     exit_trade_flag = True
# # #                     exit_price_val = candle_row['close']
# # #                     exit_reason_str = f"EOD_SIM_{sim_date_str}"

# # #                 if exit_trade_flag:
# # #                     pnl_val = (exit_price_val - active_trade['entry_price']) if active_trade['side'] == 'LONG' else (active_trade['entry_price'] - exit_price_val)
# # #                     active_trade.update({
# # #                         'exit_time': dt_index,
# # #                         'exit_price': exit_price_val,
# # #                         'pnl': pnl_val,
# # #                         'status': 'closed',
# # #                         'exit_reason': exit_reason_str
# # #                     })
# # #                     print_trade_action_tf(active_trade, "EXIT", timeframe)
# # #                     if strat_name not in trades_for_this_tf_and_strat:
# # #                         trades_for_this_tf_and_strat[strat_name] = []
# # #                     trades_for_this_tf_and_strat[strat_name].append(active_trade.copy())
# # #                     with all_trades_lock:
# # #                         all_trades_g.append(active_trade.copy())
# # #                     active_trades_sim[strat_name] = {'status': 'closed'}

# # #     for strat_name, trade_to_close in active_trades_sim.items():
# # #         if trade_to_close and trade_to_close.get('status') == 'open':
# # #             final_candle = simulated_day_with_ind.iloc[-1]
# # #             exit_price_eod = final_candle['close']
# # #             pnl_eod = (exit_price_eod - trade_to_close['entry_price']) if trade_to_close['side'] == 'LONG' else (trade_to_close['entry_price'] - exit_price_eod)
# # #             trade_to_close.update({
# # #                 'exit_time': simulated_day_with_ind.index[-1],
# # #                 'exit_price': exit_price_eod,
# # #                 'pnl': pnl_eod,
# # #                 'status': 'closed',
# # #                 'exit_reason': f"EOD_SIM_FORCE_CLOSE_{sim_date_str}"
# # #             })
# # #             print_trade_action_tf(trade_to_close, "EXIT", timeframe)
# # #             if strat_name not in trades_for_this_tf_and_strat:
# # #                 trades_for_this_tf_and_strat[strat_name] = []
# # #             trades_for_this_tf_and_strat[strat_name].append(trade_to_close.copy())
# # #             with all_trades_lock:
# # #                 all_trades_g.append(trade_to_close.copy())

# # #     for strat_name_key, strat_trades_list in trades_for_this_tf_and_strat.items():
# # #         if strat_trades_list:
# # #             save_trade_to_timeframe_strategy_csv(strat_trades_list, tf_paths["trades_dir"], strat_name_key, timeframe)
# # #     app_logger_g.info(f"[{timeframe}] Finished strategy run for {sim_date_str}. All trades for this TF saved.")

# # # def run_single_simulation_task(timeframe, index_key, exchange, main_run_dir_sim, sim_date_dt, tz, strategies_config, tick_log_dir_sim):
# # #     app_logger_g.info(f"SIM_TASK [{timeframe}] Starting simulation for date: {sim_date_dt.date()}")
# # #     symbol_config = INDEX_SYMBOL_MAP[index_key]
# # #     symbol_hist_name = symbol_config["hist_symbol"]
# # #     symbol_name_safe = symbol_hist_name.replace(" ", "_")
# # #     tf_paths = get_timeframe_specific_paths(main_run_dir_sim, timeframe, symbol_name_safe)
# # #     tf_tick_logger_sim = get_timeframe_tick_logger(timeframe, run_id_g + "_sim", tick_log_dir_sim)

# # #     df_hist_ind = load_and_prepare_data_tf(symbol_hist_name, exchange, timeframe, tf_paths)
# # #     if df_hist_ind.empty:
# # #         app_logger_g.error(f"SIM_TASK [{timeframe}] Failed to load historical data. Aborting.")
# # #         return timeframe

# # #     df_sim_future = simulate_future_candles_tf(df_hist_ind.iloc[[-1]].reset_index(), sim_date_dt, timeframe, tz, symbol_config["hist_token"], tf_tick_logger_sim)
# # #     df_all_data_with_ind = merge_and_compute_indicators_sim_tf(df_hist_ind.copy(), df_sim_future.copy(), tf_paths, symbol_name_safe, timeframe, sim_date_dt.strftime("%Y%m%d"))
# # #     if df_all_data_with_ind.empty:
# # #         app_logger_g.error(f"SIM_TASK [{timeframe}] Failed after merging/indicators. Aborting.")
# # #         return timeframe

# # #     simulated_day_data = df_all_data_with_ind.loc[df_all_data_with_ind.index.date == sim_date_dt.date()].copy()
# # #     if simulated_day_data.empty:
# # #         app_logger_g.warning(f"SIM_TASK [{timeframe}] No data found for simulated date {sim_date_dt.date()}. Using latest available.")
# # #         target_date_to_run = df_all_data_with_ind.index.max().date()
# # #         simulated_day_data = df_all_data_with_ind.loc[df_all_data_with_ind.index.date == target_date_to_run].copy()
# # #         if simulated_day_data.empty:
# # #             app_logger_g.error(f"SIM_TASK [{timeframe}] Still no data to run strategies. Aborting.")
# # #             return timeframe

# # #     run_strategies_on_simulated_day_tf(simulated_day_data, df_all_data_with_ind, strategies_config, timeframe, tf_paths, sim_date_dt.strftime("%Y%m%d"))
# # #     app_logger_g.info(f"SIM_TASK [{timeframe}] Simulation task finished.")
# # #     return timeframe

# # # def run_simulation_orchestrator():
# # #     global base_run_dir_g, run_id_g, fetcher_g
# # #     run_id_g = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:4]
# # #     base_run_dir_g = os.path.join("simulation_runs", run_id_g)
# # #     log_dir = os.path.join(base_run_dir_g, "logs")
# # #     setup_logging(run_id_g, log_dir)
# # #     app_logger_g.info(f"========== MULTI-TIMEFRAME SIMULATION MODE (Run ID: {run_id_g}) ==========")

# # #     index_key = "NIFTY"
# # #     exchange = "NSE"
# # #     timeframes_to_sim = TIMEFRAMES
# # #     sim_date_dt = datetime(2025, 5, 28)
# # #     tz = pytz.timezone("Asia/Kolkata")
# # #     strategies_config = {name: default_params.get(name, {}) for name in strategy_factories.keys()}
# # #     all_trades_g.clear()
# # #     fetcher_g = None  # Reset fetcher

# # #     with ThreadPoolExecutor(max_workers=len(timeframes_to_sim), thread_name_prefix="SimWorker") as executor:
# # #         futures = []
# # #         for tf in timeframes_to_sim:
# # #             future = executor.submit(run_single_simulation_task, tf, index_key, exchange, base_run_dir_g, sim_date_dt, tz, strategies_config, log_dir)
# # #             futures.append(future)
# # #             time.sleep(2)  # Stagger thread starts
# # #         for future in futures:
# # #             try:
# # #                 completed_tf = future.result()
# # #                 app_logger_g.info(f"Simulation future for timeframe {completed_tf} completed.")
# # #             except Exception as exc:
# # #                 app_logger_g.error(f"A simulation task generated an exception: {exc}", exc_info=True)

# # #     app_logger_g.info("All simulation tasks submitted and processed.")
# # #     generate_consolidated_report(all_trades_g, base_run_dir_g, report_name_prefix=f"sim_{sim_date_dt.strftime('%Y%m%d')}")
# # #     app_logger_g.info(f"[DONE] All simulations complete for {sim_date_dt.strftime('%d-%b-%Y')}. Details in: {base_run_dir_g}")

# # # def live_timeframe_processor_thread_worker(timeframe, index_key, exchange, main_run_dir_live, strategies_config, tz, tick_log_dir_live):
# # #     if LiveDataFeeder is None:
# # #         app_logger_g.error(f"[{timeframe}] LiveDataFeeder not available. Cannot start live processor.")
# # #         return

# # #     thread_name = threading.current_thread().name
# # #     app_logger_g.info(f"[{timeframe}/{thread_name}] Initializing live processor...")
# # #     interval_minutes = int(timeframe.replace("min", ""))
# # #     symbol_config = INDEX_SYMBOL_MAP[index_key]
# # #     symbol_hist_name = symbol_config["hist_symbol"]
# # #     symbol_name_safe = symbol_hist_name.replace(" ", "_")
# # #     tick_token_live = symbol_config["tick_token"]
# # #     live_feed_tokens = [{"exchangeType": 1, "tokens": [tick_token_live]}]
# # #     tf_paths = get_timeframe_specific_paths(main_run_dir_live, timeframe, symbol_name_safe)
# # #     tf_tick_logger = get_timeframe_tick_logger(timeframe, run_id_g + "_live", tick_log_dir_live)

# # #     current_history_with_ind = load_and_prepare_data_tf(symbol_hist_name, exchange, timeframe, tf_paths)
# # #     if current_history_with_ind.empty:
# # #         app_logger_g.error(f"[{timeframe}/{thread_name}] Failed to load initial historical data. Live processor stopping.")
# # #         return
# # #     app_logger_g.info(f"[{timeframe}/{thread_name}] Initial historical data loaded. Last record: {current_history_with_ind.index[-1]}")

# # #     active_trades_live_tf = {}

# # #     def handle_new_live_candle(candle_dict):
# # #         nonlocal current_history_with_ind, active_trades_live_tf
# # #         try:
# # #             dt_object = candle_dict.get('datetime')
# # #             if not dt_object or not isinstance(dt_object, datetime):
# # #                 app_logger_g.error(f"[{timeframe}] Invalid or missing datetime in candle: {candle_dict}")
# # #                 return

# # #             if dt_object.tzinfo is None:
# # #                 dt_object = tz.localize(dt_object)
# # #             else:
# # #                 dt_object = dt_object.astimezone(tz)
# # #             candle_dict['datetime'] = dt_object

# # #             cprint(f"[{timeframe}] Candle: {dt_object.strftime('%H:%M:%S')} O:{candle_dict['open']:.2f} H:{candle_dict['high']:.2f} L:{candle_dict['low']:.2f} C:{candle_dict['close']:.2f} V:{candle_dict.get('volume',0)}", "blue")

# # #             new_candle_df = pd.DataFrame([candle_dict])
# # #             new_candle_df.set_index('datetime', inplace=True)
# # #             temp_history = pd.concat([current_history_with_ind, new_candle_df], axis=0)
# # #             temp_history.sort_index(inplace=True)
# # #             temp_history = temp_history[~temp_history.index.duplicated(keep='last')]

# # #             out_file_path = os.path.join(tf_paths["ind_dir"], f"{symbol_name_safe}_{timeframe}_live_temp.csv")
# # #             updated_full_history_with_ind = compute_indicators(temp_history.copy(), out_file_path)
# # #             if updated_full_history_with_ind is None or updated_full_history_with_ind.empty:
# # #                 app_logger_g.error(f"[{timeframe}] Indicator computation failed for new candle. Skipping strategies.")
# # #                 return
# # #             current_history_with_ind = updated_full_history_with_ind

# # #             current_candle_data_row = current_history_with_ind.loc[dt_object]
# # #             history_for_strategies = current_history_with_ind.loc[:dt_object]

# # #             for strat_name, strat_params in strategies_config.items():
# # #                 if stop_event_g.is_set():
# # #                     break
# # #                 if strat_name not in strategy_factories:
# # #                     continue
# # #                 strat_func = strategy_factories[strat_name]
# # #                 out = strat_func(current_candle_data_row, history_for_strategies, strat_params)
# # #                 active_trade = active_trades_live_tf.get(strat_name)

# # #                 if out['signal'] in ('buy_potential', 'sell_potential'):
# # #                     if not active_trade or active_trade['status'] == 'closed':
# # #                         trade = {
# # #                             'run_id': run_id_g,
# # #                             'timeframe': timeframe,
# # #                             'strategy': strat_name,
# # #                             'entry_time': dt_object,
# # #                             'entry_price': current_candle_data_row['close'],
# # #                             'side': 'LONG' if out['signal'] == 'buy_potential' else 'SHORT',
# # #                             'sl': out.get('sl'),
# # #                             'tp': out.get('tp'),
# # #                             'status': 'open',
# # #                             'exit_time': pd.NaT,
# # #                             'exit_price': np.nan,
# # #                             'pnl': np.nan,
# # #                             'exit_reason': None
# # #                         }
# # #                         active_trades_live_tf[strat_name] = trade
# # #                         print_trade_action_tf(trade, "OPEN", timeframe)

# # #                 elif active_trade and active_trade['status'] == 'open':
# # #                     exit_trade_flag = False
# # #                     exit_price_val = np.nan
# # #                     exit_reason_str = ""
# # #                     sl = active_trade.get('sl')
# # #                     tp = active_trade.get('tp')

# # #                     if sl and pd.notna(sl):
# # #                         if (active_trade['side'] == 'LONG' and current_candle_data_row['low'] <= sl) or \
# # #                            (active_trade['side'] == 'SHORT' and current_candle_data_row['high'] >= sl):
# # #                             exit_trade_flag = True
# # #                             exit_price_val = sl
# # #                             exit_reason_str = "Stop Loss"

# # #                     if not exit_trade_flag and tp and pd.notna(tp):
# # #                         if (active_trade['side'] == 'LONG' and current_candle_data_row['high'] >= tp) or \
# # #                            (active_trade['side'] == 'SHORT' and current_candle_data_row['low'] <= tp):
# # #                             exit_trade_flag = True
# # #                             exit_price_val = tp
# # #                             exit_reason_str = "Target Profit"

# # #                     market_official_close_dt = dttime(15, 30)
# # #                     if not exit_trade_flag and (dt_object + timedelta(minutes=interval_minutes)).time() >= market_official_close_dt:
# # #                         exit_trade_flag = True
# # #                         exit_price_val = current_candle_data_row['close']
# # #                         exit_reason_str = "Market End Closure"

# # #                     if exit_trade_flag:
# # #                         pnl_val = (exit_price_val - active_trade['entry_price']) if active_trade['side'] == 'LONG' else (active_trade['entry_price'] - exit_price_val)
# # #                         active_trade.update({
# # #                             'exit_time': dt_object,
# # #                             'exit_price': exit_price_val,
# # #                             'pnl': pnl_val,
# # #                             'status': 'closed',
# # #                             'exit_reason': exit_reason_str
# # #                         })
# # #                         print_trade_action_tf(active_trade, "EXIT", timeframe)
# # #                         save_trade_to_timeframe_strategy_csv(active_trade.copy(), tf_paths["trades_dir"], strat_name, timeframe)
# # #                         with all_trades_lock:
# # #                             all_trades_g.append(active_trade.copy())
# # #                         active_trades_live_tf[strat_name] = {'status': 'closed'}

# # #         except Exception as e_candle:
# # #             app_logger_g.error(f"[{timeframe}] CRITICAL ERROR in handle_new_live_candle: {e_candle}", exc_info=True)

# # #     def handle_new_live_tick(tick_data):
# # #         if tf_tick_logger:
# # #             tf_tick_logger.info(f"LIVE_TICK: {tick_data}")

# # #     feeder_instance = None
# # #     try:
# # #         feeder_instance = LiveDataFeeder(
# # #             tokens=live_feed_tokens,
# # #             interval_minutes=interval_minutes,
# # #             candle_callback=handle_new_live_candle,
# # #             tick_callback=handle_new_live_tick
# # #         )
# # #         with all_trades_lock:
# # #             live_feeders_g.append(feeder_instance)
# # #         feeder_instance.start()
# # #         app_logger_g.info(f"[{timeframe}/{thread_name}] LiveDataFeeder started.")

# # #         while not stop_event_g.is_set():
# # #             if hasattr(feeder_instance, 'is_alive') and not feeder_instance.is_alive():
# # #                 app_logger_g.warning(f"[{timeframe}/{thread_name}] Feeder is no longer alive. Worker stopping.")
# # #                 break
# # #             now_live_tf = datetime.now(tz)
# # #             if now_live_tf.time() >= dttime(15, 30, tzinfo=tz):
# # #                 app_logger_g.info(f"[{timeframe}/{thread_name}] Market hours ended. Requesting feeder stop.")
# # #                 break
# # #             time.sleep(2)

# # #     except Exception as e_feeder_setup:
# # #         app_logger_g.error(f"[{timeframe}/{thread_name}] Failed to initialize or run LiveDataFeeder: {e_feeder_setup}", exc_info=True)
# # #     finally:
# # #         app_logger_g.info(f"[{timeframe}/{thread_name}] Live processor thread worker stopping...")
# # #         if feeder_instance and hasattr(feeder_instance, 'stop'):
# # #             try:
# # #                 app_logger_g.info(f"[{timeframe}/{thread_name}] Calling feeder.stop().")
# # #                 feeder_instance.stop()
# # #             except Exception as e_feeder_stop_final:
# # #                 app_logger_g.error(f"[{timeframe}/{thread_name}] Error stopping feeder: {e_feeder_stop_final}", exc_info=True)

# # # def run_live_orchestrator():
# # #     global base_run_dir_g, run_id_g, live_feeders_g, fetcher_g
# # #     run_id_g = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:4]
# # #     base_run_dir_g = os.path.join("live_session_runs", run_id_g)
# # #     log_dir = os.path.join(base_run_dir_g, "logs")
# # #     setup_logging(run_id_g, log_dir)
# # #     app_logger_g.info(f"========== MULTI-TIMEFRAME LIVE MODE (Run ID: {run_id_g}) ==========")

# # #     index_key = "NIFTY"
# # #     exchange = "NSE"
# # #     timeframes_to_run = TIMEFRAMES
# # #     tz = pytz.timezone("Asia/Kolkata")
# # #     strategies_config = {name: default_params.get(name, {}) for name in strategy_factories.keys()}
# # #     all_trades_g.clear()
# # #     live_feeders_g.clear()
# # #     stop_event_g.clear()
# # #     fetcher_g = None  # Reset fetcher

# # #     live_threads = []
# # #     for tf_str in timeframes_to_run:
# # #         thread = threading.Thread(
# # #             target=live_timeframe_processor_thread_worker,
# # #             args=(tf_str, index_key, exchange, base_run_dir_g, strategies_config, tz, log_dir),
# # #             name=f"LiveWorker-{tf_str}"
# # #         )
# # #         live_threads.append(thread)
# # #         thread.daemon = True
# # #         thread.start()
# # #         time.sleep(2)  # Stagger thread starts

# # #     app_logger_g.info("All live timeframe processor threads initiated.")
# # #     cprint("Live session started. Press Ctrl+C to stop and generate report.", "magenta")

# # #     try:
# # #         while not stop_event_g.is_set():
# # #             now = datetime.now(tz)
# # #             if now.time() >= dttime(15, 32, tzinfo=tz):
# # #                 app_logger_g.info("Market hours officially over. Initiating EOD shutdown.")
# # #                 stop_event_g.set()
# # #                 break
# # #             if not any(t.is_alive() for t in live_threads) and live_threads:
# # #                 app_logger_g.warning("All live worker threads seem to have terminated prematurely. Shutting down.")
# # #                 stop_event_g.set()
# # #                 break
# # #             time.sleep(5)

# # #     except KeyboardInterrupt:
# # #         app_logger_g.info("Ctrl+C pressed by user. Orchestrator initiating shutdown.")
# # #     finally:
# # #         app_logger_g.info("Orchestrator shutdown sequence started...")
# # #         if not stop_event_g.is_set():
# # #             stop_event_g.set()

# # #         app_logger_g.info("Requesting all LiveDataFeeders to stop...")
# # #         feeders_to_stop = []
# # #         with all_trades_lock:
# # #             feeders_to_stop = list(live_feeders_g)

# # #         for feeder_instance in feeders_to_stop:
# # #             if hasattr(feeder_instance, 'stop') and callable(feeder_instance.stop):
# # #                 try:
# # #                     app_logger_g.debug(f"Stopping feeder: {feeder_instance}")
# # #                     feeder_instance.stop()
# # #                 except Exception as e_feeder_stop_orch:
# # #                     app_logger_g.error(f"Error stopping a feeder during orchestration shutdown: {e_feeder_stop_orch}", exc_info=True)

# # #         app_logger_g.info("Waiting for live threads to complete (max 15s each)...")
# # #         for t_live in live_threads:
# # #             t_live.join(timeout=15)
# # #             if t_live.is_alive():
# # #                 app_logger_g.warning(f"Thread {t_live.name} did not terminate gracefully after 15s.")

# # #         app_logger_g.info("Generating final consolidated report for live session...")
# # #         generate_consolidated_report(all_trades_g, base_run_dir_g, report_name_prefix="live_session_final")
# # #         app_logger_g.info(f"[END OF SESSION] Live session (ID: {run_id_g}) concluded. Details in: {base_run_dir_g}")
# # #         cprint(f"Session logs and reports saved in: {base_run_dir_g}", "green")

# # # def main_signal_handler(sig, frame):
# # #     if not stop_event_g.is_set():
# # #         app_logger_g.warning(f"Signal {sig} received. Setting global stop event.")
# # #         cprint("\nCtrl+C detected! Initiating graceful shutdown. Please wait...", "orange", attrs=['bold'])
# # #         stop_event_g.set()

# # # if __name__ == "__main__":
# # #     signal.signal(signal.SIGINT, main_signal_handler)
# # #     parser = argparse.ArgumentParser(description="Multi-Timeframe AlgoTrading System (Simulation/Live)")
# # #     parser.add_argument("--mode", choices=["sim", "live"], default="sim", help="Choose mode: sim for simulation, live for live market.")
# # #     parser.add_argument("--index", choices=list(INDEX_SYMBOL_MAP.keys()), default="NIFTY", help="Index to trade/simulate.")
# # #     args = parser.parse_args()

# # #     all_trades_g = []
# # #     stop_event_g.clear()

# # #     print(f"Starting application in '{args.mode}' mode for index '{args.index}'.")
# # #     if args.mode == "live" and LiveDataFeeder is None:
# # #         msg_err = "ERROR: LiveDataFeeder could not be imported. Live mode is disabled."
# # #         if app_logger_g:
# # #             app_logger_g.critical(msg_err)
# # #         else:
# # #             print(msg_err)
# # #         cprint(msg_err, "red", attrs=["bold"])
# # #         cprint("Please ensure 'app.LiveDataFeeder' is correctly implemented and importable.", "red")
# # #     elif args.mode == "live":
# # #         run_live_orchestrator()
# # #     else:
# # #         run_simulation_orchestrator()

# # #     if app_logger_g:
# # #         app_logger_g.info("Application finished.")
# # #     else:
# # #         print("Application finished.")
# import argparse
# import os
# import pandas as pd
# import numpy as np
# import random
# import uuid
# import time
# from datetime import datetime, timedelta, time as dttime
# import pytz
# from termcolor import cprint, colored
# from tabulate import tabulate
# import threading
# import logging
# import signal
# import sys
# from concurrent.futures import ThreadPoolExecutor
# import backoff

# from app.compute_indicators import compute_indicators
# from app.strategies import strategy_factories
# from angel_one.angel_data_fetcher import AngelDataFetcher
# from app.tick_aggregator import TickAggregator
# from config import load_config, default_params

# try:
#     from app.LiveDataFeeder import LiveDataFeeder
# except ImportError as e:
#     print("INFO: Could not import LiveDataFeeder. Live mode will not be available.:", e)
#     LiveDataFeeder = None

# INDEX_SYMBOL_MAP = {
#     "NIFTY": {
#         "hist_symbol": "Nifty 50",
#         "hist_token": "99926000",
#         "tick_symbol": "NIFTY",
#         "tick_token": "26000"
#     },
#     "BANKNIFTY": {
#         "hist_symbol": "Bank Nifty",
#         "hist_token": "99926009",
#         "tick_symbol": "BANKNIFTY",
#         "tick_token": "26009"
#     },
# }

# # TIMEFRAMES = ["1min", "3min", "5min", "15min"]
# # TIMEFRAME_MINUTES = {"1min": 1, "3min": 3, "5min": 5, "15min": 15}

# TIMEFRAMES = ["5min"]
# TIMEFRAME_MINUTES = {"5min": 5}
# all_trades_g = []
# all_trades_lock = threading.Lock()
# stop_event_g = threading.Event()
# live_feeders_g = []
# base_run_dir_g = ""
# run_id_g = ""
# app_logger_g = None
# fetcher_g = None  # Global fetcher instance
# fetcher_lock = threading.Lock()

# def setup_logging(run_id, base_log_dir):
#     global app_logger_g
#     os.makedirs(base_log_dir, exist_ok=True)
#     log_file_app = os.path.join(base_log_dir, f"app_{run_id}.log")
#     logger = logging.getLogger("AlgoTradingApp")
#     logger.setLevel(logging.INFO)
#     if not logger.handlers:
#         fh_app = logging.FileHandler(log_file_app)
#         fh_app.setLevel(logging.INFO)
#         ch_app = logging.StreamHandler(sys.stdout)
#         ch_app.setLevel(logging.INFO)
#         formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
#         fh_app.setFormatter(formatter)
#         ch_app.setFormatter(formatter)
#         logger.addHandler(fh_app)
#         logger.addHandler(ch_app)
#     app_logger_g = logger
#     return logger

# def get_timeframe_tick_logger(timeframe_str, run_id, base_log_dir):
#     os.makedirs(base_log_dir, exist_ok=True)
#     log_file_tf = os.path.join(base_log_dir, f"ticks_{timeframe_str}_{run_id}.log")
#     tf_logger = logging.getLogger(f"TickLogger_{timeframe_str}")
#     tf_logger.setLevel(logging.INFO)
#     if not tf_logger.handlers:
#         fh_tf = logging.FileHandler(log_file_tf)
#         fh_tf.setLevel(logging.INFO)
#         formatter = logging.Formatter('%(asctime)s - %(message)s')
#         fh_tf.setFormatter(formatter)
#         tf_logger.addHandler(fh_tf)
#         tf_logger.propagate = False
#     return tf_logger

# @backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=60, base=2)
# def fetch_historical_with_retry(fetcher, symbol, timeframe, days, exchange):
#     if fetcher is None:
#         raise ValueError("AngelDataFetcher is not initialized")
#     return fetcher.fetch_historical_candles(symbol, timeframe, days=days, exchange=exchange)

# def initialize_fetcher():
#     global fetcher_g
#     with fetcher_lock:
#         if fetcher_g is None:
#             config = load_config()
#             try:
#                 fetcher_g = AngelDataFetcher(config)
#                 app_logger_g.info("Initialized global AngelDataFetcher instance")
#             except Exception as e:
#                 app_logger_g.error(f"Failed to initialize AngelDataFetcher: {e}")
#                 fetcher_g = None
#     return fetcher_g

# def load_and_prepare_data_tf(symbol_hist_name, exchange, timeframe, tf_paths):
#     raw_csv_path = tf_paths["raw_csv"]
#     hist_ind_csv_path = tf_paths["hist_ind_csv"]

#     if not os.path.exists(raw_csv_path):
#         app_logger_g.info(f"[{timeframe}] Fetching historical data for {symbol_hist_name}...")
#         fetcher = initialize_fetcher()
#         if fetcher is None:
#             app_logger_g.error(f"[{timeframe}] AngelDataFetcher initialization failed. Cannot fetch data.")
#             return pd.DataFrame()
#         try:
#             df_raw = fetch_historical_with_retry(fetcher, symbol_hist_name, timeframe, 90, exchange)
#             if df_raw.empty:
#                 app_logger_g.error(f"[{timeframe}] No data fetched for {symbol_hist_name}.")
#                 return pd.DataFrame()
#             df_raw.to_csv(raw_csv_path, index=False)
#         except Exception as e:
#             app_logger_g.error(f"[{timeframe}] Failed to fetch historical data: {e}")
#             return pd.DataFrame()
#     else:
#         app_logger_g.info(f"[{timeframe}] Loading raw data from {raw_csv_path}")
#         df_raw = pd.read_csv(raw_csv_path, parse_dates=['datetime'])

#     if df_raw.empty:
#         return pd.DataFrame()

#     if 'datetime' not in df_raw.columns:
#         if 'date' in df_raw.columns:
#             df_raw.rename(columns={'date': 'datetime'}, inplace=True)
#         elif df_raw.index.name == 'datetime':
#             df_raw.reset_index(inplace=True)
#         else:
#             app_logger_g.error(f"[{timeframe}] 'datetime' column not found in {raw_csv_path}.")
#             return pd.DataFrame()

#     df_raw = ensure_datetime_tz_aware(df_raw.copy(), 'datetime',timeframe, 'Asia/Kolkata')
#     df_raw.set_index('datetime', inplace=True)
#     df_raw.sort_index(inplace=True)
#     df_raw = df_raw[~df_raw.index.duplicated(keep='last')]

#     if not os.path.exists(hist_ind_csv_path):
#         app_logger_g.info(f"[{timeframe}] Computing indicators for historical data and saving to {hist_ind_csv_path}")
#         df_hist_with_ind = compute_indicators(df_raw.copy(), hist_ind_csv_path)
#         if df_hist_with_ind is None or df_hist_with_ind.empty:
#             app_logger_g.error(f"[{timeframe}] Indicator computation failed.")
#             return pd.DataFrame()
#         df_hist_with_ind.to_csv(hist_ind_csv_path)
#     else:
#         app_logger_g.info(f"[{timeframe}] Loading historical indicators from {hist_ind_csv_path}")
#         df_hist_with_ind = pd.read_csv(hist_ind_csv_path, parse_dates=['datetime'])
#         df_hist_with_ind = ensure_datetime_tz_aware(df_hist_with_ind.copy(), 'datetime', 'Asia/Kolkata')
#         df_hist_with_ind.set_index('datetime', inplace=True)
#         df_hist_with_ind.sort_index(inplace=True)
#         df_hist_with_ind = df_hist_with_ind[~df_hist_with_ind.index.duplicated(keep='last')]

#     return df_hist_with_ind

# def ensure_datetime_tz_aware(df, col='datetime',timeframe=None, tz_str='Asia/Kolkata'):
#     target_tz = pytz.timezone(tz_str)
#     if col not in df.columns:
#         if df.index.name == col:
#             df = df.reset_index()
#         else:
#             app_logger_g.error(f"'{col}' column or index not found in DataFrame.")
#             raise ValueError(f"'{col}' column or index not found in DataFrame.")

#     df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.tz_convert(target_tz)
#     if df[col].isnull().any():
#         app_logger_g.warning(f"[{timeframe}] NaNs introduced in '{col}' during conversion.")
#     return df

# def get_timeframe_specific_paths(main_run_dir, timeframe, symbol_name_safe):
#     tf_dir = os.path.join(main_run_dir, timeframe)
#     data_dir = os.path.join(tf_dir, "data")
#     raw_dir = os.path.join(data_dir, "raw")
#     ind_dir = os.path.join(data_dir, "datawithindicators")
#     trades_dir = os.path.join(tf_dir, "trades")

#     for d in [raw_dir, ind_dir, trades_dir]:
#         os.makedirs(d, exist_ok=True)

#     return {
#         "tf_base": tf_dir, "data_dir": data_dir, "raw_dir": raw_dir, "ind_dir": ind_dir,
#         "trades_dir": trades_dir,
#         "raw_csv": os.path.join(raw_dir, f"{symbol_name_safe}_{timeframe}_historical.csv"),
#         "hist_ind_csv": os.path.join(ind_dir, f"{symbol_name_safe}_{timeframe}_historical_with_indicators.csv")
#     }

# def print_trade_action_tf(trade, action_type, timeframe):
#     strat = trade['strategy']
#     side = trade['side']
#     price = trade['entry_price'] if action_type == "OPEN" else trade.get('exit_price', 'N/A')
#     dt = trade['entry_time'] if action_type == "OPEN" else trade.get('exit_time', 'N/A')
#     log_msg = ""
#     console_color = "white"
#     attrs = ['bold']

#     if action_type == "OPEN":
#         arrow = "🟩" if side == 'LONG' else "🟥"
#         console_color = 'green' if side == 'LONG' else 'red'
#         log_msg = (f"[{timeframe}] {arrow} [{strat}] {action_type} {side} at {price:.2f} "
#                    f"(SL={trade.get('sl', 'N/A')}, TP={trade.get('tp', 'N/A')}) @ {dt}")
#     else:
#         pnl = trade.get('pnl', 0)
#         emoji = '✔️' if pnl > 0 else ('❌' if pnl < 0 else '➖')
#         console_color = 'green' if pnl > 0 else ('red' if pnl < 0 else 'white')
#         log_msg = (f"[{timeframe}] 🔴 [{strat}] EXIT {side} at {price:.2f} | PnL: {pnl:.2f} {emoji} "
#                    f"({trade.get('exit_reason', 'N/A')}) @ {dt}")

#     app_logger_g.info(log_msg)
#     cprint(log_msg, console_color, attrs=attrs)

# def save_trade_to_timeframe_strategy_csv(trade_details, trades_dir_tf, strategy_name, timeframe):
#     filename = os.path.join(trades_dir_tf, f"trades_{strategy_name}_{timeframe}.csv")
#     df_trade = pd.DataFrame([trade_details])
#     for col in ['entry_time', 'exit_time']:
#         if col in df_trade.columns and pd.api.types.is_datetime64_any_dtype(df_trade[col]):
#             df_trade[col] = df_trade[col].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
#     header = not os.path.exists(filename)
#     try:
#         df_trade.to_csv(filename, mode='a', header=header, index=False)
#     except Exception as e:
#         app_logger_g.error(f"[{timeframe}] Failed to save trade to {filename}: {e}")

# def generate_consolidated_report(all_trades_list_global, main_run_dir, report_name_prefix="consolidated"):
#     if not all_trades_list_global:
#         msg = "No trades were executed across any timeframe for this session."
#         app_logger_g.info(msg)
#         cprint(f"\n{msg}", "red", attrs=['bold'])
#         return

#     df_all_trades = pd.DataFrame(all_trades_list_global)
#     df_all_trades['pnl'] = pd.to_numeric(df_all_trades['pnl'], errors='coerce').fillna(0)
#     df_all_trades['entry_price'] = pd.to_numeric(df_all_trades['entry_price'], errors='coerce')
#     df_all_trades['exit_price'] = pd.to_numeric(df_all_trades['exit_price'], errors='coerce')

#     for col_name in ['entry_time', 'exit_time']:
#         if col_name in df_all_trades.columns and not pd.api.types.is_string_dtype(df_all_trades[col_name]):
#             if pd.api.types.is_datetime64_any_dtype(df_all_trades[col_name]):
#                 df_all_trades[col_name] = df_all_trades[col_name].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
#             else:
#                 df_all_trades[col_name] = df_all_trades[col_name].astype(str)

#     report_header = f"\n==== CONSOLIDATED TRADE REPORT ({report_name_prefix}) ====\nRun ID: {run_id_g}\n"
#     app_logger_g.info(report_header)
#     cprint(report_header, 'yellow', attrs=['bold', 'underline'])

#     total_trades_overall = len(df_all_trades)
#     total_pnl_overall = df_all_trades['pnl'].sum()
#     wins_overall = (df_all_trades['pnl'] > 0).sum()
#     losses_overall = (df_all_trades['pnl'] <= 0).sum()

#     overall_summary_text = (
#         f"OVERALL SUMMARY:\n"
#         f"Total Trades: {total_trades_overall}\n"
#         f"Profitable Trades: {wins_overall}\n"
#         f"Losing/BE Trades: {losses_overall}\n"
#         f"Total PnL: {total_pnl_overall:.2f}\n"
#     )
#     app_logger_g.info(overall_summary_text)
#     cprint("OVERALL SUMMARY:", 'cyan', attrs=['bold'])
#     cprint(f"Total Trades: {total_trades_overall}", 'cyan')
#     cprint(f"Profitable Trades: {wins_overall}", 'cyan')
#     cprint(f"Losing/BE Trades: {losses_overall}", 'cyan')
#     cprint(f"Total PnL: {total_pnl_overall:.2f}\n", 'magenta', attrs=['bold'])

#     summary_by_tf = df_all_trades.groupby('timeframe')['pnl'].agg(
#         TotalTrades='count',
#         Wins=lambda x: (x > 0).sum(),
#         LossesBE=lambda x: (x <= 0).sum(),
#         TotalPnL='sum'
#     )
#     tf_summary_text = f"SUMMARY BY TIMEFRAME:\n{summary_by_tf.to_string()}\n"
#     app_logger_g.info(tf_summary_text)
#     cprint("SUMMARY BY TIMEFRAME:", 'cyan', attrs=['bold'])
#     print(tabulate(summary_by_tf, headers='keys', tablefmt='fancy_grid', floatfmt=".2f"))
#     print("\n")

#     summary_by_strat = df_all_trades.groupby('strategy')['pnl'].agg(
#         TotalTrades='count',
#         Wins=lambda x: (x > 0).sum(),
#         LossesBE=lambda x: (x <= 0).sum(),
#         TotalPnL='sum'
#     )
#     strat_summary_text = f"SUMMARY BY STRATEGY (Overall):\n{summary_by_strat.to_string()}\n"
#     app_logger_g.info(strat_summary_text)
#     cprint("SUMMARY BY STRATEGY (Overall):", 'cyan', attrs=['bold'])
#     print(tabulate(summary_by_strat, headers='keys', tablefmt='fancy_grid', floatfmt=".2f"))
#     print("\n")

#     cols_to_display = ['timeframe', 'strategy', 'side', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'pnl', 'exit_reason']
#     cols_to_display = [col for col in cols_to_display if col in df_all_trades.columns]

#     detailed_trades_text = f"DETAILED TRADES:\n{df_all_trades[cols_to_display].to_string(index=False)}\n"
#     app_logger_g.info(detailed_trades_text)
#     cprint("DETAILED TRADES:", 'yellow', attrs=['bold'])
#     print(tabulate(df_all_trades[cols_to_display], headers='keys', tablefmt='fancy_grid', showindex=False, floatfmt=".2f"))

#     report_csv_path = os.path.join(main_run_dir, f"{report_name_prefix}_trades_consolidated_report.csv")
#     df_all_trades.to_csv(report_csv_path, index=False)
#     msg_saved = f"Consolidated trade report saved to: {report_csv_path}"
#     app_logger_g.info(msg_saved)
#     cprint(f"\n{msg_saved}", "green")

# def simulate_future_candles_tf(df_raw_hist, sim_date_dt, timeframe, tz, token, tf_tick_logger):
#     if df_raw_hist.empty or 'close' not in df_raw_hist.columns:
#         app_logger_g.warning(f"[{timeframe}] Cannot simulate future candles. Historical data is empty or missing 'close'.")
#         return pd.DataFrame()

#     last_close = int(df_raw_hist['close'].iloc[-1] * 100)
#     minutes = int(timeframe.replace("min", ""))
#     total_market_minutes = int((dttime(15, 30).hour * 60 + dttime(15, 30).minute) -
#                               (dttime(9, 15).hour * 60 + dttime(9, 15).minute))

#     sim_candles_list = simulate_and_aggregate_candles(
#         start_price=last_close,
#         total_minutes=total_market_minutes,
#         interval_minutes=minutes,
#         token=token,
#         tick_logger=tf_tick_logger
#     )

#     market_open_time = tz.localize(datetime.combine(sim_date_dt, dttime(9, 15)))
#     for i, candle_dict in enumerate(sim_candles_list):
#         candle_dict['datetime'] = market_open_time + timedelta(minutes=i * minutes)

#     app_logger_g.info(f"[{timeframe}] Simulated {len(sim_candles_list)} future candles for {sim_date_dt.date()}.")
#     return pd.DataFrame(sim_candles_list)

# def simulate_and_aggregate_candles(start_price, total_minutes, interval_minutes, token, tick_logger=None):
#     aggregator = TickAggregator(interval_minutes=interval_minutes)
#     candles = []
#     current_sim_price = start_price
#     num_candles_to_sim = total_minutes // interval_minutes

#     for _ in range(num_candles_to_sim):
#         ticks_this_interval = simulate_ticks_for_interval(current_sim_price, interval_minutes * 60, num_ticks=60, token=token, tick_logger=tick_logger)
#         for tick in ticks_this_interval:
#             aggregated_candle = aggregator.process_tick(tick)
#             if aggregated_candle:
#                 candles.append(aggregated_candle)
#                 if 'close' in aggregated_candle:
#                     current_sim_price = aggregated_candle['close']
#                 break
#         last_candle = aggregator.force_close()
#         if last_candle:
#             candles.append(last_candle)
#             if 'close' in last_candle:
#                 current_sim_price = last_candle['close']

#     for c in candles:
#         for key_price in ['open', 'high', 'low', 'close']:
#             if key_price in c:
#                 c[key_price] /= 100.0
#     return candles

# def simulate_ticks_for_interval(start_price, interval_seconds, num_ticks, token, tick_logger=None):
#     ticks = []
#     current_price = start_price
#     base_sim_timestamp_ms = int(time.time() * 1000)
#     for i in range(num_ticks):
#         price_change = random.randint(-5, 5)
#         current_price = max(1, current_price + price_change)
#         tick = {
#             'subscription_mode': 1,
#             'exchange_type': 1,
#             'token': token,
#             'sequence_number': i,
#             'exchange_timestamp': base_sim_timestamp_ms + (i * (interval_seconds * 1000 // num_ticks)),
#             'last_traded_price': current_price,
#             'subscription_mode_val': 'LTP'
#         }
#         ticks.append(tick)
#         if tick_logger:
#             tick_logger.info(f"SIM_TICK: {tick}")
#     return ticks

# def merge_and_compute_indicators_sim_tf(df_raw_hist_indexed, df_sim_future, tf_paths, symbol_name_safe, timeframe, sim_date_str):
#     if 'datetime' in df_sim_future.columns:
#         df_sim_future_indexed = df_sim_future.set_index('datetime')
#     else:
#         df_sim_future_indexed = df_sim_future

#     if not df_sim_future_indexed.empty and df_sim_future_indexed.index.tzinfo is None:
#         df_sim_future_indexed = df_sim_future_indexed.tz_localize('Asia/Kolkata', ambiguous='infer', nonexistent='shift_forward')

#     df_full = pd.concat([df_raw_hist_indexed, df_sim_future_indexed], axis=0)
#     df_full.sort_index(inplace=True)
#     df_full = df_full[~df_full.index.duplicated(keep='last')]

#     out_file_path = os.path.join(tf_paths["ind_dir"], f"{symbol_name_safe}_{timeframe}_sim_{sim_date_str}_full_with_indicators.csv")
#     df_all_with_ind = compute_indicators(df_full.copy(), out_file_path)
#     if df_all_with_ind is None or df_all_with_ind.empty:
#         app_logger_g.error(f"[{timeframe}] Indicator computation failed for simulated data.")
#         return pd.DataFrame()
#     df_all_with_ind.to_csv(out_file_path)
#     app_logger_g.info(f"[{timeframe}] Saved SIM merged data with indicators to: {out_file_path}")
#     return df_all_with_ind

# def run_strategies_on_simulated_day_tf(simulated_day_with_ind, full_history_with_ind, strategies_config, timeframe, tf_paths, sim_date_str):
#     trades_for_this_tf_and_strat = {}
#     active_trades_sim = {}

#     if simulated_day_with_ind.empty:
#         app_logger_g.warning(f"[{timeframe}] No simulated day data for {sim_date_str} to run strategies.")
#         return

#     for idx, (dt_index, candle_row) in enumerate(simulated_day_with_ind.iterrows()):
#         current_history_for_signal = full_history_with_ind.loc[:dt_index]

#         for strat_name, strat_params in strategies_config.items():
#             if strat_name not in strategy_factories:
#                 app_logger_g.error(f"[{timeframe}] Strategy {strat_name} not found!")
#                 continue

#             strat_func = strategy_factories[strat_name]
#             out = strat_func(candle_row, current_history_for_signal, strat_params)
#             active_trade = active_trades_sim.get(strat_name)

#             if out['signal'] in ('buy_potential', 'sell_potential'):
#                 if not active_trade or active_trade['status'] == 'closed':
#                     trade = {
#                         'run_id': run_id_g,
#                         'timeframe': timeframe,
#                         'strategy': strat_name,
#                         'entry_time': dt_index,
#                         'entry_price': candle_row['close'],
#                         'side': 'LONG' if out['signal'] == 'buy_potential' else 'SHORT',
#                         'sl': out.get('sl'),
#                         'tp': out.get('tp'),
#                         'status': 'open',
#                         'exit_time': pd.NaT,
#                         'exit_price': np.nan,
#                         'pnl': np.nan,
#                         'exit_reason': None
#                     }
#                     active_trades_sim[strat_name] = trade
#                     print_trade_action_tf(trade, "OPEN", timeframe)

#             elif active_trade and active_trade['status'] == 'open':
#                 exit_trade_flag = False
#                 exit_price_val = np.nan
#                 exit_reason_str = ""

#                 sl = active_trade.get('sl')
#                 tp = active_trade.get('tp')

#                 if sl is not None and pd.notna(sl):
#                     if (active_trade['side'] == 'LONG' and candle_row['low'] <= sl) or \
#                        (active_trade['side'] == 'SHORT' and candle_row['high'] >= sl):
#                         exit_trade_flag = True
#                         exit_price_val = sl
#                         exit_reason_str = "Stop Loss"

#                 if not exit_trade_flag and tp is not None and pd.notna(tp):
#                     if (active_trade['side'] == 'LONG' and candle_row['high'] >= tp) or \
#                        (active_trade['side'] == 'SHORT' and candle_row['low'] <= tp):
#                         exit_trade_flag = True
#                         exit_price_val = tp
#                         exit_reason_str = "Target Profit"

#                 if not exit_trade_flag and idx == len(simulated_day_with_ind) - 1:
#                     exit_trade_flag = True
#                     exit_price_val = candle_row['close']
#                     exit_reason_str = f"EOD_SIM_{sim_date_str}"

#                 if exit_trade_flag:
#                     pnl_val = (exit_price_val - active_trade['entry_price']) if active_trade['side'] == 'LONG' else (active_trade['entry_price'] - exit_price_val)
#                     active_trade.update({
#                         'exit_time': dt_index,
#                         'exit_price': exit_price_val,
#                         'pnl': pnl_val,
#                         'status': 'closed',
#                         'exit_reason': exit_reason_str
#                     })
#                     print_trade_action_tf(active_trade, "EXIT", timeframe)
#                     if strat_name not in trades_for_this_tf_and_strat:
#                         trades_for_this_tf_and_strat[strat_name] = []
#                     trades_for_this_tf_and_strat[strat_name].append(active_trade.copy())
#                     with all_trades_lock:
#                         all_trades_g.append(active_trade.copy())
#                     active_trades_sim[strat_name] = {'status': 'closed'}

#     for strat_name, trade_to_close in active_trades_sim.items():
#         if trade_to_close and trade_to_close.get('status') == 'open':
#             final_candle = simulated_day_with_ind.iloc[-1]
#             exit_price_eod = final_candle['close']
#             pnl_eod = (exit_price_eod - trade_to_close['entry_price']) if trade_to_close['side'] == 'LONG' else (trade_to_close['entry_price'] - exit_price_eod)
#             trade_to_close.update({
#                 'exit_time': simulated_day_with_ind.index[-1],
#                 'exit_price': exit_price_eod,
#                 'pnl': pnl_eod,
#                 'status': 'closed',
#                 'exit_reason': f"EOD_SIM_FORCE_CLOSE_{sim_date_str}"
#             })
#             print_trade_action_tf(trade_to_close, "EXIT", timeframe)
#             if strat_name not in trades_for_this_tf_and_strat:
#                 trades_for_this_tf_and_strat[strat_name] = []
#             trades_for_this_tf_and_strat[strat_name].append(trade_to_close.copy())
#             with all_trades_lock:
#                 all_trades_g.append(trade_to_close.copy())

#     for strat_name_key, strat_trades_list in trades_for_this_tf_and_strat.items():
#         if strat_trades_list:
#             save_trade_to_timeframe_strategy_csv(strat_trades_list, tf_paths["trades_dir"], strat_name_key, timeframe)
#     app_logger_g.info(f"[{timeframe}] Finished strategy run for {sim_date_str}. All trades for this TF saved.")

# def run_single_simulation_task(timeframe, index_key, exchange, main_run_dir_sim, sim_date_dt, tz, strategies_config, tick_log_dir_sim):
#     app_logger_g.info(f"SIM_TASK [{timeframe}] Starting simulation for date: {sim_date_dt.date()}")
#     symbol_config = INDEX_SYMBOL_MAP[index_key]
#     symbol_hist_name = symbol_config["hist_symbol"]
#     symbol_name_safe = symbol_hist_name.replace(" ", "_")
#     tf_paths = get_timeframe_specific_paths(main_run_dir_sim, timeframe, symbol_name_safe)
#     tf_tick_logger_sim = get_timeframe_tick_logger(timeframe, run_id_g + "_sim", tick_log_dir_sim)

#     df_hist_ind = load_and_prepare_data_tf(symbol_hist_name, exchange, timeframe, tf_paths)
#     if df_hist_ind.empty:
#         app_logger_g.error(f"SIM_TASK [{timeframe}] Failed to load historical data. Aborting.")
#         return timeframe

#     df_sim_future = simulate_future_candles_tf(df_hist_ind.iloc[[-1]].reset_index(), sim_date_dt, timeframe, tz, symbol_config["hist_token"], tf_tick_logger_sim)
#     df_all_data_with_ind = merge_and_compute_indicators_sim_tf(df_hist_ind.copy(), df_sim_future.copy(), tf_paths, symbol_name_safe, timeframe, sim_date_dt.strftime("%Y%m%d"))
#     if df_all_data_with_ind.empty:
#         app_logger_g.error(f"SIM_TASK [{timeframe}] Failed after merging/indicators. Aborting.")
#         return timeframe

#     simulated_day_data = df_all_data_with_ind.loc[df_all_data_with_ind.index.date == sim_date_dt.date()].copy()
#     if simulated_day_data.empty:
#         app_logger_g.warning(f"SIM_TASK [{timeframe}] No data found for simulated date {sim_date_dt.date()}. Using latest available.")
#         target_date_to_run = df_all_data_with_ind.index.max().date()
#         simulated_day_data = df_all_data_with_ind.loc[df_all_data_with_ind.index.date == target_date_to_run].copy()
#         if simulated_day_data.empty:
#             app_logger_g.error(f"SIM_TASK [{timeframe}] Still no data to run strategies. Aborting.")
#             return timeframe

#     run_strategies_on_simulated_day_tf(simulated_day_data, df_all_data_with_ind, strategies_config, timeframe, tf_paths, sim_date_dt.strftime("%Y%m%d"))
#     app_logger_g.info(f"SIM_TASK [{timeframe}] Simulation task finished.")
#     return timeframe

# def run_simulation_orchestrator():
#     global base_run_dir_g, run_id_g, fetcher_g
#     run_id_g = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:4]
#     base_run_dir_g = os.path.join("simulation_runs", run_id_g)
#     log_dir = os.path.join(base_run_dir_g, "logs")
#     setup_logging(run_id_g, log_dir)
#     app_logger_g.info(f"========== MULTI-TIMEFRAME SIMULATION MODE (Run ID: {run_id_g}) ==========")

#     index_key = "NIFTY"
#     exchange = "NSE"
#     timeframes_to_sim = TIMEFRAMES
#     sim_date_dt = datetime(2025, 5, 28)
#     tz = pytz.timezone("Asia/Kolkata")
#     strategies_config = {name: default_params.get(name, {}) for name in strategy_factories.keys()}
#     all_trades_g.clear()
#     fetcher_g = None  # Reset fetcher

#     with ThreadPoolExecutor(max_workers=len(timeframes_to_sim), thread_name_prefix="SimWorker") as executor:
#         futures = []
#         for tf in timeframes_to_sim:
#             future = executor.submit(run_single_simulation_task, tf, index_key, exchange, base_run_dir_g, sim_date_dt, tz, strategies_config, log_dir)
#             futures.append(future)
#             time.sleep(2)  # Stagger thread starts
#         for future in futures:
#             try:
#                 completed_tf = future.result()
#                 app_logger_g.info(f"Simulation future for timeframe {completed_tf} completed.")
#             except Exception as exc:
#                 app_logger_g.error(f"A simulation task generated an exception: {exc}", exc_info=True)

#     app_logger_g.info("All simulation tasks submitted and processed.")
#     generate_consolidated_report(all_trades_g, base_run_dir_g, report_name_prefix=f"sim_{sim_date_dt.strftime('%Y%m%d')}")
#     app_logger_g.info(f"[DONE] All simulations complete for {sim_date_dt.strftime('%d-%b-%Y')}. Details in: {base_run_dir_g}")

# def live_timeframe_processor_thread_worker(timeframe, index_key, exchange, main_run_dir_live, strategies_config, tz, tick_log_dir_live):
#     if LiveDataFeeder is None:
#         app_logger_g.error(f"[{timeframe}] LiveDataFeeder not available. Cannot start live processor.")
#         return

#     thread_name = threading.current_thread().name
#     app_logger_g.info(f"[{timeframe}/{thread_name}] Initializing live processor...")
#     interval_minutes = int(timeframe.replace("min", ""))
#     symbol_config = INDEX_SYMBOL_MAP[index_key]
#     symbol_hist_name = symbol_config["hist_symbol"]
#     symbol_name_safe = symbol_hist_name.replace(" ", "_")
#     tick_token_live = symbol_config["tick_token"]
#     live_feed_tokens = [{"exchangeType": 1, "tokens": [tick_token_live]}]
#     tf_paths = get_timeframe_specific_paths(main_run_dir_live, timeframe, symbol_name_safe)
#     tf_tick_logger = get_timeframe_tick_logger(timeframe, run_id_g + "_live", tick_log_dir_live)

#     current_history_with_ind = load_and_prepare_data_tf(symbol_hist_name, exchange, timeframe, tf_paths)
#     if current_history_with_ind.empty:
#         app_logger_g.error(f"[{timeframe}/{thread_name}] Failed to load initial historical data. Live processor stopping.")
#         return
#     app_logger_g.info(f"[{timeframe}/{thread_name}] Initial historical data loaded. Last record: {current_history_with_ind.index[-1]}")

#     active_trades_live_tf = {}

#     def handle_new_live_candle(candle_dict):
#         nonlocal current_history_with_ind, active_trades_live_tf
#         try:
#             dt_object = candle_dict.get('datetime')
#             if not dt_object or not isinstance(dt_object, datetime):
#                 app_logger_g.error(f"[{timeframe}] Invalid or missing datetime in candle: {candle_dict}")
#                 return

#             if dt_object.tzinfo is None:
#                 dt_object = tz.localize(dt_object)
#             else:
#                 dt_object = dt_object.astimezone(tz)
#             candle_dict['datetime'] = dt_object

#             cprint(f"[{timeframe}] Candle: {dt_object.strftime('%H:%M:%S')} O:{candle_dict['open']:.2f} H:{candle_dict['high']:.2f} L:{candle_dict['low']:.2f} C:{candle_dict['close']:.2f} V:{candle_dict.get('volume',0)}", "blue")

#             new_candle_df = pd.DataFrame([candle_dict])
#             new_candle_df.set_index('datetime', inplace=True)
#             temp_history = pd.concat([current_history_with_ind, new_candle_df], axis=0)
#             temp_history.sort_index(inplace=True)
#             temp_history = temp_history[~temp_history.index.duplicated(keep='last')]

#             out_file_path = os.path.join(tf_paths["ind_dir"], f"{symbol_name_safe}_{timeframe}_live_temp.csv")
#             updated_full_history_with_ind = compute_indicators(temp_history.copy(), out_file_path)
#             if updated_full_history_with_ind is None or updated_full_history_with_ind.empty:
#                 app_logger_g.error(f"[{timeframe}] Indicator computation failed for new candle. Skipping strategies.")
#                 return
#             current_history_with_ind = updated_full_history_with_ind

#             current_candle_data_row = current_history_with_ind.loc[dt_object]
#             history_for_strategies = current_history_with_ind.loc[:dt_object]

#             for strat_name, strat_params in strategies_config.items():
#                 if stop_event_g.is_set():
#                     break
#                 if strat_name not in strategy_factories:
#                     continue
#                 strat_func = strategy_factories[strat_name]
#                 out = strat_func(current_candle_data_row, history_for_strategies, strat_params)
#                 active_trade = active_trades_live_tf.get(strat_name)

#                 if out['signal'] in ('buy_potential', 'sell_potential'):
#                     if not active_trade or active_trade['status'] == 'closed':
#                         trade = {
#                             'run_id': run_id_g,
#                             'timeframe': timeframe,
#                             'strategy': strat_name,
#                             'entry_time': dt_object,
#                             'entry_price': current_candle_data_row['close'],
#                             'side': 'LONG' if out['signal'] == 'buy_potential' else 'SHORT',
#                             'sl': out.get('sl'),
#                             'tp': out.get('tp'),
#                             'status': 'open',
#                             'exit_time': pd.NaT,
#                             'exit_price': np.nan,
#                             'pnl': np.nan,
#                             'exit_reason': None
#                         }
#                         active_trades_live_tf[strat_name] = trade
#                         print_trade_action_tf(trade, "OPEN", timeframe)

#                 elif active_trade and active_trade['status'] == 'open':
#                     exit_trade_flag = False
#                     exit_price_val = np.nan
#                     exit_reason_str = ""
#                     sl = active_trade.get('sl')
#                     tp = active_trade.get('tp')

#                     if sl and pd.notna(sl):
#                         if (active_trade['side'] == 'LONG' and current_candle_data_row['low'] <= sl) or \
#                            (active_trade['side'] == 'SHORT' and current_candle_data_row['high'] >= sl):
#                             exit_trade_flag = True
#                             exit_price_val = sl
#                             exit_reason_str = "Stop Loss"

#                     if not exit_trade_flag and tp and pd.notna(tp):
#                         if (active_trade['side'] == 'LONG' and current_candle_data_row['high'] >= tp) or \
#                            (active_trade['side'] == 'SHORT' and current_candle_data_row['low'] <= tp):
#                             exit_trade_flag = True
#                             exit_price_val = tp
#                             exit_reason_str = "Target Profit"

#                     market_official_close_dt = dttime(15, 30)
#                     if not exit_trade_flag and (dt_object + timedelta(minutes=interval_minutes)).time() >= market_official_close_dt:
#                         exit_trade_flag = True
#                         exit_price_val = current_candle_data_row['close']
#                         exit_reason_str = "Market End Closure"

#                     if exit_trade_flag:
#                         pnl_val = (exit_price_val - active_trade['entry_price']) if active_trade['side'] == 'LONG' else (active_trade['entry_price'] - exit_price_val)
#                         active_trade.update({
#                             'exit_time': dt_object,
#                             'exit_price': exit_price_val,
#                             'pnl': pnl_val,
#                             'status': 'closed',
#                             'exit_reason': exit_reason_str
#                         })
#                         print_trade_action_tf(active_trade, "EXIT", timeframe)
#                         save_trade_to_timeframe_strategy_csv(active_trade.copy(), tf_paths["trades_dir"], strat_name, timeframe)
#                         with all_trades_lock:
#                             all_trades_g.append(active_trade.copy())
#                         active_trades_live_tf[strat_name] = {'status': 'closed'}

#         except Exception as e_candle:
#             app_logger_g.error(f"[{timeframe}] CRITICAL ERROR in handle_new_live_candle: {e_candle}", exc_info=True)

#     def handle_new_live_tick(tick_data):
#         if tf_tick_logger:
#             tf_tick_logger.info(f"LIVE_TICK: {tick_data}")

#     feeder_instance = None
#     try:
#         feeder_instance = LiveDataFeeder(
#         tokens=live_feed_tokens,
#         interval_minutes=interval_minutes,
#         candle_callback=handle_new_live_candle,
#         tick_callback=handle_new_live_tick,
#         mode=1,
#         correlation_id=run_id_g
#     )

#         with all_trades_lock:
#             live_feeders_g.append(feeder_instance)
#         feeder_instance.start()
#         app_logger_g.info(f"[{timeframe}/{thread_name}] LiveDataFeeder started.")

#         while not stop_event_g.is_set():
#             if hasattr(feeder_instance, 'is_alive') and not feeder_instance.is_alive():
#                 app_logger_g.warning(f"[{timeframe}/{thread_name}] Feeder is no longer alive. Worker stopping.")
#                 break
#             now_live_tf = datetime.now(tz)
#             if now_live_tf.time() >= dttime(15, 30, tzinfo=tz):
#                 app_logger_g.info(f"[{timeframe}/{thread_name}] Market hours ended. Requesting feeder stop.")
#                 break
#             time.sleep(2)

#     except Exception as e_feeder_setup:
#         app_logger_g.error(f"[{timeframe}/{thread_name}] Failed to initialize or run LiveDataFeeder: {e_feeder_setup}", exc_info=True)
#     finally:
#         app_logger_g.info(f"[{timeframe}/{thread_name}] Live processor thread worker stopping...")
#         if feeder_instance and hasattr(feeder_instance, 'stop'):
#             try:
#                 app_logger_g.info(f"[{timeframe}/{thread_name}] Calling feeder.stop().")
#                 feeder_instance.stop()
#             except Exception as e_feeder_stop_final:
#                 app_logger_g.error(f"[{timeframe}/{thread_name}] Error stopping feeder: {e_feeder_stop_final}", exc_info=True)

# def run_live_orchestrator():
#     global base_run_dir_g, run_id_g, live_feeders_g, fetcher_g
#     run_id_g = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:4]
#     base_run_dir_g = os.path.join("live_session_runs", run_id_g)
#     log_dir = os.path.join(base_run_dir_g, "logs")
#     setup_logging(run_id_g, log_dir)
#     app_logger_g.info(f"========== MULTI-TIMEFRAME LIVE MODE (Run ID: {run_id_g}) ==========")

#     index_key = "NIFTY"
#     exchange = "NSE"
#     timeframes_to_run = TIMEFRAMES
#     tz = pytz.timezone("Asia/Kolkata")
#     strategies_config = {name: default_params.get(name, {}) for name in strategy_factories.keys()}
#     all_trades_g.clear()
#     live_feeders_g.clear()
#     stop_event_g.clear()
#     fetcher_g = None  # Reset fetcher

#     live_threads = []
#     for tf_str in timeframes_to_run:
#         thread = threading.Thread(
#             target=live_timeframe_processor_thread_worker,
#             args=(tf_str, index_key, exchange, base_run_dir_g, strategies_config, tz, log_dir),
#             name=f"LiveWorker-{tf_str}"
#         )
#         live_threads.append(thread)
#         thread.daemon = True
#         thread.start()
#         time.sleep(2)  # Stagger thread starts

#     app_logger_g.info("All live timeframe processor threads initiated.")
#     cprint("Live session started. Press Ctrl+C to stop and generate report.", "magenta")

#     try:
#         while not stop_event_g.is_set():
#             now = datetime.now(tz)
#             if now.time() >= dttime(15, 32, tzinfo=tz):
#                 app_logger_g.info("Market hours officially over. Initiating EOD shutdown.")
#                 stop_event_g.set()
#                 break
#             if not any(t.is_alive() for t in live_threads) and live_threads:
#                 app_logger_g.warning("All live worker threads seem to have terminated prematurely. Shutting down.")
#                 stop_event_g.set()
#                 break
#             time.sleep(5)

#     except KeyboardInterrupt:
#         app_logger_g.info("Ctrl+C pressed by user. Orchestrator initiating shutdown.")
#     finally:
#         app_logger_g.info("Orchestrator shutdown sequence started...")
#         if not stop_event_g.is_set():
#             stop_event_g.set()

#         app_logger_g.info("Requesting all LiveDataFeeders to stop...")
#         feeders_to_stop = []
#         with all_trades_lock:
#             feeders_to_stop = list(live_feeders_g)

#         for feeder_instance in feeders_to_stop:
#             if hasattr(feeder_instance, 'stop') and callable(feeder_instance.stop):
#                 try:
#                     app_logger_g.debug(f"Stopping feeder: {feeder_instance}")
#                     feeder_instance.stop()
#                 except Exception as e_feeder_stop_orch:
#                     app_logger_g.error(f"Error stopping a feeder during orchestration shutdown: {e_feeder_stop_orch}", exc_info=True)

#         app_logger_g.info("Waiting for live threads to complete (max 15s each)...")
#         for t_live in live_threads:
#             t_live.join(timeout=15)
#             if t_live.is_alive():
#                 app_logger_g.warning(f"Thread {t_live.name} did not terminate gracefully after 15s.")

#         app_logger_g.info("Generating final consolidated report for live session...")
#         generate_consolidated_report(all_trades_g, base_run_dir_g, report_name_prefix="live_session_final")
#         app_logger_g.info(f"[END OF SESSION] Live session (ID: {run_id_g}) concluded. Details in: {base_run_dir_g}")
#         cprint(f"Session logs and reports saved in: {base_run_dir_g}", "green")

# def main_signal_handler(sig, frame):
#     if not stop_event_g.is_set():
#         app_logger_g.warning(f"Signal {sig} received. Setting global stop event.")
#         #cprint("\nCtrl+C detected! Initiating graceful shutdown. Please wait...", "orange", attrs=['bold'])
#         cprint("\nCtrl+C detected! Initiating graceful shutdown. Please wait...", "yellow", attrs=['bold'])

#         stop_event_g.set()

# if __name__ == "__main__":
#     signal.signal(signal.SIGINT, main_signal_handler)
#     parser = argparse.ArgumentParser(description="Multi-Timeframe AlgoTrading System (Simulation/Live)")
#     parser.add_argument("--mode", choices=["sim", "live"], default="sim", help="Choose mode: sim for simulation, live for live market.")
#     parser.add_argument("--index", choices=list(INDEX_SYMBOL_MAP.keys()), default="NIFTY", help="Index to trade/simulate.")
#     args = parser.parse_args()

#     all_trades_g = []
#     stop_event_g.clear()

#     print(f"Starting application in '{args.mode}' mode for index '{args.index}'.")
#     if args.mode == "live" and LiveDataFeeder is None:
#         msg_err = "ERROR: LiveDataFeeder could not be imported. Live mode is disabled."
#         if app_logger_g:
#             app_logger_g.critical(msg_err)
#         else:
#             print(msg_err)
#         cprint(msg_err, "red", attrs=["bold"])
#         cprint("Please ensure 'app.LiveDataFeeder' is correctly implemented and importable.", "red")
#     elif args.mode == "live":
#         run_live_orchestrator()
#     else:
#         run_simulation_orchestrator()

#     if app_logger_g:
#         app_logger_g.info("Application finished.")
#     else:
#         print("Application finished.")


import argparse
import os
import pandas as pd
import numpy as np
import random
import uuid
import time
from datetime import datetime, timedelta, time as dttime
import pytz
from termcolor import cprint, colored
from tabulate import tabulate
import threading
import logging
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import backoff

# These imports are assumed to be correct in your project structure
from app.compute_indicators import compute_indicators
from app.strategies import strategy_factories
from angel_one.angel_data_fetcher import AngelDataFetcher
from app.tick_aggregator import TickAggregator
from config import load_config, default_params

try:
    from app.LiveDataFeeder import LiveDataFeeder
except ImportError as e:
    print("INFO: Could not import LiveDataFeeder. Live mode will not be available.:", e)
    LiveDataFeeder = None

INDEX_SYMBOL_MAP = {
    "NIFTY": {
        "hist_symbol": "Nifty 50",
        "hist_token": "99926000",
        "tick_symbol": "NIFTY",
        "tick_token": "26000"
    },
    "BANKNIFTY": {
        "hist_symbol": "Bank Nifty",
        "hist_token": "99926009",
        "tick_symbol": "BANKNIFTY",
        "tick_token": "26009"
    },
}

TIMEFRAMES = ["5min"]
TIMEFRAME_MINUTES = {"5min": 5}
all_trades_g = []
all_trades_lock = threading.Lock()
stop_event_g = threading.Event()
live_feeders_g = []
base_run_dir_g = ""
run_id_g = ""
app_logger_g = None
fetcher_g = None  # Global fetcher instance
fetcher_lock = threading.Lock()

def setup_logging(run_id, base_log_dir):
    global app_logger_g
    os.makedirs(base_log_dir, exist_ok=True)
    log_file_app = os.path.join(base_log_dir, f"app_{run_id}.log")
    logger = logging.getLogger("AlgoTradingApp")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh_app = logging.FileHandler(log_file_app)
        fh_app.setLevel(logging.INFO)
        ch_app = logging.StreamHandler(sys.stdout)
        ch_app.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')
        fh_app.setFormatter(formatter)
        ch_app.setFormatter(formatter)
        logger.addHandler(fh_app)
        logger.addHandler(ch_app)
    app_logger_g = logger
    return logger

def get_timeframe_tick_logger(timeframe_str, run_id, base_log_dir):
    os.makedirs(base_log_dir, exist_ok=True)
    log_file_tf = os.path.join(base_log_dir, f"ticks_{timeframe_str}_{run_id}.log")
    tf_logger = logging.getLogger(f"TickLogger_{timeframe_str}")
    tf_logger.setLevel(logging.INFO)
    if not tf_logger.handlers:
        fh_tf = logging.FileHandler(log_file_tf)
        fh_tf.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh_tf.setFormatter(formatter)
        tf_logger.addHandler(fh_tf)
        tf_logger.propagate = False
    return tf_logger

@backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=60, base=2)
def fetch_historical_with_retry(fetcher, symbol, timeframe, days, exchange):
    if fetcher is None:
        raise ValueError("AngelDataFetcher is not initialized")
    return fetcher.fetch_historical_candles(symbol, timeframe, days=days, exchange=exchange)

def initialize_fetcher():
    global fetcher_g
    with fetcher_lock:
        if fetcher_g is None:
            config = load_config()
            try:
                fetcher_g = AngelDataFetcher(config)
                app_logger_g.info("Initialized global AngelDataFetcher instance")
            except Exception as e:
                app_logger_g.error(f"Failed to initialize AngelDataFetcher: {e}")
                fetcher_g = None
    return fetcher_g

def load_and_prepare_data_tf(symbol_hist_name, exchange, timeframe, tf_paths):
    raw_csv_path = tf_paths["raw_csv"]
    hist_ind_csv_path = tf_paths["hist_ind_csv"]

    if not os.path.exists(raw_csv_path):
        app_logger_g.info(f"[{timeframe}] Fetching historical data for {symbol_hist_name}...")
        fetcher = initialize_fetcher()
        if fetcher is None:
            app_logger_g.error(f"[{timeframe}] AngelDataFetcher initialization failed. Cannot fetch data.")
            return pd.DataFrame()
        try:
            df_raw = fetch_historical_with_retry(fetcher, symbol_hist_name, timeframe, 90, exchange)
            if df_raw.empty:
                app_logger_g.error(f"[{timeframe}] No data fetched for {symbol_hist_name}.")
                return pd.DataFrame()
            df_raw.to_csv(raw_csv_path, index=False)
        except Exception as e:
            app_logger_g.error(f"[{timeframe}] Failed to fetch historical data: {e}")
            return pd.DataFrame()
    else:
        app_logger_g.info(f"[{timeframe}] Loading raw data from {raw_csv_path}")
        df_raw = pd.read_csv(raw_csv_path, parse_dates=['datetime'])

    if df_raw.empty:
        return pd.DataFrame()

    if 'datetime' not in df_raw.columns:
        if 'date' in df_raw.columns:
            df_raw.rename(columns={'date': 'datetime'}, inplace=True)
        elif df_raw.index.name == 'datetime':
            df_raw.reset_index(inplace=True)
        else:
            app_logger_g.error(f"[{timeframe}] 'datetime' column not found in {raw_csv_path}.")
            return pd.DataFrame()

    df_raw = ensure_datetime_tz_aware(df_raw.copy(), 'datetime',timeframe, 'Asia/Kolkata')
    df_raw.set_index('datetime', inplace=True)
    df_raw.sort_index(inplace=True)
    df_raw = df_raw[~df_raw.index.duplicated(keep='last')]

    if not os.path.exists(hist_ind_csv_path):
        app_logger_g.info(f"[{timeframe}] Computing indicators for historical data and saving to {hist_ind_csv_path}")
        df_hist_with_ind = compute_indicators(df_raw.copy(), hist_ind_csv_path)
        if df_hist_with_ind is None or df_hist_with_ind.empty:
            app_logger_g.error(f"[{timeframe}] Indicator computation failed.")
            return pd.DataFrame()
        df_hist_with_ind.to_csv(hist_ind_csv_path)
    else:
        app_logger_g.info(f"[{timeframe}] Loading historical indicators from {hist_ind_csv_path}")
        df_hist_with_ind = pd.read_csv(hist_ind_csv_path, parse_dates=['datetime'])
        df_hist_with_ind = ensure_datetime_tz_aware(df_hist_with_ind.copy(), 'datetime', 'Asia/Kolkata')
        df_hist_with_ind.set_index('datetime', inplace=True)
        df_hist_with_ind.sort_index(inplace=True)
        df_hist_with_ind = df_hist_with_ind[~df_hist_with_ind.index.duplicated(keep='last')]

    return df_hist_with_ind

def ensure_datetime_tz_aware(df, col='datetime',timeframe=None, tz_str='Asia/Kolkata'):
    target_tz = pytz.timezone(tz_str)
    if col not in df.columns:
        if df.index.name == col:
            df = df.reset_index()
        else:
            app_logger_g.error(f"'{col}' column or index not found in DataFrame.")
            raise ValueError(f"'{col}' column or index not found in DataFrame.")

    df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.tz_convert(target_tz)
    if df[col].isnull().any():
        app_logger_g.warning(f"[{timeframe}] NaNs introduced in '{col}' during conversion.")
    return df

def get_timeframe_specific_paths(main_run_dir, timeframe, symbol_name_safe):
    tf_dir = os.path.join(main_run_dir, timeframe)
    data_dir = os.path.join(tf_dir, "data")
    raw_dir = os.path.join(data_dir, "raw")
    ind_dir = os.path.join(data_dir, "datawithindicators")
    trades_dir = os.path.join(tf_dir, "trades")

    for d in [raw_dir, ind_dir, trades_dir]:
        os.makedirs(d, exist_ok=True)

    return {
        "tf_base": tf_dir, "data_dir": data_dir, "raw_dir": raw_dir, "ind_dir": ind_dir,
        "trades_dir": trades_dir,
        "raw_csv": os.path.join(raw_dir, f"{symbol_name_safe}_{timeframe}_historical.csv"),
        "hist_ind_csv": os.path.join(ind_dir, f"{symbol_name_safe}_{timeframe}_historical_with_indicators.csv")
    }

def print_trade_action_tf(trade, action_type, timeframe):
    strat = trade['strategy']
    side = trade['side']
    price = trade['entry_price'] if action_type == "OPEN" else trade.get('exit_price', 'N/A')
    dt = trade['entry_time'] if action_type == "OPEN" else trade.get('exit_time', 'N/A')
    log_msg = ""
    console_color = "white"
    attrs = ['bold']

    if action_type == "OPEN":
        arrow = "🟩" if side == 'LONG' else "🟥"
        console_color = 'green' if side == 'LONG' else 'red'
        log_msg = (f"[{timeframe}] {arrow} [{strat}] {action_type} {side} at {price:.2f} "
                   f"(SL={trade.get('sl', 'N/A')}, TP={trade.get('tp', 'N/A')}) @ {dt}")
    else:
        pnl = trade.get('pnl', 0)
        emoji = '✔️' if pnl > 0 else ('❌' if pnl < 0 else '➖')
        console_color = 'green' if pnl > 0 else ('red' if pnl < 0 else 'white')
        log_msg = (f"[{timeframe}] 🔴 [{strat}] EXIT {side} at {price:.2f} | PnL: {pnl:.2f} {emoji} "
                   f"({trade.get('exit_reason', 'N/A')}) @ {dt}")

    app_logger_g.info(log_msg)
    cprint(log_msg, console_color, attrs=attrs)

def save_trade_to_timeframe_strategy_csv(trade_details, trades_dir_tf, strategy_name, timeframe):
    filename = os.path.join(trades_dir_tf, f"trades_{strategy_name}_{timeframe}.csv")
    df_trade = pd.DataFrame([trade_details])
    for col in ['entry_time', 'exit_time']:
        if col in df_trade.columns and pd.api.types.is_datetime64_any_dtype(df_trade[col]):
            df_trade[col] = df_trade[col].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
    header = not os.path.exists(filename)
    try:
        df_trade.to_csv(filename, mode='a', header=header, index=False)
    except Exception as e:
        app_logger_g.error(f"[{timeframe}] Failed to save trade to {filename}: {e}")

def generate_consolidated_report(all_trades_list_global, main_run_dir, report_name_prefix="consolidated"):
    if not all_trades_list_global:
        msg = "No trades were executed across any timeframe for this session."
        app_logger_g.info(msg)
        cprint(f"\n{msg}", "red", attrs=['bold'])
        return

    df_all_trades = pd.DataFrame(all_trades_list_global)
    df_all_trades['pnl'] = pd.to_numeric(df_all_trades['pnl'], errors='coerce').fillna(0)
    df_all_trades['entry_price'] = pd.to_numeric(df_all_trades['entry_price'], errors='coerce')
    df_all_trades['exit_price'] = pd.to_numeric(df_all_trades['exit_price'], errors='coerce')

    for col_name in ['entry_time', 'exit_time']:
        if col_name in df_all_trades.columns and not pd.api.types.is_string_dtype(df_all_trades[col_name]):
            if pd.api.types.is_datetime64_any_dtype(df_all_trades[col_name]):
                df_all_trades[col_name] = df_all_trades[col_name].dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            else:
                df_all_trades[col_name] = df_all_trades[col_name].astype(str)

    report_header = f"\n==== CONSOLIDATED TRADE REPORT ({report_name_prefix}) ====\nRun ID: {run_id_g}\n"
    app_logger_g.info(report_header)
    cprint(report_header, 'yellow', attrs=['bold', 'underline'])

    total_trades_overall = len(df_all_trades)
    total_pnl_overall = df_all_trades['pnl'].sum()
    wins_overall = (df_all_trades['pnl'] > 0).sum()
    losses_overall = (df_all_trades['pnl'] <= 0).sum()

    overall_summary_text = (
        f"OVERALL SUMMARY:\n"
        f"Total Trades: {total_trades_overall}\n"
        f"Profitable Trades: {wins_overall}\n"
        f"Losing/BE Trades: {losses_overall}\n"
        f"Total PnL: {total_pnl_overall:.2f}\n"
    )
    app_logger_g.info(overall_summary_text)
    cprint("OVERALL SUMMARY:", 'cyan', attrs=['bold'])
    cprint(f"Total Trades: {total_trades_overall}", 'cyan')
    cprint(f"Profitable Trades: {wins_overall}", 'cyan')
    cprint(f"Losing/BE Trades: {losses_overall}", 'cyan')
    cprint(f"Total PnL: {total_pnl_overall:.2f}\n", 'magenta', attrs=['bold'])

    summary_by_tf = df_all_trades.groupby('timeframe')['pnl'].agg(
        TotalTrades='count',
        Wins=lambda x: (x > 0).sum(),
        LossesBE=lambda x: (x <= 0).sum(),
        TotalPnL='sum'
    )
    tf_summary_text = f"SUMMARY BY TIMEFRAME:\n{summary_by_tf.to_string()}\n"
    app_logger_g.info(tf_summary_text)
    cprint("SUMMARY BY TIMEFRAME:", 'cyan', attrs=['bold'])
    print(tabulate(summary_by_tf, headers='keys', tablefmt='fancy_grid', floatfmt=".2f"))
    print("\n")

    summary_by_strat = df_all_trades.groupby('strategy')['pnl'].agg(
        TotalTrades='count',
        Wins=lambda x: (x > 0).sum(),
        LossesBE=lambda x: (x <= 0).sum(),
        TotalPnL='sum'
    )
    strat_summary_text = f"SUMMARY BY STRATEGY (Overall):\n{summary_by_strat.to_string()}\n"
    app_logger_g.info(strat_summary_text)
    cprint("SUMMARY BY STRATEGY (Overall):", 'cyan', attrs=['bold'])
    print(tabulate(summary_by_strat, headers='keys', tablefmt='fancy_grid', floatfmt=".2f"))
    print("\n")

    cols_to_display = ['timeframe', 'strategy', 'side', 'entry_time', 'entry_price', 'exit_time', 'exit_price', 'pnl', 'exit_reason']
    cols_to_display = [col for col in cols_to_display if col in df_all_trades.columns]

    detailed_trades_text = f"DETAILED TRADES:\n{df_all_trades[cols_to_display].to_string(index=False)}\n"
    app_logger_g.info(detailed_trades_text)
    cprint("DETAILED TRADES:", 'yellow', attrs=['bold'])
    print(tabulate(df_all_trades[cols_to_display], headers='keys', tablefmt='fancy_grid', showindex=False, floatfmt=".2f"))

    report_csv_path = os.path.join(main_run_dir, f"{report_name_prefix}_trades_consolidated_report.csv")
    df_all_trades.to_csv(report_csv_path, index=False)
    msg_saved = f"Consolidated trade report saved to: {report_csv_path}"
    app_logger_g.info(msg_saved)
    cprint(f"\n{msg_saved}", "green")

# Add this lock at the top of your file with the other global variables
trade_journal_lock = threading.Lock()


def run_simulation_orchestrator():
    global base_run_dir_g, run_id_g, fetcher_g
    run_id_g = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:4]
    base_run_dir_g = os.path.join("simulation_runs", run_id_g)
    log_dir = os.path.join(base_run_dir_g, "logs")
    setup_logging(run_id_g, log_dir)
    app_logger_g.info(f"========== MULTI-TIMEFRAME SIMULATION MODE (Run ID: {run_id_g}) ==========")

    index_key = "NIFTY"
    exchange = "NSE"
    timeframes_to_sim = TIMEFRAMES
    sim_date_dt = datetime(2025, 5, 28)
    tz = pytz.timezone("Asia/Kolkata")
    strategies_config = {name: default_params.get(name, {}) for name in strategy_factories.keys()}
    all_trades_g.clear()
    fetcher_g = None  # Reset fetcher

    with ThreadPoolExecutor(max_workers=len(timeframes_to_sim), thread_name_prefix="SimWorker") as executor:
        futures = []
        for tf in timeframes_to_sim:
            future = executor.submit(run_single_simulation_task, tf, index_key, exchange, base_run_dir_g, sim_date_dt, tz, strategies_config, log_dir)
            futures.append(future)
            time.sleep(2)  # Stagger thread starts
        for future in futures:
            try:
                completed_tf = future.result()
                app_logger_g.info(f"Simulation future for timeframe {completed_tf} completed.")
            except Exception as exc:
                app_logger_g.error(f"A simulation task generated an exception: {exc}", exc_info=True)

    app_logger_g.info("All simulation tasks submitted and processed.")
    generate_consolidated_report(all_trades_g, base_run_dir_g, report_name_prefix=f"sim_{sim_date_dt.strftime('%Y%m%d')}")
    app_logger_g.info(f"[DONE] All simulations complete for {sim_date_dt.strftime('%d-%b-%Y')}. Details in: {base_run_dir_g}")

def simulate_future_candles_tf(df_raw_hist, sim_date_dt, timeframe, tz, token, tf_tick_logger):
    if df_raw_hist.empty or 'close' not in df_raw_hist.columns:
        app_logger_g.warning(f"[{timeframe}] Cannot simulate future candles. Historical data is empty or missing 'close'.")
        return pd.DataFrame()

    last_close = int(df_raw_hist['close'].iloc[-1] * 100)
    minutes = int(timeframe.replace("min", ""))
    total_market_minutes = int((dttime(15, 30).hour * 60 + dttime(15, 30).minute) -
                              (dttime(9, 15).hour * 60 + dttime(9, 15).minute))

    sim_candles_list = simulate_and_aggregate_candles(
        start_price=last_close,
        total_minutes=total_market_minutes,
        interval_minutes=minutes,
        token=token,
        tick_logger=tf_tick_logger
    )

    market_open_time = tz.localize(datetime.combine(sim_date_dt, dttime(9, 15)))
    for i, candle_dict in enumerate(sim_candles_list):
        candle_dict['datetime'] = market_open_time + timedelta(minutes=i * minutes)

    app_logger_g.info(f"[{timeframe}] Simulated {len(sim_candles_list)} future candles for {sim_date_dt.date()}.")
    return pd.DataFrame(sim_candles_list)

def simulate_and_aggregate_candles(start_price, total_minutes, interval_minutes, token, tick_logger=None):
    aggregator = TickAggregator(interval_minutes=interval_minutes)
    candles = []
    current_sim_price = start_price
    num_candles_to_sim = total_minutes // interval_minutes

    for _ in range(num_candles_to_sim):
        ticks_this_interval = simulate_ticks_for_interval(current_sim_price, interval_minutes * 60, num_ticks=60, token=token, tick_logger=tick_logger)
        for tick in ticks_this_interval:
            aggregated_candle = aggregator.process_tick(tick)
            if aggregated_candle:
                candles.append(aggregated_candle)
                if 'close' in aggregated_candle:
                    current_sim_price = aggregated_candle['close']
                break
        last_candle = aggregator.force_close()
        if last_candle:
            candles.append(last_candle)
            if 'close' in last_candle:
                current_sim_price = last_candle['close']

    for c in candles:
        for key_price in ['open', 'high', 'low', 'close']:
            if key_price in c:
                c[key_price] /= 100.0
    return candles

def simulate_ticks_for_interval(start_price, interval_seconds, num_ticks, token, tick_logger=None):
    ticks = []
    current_price = start_price
    base_sim_timestamp_ms = int(time.time() * 1000)
    for i in range(num_ticks):
        price_change = random.randint(-5, 5)
        current_price = max(1, current_price + price_change)
        tick = {
            'subscription_mode': 1,
            'exchange_type': 1,
            'token': token,
            'sequence_number': i,
            'exchange_timestamp': base_sim_timestamp_ms + (i * (interval_seconds * 1000 // num_ticks)),
            'last_traded_price': current_price,
            'subscription_mode_val': 'LTP'
        }
        ticks.append(tick)
        if tick_logger:
            tick_logger.info(f"SIM_TICK: {tick}")
    return ticks
def run_strategies_on_simulated_day_tf(simulated_day_with_ind, full_history_with_ind, strategies_config, timeframe, tf_paths, sim_date_str):
    trades_for_this_tf_and_strat = {}
    active_trades_sim = {}

    if simulated_day_with_ind.empty:
        app_logger_g.warning(f"[{timeframe}] No simulated day data for {sim_date_str} to run strategies.")
        return

    for idx, (dt_index, candle_row) in enumerate(simulated_day_with_ind.iterrows()):
        current_history_for_signal = full_history_with_ind.loc[:dt_index]

        for strat_name, strat_params in strategies_config.items():
            if strat_name not in strategy_factories:
                app_logger_g.error(f"[{timeframe}] Strategy {strat_name} not found!")
                continue

            strat_func = strategy_factories[strat_name]
            out = strat_func(candle_row, current_history_for_signal, strat_params)
            active_trade = active_trades_sim.get(strat_name)

            if out['signal'] in ('buy_potential', 'sell_potential'):
                if not active_trade or active_trade['status'] == 'closed':
                    trade = {
                        'run_id': run_id_g,
                        'timeframe': timeframe,
                        'strategy': strat_name,
                        'entry_time': dt_index,
                        'entry_price': candle_row['close'],
                        'side': 'LONG' if out['signal'] == 'buy_potential' else 'SHORT',
                        'sl': out.get('sl'),
                        'tp': out.get('tp'),
                        'status': 'open',
                        'exit_time': pd.NaT,
                        'exit_price': np.nan,
                        'pnl': np.nan,
                        'exit_reason': None
                    }
                    active_trades_sim[strat_name] = trade
                    print_trade_action_tf(trade, "OPEN", timeframe)

            elif active_trade and active_trade['status'] == 'open':
                exit_trade_flag = False
                exit_price_val = np.nan
                exit_reason_str = ""

                sl = active_trade.get('sl')
                tp = active_trade.get('tp')

                if sl is not None and pd.notna(sl):
                    if (active_trade['side'] == 'LONG' and candle_row['low'] <= sl) or \
                       (active_trade['side'] == 'SHORT' and candle_row['high'] >= sl):
                        exit_trade_flag = True
                        exit_price_val = sl
                        exit_reason_str = "Stop Loss"

                if not exit_trade_flag and tp is not None and pd.notna(tp):
                    if (active_trade['side'] == 'LONG' and candle_row['high'] >= tp) or \
                       (active_trade['side'] == 'SHORT' and candle_row['low'] <= tp):
                        exit_trade_flag = True
                        exit_price_val = tp
                        exit_reason_str = "Target Profit"

                if not exit_trade_flag and idx == len(simulated_day_with_ind) - 1:
                    exit_trade_flag = True
                    exit_price_val = candle_row['close']
                    exit_reason_str = f"EOD_SIM_{sim_date_str}"

                if exit_trade_flag:
                    pnl_val = (exit_price_val - active_trade['entry_price']) if active_trade['side'] == 'LONG' else (active_trade['entry_price'] - exit_price_val)
                    active_trade.update({
                        'exit_time': dt_index,
                        'exit_price': exit_price_val,
                        'pnl': pnl_val,
                        'status': 'closed',
                        'exit_reason': exit_reason_str
                    })
                    print_trade_action_tf(active_trade, "EXIT", timeframe)
                    if strat_name not in trades_for_this_tf_and_strat:
                        trades_for_this_tf_and_strat[strat_name] = []
                    trades_for_this_tf_and_strat[strat_name].append(active_trade.copy())
                    with all_trades_lock:
                        all_trades_g.append(active_trade.copy())
                    active_trades_sim[strat_name] = {'status': 'closed'}

    for strat_name, trade_to_close in active_trades_sim.items():
        if trade_to_close and trade_to_close.get('status') == 'open':
            final_candle = simulated_day_with_ind.iloc[-1]
            exit_price_eod = final_candle['close']
            pnl_eod = (exit_price_eod - trade_to_close['entry_price']) if trade_to_close['side'] == 'LONG' else (trade_to_close['entry_price'] - exit_price_eod)
            trade_to_close.update({
                'exit_time': simulated_day_with_ind.index[-1],
                'exit_price': exit_price_eod,
                'pnl': pnl_eod,
                'status': 'closed',
                'exit_reason': f"EOD_SIM_FORCE_CLOSE_{sim_date_str}"
            })
            print_trade_action_tf(trade_to_close, "EXIT", timeframe)
            if strat_name not in trades_for_this_tf_and_strat:
                trades_for_this_tf_and_strat[strat_name] = []
            trades_for_this_tf_and_strat[strat_name].append(trade_to_close.copy())
            with all_trades_lock:
                all_trades_g.append(trade_to_close.copy())

    for strat_name_key, strat_trades_list in trades_for_this_tf_and_strat.items():
        if strat_trades_list:
            save_trade_to_timeframe_strategy_csv(strat_trades_list, tf_paths["trades_dir"], strat_name_key, timeframe)
    app_logger_g.info(f"[{timeframe}] Finished strategy run for {sim_date_str}. All trades for this TF saved.")



def save_or_update_trade_entry(trade_details, journal_path):
    """
    Saves a new trade entry or updates an existing one in the live trade journal CSV.

    Args:
        trade_details (dict): The dictionary containing all details of a single trade.
        journal_path (str): The full path to the live trade journal CSV file.
    """
    with trade_journal_lock:
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(journal_path), exist_ok=True)

            # Prepare the new data as a DataFrame
            # We use .copy() to avoid modifying the original dictionary in memory
            update_df = pd.DataFrame([trade_details.copy()])

            # Read existing journal if it exists
            if os.path.exists(journal_path):
                journal_df = pd.read_csv(journal_path)
                
                # Check if the trade_id already exists
                if trade_details['trade_id'] in journal_df['trade_id'].values:
                    # --- UPDATE EXISTING TRADE ---
                    # Set trade_id as index to easily update the row
                    journal_df.set_index('trade_id', inplace=True)
                    update_df.set_index('trade_id', inplace=True)
                    
                    # Update the existing row with the new data
                    journal_df.update(update_df)
                    
                    # Reset index to bring trade_id back as a column
                    journal_df.reset_index(inplace=True)
                else:
                    # --- ADD NEW TRADE ---
                    # Append the new trade if trade_id is not found
                    journal_df = pd.concat([journal_df, update_df], ignore_index=True)
            else:
                # --- CREATE NEW JOURNAL ---
                # If journal file doesn't exist, the update_df is our new journal
                journal_df = update_df

            # Define the column order for consistency
            final_columns = [
                'trade_id', 'run_id', 'timeframe', 'strategy', 'side', 'status',
                'entry_time', 'entry_price', 'exit_time', 'exit_price', 
                'sl', 'tp', 'pnl', 'exit_reason'
            ]
            # Filter out columns that might not exist in the trade_details yet
            cols_to_save = [col for col in final_columns if col in journal_df.columns]

            # Save the updated DataFrame back to the CSV
            journal_df.to_csv(journal_path, index=False, columns=cols_to_save)

        except Exception as e:
            app_logger_g.error(f"Failed to save or update trade journal at {journal_path}: {e}", exc_info=True)

def run_single_simulation_task(timeframe, index_key, exchange, main_run_dir_sim, sim_date_dt, tz, strategies_config, tick_log_dir_sim):
    app_logger_g.info(f"SIM_TASK [{timeframe}] Starting simulation for date: {sim_date_dt.date()}")
    symbol_config = INDEX_SYMBOL_MAP[index_key]
    symbol_hist_name = symbol_config["hist_symbol"]
    symbol_name_safe = symbol_hist_name.replace(" ", "_")
    tf_paths = get_timeframe_specific_paths(main_run_dir_sim, timeframe, symbol_name_safe)
    tf_tick_logger_sim = get_timeframe_tick_logger(timeframe, run_id_g + "_sim", tick_log_dir_sim)

    df_hist_ind = load_and_prepare_data_tf(symbol_hist_name, exchange, timeframe, tf_paths)
    if df_hist_ind.empty:
        app_logger_g.error(f"SIM_TASK [{timeframe}] Failed to load historical data. Aborting.")
        return timeframe

    df_sim_future = simulate_future_candles_tf(df_hist_ind.iloc[[-1]].reset_index(), sim_date_dt, timeframe, tz, symbol_config["hist_token"], tf_tick_logger_sim)
    df_all_data_with_ind = merge_and_compute_indicators_sim_tf(df_hist_ind.copy(), df_sim_future.copy(), tf_paths, symbol_name_safe, timeframe, sim_date_dt.strftime("%Y%m%d"))
    if df_all_data_with_ind.empty:
        app_logger_g.error(f"SIM_TASK [{timeframe}] Failed after merging/indicators. Aborting.")
        return timeframe

    simulated_day_data = df_all_data_with_ind.loc[df_all_data_with_ind.index.date == sim_date_dt.date()].copy()
    if simulated_day_data.empty:
        app_logger_g.warning(f"SIM_TASK [{timeframe}] No data found for simulated date {sim_date_dt.date()}. Using latest available.")
        target_date_to_run = df_all_data_with_ind.index.max().date()
        simulated_day_data = df_all_data_with_ind.loc[df_all_data_with_ind.index.date == target_date_to_run].copy()
        if simulated_day_data.empty:
            app_logger_g.error(f"SIM_TASK [{timeframe}] Still no data to run strategies. Aborting.")
            return timeframe

    run_strategies_on_simulated_day_tf(simulated_day_data, df_all_data_with_ind, strategies_config, timeframe, tf_paths, sim_date_dt.strftime("%Y%m%d"))
    app_logger_g.info(f"SIM_TASK [{timeframe}] Simulation task finished.")
    return timeframe
def merge_and_compute_indicators_sim_tf(df_raw_hist_indexed, df_sim_future, tf_paths, symbol_name_safe, timeframe, sim_date_str):
    if 'datetime' in df_sim_future.columns:
        df_sim_future_indexed = df_sim_future.set_index('datetime')
    else:
        df_sim_future_indexed = df_sim_future

    if not df_sim_future_indexed.empty and df_sim_future_indexed.index.tzinfo is None:
        df_sim_future_indexed = df_sim_future_indexed.tz_localize('Asia/Kolkata', ambiguous='infer', nonexistent='shift_forward')

    df_full = pd.concat([df_raw_hist_indexed, df_sim_future_indexed], axis=0)
    df_full.sort_index(inplace=True)
    df_full = df_full[~df_full.index.duplicated(keep='last')]

    out_file_path = os.path.join(tf_paths["ind_dir"], f"{symbol_name_safe}_{timeframe}_sim_{sim_date_str}_full_with_indicators.csv")
    df_all_with_ind = compute_indicators(df_full.copy(), out_file_path)
    if df_all_with_ind is None or df_all_with_ind.empty:
        app_logger_g.error(f"[{timeframe}] Indicator computation failed for simulated data.")
        return pd.DataFrame()
    df_all_with_ind.to_csv(out_file_path)
    app_logger_g.info(f"[{timeframe}] Saved SIM merged data with indicators to: {out_file_path}")
    return df_all_with_ind
def live_timeframe_processor_thread_worker(timeframe, index_key, exchange, main_run_dir_live, strategies_config, tz, tick_log_dir_live):
    if LiveDataFeeder is None:
        app_logger_g.error(f"[{timeframe}] LiveDataFeeder not available. Cannot start live processor.")
        return

    thread_name = threading.current_thread().name
    app_logger_g.info(f"[{timeframe}/{thread_name}] Initializing live processor...")
    interval_minutes = int(timeframe.replace("min", ""))
    symbol_config = INDEX_SYMBOL_MAP[index_key]
    symbol_hist_name = symbol_config["hist_symbol"]
    symbol_name_safe = symbol_hist_name.replace(" ", "_")
    tick_token_live = symbol_config["tick_token"]
    live_feed_tokens = [{"exchangeType": 1, "tokens": [tick_token_live]}]
    tf_paths = get_timeframe_specific_paths(main_run_dir_live, timeframe, symbol_name_safe)
    tf_tick_logger = get_timeframe_tick_logger(timeframe, run_id_g + "_live", tick_log_dir_live)

    current_history_with_ind = load_and_prepare_data_tf(symbol_hist_name, exchange, timeframe, tf_paths)
    if current_history_with_ind.empty:
        app_logger_g.error(f"[{timeframe}/{thread_name}] Failed to load initial historical data. Live processor stopping.")
        return
    app_logger_g.info(f"[{timeframe}/{thread_name}] Initial historical data loaded. Last record: {current_history_with_ind.index[-1]}")

    active_trades_live_tf = {}

   # This function should be defined inside your `live_timeframe_processor_thread_worker`
    def handle_new_live_candle(candle_dict):
        nonlocal current_history_with_ind, active_trades_live_tf
        try:
            if stop_event_g.is_set(): return

            # --- Candle and Indicator Setup (remains the same) ---
            dt_object = candle_dict.get('datetime')
            # ... (all your existing code for parsing the datetime and calculating indicators) ...
            # Ensure the section below runs after your indicators are successfully calculated
            # and current_candle_data_row is defined.
            
            # [This block is a placeholder for your existing data prep logic]
            if not dt_object or not isinstance(dt_object, datetime): return
            if dt_object.tzinfo is None: dt_object = tz.localize(dt_object)
            else: dt_object = dt_object.astimezone(tz)
            candle_dict['datetime'] = dt_object
            cprint(f"[{timeframe}] Candle: {dt_object.strftime('%H:%M:%S')} O:{candle_dict['open']:.2f} H:{candle_dict['high']:.2f} L:{candle_dict['low']:.2f} C:{candle_dict['close']:.2f} V:{candle_dict.get('volume',0)}", "blue")
            new_candle_df = pd.DataFrame([candle_dict])
            new_candle_df.set_index('datetime', inplace=True)
            temp_history = pd.concat([current_history_with_ind, new_candle_df], axis=0)
            temp_history.sort_index(inplace=True)
            temp_history = temp_history[~temp_history.index.duplicated(keep='last')]
            out_file_path = os.path.join(tf_paths["ind_dir"], f"{symbol_name_safe}_{timeframe}_live_temp.csv")
            updated_full_history_with_ind = compute_indicators(temp_history.copy(), out_file_path)
            if updated_full_history_with_ind is None or updated_full_history_with_ind.empty: return
            current_history_with_ind = updated_full_history_with_ind
            current_candle_data_row = current_history_with_ind.loc[dt_object]
            history_for_strategies = current_history_with_ind.loc[:dt_object]
            journal_path = os.path.join(base_run_dir_g, "live_trades_journal.csv")
            # [End of data prep placeholder]


            # ======================================================================
            # START OF NEW TWO-PASS LOGIC
            # ======================================================================

            # --- PASS 1: PROCESS EXITS FOR ALL ACTIVE TRADES ---
            # We iterate over a copy of the keys to allow modification of the dictionary during the loop
            for strat_name in list(active_trades_live_tf.keys()):
                active_trade = active_trades_live_tf.get(strat_name)

                if active_trade and active_trade.get('status') == 'OPEN':
                    exit_trade_flag, exit_price_val, exit_reason_str = False, np.nan, ""
                    sl, tp = active_trade.get('sl'), active_trade.get('tp')

                    if sl and pd.notna(sl) and ((active_trade['side'] == 'LONG' and current_candle_data_row['low'] <= sl) or (active_trade['side'] == 'SHORT' and current_candle_data_row['high'] >= sl)):
                        exit_trade_flag, exit_price_val, exit_reason_str = True, sl, "Stop Loss"
                    elif tp and pd.notna(tp) and ((active_trade['side'] == 'LONG' and current_candle_data_row['high'] >= tp) or (active_trade['side'] == 'SHORT' and current_candle_data_row['low'] <= tp)):
                        exit_trade_flag, exit_price_val, exit_reason_str = True, tp, "Target Profit"
                    elif (dt_object + timedelta(minutes=interval_minutes)).time() >= dttime(15, 30):
                        exit_trade_flag, exit_price_val, exit_reason_str = True, current_candle_data_row['close'], "Market End Closure"

                    if exit_trade_flag:
                        pnl_val = (exit_price_val - active_trade['entry_price']) if active_trade['side'] == 'LONG' else (active_trade['entry_price'] - exit_price_val)
                        active_trade.update({
                            'exit_time': dt_object, 'exit_price': exit_price_val, 'pnl': pnl_val,
                            'status': 'CLOSED', 'exit_reason': exit_reason_str
                        })
                        print_trade_action_tf(active_trade, "EXIT", timeframe)
                        # Your save_or_update_trade_entry function already handles this
                        # save_or_update_trade_entry(active_trade, journal_path) 
                        
                        # Update the status to CLOSED
                        active_trades_live_tf[strat_name] = active_trade 


            # --- PASS 2: CHECK FOR NEW ENTRIES (ONLY IF NO POSITIONS ARE OPEN) ---
            is_trade_active = any(trade.get('status') == 'OPEN' for trade in active_trades_live_tf.values())

            if not is_trade_active:
                # If no trades are open, now we can look for a new entry signal
                for strat_name, strat_params in strategies_config.items():
                    if stop_event_g.is_set(): break
                    if strat_name not in strategy_factories: continue

                    strat_func = strategy_factories[strat_name]
                    out = strat_func(current_candle_data_row, history_for_strategies, strat_params)

                    if out['signal'] in ('buy_potential', 'sell_potential'):
                        trade_id = f"TID_{run_id_g}_{timeframe}_{strat_name}_{int(dt_object.timestamp())}"
                        trade = {
                            'trade_id': trade_id, 'run_id': run_id_g, 'timeframe': timeframe,
                            'strategy': strat_name, 'entry_time': dt_object,
                            'entry_price': current_candle_data_row['close'],
                            'side': 'LONG' if out['signal'] == 'buy_potential' else 'SHORT',
                            'sl': out.get('sl'), 'tp': out.get('tp'), 'status': 'OPEN',
                            'exit_time': pd.NaT, 'exit_price': np.nan, 'pnl': np.nan, 'exit_reason': ''
                        }
                        active_trades_live_tf[strat_name] = trade
                        print_trade_action_tf(trade, "OPEN", timeframe)
                        # Your save_or_update_trade_entry function already handles this
                        # save_or_update_trade_entry(trade, journal_path)

                        # Since we opened a trade, we break the loop to enforce "one trade at a time"
                        break 
            
            # ======================================================================
            # END OF NEW TWO-PASS LOGIC
            # ======================================================================

        except Exception as e_candle:
            app_logger_g.error(f"[{timeframe}] CRITICAL ERROR in handle_new_live_candle: {e_candle}", exc_info=True) 
        

    def handle_new_live_tick(tick_data):
            if tf_tick_logger:
                tf_tick_logger.info(f"LIVE_TICK: {tick_data}")

    feeder_instance = None
    try:
        feeder_instance = LiveDataFeeder(
        tokens=live_feed_tokens,
        interval_minutes=interval_minutes,
        candle_callback=handle_new_live_candle,
        tick_callback=handle_new_live_tick,
        mode=1,
        correlation_id=run_id_g
    )

        with all_trades_lock:
            live_feeders_g.append(feeder_instance)
        feeder_instance.start()
        app_logger_g.info(f"[{timeframe}/{thread_name}] LiveDataFeeder started.")

        # ========== THIS IS THE ONLY SECTION THAT HAS BEEN CHANGED ==========
        while not stop_event_g.is_set():
            if hasattr(feeder_instance, 'is_alive') and not feeder_instance.is_alive():
                app_logger_g.warning(f"[{timeframe}/{thread_name}] Feeder is no longer alive. Worker stopping.")
                break
            
            now_live_tf = datetime.now(tz)
            if now_live_tf.time() >= dttime(15, 30, tzinfo=tz):
                app_logger_g.info(f"[{timeframe}/{thread_name}] Market hours ended. Requesting feeder stop.")
                break

            # This single change makes the thread responsive to Ctrl+C.
            # It waits for 2 seconds OR until the stop event is set, whichever happens first.
            if stop_event_g.wait(timeout=2):
                app_logger_g.info(f"[{timeframe}/{thread_name}] Stop event detected, exiting worker loop.")
                break
        # ====================================================================

    except Exception as e_feeder_setup:
        app_logger_g.error(f"[{timeframe}/{thread_name}] Failed to initialize or run LiveDataFeeder: {e_feeder_setup}", exc_info=True)
    finally:
        app_logger_g.info(f"[{timeframe}/{thread_name}] Live processor thread worker stopping...")
        if feeder_instance and hasattr(feeder_instance, 'stop'):
            try:
                app_logger_g.info(f"[{timeframe}/{thread_name}] Calling feeder.stop().")
                feeder_instance.stop()
            except Exception as e_feeder_stop_final:
                app_logger_g.error(f"[{timeframe}/{thread_name}] Error stopping feeder: {e_feeder_stop_final}", exc_info=True)

def run_live_orchestrator():
    global base_run_dir_g, run_id_g, live_feeders_g, fetcher_g
    run_id_g = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:4]
    base_run_dir_g = os.path.join("live_session_runs", run_id_g)
    log_dir = os.path.join(base_run_dir_g, "logs")
    setup_logging(run_id_g, log_dir)
    app_logger_g.info(f"========== MULTI-TIMEFRAME LIVE MODE (Run ID: {run_id_g}) ==========")

    index_key = "NIFTY"
    exchange = "NSE"
    timeframes_to_run = TIMEFRAMES
    tz = pytz.timezone("Asia/Kolkata")
    strategies_config = {name: default_params.get(name, {}) for name in strategy_factories.keys()}
    all_trades_g.clear()
    live_feeders_g.clear()
    stop_event_g.clear()
    fetcher_g = None  # Reset fetcher

    live_threads = []
    for tf_str in timeframes_to_run:
        thread = threading.Thread(
            target=live_timeframe_processor_thread_worker,
            args=(tf_str, index_key, exchange, base_run_dir_g, strategies_config, tz, log_dir),
            name=f"LiveWorker-{tf_str}"
        )
        live_threads.append(thread)
        thread.daemon = True
        thread.start()
        time.sleep(2)  # Stagger thread starts

    app_logger_g.info("All live timeframe processor threads initiated.")
    cprint("Live session started. Press Ctrl+C to stop and generate report.", "magenta")

    try:
        while not stop_event_g.is_set():
            now = datetime.now(tz)
            if now.time() >= dttime(15, 32, tzinfo=tz):
                app_logger_g.info("Market hours officially over. Initiating EOD shutdown.")
                stop_event_g.set()
                break
            if not any(t.is_alive() for t in live_threads) and live_threads:
                app_logger_g.warning("All live worker threads seem to have terminated prematurely. Shutting down.")
                stop_event_g.set()
                break
            time.sleep(5)

    except KeyboardInterrupt:
        app_logger_g.info("Ctrl+C pressed by user. Orchestrator initiating shutdown.")
    finally:
        app_logger_g.info("Orchestrator shutdown sequence started...")
        if not stop_event_g.is_set():
            stop_event_g.set()

        app_logger_g.info("Requesting all LiveDataFeeders to stop...")
        feeders_to_stop = []
        with all_trades_lock:
            feeders_to_stop = list(live_feeders_g)

        for feeder_instance in feeders_to_stop:
            if hasattr(feeder_instance, 'stop') and callable(feeder_instance.stop):
                try:
                    app_logger_g.debug(f"Stopping feeder: {feeder_instance}")
                    feeder_instance.stop()
                except Exception as e_feeder_stop_orch:
                    app_logger_g.error(f"Error stopping a feeder during orchestration shutdown: {e_feeder_stop_orch}", exc_info=True)

        app_logger_g.info("Waiting for live threads to complete (max 15s each)...")
        for t_live in live_threads:
            t_live.join(timeout=15)
            if t_live.is_alive():
                app_logger_g.warning(f"Thread {t_live.name} did not terminate gracefully after 15s.")

        app_logger_g.info("Generating final consolidated report for live session...")
        generate_consolidated_report(all_trades_g, base_run_dir_g, report_name_prefix="live_session_final")
        app_logger_g.info(f"[END OF SESSION] Live session (ID: {run_id_g}) concluded. Details in: {base_run_dir_g}")
        cprint(f"Session logs and reports saved in: {base_run_dir_g}", "green")

def main_signal_handler(sig, frame):
    if not stop_event_g.is_set():
        app_logger_g.warning(f"Signal {sig} received. Setting global stop event.")
        cprint("\nCtrl+C detected! Initiating graceful shutdown. Please wait...", "yellow", attrs=['bold'])

        stop_event_g.set()

if __name__ == "__main__":
    # The simulation-related functions are extensive and not directly related to the live-mode issue,
    # so they are omitted here for clarity, but would be part of your complete file.
    # run_simulation_orchestrator()
    # run_single_simulation_task(...)
    
    signal.signal(signal.SIGINT, main_signal_handler)
    parser = argparse.ArgumentParser(description="Multi-Timeframe AlgoTrading System (Simulation/Live)")
    parser.add_argument("--mode", choices=["sim", "live"], default="sim", help="Choose mode: sim for simulation, live for live market.")
    parser.add_argument("--index", choices=list(INDEX_SYMBOL_MAP.keys()), default="NIFTY", help="Index to trade/simulate.")
    args = parser.parse_args()

    # Initial setup of logger
    setup_logging("startup", os.path.join("logs", "startup_logs"))

    all_trades_g = []
    stop_event_g.clear()

    print(f"Starting application in '{args.mode}' mode for index '{args.index}'.")
    if args.mode == "live":
      if LiveDataFeeder is None:
        msg_err = "ERROR: LiveDataFeeder could not be imported. Live mode is disabled."
        app_logger_g.critical(msg_err)
        cprint(msg_err, "red", attrs=["bold"])
      else:
        run_live_orchestrator()
    else:
        # Placeholder for your simulation logic if you run in sim mode
        app_logger_g.info("Simulation mode selected. ")
        run_simulation_orchestrator()

    if app_logger_g:
        app_logger_g.info("Application finished.")
    else:
        print("Application finished.")
