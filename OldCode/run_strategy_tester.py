
# from jinja2 import Environment, FileSystemLoader
# import sys
# import os

# from trade_option_report import generate_option_report_from_csv
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import logging
# import pandas as pd
# from datetime import datetime, time
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from app.load_data import load_all_data
# from app.compute_indicators import compute_indicators
# from app.backtest_engine import SimpleBacktester, save_trades_to_csv
# from strategies import strategy_factories
# from app.generate_report import generate_strategy_report
# from app import trade_reporter as reporter
# import traceback
# import config as config

# class ColorFormatter(logging.Formatter):
#     RED = '\033[91m'
#     YELLOW = '\033[93m'
#     RESET = '\033[0m'

#     def format(self, record):
#         if record.levelno == logging.ERROR:
#             record.msg = f"{self.RED}{record.msg}{self.RESET}"
#         elif record.levelno == logging.WARNING:
#             record.msg = f"{self.YELLOW}{record.msg}{self.RESET}"
#         return super().format(record)

# handler = logging.StreamHandler()
# handler.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.addHandler(handler)

# def run_strategy(strategy_name, nifty_df, params, timeframe, symbol, exchange, expiry_date, strike_price, run_id, ce_df=None, pe_df=None):
#     """Helper function to run a single strategy's backtest."""
#     logger.info(f"Testing strategy: {strategy_name} on timeframe: {timeframe}")

#     if nifty_df.index.has_duplicates:
#         duplicate_count = nifty_df.index.duplicated().sum()
#         logger.warning(f"Found {duplicate_count} duplicate timestamps in Nifty DataFrame for {strategy_name} on {timeframe}. Removing duplicates...")
#         nifty_df = nifty_df[~nifty_df.index.duplicated(keep='first')]
#         logger.info(f"Nifty DataFrame shape after removing duplicates for {strategy_name} on {timeframe}: {nifty_df.shape}")

#     if not nifty_df.index.is_unique:
#         logger.error(f"Index is not unique for Nifty DataFrame in {strategy_name} on {timeframe} after duplicate removal!")
#         raise ValueError(f"Non-unique index for Nifty DataFrame in {strategy_name} on {timeframe}")

#     strategy_params = params.get(strategy_name, {}).copy()
#     strategy_params.update({k: v for k, v in config.common_params.items() if k not in strategy_params})

#     try:
#         backtester = SimpleBacktester(
#             nifty_df,
#             strategy_name=strategy_name,
#             strategy_func=strategy_factories[strategy_name],
#             params=strategy_params,
#             timeframe=timeframe,
#             symbol=symbol,
#             exchange=exchange,
#             ce_premiums_df=ce_df,
#             pe_premiums_df=pe_df,
#             run_id=run_id,
#             expiry_date=expiry_date,
#             strike_price=strike_price
#         )
#         result = backtester.run_simulation()
#     except Exception as e:
#         logger.error(f"Error in backtesting {strategy_name} on {timeframe}: {e}")
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         return {
#             'strategy': strategy_name,
#             'timeframe': timeframe,
#             'pnl': 0,
#             'win_rate': 0,
#             'buy_trades': 0,
#             'sell_trades': 0,
#             'exit_reasons': {'sl': 0, 'tsl': 0, 'tp': 0, 'signal': 0},
#             'total_trades': 0,
#             'profitable_trades': 0,
#             'losing_trades': 0,
#             'performance_score': 0,
#             'day_wise': {},
#             'session_wise': {},
#             'regime_wise': {},
#             'volatility_wise': {},
#             'expiry_wise': {},
#             'trades': [],
#             'error': str(e)
#         }

#     return {
#         'strategy': strategy_name,
#         'timeframe': timeframe,
#         'pnl': result['total_pnl'],
#         'win_rate': result['win_rate'],
#         'buy_trades': result['buy_trades'],
#         'sell_trades': result['sell_trades'],
#         'exit_reasons': result['exit_reasons'],
#         'total_trades': result['total_trades'],
#         'profitable_trades': result['profitable_trades'],
#         'losing_trades': result['losing_trades'],
#         'performance_score': result['performance_score'],
#         'day_wise': result['day_wise'],
#         'session_wise': result['session_wise'],
#         'regime_wise': result['regime_wise'],
#         'volatility_wise': result['volatility_wise'],
#         'expiry_wise': result['expiry_wise'],
#         'trades': result['trades']
#     }

# def process_timeframe(timeframe, default_params, symbol, exchange, expiry_date, strike_price, run_id):
#     """Process a single timeframe: load data, compute indicators on index, and run strategies."""
#     logger.info(f"Processing timeframe: {timeframe}")

#     timeframe_map = {
#         '5min': '5min',
#         # '3min': '3min',
#         # '15min': '15min',
#         # '1min': '1min',
#     }
#     if timeframe not in timeframe_map:
#         logger.error(f"Invalid timeframe: {timeframe}")
#         return [], []

#     output_path = f"data/datawithindicator/nifty_{timeframe}_with_indicators.csv"

#     symbol = "NIFTY"
#     timeframe = "5min"
#     expiry_date = "2025-05-29"
#     strike_price = "24000"

#     nifty_df, ce_df, pe_df = load_all_data(symbol, timeframe, expiry_date, strike_price)
#     if nifty_df is not None:
#         print(f"Nifty shape: {nifty_df.shape}")
#     if ce_df is not None:
#         print(f"CE shape: {ce_df.shape}")
#     if pe_df is not None:
#         print(f"PE shape: {pe_df.shape}")
#     try:
#         nifty_df =compute_indicators(nifty_df, output_path)
       
#         #logger.info(f"Loaded indicator-enriched DataFrame for {timeframe}. Columns: {list(nifty_df.columns)}")
#     except Exception as e:
#         logger.error(f"Error computing indicators for Nifty on {timeframe}: {e}")
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         return [], []

#     if nifty_df is None or nifty_df.empty:
#         logger.error(f"No Nifty data available for timeframe: {timeframe}")
#         return [], []

#     if nifty_df.index.has_duplicates:
#         nifty_df = nifty_df[~nifty_df.index.duplicated(keep='first')]

#     results = []
#     trades = []
#     max_workers = 4
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_strategy = {
#             executor.submit(
#                 run_strategy,
#                 strategy_name,
#                 nifty_df.copy(),
#                 config.default_params,
#                 timeframe,
#                 symbol,
#                 exchange,
#                 expiry_date,
#                 strike_price,
#                 run_id,
#                 ce_df,
#                 pe_df
#             ): strategy_name for strategy_name in strategy_factories.keys()
#         }
#         for future in as_completed(future_to_strategy):
#             strategy_name = future_to_strategy[future]
#             try:
#                 result = future.result()
#                 results.append(result)
#                 trades.extend(result['trades'])
#             except Exception as e:
#                 logger.error(f"Strategy {strategy_name} failed for {timeframe}: {e}")
#                 logger.error(f"Traceback: {traceback.format_exc()}")
#                 results.append({
#                     'strategy': strategy_name,
#                     'timeframe': timeframe,
#                     'pnl': 0,
#                     'win_rate': 0,
#                     'buy_trades': 0,
#                     'sell_trades': 0,
#                     'exit_reasons': {'sl': 0, 'tsl': 0, 'tp': 0, 'signal': 0},
#                     'total_trades': 0,
#                     'profitable_trades': 0,
#                     'losing_trades': 0,
#                     'performance_score': 0,
#                     'day_wise': {},
#                     'session_wise': {},
#                     'regime_wise': {},
#                     'volatility_wise': {},
#                     'expiry_wise': {},
#                     'trades': [],
#                     'error': str(e)
#                 })

#     return results, trades

# def run_strategy_tester(run_id, timeframes=['5min'],#, '3min','15min','1min'], 
#                         expiry='2025-05-29', strike='24000'):
#     logger.info(f"Starting strategy testing for Run ID: {run_id}, Timeframes: {timeframes}")

#     all_results = []
#     all_trades = []
#     max_timeframe_workers = min(4, len(timeframes))
#     logger.info(f"Using {max_timeframe_workers} workers for timeframes")
#     with ThreadPoolExecutor(max_workers=max_timeframe_workers) as executor:
#         future_to_timeframe = {
#             executor.submit(
#                 process_timeframe,
#                 timeframe,
#                 config.default_params,
#                 'NIFTY',
#                 'NSE',
#                 expiry,
#                 strike,
#                 run_id
#             ): timeframe for timeframe in timeframes
#         }
#         for future in as_completed(future_to_timeframe):
#             timeframe = future_to_timeframe[future]
#             try:
#                 results, trades = future.result()
#                 all_results.extend(results)
#                 all_trades.extend(trades)
#             except Exception as e:
#                 logger.error(f"Timeframe {timeframe} failed: {e}")
#                 logger.error(f"Traceback: {traceback.format_exc()}")

#     try:
#         save_trades_to_csv(all_trades, run_id)
#     except Exception as e:
#         logger.error(f"Error saving trades: {e}")
#         logger.error(f"Traceback: {traceback.format_exc()}")

#     option_trades_file_path = os.path.join("reports", f"run_{run_id}", "option_trade_logs", f"option_trades_{run_id}.csv")
#     option_pnl_analysis = {}

#     try:
#         df_option_trades = pd.read_csv(option_trades_file_path)
#         logger.info(f"Successfully loaded option trade data for P&L analysis.")

#         total_option_pnl = df_option_trades['pl'].sum()
#         option_pnl_analysis['total_option_pnl'] = round(total_option_pnl, 2)

#         df_option_trades['exit_time'] = pd.to_datetime(df_option_trades['exit_time'])
#         df_option_trades['exit_date'] = df_option_trades['exit_time'].dt.date
#         pnl_by_day = df_option_trades.groupby('exit_date')['pl'].sum().to_dict()
#         option_pnl_analysis['pnl_by_day'] = {str(date): round(pnl, 2) for date, pnl in pnl_by_day.items()}

#         pnl_by_option_type = df_option_trades.groupby('option_type')['pl'].sum().to_dict()
#         option_pnl_analysis['pnl_by_option_type'] = {option_type: round(pnl, 2) for option_type, pnl in pnl_by_option_type.items()}

#     except FileNotFoundError:
#         logger.warning(f"Option trade log file not found for P&L analysis at {option_trades_file_path}.")
#     except Exception as e:
#         logger.error(f"Error during option trade P&L analysis: {e}")
#         logger.error(f"Traceback: {traceback.format_exc()}")

#     #generate_option_pnl_report(option_pnl_analysis, run_id)
#     for strategy in strategy_factories.keys():
#         for tf in timeframes:
#             option_trades_file_path = f"reports/run_{run_id}/option_trade_logs/option_trades_{run_id}__{strategy}_{tf}.csv"
#             if os.path.exists(option_trades_file_path):  # <-- Only run if file exists
#                 generate_option_report_from_csv(option_trades_file_path, run_id)
#             else:
#                 print(f"[WARN] File not found: {option_trades_file_path}")

#     try:
#         generate_strategy_report(all_results, run_id, timeframes, option_pnl_analysis={})
#         logger.info(f"Completed strategy testing for Run ID: {run_id}")
#     except Exception as e:
#         logger.error(f"Error generating reports: {e}")
#         logger.error(f"Traceback: {traceback.format_exc()}")

# if __name__ == '__main__':
#     run_id = datetime.now().strftime('%Y%m%d%H%M%S')
#     run_strategy_tester(run_id, expiry='2025-05-29', strike='24000')
from jinja2 import Environment, FileSystemLoader
import sys
import os
import logging
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from load_data import load_all_data
from compute_indicators import compute_indicators
from backtest_engine import SimpleBacktester, save_trades_to_csv
from strategies import strategy_factories
from generate_report import generate_strategy_report
import trade_reporter as reporter
from trade_option_report import generate_option_report_from_csv
import traceback
import config as config

# Custom logging formatter for colored output
class ColorFormatter(logging.Formatter):
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

    def format(self, record):
        if record.levelno == logging.ERROR:
            record.msg = f"{self.RED}{record.msg}{self.RESET}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{self.YELLOW}{record.msg}{self.RESET}"
        return super().format(record)

# Configure logging
log_dir = "data/logs"
os.makedirs(log_dir, exist_ok=True)
run_id = datetime.now().strftime("%Y%m%d%H%M%S")
log_file = os.path.join(log_dir, f"app_run_{run_id}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def run_strategy(strategy_name, nifty_df, params, timeframe, symbol, exchange, expiry_date, strike_price, run_id, ce_df=None, pe_df=None):
    """Helper function to run a single strategy's backtest."""
    logger.info(f"Testing strategy: {strategy_name} on timeframe: {timeframe}")

    if nifty_df.index.has_duplicates:
        duplicate_count = nifty_df.index.duplicated().sum()
        logger.warning(f"Found {duplicate_count} duplicate timestamps in Nifty DataFrame for {strategy_name} on {timeframe}. Removing duplicates...")
        nifty_df = nifty_df[~nifty_df.index.duplicated(keep='first')]
        logger.info(f"Nifty DataFrame shape after removing duplicates for {strategy_name} on {timeframe}: {nifty_df.shape}")

    if not nifty_df.index.is_unique:
        logger.error(f"Index is not unique for Nifty DataFrame in {strategy_name} on {timeframe} after duplicate removal!")
        raise ValueError(f"Non-unique index for Nifty DataFrame in {strategy_name} on {timeframe}")

    strategy_params = params.get(strategy_name, {}).copy()
    strategy_params.update({k: v for k, v in config.common_params.items() if k not in strategy_params})

    try:
        backtester = SimpleBacktester(
            nifty_df,
            strategy_name=strategy_name,
            strategy_func=strategy_factories[strategy_name],
            params=strategy_params,
            timeframe=timeframe,
            symbol=symbol,
            exchange=exchange,
            ce_premiums_df=ce_df,
            pe_premiums_df=pe_df,
            run_id=run_id,
            expiry_date=expiry_date,
            strike_price=strike_price
        )
        result = backtester.run_simulation()
    except Exception as e:
        logger.error(f"Error in backtesting {strategy_name} on {timeframe}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'strategy': strategy_name,
            'timeframe': timeframe,
            'pnl': 0,
            'win_rate': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'exit_reasons': {'sl': 0, 'tsl': 0, 'tp': 0, 'signal': 0},
            'total_trades': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'performance_score': 0,
            'day_wise': {},
            'session_wise': {},
            'regime_wise': {},
            'volatility_wise': {},
            'expiry_wise': {},
            'trades': [],
            'error': str(e)
        }

    return {
        'strategy': strategy_name,
        'timeframe': timeframe,
        'pnl': result['total_pnl'],
        'win_rate': result['win_rate'],
        'buy_trades': result['buy_trades'],
        'sell_trades': result['sell_trades'],
        'exit_reasons': result['exit_reasons'],
        'total_trades': result['total_trades'],
        'profitable_trades': result['profitable_trades'],
        'losing_trades': result['losing_trades'],
        'performance_score': result['performance_score'],
        'day_wise': result['day_wise'],
        'session_wise': result['session_wise'],
        'regime_wise': result['regime_wise'],
        'volatility_wise': result['volatility_wise'],
        'expiry_wise': result['expiry_wise'],
        'trades': result['trades']
    }

def process_timeframe(timeframe, default_params, symbol, exchange, expiry_date, strike_price, run_id):
    """Process a single timeframe: load data, compute indicators on index, and run strategies."""
    logger.info(f"Processing timeframe: {timeframe}")

    timeframe_map = {
        '1min': '1min',
        '3min': '3min',
        '5min': '5min',
        '15min': '15min',
    }
    if timeframe not in timeframe_map:
        logger.error(f"Invalid timeframe: {timeframe}")
        return [], []

    output_path = f"data/datawithindicator/nifty_{timeframe}_with_indicators.csv"

    try:
        # Load NIFTY, CE, and PE data
        nifty_df, ce_df, pe_df = load_all_data(symbol, timeframe, expiry_date, strike_price)
        if nifty_df is not None:
            logger.info(f"Nifty shape: {nifty_df.shape}")
        if ce_df is not None:
            logger.info(f"CE shape: {ce_df.shape}")
        else:
            logger.warning(f"No CE data loaded for {timeframe}")
        if pe_df is not None:
            logger.info(f"PE shape: {pe_df.shape}")
        else:
            logger.warning(f"No PE data loaded for {timeframe}")
    except Exception as e:
        logger.error(f"Error loading data for {timeframe}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return [], []

    try:
        # Compute indicators
        nifty_df = compute_indicators(nifty_df, output_path)
        logger.info(f"Computed indicators for {timeframe}. Columns: {list(nifty_df.columns)}")
    except Exception as e:
        logger.error(f"Error computing indicators for Nifty on {timeframe}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return [], []

    if nifty_df is None or nifty_df.empty:
        logger.error(f"No Nifty data available for timeframe: {timeframe}")
        return [], []

    if nifty_df.index.has_duplicates:
        logger.warning(f"Removing duplicate timestamps in Nifty DataFrame for {timeframe}")
        nifty_df = nifty_df[~nifty_df.index.duplicated(keep='first')]

    results = []
    trades = []
    max_workers = min(4, len(strategy_factories))  # Limit workers to number of strategies or 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_strategy = {
            executor.submit(
                run_strategy,
                strategy_name,
                nifty_df.copy(),  # Use a copy to avoid shared state
                default_params,
                timeframe,
                symbol,
                exchange,
                expiry_date,
                strike_price,
                run_id,
                ce_df.copy() if ce_df is not None else None,
                pe_df.copy() if pe_df is not None else None
            ): strategy_name for strategy_name in strategy_factories.keys()
        }
        for future in as_completed(future_to_strategy):
            strategy_name = future_to_strategy[future]
            try:
                result = future.result()
                results.append(result)
                trades.extend(result['trades'])
                # Save trades for this strategy and timeframe
                if result['trades']:
                    trade_file = os.path.join(
                        "reports", f"run_{run_id}", "option_trade_logs",
                        f"option_trades_{run_id}__{strategy_name}_{timeframe}.csv"
                    )
                    os.makedirs(os.path.dirname(trade_file), exist_ok=True)
                    pd.DataFrame(result['trades']).to_csv(trade_file, index=False)
                    logger.info(f"Saved {len(result['trades'])} trades for {strategy_name} on {timeframe} to {trade_file}")
            except Exception as e:
                logger.error(f"Strategy {strategy_name} failed for {timeframe}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                results.append({
                    'strategy': strategy_name,
                    'timeframe': timeframe,
                    'pnl': 0,
                    'win_rate': 0,
                    'buy_trades': 0,
                    'sell_trades': 0,
                    'exit_reasons': {'sl': 0, 'tsl': 0, 'tp': 0, 'signal': 0},
                    'total_trades': 0,
                    'profitable_trades': 0,
                    'losing_trades': 0,
                    'performance_score': 0,
                    'day_wise': {},
                    'session_wise': {},
                    'regime_wise': {},
                    'volatility_wise': {},
                    'expiry_wise': {},
                    'trades': [],
                    'error': str(e)
                })

    return results, trades

def run_strategy_tester(run_id, timeframes=['1min', '3min', '5min', '15min'], expiry='2025-05-29', strike='25000'):
    """Run strategy tester for all specified timeframes in parallel using threading."""
    logger.info(f"Starting strategy testing for Run ID: {run_id}, Timeframes: {timeframes}")

    all_results = []
    all_trades = []
    max_timeframe_workers = min(4, len(timeframes))  # Limit to 4 workers or number of timeframes
    logger.info(f"Using {max_timeframe_workers} workers for timeframes")

    with ThreadPoolExecutor(max_workers=max_timeframe_workers) as executor:
        future_to_timeframe = {
            executor.submit(
                process_timeframe,
                timeframe,
                config.default_params,
                'NIFTY',
                'NSE',
                expiry,
                strike,
                run_id
            ): timeframe for timeframe in timeframes
        }
        for future in as_completed(future_to_timeframe):
            timeframe = future_to_timeframe[future]
            try:
                results, trades = future.result()
                all_results.extend(results)
                all_trades.extend(trades)
                logger.info(f"Completed processing for timeframe: {timeframe}")
            except Exception as e:
                logger.error(f"Timeframe {timeframe} failed: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")

    # Save all trades
    try:
        save_trades_to_csv(all_trades, run_id)
        logger.info(f"Saved all trades to reports/run_{run_id}/option_trade_logs/option_trades_{run_id}.csv")
    except Exception as e:
        logger.error(f"Error saving trades: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

    # Option P&L analysis
    option_trades_file_path = os.path.join("reports", f"run_{run_id}", "option_trade_logs", f"option_trades_{run_id}.csv")
    option_pnl_analysis = {}
    try:
        df_option_trades = pd.read_csv(option_trades_file_path)
        logger.info(f"Successfully loaded option trade data for P&L analysis.")

        total_option_pnl = df_option_trades['pl'].sum()
        option_pnl_analysis['total_option_pnl'] = round(total_option_pnl, 2)

        df_option_trades['exit_time'] = pd.to_datetime(df_option_trades['exit_time'])
        df_option_trades['exit_date'] = df_option_trades['exit_time'].dt.date
        pnl_by_day = df_option_trades.groupby('exit_date')['pl'].sum().to_dict()
        option_pnl_analysis['pnl_by_day'] = {str(date): round(pnl, 2) for date, pnl in pnl_by_day.items()}

        pnl_by_option_type = df_option_trades.groupby('option_type')['pl'].sum().to_dict()
        option_pnl_analysis['pnl_by_option_type'] = {option_type: round(pnl, 2) for option_type, pnl in pnl_by_option_type.items()}
    except FileNotFoundError:
        logger.warning(f"Option trade log file not found for P&L analysis at {option_trades_file_path}.")
    except Exception as e:
        logger.error(f"Error during option trade P&L analysis: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

    # Generate option reports for each strategy and timeframe
    for strategy in strategy_factories.keys():
        for tf in timeframes:
            option_trades_file_path = os.path.join(
                "reports", f"run_{run_id}", "option_trade_logs",
                f"option_trades_{run_id}__{strategy}_{tf}.csv"
            )
            if os.path.exists(option_trades_file_path):
                try:
                    generate_option_report_from_csv(option_trades_file_path, run_id)
                    logger.info(f"Generated option report for {strategy} on {tf}")
                except Exception as e:
                    logger.error(f"Error generating option report for {strategy} on {tf}: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                logger.warning(f"Option trade file not found: {option_trades_file_path}")

    # Generate strategy report
    try:
        generate_strategy_report(all_results, run_id, timeframes, option_pnl_analysis=option_pnl_analysis)
        logger.info(f"Completed strategy testing for Run ID: {run_id}")
    except Exception as e:
        logger.error(f"Error generating strategy report: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == '__main__':
    run_id = datetime.now().strftime('%Y%m%d%H%M%S')
    try:
        run_strategy_tester(run_id, expiry='2025-05-29', strike='25000')
    except Exception as e:
        logger.error(f"Strategy tester failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")