
# from dotenv import load_dotenv
# import json
# import logging
# import os
# from logging.handlers import TimedRotatingFileHandler
# from datetime import datetime

# # --- Project Root ---
# # Try to determine project root dynamically
# # Assumes config.py is in the project's root directory or one level down (e.g. in strategy_tester folder)
# def get_project_root_path():
#     current_path = os.path.abspath(os.path.dirname(__file__))
#     # If config.py is in strategy_tester, then project_root is one level up
#     if os.path.basename(current_path) == 'strategy_tester':
#         return os.path.dirname(current_path)
#     # If config.py is in root, current_path is project_root
#     return current_path

# PROJECT_ROOT = get_project_root_path()

# DEFAULT_CONFIG = {
#     "broker_name": "angelone", # or "zerodha", "fyers", etc.
#     "angel_api_key": "YOUR_API_KEY",
#     "angel_client_id": "YOUR_CLIENT_ID",
#     "angel_password": "YOUR_PASSWORD",
#     "angel_totp_secret": "YOUR_TOTP_SECRET",
#     "log_level": "INFO",
#     "log_to_console": True,
#     "log_file_template": "app_run_{run_id}.log", # Now includes run_id
#     "data_dir": "data",
#     "reports_dir": "reports",
#     "historical_data_path_template": "raw/{symbol}_{timeframe}_historical.csv", # For saving bulk history
#     "indicator_data_path_template": "datawithindicator/{symbol}_{timeframe}_with_indicators.csv",

#     "symbols_to_trade": ["NIFTY"], # For backtesting primarily
#     "timeframes_to_run": ["5min"], # For backtesting primarily
#     "start_date": "2023-01-01",
#     "end_date": "2023-12-31",
#     "initial_capital": 100000,
#     "strategies_to_run": { # Example, load actual from a JSON or define here
#         "SMACrossover": {
#             "params": {"short_window": 10, "long_window": 30},
#             "trade_options": False
#         },
#         "RSIStrategy": {
#             "params": {"rsi_period": 14, "overbought": 70, "oversold": 30},
#             "trade_options": False
#         }
#     },
#     "exchange": "NSE", # Default exchange

#     # --- Live Dry Run Specific Configs ---
#     "live_mode_enabled": False,
#     "live_symbol_to_run": "NIFTY 50", # Be specific for your broker, e.g. "NIFTY" or "NIFTY 50"
#     "live_timeframe_to_run": "1min",
#     "live_initial_history_days": 10, # Days of historical data for initial indicator calculation
#     "live_data_fetch_interval_seconds": 10, # How often to poll for new live data
#     # "live_symbol_token_map": { # Only if your broker needs explicit tokens not fetched dynamically
#     #     "NIFTY 50": "26000", 
#     #     "BANKNIFTY": "26009"
#     # },
#     "live_raw_data_dir_template": "raw/live", # Subdirectory for live raw data
#     "live_indicator_data_dir_template": "datawithindicator/live", # Subdirectory for live indicator data
#     "live_trade_log_dir_template": "live_runs", # Subdirectory in reports for live trade logs
# }
# common_params = {
#         'rsi_oversold': 35,
#         'rsi_overbought': 65,
#         'rsi_lower': 35,
#         'rsi_upper': 65,
#         'rsi_momentum': 45,
#         'rsi_exit': 35,
#         'rsi_buy_lower': 35,
#         'rsi_buy_upper': 65,
#         'rsi_sell_lower': 55,
#         'rsi_sell_upper': 75,
#         'stoch_oversold': 30,
#         'stoch_overbought': 70,
#         'period': 20,
#         'roc_threshold': 0.5,
#         'supert_period': 10,
#         'supert_multiplier': 2.0,
#         'adx_period': 14,
#         'adx_threshold': 20,
#         'atr_multiplier': 1.0,
#         'atr_threshold': 0.3,
#         'sl_atr_mult': 1.0,
#         'tp_atr_mult': 0.5,
#         'tsl_atr_mult': 0.5,
#         'risk_per_trade': 0.01,
#     'max_position_size': 1.0
#     }
# default_params = {
#         'AlphaTrend': {
#             'coeff': 0.6,
#             'ap': 10,
#             'macd_fast': 12,
#             'macd_slow': 26,
#             'macd_signal': 9,
#             'supertrend_period': 10,
#             'supertrend_multiplier': 3.0,
#             'rsi_buy_lower': 35,
#             'rsi_buy_upper': 65,
#             'rsi_sell_lower': 55,
#             'rsi_sell_upper': 75,
#             'volatility_threshold': 0.5,
#             'entry_condition_count': 4,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'EMACrossover': {
#             'short_window': 3,
#             'long_window': 8,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'RSIMACD': {
#             'rsi_period': 14,
#             'rsi_oversold': 35,
#             'rsi_overbought': 65,
#             'macd_fast': 12,
#             'macd_slow': 26,
#             'macd_signal': 9,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'SuperTrend_ADX': {
#             'supert_period': 10,
#             'supert_multiplier': 2.0,
#             'adx_period': 14,
#             'adx_threshold': 20,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'Bollinger_Bands': {
#             'bb_period': 20,
#             'bb_std': 2.0,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'Momentum_Breakout': {
#             'momentum_period': 14,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'ATR_Breakout': {
#             'atr_multiplier': 1.5,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'RSIMACDSuperTrend': {
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'ADXMACDEMA': {
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'RSIBollingerMACD': {
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'Breakout_ATR': {
#             'atr_multiplier': 1.5,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'SuperMomentum': {
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'Stochastic_Oscillator': {
#             'stoch_oversold': 20,
#             'stoch_overbought': 80,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'Support_Resistance': {
#             'period': 20,
#             'pivot_strength': 3,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'Mean_Reversion': {
#             'rsi_period': 14,
#             'rsi_oversold': 30,
#             'rsi_overbought': 70,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'Momentum': {
#             'roc_threshold': 0.5,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'RSI': {
#             'rsi_oversold': 30,
#             'rsi_overbought': 70,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'MACD_Crossover': {
#             'macd_fast': 12,
#             'macd_slow': 26,
#             'macd_signal': 9,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'ATRSuperAlphaTrend': {
#             'rsi_lower': 40,
#             'rsi_upper': 60,
#             'rsi_overbought': 70,
#             'atr_threshold': 0.5,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'RSIBollingerConfluence': {
#             'rsi_oversold': 30,
#             'rsi_overbought': 70,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'MACDStochasticTrend': {
#             'stoch_oversold': 20,
#             'stoch_overbought': 80,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'ADXVolatilityBreakout': {
#             'atr_multiplier': 1.5,
#             'adx_threshold': 25,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'TripleEMAMomentum': {
#             'rsi_momentum': 50,
#             'rsi_exit': 40,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'ExpiryDayOTMScalp': {
#             'bb_period': 20,
#             'bb_std': 2.0,
#             'volume_multiplier': 1.5,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'start_hour': 14,
#             'start_minute': 55,
#             'end_hour': 15,
#             'end_minute': 20
#         },
#         'MomentumBreakoutRSI': {
#             'ema_period': 20,
#             'rsi_period': 14,
#             'rsi_overbought': 60,
#             'rsi_oversold': 40,
#             'volume_multiplier': 1.5,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'start_hour': 14,
#             'start_minute': 55,
#             'end_hour': 15,
#             'end_minute': 20
#         },
#         'VWAPReversalScalp': {
#             'stoch_k_period': 14,
#             'stoch_d_period': 3,
#             'stoch_overbought': 70,
#             'stoch_oversold': 30,
#             'vwap_tolerance': 0.001,
#             'volume_multiplier': 1.5,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'start_hour': 14,
#             'start_minute': 50,
#             'end_hour': 15,
#             'end_minute': 20
#         },
#         'StraddleScalpHighVol': {
#             'vix_threshold': 15.0,
#             'atr_threshold': 0.002,
#             'range_threshold': 0.001,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'start_hour': 14,
#             'start_minute': 50,
#             'end_hour': 15,
#             'end_minute': 20
#         },
#         'ThreePMBollingerVolBreakout': {
#             'bb_period': 20,
#             'bb_std': 2.0,
#             'volume_multiplier': 1.5,
#             'rsi_lower': 40,
#             'rsi_upper': 60,
#             'squeeze_threshold': 0.7,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'start_hour': 14,
#             'start_minute': 50,
#             'end_hour': 15,
#             'end_minute': 20
#         },
#         'ClosingBellBreakoutScalp': {
#             'volume_multiplier': 1.5,
#             'atr_threshold': 0.002,
#             'rsi_lower': 40,
#             'rsi_upper': 60,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'start_hour': 15,
#             'start_minute': 0,
#             'end_hour': 15,
#             'end_minute': 30
#         },
#         'ExpiryDayVolatilitySpike': {
#             'vix_threshold': 15.0,
#             'volume_multiplier': 1.5,
#             'atr_threshold': 0.002,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'start_hour': 14,
#             'start_minute': 50,
#             'end_hour': 15,
#             'end_minute': 20
#         },
#         'MeanReversionSnapBack': {
#             'rsi_oversold': 30,
#             'rsi_overbought': 70,
#             'bb_period': 20,
#             'bb_std': 2.0,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'start_hour': 14,
#             'start_minute': 50,
#             'end_hour': 15,
#             'end_minute': 20
#         },
#         'IchimokuCloudStrategy_v1': {
#             'tenkan_period': 9,
#             'kijun_period': 26,
#             'senkou_b_period': 52,
#             'rsi_period': 14,
#             'rsi_overbought': 70,
#             'rsi_oversold': 30,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'KeltnerChannelBreakoutStrategy_v1': {
#             'ema_period': 20,
#             'atr_period': 10,
#             'multiplier': 1.5,
#             'rsi_period': 14,
#             'rsi_overbought': 70,
#             'rsi_oversold': 30,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#         'DonchianBreakoutStrategy_v1': {
#             'lookback_period': 20,
#             'confirm_rsi_period': 14,
#             'confirm_rsi_threshold': 50,
#             'sl_atr_mult': 1.0,
#             'tp_atr_mult': 0.5,
#             'tsl_atr_mult': 0.5
#         },
#        'BollingerStochasticCrossoverStrategy': {
#         'bb_length': 5,       # Ultra-short for maximum signals
#         'bb_std': 1.0,        # Extremely narrow bands
#         'stoch_k': 3,         # Fastest Stochastic
#         'stoch_d': 2,
#         'stoch_smooth': 1,
#         'sl_atr_mult': 2.5,   # Very wide SL
#         'tp_atr_mult': 3.0,   # Very wide TP
#         'tsl_atr_mult': 0.2   # Minimal TSL
#     }
#     }
#         # --- Credentials (Loaded from .env file or environment) ---
#     # IMPORTANT: Ensure these are set in your .env file and kept secure
# ANGELONE_API_KEY        = os.getenv("ANGELONE_API_KEY", None) # Default to None if not set
# ANGELONE_CLIENT_CODE    = os.getenv("ANGELONE_CLIENT_CODE", None)
# ANGELONE_PASSWORD_OR_PIN= os.getenv("ANGELONE_PASSWORD", None) # Renamed for clarity
# ANGELONE_TOTP_SECRET    = os.getenv("ANGELONE_TOTP_SECRET", None)
# class ConfigManager:
#     _instance = None

#     def __new__(cls, *args, **kwargs):
#         if not cls._instance:
#             cls._instance = super(ConfigManager, cls).__new__(cls)
#             cls._instance._initialized = False
#         return cls._instance

#     def __init__(self, config_file_path=None):
#         if self._initialized:
#             return
        
#         self.config = DEFAULT_CONFIG.copy()
#         self.project_root = PROJECT_ROOT # Store project root

#         if config_file_path:
#             self.config_file = os.path.join(self.project_root, config_file_path) # Ensure path is absolute
#             self._load_config_from_file()
#         else:
#             # Try to load default config.json if it exists in project root
#             default_config_path = os.path.join(self.project_root, "config.json")
#             if os.path.exists(default_config_path):
#                 self.config_file = default_config_path
#                 self._load_config_from_file()
#             else:
#                 self.config_file = None
        
#         self._initialized = True
#         logging.info(f"ConfigManager initialized. Project root: {self.project_root}")
#         if self.config_file:
#             logging.info(f"Config loaded from: {self.config_file}")
#         else:
#             logging.info("Using default configuration.")


#     def _load_config_from_file(self):
#         try:
#             if self.config_file and os.path.exists(self.config_file):
#                 with open(self.config_file, 'r') as f:
#                     file_config = json.load(f)
#                     self.config.update(file_config) # Override defaults with file config
#                     logging.info(f"Configuration loaded from {self.config_file}")
#             else:
#                 logging.warning(f"Config file {self.config_file} not found. Using default config.")
#         except Exception as e:
#             logging.error(f"Error loading config file {self.config_file}: {e}. Using default config.")

#     def get(self, key, default=None):
#         return self.config.get(key, default)

#     def set(self, key, value): # Useful for dynamic changes if needed
#         self.config[key] = value

#     def get_all(self):
#         return self.config.copy()

# # --- Global Config Instance ---
# # Initialize with a potential config file name, e.g., "config.json"
# # This allows other modules to import 'Config' and use it directly.
# _config_manager = ConfigManager("config.json") # Assuming config.json is in PROJECT_ROOT

# def get_config_manager():
#     global _config_manager
#     if not _config_manager._initialized: # Re-initialize if accessed before main init
#         _config_manager = ConfigManager("config.json")
#     return _config_manager

# def Config(): # Callable to get the manager instance
#     return get_config_manager()

# def load_config(config_file_path="config.json"): # Explicit load if needed
#     global _config_manager
#     _config_manager = ConfigManager(config_file_path)
#     return _config_manager

# # --- Path Helpers (using PROJECT_ROOT from ConfigManager) ---
# def get_data_dir():
#     cm = Config()
#     data_dir_name = cm.get("data_dir", "data")
#     return os.path.join(cm.project_root, data_dir_name)

# def get_reports_dir(run_id=None):
#     cm = Config()
#     reports_dir_name = cm.get("reports_dir", "reports")
#     base_reports_dir = os.path.join(cm.project_root, reports_dir_name)
#     if run_id:
#         return os.path.join(base_reports_dir, f"run_{run_id}")
#     return base_reports_dir

# # --- Logging Setup ---
# LOG_FILE_RUN_ID = datetime.now().strftime("%Y%m%d%H%M%S") # Global run_id for log file name for this session

# def setup_logging(log_level_str='INFO', log_file_template="app_run_{run_id}.log", run_id=None):
#     global LOG_FILE_RUN_ID
#     if run_id: # If a specific run_id is passed (e.g. for backtest runs)
#         current_run_id_for_log = run_id
#     else: # For general app logging or live mode session
#         current_run_id_for_log = LOG_FILE_RUN_ID

#     numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
#     # Create logs directory if it doesn't exist (relative to project root)
#     logs_dir = os.path.join(PROJECT_ROOT, "logs", datetime.now().strftime("%Y-%m-%d"))
#     os.makedirs(logs_dir, exist_ok=True)
    
#     log_filename = log_file_template.format(run_id=current_run_id_for_log)
#     log_filepath = os.path.join(logs_dir, log_filename)

#     # Basic config for root logger
#     logging.basicConfig(
#         level=numeric_level,
#         format='%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s',
#         handlers=[] # Remove default handlers to avoid duplicate console logs if any already set
#     )
    
#     # Get the root logger
#     logger = logging.getLogger()
#     logger.setLevel(numeric_level) # Ensure root logger level is set

#     # Clear existing handlers on the root logger to prevent duplication if setup_logging is called multiple times
#     for handler in logger.handlers[:]:
#         logger.removeHandler(handler)
#         handler.close()

#     # File Handler with rotation
#     file_handler = TimedRotatingFileHandler(log_filepath, when="midnight", interval=1, backupCount=7)
#     file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s'))
#     logger.addHandler(file_handler)

#     # Console Handler
#     if Config().get("log_to_console", True):
#         console_handler = logging.StreamHandler()
#         console_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
#         # console_handler.setLevel(numeric_level) # Console handler respects root logger level or can have its own
#         logger.addHandler(console_handler)
    
#     logging.info(f"Logging setup complete. Level: {log_level_str}. File: {log_filepath}")


# # --- Expose methods for easier access ---
# # This makes it so you can do `from config import Config` and then `Config().get(...)`
# # Or, for more direct access to underlying manager (if needed, though Config() is preferred):
# # `from config import get_config_manager` then `get_config_manager().get(...)`


# # Auto-initialize logging when config module is imported for the first time in an app run
# # This ensures logging is set up early.
# # setup_logging(Config().get('log_level'), Config().get('log_file_template'))
# # Call setup_logging explicitly in your main app entry point (e.g. run_strategy_tester.py)
# # after loading any specific run_id if necessary.
# config.py
from dotenv import load_dotenv
import logging
import os
import uuid
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# --- Project Root ---
def get_project_root_path():
    current_path = os.path.abspath(os.path.dirname(__file__))
    if os.path.basename(current_path) == 'strategy_tester':
        return os.path.dirname(current_path)
    return current_path

PROJECT_ROOT = get_project_root_path()

# Fetch credentials from environment variables
ANGELONE_API_KEY = os.getenv("ANGELONE_API_KEY", None)
ANGELONE_CLIENT_CODE = os.getenv("ANGELONE_CLIENT_CODE", None)
ANGELONE_PASSWORD_OR_PIN = os.getenv("ANGELONE_PASSWORD", None)
ANGELONE_TOTP_SECRET = os.getenv("ANGELONE_TOTP_SECRET", None)

# Warn if credentials are missing
if not all([ANGELONE_API_KEY, ANGELONE_CLIENT_CODE, ANGELONE_PASSWORD_OR_PIN, ANGELONE_TOTP_SECRET]):
    logging.warning("One or more AngelOne credentials are missing. Please set ANGELONE_API_KEY, ANGELONE_CLIENT_CODE, ANGELONE_PASSWORD, and ANGELONE_TOTP_SECRET in your .env file.")

DEFAULT_CONFIG = {
    "broker_name": "angelone",
    "angel_api_key": ANGELONE_API_KEY,
    "angel_client_id": ANGELONE_CLIENT_CODE,
    "angel_password": ANGELONE_PASSWORD_OR_PIN,
    "angel_totp_secret": ANGELONE_TOTP_SECRET,
    "angel_session_expiry_hours": 6,
    "angel_instrument_list_url": "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
    "log_level": "INFO",
    "log_to_console": True,
    "log_file_template": "app_run_{run_id}.log",
    "data_dir": "data",
    "reports_dir": "reports",
    "historical_data_path_template": "raw/{symbol}_{timeframe}_historical.csv",
    "indicator_data_path_template": "datawithindicator/{symbol}_{timeframe}_with_indicators.csv",
    "live_raw_data_dir_template": "raw/live",
    "live_indicator_data_dir_template": "datawithindicator/live",
    "live_trade_log_dir_template": "live_runs",
    "symbols_to_trade": ["NIFTY"],
    "timeframes_to_run": ["5min"],
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 100000,
    "exchange": "NSE",
    "live_mode_enabled": False,
    "live_symbol_to_run": "NIFTY 50",
    "live_timeframe_to_run": "5min",
    "live_initial_history_days": 90,
    "live_data_fetch_interval_seconds": 10,
}

# Strategy parameters (kept as-is for your broader project, but not used in current task)
common_params = {
    'rsi_oversold': 35,
    'rsi_overbought': 65,
    'rsi_lower': 35,
    'rsi_upper': 65,
    'rsi_momentum': 45,
    'rsi_exit': 35,
    'rsi_buy_lower': 35,
    'rsi_buy_upper': 65,
    'rsi_sell_lower': 55,
    'rsi_sell_upper': 75,
    'stoch_oversold': 30,
    'stoch_overbought': 70,
    'period': 20,
    'roc_threshold': 0.5,
    'supert_period': 10,
    'supert_multiplier': 2.0,
    'adx_period': 14,
    'adx_threshold': 20,
    'atr_multiplier': 1.0,
    'atr_threshold': 0.3,
    'sl_atr_mult': 1.0,
    'tp_atr_mult': 0.5,
    'tsl_atr_mult': 0.5,
    'risk_per_trade': 0.01,
    'max_position_size': 1.0
}

default_params = {
    'AlphaTrend': {
        'coeff': 0.6,
        'ap': 10,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'supertrend_period': 10,
        'supertrend_multiplier': 3.0,
        'rsi_buy_lower': 35,
        'rsi_buy_upper': 65,
        'rsi_sell_lower': 55,
        'rsi_sell_upper': 75,
        'volatility_threshold': 0.5,
        'entry_condition_count': 4,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'EMACrossover': {
        'short_window': 3,
        'long_window': 8,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'RSIMACD': {
        'rsi_period': 14,
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'SuperTrend_ADX': {
        'supert_period': 10,
        'supert_multiplier': 2.0,
        'adx_period': 14,
        'adx_threshold': 20,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'Bollinger_Bands': {
        'bb_period': 20,
        'bb_std': 2.0,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'Momentum_Breakout': {
        'momentum_period': 14,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'ATR_Breakout': {
        'atr_multiplier': 1.5,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'RSIMACDSuperTrend': {
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'ADXMACDEMA': {
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'RSIBollingerMACD': {
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'Breakout_ATR': {
        'atr_multiplier': 1.5,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'SuperMomentum': {
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'Stochastic_Oscillator': {
        'stoch_oversold': 20,
        'stoch_overbought': 80,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'Support_Resistance': {
        'period': 20,
        'pivot_strength': 3,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'Mean_Reversion': {
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'Momentum': {
        'roc_threshold': 0.5,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'RSI': {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'MACD_Crossover': {
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'ATRSuperAlphaTrend': {
        'rsi_lower': 40,
        'rsi_upper': 60,
        'rsi_overbought': 70,
        'atr_threshold': 0.5,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'RSIBollingerConfluence': {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'MACDStochasticTrend': {
        'stoch_oversold': 20,
        'stoch_overbought': 80,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'ADXVolatilityBreakout': {
        'atr_multiplier': 1.5,
        'adx_threshold': 25,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'TripleEMAMomentum': {
        'rsi_momentum': 50,
        'rsi_exit': 40,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'ExpiryDayOTMScalp': {
        'bb_period': 20,
        'bb_std': 2.0,
        'volume_multiplier': 1.5,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'start_hour': 14,
        'start_minute': 55,
        'end_hour': 15,
        'end_minute': 20
    },
    'MomentumBreakoutRSI': {
        'ema_period': 20,
        'rsi_period': 14,
        'rsi_overbought': 60,
        'rsi_oversold': 40,
        'volume_multiplier': 1.5,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'start_hour': 14,
        'start_minute': 55,
        'end_hour': 15,
        'end_minute': 20
    },
    'VWAPReversalScalp': {
        'stoch_k_period': 14,
        'stoch_d_period': 3,
        'stoch_overbought': 70,
        'stoch_oversold': 30,
        'vwap_tolerance': 0.001,
        'volume_multiplier': 1.5,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'start_hour': 14,
        'start_minute': 50,
        'end_hour': 15,
        'end_minute': 20
    },
    'StraddleScalpHighVol': {
        'vix_threshold': 15.0,
        'atr_threshold': 0.002,
        'range_threshold': 0.001,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'start_hour': 14,
        'start_minute': 50,
        'end_hour': 15,
        'end_minute': 20
    },
    'ThreePMBollingerVolBreakout': {
        'bb_period': 20,
        'bb_std': 2.0,
        'volume_multiplier': 1.5,
        'rsi_lower': 40,
        'rsi_upper': 60,
        'squeeze_threshold': 0.7,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'start_hour': 14,
        'start_minute': 50,
        'end_hour': 15,
        'end_minute': 20
    },
    'ClosingBellBreakoutScalp': {
        'volume_multiplier': 1.5,
        'atr_threshold': 0.002,
        'rsi_lower': 40,
        'rsi_upper': 60,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'start_hour': 15,
        'start_minute': 0,
        'end_hour': 15,
        'end_minute': 30
    },
    'ExpiryDayVolatilitySpike': {
        'vix_threshold': 15.0,
        'volume_multiplier': 1.5,
        'atr_threshold': 0.002,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'start_hour': 14,
        'start_minute': 50,
        'end_hour': 15,
        'end_minute': 20
    },
    'MeanReversionSnapBack': {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'bb_period': 20,
        'bb_std': 2.0,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'start_hour': 14,
        'start_minute': 50,
        'end_hour': 15,
        'end_minute': 20
    },
    'IchimokuCloudStrategy_v1': {
        'tenkan_period': 9,
        'kijun_period': 26,
        'senkou_b_period': 52,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'KeltnerChannelBreakoutStrategy_v1': {
        'ema_period': 20,
        'atr_period': 10,
        'multiplier': 1.5,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'DonchianBreakoutStrategy_v1': {
        'lookback_period': 20,
        'confirm_rsi_period': 14,
        'confirm_rsi_threshold': 50,
        'sl_atr_mult': 1.0,
        'tp_atr_mult': 0.5,
        'tsl_atr_mult': 0.5
    },
    'BollingerStochasticCrossoverStrategy': {
        'bb_length': 5,
        'bb_std': 1.0,
        'stoch_k': 3,
        'stoch_d': 2,
        'stoch_smooth': 1,
        'sl_atr_mult': 2.5,
        'tp_atr_mult': 3.0,
        'tsl_atr_mult': 0.2
    }
}

class Config:
    def __init__(self):
        self._config = DEFAULT_CONFIG.copy()
        self.run_id = str(uuid.uuid4())

    def get(self, key, default=None):
        return self._config.get(key, default)

    def set(self, key, value):
        self._config[key] = value

    def get_all(self):
        return self._config.copy()

def get_project_root():
    return PROJECT_ROOT

def get_data_dir():
    data_dir = os.path.join(PROJECT_ROOT, DEFAULT_CONFIG["data_dir"])
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def get_reports_dir(run_id=None):
    reports_dir = os.path.join(PROJECT_ROOT, DEFAULT_CONFIG["reports_dir"])
    if run_id:
        reports_dir = os.path.join(reports_dir, f"run_{run_id}")
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir

def setup_logging(log_level_str='INFO', log_file_template="app_run_{run_id}.log", run_id=None):
    import logging
    log_dir = os.path.join(get_data_dir(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    run_id = run_id or str(uuid.uuid4())
    log_file = os.path.join(log_dir, log_file_template.format(run_id=run_id))

    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, log_level_str.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    print(f"✅ Logging setup complete. Level: {log_level_str}. File: {log_file}")

def load_config():
    global config_instance  # move this to the top before using config_instance

    config_instance = Config()
    run_id = config_instance.run_id

    numeric_level = getattr(logging, DEFAULT_CONFIG["log_level"].upper(), logging.INFO)

    # Save logs in data/logs/ directory
    logs_dir = os.path.join(get_data_dir(), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_filename = DEFAULT_CONFIG["log_file_template"].format(run_id=run_id)
    log_filepath = os.path.join(logs_dir, log_filename)

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler() if DEFAULT_CONFIG["log_to_console"] else logging.NullHandler()
        ]
    )

    logging.info(f"✅ Logging setup complete. Level: {DEFAULT_CONFIG['log_level']}. File: {log_filepath}")
    return config_instance