import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class Config:
    """ Application configuration class """

    # --- Credentials (Loaded from .env file or environment) ---
    # IMPORTANT: Ensure these are set in your .env file and kept secure
    ANGELONE_API_KEY        = os.getenv("ANGELONE_API_KEY", None) # Default to None if not set
    ANGELONE_CLIENT_CODE    = os.getenv("ANGELONE_CLIENT_CODE", None)
    ANGELONE_PASSWORD_OR_PIN= os.getenv("ANGELONE_PASSWORD", None) # Renamed for clarity
    ANGELONE_TOTP_SECRET    = os.getenv("ANGELONE_TOTP_SECRET", None)
 # — Simulation defaults —
  
    OFFLINE_DATA_PATH       = os.getenv("OFFLINE_DATA_PATH", "data") # Path relative to project root
    # Mapping from user interval selection to filename
    OFFLINE_FILES = {
        "1minute":   "nifty_historical_data_1min.csv",
        "3minute":   "nifty_historical_data_3min.csv",
        "5minute":   "nifty_historical_data_5min.csv",
        "10minute":  "nifty_historical_data_10min.csv",
        "15minute":  "nifty_historical_data_15min.csv",
        "30minute":  "nifty_historical_data_30min.csv",
        "1hour":     "nifty_historical_data_1h.csv",
        "1day":      "nifty_historical_data_1d.csv",
    }
    # --- Simulation defaults ---
    DEFAULT_SL_PCT       = float(os.getenv("DEFAULT_SL_PCT", 1.0))
    DEFAULT_TRAIL_PCT    = float(os.getenv("DEFAULT_TRAIL_PCT", 0.5))
    COMMISSION_PER_TRADE = float(os.getenv("COMMISSION_PER_TRADE", 0.0))
    SLIPPAGE_PER_TRADE   = float(os.getenv("SLIPPAGE_PER_TRADE", 0.0))
    INITIAL_CAPITAL      = float(os.getenv('INITIAL_CAPITAL', 100000)) # Moved here

    # --- Indicator Defaults (Add these) ---
    INDICATOR_SMA_PERIODS = (10, 20, 50) # Example values, adjust as needed
    INDICATOR_EMA_PERIODS = (9, 21, 50)  # Example values, adjust as needed
    INDICATOR_RSI_PERIOD  = 14
    INDICATOR_ATR_PERIOD  = 14
    INDICATOR_BBANDS_PERIOD= 20
    INDICATOR_STOCH_PERIOD= 14
    INDICATOR_DMI_LENGTH  = 14
    INDICATOR_ADX_SMOOTHING= 14
    INDICATOR_VOL_MA_LEN  = 20
    INDICATOR_MACD_FAST   = 12
    INDICATOR_MACD_SLOW   = 26
    INDICATOR_MACD_SIGNAL = 9
    VWAP_ENABLED          = False      # Example, set based on your needs
    VWAP_TYPE             = 'typical'  # Example, set based on your needs

    # --- Offline Data Config ---
    # ... (keep existing offline data lines) ...
    FLASK_DEBUG = True # Keep or adjust as needed
        # Add more mappings here if you have more CSV files
        # e.g., "1day": os.getenv("OFFLINE_FILE_1DAY", "nifty_historical_data_1day.csv"),
    

    # --- Other Potential Config ---
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 100000))


# Create a single instance of the Config class for the application to use
config = Config()

# Example Check (Optional): Print a warning if essential credentials are missing
if not all([config.ANGELONE_API_KEY, config.ANGELONE_CLIENT_CODE, config.ANGELONE_PASSWORD_OR_PIN, config.ANGELONE_TOTP_SECRET]):
     print("\nWARNING: Angel One API credentials not fully set in environment/.env file. Online data fetching will fail.\n")


print("File 'config.py' updated with Config class structure.")