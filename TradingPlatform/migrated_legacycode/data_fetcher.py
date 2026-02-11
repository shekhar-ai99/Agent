# data_fetcher.py corrected for Python 3.9 type hints and Angel One integration

import pandas as pd
import os
import logging
import datetime
import time
import pytz
import pyotp
import requests # Import requests for fallback instrument list download
from SmartApi import SmartConnect
# Import Union and Optional for type hinting compatibility
from typing import Union, Optional, Dict, Any, List
# Corrected import using direct path - sys.path must be modified in main.py
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def map_interval_to_smartapi(interval_str: str) -> str:
    """ Maps user-friendly interval strings to SmartAPI required strings. """
    mapping = {
        '1minute': 'ONE_MINUTE', '3minute': 'THREE_MINUTE', '5minute': 'FIVE_MINUTE',
        '10minute': 'TEN_MINUTE', '15minute': 'FIFTEEN_MINUTE', '30minute': 'THIRTY_MINUTE',
        '1hour': 'SIXTY_MINUTE', '1day': 'ONE_DAY',
        '1week': 'ONE_DAY', '1month': 'ONE_DAY', # Fetch daily and resample later if needed
    }
    api_interval = mapping.get(interval_str)
    if not api_interval:
        logger.warning(f"Interval '{interval_str}' unknown, defaulting to 'ONE_DAY'.")
        return 'ONE_DAY'
    if interval_str in ['1week', '1month']:
        logger.warning(f"Fetching 'ONE_DAY' for '{interval_str}'. Resample needed.")
    return api_interval

# --- Instrument Token Handling ---
# Global variable to cache instrument list
# Use Python 3.9 compatible type hint Optional[pd.DataFrame] instead of pd.DataFrame | None
instrument_list_df: Optional[pd.DataFrame] = None

def fetch_instrument_list(smart_api: SmartConnect) -> Optional[pd.DataFrame]: # Use Optional
    """ Fetches and caches the full instrument list from Angel One. """
    global instrument_list_df
    if instrument_list_df is not None:
        return instrument_list_df
    try:
        logger.info("Fetching instrument list from Angel One...")
        # Try the library method first (assuming get_instrument_list or similar exists)
        # If it fails with AttributeError or returns invalid data, try direct download
        response = None
        try:
            # Use getScripMaster based on user's working script structure
            response = smart_api.getScripMaster()
            logger.info("Using smart_api.getScripMaster() for instrument list.")
        except AttributeError:
            logger.warning("'getScripMaster()' method not found. Check smartapi-python version or documentation.")
            # Fallback: Try get_instrument_list if getScripMaster failed
            try:
                 response = smart_api.get_instrument_list()
                 logger.info("Using smart_api.get_instrument_list() for instrument list.")
            except AttributeError:
                 logger.warning("'get_instrument_list()' method also not found. Trying direct JSON download.")
                 response = None # Ensure response is None to trigger fallback

        data_list = None
        # Process response if library call seemed successful
        if response and isinstance(response, dict) and response.get("status") and isinstance(response.get("data"), list):
             data_list = response["data"]
        elif isinstance(response, list): # Handle case where method returns list directly
            data_list = response

        # --- Fallback: Download JSON directly if method failed or returned unexpected data ---
        if not data_list:
             logger.warning("Library method failed or returned invalid data. Trying direct JSON download...")
             instrument_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
             headers = {'User-Agent': 'Mozilla/5.0'} # Some sites require a user agent
             http_response = requests.get(instrument_url, headers=headers, timeout=10) # Added timeout
             http_response.raise_for_status() # Raise error for bad status codes
             data_list = http_response.json()
             if not isinstance(data_list, list):
                  logger.error(f"Direct JSON download did not return a list. Content: {str(data_list)[:200]}...")
                  return None

        # --- Process the data list ---
        if data_list:
             instrument_list_df = pd.DataFrame(data_list)
             required_cols = ['token', 'symbol', 'exch_seg'] # Minimal required columns
             if not all(col in instrument_list_df.columns for col in required_cols):
                 logger.error(f"Instrument list missing required columns (Need token, symbol, exch_seg). Found: {instrument_list_df.columns}")
                 instrument_list_df = None; return None
             logger.info(f"Instrument list fetched/cached successfully: {len(instrument_list_df)} instruments.")
             return instrument_list_df
        else:
            logger.error("Failed to fetch instrument list via all methods.")
            return None

    except Exception as e:
        logger.error(f"General exception fetching instrument list: {e}", exc_info=True)
        instrument_list_df = None # Ensure cache is cleared on error
        return None


def get_instrument_token(smart_api: SmartConnect, ticker: str, exchange: str = 'NSE') -> str:
    """ Looks up instrument token from the cached list based on ticker and exchange. """
    global instrument_list_df
    if instrument_list_df is None:
        instrument_list_df = fetch_instrument_list(smart_api)
        if instrument_list_df is None: raise ValueError("Could not fetch instrument list to find token.")

    ticker_upper = ticker.strip().upper(); exchange_upper = exchange.strip().upper()
    search_symbol = ticker_upper

    # Handle common Index name variations
    if exchange_upper == 'NSE' and ticker_upper in ['NIFTY 50', '^NSEI', 'NIFTY']: search_symbol = 'NIFTY 50'
    elif exchange_upper == 'NSE' and ticker_upper in ['BANKNIFTY', 'NIFTY BANK']: search_symbol = 'BANKNIFTY'
    # Add other index mappings if needed

    logger.debug(f"Searching token: Symbol='{search_symbol}', Exchange='{exchange_upper}'")

    # Ensure required columns exist before filtering
    if not all(col in instrument_list_df.columns for col in ['symbol', 'exch_seg', 'token']):
         raise ValueError("Instrument list DataFrame is missing required columns for token lookup.")

    # Filter by symbol and exchange segment
    filtered_df = instrument_list_df[
        (instrument_list_df['symbol'].str.upper() == search_symbol) &
        (instrument_list_df['exch_seg'].str.upper() == exchange_upper)
    ]

    # Fallback: If symbol search fails, try searching by 'name' column if it exists
    if filtered_df.empty and 'name' in instrument_list_df.columns:
         logger.debug(f"Symbol search failed for '{search_symbol}', trying search by name.")
         filtered_df = instrument_list_df[
            (instrument_list_df['name'].str.upper() == search_symbol) &
            (instrument_list_df['exch_seg'].str.upper() == exchange_upper)
         ]

    # Check results
    if filtered_df.empty:
        logger.error(f"Instrument token not found for Symbol='{ticker}', Exchange='{exchange}'. Check symbol/exchange and instrument list content.")
        raise ValueError(f"Token not found for Ticker: '{ticker}', Exchange: '{exchange}'")
    elif len(filtered_df) > 1:
        logger.warning(f"Multiple tokens found for {ticker}@{exchange}. Using first found: {filtered_df.iloc[0]['token']}. Matches:\n{filtered_df[['token', 'symbol', 'name', 'exch_seg']]}")

    token = str(filtered_df.iloc[0]['token']) # Ensure token is string
    logger.info(f"Found token '{token}' for {ticker}@{exchange}")
    return token

# --- Session Management ---
# Corrected Type Hint for Python 3.9
smart_api_obj: Optional[SmartConnect] = None
session_expiry_time: Optional[datetime.datetime] = None
MAX_RETRIES = 3
RETRY_DELAY = 5

def get_smartapi_session() -> SmartConnect: # Return should be SmartConnect on success
    """ Creates or returns an existing valid SmartAPI session object. """
    global smart_api_obj, session_expiry_time
    # Check if session exists and is notionally not expired
    if smart_api_obj and session_expiry_time and session_expiry_time > datetime.datetime.now():
        logger.info("Using existing SmartAPI session.")
        # TODO: Add a real validity check if possible (e.g., getProfile)
        return smart_api_obj
    elif smart_api_obj:
         logger.info("Existing SmartAPI session likely expired. Re-authenticating.")
         smart_api_obj = None # Force re-login


    logger.info("Attempting to initialize and authenticate SmartAPI...")
    try:
        api_key=config.ANGELONE_API_KEY; client_code=config.ANGELONE_CLIENT_CODE; password=config.ANGELONE_PASSWORD_OR_PIN; totp_secret=config.ANGELONE_TOTP_SECRET
        if not all([api_key, client_code, password, totp_secret]): raise ValueError("Missing Angel One credentials")
        smart_api = SmartConnect(api_key)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Auth attempt {attempt}/{MAX_RETRIES}")
                totp = pyotp.TOTP(totp_secret).now()
                resp = smart_api.generateSession(client_code, password, totp)
                if resp.get("status") and resp.get("data") and resp["data"].get("jwtToken"):
                    logger.info("✅ SmartAPI login successful")
                    smart_api_obj = smart_api
                    # Set expiry based on response if possible, else default (e.g., 6 hours)
                    # Note: Actual expiry depends on Angel One policy
                    session_expiry_time = datetime.datetime.now() + datetime.timedelta(hours=6)
                    fetch_instrument_list(smart_api_obj) # Pre-fetch/cache instrument list
                    return smart_api_obj # Return the successful object
                logger.warning(f"Login attempt {attempt} failed: {resp.get('message', 'Unknown error')}")
            except Exception as e: logger.error(f"Login attempt {attempt} exception: {e}")
            if attempt < MAX_RETRIES: time.sleep(RETRY_DELAY)
        raise ConnectionError(f"Could not login after {MAX_RETRIES} attempts")
    except Exception as e:
        logger.error(f"API initialization failed: {e}", exc_info=True)
        smart_api_obj = None; session_expiry_time = None
        raise ConnectionError(f"Could not connect/authenticate Angel One: {e}")

# --- Main Fetch Function ---
def fetch_data(
    source: str,
    interval: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    ticker: str = 'NIFTY 50',
    exchange: str = 'NSE'
) -> pd.DataFrame:
    logger.info(f"Fetching data: source={source} interval={interval} ticker={ticker} exch={exchange} from={start_date} to={end_date}")
   
    if source.lower() == 'offline':
        # --- OFFLINE LOGIC ---
        if not hasattr(config, 'OFFLINE_DATA_PATH') or not hasattr(config, 'OFFLINE_FILES'): raise ValueError("Offline config missing")
        fname = config.OFFLINE_FILES.get(interval)
        if not fname: raise ValueError(f"No offline file for interval '{interval}'")
        path = os.path.join(config.OFFLINE_DATA_PATH, fname)
        if not os.path.exists(path): raise FileNotFoundError(f"Missing offline file: {path}")
        try:
            df = pd.read_csv(path);
            if 'datetime' not in df.columns: raise ValueError("CSV lacks 'datetime'")
            try: df['datetime'] = pd.to_datetime(df['datetime'])
            except Exception as e: raise ValueError(f"Could not parse 'datetime' in {path}: {e}")
            is_localized = pd.api.types.is_datetime64_any_dtype(df['datetime']) and df['datetime'].dt.tz is not None
            if not is_localized:
                logger.warning(f"Offline data {path} is naive. Assuming Asia/Kolkata.")
                try: df['datetime'] = df['datetime'].dt.tz_localize('Asia/Kolkata', ambiguous='infer')
                except Exception as tz_err: logger.error(f"Failed to localize {path}: {tz_err}.")
            df.set_index('datetime', inplace=True); df = df.rename(columns=str.lower)
            current_tz = df.index.tz
            if start_date: start_dt_naive=pd.to_datetime(start_date); start_dt=start_dt_naive.tz_localize(current_tz) if current_tz else start_dt_naive; df=df[df.index >= start_dt]
            if end_date: end_dt_naive=pd.to_datetime(end_date)+pd.Timedelta(days=1); end_dt=end_dt_naive.tz_localize(current_tz) if current_tz else end_dt_naive; df=df[df.index < end_dt]
            logger.info(f"✅ Offline data loaded: {path} → {df.shape}"); return df
        except Exception as e: logger.error(f"Error loading offline {path}: {e}"); raise

    elif source.lower() == 'online':
            try:
                sc = get_smartapi_session()
                token = get_instrument_token(sc, ticker, exchange)
                api_int = map_interval_to_smartapi(interval)
                ist = pytz.timezone("Asia/Kolkata")
                now = datetime.datetime.now(ist)
                # Parse dates robustly
                if end_date:
                    end_dt = pd.to_datetime(end_date).tz_localize(ist).replace(hour=15, minute=30)
                else:
                    end_dt = now.replace(hour=15, minute=30) if now.time() >= datetime.time(9, 15) else (now - datetime.timedelta(days=1)).replace(hour=15, minute=30, tzinfo=ist)
                end_dt = min(end_dt, now)
                if start_date:
                    start_dt = pd.to_datetime(start_date).tz_localize(ist).replace(hour=9, minute=15)
                else:
                    start_dt = end_dt - datetime.timedelta(days=30)
                    start_dt = start_dt.replace(hour=9, minute=15)
                if start_dt >= end_dt:
                    logger.error(f"Start date {start_dt} not before end date {end_dt}")
                    return pd.DataFrame()
                # Dynamic chunking based on interval
                if api_int == "ONE_DAY":
                    chunk_delta = datetime.timedelta(days=200)
                elif api_int == "SIXTY_MINUTE":
                    chunk_delta = datetime.timedelta(days=50)
                elif api_int in ["ONE_MINUTE", "THREE_MINUTE", "FIVE_MINUTE"]:
                    chunk_delta = datetime.timedelta(days=5)  # Smaller chunks for high-frequency
                else:
                    chunk_delta = datetime.timedelta(days=10)
                all_candles = []
                chunk_start_dt = start_dt
                while chunk_start_dt < end_dt:
                    chunk_end_dt = min(chunk_start_dt + chunk_delta, end_dt)
                    from_date_chunk = chunk_start_dt.strftime("%Y-%m-%d %H:%M")
                    to_date_chunk = chunk_end_dt.strftime("%Y-%m-%d %H:%M")
                    if from_date_chunk >= to_date_chunk:
                        break
                    logger.info(f"▶️ Fetching chunk: {token} {exchange} {api_int} From='{from_date_chunk}', To='{to_date_chunk}'")
                    params = {"exchange": exchange.upper(), "symboltoken": token, "interval": api_int, "fromdate": from_date_chunk, "todate": to_date_chunk}
                    resp_data = None
                    for attempt in range(1, MAX_RETRIES + 1):
                        try:
                            resp = sc.getCandleData(params)
                            if resp and resp.get("status") and resp.get("data") is not None:
                                resp_data = resp.get("data", [])
                                break
                            logger.warning(f"Chunk {attempt} failed: {resp.get('message', 'No status/data') if resp else 'No response'}")
                        except Exception as ex:
                            logger.error(f"Chunk {attempt} exception: {ex}")
                        if attempt == MAX_RETRIES:
                            raise ConnectionError(f"Failed fetch chunk: {resp.get('message') if resp else 'Exception occurred'}")
                        time.sleep(RETRY_DELAY / 2.0)
                    if resp_data:
                        all_candles.extend(resp_data)
                    chunk_start_dt = chunk_end_dt
                    time.sleep(0.5)
                if not all_candles:
                    logger.warning("No online data returned")
                    return pd.DataFrame()
                df = pd.DataFrame(all_candles, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.drop_duplicates(subset=['datetime'], keep='first').set_index('datetime').sort_index()
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                df.dropna(inplace=True)
                logger.info(f"✅ Online data processed: {df.shape}")
                if df.index.tz is None:
                    logger.warning("Angel One data index naive, localizing to Asia/Kolkata")
                    df = df.tz_localize('Asia/Kolkata', ambiguous='infer')
                return df
            except Exception as e:
                logger.error(f"Error processing online request: {e}", exc_info=True)
                raise RuntimeError(f"Online fetch failed: {e}")
    else:
            raise ValueError(f"Unknown source '{source}'")
# print statement to confirm update (optional)
# print("File 'data_fetcher.py' updated with Python 3.9 compatible type hints.")