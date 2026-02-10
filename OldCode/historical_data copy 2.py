import sys
import traceback
import pandas as pd
import os
from SmartApi import SmartConnect
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz
import pyotp
import logging
from pathlib import Path
import time
import argparse

from compute_indicators import compute_indicators
# Import AngelOneAPI and BrokerClient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from angel_one.angel_one_api import AngelOneAPI
# Import your Config class and load_config function
from config import Config, load_config
from angel_one.angel_one_instrument_manager import InstrumentManager # You can keep this import

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
# Initialize AngelOneAPI to get InstrumentManager
config = load_config()
api_client = AngelOneAPI(config)
# Use the InstrumentManager instance from api_client
instrument_manager = api_client.get_instrument_manager()
smart_api = api_client.get_smart_connect_object()



class HistoricalDataFetcher:
    def __init__(self, interval: str):
        self.NIFTY_SYMBOL = "99926000"  # Token for NIFTY 50
        self.HISTORICAL_INTERVAL = interval
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 5  # seconds

    def initialize_api(self) -> SmartConnect:
        try:
            api_key = os.getenv("ANGELONE_API_KEY")
            client_code = os.getenv("ANGELONE_CLIENT_CODE")
            password = os.getenv("ANGELONE_PASSWORD")
            totp_secret = os.getenv("ANGELONE_TOTP_SECRET")

            if not all([api_key, client_code, password, totp_secret]):
                raise ValueError("Missing required environment variables")

            smart_api = SmartConnect.SmartConnect(api_key)
            for attempt in range(1, self.MAX_RETRIES + 1):
                try:
                    logger.info(f"Authentication attempt {attempt}/{self.MAX_RETRIES}")
                    totp = pyotp.TOTP(totp_secret).now()
                    login_data = smart_api.generateSession(client_code, password, totp)
                    if login_data.get("status"):
                        logger.info("API authentication successful")
                        return smart_api
                    error_msg = login_data.get("message", "Unknown error")
                    logger.warning(f"Login failed: {error_msg}")
                except Exception as e:
                    logger.error(f"Login error: {str(e)}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY)
            raise Exception(f"Max login attempts ({self.MAX_RETRIES}) reached")
        except Exception as e:
            logger.error(f"API initialization failed: {str(e)}")
            raise

    def fetch_historical_data(self, smart_api: SmartConnect, days: int = 30) -> pd.DataFrame: # Changed default days to 60
        try:
            ist = pytz.timezone('Asia/Kolkata')
            now = datetime.now(ist)
            start_time = now - timedelta(days=days)

            params = {
                "exchange": "NSE",
                "symboltoken": self.NIFTY_SYMBOL,
                "interval": self.HISTORICAL_INTERVAL,
                "fromdate": start_time.strftime('%Y-%m-%d %H:%M'),
                "todate": now.strftime('%Y-%m-%d %H:%M')
            }

            logger.info(f"Fetching {self.HISTORICAL_INTERVAL} data for NIFTY with params: {params}")
            response = smart_api.getCandleData(params)
            if not response["status"]:
                raise Exception(f"API response error: {response['message']}")
            if not response.get("data"):
                raise Exception("No data returned from API")

            df = pd.DataFrame(
                response["data"],
                columns=["datetime", "open", "high", "low", "close", "volume"]
            )
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)

            for col in ["open", "high", "low", "close"]:
                if (df[col] <= 0).any():
                    raise ValueError(f"Invalid {col} prices in data")

            logger.info(f"Fetched {len(df)} records for NIFTY at interval {self.HISTORICAL_INTERVAL}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {self.HISTORICAL_INTERVAL} data for NIFTY: {str(e)}")
            raise

    def fetch_historical_candles(
        self,
        smart_api,
        symboltoken: str,
        exchange: str,
        interval: str,
        fromdate: str,
        todate: str,
        instrument_name: str = "Instrument" # Added instrument name for logging
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for any instrument (index, stock, option) using AngelOne SmartAPI.
        """
        import pandas as pd
        import logging
        logger = logging.getLogger(__name__)

        params = {
            "exchange": exchange,
            "symboltoken": symboltoken,
            "interval": interval,
            "fromdate": fromdate,
            "todate": todate
        }
        logger.info(f"Fetching {interval} candles from {fromdate} to {todate} for {instrument_name} token {symboltoken} ({exchange})")
        response = smart_api.getCandleData(params)
        if not response.get("status"):
            raise Exception(f"API error: {response.get('message', 'Unknown error')}")
        if not response.get("data"):
            raise Exception("No data returned from API")

        df = pd.DataFrame(
            response["data"],
            columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)

        # Sanity check: all OHLC should be > 0
        for col in ["open", "high", "low", "close"]:
            if (df[col] <= 0).any():
                raise ValueError(f"Invalid {col} prices in data for {symboltoken}")

        logger.info(f"Fetched {len(df)} records for {instrument_name} token {symboltoken} ({exchange})")
        return df
# def main(output: str, duration_months: int = 2, expiry: str = "2025-05-29", strike: str = "24000"):
#     try:
#         output_path = Path(output)
#         if output_path.is_dir():
#             base_dir = output_path
#             base_name = "historical_data"
#             ext = ".csv"
#         else:
#             base_dir = output_path.parent
#             base_name = output_path.stem
#             ext = output_path.suffix if output_path.suffix else ".csv"

#         base_dir.mkdir(parents=True, exist_ok=True)

#         ist = pytz.timezone('Asia/Kolkata')
#         now = datetime.now(ist)
#         start_time = now - timedelta(days=duration_months * 30)

#         intervals = {
#             "FIVE_MINUTE": "5min"
#         }

#         # Initialize AngelOneAPI to get InstrumentManager
#         config = load_config()
#         api_client = AngelOneAPI(config)
#         instrument_manager = api_client.get_instrument_manager()
#         smart_api = api_client.get_smart_connect_object()

#         if instrument_manager is None or smart_api is None:
#             logger.error("Failed to initialize AngelOne API or InstrumentManager.")
#             return

#         for interval_key, suffix in intervals.items():
#             logger.info(f"Starting fetch for interval: {interval_key}")
#             fetcher = HistoricalDataFetcher(interval=interval_key)

#             # Fetch NIFTY data
#             nifty_df = fetcher.fetch_historical_data(smart_api, days=duration_months * 30)
#             enriched_nifty_path = base_dir / f"nifty_{suffix}_with_indicators.csv"
#             compute_indicators(nifty_df, output_path=enriched_nifty_path)
#             logger.info(f"Enriched NIFTY {interval_key} data saved to {enriched_nifty_path}")

#             # --- Fetch CE option data ---
#             #from datetime import datetime
#             expiry_formatted = datetime.strptime(expiry, "%Y-%m-%d").strftime("%d%b%y").upper()
#             ce_tradingsymbol = f"NIFTY{expiry_formatted}{strike}CE"
#             ce_token = instrument_manager.get_instrument_token(ce_tradingsymbol, "NFO")
#             ce_instrument_name = f"NIFTY{strike}CE{expiry_formatted}"
#             if ce_token:
#                 ce_df = fetcher.fetch_historical_candles(
#                     smart_api,
#                     symboltoken=ce_token,
#                     exchange="NFO",
#                     interval=interval_key,
#                     fromdate=start_time.strftime('%Y-%m-%d %H:%M'),
#                     todate=now.strftime('%Y-%m-%d %H:%M'),
#                     instrument_name=ce_instrument_name
#                 )
#                 if ce_df is not None and not ce_df.empty:
#                     enriched_ce_path = base_dir / f"{ce_instrument_name}_{suffix}.csv"
#                     ce_df.to_csv(enriched_ce_path)
#                     logger.info(f"Fetched and saved CE option data to {enriched_ce_path}")
#                 else:
#                     logger.warning(f"Could not fetch data for CE option: {ce_instrument_name} with token {ce_token}")
#             else:
#                 logger.warning(f"Could not retrieve token for CE option: {ce_tradingsymbol}")

#             # --- Fetch PE option data ---
#             pe_tradingsymbol = f"NIFTY{expiry_formatted}{strike}PE"
#             pe_token = instrument_manager.get_instrument_token(pe_tradingsymbol, "NFO")
#             pe_instrument_name = f"NIFTY{strike}PE{expiry_formatted}"
#             if pe_token:
#                 pe_df = fetcher.fetch_historical_candles(
#                     smart_api,
#                     symboltoken=pe_token,
#                     exchange="NFO",
#                     interval=interval_key,
#                     fromdate=start_time.strftime('%Y-%m-%d %H:%M'),
#                     todate=now.strftime('%Y-%m-%d %H:%M'),
#                     instrument_name=pe_instrument_name
#                 )
#                 if pe_df is not None and not pe_df.empty:
#                     enriched_pe_path = base_dir / f"{pe_instrument_name}_{suffix}.csv"
#                     pe_df.to_csv(enriched_pe_path)
#                     logger.info(f"Fetched and saved PE option data to {pe_instrument_name}")
#                 else:
#                     logger.warning(f"Could not fetch data for PE option: {pe_instrument_name} with token {pe_token}")
#             else:
#                 logger.warning(f"Could not retrieve token for PE option: {pe_tradingsymbol}")

#     except Exception as e:
#         logger.error(f"Script execution failed: {str(e)}")
#         raise
def main(output: str, duration_months: int = 2, expiry: str = "2025-05-29", strike: str = "24000"):
    try:
        base_output_path = Path(output)
        raw_data_dir = base_output_path / "raw"
        option_data_dir = base_output_path / "option"

        raw_data_dir.mkdir(parents=True, exist_ok=True)
        option_data_dir.mkdir(parents=True, exist_ok=True)

        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        start_time = now - timedelta(days=duration_months * 30)

        intervals = {
            "FIVE_MINUTE": "5min","THREE_MINUTE": "3min","ONE_MINUTE": "1min","FIFTEEN_MINUTE": "15min"
        }

        # Initialize AngelOneAPI to get InstrumentManager
        config = load_config()
        api_client = AngelOneAPI(config)
        instrument_manager = api_client.get_instrument_manager()
        smart_api = api_client.get_smart_connect_object()

        if instrument_manager is None or smart_api is None:
            logger.error("Failed to initialize AngelOne API or InstrumentManager.")
            return

        for interval_key, suffix in intervals.items():
            logger.info(f"Starting fetch for interval: {interval_key}")
            fetcher = HistoricalDataFetcher(interval=interval_key)

            # Fetch NIFTY data and save to raw folder
            nifty_df = fetcher.fetch_historical_data(smart_api, days=duration_months * 30)
            nifty_path = raw_data_dir / f"nifty_{suffix}_historical.csv"
            nifty_df.to_csv(nifty_path)
            logger.info(f"NIFTY {interval_key} data saved to {nifty_path}")

            # --- Fetch CE option data and save to option folder ---
            
            expiry_formatted = datetime.strptime(expiry, "%Y-%m-%d").strftime("%d%b%y").upper()
            ce_tradingsymbol = f"NIFTY{expiry_formatted}{strike}CE"
            ce_token = instrument_manager.get_instrument_token(ce_tradingsymbol, "NFO")
            ce_instrument_name = f"NIFTY{strike}CE{expiry_formatted}"
            if ce_token:
                ce_df = fetcher.fetch_historical_candles(
                    smart_api,
                    symboltoken=ce_token,
                    exchange="NFO",
                    interval=interval_key,
                    fromdate=start_time.strftime('%Y-%m-%d %H:%M'),
                    todate=now.strftime('%Y-%m-%d %H:%M'),
                    instrument_name=ce_instrument_name
                )
                if ce_df is not None and not ce_df.empty:
                    ce_path = option_data_dir / f"{ce_instrument_name}_{suffix}.csv"
                    ce_df.to_csv(ce_path)
                    logger.info(f"Fetched and saved CE option data to {ce_path}")
                else:
                    logger.warning(f"Could not fetch data for CE option: {ce_instrument_name} with token {ce_token}")
            else:
                logger.warning(f"Could not retrieve token for CE option: {ce_tradingsymbol}")

            # --- Fetch PE option data and save to option folder ---
            pe_tradingsymbol = f"NIFTY{expiry_formatted}{strike}PE"
            pe_token = instrument_manager.get_instrument_token(pe_tradingsymbol, "NFO")
            pe_instrument_name = f"NIFTY{strike}PE{expiry_formatted}"
            if pe_token:
                pe_df = fetcher.fetch_historical_candles(
                    smart_api,
                    symboltoken=pe_token,
                    exchange="NFO",
                    interval=interval_key,
                    fromdate=start_time.strftime('%Y-%m-%d %H:%M'),
                    todate=now.strftime('%Y-%m-%d %H:%M'),
                    instrument_name=pe_instrument_name
                )
                if pe_df is not None and not pe_df.empty:
                    pe_path = option_data_dir / f"{pe_instrument_name}_{suffix}.csv"
                    pe_df.to_csv(pe_path)
                    logger.info(f"Fetched and saved PE option data to {pe_path}")
                else:
                    logger.warning(f"Could not fetch data for PE option: {pe_instrument_name} with token {pe_token}")
            else:
                logger.warning(f"Could not retrieve token for PE option: {pe_tradingsymbol}")

    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        raise
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/",
                        help="Base output directory")
    parser.add_argument("--duration", type=int, default=2,
                        help="Number of months of historical data to fetch")
    parser.add_argument("--expiry", default="2025-05-29",
                        help="Expiry date for the options (YYYY-MM-DD)")
    parser.add_argument("--strike", default="24000",
                        help="Strike price for the options")
    args = parser.parse_args()
    main(args.output, args.duration, args.expiry, args.strike)