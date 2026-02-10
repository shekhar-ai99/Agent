import sys
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class HistoricalDataFetcher:
    def __init__(self, interval: str):
        self.NIFTY_SYMBOL = "99926000"  # Token for NIFTY 50
        self.HISTORICAL_INTERVAL = interval
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 5  # seconds

    def initialize_api(self) -> SmartConnect:
        """Initialize and authenticate with the SmartAPI"""
        try:
            api_key = os.getenv("ANGELONE_API_KEY")
            client_code = os.getenv("ANGELONE_CLIENT_CODE")
            password = os.getenv("ANGELONE_PASSWORD")
            totp_secret = os.getenv("ANGELONE_TOTP_SECRET")

            if not all([api_key, client_code, password, totp_secret]):
                raise ValueError("Missing required environment variables")

            smart_api = SmartConnect(api_key)
            
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

    def fetch_historical_data(self, smart_api: SmartConnect, days: int = 30) -> pd.DataFrame:
        """Fetch historical candle data"""
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

            logger.info(f"Fetching {self.HISTORICAL_INTERVAL} data with params: {params}")
            
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
            
            # Basic data validation
            for col in ["open", "high", "low", "close"]:
                if (df[col] <= 0).any():
                    raise ValueError(f"Invalid {col} prices in data")
            
            logger.info(f"Fetched {len(df)} records for interval {self.HISTORICAL_INTERVAL}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch {self.HISTORICAL_INTERVAL} data: {str(e)}")
            raise

def main(output: str):
    """
    Main function to fetch historical data.
    If no output is provided, it defaults to "nifty_historical_data.csv".
    The script outputs three CSV files (for 3min, 5min, and 15min intervals) using the base file name.
    """
    try:
        # Determine if output is a directory or a file path with a basename.
        output_path = Path(output)
        if output_path.is_dir():
            base_dir = output_path
            base_name = "nifty_historical_data"
            ext = ".csv"
        else:
            base_dir = output_path.parent
            base_name = output_path.stem
            ext = output_path.suffix if output_path.suffix else ".csv"

        # Ensure the output directory exists
        base_dir.mkdir(parents=True, exist_ok=True)
        
        intervals = {
             "ONE_MINUTE": "1min",
            "THREE_MINUTE": "3min",
            "FIVE_MINUTE": "5min",
            "FIFTEEN_MINUTE": "15min"
        }
        
        for interval_key, suffix in intervals.items():
            logger.info(f"Starting fetch for interval: {interval_key}")
            fetcher = HistoricalDataFetcher(interval=interval_key)
            smart_api = fetcher.initialize_api()
            df = fetcher.fetch_historical_data(smart_api)
            
            # Construct output file name
            filename = f"{base_name}_{suffix}{ext}"
            full_path = base_dir / filename
            
            df.to_csv(full_path)
            logger.info(f"{interval_key} data saved to {full_path}")
        
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/data", 
                        help="(Optional) Output file path (e.g., data/nifty_historical_data.csv) or directory (e.g., data/). Defaults to 'nifty_historical_data.csv'.")
    args = parser.parse_args()
    
    main(args.output)
