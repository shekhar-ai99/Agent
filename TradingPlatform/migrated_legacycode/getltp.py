import os
import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Retrieve credentials from environment variables
API_KEY = os.getenv("ANGELONE_API_KEY")
CLIENT_CODE = os.getenv("ANGELONE_CLIENT_CODE")
PASSWORD = os.getenv("ANGELONE_PASSWORD")
TOTP_SECRET = os.getenv("ANGELONE_TOTP_SECRET")

# --- Parameters ----
SYMBOL_TOKEN = "99926000"       # Example: SBIN-EQ token on NSE
TRADING_SYMBOL = "NIFTY 50" # The trading symbol
EXCHANGE = "NSE"
# -------------------

if not all([API_KEY, CLIENT_CODE, PASSWORD, TOTP_SECRET]):
    logger.error("Missing required environment variables in .env file")
    sys.exit()

obj = None # Initialize obj to None

try:
    # Initialize SmartConnect
    obj = SmartConnect(API_KEY)

    # Generate TOTP
    totp = pyotp.TOTP(TOTP_SECRET).now()

    # Login / Generate Session
    logger.info("Attempting to login...")
    login_data = obj.generateSession(CLIENT_CODE, PASSWORD, totp)

    if not login_data or login_data.get("status") is False:
        logger.error(f"Login Failed: {login_data.get('message', 'Unknown error')}")
        sys.exit()

    refreshToken = login_data['data']['refreshToken']
    logger.info("Login successful.")

    # Prepare parameters for market data request (using marketData method)
    # Mode: 1 for LTP, 2 for Quote, 3 for Full Snap Quote
    market_data_params = {
        "mode": 1, # Mode 1 = LTP
        "exchangeTokens": {
            EXCHANGE: [SYMBOL_TOKEN] # e.g., {"NSE": ["99926000"]} based on your params
        }
    }

    logger.info(f"Fetching market data (LTP) for {TRADING_SYMBOL} ({SYMBOL_TOKEN}) using marketData method")
    # Fetch market data (LTP snapshot)
    snapshot_data = obj.marketData(market_data_params) # <-- Use marketData here

    # Check the response structure - it might differ slightly
    # Data is often in snapshot_data['data']['fetched'][0] for the first token
    if snapshot_data and snapshot_data.get("status") and snapshot_data.get("data") and snapshot_data["data"].get("fetched"):
        logger.info("Successfully fetched market data.")
        fetched_data = snapshot_data["data"]["fetched"]
        if fetched_data:
            print("\nMarket Data Snapshot (LTP Mode):")
            print(fetched_data[0]) # Print the data for the first token
            # Example Access:
            # ltp = fetched_data[0].get('ltp')
            # last_traded_time = fetched_data[0].get('lastTradedTime') # Check available fields in the output
            # print(f"LTP: {ltp}")
            # print(f"Last Traded Time: {datetime.fromtimestamp(last_traded_time/1000) if last_traded_time else 'N/A'}") # Example time conversion
        else:
            print("No data points found in 'fetched' list within the response.")

    else:
        logger.error(f"Could not fetch market data: {snapshot_data.get('message', 'API error or no data')}")
        # Optional: Log the full response if debugging is needed
        # logger.debug(f"Full response: {snapshot_data}")

except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)

finally:
    # Logout (always recommended)
    if obj and 'refreshToken' in locals() and refreshToken:
         try:
             logger.info("Attempting to logout...")
             logout_status = obj.terminateSession(CLIENT_CODE)
             logger.info(f"Logout status: {logout_status}")
         except Exception as logout_err:
             logger.error(f"Logout failed: {logout_err}", exc_info=True)