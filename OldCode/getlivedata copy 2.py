import os
import pyotp
import time
from logzero import logger
from dotenv import load_dotenv
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
import sys

# Load environment variables from .env file
load_dotenv()

# Retrieve credentials from environment variables
API_KEY = os.getenv("ANGELONE_API_KEY")
CLIENT_CODE = os.getenv("ANGELONE_CLIENT_CODE")
PASSWORD = os.getenv("ANGELONE_PASSWORD")
TOTP_SECRET = os.getenv("ANGELONE_TOTP_SECRET")

# --- Parameters ----
# Define the instruments to subscribe to
# Format: [{"exchangeType": 1 for NSE, 2 for NFO, etc., "tokens": ["token1", "token2", ...]}]
# Find exchangeType codes in Angel One API documentation
TOKEN_LIST = [
    #{"exchangeType": 1, "tokens": ["3045"]},  # NSE: SBIN-EQ
    {"exchangeType": 1, "tokens": ["26000"]}, # NSE: NIFTY BANK Index
     #{"exchangeType": 1, "tokens": ["26009"]}, # NSE: NIFTY 50 Index
]
CORRELATION_ID = "my_unique_id_123" # Can be anything unique
ACTION = 1  # 1 for Subscribe, 0 for Unsubscribe
MODE = 1    # 1 for LTP, 2 for Quote, 3 for Snap Quote (Full data including depth)
# -------------------


if not all([API_KEY, CLIENT_CODE, PASSWORD, TOTP_SECRET]):
    logger.error("Missing required environment variables in .env file")
    sys.exit()

obj = None
sws = None

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

    AUTH_TOKEN = login_data['data']['jwtToken'] # Use JWT Token for WebSocket V2
    FEED_TOKEN = obj.getfeedToken()             # Get Feed Token
    refreshToken = login_data['data']['refreshToken'] # Needed for potential re-login/refresh logic (not used here)
    logger.info("Login successful. Obtained Auth and Feed tokens.")

    # Initialize WebSocket V2
    sws = SmartWebSocketV2(AUTH_TOKEN, API_KEY, CLIENT_CODE, FEED_TOKEN)

    # --- Define WebSocket Callbacks ---
    from tick_aggregator import TickAggregator

    agg = TickAggregator(interval_minutes=5)
    live_candles = []
    def on_data(wsapp, message):
        logger.info("Tick Data: {}".format(message))
        candle = agg.process_tick(message)
        if candle:
            live_candles.append(candle)
        # You can process the 'message' dictionary here
        # Example: check message type, extract LTP, etc.
        # if message.get('token') == '3045':
        #     ltp = message.get('ltp')
        #     if ltp:
        #         print(f"SBIN LTP: {ltp}")


    def on_open(wsapp):
        logger.info("WebSocket connection opened.")
        logger.info(f"Subscribing to tokens with mode {MODE}...")
        sws.subscribe(CORRELATION_ID, MODE, TOKEN_LIST)


    def on_error(wsapp, error):
        logger.error(f"WebSocket Error: {error}")


    def on_close(wsapp):
        logger.info("WebSocket connection closed.")

    # Assign callbacks
    sws.on_open = on_open
    sws.on_data = on_data
    sws.on_error = on_error
    sws.on_close = on_close

    # Connect to WebSocket (this runs in a background thread)
    logger.info("Connecting to WebSocket...")
    sws.connect()

    # Keep the main thread alive to receive ticks
    logger.info("WebSocket connected. Waiting for ticks (Press Ctrl+C to stop)...")
    while True:
        time.sleep(1) # Keep the script running

except KeyboardInterrupt:
    logger.info("Interrupted by user.")
except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
finally:
    # Close WebSocket connection
    if sws and sws.is_connected():
         logger.info("Closing WebSocket connection...")
         sws.close_connection() # Important to close gracefully

    # Logout from REST API session (if logged in)
    if obj and 'refreshToken' in locals() and refreshToken:
         try:
             logger.info("Attempting to logout...")
             logout_status = obj.terminateSession(CLIENT_CODE)
             logger.info(f"Logout status: {logout_status}")
         except Exception as logout_err:
             logger.error(f"Logout failed: {logout_err}", exc_info=True)

    logger.info("Script finished.")