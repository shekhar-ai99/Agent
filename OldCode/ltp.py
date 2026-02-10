import os
import pyotp
import http.client # Import http.client
import json       # Import json for handling payload and response
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
API_KEY = os.getenv("ANGELONE_API_KEY2")
CLIENT_CODE = os.getenv("ANGELONE_CLIENT_CODE")
PASSWORD = os.getenv("ANGELONE_PASSWORD")
TOTP_SECRET = os.getenv("ANGELONE_TOTP_SECRET")

# --- Parameters ----
SYMBOL_TOKEN = "3045"       # Example: NIFTY 50 token
TRADING_SYMBOL = "SBIN-EQ"     # The trading symbol
EXCHANGE = "NSE"
# --- API Endpoint details ---
API_HOST = "apiconnect.angelone.in"
LTP_ENDPOINT = "/rest/secure/angelbroking/order/v1/getLtpData"
# -------------------

if not all([API_KEY, CLIENT_CODE, PASSWORD, TOTP_SECRET]):
    logger.error("Missing required environment variables in .env file")
    sys.exit()

obj = None # Initialize obj to None
refreshToken = None
AUTH_TOKEN = None # Initialize auth token

try:
    # Initialize SmartConnect (still needed for login to get the token)
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
    # Ensure you are getting the JWT token, usually under 'jwtToken'
    AUTH_TOKEN = login_data['data'].get('jwtToken') # Use .get() for safety

    if not AUTH_TOKEN:
        logger.error("Could not find 'jwtToken' in login response data.")
        logger.debug(f"Login response data keys: {login_data['data'].keys()}")
        sys.exit()

    logger.info("Login successful. Obtained Auth Token.")
    # ----> ADD DEBUG PRINT FOR TOKEN <----
    # Print only parts of the token for security
    logger.debug(f"Using Auth Token (partial): Bearer {AUTH_TOKEN[:10]}...{AUTH_TOKEN[-10:]}")
    # ------------------------------------

    # --- Direct HTTP Request using http.client ---
    logger.info(f"Fetching LTP data for {TRADING_SYMBOL} ({SYMBOL_TOKEN}) using http.client")

    conn = http.client.HTTPSConnection(API_HOST)

    payload_dict = {
        "exchange": EXCHANGE,
        "tradingsymbol": TRADING_SYMBOL,
        "symboltoken": SYMBOL_TOKEN
    }
    payload = json.dumps(payload_dict)

    headers = {
      'Authorization': f'Bearer {AUTH_TOKEN}',
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'X-UserType': 'USER',
      'X-SourceID': 'WEB',
      'X-ClientLocalIP': '192.168.1.1',
      'X-ClientPublicIP': '1.1.1.1',
      'X-MACAddress': '00:00:00:00:00:00',
      'X-PrivateKey': API_KEY
    }

    # ----> ADD DEBUG PRINT FOR HEADERS <----
    logger.debug(f"Request Headers: {json.dumps(headers, indent=2)}")
    # -------------------------------------

    # Make the POST request
    conn.request("POST", LTP_ENDPOINT, payload, headers)
    # Make the POST request


    # Get the response
    res = conn.getresponse()
    # Check if the request was successful (HTTP status code 200)
    if res.status != 200:
         logger.error(f"HTTP Request failed with status {res.status} - {res.reason}")
         raw_error_data = res.read().decode("utf-8")
         logger.error(f"Error Response: {raw_error_data}")
    else:
        # Read and decode the response data
        data_bytes = res.read()
        data_string = data_bytes.decode("utf-8")

        # Parse the JSON response
        ltp_data = json.loads(data_string)

        # Process the response
        if ltp_data and ltp_data.get("status") is True and ltp_data.get("data"):
            logger.info("Successfully fetched LTP data via http.client.")
            print("\nLTP Data:")
            print(json.dumps(ltp_data["data"], indent=2)) # Pretty print the data dict
        else:
            logger.error(f"API call successful but failed to get LTP data: {ltp_data.get('message', 'No data found')}")
            logger.debug(f"Full response: {ltp_data}")

except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)

finally:
    # Logout using the SDK (still recommended)
    if obj and refreshToken:
         try:
             logger.info("Attempting to logout...")
             logout_status = obj.terminateSession(CLIENT_CODE)
             logger.info(f"Logout status: {logout_status}")
         except Exception as logout_err:
             logger.error(f"Logout failed: {logout_err}", exc_info=True)