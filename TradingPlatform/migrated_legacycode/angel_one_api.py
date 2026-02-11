# angel_one_api.py

import os
import pyotp
import time
from logzero import logger
from dotenv import load_dotenv
from SmartApi import SmartConnect
import sys
import logging
from typing import Optional
import datetime as dt

from angel_one.angel_one_instrument_manager import InstrumentManager
from angel_one.angel_one_market_data import AngelMarketDataClient

# Load environment variables from .env file
load_dotenv()

# Retrieve credentials from environment variables
API_KEY = os.getenv("ANGELONE_API_KEY")
CLIENT_CODE = os.getenv("ANGELONE_CLIENT_CODE")
PASSWORD = os.getenv("ANGELONE_PASSWORD")
TOTP_SECRET = os.getenv("ANGELONE_TOTP_SECRET")

if not all([API_KEY, CLIENT_CODE, PASSWORD, TOTP_SECRET]):
    logger.error("Missing required environment variables in .env file")
    sys.exit()

class AngelOneAPI:
    def __init__(self, config):
        self.config = config
        self.smart_api_obj: Optional[SmartConnect] = None
        self.jwt_token: Optional[str] = None
        self.feed_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.session_expiry_time: Optional[dt.datetime] = None
        self.login_successful = False
        self.api_key = API_KEY
        self.client_code =CLIENT_CODE
        self.instrument_manager = None
        self._initialize_session()
        if self.login_successful:
            self.market_data_client = AngelMarketDataClient(self.jwt_token)
            self.instrument_manager = InstrumentManager(self.config) # Pass self as broker_client


    def _initialize_session(self):
        MAX_RETRIES = 3
        RETRY_DELAY = 5

        password = PASSWORD
        totp_secret =TOTP_SECRET

        if not all([self.api_key, self.client_code, password, totp_secret]):
            logger.error("Missing AngelOne credentials in config.")
            return

        self.smart_api_obj = SmartConnect(api_key=self.api_key)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"AngelOne Auth attempt {attempt}/{MAX_RETRIES}")
                totp = pyotp.TOTP(totp_secret).now()
                login_response = self.smart_api_obj.generateSession(self.client_code, password, totp)

                if login_response.get("status"):
                    logger.info("✅ AngelOne SmartAPI login successful")
                    self.jwt_token = login_response["data"]["jwtToken"]
                    self.feed_token = self.smart_api_obj.getfeedToken()
                    self.refresh_token = login_response["data"]["refreshToken"]
                    self.session_expiry_time = dt.datetime.now() + dt.timedelta(hours=self.config.get("angel_session_expiry_hours", 6))
                    self.login_successful = True
                    return
                else:
                    logger.warning(f"Login failed: {login_response.get('message', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Login attempt {attempt} failed: {e}")
                if attempt < MAX_RETRIES:
                    logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)

        logger.error("Failed to initialize AngelOne SmartAPI after max retries.")

    def is_session_valid(self) -> bool:
        if not self.login_successful or not self.session_expiry_time:
            return False
        if self.session_expiry_time <= dt.datetime.now():
            logger.warning("Session expired. Attempting to refresh...")
            return self._renew_session()
        return True

    def _renew_session(self) -> bool:
        if not self.refresh_token:
            logger.error("No refresh token available for session renewal")
            return False

        MAX_RETRIES = 3
        RETRY_DELAY = 5

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"Session refresh attempt {attempt}/{MAX_RETRIES}")
                refresh_response = self.smart_api_obj.renewAccessToken(self.refresh_token)

                if refresh_response.get("status"):
                    self.jwt_token = refresh_response['data'].get('jwtToken')
                    self.feed_token = self.smart_api_obj.getfeedToken()
                    self.refresh_token = refresh_response['data'].get('refreshToken')
                    self.session_expiry_time = dt.datetime.now() + dt.timedelta(hours=self.config.get("angel_session_expiry_hours", 6))
                    logger.info("✅ Session refreshed successfully")
                    return True
            except Exception as e:
                logger.error(f"Session refresh attempt {attempt} failed: {e}")
                if attempt < MAX_RETRIES:
                    logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)

        logger.error("Failed to refresh session after max retries")
        self.login_successful = False
        return False

    def get_smart_connect_object(self):
        return self.smart_api_obj

    def get_auth_token(self):
        return self.jwt_token

    def get_feed_token(self):
        return self.feed_token

    def get_instrument_manager(self):
        return self.instrument_manager