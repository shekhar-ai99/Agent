import os
import sys
import time
from logzero import logger
from dotenv import load_dotenv
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from app.tick_aggregator import TickAggregator
import pyotp
from datetime import datetime
import pytz
class LiveDataFeeder:
    """
    Handles live tick data streaming, aggregates into OHLCV candles, and calls callbacks.
    """
    def _setup(self):
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                login_data = self.obj.generateSession(self.CLIENT_CODE, self.PASSWORD, self.TOTP_SECRET)
                return login_data
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    delay = 2 * (2 ** attempt)
                    print(f"Rate limit error, retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise
        raise Exception("Failed to login after max attempts")
    def __init__(
        self,
        tokens,                   # List of dicts: [{"exchangeType": 1, "tokens": ["26000", ...]}, ...]
        interval_minutes=5,       # Candle interval
        candle_callback=None,     # Function to call with each completed candle
        tick_callback=None,       # Function to call with each raw tick
        mode=1,                   # LTP mode (1=LTP, 2=Quote, etc)
        correlation_id="live_correlation_1",
        api_key=None, client_code=None, password=None, totp_secret=None
    ):
        load_dotenv()  # loads from .env by default

        self.API_KEY = api_key or os.getenv("ANGELONE_API_KEY")
        self.CLIENT_CODE = client_code or os.getenv("ANGELONE_CLIENT_CODE")
        self.PASSWORD = password or os.getenv("ANGELONE_PASSWORD")
        self.TOTP_SECRET = totp_secret or os.getenv("ANGELONE_TOTP_SECRET")
        self.TOKEN_LIST = tokens
        self.CORRELATION_ID = correlation_id
        self.MODE = mode

        self.agg = TickAggregator(interval_minutes=interval_minutes)
        self.candle_callback = candle_callback
        self.tick_callback = tick_callback

        self.obj = None  # SmartConnect
        self.sws = None  # SmartWebSocketV2
        self.running = False  # Track running state
        self._setup()

    def _setup(self):
        # --- Authentication ---
        if not all([self.API_KEY, self.CLIENT_CODE, self.PASSWORD, self.TOTP_SECRET]):
            logger.error("Missing required environment variables or args for Angel One API")
            sys.exit(1)

        self.obj = SmartConnect(self.API_KEY)
        totp = pyotp.TOTP(self.TOTP_SECRET).now()

        logger.info("Attempting to login to Angel One...")
        login_data = self.obj.generateSession(self.CLIENT_CODE, self.PASSWORD, totp)
        if not login_data or login_data.get("status") is False:
            logger.error(f"Login Failed: {login_data.get('message', 'Unknown error')}")
            sys.exit(1)
        self.AUTH_TOKEN = login_data['data']['jwtToken']
        self.FEED_TOKEN = self.obj.getfeedToken()
        logger.info("Login successful. Obtained Auth and Feed tokens.")

        # --- WebSocket ---
        self.sws = SmartWebSocketV2(self.AUTH_TOKEN, self.API_KEY, self.CLIENT_CODE, self.FEED_TOKEN)
        self.sws.on_data = self.on_data
        self.sws.on_open = self.on_open
        self.sws.on_error = self.on_error
        self.sws.on_close = self.on_close

    def on_data(self, wsapp, message):
        """
        Receives each tick, processes it to aggregate, calls callbacks.
        """
        # Optional: add latency measurement
        #recv_time = datetime.now()
        recv_time = datetime.now().astimezone(pytz.timezone("Asia/Kolkata"))
       # logger.info(f"[Tick] {recv_time.strftime('%H:%M:%S.%f')[:-3]} - Data: {message}")

        # --- Raw tick callback ---
        if self.tick_callback and callable(self.tick_callback):
            try:
                self.tick_callback(message)
            except Exception as e_tick_cb:
                logger.error(f"Error in tick_callback: {e_tick_cb}", exc_info=True)

        # --- Aggregation ---
        try:
            candle = self.agg.process_tick(message)
            if candle and self.candle_callback:
                # Optional latency between first tick in candle and now
                latency_ms = (recv_time - candle['datetime']).total_seconds() * 1000
                logger.info(f"[Candle] {candle['datetime']} - latency {latency_ms:.1f} ms")

                self.candle_callback(candle)
        except Exception as e_agg:
            logger.error(f"Error in candle aggregation: {e_agg}", exc_info=True)

    def on_open(self, wsapp):
        logger.info("WebSocket connection opened.")
        logger.info(f"Subscribing to tokens with mode {self.MODE}...")
        try:
            self.sws.subscribe(self.CORRELATION_ID, self.MODE, self.TOKEN_LIST)
            logger.info("Subscription sent.")
        except Exception as e_sub:
            logger.error(f"Subscription failed: {e_sub}", exc_info=True)

    def on_error(self, wsapp, error):
        logger.error(f"WebSocket Error: {error}")

    def on_close(self, wsapp):
        logger.info("WebSocket connection closed by server/user.")
        self.running = False

    def start(self):
        logger.info("Connecting to WebSocket...")
        self.running = True
        try:
            self.sws.connect()
            logger.info("WebSocket connected. Waiting for ticks (Press Ctrl+C to stop)...")
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C).")
            self.stop()
        except Exception as e_start:
            logger.error(f"Error during WebSocket run: {e_start}", exc_info=True)
            self.stop()

    def stop(self):
        logger.info("Initiating shutdown...")
        if self.sws and hasattr(self.sws, "is_connected") and self.sws.is_connected():
            logger.info("Closing WebSocket connection...")
            self.sws.close_connection()
        if self.obj:
            try:
                logger.info("Attempting to logout...")
                logout_status = self.obj.terminateSession(self.CLIENT_CODE)
                logger.info(f"Logout status: {logout_status}")
            except Exception as logout_err:
                logger.error(f"Logout failed: {logout_err}", exc_info=True)
        self.running = False
        logger.info("LiveDataFeeder shutdown complete.")

    def is_alive(self):
        return self.running
