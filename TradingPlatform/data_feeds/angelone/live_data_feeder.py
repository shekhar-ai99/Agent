"""
Live Data Feeder for Angel One

Handles live tick data streaming from Angel One broker, aggregates into candles,
and provides callbacks for ticks and completed candles.
"""

import os
import sys
import time
from datetime import datetime
import pytz
import pyotp
from logzero import logger
from dotenv import load_dotenv
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

from .tick_aggregator import TickAggregator


class LiveDataFeeder:
    """
    Handles live tick data streaming, aggregates into OHLCV candles, and calls callbacks.
    
    Features:
    - Angel One SmartConnect API authentication with TOTP
    - SmartWebSocketV2 for live tick streaming
    - Automatic tick aggregation into time-based candles
    - Configurable callbacks for raw ticks and completed candles
    - Graceful shutdown with session termination
    """

    def __init__(
        self,
        tokens,                       # List of dicts: [{"exchangeType": 1, "tokens": ["26000", ...]}]
        interval_minutes: int = 5,    # Candle interval
        candle_callback=None,         # Function to call with each completed candle
        tick_callback=None,           # Function to call with each raw tick
        mode: int = 1,                # LTP mode (1=LTP, 2=Quote, etc)
        correlation_id: str = "live_correlation_1",
        api_key=None,
        client_code=None,
        password=None,
        totp_secret=None,
    ):
        """
        Initialize LiveDataFeeder.
        
        Args:
            tokens: List of token subscriptions with exchangeType and tokens
            interval_minutes: Candle interval in minutes
            candle_callback: Function(candle_dict) called on completed candles
            tick_callback: Function(tick_dict) called on every tick
            mode: Angel One mode (1=LTP, 2=Quote, 3=Snapshot)
            correlation_id: Correlation ID for subscription
            api_key: Angel One API key (defaults to env var)
            client_code: Client code (defaults to env var)
            password: Password (defaults to env var)
            totp_secret: TOTP secret (defaults to env var)
        """
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
        """
        Authenticate with Angel One and establish WebSocket connection.
        
        Raises:
            SystemExit: If credentials are missing or authentication fails
        """
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

    def on_data(self, wsapp, message: dict) -> None:
        """
        Callback for incoming tick data from WebSocket.
        
        Processes raw ticks and aggregates them into candles.
        
        Args:
            wsapp: WebSocket app instance
            message: Tick message dict
        """
        recv_time = datetime.now().astimezone(pytz.timezone("Asia/Kolkata"))

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

    def on_open(self, wsapp) -> None:
        """
        Callback when WebSocket connection opens.
        
        Subscribes to configured tokens.
        
        Args:
            wsapp: WebSocket app instance
        """
        logger.info("WebSocket connection opened.")
        logger.info(f"Subscribing to tokens with mode {self.MODE}...")
        try:
            self.sws.subscribe(self.CORRELATION_ID, self.MODE, self.TOKEN_LIST)
            logger.info("Subscription sent.")
        except Exception as e_sub:
            logger.error(f"Subscription failed: {e_sub}", exc_info=True)

    def on_error(self, wsapp, error) -> None:
        """
        Callback for WebSocket errors.
        
        Args:
            wsapp: WebSocket app instance
            error: Error message
        """
        logger.error(f"WebSocket Error: {error}")

    def on_close(self, wsapp) -> None:
        """
        Callback when WebSocket closes.
        
        Args:
            wsapp: WebSocket app instance
        """
        logger.info("WebSocket connection closed by server/user.")
        self.running = False

    def start(self) -> None:
        """
        Start the live data feed and listen for ticks.
        
        Runs blocking until stop() is called or Ctrl+C is pressed.
        """
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

    def stop(self) -> None:
        """
        Stop the live data feed and gracefully close connections.
        
        Closes WebSocket and terminates Angel One session.
        """
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

    def is_alive(self) -> bool:
        """
        Check if feeder is running.
        
        Returns:
            True if running, False otherwise
        """
        return self.running
