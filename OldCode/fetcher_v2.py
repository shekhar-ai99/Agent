import os
import time
import pytz
import pyotp
import pandas as pd
from datetime import datetime, timedelta
from SmartApi import SmartConnect
import logging

# ——— Logging Setup ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HistoricalDataFetcher:
    """
    Encapsulates Angel One SmartAPI authentication and historical‐candle fetching,
    plus offline CSV fallback.
    """

    # Default mapping of friendly intervals → SmartAPI codes
    INTERVAL_MAP = {
        "1minute":  "ONE_MINUTE",
        "3minute":  "THREE_MINUTE",
        "5minute":  "FIVE_MINUTE",
        "10minute": "TEN_MINUTE",
        "15minute": "FIFTEEN_MINUTE",
        "30minute": "THIRTY_MINUTE",
        "1hour":    "SIXTY_MINUTE",
        "1day":     "ONE_DAY",
    }

    # For offline CSV lookups
    OFFLINE_MAP = {
        "3minute":  "nifty_historical_data_3min.csv",
        "5minute":  "nifty_historical_data_5min.csv",
        "15minute": "nifty_historical_data_15min.csv",
    }

    def __init__(self, interval: str):
        self.interval = interval
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 5  # seconds

        # You can override this via env if you want another token
        self.NIFTY_SYMBOL = os.getenv("NIFTY_TOKEN", "99926000")

    def _to_smartapi_interval(self):
        return self.INTERVAL_MAP.get(self.interval, "ONE_DAY")

    def initialize_api(self) -> SmartConnect:
        """
        Create & login SmartConnect client, retrying up to MAX_RETRIES.
        """
        api_key     = os.getenv("ANGELONE_API_KEY")
        client_code = os.getenv("ANGELONE_CLIENT_CODE")
        password    = os.getenv("ANGELONE_PASSWORD_OR_PIN")
        totp_secret = os.getenv("ANGELONE_TOTP_SECRET")

        if not all([api_key, client_code, password, totp_secret]):
            raise ValueError("Missing Angel One credentials in environment")

        smart_api = SmartConnect(api_key)

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.info(f"Auth attempt {attempt}/{self.MAX_RETRIES}")
                totp = pyotp.TOTP(totp_secret).now()
                resp = smart_api.generateSession(client_code, password, totp)
                if resp.get("status") and resp["data"].get("jwtToken"):
                    logger.info("✅ SmartAPI login successful")
                    return smart_api
                logger.warning(f"Login failed: {resp.get('message')}")
            except Exception as ex:
                logger.error(f"Login exception: {ex}")
            time.sleep(self.RETRY_DELAY)

        raise ConnectionError(f"Could not login to SmartAPI after {self.MAX_RETRIES} attempts")

    def fetch_online(
        self,
        smart_api: SmartConnect,
        from_dt: datetime = None,
        to_dt:   datetime = None,
        ticker:  str      = None,
        exchange:str      = "NSE"
    ) -> pd.DataFrame:
        """
        Fetch from Angel One in one or more chunks.
        Defaults to last 30 days if from_dt/to_dt not given.
        """
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)

        if not to_dt:
            to_dt = now
        if not from_dt:
            from_dt = now - timedelta(days=30)

        token = ticker or self.NIFTY_SYMBOL
        api_interval = self._to_smartapi_interval()

        all_rows = []
        chunk_start = from_dt
        # simple 5-day chunks
        while chunk_start < to_dt:
            chunk_end = min(chunk_start + timedelta(days=5), to_dt)
            params = {
                "exchange":    exchange.upper(),
                "symboltoken": token,
                "interval":    api_interval,
                "fromdate":    chunk_start.strftime("%Y-%m-%d %H:%M"),
                "todate":      chunk_end  .strftime("%Y-%m-%d %H:%M"),
            }
            logger.info(f"▶️ Fetching chunk {params}")
            for attempt in range(1, self.MAX_RETRIES + 1):
                resp = smart_api.getCandleData(params)
                if resp.get("status") and resp.get("data"):
                    all_rows += resp["data"]
                    break
                logger.warning(f" chunk {attempt} failed: {resp.get('message')}")
                time.sleep(self.RETRY_DELAY)
            chunk_start = chunk_end
            time.sleep(0.5)

        if not all_rows:
            logger.warning("No online data returned")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows, columns=[
            "datetime", "open", "high", "low", "close", "volume"
        ])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(inplace=True)
        logger.info(f"✅ Online data loaded: {df.shape}")
        return df

    def fetch_offline(self) -> pd.DataFrame:
        """
        Load from CSV under data/ matching OFFLINE_MAP interval.
        """
        fname = self.OFFLINE_MAP.get(self.interval)
        if not fname:
            raise ValueError(f"No CSV mapped for interval '{self.interval}'")
        path = os.path.join("data", fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing offline file: {path}")

        df = pd.read_csv(path)
        if "datetime" not in df.columns:
            raise ValueError("CSV lacks 'datetime' column")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        logger.info(f"✅ Offline data loaded: {path} → {df.shape}")
        return df

    def fetch_data(
        self,
        source:  str,
        interval: str,
        from_dt: datetime = None,
        to_dt:   datetime = None,
        ticker:  str      = None,
        exchange:str      = "NSE"
    ) -> pd.DataFrame:
        """
        Unified entrypoint. source='online' or 'offline'.
        Other parameters only used for online.
        """
        if source.lower() == "offline":
            return self.fetch_offline()
        elif source.lower() == "online":
            api = self.initialize_api()
            return self.fetch_online(api, from_dt, to_dt, ticker, exchange)
        else:
            raise ValueError(f"Unknown source '{source}'")
