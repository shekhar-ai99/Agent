import pandas as pd
import logging
import time
from datetime import datetime, timedelta
import pytz

logger = logging.getLogger(__name__)

class AngelDataFetcher:
    def __init__(self, config):
        from angel_one.angel_one_api import AngelOneAPI
        self.api = AngelOneAPI(config)
        self.config = config

    def fetch_historical_candles(self, symbol, timeframe, days=90, exchange=None, retries=3):
        token = self.api.get_instrument_manager().get_instrument_token(symbol, exchange or self.config.get("exchange", "NSE"))
        if not token:
            logger.error(f"[AngelDataFetcher] Token not found for {symbol} ({exchange})")
            return pd.DataFrame()
        tf_map = {
            "1min": "ONE_MINUTE",
            "3min": "THREE_MINUTE",
            "5min": "FIVE_MINUTE",
            "15min": "FIFTEEN_MINUTE",
            "30min": "THIRTY_MINUTE",
            "60min": "ONE_HOUR",
            "1day": "ONE_DAY"
        }
        interval = tf_map.get(timeframe, "FIVE_MINUTE")
        ist = pytz.timezone('Asia/Kolkata')
        end_dt = datetime.now(ist)
        start_dt = end_dt - timedelta(days=days)
        params = {
            "exchange": exchange or self.config.get("exchange", "NSE"),
            "symboltoken": str(token),
            "interval": interval,
            "fromdate": start_dt.strftime('%Y-%m-%d %H:%M'),
            "todate": end_dt.strftime('%Y-%m-%d %H:%M')
        }
        for attempt in range(retries):
            try:
                api = self.api.get_smart_connect_object()
                logger.info(f"Fetch historical {symbol} {interval}: {params}")
                resp = api.getCandleData(params)
                if resp.get("status") and resp.get("data"):
                    df = pd.DataFrame(resp["data"], columns=["datetime", "open", "high", "low", "close", "volume"])
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df.set_index("datetime", inplace=True)
                    logger.info(f"Fetched {len(df)} rows for {symbol} ({interval})")
                    return df
                logger.warning(f"Fetch attempt {attempt+1}: {resp.get('message', 'No data')}")
                time.sleep(2)
            except Exception as e:
                logger.error(f"Fetch error {symbol}: {e}")
                time.sleep(2)
        logger.error(f"[AngelDataFetcher] All fetch attempts failed for {symbol}")
        return pd.DataFrame()
