
import sys
import os

import requests


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import logging
import datetime as dt
import pytz
import time
from typing import List, Optional, Tuple
from enum import Enum
from SmartApi.smartExceptions import DataException
from config import Config, get_data_dir, setup_logging
from angel_one.angel_one_api import AngelOneAPI
from angel_one.angel_one_market_data import AngelMarketDataClient
from angel_one.angel_one_instrument_manager import InstrumentManager
# from liveDryrun.broker_client import BrokerClient
# from liveDryrun.market_data_client import AngelMarketDataClient
# from liveDryrun.instrument_manager import InstrumentManager

import struct
import ssl
import json
import websocket
import traceback

logger = logging.getLogger(__name__)

class SmartWebSocketV2:
    ROOT_URI = "ws://smartapisocket.angelone.in/smart-stream"
    HEART_BEAT_MESSAGE = "ping"
    HEAR_BEAT_INTERVAL = 30
    LITTLE_ENDIAN_BYTE_ORDER = "<"
    RESUBSCRIBE_FLAG = False
    MAX_RETRY_ATTEMPT = 3

    SUBSCRIBE_ACTION = 1
    UNSUBSCRIBE_ACTION = 0
    LTP_MODE = 1
    QUOTE = 2
    SNAP_QUOTE = 3

    NSE_CM = 1
    NSE_FO = 2
    BSE_CM = 3
    BSE_FO = 4
    MCX_FO = 5
    NCX_FO = 7
    CDE_FO = 13

    SUBSCRIPTION_MODE_MAP = {1: "LTP", 2: "QUOTE", 3: "SNAP_QUOTE"}

    wsapp = None
    input_request_dict = {}

    def __init__(self, auth_token, api_key, client_code, feed_token, on_open=None, on_data=None, on_error=None, on_close=None):
        self.auth_token = auth_token
        self.api_key = api_key
        self.client_code = client_code
        self.feed_token = feed_token
        self.retry_count = 0

        # Attach overrides
        if on_open: self.on_open = on_open
        if on_data: self.on_data = on_data
        if on_error: self.on_error = on_error
        if on_close: self.on_close = on_close

    def _sanity_check(self):
        return bool(self.auth_token and self.api_key and self.client_code and self.feed_token)

    def _on_data(self, wsapp, data, data_type, continue_flag):
        if data_type == 2:
            parsed_message = self._parse_binary_data(data)
            self.on_data(wsapp, parsed_message)
        else:
            self.on_data(wsapp, data)

    def _on_open(self, wsapp):
        if self.RESUBSCRIBE_FLAG:
            self.resubscribe()
        else:
            self.RESUBSCRIBE_FLAG = True
            self.on_open(wsapp)

    def _on_pong(self, wsapp, data):
        logger.debug(f"Pong received: {data}")

    def _on_ping(self, wsapp, data):
        logger.debug(f"Ping received: {data}")

    def subscribe(self, correlation_id, mode, token_list):
        try:
            request_data = {
                "correlationID": correlation_id,
                "action": self.SUBSCRIBE_ACTION,
                "params": {"mode": mode, "tokenList": token_list}
            }
            if self.input_request_dict.get(mode) is None:
                self.input_request_dict[mode] = {}
            for token in token_list:
                if token['exchangeType'] in self.input_request_dict[mode]:
                    self.input_request_dict[mode][token['exchangeType']].extend(token["tokens"])
                else:
                    self.input_request_dict[mode][token['exchangeType']] = token["tokens"]
            self.wsapp.send(json.dumps(request_data))
            self.RESUBSCRIBE_FLAG = True
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
            raise

    def unsubscribe(self, correlation_id, mode, token_list):
        try:
            request_data = {
                "correlationID": correlation_id,
                "action": self.UNSUBSCRIBE_ACTION,
                "params": {"mode": mode, "tokenList": token_list}
            }
            self.wsapp.send(json.dumps(request_data))
            self.RESUBSCRIBE_FLAG = True
        except Exception as e:
            logger.error(f"Unsubscription failed: {e}")
            raise

    def resubscribe(self):
        try:
            for key, val in self.input_request_dict.items():
                token_list = []
                for key1, val1 in val.items():
                    token_list.append({'exchangeType': key1, 'tokens': val1})
                request_data = {
                    "action": self.SUBSCRIBE_ACTION,
                    "params": {"mode": key, "tokenList": token_list}
                }
                self.wsapp.send(json.dumps(request_data))
        except Exception as e:
            logger.error(f"Resubscription failed: {e}")
            raise

    def connect(self):
        try:
            headers = {
                "Authorization": self.auth_token,
                "x-api-key": self.api_key,
                "x-client-code": self.client_code,
                "x-feed-token": self.feed_token
            }
            logger.debug(f"Connecting to WebSocket with headers: {headers}")
            self.wsapp = websocket.WebSocketApp(
                self.ROOT_URI,
                header=headers,
                on_open=self._on_open,
                on_error=self._on_error,
                on_close=self._on_close,
                on_data=self._on_data,
                on_ping=self._on_ping,
                on_pong=self._on_pong)
            self.wsapp.run_forever(
                sslopt={"cert_reqs": ssl.CERT_NONE},
                ping_interval=self.HEAR_BEAT_INTERVAL,
                ping_payload=self.HEART_BEAT_MESSAGE)
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise

    def close_connection(self):
        self.RESUBSCRIBE_FLAG = False
        if self.wsapp:
            self.wsapp.close()
            logger.info("WebSocket connection closed")

    def _on_error(self, wsapp, error):
        logger.error(f"WebSocket error: {error}")
        self.retry_count += 1
        if self.retry_count >= self.MAX_RETRY_ATTEMPT:
            logger.error("Max retry attempts reached. Aborting WebSocket connection.")
            self.wsapp.keep_running = False
        else:
            logger.info(f"Retrying connection (attempt {self.retry_count + 1}/{self.MAX_RETRY_ATTEMPT}) in 10 seconds...")
            time.sleep(10)  # Increased retry delay
            self.connect()

    def _on_close(self, wsapp, close_status_code, close_msg):
        logger.info(f"WebSocket closed: {close_status_code}, {close_msg}")

    def _parse_binary_data(self, binary_data):
        try:
            parsed_data = {
                "subscription_mode": self._unpack_data(binary_data, 0, 1, byte_format="B")[0],
                "exchange_type": self._unpack_data(binary_data, 1, 2, byte_format="B")[0],
                "token": self._parse_token_value(binary_data[2:27]),
                "sequence_number": self._unpack_data(binary_data, 27, 35, byte_format="q")[0],
                "exchange_timestamp": self._unpack_data(binary_data, 35, 43, byte_format="q")[0],
                "last_traded_price": self._unpack_data(binary_data, 43, 51, byte_format="q")[0]
            }
            parsed_data["subscription_mode_val"] = self.SUBSCRIPTION_MODE_MAP.get(parsed_data["subscription_mode"])
            if parsed_data["subscription_mode"] in [self.QUOTE, self.SNAP_QUOTE]:
                parsed_data["last_traded_quantity"] = self._unpack_data(binary_data, 51, 59, byte_format="q")[0]
                parsed_data["average_traded_price"] = self._unpack_data(binary_data, 59, 67, byte_format="q")[0]
                parsed_data["volume_trade_for_the_day"] = self._unpack_data(binary_data, 67, 75, byte_format="q")[0]
                parsed_data["total_buy_quantity"] = self._unpack_data(binary_data, 75, 83, byte_format="d")[0]
                parsed_data["total_sell_quantity"] = self._unpack_data(binary_data, 83, 91, byte_format="d")[0]
                parsed_data["open_price_of_the_day"] = self._unpack_data(binary_data, 91, 99, byte_format="q")[0]
                parsed_data["high_price_of_the_day"] = self._unpack_data(binary_data, 99, 107, byte_format="q")[0]
                parsed_data["low_price_of_the_day"] = self._unpack_data(binary_data, 107, 115, byte_format="q")[0]
                parsed_data["closed_price"] = self._unpack_data(binary_data, 115, 123, byte_format="q")[0]
            if parsed_data["subscription_mode"] == self.SNAP_QUOTE:
                parsed_data["last_traded_timestamp"] = self._unpack_data(binary_data, 123, 131, byte_format="q")[0]
                parsed_data["open_interest"] = self._unpack_data(binary_data, 131, 139, byte_format="q")[0]
                parsed_data["open_interest_change_percentage"] = self._unpack_data(binary_data, 139, 147, byte_format="q")[0]
                parsed_data["upper_circuit_limit"] = self._unpack_data(binary_data, 347, 355, byte_format="q")[0]
                parsed_data["lower_circuit_limit"] = self._unpack_data(binary_data, 355, 363, byte_format="q")[0]
                parsed_data["52_week_high_price"] = self._unpack_data(binary_data, 363, 371, byte_format="q")[0]
                parsed_data["52_week_low_price"] = self._unpack_data(binary_data, 371, 379, byte_format="q")[0]
                best_5_data = self._parse_best_5_buy_and_sell_data(binary_data[147:347])
                parsed_data["best_5_buy_data"] = best_5_data["best_5_buy_data"]
                parsed_data["best_5_sell_data"] = best_5_data["best_5_sell_data"]
            return parsed_data
        except Exception as e:
            logger.error(f"Failed to parse binary data: {e}")
            raise

    def _unpack_data(self, binary_data, start, end, byte_format="I"):
        return struct.unpack(self.LITTLE_ENDIAN_BYTE_ORDER + byte_format, binary_data[start:end])

    @staticmethod
    def _parse_token_value(binary_packet):
        token = ""
        for i in range(len(binary_packet)):
            if chr(binary_packet[i]) == '\x00':
                return token
            token += chr(binary_packet[i])
        return token

    def _parse_best_5_buy_and_sell_data(self, binary_data):
        def split_packets(binary_packets):
            packets = []
            i = 0
            while i < len(binary_packets):
                packets.append(binary_packets[i: i + 20])
                i += 20
            return packets
        best_5_buy_sell_packets = split_packets(binary_data)
        best_5_buy_data = []
        best_5_sell_data = []
        for packet in best_5_buy_sell_packets:
            each_data = {
                "flag": self._unpack_data(packet, 0, 2, byte_format="H")[0],
                "quantity": self._unpack_data(packet, 2, 10, byte_format="q")[0],
                "price": self._unpack_data(packet, 10, 18, byte_format="q")[0],
                "no of orders": self._unpack_data(packet, 18, 20, byte_format="H")[0]
            }
            if each_data["flag"] == 0:
                best_5_buy_data.append(each_data)
            else:
                best_5_sell_data.append(each_data)
        return {"best_5_buy_data": best_5_buy_data, "best_5_sell_data": best_5_sell_data}

    def on_data(self, wsapp, data):
        pass

    def on_close(self, wsapp):
        pass

    def on_open(self, wsapp):
        pass

class Timeframe(Enum):
    ONE_MINUTE = "1min"
    THREE_MINUTE = "3min"
    FIVE_MINUTE = "5min"
    TEN_MINUTE = "10min"
    FIFTEEN_MINUTE = "15min"
    THIRTY_MINUTE = "30min"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"

class DataTypeGainersLosers(Enum):
    PERC_OI_GAINERS = "PercOIGainers"
    PERC_OI_LOSERS = "PercOILosers"
    PERC_PRICE_GAINERS = "PercPriceGainers"
    PERC_PRICE_LOSERS = "PercPriceLosers"

class DataTypeOIBuildup(Enum):
    LONG_BUILT_UP = "Long Built Up"
    SHORT_BUILT_UP = "Short Built Up"
    SHORT_COVERING = "Short Covering"
    LONG_UNWINDING = "Long Unwinding"

class ExpiryType(Enum):
    NEAR = "NEAR"
    NEXT = "NEXT"
    FAR = "FAR"

class DataFetcher:
    TIMEFRAME_MAPPING = {
        "1min": "ONE_MINUTE",
        "3min": "THREE_MINUTE",
        "5min": "FIVE_MINUTE",
        "10min": "TEN_MINUTE",
        "15min": "FIFTEEN_MINUTE",
        "30min": "THIRTY_MINUTE",
        "1h": "ONE_HOUR",
        "1d": "ONE_DAY"
    }

    def __init__(self, broker_client: BrokerClient, instrument_manager: InstrumentManager, config: Config):
        self.broker_client = broker_client
        self.instrument_manager = instrument_manager
        self.config = config
        if not self.broker_client.broker_client_initialised:
            logger.error("Broker client not initialized. Cannot proceed.")
            raise Exception("Broker client initialization failed")
        self._validate_tokens()
        logger.info("DataFetcher initialized successfully.")

    def _validate_tokens(self):
        jwt_token = getattr(self.broker_client, 'jwt_token', None)
        feed_token = getattr(self.broker_client, 'feed_token', None)
        if not jwt_token or not feed_token:
            logger.error(f"Invalid tokens: JWT={bool(jwt_token)}, Feed={bool(feed_token)}")
            raise Exception("Missing JWT or Feed token")
        logger.debug(f"Tokens validated: JWT={jwt_token[:10]}..., Feed={feed_token[:10]}...")

    def retry_request(self, func, *args, max_retries: int = 3, retry_delay: int = 5, rate_limit_delay: int = 10, **kwargs):
        for attempt in range(1, max_retries + 1):
            try:
                result = func(*args, **kwargs)
                #logger.debug(f"API response: {json.dumps(result, indent=2) if result else 'No response'}")
                if isinstance(result, dict):
                    logger.debug(f"API response: {json.dumps(result, indent=2)}")
                elif hasattr(result, "text"):
                    logger.debug(f"API raw response text: {result.text}")
                else:
                    logger.debug(f"API response: {result}")

                return result
            except DataException as e:
                if "exceeding access rate" in str(e).lower():
                    logger.warning(f"Rate limit hit on attempt {attempt}. Sleeping {rate_limit_delay}s...")
                    time.sleep(rate_limit_delay)
                    continue
                if "invalid token" in str(e).lower():
                    logger.warning(f"Invalid token on attempt {attempt}. Attempting to refresh session...")
                    if self.broker_client.renew_session():
                        logger.info("Session refreshed successfully. Retrying...")
                        continue
                    logger.error("Session refresh failed.")
                logger.error(f"Attempt {attempt} failed: {e}")
            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {e}\n{traceback.format_exc()}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        logger.error("All retry attempts failed.")
        return None

    def _is_nse_holiday(self, date: dt.datetime) -> bool:
        nse_holidays_2025 = [
            dt.datetime(2025, 1, 26), dt.datetime(2025, 3, 14), dt.datetime(2025, 4, 14),
            dt.datetime(2025, 4, 18), dt.datetime(2025, 5, 1), dt.datetime(2025, 8, 15),
            dt.datetime(2025, 10, 2), dt.datetime(2025, 10, 21), dt.datetime(2025, 11, 7),
            dt.datetime(2025, 12, 25)
        ]
        date_only = date.date()
        return any(holiday.date() == date_only for holiday in nse_holidays_2025)

    def _adjust_to_market_hours(self, date: dt.datetime, direction: str = 'forward') -> dt.datetime:
        ist = pytz.timezone('Asia/Kolkata')
        date = date.astimezone(ist) if date.tzinfo else ist.localize(date)
        while True:
            while date.weekday() >= 5 or self._is_nse_holiday(date):
                if direction == 'forward':
                    date = date + dt.timedelta(days=1)
                    date = date.replace(hour=9, minute=15, second=0, microsecond=0)
                else:
                    date = date - dt.timedelta(days=1)
                    date = date.replace(hour=15, minute=30, second=0, microsecond=0)
            market_open = date.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = date.replace(hour=15, minute=30, second=0, microsecond=0)
            if direction == 'forward':
                if date < market_open:
                    return market_open
                elif date > market_close:
                    date = date + dt.timedelta(days=1)
                    date = date.replace(hour=9, minute=15, second=0, microsecond=0)
                    continue
            else:
                if date > market_close:
                    return market_close
                elif date < market_open:
                    date = date - dt.timedelta(days=1)
                    date = date.replace(hour=15, minute=30, second=0, microsecond=0)
                    continue
            return date

    def fetch_websocket_ltp(self, ticker: str, exchange: str, timeout: int = 10) -> float:
        if not self.broker_client.broker_client_initialised:
            logger.error("Broker client not initialized. Cannot fetch WebSocket LTP.")
            return 0.0
        token = self.instrument_manager.get_instrument_token(ticker, exchange)
        if not token:
            logger.error(f"Cannot fetch WebSocket LTP for {ticker} without a valid token.")
            return 0.0
        exchange_type = {"NSE": 1, "NFO": 2, "BSE": 3}.get(exchange, 2)
        ltp_result = [0.0]
        received = [False]

        def on_data(wsapp, data):
            if isinstance(data, dict) and data.get("token") == token:
                ltp_result[0] = float(data.get("last_traded_price", 0.0)) / 100
                received[0] = True
                logger.info(f"Received WebSocket LTP for {ticker}: {ltp_result[0]}")
                wsapp.close()

        def on_open(wsapp):
            logger.info(f"WebSocket opened for {ticker}")
            wsapp.subscribe("123_qwerty", SmartWebSocketV2.LTP_MODE, [{"exchangeType": exchange_type, "tokens": [token]}])

        def on_error(wsapp, error):
            logger.error(f"WebSocket error for {ticker}: {error}")
            received[0] = True

        def on_close(wsapp, close_status_code, close_msg):
            logger.info(f"WebSocket closed for {ticker}: {close_status_code}, {close_msg}")

        sws = SmartWebSocketV2(
            auth_token=self.broker_client.jwt_token,
            api_key=self.broker_client.api_key,
            client_code=self.broker_client.client_code,
            feed_token=self.broker_client.feed_token
        )
        sws.on_open = on_open
        sws.on_data = on_data
        sws.on_error = on_error
        sws.on_close = on_close

        try:
            import threading
            ws_thread = threading.Thread(target=sws.connect)
            ws_thread.daemon = True
            ws_thread.start()
            start_time = time.time()
            while not received[0] and time.time() - start_time < timeout:
                time.sleep(0.1)
            if not received[0]:
                logger.error(f"Timeout waiting for WebSocket LTP for {ticker}")
                sws.close_connection()
                return 0.0
            return ltp_result[0]
        except Exception as e:
            logger.error(f"WebSocket LTP fetch failed for {ticker}: {e}\n{traceback.format_exc()}")
            sws.close_connection()
            return 0.0

    def _fetch_oi_data(self, symbol: str, timeframe: str, ticker: str, exchange: str, parsed_start_date: dt.datetime, parsed_end_date: dt.datetime) -> pd.DataFrame:
        if not self.broker_client.broker_client_initialised:
            logger.error("Broker client not initialized. Cannot fetch OI data.")
            return pd.DataFrame()
        if exchange != "NFO":
            logger.warning(f"OI data is only available for NFO exchange, not {exchange}. Skipping OI fetch.")
            return pd.DataFrame()
        api_interval = self.TIMEFRAME_MAPPING.get(timeframe.lower(), "FIVE_MINUTE")
        logger.info(f"Fetching OI data for {symbol} ({timeframe}) from {parsed_start_date} to {parsed_end_date} on {exchange}")
        token = self.instrument_manager.get_instrument_token(ticker, exchange)
        if not token:
            logger.error(f"Cannot fetch OI data for {ticker} without a valid token.")
            return pd.DataFrame()
        all_oi_data: List[pd.DataFrame] = []
        current_chunk_start_dt = parsed_start_date
        chunk_delta = dt.timedelta(days=10) if api_interval in ["ONE_MINUTE", "THREE_MINUTE", "FIVE_MINUTE", "TEN_MINUTE", "FIFTEEN_MINUTE", "THIRTY_MINUTE"] else dt.timedelta(days=200)
        while current_chunk_start_dt < parsed_end_date:
            current_chunk_end_dt = min(current_chunk_start_dt + chunk_delta, parsed_end_date)
            from_date_chunk_str = current_chunk_start_dt.strftime('%Y-%m-%d %H:%M')
            to_date_chunk_str = current_chunk_end_dt.strftime('%Y-%m-%d %H:%M')
            params = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": api_interval,
                "fromdate": from_date_chunk_str,
                "todate": to_date_chunk_str
            }
            logger.info(f"Fetching OI chunk: {token} {exchange} {api_interval} From='{from_date_chunk_str}', To='{to_date_chunk_str}'")
            # resp = self.retry_request(
            #     self.broker_client.smart_api_obj._request,
            #     'POST',
            #     'Get OI Data',
            #     params
            # )
            url = "https://apiconnect.angelone.in/rest/secure/angelbroking/marketData/v1/optionOpenInterest"
            headers = {
    "Authorization": f"Bearer {self.broker_client.jwt_token}",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "X-UserType": "USER",
    "X-SourceID": "WEB",
    "X-ClientLocalIP": "127.0.0.1",
    "X-ClientPublicIP": "127.0.0.1",
    "X-MACAddress": "00:00:00:00:00:00"
}
            resp = self.retry_request(
                requests.post,
                url,
                headers=headers,
                json=params
            )

            if not resp or not isinstance(resp, dict) or not resp.get("status", False):
                logger.error(f"API error: {resp.get('message', 'Unknown error') if isinstance(resp, dict) else 'No response or invalid response'}")
                break
            chunk_data = resp.get("data", [])
            if not chunk_data:
                logger.warning(f"No OI data returned for chunk: {from_date_chunk_str} to {to_date_chunk_str}")
                break
            chunk_df = pd.DataFrame(chunk_data)
            chunk_df["time"] = pd.to_datetime(chunk_df["time"], format='%Y-%m-%d %H:%M:%S')
            all_oi_data.append(chunk_df)
            current_chunk_start_dt = current_chunk_end_dt + dt.timedelta(minutes=1)
            time.sleep(2)
        if not all_oi_data:
            logger.warning(f"No OI data returned for {symbol}.")
            return pd.DataFrame()
        df = pd.concat(all_oi_data, ignore_index=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info(f"✅ Processed OI data for {symbol}: {len(df)} rows from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
        logger.info(f"OI stats: min={df['oi'].min()}, max={df['oi'].max()}, mean={df['oi'].mean():.2f}")
        logger.info(f"Fetch successful. Shape: {df.shape}")
        logger.info(f"Sample OI data:\n{df.tail()}")
        return df

    def _fetch_data(self, symbol: str, timeframe: str, ticker: str, exchange: str, parsed_start_date: dt.datetime, parsed_end_date: dt.datetime, fetch_oi: bool = False) -> pd.DataFrame:
        if not self.broker_client.broker_client_initialised:
            logger.error("Broker client not initialized. Cannot fetch data.")
            return pd.DataFrame()
        api_interval = self.TIMEFRAME_MAPPING.get(timeframe.lower(), "FIVE_MINUTE")
        logger.info(f"Fetching OHLCV data for {symbol} ({timeframe}) from {parsed_start_date} to {parsed_end_date} on {exchange}")
        token = self.instrument_manager.get_instrument_token(ticker, exchange)
        if not token:
            logger.error(f"Cannot fetch data for {ticker} without a valid token.")
            return pd.DataFrame()
        all_data: List[pd.DataFrame] = []
        current_chunk_start_dt = parsed_start_date
        chunk_delta = dt.timedelta(days=10) if api_interval in ["ONE_MINUTE", "THREE_MINUTE", "FIVE_MINUTE", "TEN_MINUTE", "FIFTEEN_MINUTE", "THIRTY_MINUTE"] else dt.timedelta(days=200)
        while current_chunk_start_dt < parsed_end_date:
            current_chunk_end_dt = min(current_chunk_start_dt + chunk_delta, parsed_end_date)
            from_date_chunk_str = current_chunk_start_dt.strftime('%Y-%m-%d %H:%M')
            to_date_chunk_str = current_chunk_end_dt.strftime('%Y-%m-%d %H:%M')
            params = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": api_interval,
                "fromdate": from_date_chunk_str,
                "todate": to_date_chunk_str
            }
            logger.info(f"Fetching OHLCV chunk: {token} {exchange} {api_interval} From='{from_date_chunk_str}', To='{to_date_chunk_str}'")
            resp = self.retry_request(
                self.broker_client.smart_api_obj.getCandleData,
                params
            )
            if not resp or not isinstance(resp, dict) or not resp.get("status", False):
                logger.error(f"API error: {resp.get('message', 'Unknown error') if isinstance(resp, dict) else 'No response or invalid response'}")
                break
            chunk_data = resp.get("data", [])
            if not chunk_data:
                logger.warning(f"No OHLCV data returned for chunk: {from_date_chunk_str} to {to_date_chunk_str}")
                break
            chunk_df = pd.DataFrame(
                chunk_data,
                columns=["datetime", "open", "high", "low", "close", "volume"]
            )
            #chunk_df["datetime"] = pd.to_datetime(chunk_df["datetime"], format='%Y-%m-%d %H:%M:%S')
            chunk_df["datetime"] = pd.to_datetime(chunk_df["datetime"], format='ISO8601')

            #chunk_df["datetime"] = pd.to_datetime(chunk_df["datetime"])

            if chunk_df["volume"].eq(0).all():
                logger.warning(f"Chunk from {from_date_chunk_str} to {to_date_chunk_str} has all-zero volume. Skipping.")
                break
            all_data.append(chunk_df)
            current_chunk_start_dt = current_chunk_end_dt + dt.timedelta(minutes=1)
            time.sleep(2)
        if not all_data:
            logger.warning(f"No OHLCV data returned for {symbol}.")
            return pd.DataFrame()
        df = pd.concat(all_data, ignore_index=True)
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        if df["volume"].eq(0).any():
            logger.warning(f"Found {df['volume'].eq(0).sum()} rows with zero volume. Replacing with 1.")
            df["volume"] = df["volume"].replace(0, 1)
        logger.info(f"✅ Processed OHLCV data for {symbol}: {len(df)} rows from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
        logger.info(f"Volume stats: min={df['volume'].min()}, max={df['volume'].max()}, mean={df['volume'].mean():.2f}")
        logger.info(f"Fetch successful. Shape: {df.shape}")
        logger.info(f"Sample OHLCV data:\n{df.tail()}")
        if fetch_oi:
            oi_df = self._fetch_oi_data(symbol, timeframe, ticker, exchange, parsed_start_date, parsed_end_date)
            if not oi_df.empty:
                oi_df = oi_df.rename(columns={"time": "datetime"})
                df = pd.merge_asof(
                    df.sort_values("datetime"),
                    oi_df[["datetime", "oi"]].sort_values("datetime"),
                    on="datetime",
                    direction="nearest"
                )
                logger.info(f"Merged OI data with OHLCV data. New shape: {df.shape}")
        return df

    def _fetch_current_price(self, ticker: str, exchange: str) -> float:
        ltp = self.fetch_websocket_ltp(ticker, exchange)
        if ltp > 0.0:
            return ltp
        logger.warning(f"WebSocket LTP fetch failed for {ticker}. Falling back to HTTP.")
        if not self.broker_client.broker_client_initialised:
            logger.error("Broker client not initialized. Cannot fetch current price.")
            return 0.0
        token = self.instrument_manager.get_instrument_token(ticker, exchange)
        if not token:
            logger.error(f"Cannot fetch current price for {ticker} without a valid token.")
            return 0.0
        params = {
            "exchange": exchange,
            "tradingsymbol": ticker,
            "symboltoken": token
        }
        resp = self.retry_request(
            self.broker_client.smart_api_obj.ltpData,
            exchange,
            ticker,
            token
        )
        if not resp or not isinstance(resp, dict) or not resp.get("status", False):
            logger.error(f"API error: {resp.get('message', 'Unknown error') if isinstance(resp, dict) else 'No response or invalid response'}")
            return 0.0
        ltp_data = resp.get("data", {})
        ltp = ltp_data.get("ltp", 0.0)
        logger.info(f"Current price for {ticker} on {exchange}: {ltp}")
        return float(ltp)

    def fetch_fno_data(self, exchange: str, underlying: str, expiry_date: str, duration: int, timeframe: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not self.broker_client.broker_client_initialised:
            logger.error("Broker client not initialized. Cannot fetch F&O data.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        if exchange != "NFO":
            logger.error(f"F&O data is only available for NFO exchange, not {exchange}.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        if timeframe.lower() not in self.TIMEFRAME_MAPPING:
            logger.error(f"Invalid timeframe: {timeframe}. Must be one of: {list(self.TIMEFRAME_MAPPING.keys())}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        if duration < 1:
            logger.error(f"Invalid duration: {duration}. Must be at least 1 day.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        try:
            expiry_dt = dt.datetime.strptime(expiry_date, '%d%b%Y')
            logger.info(f"Parsed expiry date: {expiry_dt.strftime('%d%b%Y')}")
        except ValueError:
            logger.error(f"Invalid expiry date format: {expiry_date}. Expected format: DDMMMYYYY (e.g., 29MAY2025).")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        current_date = dt.datetime.now(pytz.timezone('Asia/Kolkata'))
        valid_expiries = self.instrument_manager.get_valid_expiries(underlying)
        if not valid_expiries:
            logger.error(f"No valid expiries found for {underlying}. Ensure instrument list is loaded.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        if expiry_date not in valid_expiries:
            logger.warning(f"Expiry {expiry_date} is not valid for {underlying}. Valid expiries: {sorted(valid_expiries)}")
            expiry_date = min(valid_expiries, key=lambda x: abs((dt.datetime.strptime(x, '%d%b%Y') - expiry_dt).days))
            expiry_dt = dt.datetime.strptime(expiry_date, '%d%b%Y')
            logger.info(f"Using nearest valid expiry: {expiry_date}")
        if expiry_dt.date() < current_date.date():
            logger.warning(f"Expiry date {expiry_date} is in the past as of {current_date.date()}. Options data may not be available.")
        underlying = underlying.strip().upper()
        if underlying == "NIFTY":
            underlying_symbol = "NIFTY"
        elif underlying == "BANKNIFTY":
            underlying_symbol = "BANKNIFTY"
        else:
            logger.error(f"Unsupported underlying: {underlying}. Supported: NIFTY, BANKNIFTY.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        expiry_str = expiry_dt.strftime('%y%b').upper()
        expiry_str_full = expiry_dt.strftime('%d%b%Y').upper()
        expiry_str_short = expiry_dt.strftime('%b%y').upper()
        expiry_str_day = expiry_dt.strftime('%d%b%y').upper()
        futures_ticker_formats = [
            f"{underlying_symbol}{expiry_str_day}FUT",  # e.g., NIFTY29MAY25FUT
            f"{underlying_symbol}{expiry_str}FUT",  # e.g., NIFTY25MAYFUT
            f"{underlying_symbol}{expiry_str_full}FUT",  # e.g., NIFTY29MAY2025FUT
            f"{underlying_symbol}{expiry_str_short}FUT"  # e.g., NIFTYMAY25FUT
        ]
        futures_ticker = None
        futures_token = None
        futures_exchange = "NFO"
        for fmt_ticker in futures_ticker_formats:
            token = self.instrument_manager.get_instrument_token(fmt_ticker, futures_exchange)
            if token:
                futures_ticker = fmt_ticker
                futures_token = token
                logger.info(f"Found futures ticker: {futures_ticker}, Token: {futures_token}")
                break
        if not futures_ticker:
            logger.error(f"No instrument found for {underlying} futures with expiry {expiry_date} on {futures_exchange}. Tried formats: {futures_ticker_formats}")
            nfo_futures = self.instrument_manager.instrument_df[
                (self.instrument_manager.instrument_df['exchange'] == 'NFO') &
                (self.instrument_manager.instrument_df['name'] == underlying) &
                (self.instrument_manager.instrument_df['instrumenttype'] == 'FUTIDX')
            ]
            if not nfo_futures.empty:
                logger.info(f"Available NIFTY futures tickers: {nfo_futures['tradingsymbol'].tolist()}")
            self.instrument_manager.refresh_instrument_list()
            for fmt_ticker in futures_ticker_formats:
                token = self.instrument_manager.get_instrument_token(fmt_ticker, futures_exchange)
                if token:
                    futures_ticker = fmt_ticker
                    futures_token = token
                    logger.info(f"Found futures ticker after refresh: {futures_ticker}, Token: {futures_token}")
                    break
            if not futures_ticker:
                logger.error(f"Still no instrument found after refresh for {underlying} futures with expiry {expiry_date}.")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        current_price = self._fetch_current_price(futures_ticker, futures_exchange)
        if current_price == 0.0:
            logger.error(f"Failed to fetch current price for {futures_ticker}. Cannot determine ATM strike.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        logger.info(f"Current {underlying} futures price: {current_price}")
        atm_strike = self.instrument_manager.get_atm_strike(underlying_symbol, expiry_date, current_price)
        if atm_strike is None:
            logger.error(f"Failed to determine ATM strike for {underlying} with expiry {expiry_date}.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        logger.info(f"ATM strike for {underlying} with expiry {expiry_date}: {atm_strike}")
        call_formats = [
            f"{underlying_symbol}{expiry_dt.strftime('%d%b%Y').upper()}{int(atm_strike)}CE",
            f"{underlying_symbol}{expiry_dt.strftime('%d%b').upper()}{int(atm_strike)}CE",
            f"{underlying_symbol}{expiry_dt.strftime('%d%b%y').upper()}{int(atm_strike)}CE",
            f"{underlying_symbol}{expiry_dt.strftime('%y%b%d').upper()}{int(atm_strike)}CE"
        ]
        put_formats = [s.replace("CE", "PE") for s in call_formats]
        call_ticker = None
        put_ticker = None
        for fmt in call_formats:
            if self.instrument_manager.get_instrument_token(fmt, "NFO"):
                call_ticker = fmt
                break
        for fmt in put_formats:
            if self.instrument_manager.get_instrument_token(fmt, "NFO"):
                put_ticker = fmt
                break
        if not call_ticker or not put_ticker:
            logger.error(f"Failed to find token for ATM call or put. Tried call formats: {call_formats}, put formats: {put_formats}")
            nfo_options = self.instrument_manager.instrument_df[
                (self.instrument_manager.instrument_df['exchange'] == 'NFO') &
                (self.instrument_manager.instrument_df['name'] == underlying) &
                (self.instrument_manager.instrument_df['instrumenttype'].isin(['OPTIDX'])) &
                (pd.to_datetime(self.instrument_manager.instrument_df['expiry'], format='%Y-%m-%d') == expiry_dt)
            ]
            if not nfo_options.empty:
                logger.info(f"Available NIFTY option tickers for {expiry_date}: {nfo_options['tradingsymbol'].tolist()}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        call_token = self.instrument_manager.get_instrument_token(call_ticker, "NFO")
        put_token = self.instrument_manager.get_instrument_token(put_ticker, "NFO")
        if not call_token:
            logger.error(f"Failed to find token for ATM call: {call_ticker}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        if not put_token:
            logger.error(f"Failed to find token for ATM put: {put_ticker}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        logger.info(f"ATM Call: {call_ticker}, Token: {call_token}")
        logger.info(f"ATM Put: {put_ticker}, Token: {put_token}")
        ist = pytz.timezone('Asia/Kolkata')
        parsed_end_date = dt.datetime.now(ist)
        parsed_start_date = parsed_end_date - dt.timedelta(days=duration)
        parsed_start_date = self._adjust_to_market_hours(parsed_start_date, direction='forward')
        parsed_end_date = self._adjust_to_market_hours(parsed_end_date, direction='backward')
        call_ohlcv_df = self._fetch_data(
            symbol=call_ticker,
            timeframe=timeframe,
            ticker=call_ticker,
            exchange="NFO",
            parsed_start_date=parsed_start_date,
            parsed_end_date=parsed_end_date,
            fetch_oi=False
        )
        call_oi_df = self._fetch_oi_data(
            symbol=call_ticker,
            timeframe=timeframe,
            ticker=call_ticker,
            exchange="NFO",
            parsed_start_date=parsed_start_date,
            parsed_end_date=parsed_end_date
        )
        put_ohlcv_df = self._fetch_data(
            symbol=put_ticker,
            timeframe=timeframe,
            ticker=put_ticker,
            exchange="NFO",
            parsed_start_date=parsed_start_date,
            parsed_end_date=parsed_end_date,
            fetch_oi=False
        )
        put_oi_df = self._fetch_oi_data(
            symbol=put_ticker,
            timeframe=timeframe,
            ticker=put_ticker,
            exchange="NFO",
            parsed_start_date=parsed_start_date,
            parsed_end_date=parsed_end_date
        )
        data_dir = get_data_dir()
        underlying_cleaned = underlying_symbol.replace(" ", "_")
        expiry_cleaned = expiry_date.upper()
        call_ohlcv_path = os.path.join(data_dir, "raw", f"{underlying_cleaned}_{expiry_cleaned}_ATM_call_ohlcv.csv")
        os.makedirs(os.path.dirname(call_ohlcv_path), exist_ok=True)
        if not call_ohlcv_df.empty:
            call_ohlcv_df.to_csv(call_ohlcv_path, index=False)
            logger.info(f"Call OHLCV data saved to {call_ohlcv_path}")
        call_oi_path = os.path.join(data_dir, "raw", f"{underlying_cleaned}_{expiry_cleaned}_ATM_call_oi.csv")
        if not call_oi_df.empty:
            call_oi_df.to_csv(call_oi_path, index=False)
            logger.info(f"Call OI data saved to {call_oi_path}")
        put_ohlcv_path = os.path.join(data_dir, "raw", f"{underlying_cleaned}_{expiry_cleaned}_ATM_put_ohlcv.csv")
        if not put_ohlcv_df.empty:
            put_ohlcv_df.to_csv(put_ohlcv_path, index=False)
            logger.info(f"Put OHLCV data saved to {put_ohlcv_path}")
        put_oi_path = os.path.join(data_dir, "raw", f"{underlying_cleaned}_{expiry_cleaned}_ATM_put_oi.csv")
        if not put_oi_df.empty:
            put_oi_df.to_csv(put_oi_path, index=False)
            logger.info(f"Put OI data saved to {put_oi_path}")
        # greeks_df = self.fetch_option_greeks(underlying_symbol, expiry_date)
        # if not greeks_df.empty and "strikePrice" in greeks_df.columns:
        #     greeks_filtered = greeks_df[greeks_df['strikePrice'] == atm_strike]
        #     ce_greeks = greeks_filtered[greeks_filtered['optionType'] == 'CE']
        #     pe_greeks = greeks_filtered[greeks_filtered['optionType'] == 'PE']
        #     option_dir = os.path.join(get_data_dir(), "raw", "options")
        #     os.makedirs(option_dir, exist_ok=True)
        #     ce_greeks_path = os.path.join(option_dir, f"{underlying_cleaned}_{expiry_cleaned}_{int(atm_strike)}CE_greeks.csv")
        #     pe_greeks_path = os.path.join(option_dir, f"{underlying_cleaned}_{expiry_cleaned}_{int(atm_strike)}PE_greeks.csv")
        #     if not ce_greeks.empty:
        #         ce_greeks.to_csv(ce_greeks_path, index=False)
        #         logger.info(f"CE Greeks saved to {ce_greeks_path}")
        #     if not pe_greeks.empty:
        #         pe_greeks.to_csv(pe_greeks_path, index=False)
        #         logger.info(f"PE Greeks saved to {pe_greeks_path}")
        # else:
        #     logger.warning("Skipping Greeks save due to empty or invalid Greeks data.")
        return call_ohlcv_df, call_oi_df, put_ohlcv_df, put_oi_df

    def fetch_option_greeks(self, underlying: str, expiry_date: str) -> pd.DataFrame:
        if not self.broker_client.broker_client_initialised:
            logger.error("Broker client not initialized. Cannot fetch Option Greeks.")
            return pd.DataFrame()
        try:
            expiry_dt = dt.datetime.strptime(expiry_date, '%d%b%Y')
        except ValueError:
            logger.error(f"Invalid expiry date format: {expiry_date}. Expected format: DDMMMYYYY.")
            return pd.DataFrame()
        params = {
            "name": underlying.strip().upper(),
            "expirydate": expiry_dt.strftime('%d%b%Y').upper()
        }
        logger.info(f"Fetching Option Greeks for {params['name']} with expiry {params['expirydate']}")
        # resp = self.retry_request(
        #     self.broker_client.smart_api_obj._request,
        #     'POST',
        #     'rest/secure/angelbroking/marketData/v1/optionGreek',
        #     params
        # )
        if not self.broker_client.is_session_valid():
            logger.warning("Session expired. Refreshing before fetching Greeks...")
            self.broker_client.renew_session()

        resp = self.retry_request(
        self.broker_client.market_data_client.get_option_greeks,
        underlying.strip().upper(),
        expiry_dt.strftime('%d%b%Y').upper()
    )

        if not resp or not isinstance(resp, dict):
            logger.error(f"Option Greeks API error: No response or invalid response received")
            return pd.DataFrame()
        if not resp.get("status", False):
            logger.error(f"Option Greeks API error: {resp.get('message', 'Unknown error')}")
            return pd.DataFrame()
        data = resp.get("data", [])
        if not data:
            logger.warning("Option Greeks response contains no data.")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        for col in ["strikePrice", "delta", "gamma", "theta", "vega", "impliedVolatility", "tradeVolume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        logger.info(f"✅ Retrieved {len(df)} Option Greeks entries.")
        return df

    def fetch_top_gainers_losers(self, data_type: str, expiry_type: str) -> pd.DataFrame:
        if not self.broker_client.broker_client_initialised:
            logger.error("Broker client not initialized. Cannot fetch Top Gainers/Losers.")
            return pd.DataFrame()
        try:
            data_type_enum = DataTypeGainersLosers(data_type)
        except ValueError:
            logger.error(f"Invalid data_type: {data_type}. Must be one of: {[e.value for e in DataTypeGainersLosers]}")
            return pd.DataFrame()
        try:
            expiry_type_enum = ExpiryType(expiry_type)
        except ValueError:
            logger.error(f"Invalid expiry_type: {expiry_type}. Must be one of: {[e.value for e in ExpiryType]}")
            return pd.DataFrame()
        params = {
            "datatype": data_type_enum.value,
            "expirytype": expiry_type_enum.value
        }
        logger.info(f"Fetching Top Gainers/Losers: {data_type} for {expiry_type} expiry")
        # resp = self.retry_request(
        #     self.broker_client.smart_api_obj._request,
        #     'POST',
        #     'Gainers Losers',
        #     params
        # )
        resp = self.retry_request(
        self.broker_client.market_data_client.get_top_gainers_losers,
        data_type_enum.value,
        expiry_type_enum.value
    )

        if not resp or not isinstance(resp, dict) or not resp.get("status", False):
            logger.error(f"API error: {resp.get('message', 'Unknown error') if isinstance(resp, dict) else 'No response or invalid response'}")
            return pd.DataFrame()
        gainers_losers_data = resp.get("data", [])
        if not gainers_losers_data:
            logger.warning(f"No Top Gainers/Losers data returned for {data_type} with expiry {expiry_type}")
            return pd.DataFrame()
        df = pd.DataFrame(gainers_losers_data)
        df["percentChange"] = df["percentChange"].astype(float)
        df["opnInterest"] = df["opnInterest"].astype(float)
        df["netChangeOpnInterest"] = df["netChangeOpnInterest"].astype(float)
        logger.info(f"✅ Fetched Top Gainers/Losers for {data_type}: {len(df)} records")
        logger.info(f"Sample Top Gainers/Losers data:\n{df.head()}")
        data_dir = get_data_dir()
        file_path = os.path.join(data_dir, "raw", "derivatives", f"{data_type}_{expiry_type}_gainers_losers.csv")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Top Gainers/Losers data saved to {file_path}")
        return df

    def fetch_pcr_volume(self) -> pd.DataFrame:
        if not self.broker_client.broker_client_initialised:
            logger.error("Broker client not initialized. Cannot fetch PCR Volume.")
            return pd.DataFrame()
        logger.info("Fetching PCR Volume data")
        # resp = self.retry_request(
        #     self.broker_client.smart_api_obj._request,
        #     'GET',
        #     'Put Call Ratio',
        #     {}
        # )
        resp = self.retry_request(
    self.broker_client.market_data_client.get_pcr_volume
)

        if not resp or not isinstance(resp, dict) or not resp.get("status", False):
            logger.error(f"API error: {resp.get('message', 'Unknown error') if isinstance(resp, dict) else 'No response or invalid response'}")
            return pd.DataFrame()
        pcr_data = resp.get("data", [])
        if not pcr_data:
            logger.warning("No PCR Volume data returned")
            return pd.DataFrame()
        df = pd.DataFrame(pcr_data)
        df["pcr"] = df["pcr"].astype(float)
        logger.info(f"✅ Fetched PCR Volume data: {len(df)} records")
        logger.info(f"Sample PCR Volume data:\n{df.head()}")
        data_dir = get_data_dir()
        current_date = dt.datetime.now().strftime('%Y%m%d')
        file_path = os.path.join(data_dir, "raw", "derivatives", f"pcr_volume_{current_date}.csv")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"PCR Volume data saved to {file_path}")
        return df

    def fetch_oi_buildup(self, data_type: str, expiry_type: str) -> pd.DataFrame:
        if not self.broker_client.broker_client_initialised:
            logger.error("Broker client not initialized. Cannot fetch OI Buildup.")
            return pd.DataFrame()
        try:
            data_type_enum = DataTypeOIBuildup(data_type)
        except ValueError:
            logger.error(f"Invalid data_type: {data_type}. Must be one of: {[e.value for e in DataTypeOIBuildup]}")
            return pd.DataFrame()
        try:
            expiry_type_enum = ExpiryType(expiry_type)
        except ValueError:
            logger.error(f"Invalid expiry_type: {expiry_type}. Must be one of: {[e.value for e in ExpiryType]}")
            return pd.DataFrame()
        params = {
            "datatype": data_type_enum.value,
            "expirytype": expiry_type_enum.value
        }
        logger.info(f"Fetching OI Buildup: {data_type} for {expiry_type} expiry")
            # resp = self.retry_request(
            #     self.broker_client.smart_api_obj._request,
            #     'POST',
            #     'OI Buildup',
            #     params
            # )
        resp = self.retry_request(
        self.broker_client.market_data_client.get_oi_buildup,
        data_type_enum.value,
        expiry_type_enum.value
    )

        if not resp or not isinstance(resp, dict) or not resp.get("status", False):
            logger.error(f"API error: {resp.get('message', 'Unknown error') if isinstance(resp, dict) else 'No response or invalid response'}")
            return pd.DataFrame()
        oi_buildup_data = resp.get("data", [])
        if not oi_buildup_data:
            logger.warning(f"No OI Buildup data returned for {data_type} with expiry {expiry_type}")
            return pd.DataFrame()
        df = pd.DataFrame(oi_buildup_data)
        df["ltp"] = df["ltp"].astype(float)
        df["netChange"] = df["netChange"].astype(float)
        df["percentChange"] = df["percentChange"].astype(float)
        df["opnInterest"] = df["opnInterest"].astype(float)
        df["netChangeOpnInterest"] = df["netChangeOpnInterest"].astype(float)
        logger.info(f"✅ Fetched OI Buildup for {data_type}: {len(df)} records")
        logger.info(f"Sample OI Buildup data:\n{df.head()}")
        data_dir = get_data_dir()
        file_path = os.path.join(data_dir, "raw", "derivatives", f"{data_type.replace(' ', '_')}_{expiry_type}_oi_buildup.csv")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"OI Buildup data saved to {file_path}")
        return df

    def fetch_data(self, symbol: str, duration: int, timeframe: str, ticker: str, exchange: Optional[str] = None, fetch_oi: bool = False) -> pd.DataFrame:
        ist = pytz.timezone('Asia/Kolkata')
        parsed_end_date = dt.datetime.now(ist)
        parsed_start_date = parsed_end_date - dt.timedelta(days=duration)
        parsed_start_date = self._adjust_to_market_hours(parsed_start_date, direction='forward')
        parsed_end_date = self._adjust_to_market_hours(parsed_end_date, direction='backward')
        df = self._fetch_data(symbol, timeframe, ticker, exchange, parsed_start_date, parsed_end_date, fetch_oi=fetch_oi)
        if not df.empty:
            data_dir = get_data_dir()
            symbol_cleaned = symbol.replace(" ", "_")
            historical_path_template = self.config.get("historical_data_path_template", "raw/{symbol}_{timeframe}_historical.csv")
            historical_path = os.path.join(data_dir, historical_path_template.format(symbol=symbol_cleaned, timeframe=timeframe))
            os.makedirs(os.path.dirname(historical_path), exist_ok=True)
            df.to_csv(historical_path, index=False)
            logger.info(f"Historical data saved to {historical_path}")
        return df

    def fetch_live_data(self, symbol: str, duration: int, timeframe: str, ticker: str, exchange: Optional[str] = None, fetch_oi: bool = False) -> pd.DataFrame:
        ist = pytz.timezone('Asia/Kolkata')
        now = dt.datetime.now(ist)
        parsed_end_date = self._adjust_to_market_hours(now, direction='backward')
        parsed_start_date = parsed_end_date - dt.timedelta(days=duration)
        parsed_start_date = self._adjust_to_market_hours(parsed_start_date, direction='forward')
        if parsed_start_date >= parsed_end_date:
            logger.error(f"Invalid date range: start {parsed_start_date} >= end {parsed_end_date}")
            return pd.DataFrame()
        df = self._fetch_data(symbol, timeframe, ticker, exchange, parsed_start_date, parsed_end_date, fetch_oi=fetch_oi)
        if not df.empty:
            data_dir = get_data_dir()
            symbol_cleaned = symbol.replace(" ", "_")
            live_raw_dir = os.path.join(data_dir, self.config.get("live_raw_data_dir_template", "raw/live"))
            live_path = os.path.join(live_raw_dir, f"{symbol_cleaned}_{timeframe}_live.csv")
            os.makedirs(os.path.dirname(live_path), exist_ok=True)
            df.to_csv(live_path, index=False)
            logger.info(f"Live data saved to {live_path}")
        return df

if __name__ == "__main__":
    import argparse
    from config import Config, setup_logging
    parser = argparse.ArgumentParser(description="Fetch market data from AngelOne SmartAPI.")
    parser.add_argument("--symbol", default="SBIN-EQ", help="Symbol to fetch data for (e.g., SBIN-EQ)")
    parser.add_argument("--timeframe", default="5min", help="Timeframe (e.g., 5min, 1h, 1d)")
    parser.add_argument("--days", type=int, default=90, help="Number of days of historical data")
    parser.add_argument("--fetch-oi", action="store_true", help="Fetch OI data (for NFO symbols)")
    parser.add_argument("--fetch-greeks", action="store_true", help="Fetch Option Greeks")
    parser.add_argument("--underlying", default="NIFTY", help="Underlying symbol for F&O data or Option Greeks (e.g., NIFTY, BANKNIFTY)")
    parser.add_argument("--expiry", default="29MAY2025", help="Expiry date for Option Greeks or F&O data (DDMMMYYYY)")
    parser.add_argument("--fetch-gainers-losers", action="store_true", help="Fetch Top Gainers/Losers")
    parser.add_argument("--gainers-losers-type", default="PercOIGainers", help="Data type for Gainers/Losers")
    parser.add_argument("--expiry-type", default="NEAR", help="Expiry type (NEAR, NEXT, FAR)")
    parser.add_argument("--fetch-pcr", action="store_true", help="Fetch PCR Volume data")
    parser.add_argument("--fetch-oi-buildup", action="store_true", help="Fetch OI Buildup data")
    parser.add_argument("--oi-buildup-type", default="Long Built Up", help="Data type for OI Buildup")
    parser.add_argument("--fno", action="store_true", help="Fetch F&O data (ATM call and put options)")
    parser.add_argument("--exchange", default="NSE", help="Exchange for F&O data (e.g., NFO)")
    args = parser.parse_args()
    config = Config()
    setup_logging(config.get('log_level', 'INFO'), config.get('log_file_template', 'data_fetcher_test_{run_id}.log'))
    try:
        broker_client = BrokerClient(config)
        if not broker_client.broker_client_initialised:
            logger.error("Failed to initialize BrokerClient. Exiting.")
            sys.exit(1)
        instrument_manager = InstrumentManager(broker_client, config)
        fetcher = DataFetcher(broker_client, instrument_manager, config)
        if args.fno:
            call_ohlcv_df, call_oi_df, put_ohlcv_df, put_oi_df = fetcher.fetch_fno_data(
                exchange=args.exchange,
                underlying=args.underlying,
                expiry_date=args.expiry,
                duration=args.days,
                timeframe=args.timeframe
            )
            if not call_ohlcv_df.empty and not call_oi_df.empty and not put_ohlcv_df.empty and not put_oi_df.empty:
                logger.info("F&O data fetch successful.")
            else:
                logger.error("F&O data fetch failed.")
        if args.fetch_greeks:
            greeks_df = fetcher.fetch_option_greeks(
                underlying=args.underlying,
                expiry_date=args.expiry
            )
            if not greeks_df.empty:
                logger.info("Option Greeks fetch successful.")
            else:
                logger.error("Option Greeks fetch failed.")
        if args.fetch_gainers_losers:
            gainers_losers_df = fetcher.fetch_top_gainers_losers(
                data_type=args.gainers_losers_type,
                expiry_type=args.expiry_type
            )
            if not gainers_losers_df.empty:
                logger.info("Top Gainers/Losers fetch successful.")
            else:
                logger.error("Top Gainers/Losers fetch failed.")
        if args.fetch_pcr:
            pcr_df = fetcher.fetch_pcr_volume()
            if not pcr_df.empty:
                logger.info("PCR Volume fetch successful.")
            else:
                logger.error("PCR Volume fetch failed.")
        if args.fetch_oi_buildup:
            oi_buildup_df = fetcher.fetch_oi_buildup(
                data_type=args.oi_buildup_type,
                expiry_type=args.expiry_type
            )
            if not oi_buildup_df.empty:
                logger.info("OI Buildup fetch successful.")
            else:
                logger.error("OI Buildup fetch failed.")
    except Exception as e:
        logger.error(f"Script execution failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)
