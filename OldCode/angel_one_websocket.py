# angel_one_websocket.py

import logging
import websocket
import ssl
import json
import struct
import time

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