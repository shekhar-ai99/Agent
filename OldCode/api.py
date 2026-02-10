import requests
import time
import hmac
import hashlib

class DeltaAPIClient:
    def __init__(self, api_key, api_secret, base_url):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url

    def _generate_signature(self, method, path, timestamp, body=""):
        message = f"{timestamp}{method.upper()}{path}{body}".encode()
        return hmac.new(self.api_secret.encode(), message, hashlib.sha256).hexdigest()

    def _headers(self, method, path, body=""):
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(method, path, timestamp, body)
        return {
            "api-key": self.api_key,
            "timestamp": timestamp,
            "signature": signature,
            "Content-Type": "application/json"
        }

    def get_ticker(self, product_symbol="BTCUSD"):
        path = f"/v2/tickers/{product_symbol}"
        url = self.base_url + path
        headers = self._headers("GET", path)
        response = requests.get(url, headers=headers)
        return response.json()

    def place_order(self, product_symbol, size, side="buy", order_type="market"):
        path = "/v2/orders"
        url = self.base_url + path
        body = {
            "product_symbol": product_symbol,
            "size": size,
            "side": side,
            "order_type": order_type
        }
        import json
        body_json = json.dumps(body)
        headers = self._headers("POST", path, body_json)
        response = requests.post(url, headers=headers, data=body_json)
        return response.json()
