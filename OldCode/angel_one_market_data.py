# angel_one_market_data.py

import requests
import logging

logger = logging.getLogger(__name__)

class AngelMarketDataClient:
    def __init__(self, access_token: str, client_local_ip="127.0.0.1", client_public_ip="127.0.0.1", mac_address="00:00:00:00:00:00"):
        self.base_url = "https://apiconnect.angelone.in/rest/secure/angelbroking/marketData/v1"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-UserType": "USER",
            "X-SourceID": "WEB",
            "X-ClientLocalIP": client_local_ip,
            "X-ClientPublicIP": client_public_ip,
            "X-MACAddress": mac_address
        }

    def get_top_gainers_losers(self, datatype="PercOIGainers", expirytype="NEAR"):
        url = f"{self.base_url}/gainersLosers"
        payload = {
            "datatype": datatype,
            "expirytype": expirytype
        }
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error fetching top gainers/losers: {response.status_code} - {response.text}")
            return None

    def get_pcr_volume(self):
        url = f"{self.base_url}/putCallRatio"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error fetching PCR volume: {response.status_code} - {response.text}")
            return None

    def get_oi_buildup(self, datatype="Long Built Up", expirytype="NEAR"):
        url = f"{self.base_url}/OIBuildup"
        payload = {
            "datatype": datatype,
            "expirytype": expirytype
        }
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error fetching OI buildup: {response.status_code} - {response.text}")
            return None

    def get_option_greeks(self, name: str, expirydate: str):
        url = f"{self.base_url}/optionGreek"
        payload = {
            "name": name,
            "expirydate": expirydate
        }
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error fetching option greeks: {response.status_code} - {response.text}")
            return None