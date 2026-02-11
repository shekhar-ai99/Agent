# File: real_time_data.py
import pandas as pd
import os
from SmartApi import SmartConnect
from dotenv import load_dotenv
from datetime import datetime, time, timedelta
import pytz
import pyotp

# Load environment variables
load_dotenv()

# Define constants
NIFTY_SYMBOL = "99926000"  # Token for NIFTY 50
REALTIME_INTERVAL = "THREE_MINUTE"
data_folder = "data/"

def initialize_api():
    api_key = os.getenv("ANGELONE_API_KEY")
    client_code = os.getenv("ANGELONE_CLIENT_CODE")
    password = os.getenv("ANGELONE_PASSWORD")
    totp_secret = os.getenv("ANGELONE_TOTP_SECRET")
    
    smart_api = SmartConnect(api_key)
    totp = pyotp.TOTP(totp_secret).now()
    login_data = smart_api.generateSession(client_code, password, totp)
    
    if not login_data["status"]:
        raise Exception(f"Login failed: {login_data}")
    
    return smart_api

def fetch_real_time_data(smart_api):
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    start_time = now - timedelta(days=2)
    
    params = {
        "exchange": "NSE",
        "symboltoken": NIFTY_SYMBOL,
        "interval": REALTIME_INTERVAL,
        "fromdate": start_time.strftime('%Y-%m-%d %H:%M'),
        "todate": now.strftime('%Y-%m-%d %H:%M')
    }
    
    response = smart_api.getCandleData(params)
    if not response["status"] or not response.get("data"):
        raise Exception(f"Error fetching data: {response}")
    
    df = pd.DataFrame(response["data"], columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    return df
# Add to real_time_data.py
def continuous_data_fetch(smart_api, interval_minutes=3):
    """Continuously fetch real-time data"""
    while True:
        try:
            now = datetime.now(pytz.timezone('Asia/Kolkata'))
            start_time = now - timedelta(minutes=interval_minutes*3)
            
            params = {
                "exchange": "NSE",
                "symboltoken": NIFTY_SYMBOL,
                "interval": REALTIME_INTERVAL,
                "fromdate": start_time.strftime('%Y-%m-%d %H:%M'),
                "todate": now.strftime('%Y-%m-%d %H:%M')
            }
            
            response = smart_api.getCandleData(params)
            if response["status"] and response.get("data"):
                df = pd.DataFrame(response["data"], columns=["datetime", "open", "high", "low", "close", "volume"])
                df["datetime"] = pd.to_datetime(df["datetime"])
                df.set_index("datetime", inplace=True)
                
                output_file = os.path.join(data_folder, "nifty_realtime_data.csv")
                df.to_csv(output_file)
                print(f"{now}: Data updated")
            
            time.sleep(interval_minutes * 60)  # Wait for next interval
            
        except Exception as e:
            print(f"Error in continuous fetch: {e}")
            time.sleep(60)  # Wait before retrying

if __name__ == "__main__":
    import sys
    smart_api = initialize_api()
    
    if "--continuous" in sys.argv:
        print("Running in continuous mode...")
        continuous_data_fetch(smart_api)
    else:
        # One-time fetch
        real_time_data = fetch_real_time_data(smart_api)
        real_time_file = os.path.join(data_folder, "nifty_realtime_data.csv")
        real_time_data.to_csv(real_time_file)
        print(f"Real-time data saved to {real_time_file}")