import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def get_candlestick_data(pair, interval, start_time, end_time, limit=1000, max_retries=3):
    interval_seconds = {"15m": 900}[interval]
    # interval_seconds = {"1m": 60, "3m": 180, "5m": 300, "15m": 900}[interval]
    all_candles = []
    current_start = start_time

    while current_start < end_time:
        chunk_end = min(current_start + timedelta(seconds=interval_seconds * limit), end_time)
        start_ts = int(current_start.timestamp() * 1000)
        end_ts = int(chunk_end.timestamp() * 1000)
        url = (f"https://public.coindcx.com/market_data/candles"
               f"?pair={pair}&interval={interval}&startTime={start_ts}&endTime={end_ts}&limit={limit}")

        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        all_candles.extend(data)
                        print(f"Fetched {len(data)} candles from {current_start} to {chunk_end} ({interval})")
                    else:
                        print(f"Empty data for {current_start} to {chunk_end} ({interval})")
                    break
                else:
                    print(f"API Error ({response.status_code}) at {current_start}, attempt {attempt+1}")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception at {current_start}, attempt {attempt+1}: {e}")
                time.sleep(2)
        else:
            print(f"Failed to fetch data for {current_start} after {max_retries} attempts. Skipping.")
        
        current_start = chunk_end + timedelta(seconds=interval_seconds)
        time.sleep(0.1)
    
    return all_candles

def save_candlestick_data_to_csv(data, pair, interval, filename):
    if data:
        # CoinDCX returns [time, open, high, low, close, volume] for each candle
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"])
        # Cast numeric fields if needed
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Convert timestamp
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        # Remove duplicates and sort by time
        df = df.drop_duplicates(subset=["time"]).sort_values("time")
        
        # Parse symbol from pair (assumes format "B-BTC_USDT")
        symbol = pair.split('-')[-1]
        dir_path = os.path.join("data", "coinDCX", symbol)
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, filename)
        
        df.to_csv(file_path, index=False)
        print(f"Saved {interval} data to {file_path} ({len(df)} rows)")
    else:
        print(f"No data to save for {interval}")

if __name__ == "__main__":
    start = datetime(2025, 5, 1)
    end = datetime(2025, 6, 20, 23, 59, 59)
    intervals = ["1m", "3m", "5m", "15m"]
    pair = "B-BTC_USDT"
    symbol = pair.split('-')[-1]
    for interval in intervals:
        print(f"Fetching {interval} candlestick data...")
        candles = get_candlestick_data(pair, interval, start, end)
        filename = f"btcusd_{interval}_20250501_20250620.csv"
        save_candlestick_data_to_csv(candles, pair, interval, filename)
