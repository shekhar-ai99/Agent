"""
FEED MAPPING NOTES (STEP 1)

This file documents ALL legacy data feeds found in the repository.
DO NOT MODIFY FEED LOGIC YET. This is an inventory only.

Fields captured per feed:
- broker
- market
- mode (live / historical)
- timeframe support
- file
- entry function / class
- dependencies
"""

FEED_MAP = [
    # --- Angel One (India) ---
    {
        "broker": "angelone",
        "market": "india",
        "mode": "live",
        "timeframes": "tick -> aggregated to candles",
        "file": "IndianMarket/strategy_tester_app/app/LiveDataFeeder.py",
        "entry": "class LiveDataFeeder (start/stop, on_data, TickAggregator)",
        "deps": ["SmartApi", "SmartWebSocketV2", "TickAggregator", "pyotp", "dotenv"],
    },
    {
        "broker": "angelone",
        "market": "india",
        "mode": "live",
        "timeframes": "LTP (tick) + websocket callbacks",
        "file": "IndianMarket/strategy_tester_app/data_fetcher.py",
        "entry": "class SmartWebSocketV2 + DataFetcher.fetch_websocket_ltp",
        "deps": ["websocket", "SmartApi", "InstrumentManager", "BrokerClient"],
    },
    {
        "broker": "angelone",
        "market": "india",
        "mode": "live",
        "timeframes": "LTP/QUOTE/SNAP (SmartWebSocketV2)",
        "file": "IndianMarket/strategy_tester_app/angel_one/angel_one_websocket.py",
        "entry": "class SmartWebSocketV2 (connect/subscribe/resubscribe)",
        "deps": ["websocket", "ssl", "struct"],
    },
    {
        "broker": "angelone",
        "market": "india",
        "mode": "live (example script)",
        "timeframes": "tick",
        "file": "IndianMarket/strategy_tester_app/app/getlivedata.py",
        "entry": "script-level SmartWebSocketV2 usage",
        "deps": ["SmartApi", "SmartWebSocketV2", "TickAggregator"],
    },
    {
        "broker": "angelone",
        "market": "india",
        "mode": "live LTP (REST)",
        "timeframes": "LTP",
        "file": "IndianMarket/BOT/trading_bot_final/data/ltp.py",
        "entry": "script-level http.client POST getLtpData",
        "deps": ["SmartApi", "http.client", "pyotp"],
    },

    # --- Angel One (India) Historical ---
    {
        "broker": "angelone",
        "market": "india",
        "mode": "historical",
        "timeframes": "1min,3min,5min,15min,30min,60min,1day",
        "file": "IndianMarket/strategy_tester_app/angel_one/angel_data_fetcher.py",
        "entry": "class AngelDataFetcher.fetch_historical_candles",
        "deps": ["AngelOneAPI", "InstrumentManager", "SmartConnect"],
    },
    {
        "broker": "angelone",
        "market": "india",
        "mode": "historical",
        "timeframes": "interval passed via HistoricalDataFetcher(interval)",
        "file": "IndianMarket/strategy_tester_app/app/historical_data.py",
        "entry": "class HistoricalDataFetcher.fetch_historical_data / fetch_historical_candles",
        "deps": ["SmartApi", "AngelOneAPI", "InstrumentManager", "pyotp"],
    },
    {
        "broker": "angelone",
        "market": "india",
        "mode": "historical",
        "timeframes": "online/offline; interval mapping via map_interval_to_smartapi",
        "file": "IndianMarket/BOT/trading_bot_final/data/data_fetcher.py",
        "entry": "fetch_data(source=online/offline)",
        "deps": ["SmartApi", "requests", "pyotp", "OFFLINE_FILES config"],
    },
    {
        "broker": "angelone",
        "market": "india",
        "mode": "historical",
        "timeframes": "interval from HistoricalDataFetcher(interval)",
        "file": "IndianMarket/BOT/trading_bot_final/data/historical_data.py",
        "entry": "class HistoricalDataFetcher.fetch_historical_data",
        "deps": ["SmartApi", "pyotp"],
    },
    {
        "broker": "angelone",
        "market": "india",
        "mode": "historical (online/offline)",
        "timeframes": "interval map + offline CSV fallback",
        "file": "IndianMarket/BOT/trading_bot_final/data/fetcher_v2.py",
        "entry": "class HistoricalDataFetcher.fetch_data",
        "deps": ["SmartApi", "pyotp", "pandas"],
    },
    {
        "broker": "angelone",
        "market": "india",
        "mode": "historical (csv loader)",
        "timeframes": "csv based",
        "file": "IndianMarket/strategy_tester_app/app/load_data.py",
        "entry": "class HistoricalDataLoader.load_historical_data / fill_missing_ohlcv_bars",
        "deps": ["pandas", "pytz"],
    },

    # --- CoinDCX (Crypto) ---
    {
        "broker": "coindcx",
        "market": "crypto",
        "mode": "historical",
        "timeframes": "1m,3m,5m,15m (interval param in get_candlestick_data)",
        "file": "IndianMarket/strategy_tester_app/app/getCoinDCX.py",
        "entry": "get_candlestick_data(pair, interval, start_time, end_time)",
        "deps": ["requests", "pandas"],
    },

    # --- Delta Exchange (Crypto) ---
    {
        "broker": "delta",
        "market": "crypto",
        "mode": "api client (no feed found)",
        "timeframes": "n/a",
        "file": "Bitcoin/bitcoin_strategy_tester_package/bitcoin_strategy_tester/api.py",
        "entry": "class DeltaAPIClient.get_ticker / place_order",
        "deps": ["requests", "hmac", "hashlib"],
    },

    # --- Legacy references (placeholders, code missing in repo) ---
    {
        "broker": "angelone",
        "market": "india",
        "mode": "live/historical (legacy reference only)",
        "timeframes": "n/a",
        "file": "TradingPlatform/execution/modes/real_mode.py",
        "entry": "SmartAPIClient (brokers/angel_one_backup/angel_one_api.py) - NOT FOUND",
        "deps": ["missing"],
    },
    {
        "broker": "delta",
        "market": "crypto",
        "mode": "live/historical (legacy reference only)",
        "timeframes": "n/a",
        "file": "TradingPlatform/execution/modes/real_mode.py",
        "entry": "DeltaClient (brokers/delta_backup/delta_api.py) - NOT FOUND",
        "deps": ["missing"],
    },
]
