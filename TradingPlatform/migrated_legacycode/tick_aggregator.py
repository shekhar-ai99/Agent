# import os
# import pandas as pd
# import pytz
# from datetime import datetime
# from collections import defaultdict

# class TickAggregator:
#     def __init__(self, symbol, token, exchange_type, timeframes, on_candle):
#         self.symbol = symbol
#         self.token = token
#         self.exchange_type = exchange_type
#         self.timeframes = timeframes
#         self.on_candle = on_candle
#         self.TIMEZ = pytz.timezone("Asia/Kolkata")
#         self.tick_buffer = defaultdict(list)
#         self.last_candle_time = {tf: None for tf in timeframes}

#     def get_candle_time(self, ts, tf):
#         dt = ts.replace(second=0, microsecond=0)
#         minute = (dt.minute // tf) * tf
#         return dt.replace(minute=minute)

#     def aggregate_ticks(self, ticks, tf):
#         if not ticks: return None
#         opens = ticks[0]['ltp']
#         closes = ticks[-1]['ltp']
#         highs = max(t['ltp'] for t in ticks)
#         lows = min(t['ltp'] for t in ticks)
#         vols = sum(t.get('qty', 0) for t in ticks)
#         candle_time = self.get_candle_time(ticks[0]['dt'], tf)
#         return {
#             'datetime': candle_time,
#             'open': opens,
#             'high': highs,
#             'low': lows,
#             'close': closes,
#             'volume': vols
#         }

#     def on_tick(self, message):
#         ts = datetime.fromtimestamp(message['exchange_timestamp']/1000, tz=self.TIMEZ)
#         tick = {
#             'dt': ts,
#             'ltp': message['last_traded_price'] / 100.0,
#             'qty': 0  # Fill from snap-quote if you want actual volume
#         }
#         for tf in self.timeframes:
#             c_time = self.get_candle_time(ts, tf)
#             if self.last_candle_time[tf] is not None and c_time > self.last_candle_time[tf]:
#                 prev_ticks = [tick for tick in self.tick_buffer[tf] if self.get_candle_time(tick['dt'], tf) == self.last_candle_time[tf]]
#                 ohlcv = self.aggregate_ticks(prev_ticks, tf)
#                 if ohlcv:
#                     self.on_candle(tf, ohlcv)
#                 # Remove those ticks
#                 self.tick_buffer[tf] = [t for t in self.tick_buffer[tf] if self.get_candle_time(t['dt'], tf) != self.last_candle_time[tf]]
#             self.tick_buffer[tf].append(tick)
#             self.last_candle_time[tf] = c_time

#     def run(self, ws_connect_func):
#         """
#         Start the WebSocket and process ticks. 
#         ws_connect_func: function(on_tick_callback) that connects and passes each tick to on_tick_callback.
#         """
#         ws_connect_func(self.on_tick)
# tick_aggregator.py
from datetime import datetime, timedelta
import pytz

class TickAggregator:
    def __init__(self, interval_minutes=5, tz='Asia/Kolkata'):
        self.interval_minutes = interval_minutes
        self.tz = pytz.timezone(tz)
        self.current_candle = None
        self.candle_start_time = None

    def process_tick(self, tick):
        # Angel One: tick['exchange_timestamp'] is ms epoch (UTC)
        ts_utc = int(tick.get('exchange_timestamp')) // 1000  # ms to s
        dt_utc = datetime.utcfromtimestamp(ts_utc).replace(tzinfo=pytz.utc)
        dt = dt_utc.astimezone(self.tz)

        # AngelOne gives LTP in paise (divide by 100)
        price = tick.get('last_traded_price') / 100.0

        # Round down to interval
        minute = (dt.minute // self.interval_minutes) * self.interval_minutes
        candle_time = dt.replace(minute=minute, second=0, microsecond=0)

        # Start new candle if it's a new interval
        if (self.current_candle is None) or (candle_time != self.candle_start_time):
            finished = self.current_candle
            self.current_candle = {
                'datetime': candle_time,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': tick.get('volume', 0)  # Use 0 if not present
            }
            self.candle_start_time = candle_time
            return finished
        else:
            # Update current candle
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price
            # If tick has volume, accumulate; else just count ticks if you want
            self.current_candle['volume'] += tick.get('volume', 0)
            return None

    def force_close(self):
        finished = self.current_candle
        self.current_candle = None
        return finished
