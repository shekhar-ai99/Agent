"""
Tick Aggregator for Angel One Live Data Feed

Aggregates individual tick data into OHLCV candles at specified intervals.
Used by LiveDataFeeder to convert tick stream into candle stream.
"""

from datetime import datetime
import pytz


class TickAggregator:
    """
    Aggregates ticks into time-based OHLCV candles.
    
    Handles Angel One tick format where:
    - exchange_timestamp is in milliseconds (UTC)
    - last_traded_price is in paise (1/100 of rupee)
    """

    def __init__(self, interval_minutes: int = 5, tz: str = 'Asia/Kolkata'):
        """
        Initialize TickAggregator.
        
        Args:
            interval_minutes: Candle interval in minutes (default 5)
            tz: Timezone string (default 'Asia/Kolkata')
        """
        self.interval_minutes = interval_minutes
        self.tz = pytz.timezone(tz)
        self.current_candle = None
        self.candle_start_time = None

    def process_tick(self, tick: dict) -> dict | None:
        """
        Process a single tick and return completed candle if interval boundary crossed.
        
        Args:
            tick: Tick dict with keys: exchange_timestamp (ms), last_traded_price (paise), volume
            
        Returns:
            Completed candle dict if new interval started, else None
        """
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

    def force_close(self) -> dict | None:
        """
        Force close the current candle (e.g., at market close).
        
        Returns:
            The current candle if one exists, else None
        """
        finished = self.current_candle
        self.current_candle = None
        return finished
