from .base_strategy import BaseStrategy

class MovingAverageCrossover(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.short_window = config.get("short_window", 20)
        self.long_window = config.get("long_window", 50)
        self.prices = []

    def on_data(self, row):
        price = row["close"]
        self.prices.append(price)
        if len(self.prices) < self.long_window:
            return None
        short_ma = sum(self.prices[-self.short_window:]) / self.short_window
        long_ma = sum(self.prices[-self.long_window:]) / self.long_window
        if short_ma > long_ma:
            return "buy"
        elif short_ma < long_ma:
            return "sell"
        return None
