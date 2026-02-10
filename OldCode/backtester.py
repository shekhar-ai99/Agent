import pandas as pd

class SimpleBacktester:
    def __init__(self, strategy, df, initial_balance=10000, trade_size=1, logger=None):
        self.strategy = strategy
        self.df = df
        self.cash = initial_balance
        self.position = 0
        self.trade_size = trade_size
        self.trades = []
        self.logger = logger

    def run(self):
        for _, row in self.df.iterrows():
            signal = self.strategy.on_data(row)
            price = row["close"]
            if signal == "buy" and self.position == 0:
                cost = price * self.trade_size
                if self.cash >= cost:
                    self.cash -= cost
                    self.position = self.trade_size
                    self.trades.append((row["datetime"], "BUY", price, self.trade_size))
                    if self.logger:
                        self.logger.info(f"BUY @ {price}")
            elif signal == "sell" and self.position > 0:
                self.cash += price * self.position
                self.trades.append((row["datetime"], "SELL", price, self.position))
                if self.logger:
                    self.logger.info(f"SELL @ {price}")
                self.position = 0
        final_value = self.cash + self.position * self.df.iloc[-1]["close"]
        return final_value, self.trades
