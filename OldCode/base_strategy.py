class BaseStrategy:
    def __init__(self, config):
        self.params = config

    def on_data(self, row):
        raise NotImplementedError("Must implement on_data method.")
