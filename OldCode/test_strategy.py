# tests/test_strategy.py
import pandas as pd
#import pytest
from strategy.base_strategy import AlphaTrendStrategy

def test_alphatrend_signals():
    data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105],
        'high': [101, 102, 103, 104, 105, 106],
        'low': [99, 100, 101, 102, 103, 104],
        'close': [100, 101, 102, 103, 104, 105],
        'volume': [1000] * 6
    }, index=pd.date_range("2025-04-29 09:15", periods=6, freq="5min", tz="Asia/Kolkata"))
    data['ATR_SMA_10'] = [2] * 6
    data['RSI_10'] = [40, 50, 55, 60, 65, 70]
    data['rsi_slope'] = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
    data['macd'] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    data['macd_signal'] = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    data['SUPERTd_10_3.0'] = [0, 0, 1, 1, 1, 1]
    strategy = AlphaTrendStrategy(coeff=0.6, ap=10, use_rsi_condition=True, gap_threshold=1.0)
    result = strategy.generate_signals(data, prev_close=80)
    assert result['signal'].sum() >= 1, "Expected at least one buy signal"
    assert result['morning_entry'].iloc[0], "Expected morning entry at 09:15"