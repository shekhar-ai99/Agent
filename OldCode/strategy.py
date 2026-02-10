from abc import ABC, abstractmethod
import math
import pandas as pd
import numpy as np
from typing import Dict, Optional

class Strategy(ABC):
    include = True

    @abstractmethod
    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        pass

class EMACrossover(Strategy):
    include = True
    required_indicators = ['ema_9', 'ema_21', 'adx', 'volume', 'vol_ma', 'atr', 'close']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        adx_threshold = params.get('adx_threshold', 25)
        volume_multiplier = params.get('volume_multiplier', 1.5)
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        if idx == 0:
            return 'hold'
        prev_row = data.iloc[idx-1]
        if current_row['adx'] < adx_threshold or current_row['volume'] < current_row['vol_ma'] * volume_multiplier:
            return 'hold'
        if current_row['ema_9'] > current_row['ema_21'] and prev_row['ema_9'] <= prev_row['ema_21']:
            data.at[current_row.name, 'ema_crossover_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'ema_crossover_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif current_row['ema_9'] < current_row['ema_21'] and prev_row['ema_9'] >= prev_row['ema_21']:
            data.at[current_row.name, 'ema_crossover_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'ema_crossover_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class RSIThreshold(Strategy):
    include = True
    required_indicators = ['rsi', 'adx', 'atr', 'close']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        rsi_oversold = params.get('rsi_oversold', 35)
        rsi_overbought = params.get('rsi_overbought', 65)
        adx_threshold = params.get('adx_threshold', 20)
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        if idx == 0:
            return 'hold'
        prev_row = data.iloc[idx-1]
        if current_row['adx'] < adx_threshold:
            return 'hold'
        if current_row['rsi'] > rsi_oversold and prev_row['rsi'] <= rsi_oversold:
            data.at[current_row.name, 'rsi_threshold_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'rsi_threshold_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif current_row['rsi'] < rsi_overbought and prev_row['rsi'] >= rsi_overbought:
            data.at[current_row.name, 'rsi_threshold_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'rsi_threshold_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class MACDCross(Strategy):
    include = True
    required_indicators = ['macd', 'macd_signal', 'macd_hist', 'ema_50', 'atr', 'close']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        ema_trend_period = params.get('ema_trend_period', 50)
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        if idx == 0:
            return 'hold'
        prev_row = data.iloc[idx-1]
        if current_row['close'] < current_row[f'ema_{ema_trend_period}'] and current_row['macd'] > current_row['macd_signal']:
            return 'hold'
        if current_row['macd'] > current_row['macd_signal'] and prev_row['macd'] <= prev_row['macd_signal'] and current_row['macd_hist'] > 0:
            data.at[current_row.name, 'macd_cross_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'macd_cross_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif current_row['macd'] < current_row['macd_signal'] and prev_row['macd'] >= prev_row['macd_signal'] and current_row['macd_hist'] < 0:
            data.at[current_row.name, 'macd_cross_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'macd_cross_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class BBBreakout(Strategy):
    include = True
    required_indicators = ['bollinger_bandwidth', 'close', 'bollinger_upper', 'bollinger_lower', 'adx', 'volume', 'vol_ma', 'atr']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        adx_threshold = params.get('adx_threshold', 25)
        volume_multiplier = params.get('volume_multiplier', 1.5)
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        squeeze_lookback = params.get('squeeze_lookback', 20)
        if idx < squeeze_lookback:
            return 'hold'
        prev_rows = data.iloc[idx-squeeze_lookback:idx]
        bandwidth_threshold = prev_rows['bollinger_bandwidth'].quantile(0.15)
        prev_bandwidth = data.iloc[idx-1]['bollinger_bandwidth']
        if pd.isna(prev_bandwidth):
            return 'hold'
        in_squeeze_prev = prev_bandwidth < bandwidth_threshold
        breaking_out_up = current_row['close'] > current_row['bollinger_upper']
        breaking_out_down = current_row['close'] < current_row['bollinger_lower']
        if in_squeeze_prev and breaking_out_up and current_row['adx'] > adx_threshold and current_row['volume'] > current_row['vol_ma'] * volume_multiplier:
            data.at[current_row.name, 'bb_breakout_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'bb_breakout_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif in_squeeze_prev and breaking_out_down and current_row['adx'] > adx_threshold and current_row['volume'] > current_row['vol_ma'] * volume_multiplier:
            data.at[current_row.name, 'bb_breakout_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'bb_breakout_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class CombinedMomentum(Strategy):
    include = True
    required_indicators = ['ema_9', 'ema_21', 'rsi', 'bollinger_bandwidth', 'atr', 'close']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        rsi_confirm_buy = params.get('rsi_confirm_buy', 55)
        rsi_confirm_sell = params.get('rsi_confirm_sell', 45)
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        if idx == 0:
            return 'hold'
        prev_row = data.iloc[idx-1]
        bandwidth_threshold = data.iloc[idx-20:idx]['bollinger_bandwidth'].quantile(0.25) if idx >= 20 else np.inf
        if current_row['bollinger_bandwidth'] < bandwidth_threshold:
            return 'hold'
        ema_crossed_up = current_row['ema_9'] > current_row['ema_21'] and prev_row['ema_9'] <= prev_row['ema_21']
        ema_crossed_down = current_row['ema_9'] < current_row['ema_21'] and prev_row['ema_9'] >= prev_row['ema_21']
        if ema_crossed_up and current_row['rsi'] > rsi_confirm_buy:
            data.at[current_row.name, 'combined_momentum_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'combined_momentum_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif ema_crossed_down and current_row['rsi'] < rsi_confirm_sell:
            data.at[current_row.name, 'combined_momentum_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'combined_momentum_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class AIPoweredV6(Strategy):
    include = True
    required_indicators = ['ema_9', 'ema_50', 'macd', 'macd_signal', 'macd_hist', 'rsi', 'plus_di', 'minus_di', 'adx', 'close', 'atr']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        ema9_length = params.get('ema9_length', 9)
        ema50_length = params.get('ema50_length', 50)
        adx_threshold = params.get('adx_threshold', 20)
        ai_buy_threshold = params.get('ai_buy_threshold', 0.2)
        ai_sell_threshold = params.get('ai_sell_threshold', -0.2)
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        if idx == 0:
            return 'hold'
        prev_row = data.iloc[idx - 1]
        if pd.isna(prev_row['macd']) or pd.isna(prev_row['macd_signal']):
            return 'hold'
        macd_line = current_row['macd']
        macd_signal_line = current_row['macd_signal']
        macd_hist = current_row['macd_hist']
        rsi = current_row['rsi']
        adx = current_row['adx']
        close = current_row['close']
        ema9 = current_row[f'ema_{ema9_length}']
        ema50 = current_row[f'ema_{ema50_length}']
        hist_lookback = params.get('hist_lookback', 5)
        if idx >= hist_lookback:
            trend_factor = data['macd_hist'].iloc[idx-hist_lookback+1:idx+1].mean()
            if pd.isna(trend_factor):
                trend_factor = macd_hist
        else:
            trend_factor = macd_hist
        w_trend = params.get('w_trend', 1.0)
        w_rsi = params.get('w_rsi', 1.0)
        rsi_factor = (rsi - 50) / 10
        ai_confidence = (w_trend * trend_factor + w_rsi * rsi_factor) / max((w_trend + w_rsi), 1e-6)
        ai_buy_signal = ai_confidence > ai_buy_threshold
        ai_sell_signal = ai_confidence < ai_sell_threshold
        is_trending = adx > adx_threshold
        macd_crossover = macd_line > macd_signal_line and prev_row['macd'] <= prev_row['macd_signal']
        macd_crossunder = macd_line < macd_signal_line and prev_row['macd'] >= prev_row['macd_signal']
        close_gt_ema9 = close > ema9
        close_gt_ema50 = close > ema50
        close_lt_ema9 = close < ema9
        close_lt_ema50 = close < ema50
        if macd_crossover and ai_buy_signal and close_gt_ema9 and close_gt_ema50 and is_trending:
            data.at[current_row.name, 'ai_powered_v6_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'ai_powered_v6_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif macd_crossunder and ai_sell_signal and close_lt_ema9 and close_lt_ema50 and is_trending:
            data.at[current_row.name, 'ai_powered_v6_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'ai_powered_v6_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class BBADXTrend(Strategy):
    include = True
    required_indicators = ['bollinger_upper', 'bollinger_lower', 'adx', 'plus_di', 'minus_di', 'close', 'ema_50', 'atr']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        adx_trend_threshold = params.get('adx_trend_threshold', 30)
        ema_period = params.get('ema_period', 50)
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        if idx == 0:
            return 'hold'
        prev_row = data.iloc[idx - 1]
        if pd.isna(prev_row['close']) or pd.isna(prev_row['bollinger_upper']) or pd.isna(prev_row['bollinger_lower']):
            return 'hold'
        bb_upper = current_row['bollinger_upper']
        bb_lower = current_row['bollinger_lower']
        adx = current_row['adx']
        plus_di = current_row['plus_di']
        minus_di = current_row['minus_di']
        close = current_row['close']
        is_trending = adx > adx_trend_threshold
        is_uptrend_dmi = plus_di > minus_di
        is_downtrend_dmi = minus_di > plus_di
        prev_close_inside_bands = (prev_row['close'] < prev_row['bollinger_upper']) and (prev_row['close'] > prev_row['bollinger_lower'])
        long_condition = (close > bb_upper) and prev_close_inside_bands and is_trending and is_uptrend_dmi and close > current_row[f'ema_{ema_period}']
        short_condition = (close < bb_lower) and prev_close_inside_bands and is_trending and is_downtrend_dmi and close < current_row[f'ema_{ema_period}']
        if long_condition:
            data.at[current_row.name, 'bb_adx_trend_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'bb_adx_trend_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif short_condition:
            data.at[current_row.name, 'bb_adx_trend_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'bb_adx_trend_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class RSIConfirmed(Strategy):
    include = True
    required_indicators = ['rsi', 'macd', 'macd_signal', 'atr', 'close']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        rsi_oversold = params.get('rsi_oversold', 35)
        rsi_overbought = params.get('rsi_overbought', 65)
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        if idx == 0:
            return 'hold'
        prev_row = data.iloc[idx-1]
        if pd.isna(prev_row['rsi']):
            return 'hold'
        rsi_crossed_above_os = current_row['rsi'] > rsi_oversold and prev_row['rsi'] <= rsi_oversold
        rsi_crossed_below_ob = current_row['rsi'] < rsi_overbought and prev_row['rsi'] >= rsi_overbought
        macd_bullish = current_row['macd'] > current_row['macd_signal'] and current_row['macd'] > 0
        macd_bearish = current_row['macd'] < current_row['macd_signal'] and current_row['macd'] < 0
        if rsi_crossed_above_os and macd_bullish:
            data.at[current_row.name, 'rsi_confirmed_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'rsi_confirmed_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif rsi_crossed_below_ob and macd_bearish:
            data.at[current_row.name, 'rsi_confirmed_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'rsi_confirmed_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class EMACrossoverFiltered(Strategy):
    include = True
    required_indicators = ['ema_9', 'ema_21', 'ema_50', 'close', 'atr']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        ema_fast = params.get('ema_fast', 9)
        ema_slow = params.get('ema_slow', 21)
        ema_trend = params.get('ema_trend', 50)
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        if idx == 0:
            return 'hold'
        prev_row = data.iloc[idx-1]
        if pd.isna(prev_row[f'ema_{ema_fast}']) or pd.isna(prev_row[f'ema_{ema_slow}']):
            return 'hold'
        crossed_up = current_row[f'ema_{ema_fast}'] > current_row[f'ema_{ema_slow}'] and prev_row[f'ema_{ema_fast}'] <= prev_row[f'ema_{ema_slow}']
        crossed_down = current_row[f'ema_{ema_fast}'] < current_row[f'ema_{ema_slow}'] and prev_row[f'ema_{ema_fast}'] >= prev_row[f'ema_{ema_slow}']
        is_uptrend = current_row['close'] > current_row[f'ema_{ema_trend}']
        is_downtrend = current_row['close'] < current_row[f'ema_{ema_trend}']
        atr_filter = current_row['atr'] > data['atr'].rolling(20).mean().iloc[idx] if idx >= 20 else True
        if crossed_up and is_uptrend and atr_filter:
            data.at[current_row.name, 'ema_crossover_filtered_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'ema_crossover_filtered_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif crossed_down and is_downtrend and atr_filter:
            data.at[current_row.name, 'ema_crossover_filtered_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'ema_crossover_filtered_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class ConsolidatedSimplified(Strategy):
    include = True
    required_indicators = ['ema_9', 'ema_14', 'ema_21', 'macd', 'macd_signal', 'rsi', 'volume', 'vol_ma', 'plus_di', 'minus_di', 'adx', 'close', 'open', 'atr']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        ema_fast_len = params.get('ema_fast_len', 9)
        ema_med_len = params.get('ema_med_len', 14)
        ema_slow_len = params.get('ema_slow_len', 21)
        rsi_buy_level = params.get('rsi_buy_level', 55.0)
        rsi_sell_level = params.get('rsi_sell_level', 45.0)
        adx_threshold = params.get('adx_threshold', 25.0)
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        if idx == 0:
            return 'hold'
        prev_row = data.iloc[idx - 1]
        ema_fast = current_row[f'ema_{ema_fast_len}']
        ema_med = current_row[f'ema_{ema_med_len}']
        ema_slow = current_row[f'ema_{ema_slow_len}']
        macd_line = current_row['macd']
        signal_line = current_row['macd_signal']
        price_rsi = current_row['rsi']
        di_pos = current_row['plus_di']
        di_neg = current_row['minus_di']
        adx_val = current_row['adx']
        close = current_row['close']
        cond_ema_fast_slow_cross_buy = macd_line > signal_line and prev_row['macd'] <= prev_row['macd_signal']
        cond_ema_fast_slow_cross_sell = macd_line < signal_line and prev_row['macd'] >= prev_row['macd_signal']
        cond_rsi_ok_buy = price_rsi > rsi_buy_level
        cond_rsi_ok_sell = price_rsi < rsi_sell_level
        cond_ema_trend_ok_buy = ema_med > ema_slow
        cond_ema_trend_ok_sell = ema_med < ema_slow
        cond_adx_strength_ok = adx_val > adx_threshold
        cond_adx_direction_ok_buy = di_pos > di_neg
        cond_adx_direction_ok_sell = di_neg > di_pos
        cond_adx_filter_ok_buy = cond_adx_strength_ok and cond_adx_direction_ok_buy
        cond_adx_filter_ok_sell = cond_adx_strength_ok and cond_adx_direction_ok_sell
        if (cond_ema_fast_slow_cross_buy and cond_rsi_ok_buy and cond_ema_trend_ok_buy and cond_adx_filter_ok_buy):
            data.at[current_row.name, 'consolidated_simplified_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'consolidated_simplified_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif (cond_ema_fast_slow_cross_sell and cond_rsi_ok_sell and cond_ema_trend_ok_sell and cond_adx_filter_ok_sell):
            data.at[current_row.name, 'consolidated_simplified_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'consolidated_simplified_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class ZLEMAKalmanCross(Strategy):
    include = True
    required_indicators = ['zlema_8', 'zlema_21', 'adx', 'atr', 'close']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        period1 = params.get('period1', 8)
        period2 = params.get('period2', 21)
        show_cross = params.get('show_cross', True)
        adx_threshold = params.get('adx_threshold', 20)
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        if not show_cross:
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        if idx == 0:
            return 'hold'
        prev_row = data.iloc[idx - 1]
        if prev_row[self.required_indicators].isna().any():
            return 'hold'
        zlema1_curr = current_row[f'zlema_{period1}']
        zlema2_curr = current_row[f'zlema_{period2}']
        zlema1_prev = prev_row[f'zlema_{period1}']
        zlema2_prev = prev_row[f'zlema_{period2}']
        crossed_above = zlema1_curr > zlema2_curr and zlema1_prev <= zlema2_prev
        crossed_below = zlema1_curr < zlema2_curr and zlema1_prev >= zlema2_prev
        if crossed_above and current_row['adx'] > adx_threshold:
            data.at[current_row.name, 'zlema_kalman_cross_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'zlema_kalman_cross_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif crossed_below and current_row['adx'] > adx_threshold:
            data.at[current_row.name, 'zlema_kalman_cross_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'zlema_kalman_cross_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class ICTTurtleSoup(Strategy):
    include = True
    required_indicators = ['high', 'low', 'close', 'open', 'htf_high_shifted', 'htf_low_shifted', 'mss_high_shifted', 'mss_low_shifted', 'volume', 'vol_ma', 'atr']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        bar_length = params.get('bar_length', 20)
        mss_offset = params.get('mss_offset', 10)
        breakout_method = params.get('breakout_method', 'Wick')
        volume_multiplier = params.get('volume_multiplier', 1.5)
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx_loc = data.index.get_loc(current_row.name)
        if idx_loc < bar_length:
            return 'hold'
        current_high = current_row['high']
        current_low = current_row['low']
        current_close = current_row['close']
        prev_mss_high = current_row['mss_high_shifted']
        prev_mss_low = current_row['mss_low_shifted']
        break_high_price = current_high if breakout_method == 'Wick' else current_close
        break_low_price = current_low if breakout_method == 'Wick' else current_close
        lookback_start_index = idx_loc - bar_length
        if lookback_start_index < 0:
            lookback_start_index = 0
        recent_df_slice = data.iloc[lookback_start_index:idx_loc]
        recent_high_liq_grab = (recent_df_slice['high'].values > recent_df_slice['htf_high_shifted'].values).any()
        recent_low_liq_grab = (recent_df_slice['low'].values < recent_df_slice['htf_low_shifted'].values).any()
        high_volume = current_row['volume'] > current_row['vol_ma'] * volume_multiplier
        if recent_high_liq_grab and break_low_price < prev_mss_low and high_volume:
            data.at[current_row.name, 'ict_turtle_soup_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'ict_turtle_soup_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        elif recent_low_liq_grab and break_high_price > prev_mss_high and high_volume:
            data.at[current_row.name, 'ict_turtle_soup_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'ict_turtle_soup_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        return 'hold'

class ChannelBreakout(Strategy):
    include = True
    required_indicators = ['highest_10_shifted', 'lowest_10_shifted', 'high', 'low', 'adx', 'atr', 'close']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        length = params.get('length', 10)
        adx_threshold = params.get('adx_threshold', 25)
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        prev_upBound = current_row[f'highest_{length}_shifted']
        prev_downBound = current_row[f'lowest_{length}_shifted']
        current_high = current_row['high']
        current_low = current_row['low']
        if current_high > prev_upBound and current_row['adx'] > adx_threshold:
            data.at[current_row.name, 'channel_breakout_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'channel_breakout_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif current_low < prev_downBound and current_row['adx'] > adx_threshold:
            data.at[current_row.name, 'channel_breakout_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'channel_breakout_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class CSAlpha(Strategy):
    include = True
    required_indicators = ['close', 'ema_50', 'rsi', 'adx', 'volume', 'vol_ma', 'atr']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        ema_period = params.get('ema_period', 50)
        rsi_period = params.get('rsi_period', 14)
        volume_multiplier = params.get('volume_multiplier', 2.0)
        adx_threshold = params.get('adx_threshold', 25)
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        if idx < 20:
            return 'hold'
        prev_row = data.iloc[idx-1]
        ema_slope = (current_row[f'ema_{ema_period}'] - prev_row[f'ema_{ema_period}']) / prev_row[f'ema_{ema_period}']
        rsi_divergence = current_row['rsi'] - data['rsi'].rolling(20).mean().iloc[idx]
        volume_spike = current_row['volume'] > current_row['vol_ma'] * volume_multiplier
        is_trending = current_row['adx'] > adx_threshold
        if ema_slope > 0 and rsi_divergence > 0 and volume_spike and is_trending:
            data.at[current_row.name, 'cs_alpha_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'cs_alpha_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif ema_slope < 0 and rsi_divergence < 0 and volume_spike and is_trending:
            data.at[current_row.name, 'cs_alpha_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'cs_alpha_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class AIAdaptive(Strategy):
    include = True
    required_indicators = ['close', 'high', 'low', 'atr', 'adx', 'rsi', 'volume', 'vol_ma', 'ema_9', 'ema_21']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        base_ema_fast = params.get('base_ema_fast', 9)
        base_ema_slow = params.get('base_ema_slow', 21)
        rsi_buy_level = params.get('rsi_buy_level', 55)
        rsi_sell_level = params.get('rsi_sell_level', 45)
        adx_threshold = params.get('adx_threshold', 25)
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        if idx < 20:
            return 'hold'
        prev_row = data.iloc[idx-1]
        atr_ratio = current_row['atr'] / data['atr'].rolling(20).mean().iloc[idx]
        ema_fast = int(base_ema_fast / atr_ratio) if atr_ratio > 0 else base_ema_fast
        ema_slow = int(base_ema_slow / atr_ratio) if atr_ratio > 0 else base_ema_slow
        ema_fast = max(3, min(50, ema_fast))
        ema_slow = max(10, min(100, ema_slow))
        if f'ema_{ema_fast}' not in data.columns or f'ema_{ema_slow}' not in data.columns:
            return 'hold'
        crossed_up = current_row[f'ema_{ema_fast}'] > current_row[f'ema_{ema_slow}'] and prev_row[f'ema_{ema_fast}'] <= prev_row[f'ema_{ema_slow}']
        crossed_down = current_row[f'ema_{ema_fast}'] < current_row[f'ema_{ema_slow}'] and prev_row[f'ema_{ema_fast}'] >= prev_row[f'ema_{ema_slow}']
        is_trending = current_row['adx'] > adx_threshold
        volume_confirmed = current_row['volume'] > current_row['vol_ma']
        if crossed_up and current_row['rsi'] > rsi_buy_level and is_trending and volume_confirmed:
            data.at[current_row.name, 'ai_adaptive_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'ai_adaptive_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif crossed_down and current_row['rsi'] < rsi_sell_level and is_trending and volume_confirmed:
            data.at[current_row.name, 'ai_adaptive_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'ai_adaptive_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class EnhancedTrading(Strategy):
    include = True
    required_indicators = [
        'ema_9', 'ema_21', 'rsi', 'bollinger_lower', 'bollinger_mid', 'bollinger_upper',
        'atr', 'adx', 'SUPERT_10_3.0', 'macd', 'macd_signal', 'stochastic_k', 'close'
    ]

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        state = state or {'regime': 'wait', 'prev_signal': 0, 'prev_sl': None, 'prev_tp': None}
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        if idx < 20:
            state['regime'] = 'wait'
            return 'hold'
        prev_row = data.iloc[idx-1]
        bb_squeeze = (current_row['bollinger_upper'] - current_row['bollinger_lower']) < 1.5 * current_row['atr']
        strong_trend = (current_row['adx'] > 25) and (
            abs(current_row['ema_9'] - current_row['ema_21']) > 0.005 * current_row['close']
        )
        macd_bullish = (
            current_row['macd'] > current_row['macd_signal'] and
            prev_row['macd'] <= prev_row['macd_signal']
        )
        macd_bearish = (
            current_row['macd'] < current_row['macd_signal'] and
            prev_row['macd'] >= prev_row['macd_signal']
        )
        if strong_trend and (macd_bullish or macd_bearish):
            state['regime'] = 'trend'
        elif bb_squeeze and current_row['adx'] < 20:
            state['regime'] = 'range'
        else:
            state['regime'] = 'wait'
        signal = 0
        sl = np.nan
        tp = np.nan
        if state['regime'] == 'trend':
            if (
                current_row['ema_9'] > current_row['ema_21'] and
                current_row['close'] > current_row['SUPERT_10_3.0'] and
                current_row['macd'] > 0
            ):
                signal = 1
                sl = current_row['close'] - 1.5 * current_row['atr']
                tp = current_row['close'] + 2.5 * current_row['atr']
            elif (
                current_row['ema_9'] < current_row['ema_21'] and
                current_row['close'] < current_row['SUPERT_10_3.0'] and
                current_row['macd'] < 0
            ):
                signal = -1
                sl = current_row['close'] + 1.5 * current_row['atr']
                tp = current_row['close'] - 2.5 * current_row['atr']
        elif state['regime'] == 'range':
            if (
                current_row['close'] <= current_row['bollinger_lower'] and
                current_row['rsi'] < 35 and
                current_row['stochastic_k'] < 30
            ):
                signal = 1
                sl = current_row['close'] - 0.8 * current_row['atr']
                tp = current_row['bollinger_mid']
            elif (
                current_row['close'] >= current_row['bollinger_upper'] and
                current_row['rsi'] > 65 and
                current_row['stochastic_k'] > 70
            ):
                signal = -1
                sl = current_row['close'] + 0.8 * current_row['atr']
                tp = current_row['bollinger_mid']
        if state['prev_signal'] != 0:
            if state['regime'] == 'trend' and signal == 0:
                if state['prev_signal'] == 1:
                    sl = max(state['prev_sl'] or 0, current_row['close'] - 1.2 * current_row['atr'])
                elif state['prev_signal'] == -1:
                    sl = min(state['prev_sl'] or float('inf'), current_row['close'] + 1.2 * current_row['atr'])
            elif state['regime'] == 'range' and signal == 0:
                if (state['prev_signal'] == 1 and current_row['rsi'] > 55) or (
                    state['prev_signal'] == -1 and current_row['rsi'] < 45
                ):
                    signal = -state['prev_signal']
                    tp = current_row['close']
                    sl = state['prev_sl']
            else:
                sl = state['prev_sl']
                tp = state['prev_tp']
        state['prev_signal'] = signal
        state['prev_sl'] = sl
        state['prev_tp'] = tp
        if signal != 0:
            data.at[current_row.name, 'enhanced_trading_sl'] = sl
            data.at[current_row.name, 'enhanced_trading_tp'] = tp
        return 'buy' if signal == 1 else 'sell' if signal == -1 else 'hold'

class OriginalMeanReversion(Strategy):
    include = True
    required_indicators = ['close', 'support', 'resistance', 'atr']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        if idx == 0:
            return 'hold'
        prev_row = data.iloc[idx-1]
        buy_signal = (current_row['close'] < current_row['support'] * 1.01) and (prev_row['close'] > prev_row['support'] * 1.01)
        sell_signal = (current_row['close'] > current_row['resistance'] * 0.99) and (prev_row['close'] < prev_row['resistance'] * 0.99)
        if buy_signal:
            data.at[current_row.name, 'original_mean_reversion_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'original_mean_reversion_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif sell_signal:
            data.at[current_row.name, 'original_mean_reversion_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'original_mean_reversion_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class RefinedMeanReversion(Strategy):
    include = True
    required_indicators = ['close', 'support', 'resistance', 'rsi', 'atr']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        if idx == 0:
            return 'hold'
        prev_row = data.iloc[idx-1]
        buy_signal = (
            (current_row['close'] < current_row['support'] * 1.01) and
            (prev_row['close'] > prev_row['support'] * 1.01) and
            (current_row['rsi'] < 35)
        )
        sell_signal = (
            (current_row['close'] > current_row['resistance'] * 0.99) and
            (prev_row['close'] < prev_row['resistance'] * 0.99) and
            (current_row['rsi'] > 65)
        )
        if buy_signal:
            data.at[current_row.name, 'refined_mean_reversion_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'refined_mean_reversion_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif sell_signal:
            data.at[current_row.name, 'refined_mean_reversion_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'refined_mean_reversion_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class MomentumBreakout(Strategy):
    include = True
    required_indicators = ['close', 'breakout', 'atr', 'max_atr']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        buy_signal = (current_row['close'] > current_row['breakout']) and (current_row['atr'] > current_row['max_atr'])
        if buy_signal:
            data.at[current_row.name, 'momentum_breakout_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'momentum_breakout_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        return 'hold'

class SMACrossover(Strategy):
    include = True
    required_indicators = ['sma10', 'sma50', 'close', 'atr']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        if idx == 0:
            return 'hold'
        prev_row = data.iloc[idx-1]
        buy_signal = (current_row['sma10'] > current_row['sma50']) and (prev_row['sma10'] <= prev_row['sma50'])
        sell_signal = (current_row['sma10'] < current_row['sma50']) and (prev_row['sma10'] >= prev_row['sma50'])
        if buy_signal:
            data.at[current_row.name, 'sma_crossover_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'sma_crossover_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif sell_signal:
            data.at[current_row.name, 'sma_crossover_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'sma_crossover_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class BollingerScalping(Strategy):
    include = True
    required_indicators = ['close', 'upper', 'lower', 'atr']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        buy_signal = current_row['close'] < current_row['lower']
        sell_signal = current_row['close'] > current_row['upper']
        if buy_signal:
            data.at[current_row.name, 'bollinger_scalping_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'bollinger_scalping_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        elif sell_signal:
            data.at[current_row.name, 'bollinger_scalping_sl'] = current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'bollinger_scalping_tp'] = current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'sell'
        return 'hold'

class AdjustedAlphatrend(Strategy):
    include = True
    required_indicators = ['close', 'rsi', 'rsi_slope', 'macd', 'macd_signal', 'momentum', 'atr']

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        if not all(col in current_row.index for col in self.required_indicators) or current_row[self.required_indicators].isna().any():
            return 'hold'
        idx = data.index.get_loc(current_row.name)
        if idx < 2:
            return 'hold'
        prev_row = data.iloc[idx-1]
        buy_signal = (
            (current_row['close'] > data.iloc[idx-2]['close']) and
            (prev_row['close'] < data.iloc[idx-2]['close']) and
            (current_row['momentum'] > 1) and
            (current_row['rsi'] > 50) and (current_row['rsi'] < 70) and
            (current_row['rsi_slope'] > 0) and
            (current_row['macd'] > current_row['macd_signal'])
        )
        if buy_signal:
            data.at[current_row.name, 'adjusted_alphatrend_sl'] = current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
            data.at[current_row.name, 'adjusted_alphatrend_tp'] = current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
            return 'buy'
        return 'hold'

class VotingEnsemble(Strategy):
    include = True
    # Include signal columns for all constituent strategies
    required_indicators = [
        'ema_crossover_signal', 'rsi_threshold_signal', 'macd_cross_signal',
        'bb_breakout_signal', 'combined_momentum_signal', 'ai_powered_v6_signal',
        'bb_adx_trend_signal', 'rsi_confirmed_signal', 'ema_crossover_filtered_signal',
        'consolidated_simplified_signal', 'zlema_kalman_cross_signal',
        'ict_turtle_soup_signal', 'channel_breakout_signal', 'cs_alpha_signal',
        'ai_adaptive_signal', 'enhanced_trading_signal', 'original_mean_reversion_signal',
        'refined_mean_reversion_signal', 'momentum_breakout_signal', 'sma_crossover_signal',
        'bollinger_scalping_signal', 'adjusted_alphatrend_signal', 'close', 'atr'
    ]

    def execute(self, current_row: pd.Series, data: pd.DataFrame = None, params: Dict = None, state: Dict = None) -> str:
        params = params or {}
        state = state or {'active_trade': False, 'position': ''}
        min_votes_entry = params.get('min_votes_entry', 2)
        constituent_strategies = params.get('constituent_strategies', [])
        exit_vote_percentage = params.get('exit_vote_percentage', 0.40)
        if not constituent_strategies:
            return 'hold'
        total_strategies = len(constituent_strategies)
        if total_strategies == 0:
            return 'hold'
        min_opposing_votes_exit = math.ceil(total_strategies * exit_vote_percentage)
        buy_votes = 0
        sell_votes = 0
        for name in constituent_strategies:
            signal_col = f'{name}_signal'.lower()
            if signal_col in current_row.index:
                signal = current_row[signal_col]
                if signal == 'buy':
                    buy_votes += 1
                elif signal == 'sell':
                    sell_votes += 1
        current_position = state.get('position', '')
        if current_position == 'long' and sell_votes >= min_opposing_votes_exit:
            state['active_trade'] = False
            state['position'] = ''
            return 'sell'
        elif current_position == 'short' and buy_votes >= min_opposing_votes_exit:
            state['active_trade'] = False
            state['position'] = ''
            return 'buy'
        signal_to_return = 'hold'
        if not state.get('active_trade', False):
            if buy_votes >= min_votes_entry:
                signal_to_return = 'buy'
                state['active_trade'] = True
                state['position'] = 'long'
            elif sell_votes >= min_votes_entry:
                signal_to_return = 'sell'
                state['active_trade'] = True
                state['position'] = 'short'
        if signal_to_return in ['buy', 'sell']:
            data.at[current_row.name, 'voting_ensemble_sl'] = (
                current_row['close'] - params.get('sl_atr_mult', 2.0) * current_row['atr']
                if signal_to_return == 'buy' else
                current_row['close'] + params.get('sl_atr_mult', 2.0) * current_row['atr']
            )
            data.at[current_row.name, 'voting_ensemble_tp'] = (
                current_row['close'] + params.get('tp_atr_mult', 3.0) * current_row['atr']
                if signal_to_return == 'buy' else
                current_row['close'] - params.get('tp_atr_mult', 3.0) * current_row['atr']
            )
        return signal_to_return