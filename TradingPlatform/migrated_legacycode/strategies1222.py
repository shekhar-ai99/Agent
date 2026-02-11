
from datetime import time
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Base Strategy class
class BaseStrategy:
    def __init__(self, **kwargs):
        self.params = kwargs

    def generate_signals(self, data):
        raise NotImplementedError("Each strategy must implement generate_signals()")

class MomentumBreakoutStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            high_breakout = df['close'] > df['close'].rolling(10).max().shift(1)
            low_breakdown = df['close'] < df['close'].rolling(10).min().shift(1)
            df.loc[high_breakout, 'signal'] = 1
            df.loc[low_breakdown, 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = high_breakout.sum()
            sell_signals = low_breakdown.sum()
            logger.debug(f"MomentumBreakout: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"MomentumBreakout conditions: close={df['close'].iloc[-1]:.2f}, "
                            f"rolling_max={df['close'].rolling(10).max().shift(1).iloc[-1]:.2f}, "
                            f"rolling_min={df['close'].rolling(10).min().shift(1).iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in MomentumBreakout: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

class BreakoutATRStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            rolling_high = df['high'].rolling(20).max().shift(1)
            rolling_low = df['low'].rolling(20).min().shift(1)
            #df['buy'] = (df['close'] > rolling_high) & (df['close'] > df['supertrend_10_3.0'])
            # Make Supertrend less strict
            df['buy'] = (df['close'] > rolling_high) & (df['close'] > df['supertrend_10_2.0'])

            df['sell'] = (df['close'] < rolling_low)
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"BreakoutATR: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"BreakoutATR conditions: close={df['close'].iloc[-1]:.2f}, "
                            f"rolling_high={rolling_high.iloc[-1]:.2f}, "
                            f"rolling_low={rolling_low.iloc[-1]:.2f}, "
                            f"supertrend_10_3.0={df['supertrend_10_3.0'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in BreakoutATR: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

class SupportResistanceStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            period = self.params.get('period', 20)
            df['support'] = df['low'].rolling(period).min()
            df['resistance'] = df['high'].rolling(period).max()
            # df['buy'] = (df['close'] <= df['support']) & (df['close'].shift(1) > df['support'].shift(1))
            # df['sell'] = (df['close'] >= df['resistance']) & (df['close'].shift(1) < df['resistance'].shift(1))
            # Use ATR or slight threshold to trigger trade near support
            df['buy'] = (df['close'] <= df['support'] * 1.005)
            df['sell'] = (df['close'] >= df['resistance'] * 0.995)

            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"SupportResistance: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"SupportResistance conditions: close={df['close'].iloc[-1]:.2f}, "
                            f"support={df['support'].iloc[-1]:.2f}, "
                            f"resistance={df['resistance'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in SupportResistance: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

class MeanReversionStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['sma_20'] = df['close'].rolling(20).mean()
            # df['buy'] = df['close'] < df['sma_20']
            # df['sell'] = df['close'] > df['sma_20']
            # Use deviation from mean (Bollinger or % drop) instead
            df['buy'] = df['close'] < df['sma_20'] * 0.99
            df['sell'] = df['close'] > df['sma_20'] * 1.01

            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"MeanReversion: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"MeanReversion conditions: close={df['close'].iloc[-1]:.2f}, "
                            f"sma_20={df['sma_20'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in MeanReversion: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

class MomentumStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['roc_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
            # df['buy'] = df['roc_5'] > self.params.get('roc_threshold', 0)
            # df['sell'] = df['roc_5'] < -self.params.get('roc_threshold', 0)
            # Add small threshold if not provided
            roc_thresh = self.params.get('roc_threshold', 0.5)
            df['buy'] = df['roc_5'] > roc_thresh
            df['sell'] = df['roc_5'] < -roc_thresh

            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"Momentum: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"Momentum conditions: roc_5={df['roc_5'].iloc[-1]:.2f}, "
                            f"close={df['close'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in Momentum: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

class RSIATRReversalStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            # df['buy'] = (df['rsi_14'] < 30) & (df['atr_14'] > df['atr_14'].rolling(14).mean())
            # df['sell'] = (df['rsi_14'] > 70) & (df['atr_14'] > df['atr_14'].rolling(14).mean())
            # Loosen condition: allow moderate ATR rise and slightly higher RSI threshold
            df['buy'] = (df['rsi_14'] < 35) & (df['atr_14'] > df['atr_14'].rolling(14).mean() * 0.9)
            df['sell'] = (df['rsi_14'] > 65) & (df['atr_14'] > df['atr_14'].rolling(14).mean() * 0.9)

            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = df['close'] * 0.995
            df['tp'] = df['close'] * 1.01
            df['tsl'] = df['close'] - df['atr_14']
        except Exception as e:
            logger.error(f"RSIATRReversalStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class MACDTrendVolumeStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            avg_volume = df['volume'].rolling(20).mean()
            #df['volume_breakout'] = df['volume'] > 1.5 * avg_volume
            # Volume spike threshold too strict — try 1.2x instead
            df['volume_breakout'] = df['volume'] > 1.2 * avg_volume

            df['macd_trend'] = df['macd_12_26_9'] > df['macds_12_26_9']
            df['buy'] = df['macd_trend'] & df['volume_breakout']
            df['sell'] = (~df['macd_trend']) & df['volume_breakout']
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = df['close'] * 0.995
            df['tp'] = df['close'] * 1.01
            df['tsl'] = df['close'] - df['atr_14']
        except Exception as e:
            logger.error(f"MACDTrendVolumeStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class ORBVolumeMACDStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            open_time = df.index[0].time()
            # Calculate actual ORB from the first 3 bars only
            orb_range = df.iloc[:3]
            orb_high = orb_range['high'].max()
            orb_low = orb_range['low'].min()
            df['orb_high'] = orb_high
            df['orb_low'] = orb_low

            # df['orb_high'] = df['high'].rolling(window=3, min_periods=1).max()
            # df['orb_low'] = df['low'].rolling(window=3, min_periods=1).min()
            df['volume_spike'] = df['volume'] > df['volume'].rolling(10).mean() * 1.5
            df['macd_confirm'] = df['macd_12_26_9'] > df['macds_12_26_9']
            df['buy'] = (df['close'] > df['orb_high']) & df['volume_spike'] & df['macd_confirm']
            df['sell'] = (df['close'] < df['orb_low']) & df['volume_spike'] & (~df['macd_confirm'])
            

            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = df['close'] * 0.995
            df['tp'] = df['close'] * 1.01
            df['tsl'] = df['close'] - df['atr_14']
        except Exception as e:
            logger.error(f"ORBVolumeMACDStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class ThreePMBreakoutPowerBarStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            #df['is_3pm'] = df.index.map(lambda ts: ts.hour == 14 and ts.minute >= 50)
            # Loosen timing window to include more 3 PM candles
            df['is_3pm'] = df.index.map(lambda ts: ts.hour == 14 and ts.minute >= 45)

            df['body'] = abs(df['close'] - df['open'])
            df['avg_volume'] = df['volume'].rolling(20).mean()
            #df['strong_bull'] = (df['close'] > df['open']) & (df['body'] > df['body'].rolling(20).mean())
            df['strong_bull'] = (df['close'] > df['open']) & (df['body'] > df['body'].rolling(20).mean() * 0.8)

            df['volume_spike'] = df['volume'] > 1.5 * df['avg_volume']
            df['buy'] = df['is_3pm'] & df['strong_bull'] & df['volume_spike'] & (df['close'] > df['vwap'])
            df['sell'] = df['is_3pm'] & (~df['strong_bull']) & df['volume_spike'] & (df['close'] < df['vwap'])
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = df['close'] * 0.995
            df['tp'] = df['close'] * 1.01
            df['tsl'] = df['close'] - df['atr_14']
        except Exception as e:
            logger.error(f"ThreePMBreakoutPowerBarStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class BollingerSqueezeSpikeStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['is_3pm'] = df.index.map(lambda ts: ts.hour == 14 and ts.minute >= 50)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
           # df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.7
            df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.9

            #df['volume_spike'] = df['volume'] > df['volume'].rolling(20).mean() * 1.5
            df['volume_spike'] = df['volume'] > df['volume'].rolling(20).mean() * 1.2

            df['buy'] = df['is_3pm'] & df['bb_squeeze'] & df['volume_spike'] & (df['close'] > df['bb_upper'])
            df['sell'] = df['is_3pm'] & df['bb_squeeze'] & df['volume_spike'] & (df['close'] < df['bb_lower'])
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = df['close'] * 0.995
            df['tp'] = df['close'] * 1.01
            df['tsl'] = df['close'] - df['atr_14']
        except Exception as e:
            logger.error(f"BollingerSqueezeSpikeStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class AlphaTrendStrategy_EnhancedV1(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['sl'] = np.nan
        df['tp'] = np.nan
        df['tsl'] = np.nan
        try:
            required_cols = ['rsi_10', 'supertrend_10_3.0', 'ema_50', 'volume', 'volume_avg', 'adx_14', 'atr_14']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in AlphaTrendStrategy_EnhancedV1: {missing_cols}")
                return df
            is_uptrend_regime = df['close'] > df['ema_50']
            is_downtrend_regime = df['close'] < df['ema_50']
            has_volume_confirmation = df['volume'] > df['volume_avg']
            has_trend_strength = df['adx_14'] > self.params.get('adx_min_strength', 15)
            df['buy_cond_enhanced'] = (
                (df['rsi_10'] > self.params.get('rsi_buy_lower', 30)) &
                (df['rsi_10'] < self.params.get('rsi_buy_upper', 70)) &
                (df['supertrend_10_3.0'] < df['close']) &
                is_uptrend_regime & has_volume_confirmation & has_trend_strength
            )
            df['sell_cond_enhanced'] = (
                (df['rsi_10'] > self.params.get('rsi_sell_lower', 50)) &
                (df['rsi_10'] < self.params.get('rsi_sell_upper', 80)) &
                (df['supertrend_10_3.0'] > df['close']) &
                is_downtrend_regime & has_volume_confirmation & has_trend_strength
            )
            df.loc[df['buy_cond_enhanced'], 'signal'] = 1
            df.loc[df['sell_cond_enhanced'], 'signal'] = -1
            df.loc[df['signal'] == 1, 'sl'] = df['close'] - df['atr_14'] * self.params.get('atr_sl_mult', 1.5)
            df.loc[df['signal'] == 1, 'tp'] = df['close'] + df['atr_14'] * self.params.get('atr_tp_mult', 2.0)
            df.loc[df['signal'] == -1, 'sl'] = df['close'] + df['atr_14'] * self.params.get('atr_sl_mult', 1.5)
            df.loc[df['signal'] == -1, 'tp'] = df['close'] - df['atr_14'] * self.params.get('atr_tp_mult', 2.0)
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            buy_signals = df['buy_cond_enhanced'].sum()
            sell_signals = df['sell_cond_enhanced'].sum()
            logger.debug(f"AlphaTrend_EnhancedV1: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
        except Exception as e:
            logger.error(f"AlphaTrend_EnhancedV1 error: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

class ClosingBellBreakoutScalpStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['sl'] = np.nan
        df['tp'] = np.nan
        df['tsl'] = np.nan
        try:
            if not df.index.tz:
                logger.warning("DataFrame index is not timezone-aware. Localizing to IST.")
                df.index = df.index.tz_localize('Asia/Kolkata')
            df['is_timeslot'] = df.index.to_series().apply(
                lambda ts: (ts.time() >= time(14, 30) and ts.time() <= time(15, 30))
            )
            timeslot_count = df['is_timeslot'].sum()
            logger.debug(f"ClosingBellBreakoutScalp: {timeslot_count} rows in time window 2:30-3:30 PM")
            df['body'] = abs(df['close'] - df['open'])
            df['body_avg'] = df['body'].rolling(window=20, min_periods=5).mean()
            df['volume_avg'] = df['volume'].rolling(window=20, min_periods=5).mean()
            df['strong_bull_candle'] = (df['close'] > df['open']) & (df['body'] > df['body_avg'] * 1.0)
            df['strong_bear_candle'] = (df['open'] > df['close']) & (df['body'] > df['body_avg'] * 1.0)
            df['volume_spike'] = df['volume'] > df['volume_avg'] * 1.2
            if 'vwap' not in df.columns or df['vwap'].isna().all():
                logger.error("VWAP not found or all NaN in data. Cannot generate signals.")
                return df
            df['vwap_buy_condition'] = df['close'] > df['vwap']
            df['vwap_sell_condition'] = df['close'] < df['vwap']
            atr_for_sl_tp = df['atr_14'] if 'atr_14' in df.columns else df['close'] * 0.003
            buy_conditions = (
                df['is_timeslot'] &
                df['strong_bull_candle'] &
                df['volume_spike'] &
                df['vwap_buy_condition']
            )
            sell_conditions = (
                df['is_timeslot'] &
                df['strong_bear_candle'] &
                df['volume_spike'] &
                df['vwap_sell_condition']
            )
            df.loc[buy_conditions, 'signal'] = 1
            df.loc[sell_conditions, 'signal'] = -1
            df.loc[df['signal'] == 1, 'sl'] = df['close'] - (atr_for_sl_tp * 1.2)
            df.loc[df['signal'] == 1, 'tp'] = df['close'] + (atr_for_sl_tp * 2.5)
            df.loc[df['signal'] == -1, 'sl'] = df['close'] + (atr_for_sl_tp * 1.2)
            df.loc[df['signal'] == -1, 'tp'] = df['close'] - (atr_for_sl_tp * 2.5)
            buy_signals = buy_conditions.sum()
            sell_signals = sell_conditions.sum()
            logger.debug(f"ClosingBellBreakoutScalp: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"ClosingBellBreakoutScalp conditions: "
                            f"strong_bull={df['strong_bull_candle'].iloc[-1]}, "
                            f"volume_spike={df['volume_spike'].iloc[-1]}, "
                            f"vwap_buy={df['vwap_buy_condition'].iloc[-1]}, "
                            f"close={df['close'].iloc[-1]:.2f}, vwap={df['vwap'].iloc[-1]:.2f}")
        except Exception as e:
            logger.error(f"ClosingBellBreakoutScalpStrategy error: {e}")
        return df

class ExpiryDayVolatilitySpikeStrategy(BaseStrategy):
    def __init__(self, **params):
        super().__init__(**params)
        self.bb_window = self.params.get('bb_window', 20)
        self.bb_std_dev = self.params.get('bb_std_dev', 2)
        self.squeeze_threshold_multiplier = self.params.get('squeeze_threshold_multiplier', 0.8)

    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['sl'] = np.nan
        df['tp'] = np.nan
        df['tsl'] = np.nan
        try:
            if not df.index.tz:
                logger.warning("DataFrame index is not timezone-aware. Localizing to IST.")
                df.index = df.index.tz_localize('Asia/Kolkata')
            df['is_timeslot'] = df.index.to_series().apply(
                lambda ts: (ts.time() >= time(14, 30) and ts.time() <= time(15, 30))
            )
            timeslot_count = df['is_timeslot'].sum()
            logger.debug(f"ExpiryDayVolatilitySpike: {timeslot_count} rows in time window 2:30-3:30 PM")
            if 'is_expiry_day' not in df.columns:
                logger.warning("is_expiry_day not found. Assuming all days are non-expiry.")
                df['is_expiry_day'] = False
            expiry_count = df['is_expiry_day'].sum()
            logger.debug(f"ExpiryDayVolatilitySpike: {expiry_count} expiry day rows")
            df['bb_mid'] = df['close'].rolling(window=self.bb_window).mean()
            df['bb_std'] = df['close'].rolling(window=self.bb_window).std()
            df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * self.bb_std_dev)
            df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * self.bb_std_dev)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            if 'atr_14' not in df.columns:
                logger.warning("atr_14 not found. Using IQR-based fallback.")
                df['squeeze_ref_atr'] = df['close'].rolling(window=self.bb_window).apply(
                    lambda x: np.percentile(x, 75) - np.percentile(x, 25), raw=True
                ) / 2
                df['squeeze_ref_atr'] = df['squeeze_ref_atr'].fillna(method='bfill').fillna(method='ffill')
            else:
                df['squeeze_ref_atr'] = df['atr_14']
            df['is_squeeze'] = df['bb_width'] < (df['squeeze_ref_atr'] * self.squeeze_threshold_multiplier)
            df['volume_avg'] = df['volume'].rolling(window=20, min_periods=5).mean()
            df['volume_spike'] = df['volume'] > df['volume_avg'] * 1.2
            buy_conditions = (
                df['is_timeslot'] &
                df['is_squeeze'].shift(1) &
                (df['close'] > df['bb_upper'].shift(1)) &
                df['volume_spike']
            )
            sell_conditions = (
                df['is_timeslot'] &
                df['is_squeeze'].shift(1) &
                (df['close'] < df['bb_lower'].shift(1)) &
                df['volume_spike']
            )
            df.loc[buy_conditions, 'signal'] = 1
            df.loc[sell_conditions, 'signal'] = -1
            atr_for_sl_tp = df['squeeze_ref_atr']
            df.loc[df['signal'] == 1, 'sl'] = df['close'] - (atr_for_sl_tp * 1.5)
            df.loc[df['signal'] == 1, 'tp'] = df['close'] + (atr_for_sl_tp * 3.0)
            df.loc[df['signal'] == -1, 'sl'] = df['close'] + (atr_for_sl_tp * 1.5)
            df.loc[df['signal'] == -1, 'tp'] = df['close'] - (atr_for_sl_tp * 3.0)
            buy_signals = buy_conditions.sum()
            sell_signals = sell_conditions.sum()
            logger.debug(f"ExpiryDayVolatilitySpike: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"ExpiryDayVolatilitySpike conditions: "
                            f"is_squeeze={df['is_squeeze'].shift(1).iloc[-1]}, "
                            f"close={df['close'].iloc[-1]:.2f}, "
                            f"bb_upper={df['bb_upper'].shift(1).iloc[-1]:.2f}, "
                            f"volume_spike={df['volume_spike'].iloc[-1]}")
        except Exception as e:
            logger.error(f"ExpiryDayVolatilitySpikeStrategy error: {e}")
        return df

class ExpiryDayOTMScalpStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['sl'] = np.nan
        df['tp'] = np.nan
        df['tsl'] = np.nan
        try:
            bb_period = self.params.get('bb_period', 20)
            bb_std = self.params.get('bb_std', 2.0)
            volume_multiplier = self.params.get('volume_multiplier', 1.5)
            sl_atr_mult = self.params.get('sl_atr_mult', 1.0)
            tp_atr_mult = self.params.get('tp_atr_mult', 0.5)
            start_hour = self.params.get('start_hour', 14)
            start_minute = self.params.get('start_minute', 55)
            end_hour = self.params.get('end_hour', 15)
            end_minute = self.params.get('end_minute', 20)
            df['is_3pm'] = df.index.map(lambda ts: (ts.hour == start_hour and ts.minute >= start_minute) or 
                                         (ts.hour == end_hour and ts.minute <= end_minute))
            df['avg_volume'] = df['volume'].rolling(window=5).mean()
            df['volume_spike'] = df['volume'] > volume_multiplier * df['avg_volume']
            df['buy'] = (df['is_3pm'] & 
                        (df['close'] > df['bollinger_hband']) & 
                        df['volume_spike'])
            df['sell'] = (df['is_3pm'] & 
                         (df['close'] < df['bollinger_lband']) & 
                         df['volume_spike'])
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df.loc[df['buy'], 'sl'] = df['close'] - sl_atr_mult * df['atr']
            df.loc[df['buy'], 'tp'] = df['close'] + tp_atr_mult * df['atr']
            df.loc[df['buy'], 'tsl'] = df['close'] - sl_atr_mult * df['atr'] * 0.5
            df.loc[df['sell'], 'sl'] = df['close'] + sl_atr_mult * df['atr']
            df.loc[df['sell'], 'tp'] = df['close'] - tp_atr_mult * df['atr']
            df.loc[df['sell'], 'tsl'] = df['close'] + sl_atr_mult * df['atr'] * 0.5
        except Exception as e:
            logger.error(f"ExpiryDayOTMScalpStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class MomentumBreakoutRSIStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['sl'] = np.nan
        df['tp'] = np.nan
        df['tsl'] = np.nan
        try:
            ema_period = self.params.get('ema_period', 20)
            rsi_period = self.params.get('rsi_period', 14)
            rsi_overbought = self.params.get('rsi_overbought', 60)
            rsi_oversold = self.params.get('rsi_oversold', 40)
            volume_multiplier = self.params.get('volume_multiplier', 1.5)
            sl_atr_mult = self.params.get('sl_atr_mult', 1.0)
            tp_atr_mult = self.params.get('tp_atr_mult', 0.5)
            start_hour = self.params.get('start_hour', 14)
            start_minute = self.params.get('start_minute', 55)
            end_hour = self.params.get('end_hour', 15)
            end_minute = self.params.get('end_minute', 20)
            df['is_3pm'] = df.index.map(lambda ts: (ts.hour == start_hour and ts.minute >= start_minute) or 
                                         (ts.hour == end_hour and ts.minute <= end_minute))
            df['avg_volume'] = df['volume'].rolling(window=5).mean()
            df['volume_spike'] = df['volume'] > volume_multiplier * df['avg_volume']
            df['buy'] = (df['is_3pm'] & 
                        (df['close'] > df[f'ema_{ema_period}']) & 
                        (df['rsi'] > rsi_overbought) & 
                        df['volume_spike'])
            df['sell'] = (df['is_3pm'] & 
                         (df['close'] < df[f'ema_{ema_period}']) & 
                         (df['rsi'] < rsi_oversold) & 
                         df['volume_spike'])
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df.loc[df['buy'], 'sl'] = df['close'] - sl_atr_mult * df['atr']
            df.loc[df['buy'], 'tp'] = df['close'] + tp_atr_mult * df['atr']
            df.loc[df['buy'], 'tsl'] = df['close'] - sl_atr_mult * df['atr'] * 0.5
            df.loc[df['sell'], 'sl'] = df['close'] + sl_atr_mult * df['atr']
            df.loc[df['sell'], 'tp'] = df['close'] - tp_atr_mult * df['atr']
            df.loc[df['sell'], 'tsl'] = df['close'] + sl_atr_mult * df['atr'] * 0.5
        except Exception as e:
            logger.error(f"MomentumBreakoutRSIStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class VWAPReversalScalpStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['sl'] = np.nan
        df['tp'] = np.nan
        df['tsl'] = np.nan
        try:
            stoch_k_period = self.params.get('stoch_k_period', 14)
            stoch_d_period = self.params.get('stoch_d_period', 3)
            stoch_overbought = self.params.get('stoch_overbought', 70)
            stoch_oversold = self.params.get('stoch_oversold', 30)
            #vwap_tolerance = self.params.get('vwap_tolerance', 0.001)
            vwap_tolerance = self.params.get('vwap_tolerance', 0.002)
            volume_multiplier = self.params.get('volume_multiplier', 1.5)
            sl_atr_mult = self.params.get('sl_atr_mult', 1.0)
            tp_atr_mult = self.params.get('tp_atr_mult', 0.5)
            start_hour = self.params.get('start_hour', 14)
            start_minute = self.params.get('start_minute', 50)
            end_hour = self.params.get('end_hour', 15)
            end_minute = self.params.get('end_minute', 20)
            df['is_3pm'] = df.index.map(lambda ts: (ts.hour == start_hour and ts.minute >= start_minute) or 
                                         (ts.hour == end_hour and ts.minute <= end_minute))
            df['avg_volume'] = df['volume'].rolling(window=5).mean()
            df['volume_spike'] = df['volume'] > volume_multiplier * df['avg_volume']
            #df['vwap_proximity'] = (df['close'].shift(1) - df['vwap']).abs() / df['vwap'] <= vwap_tolerance
            df['stoch_k_cross_d'] = df['stoch_k'] > df['stoch_d']
            df['stoch_k_cross_d_reverse'] = df['stoch_k'] < df['stoch_d']
            df['buy'] = (df['is_3pm'] & 
                       # df['vwap_proximity'] & 
                        (df['close'] > df['vwap']) & 
                        df['stoch_k_cross_d'] & 
                        (df['stoch_k'] < stoch_oversold) & 
                        df['volume_spike'])
            df['sell'] = (df['is_3pm'] & 
                        # df['vwap_proximity'] & 
                         (df['close'] < df['vwap']) & 
                         df['stoch_k_cross_d_reverse'] & 
                         (df['stoch_k'] > stoch_overbought) & 
                         df['volume_spike'])
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df.loc[df['buy'], 'sl'] = df['close'] - sl_atr_mult * df['atr']
            df.loc[df['buy'], 'tp'] = df['close'] + tp_atr_mult * df['atr']
            df.loc[df['buy'], 'tsl'] = df['close'] - sl_atr_mult * df['atr'] * 0.5
            df.loc[df['sell'], 'sl'] = df['close'] + sl_atr_mult * df['atr']
            df.loc[df['sell'], 'tp'] = df['close'] - tp_atr_mult * df['atr']
            df.loc[df['sell'], 'tsl'] = df['close'] + sl_atr_mult * df['atr'] * 0.5
        except Exception as e:
            logger.error(f"VWAPReversalScalpStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class StraddleScalpHighVolStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['sl'] = np.nan
        df['tp'] = np.nan
        df['tsl'] = np.nan
        try:
            # Check for required columns
            required_cols = ['close', 'high', 'low', 'atr_14', 'volatility_20']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in StraddleScalpHighVolStrategy: {missing_cols}")
                return df

            # Strategy parameters
            vix_threshold = self.params.get('vix_threshold', 8.0)  # Relaxed from 10.0
            atr_threshold = self.params.get('atr_threshold', 0.0005)  # Relaxed from 0.001
            #range_threshold = self.params.get('range_threshold', 0.007)  # Relaxed from 0.005
            range_threshold = self.params.get('range_threshold', 0.01)

            sl_atr_mult = self.params.get('sl_atr_mult', 1.0)
            tp_atr_mult = self.params.get('tp_atr_mult', 0.5)
            volume_multiplier = self.params.get('volume_multiplier', 1.1)  # Relaxed from 1.2

            # Remove time restriction for broader applicability
            df['is_3pm'] = True

            # Calculate recent price range
            df['recent_range'] = (df['high'].rolling(window=5).max() - 
                                 df['low'].rolling(window=5).min()) / df['close']

            # High volatility condition
            df['high_vol'] = (df['volatility_20'] * 100 > vix_threshold) & (df['atr_14'] / df['close'] > atr_threshold)

            # Volume condition
            df['volume_avg'] = df['volume'].rolling(window=20, min_periods=5).mean()
            df['volume_spike'] = df['volume'] > df['volume_avg'] * volume_multiplier

            # Straddle condition
            df['straddle'] = (df['recent_range'] <= range_threshold) & df['high_vol'] & df['volume_spike']

            # Assign signals (1 for buy straddle, 0 for no trade)
            df.loc[df['straddle'], 'signal'] = 1  # Buy straddle (call + put)

            # Set SL, TP, and TSL
            df.loc[df['signal'] == 1, 'sl'] = df['close'] - sl_atr_mult * df['atr_14']
            df.loc[df['signal'] == 1, 'tp'] = df['close'] + tp_atr_mult * df['atr_14']
            df.loc[df['signal'] == 1, 'tsl'] = df['close'] - sl_atr_mult * df['atr_14'] * 0.5

            # Debug logging
            straddle_signals = df['straddle'].sum()
            logger.debug(f"StraddleScalpHighVol: Straddle signals: {straddle_signals}")
            if straddle_signals == 0:
                logger.debug(f"StraddleScalpHighVol conditions: "
                             f"recent_range={df['recent_range'].iloc[-1]:.4f}, "
                             f"high_vol={df['high_vol'].iloc[-1]}, "
                             f"volume_spike={df['volume_spike'].iloc[-1]}, "
                             f"volatility_20={df['volatility_20'].iloc[-1]*100:.2f}, "
                             f"atr_ratio={df['atr_14'].iloc[-1]/df['close'].iloc[-1]:.4f}, "
                             f"params={self.params}")

        except Exception as e:
            logger.error(f"StraddleScalpHighVolStrategy error: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan

        return df
class MeanReversionSnapBackStrategy(BaseStrategy):
    def __init__(self, **params):
        super().__init__(**params)
        self.rsi_period = self.params.get('rsi_period', 7)
        self.rsi_ob = self.params.get('rsi_ob', 80)
        self.rsi_os = self.params.get('rsi_os', 20)
        self.ema_period = self.params.get('ema_period', 9)

    def _calculate_rsi(self, series, period):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['sl'] = np.nan
        df['tp'] = np.nan
        df['tsl'] = np.nan
        try:
            if not df.index.tz:
                logger.warning("DataFrame index is not timezone-aware. Localizing to IST.")
                df.index = df.index.tz_localize('Asia/Kolkata')
            required_cols = ['close', 'high', 'low']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in MeanReversionSnapBackStrategy: {missing_cols}")
                return df
            df['is_timeslot'] = df.index.to_series().apply(
                lambda ts: (ts.time() >= time(14, 30) and ts.time() <= time(15, 30))
            )
            timeslot_count = df['is_timeslot'].sum()
            logger.debug(f"MeanReversionSnapBack: {timeslot_count} rows in time window 2:30-3:30 PM")
            df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
            df['ema_short'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
            atr_for_sl_tp = df['atr_14'] if 'atr_14' in df.columns else df['close'] * 0.002
            df['was_oversold'] = (df['rsi'].shift(1) < self.rsi_os) | (df['rsi'].shift(2) < self.rsi_os)
            buy_reversal_confirmation = (df['close'] > df['low'].shift(1))
            buy_conditions = (
                df['is_timeslot'] &
                df['was_oversold'] &
                buy_reversal_confirmation
            )
            df['was_overbought'] = (df['rsi'].shift(1) > self.rsi_ob) | (df['rsi'].shift(2) > self.rsi_ob)
            sell_reversal_confirmation = (df['close'] < df['high'].shift(1))
            sell_conditions = (
                df['is_timeslot'] &
                df['was_overbought'] &
                sell_reversal_confirmation
            )
            df.loc[buy_conditions, 'signal'] = 1
            df.loc[sell_conditions, 'signal'] = -1
            df.loc[df['signal'] == 1, 'sl'] = df['low'].shift(1) - atr_for_sl_tp * 0.3
            df.loc[df['signal'] == 1, 'tp'] = df['close'] + (atr_for_sl_tp * 1.5)
            df.loc[df['signal'] == -1, 'sl'] = df['high'].shift(1) + atr_for_sl_tp * 0.3
            df.loc[df['signal'] == -1, 'tp'] = df['close'] - (atr_for_sl_tp * 1.5)
            buy_signals = buy_conditions.sum()
            sell_signals = sell_conditions.sum()
            logger.debug(f"MeanReversionSnapBack: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                rsi_val = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 'NaN'
                was_oversold = df['was_oversold'].iloc[-1] if not pd.isna(df['was_oversold'].iloc[-1]) else 'NaN'
                close_val = df['close'].iloc[-1] if not pd.isna(df['close'].iloc[-1]) else 'NaN'
                low_prev = df['low'].shift(1).iloc[-1] if not pd.isna(df['low'].shift(1).iloc[-1]) else 'NaN'
                logger.debug(f"MeanReversionSnapBack conditions: "
                            f"rsi={rsi_val}, "
                            f"was_oversold={was_oversold}, "
                            f"close={close_val}, "
                            f"low_prev={low_prev}")
        except Exception as e:
            logger.error(f"MeanReversionSnapBackStrategy error: {e}")
        return df

class ThreePMBollingerVolBreakoutStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['sl'] = np.nan
        df['tp'] = np.nan
        df['tsl'] = np.nan
        try:
            bb_period = self.params.get('bb_period', 20)
            bb_std = self.params.get('bb_std', 2.0)
            volume_multiplier = self.params.get('volume_multiplier', 1.5)
            rsi_lower = self.params.get('rsi_lower', 40)
            rsi_upper = self.params.get('rsi_upper', 60)
            squeeze_threshold = self.params.get('squeeze_threshold', 0.7)
            sl_atr_mult = self.params.get('sl_atr_mult', 1.0)
            tp_atr_mult = self.params.get('tp_atr_mult', 0.5)
            start_hour = self.params.get('start_hour', 14)
            start_minute = self.params.get('start_minute', 50)
            end_hour = self.params.get('end_hour', 15)
            end_minute = self.params.get('end_minute', 20)
            df['is_3pm'] = df.index.map(lambda ts: (ts.hour == start_hour and ts.minute >= start_minute) or
                                        (ts.hour == end_hour and ts.minute <= end_minute))
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            #df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(bb_period).mean() * squeeze_threshold
            df['avg_volume'] = df['volume'].rolling(window=5).mean()
            df['volume_spike'] = df['volume'] > volume_multiplier * df['avg_volume']
            #df['rsi_neutral'] = (df['rsi_14'] >= rsi_lower) & (df['rsi_14'] <= rsi_upper)
            df['rsi_neutral'] = (df['rsi_14'] >= 35) & (df['rsi_14'] <= 65)
            df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(bb_period).mean() * 0.6

            df['buy'] = (df['is_3pm'] &
                         df['bb_squeeze'] &
                         df['volume_spike'] &
                         df['rsi_neutral'] &
                         (df['close'] > df['bb_upper']))
            df['sell'] = (df['is_3pm'] &
                          df['bb_squeeze'] &
                          df['volume_spike'] &
                          df['rsi_neutral'] &
                          (df['close'] < df['bb_lower']))
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df.loc[df['buy'], 'sl'] = df['close'] - sl_atr_mult * df['atr_14']
            df.loc[df['buy'], 'tp'] = df['close'] + tp_atr_mult * df['atr_14']
            df.loc[df['buy'], 'tsl'] = df['close'] - sl_atr_mult * df['atr_14'] * 0.5
            df.loc[df['sell'], 'sl'] = df['close'] + sl_atr_mult * df['atr_14']
            df.loc[df['sell'], 'tp'] = df['close'] - tp_atr_mult * df['atr_14']
            df.loc[df['sell'], 'tsl'] = df['close'] + sl_atr_mult * df['atr_14'] * 0.5
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"ThreePMBollingerVolBreakout: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"ThreePMBollingerVolBreakout conditions: "
                             f"close={df['close'].iloc[-1]:.2f}, "
                             f"bb_upper={df['bb_upper'].iloc[-1]:.2f}, "
                             f"bb_lower={df['bb_lower'].iloc[-1]:.2f}, "
                             f"rsi_14={df['rsi_14'].iloc[-1]:.2f}, "
                             f"volume={df['volume'].iloc[-1]:.2f}, "
                             f"avg_volume={df['avg_volume'].iloc[-1]:.2f}, "
                             f"params={self.params}")
        except Exception as e:
            logger.error(f"ThreePMBollingerVolBreakoutStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class EMACrossoverStrategy_EnhancedV1(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            required_cols = ['ema_9', 'ema_21', 'adx_14', 'volume', 'volume_avg', 'atr_14', 'close']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing column for EMACrossoverStrategy_EnhancedV1: {col}.")
                    return df
            basic_buy_crossover = (df['ema_9'] > df['ema_21']) & (df['ema_9'].shift(1) <= df['ema_21'].shift(1))
            basic_sell_crossover = (df['ema_9'] < df['ema_21']) & (df['ema_9'].shift(1) >= df['ema_21'].shift(1))
            adx_threshold = self.params.get('adx_threshold', 20)
            is_trending = df['adx_14'] > adx_threshold
            volume_multiplier = self.params.get('volume_multiplier', 1.2)
            has_volume_confirmation = df['volume'] > (df['volume_avg'] * volume_multiplier)
            use_regime_filter = self.params.get('use_regime_filter', False)
            if use_regime_filter:
                if 'ema_200' not in df.columns:
                    logger.warning("ema_200 not found for regime filter in EMACrossover_EnhancedV1. Skipping regime filter.")
                    is_bullish_regime = pd.Series([True] * len(df), index=df.index)
                    is_bearish_regime = pd.Series([True] * len(df), index=df.index)
                else:
                    is_bullish_regime = df['close'] > df['ema_200']
                    is_bearish_regime = df['close'] < df['ema_200']
            else:
                is_bullish_regime = pd.Series([True] * len(df), index=df.index)
                is_bearish_regime = pd.Series([True] * len(df), index=df.index)
            df['buy'] = (
                basic_buy_crossover &
                is_trending &
                has_volume_confirmation &
                is_bullish_regime
            )
            df['sell'] = (
                basic_sell_crossover &
                is_trending &
                has_volume_confirmation &
                is_bearish_regime
            )
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            sl_atr_mult = self.params.get('sl_atr_mult', 1.5)
            tp_atr_mult = self.params.get('tp_atr_mult', 2.0)
            df['sl'] = np.where(df['signal'] == 1, df['close'] - (df['atr_14'] * sl_atr_mult),
                                np.where(df['signal'] == -1, df['close'] + (df['atr_14'] * sl_atr_mult), np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] + (df['atr_14'] * tp_atr_mult),
                                np.where(df['signal'] == -1, df['close'] - (df['atr_14'] * tp_atr_mult), np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"EMACrossover_EnhancedV1: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
        except KeyError as e:
            logger.error(f"KeyError in EMACrossoverStrategy_EnhancedV1: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = 0; df['sl'] = np.nan; df['tp'] = np.nan; df['tsl'] = np.nan
        except Exception as e:
            logger.error(f"General error in EMACrossoverStrategy_EnhancedV1: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = 0; df['sl'] = np.nan; df['tp'] = np.nan; df['tsl'] = np.nan
        return df

class ThreePMBreakoutPowerBarStrategy_EnhancedV1(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            required_cols = ['close', 'open', 'volume', 'vwap', 'atr_14', 'rsi_14']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing column for ThreePMBreakoutPowerBarStrategy_EnhancedV1: {col}.")
                    return df
            df['is_3pm_window'] = df.index.map(
                lambda ts: (ts.hour == 14 and ts.minute >= self.params.get('start_minute', 50)) or \
                           (ts.hour == 15 and ts.minute <= self.params.get('end_minute', 15))
            )
            df['body'] = abs(df['close'] - df['open'])
            df['body_avg'] = df['body'].rolling(window=self.params.get('body_avg_period', 20), min_periods=5).mean()
            df['volume_avg'] = df['volume'].rolling(window=self.params.get('vol_avg_period', 20), min_periods=5).mean()
            is_strong_bull_base = (df['close'] > df['open']) & (df['body'] > df['body_avg'])
            is_strong_bear_base = (df['open'] > df['close']) & (df['body'] > df['body_avg'])
            has_volume_spike = df['volume'] > (df['volume_avg'] * self.params.get('vol_spike_mult', 1.5))
            atr_body_mult = self.params.get('atr_body_mult', 1.0)
            is_atr_power_bar = df['body'] > (df['atr_14'] * atr_body_mult)
            closed_above_vwap = df['close'] > df['vwap']
            closed_below_vwap = df['close'] < df['vwap']
            opened_near_below_vwap = (df['open'] <= df['vwap'])
            opened_near_above_vwap = (df['open'] >= df['vwap'])
            vwap_buy_confirm = closed_above_vwap
            vwap_sell_confirm = closed_below_vwap
            rsi_momentum_buy = df['rsi_14'] > self.params.get('rsi_buy_thresh', 55)
            rsi_momentum_sell = df['rsi_14'] < self.params.get('rsi_sell_thresh', 45)
            use_rsi_confirm = self.params.get('use_rsi_confirm_3pm', False)
            df['buy'] = (
                df['is_3pm_window'] &
                is_strong_bull_base &
                has_volume_spike &
                is_atr_power_bar &
                vwap_buy_confirm
            )
            if use_rsi_confirm:
                df['buy'] = df['buy'] & rsi_momentum_buy
            df['sell'] = (
                df['is_3pm_window'] &
                is_strong_bear_base &
                has_volume_spike &
                is_atr_power_bar &
                vwap_sell_confirm
            )
            if use_rsi_confirm:
                df['sell'] = df['sell'] & rsi_momentum_sell
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            sl_atr_mult_3pm = self.params.get('sl_atr_mult_3pm', 1.0)
            tp_atr_mult_3pm = self.params.get('tp_atr_mult_3pm', 1.5)
            df.loc[df['signal'] == 1, 'sl'] = df['open'] - (df['atr_14'] * 0.25)
            df.loc[df['signal'] == 1, 'tp'] = df['close'] + (df['atr_14'] * tp_atr_mult_3pm)
            df.loc[df['signal'] == -1, 'sl'] = df['open'] + (df['atr_14'] * 0.25)
            df.loc[df['signal'] == -1, 'tp'] = df['close'] - (df['atr_14'] * tp_atr_mult_3pm)
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"ThreePMBreakoutPowerBar_EnhancedV1: Buy: {buy_signals}, Sell: {sell_signals}")
        except KeyError as e:
            logger.error(f"KeyError in ThreePMBreakoutPowerBarStrategy_EnhancedV1: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = 0; df['sl'] = np.nan; df['tp'] = np.nan; df['tsl'] = np.nan
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan

        except Exception as e:
            logger.error(f"General error in ThreePMBreakoutPowerBarStrategy_EnhancedV1: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = 0; df['sl'] = np.nan; df['tp'] = np.nan; df['tsl'] = np.nan
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

class MeanReversionSnapBackStrategy_EnhancedV1(BaseStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rsi_period = self.params.get('rsi_period', 14)
        self.rsi_ob = self.params.get('rsi_ob', 60)
        self.rsi_os = self.params.get('rsi_os', 40)
        self.ema_period = self.params.get('ema_period', 9)
        self.atr_period_for_vol = self.params.get('atr_period_for_vol', 14)
        self.max_atr_multiplier_for_entry = self.params.get('max_atr_multiplier_for_entry', 2.5)
        self.reversal_vol_spike_mult = self.params.get('reversal_vol_spike_mult', 1.0)

    def _calculate_rsi(self, series, period):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['sl'] = np.nan
        df['tp'] = np.nan
        df['tsl'] = np.nan
        try:
            required_cols = ['close', 'open', 'high', 'low', 'volume', 'atr_14']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing column for MeanReversionSnapBackStrategy_EnhancedV1: {col}.")
                    return df
                
            df['volume_avg'] = df['volume'].rolling(window=20, min_periods=5).mean()
            df['is_timeslot'] = True  # Removed time restriction
            df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
            df['ema_short'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
            has_reversal_volume_spike = df['volume'] > (df['volume_avg'] * self.reversal_vol_spike_mult)
            prev_bar_midpoint = (df['high'].shift(1) + df['low'].shift(1)) / 2
            prev_bar_was_bearish = df['close'].shift(1) < df['open'].shift(1)
            buy_reversal_candle_confirm = (df['close'] > prev_bar_midpoint) & prev_bar_was_bearish & (df['close'] > df['open'])
            prev_bar_was_bullish = df['close'].shift(1) > df['open'].shift(1)
            sell_reversal_candle_confirm = (df['close'] < prev_bar_midpoint) & prev_bar_was_bullish & (df['close'] < df['open'])
            df['was_oversold'] = df['rsi'].shift(1) < self.rsi_os
            df['was_overbought'] = df['rsi'].shift(1) > self.rsi_ob
            df['buy'] = df['was_oversold'] & has_reversal_volume_spike & buy_reversal_candle_confirm
            df['sell'] = df['was_overbought'] & has_reversal_volume_spike & sell_reversal_candle_confirm
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df.loc[df['signal'] == 1, 'sl'] = df['low'] - (df['atr_14'] * 0.25)
            df.loc[df['signal'] == 1, 'tp'] = df['close'] + (df['atr_14'] * self.params.get('tp_atr_mult_mr', 1.5))
            df.loc[df['signal'] == -1, 'sl'] = df['high'] + (df['atr_14'] * 0.25)
            df.loc[df['signal'] == -1, 'tp'] = df['close'] - (df['atr_14'] * self.params.get('tp_atr_mult_mr', 1.5))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"MeanReversionSnapBack_EnhancedV1: Buy: {buy_signals}, Sell: {sell_signals}")
        except KeyError as e:
            logger.error(f"KeyError in MeanReversionSnapBackStrategy_EnhancedV1: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = 0; df['sl'] = np.nan; df['tp'] = np.nan; df['tsl'] = np.nan
        except Exception as e:
            logger.error(f"General error in MeanReversionSnapBackStrategy_EnhancedV1: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = 0; df['sl'] = np.nan; df['tp'] = np.nan; df['tsl'] = np.nan
        return df

class MomentumBreakout_enhancedV1(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            for col in ['volume', 'ema_9', 'ema_21', 'atr_14']:
                if col not in df.columns:
                    logger.error(f"Missing {col} in MomentumBreakout_enhancedV1")
                    return df
            df['roc'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
            df['momentum_up'] = df['roc'] > self.params.get('roc_threshold', 0.5)
            df['volume_spike'] = df['volume'] > 1.5 * df['volume'].rolling(10).mean()
            df['ema_filter'] = df['ema_9'] > df['ema_21']
            df['buy'] = df['momentum_up'] & df['volume_spike'] & df['ema_filter']
            df['sell'] = (~df['momentum_up']) & df['volume_spike'] & (~df['ema_filter'])
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = df['close'] * 0.995
            df['tp'] = df['close'] * 1.01
            df['tsl'] = df['close'] - df['atr_14']
        except Exception as e:
            logger.error(f"MomentumBreakout_enhancedV1 error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class MeanReversion_enhancedV1(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            required_cols = ['rsi_14', 'macd_12_26_9', 'macds_12_26_9', 'atr_14']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in AlphaTrendStrategy: {missing_cols}")
                return df

            df['sma_20'] = df['close'].rolling(20).mean()
            df['boll_dist'] = (df['close'] - df['sma_20']) / df['sma_20']
            df['rsi_extreme'] = (df['rsi_14'] < 30) | (df['rsi_14'] > 70)
            df['macd_reversal'] = (df['macd_12_26_9'] < df['macds_12_26_9'])
            df['buy'] = (df['close'] < df['sma_20']) & (df['boll_dist'] < -0.01) & (df['rsi_14'] < 30)
            df['sell'] = (df['close'] > df['sma_20']) & (df['boll_dist'] > 0.01) & (df['rsi_14'] > 70)
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = df['close'] * 0.995
            df['tp'] = df['sma_20']
            df['tsl'] = df['close'] - df['atr_14']
        except Exception as e:
            logger.error(f"MeanReversion_enhancedV1 error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class IchimokuCloudStrategy_v1(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            if len(df) < 52:
                logger.warning("Not enough data for IchimokuCloudStrategy")
                df['signal'] = 0
                df['sl'] = np.nan
                df['tp'] = np.nan
                df['tsl'] = np.nan
                return df


            df['tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
            df['kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
            df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
            df['senkou_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
            df['chikou'] = df['close'].shift(-26)
            bullish = (df['close'] > df['senkou_a']) & (df['close'] > df['senkou_b']) & (df['tenkan'] > df['kijun'])
            bearish = (df['close'] < df['senkou_a']) & (df['close'] < df['senkou_b']) & (df['tenkan'] < df['kijun'])
            df.loc[bullish, 'signal'] = 1
            df.loc[bearish, 'signal'] = -1
            df['sl'] = df['close'] * 0.995
            df['tp'] = df['close'] * 1.01
            df['tsl'] = df['close'] - df['atr_14']
        except Exception as e:
            logger.error(f"IchimokuCloudStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class DonchianBreakoutStrategy_v1(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            required_cols = ['high', 'low', 'close', 'atr_14']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in AlphaTrendStrategy: {missing_cols}")
                return df

            high_20 = df['high'].rolling(20).max()
            low_20 = df['low'].rolling(20).min()
            buy = df['close'] > high_20.shift(1)
            sell = df['close'] < low_20.shift(1)
            df.loc[buy, 'signal'] = 1
            df.loc[sell, 'signal'] = -1
            df['sl'] = df['close'] * 0.995
            df['tp'] = df['close'] * 1.01
            df['tsl'] = df['close'] - df['atr_14']
        except Exception as e:
            logger.error(f"DonchianBreakoutStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class AlphaTrendStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['sl'] = np.nan
        df['tp'] = np.nan
        df['tsl'] = np.nan
        try:
            # Handle duplicate timestamps
            if df.index.has_duplicates:
                duplicate_count = df.index.duplicated().sum()
                logger.warning(f"AlphaTrendStrategy received DF with {duplicate_count} duplicate timestamps")
                logger.warning(f"Sample duplicates: {df.index[df.index.duplicated(keep=False)].unique()[:10].tolist()}")
                df = df[~df.index.duplicated(keep='last')]

            # Check for required columns
            required_cols = ['rsi_14', 'supertrend_10_3.0', 'volume', 'volume_avg', 'atr_14', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in AlphaTrendStrategy: {missing_cols}")
                return df

            # Define trading conditions
            volume_confirm = df['volume'] > df['volume_avg'] * 1.1  # Relaxed volume threshold
            buy_condition = (
                (df['rsi_14'] > 40) &  # Relaxed RSI threshold
                (df['close'] > df['supertrend_10_3.0'])  # Price above Supertrend
            )
            sell_condition = (
                (df['rsi_14'] < 60) &  # Relaxed RSI threshold
                (df['close'] < df['supertrend_10_3.0'])  # Price below Supertrend
            )

            # Apply volume confirmation
            df['buy'] = buy_condition & volume_confirm
            df['sell'] = sell_condition & volume_confirm

            # Assign signals
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1

            # Set SL, TP, and TSL
            df.loc[df['signal'] == 1, 'sl'] = df['close'] - df['atr_14'] * 1.0
            df.loc[df['signal'] == 1, 'tp'] = df['close'] + df['atr_14'] * 1.5
            df.loc[df['signal'] == -1, 'sl'] = df['close'] + df['atr_14'] * 1.0
            df.loc[df['signal'] == -1, 'tp'] = df['close'] - df['atr_14'] * 1.5
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'] * 0.5, np.nan)

            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"AlphaTrendStrategy: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"AlphaTrendStrategy conditions: "
                             f"rsi_14={df['rsi_14'].iloc[-1]:.2f}, "
                             f"close={df['close'].iloc[-1]:.2f}, "
                             f"supertrend={df['supertrend_10_3.0'].iloc[-1]:.2f}, "
                             f"volume={df['volume'].iloc[-1]:.2f}, "
                             f"volume_avg={df['volume_avg'].iloc[-1]:.2f}, "
                             f"params={self.params}")
            
        


        except Exception as e:
            logger.error(f"AlphaTrendStrategy error: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan

        return df

class BollingerStochasticCrossoverStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['buy'] = (
                (df['close'] < df['bb_lower']) &
                (df['stoch_k_14'] > df['stoch_d_14'])
            )
            df['sell'] = (
                (df['close'] > df['bb_upper']) &
                (df['stoch_k_14'] < df['stoch_d_14'])
            )
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = df['close'] * 0.995
            df['tp'] = df['close'] * 1.01
            df['tsl'] = df['close'] - df['atr_14']
        except Exception as e:
            logger.error(f"BollingerStochasticCrossoverStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df
class VWAPMomentumStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['vwap_above'] = df['close'] > df['vwap']
            df['momentum_up'] = df['roc_5'] > 0
            df['buy'] = df['vwap_above'] & df['momentum_up']
            df['sell'] = ~df['vwap_above'] & ~df['momentum_up']
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = df['close'] * 0.995
            df['tp'] = df['close'] * 1.01
            df['tsl'] = df['close'] - df['atr_14']
        except Exception as e:
            logger.error(f"VWAPMomentumStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df
class MACDStochasticTrendStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['sma_20'] = df['close'].rolling(20).mean()
            df['buy'] = (
                (df['macd_12_26_9'] > df['macds_12_26_9']) &
                (df['stoch_k_14'] > df['stoch_d_14']) &
                (df['stoch_k_14'] > self.params.get('stoch_oversold', 20)) &
                (df['stoch_k_14'] < self.params.get('stoch_overbought', 80)) &
                (df['close'] > df['sma_20'])
            )
            df['sell'] = (
                (df['macd_12_26_9'] < df['macds_12_26_9']) |
                ((df['stoch_k_14'] < df['stoch_d_14']) & (df['stoch_k_14'] > self.params.get('stoch_overbought', 80)))
            )
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"MACDStochasticTrend: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"MACDStochasticTrend conditions: macd_12_26_9={df['macd_12_26_9'].iloc[-1]:.2f}, "
                            f"macds_12_26_9={df['macds_12_26_9'].iloc[-1]:.2f}, "
                            f"stoch_k_14={df['stoch_k_14'].iloc[-1]:.2f}, "
                            f"stoch_d_14={df['stoch_d_14'].iloc[-1]:.2f}, "
                            f"sma_20={df['sma_20'].iloc[-1]:.2f}, "
                            f"close={df['close'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in MACDStochasticTrend: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

def make_strategy_func(cls):
    def strategy_func(row, history, params):
        if not hasattr(strategy_func, 'instance'):
            strategy_func.instance = cls(**params)
        
        if history.index.has_duplicates:
            duplicate_count = history.index.duplicated().sum()
            logger.warning(f"History DF has {duplicate_count} duplicate timestamps for {cls.__name__}")
            logger.warning(f"Sample duplicates: {history.index[history.index.duplicated(keep=False)].unique()[:10].tolist()}")
            history = history[~history.index.duplicated(keep='last')]
        
        history_slice = history.tail(5)
        
        logger.debug(f"Row name: {row.name}, Type: {type(row.name)}")
        
        current_row_df = row.to_frame().T
        try:
            if row.name is None:
                logger.error(f"Row name is None for {cls.__name__}")
                raise ValueError("Row name cannot be None")
            elif isinstance(row.name, pd.Timestamp):
                current_row_df.index = pd.DatetimeIndex([row.name])
            elif isinstance(row.name, str):
                current_row_df.index = pd.DatetimeIndex([pd.to_datetime(row.name)])
            else:
                logger.error(f"Unsupported row.name type for {cls.__name__}: {type(row.name)}")
                raise ValueError(f"Unsupported row.name type: {type(row.name)}")
        except Exception as e:
            logger.error(f"Failed to convert row timestamp for {cls.__name__}: {e}")
            raise ValueError(f"Invalid row timestamp: {row.name}")
        
        if row.name in history_slice.index:
            #logger.warning(f"Row timestamp {row.name} already in history for {cls.__name__}")
            history_slice = history_slice[history_slice.index != row.name]
        
        df = pd.concat([history_slice, current_row_df])
        
        if df.index.has_duplicates:
            logger.warning(f"Duplicate index labels found in small df for {cls.__name__} at timestamp {row.name}")
            logger.warning(f"Small df index: {df.index.tolist()}")
            df = df[~df.index.duplicated(keep='last')]
            logger.info(f"Small df shape after removing duplicates: {df.shape}")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(f"Small df index is not DatetimeIndex for {cls.__name__}: {type(df.index)}")
            raise ValueError("Small df must have DatetimeIndex")
        
        result = strategy_func.instance.generate_signals(df)
        signal = result['signal'].iloc[-1]
        sl = result['sl'].iloc[-1]
        tp = result['tp'].iloc[-1]
        tsl = result.get('tsl', np.nan)
        return {
            'signal': 'buy_potential' if signal == 1 else 'sell_potential' if signal == -1 else 'hold',
            'sl': sl,
            'tp': tp,
            'tsl': tsl
        }
    return strategy_func

strategy_factories = {
    # 'Support_Resistance': make_strategy_func(SupportResistanceStrategy),
    # 'Mean_Reversion': make_strategy_func(MeanReversionStrategy),
    # 'Momentum': make_strategy_func(MomentumStrategy),
    # 'BollingerSqueezeSpike': make_strategy_func(BollingerSqueezeSpikeStrategy),
    # 'ClosingBellBreakoutScalp': make_strategy_func(ClosingBellBreakoutScalpStrategy),
    # 'ExpiryDayVolatilitySpike': make_strategy_func(ExpiryDayVolatilitySpikeStrategy),
    # 'MeanReversionSnapBack': make_strategy_func(MeanReversionSnapBackStrategy),
    # 'ExpiryDayOTMScalp': make_strategy_func(ExpiryDayOTMScalpStrategy),
    # 'MomentumBreakoutRSI': make_strategy_func(MomentumBreakoutRSIStrategy),
    # 'VWAPReversalScalp': make_strategy_func(VWAPReversalScalpStrategy),
    # 'StraddleScalpHighVol': make_strategy_func(StraddleScalpHighVolStrategy),
    # 'ThreePMBollingerVolBreakout': make_strategy_func(ThreePMBollingerVolBreakoutStrategy), 
    # 'AlphaTrendStrategy_EnhancedV1': make_strategy_func(AlphaTrendStrategy_EnhancedV1),
    # 'EMACrossoverStrategy_EnhancedV1': make_strategy_func(EMACrossoverStrategy_EnhancedV1),
    # 'ThreePMBreakoutPowerBarStrategy_EnhancedV1': make_strategy_func(ThreePMBreakoutPowerBarStrategy_EnhancedV1),
    # 'MeanReversionSnapBackStrategy_EnhancedV1': make_strategy_func(MeanReversionSnapBackStrategy_EnhancedV1),
    # 'Momentum_Breakout': make_strategy_func(MomentumBreakoutStrategy),
    # 'MomentumBreakout_enhancedV1': make_strategy_func(MomentumBreakout_enhancedV1),
    # 'MeanReversion_enhancedV1': make_strategy_func(MeanReversion_enhancedV1),
    # 'IchimokuCloudStrategy_v1': make_strategy_func(IchimokuCloudStrategy_v1),
    # 'DonchianBreakoutStrategy_v1': make_strategy_func(DonchianBreakoutStrategy_v1),
    # 'RSIATRReversal': make_strategy_func(RSIATRReversalStrategy),
    # 'MACDTrendVolume': make_strategy_func(MACDTrendVolumeStrategy),
    # 'ORBVolumeMACD': make_strategy_func(ORBVolumeMACDStrategy),
    # 'ThreePMBreakoutPowerBar': make_strategy_func(ThreePMBreakoutPowerBarStrategy),
    # 'Breakout_ATR': make_strategy_func(BreakoutATRStrategy),
    'BollingerStochasticCrossoverStrategy': make_strategy_func(BollingerStochasticCrossoverStrategy),
    # 'VWAPMomentumStrategy': make_strategy_func(VWAPMomentumStrategy),
    # 'MACDStochasticTrendStrategy': make_strategy_func(MACDStochasticTrendStrategy),

   }
