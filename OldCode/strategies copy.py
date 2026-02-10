
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

# ---- CORE STRATEGIES ---- #


class EMACrossoverStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            short_p = self.params.get('short_window', 9)
            long_p = self.params.get('long_window', 21)
            ema_short_col = f'ema_{short_p}'
            ema_long_col = f'ema_{long_p}'

            # Ensure these columns exist! (Check against compute_indicators.py output)
            if not all(c in df.columns for c in [ema_short_col, ema_long_col]):
                logger.error(f"Missing EMA columns {ema_short_col} or {ema_long_col}")
                # Set signal to 0 and return or handle appropriately
                df['signal'] = 0; df['sl'] = np.nan; df['tp'] = np.nan; df['tsl'] = np.nan
                return df

            df['buy'] = (df[ema_short_col] > df[ema_long_col]) & \
                        (df[ema_short_col].shift(1) <= df[ema_long_col].shift(1))
            df['sell'] = (df[ema_short_col] < df[ema_long_col]) & \
                         (df[ema_short_col].shift(1) >= df[ema_long_col].shift(1))
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"EMACrossover: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"EMACrossover conditions: ema_9={df['ema_9'].iloc[-1]:.2f}, "
                            f"ema_21={df['ema_21'].iloc[-1]:.2f}, "
                            f"close={df['close'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in EMACrossover: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

class RSIMACDStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['buy'] = (df['rsi_14'] < self.params.get('rsi_oversold', 35)) & (df['macd_12_26_9'] > df['macds_12_26_9'])
            df['sell'] = (df['rsi_14'] > self.params.get('rsi_overbought', 65)) & (df['macd_12_26_9'] < df['macds_12_26_9'])
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"RSIMACD: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"RSIMACD conditions: rsi_14={df['rsi_14'].iloc[-1]:.2f}, "
                            f"macd_12_26_9={df['macd_12_26_9'].iloc[-1]:.2f}, "
                            f"macds_12_26_9={df['macds_12_26_9'].iloc[-1]:.2f}, "
                            f"close={df['close'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in RSIMACD: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

class SuperTrendADXStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        st_col = f"supertrend_{self.params.get('supert_period', 10)}_{self.params.get('supert_multiplier', 2.0)}"
        adx_col = f"adx_{self.params.get('adx_period', 14)}"
        df['signal'] = 0
        try:
            df['buy'] = (df[adx_col] > self.params.get('adx_threshold', 20)) & (df['close'] > df[st_col])
            df['sell'] = (df[adx_col] > self.params.get('adx_threshold', 20)) & (df['close'] < df[st_col])
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"SuperTrendADX: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"SuperTrendADX conditions: {adx_col}={df[adx_col].iloc[-1]:.2f}, "
                            f"{st_col}={df[st_col].iloc[-1]:.2f}, "
                            f"close={df['close'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in SuperTrendADX: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

class BollingerMeanReversionStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['buy'] = df['close'] < df['bb_lower']
            df['sell'] = df['close'] > df['bb_upper']
            df.loc[df['buy'], 'signal'] = 1
            df['sell'] = df['close'] > df['bb_upper']
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"BollingerMeanReversion: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"BollingerMeanReversion conditions: close={df['close'].iloc[-1]:.2f}, "
                            f"bb_lower={df['bb_lower'].iloc[-1]:.2f}, "
                            f"bb_upper={df['bb_upper'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in BollingerMeanReversion: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

class MomentumBreakoutStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            high_breakout = df['close'] > df['close'].rolling(20).max().shift(1)
            low_breakdown = df['close'] < df['close'].rolling(20).min().shift(1)
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
                            f"rolling_max={df['close'].rolling(20).max().shift(1).iloc[-1]:.2f}, "
                            f"rolling_min={df['close'].rolling(20).min().shift(1).iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in MomentumBreakout: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

class ATRBreakoutStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            atr_mult = self.params.get('atr_multiplier', 2.0)
            df['upper_band'] = df['close'].shift(1) + atr_mult * df['atr_14']
            df['lower_band'] = df['close'].shift(1) - atr_mult * df['atr_14']
            df.loc[df['close'] > df['upper_band'], 'signal'] = 1
            df.loc[df['close'] < df['lower_band'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = (df['close'] > df['upper_band']).sum()
            sell_signals = (df['close'] < df['lower_band']).sum()
            logger.debug(f"ATRBreakout: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"ATRBreakout conditions: close={df['close'].iloc[-1]:.2f}, "
                            f"upper_band={df['upper_band'].iloc[-1]:.2f}, "
                            f"lower_band={df['lower_band'].iloc[-1]:.2f}, "
                            f"atr_14={df['atr_14'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in ATRBreakout: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

# ---- ADVANCED MULTI-INDICATOR STRATEGIES ---- #

class RSIMACDSuperTrendStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['buy'] = (
                (df['rsi_14'] > 45) & (df['rsi_14'] < 60) &
                (df['macd_12_26_9'] > df['macds_12_26_9']) &
                (df['close'] > df['supertrend_10_3.0'])
            )
            df['sell'] = (
                (df['rsi_14'] > 65) & (df['close'] < df['supertrend_10_3.0'])
            )
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"RSIMACDSuperTrend: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"RSIMACDSuperTrend conditions: rsi_14={df['rsi_14'].iloc[-1]:.2f}, "
                            f"macd_12_26_9={df['macd_12_26_9'].iloc[-1]:.2f}, "
                            f"macds_12_26_9={df['macds_12_26_9'].iloc[-1]:.2f}, "
                            f"supertrend_10_3.0={df['supertrend_10_3.0'].iloc[-1]:.2f}, "
                            f"close={df['close'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in RSIMACDSuperTrend: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

class ADXMACDEMAStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['buy'] = (
                (df['adx_14'] > 20) &
                (df['ema_9'] > df['ema_21']) &
                (df['macd_12_26_9'] > df['macds_12_26_9'])
            )
            df['sell'] = (
                (df['adx_14'] < 20) | (df['ema_9'] < df['ema_21'])
            )
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"ADXMACDEMA: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"ADXMACDEMA conditions: adx_14={df['adx_14'].iloc[-1]:.2f}, "
                            f"ema_9={df['ema_9'].iloc[-1]:.2f}, "
                            f"ema_21={df['ema_21'].iloc[-1]:.2f}, "
                            f"macd_12_26_9={df['macd_12_26_9'].iloc[-1]:.2f}, "
                            f"macds_12_26_9={df['macds_12_26_9'].iloc[-1]:.2f}, "
                            f"close={df['close'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in ADXMACDEMA: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

class RSIBollingerMACDStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['buy'] = (
                (df['close'] < df['bb_lower']) &
                (df['rsi_14'] < 30) &
                (df['macd_12_26_9'] > df['macds_12_26_9'])
            )
            df['sell'] = (
                (df['close'] > df['bb_upper']) & (df['rsi_14'] > 70)
            )
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"RSIBollingerMACD: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"RSIBollingerMACD conditions: close={df['close'].iloc[-1]:.2f}, "
                            f"bb_lower={df['bb_lower'].iloc[-1]:.2f}, "
                            f"bb_upper={df['bb_upper'].iloc[-1]:.2f}, "
                            f"rsi_14={df['rsi_14'].iloc[-1]:.2f}, "
                            f"macd_12_26_9={df['macd_12_26_9'].iloc[-1]:.2f}, "
                            f"macds_12_26_9={df['macds_12_26_9'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in RSIBollingerMACD: {e}")
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
            df['buy'] = (df['close'] > rolling_high) & (df['close'] > df['supertrend_10_3.0'])
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

class SuperMomentumStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['score'] = 0
        df['signal'] = 0
        try:
            df['score'] += (df['rsi_14'] > 45) & (df['rsi_14'] < 65)
            df['score'] += (df['macd_12_26_9'] > df['macds_12_26_9'])
            df['score'] += df['close'] > df['supertrend_10_3.0']
            df['score'] += df['adx_14'] > 20
            df.loc[df['score'] >= 3, 'signal'] = 1
            df.loc[df['score'] <= 1, 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = (df['score'] >= 3).sum()
            sell_signals = (df['score'] <= 1).sum()
            logger.debug(f"SuperMomentum: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"SuperMomentum conditions: score={df['score'].iloc[-1]}, "
                            f"rsi_14={df['rsi_14'].iloc[-1]:.2f}, "
                            f"macd_12_26_9={df['macd_12_26_9'].iloc[-1]:.2f}, "
                            f"macds_12_26_9={df['macds_12_26_9'].iloc[-1]:.2f}, "
                            f"supertrend_10_3.0={df['supertrend_10_3.0'].iloc[-1]:.2f}, "
                            f"adx_14={df['adx_14'].iloc[-1]:.2f}, "
                            f"close={df['close'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in SuperMomentum: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

# ---- ADDITIONAL STRATEGIES ---- #

class StochasticOscillatorStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['buy'] = (
                (df['stoch_k_14'] > df['stoch_d_14']) &
                (df['stoch_k_14'].shift(1) <= df['stoch_d_14'].shift(1)) &
                (df['stoch_k_14'] < self.params.get('stoch_oversold', 30))
            )
            df['sell'] = (
                (df['stoch_k_14'] < df['stoch_d_14']) &
                (df['stoch_k_14'].shift(1) >= df['stoch_d_14'].shift(1)) &
                (df['stoch_k_14'] > self.params.get('stoch_overbought', 70))
            )
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"StochasticOscillator: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"StochasticOscillator conditions: stoch_k_14={df['stoch_k_14'].iloc[-1]:.2f}, "
                            f"stoch_d_14={df['stoch_d_14'].iloc[-1]:.2f}, "
                            f"oversold={self.params.get('stoch_oversold', 30)}, "
                            f"overbought={self.params.get('stoch_overbought', 70)}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in StochasticOscillator: {e}")
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
            df['buy'] = (df['close'] <= df['support']) & (df['close'].shift(1) > df['support'].shift(1))
            df['sell'] = (df['close'] >= df['resistance']) & (df['close'].shift(1) < df['resistance'].shift(1))
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
            df['buy'] = df['close'] < df['sma_20']
            df['sell'] = df['close'] > df['sma_20']
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
            df['buy'] = df['roc_5'] > self.params.get('roc_threshold', 0)
            df['sell'] = df['roc_5'] < -self.params.get('roc_threshold', 0)
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

class RSIStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['buy'] = df['rsi_14'] < self.params.get('rsi_oversold', 30)
            df['sell'] = df['rsi_14'] > self.params.get('rsi_overbought', 70)
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"RSI: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"RSI conditions: rsi_14={df['rsi_14'].iloc[-1]:.2f}, "
                            f"oversold={self.params.get('rsi_oversold', 30)}, "
                            f"overbought={self.params.get('rsi_overbought', 70)}, "
                            f"close={df['close'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in RSI: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

class MACDCrossoverStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['buy'] = (df['macd_12_26_9'] > df['macds_12_26_9']) & (df['macd_12_26_9'].shift(1) <= df['macds_12_26_9'].shift(1))
            df['sell'] = (df['macd_12_26_9'] < df['macds_12_26_9']) & (df['macd_12_26_9'].shift(1) >= df['macds_12_26_9'].shift(1))
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"MACDCrossover: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"MACDCrossover conditions: macd_12_26_9={df['macd_12_26_9'].iloc[-1]:.2f}, "
                            f"macds_12_26_9={df['macds_12_26_9'].iloc[-1]:.2f}, "
                            f"close={df['close'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in MACDCrossover: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

# ---- NEW CUSTOM STRATEGIES ---- #

class ATRSuperAlphaTrendStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['buy_cond'] = (
                (df['rsi_14'] > self.params.get('rsi_lower', 40)) &
                (df['rsi_14'] < self.params.get('rsi_upper', 60)) &
                (df['close'] > df['supertrend_10_3.0']) &
                (df['atr_14'] > self.params.get('atr_threshold', 0.5))
            )
            df['sell_cond'] = (
                (df['rsi_14'] > self.params.get('rsi_overbought', 70)) |
                (df['close'] < df['supertrend_10_3.0'])
            )
            df.loc[df['buy_cond'], 'signal'] = 1
            df.loc[df['sell_cond'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy_cond'].sum()
            sell_signals = df['sell_cond'].sum()
            logger.debug(f"ATRSuperAlphaTrend: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"ATRSuperAlphaTrend conditions: rsi_14={df['rsi_14'].iloc[-1]:.2f}, "
                            f"supertrend_10_3.0={df['supertrend_10_3.0'].iloc[-1]:.2f}, "
                            f"atr_14={df['atr_14'].iloc[-1]:.2f}, "
                            f"close={df['close'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in ATRSuperAlphaTrend: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

class RSIBollingerConfluenceStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['buy'] = (
                (df['close'] < df['bb_lower']) &
                (df['rsi_14'] < self.params.get('rsi_oversold', 30)) &
                (df['close'] > df['ema_9'])
            )
            df['sell'] = (
                (df['close'] > df['bb_upper']) &
                (df['rsi_14'] > self.params.get('rsi_overbought', 70))
            )
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"RSIBollingerConfluence: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"RSIBollingerConfluence conditions: close={df['close'].iloc[-1]:.2f}, "
                            f"bb_lower={df['bb_lower'].iloc[-1]:.2f}, "
                            f"bb_upper={df['bb_upper'].iloc[-1]:.2f}, "
                            f"rsi_14={df['rsi_14'].iloc[-1]:.2f}, "
                            f"ema_9={df['ema_9'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in RSIBollingerConfluence: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
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

class ADXVolatilityBreakoutStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            atr_mult = self.params.get('atr_multiplier', 1.5)
            df['upper_band'] = df['ema_21'] + atr_mult * df['atr_14']
            df['lower_band'] = df['ema_21'] - atr_mult * df['atr_14']
            df['buy'] = (
                (df['adx_14'] > self.params.get('adx_threshold', 25)) &
                (df['close'] > df['upper_band']) &
                (df['close'] > df['ema_21'])
            )
            df['sell'] = (
                (df['adx_14'] < self.params.get('adx_threshold', 20)) |
                (df['close'] < df['lower_band'])
            )
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"ADXVolatilityBreakout: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"ADXVolatilityBreakout conditions: adx_14={df['adx_14'].iloc[-1]:.2f}, "
                            f"close={df['close'].iloc[-1]:.2f}, "
                            f"upper_band={df['upper_band'].iloc[-1]:.2f}, "
                            f"lower_band={df['lower_band'].iloc[-1]:.2f}, "
                            f"ema_21={df['ema_21'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in ADXVolatilityBreakout: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
        return df

class TripleEMAMomentumStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['buy'] = (
                (df['ema_5'] > df['ema_13']) &
                (df['ema_13'] > df['ema_34']) &
                (df['rsi_14'] > self.params.get('rsi_momentum', 50))
            )
            df['sell'] = (
                (df['ema_5'] < df['ema_13']) |
                (df['rsi_14'] < self.params.get('rsi_exit', 40))
            )
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995, np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01, np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"TripleEMAMomentum: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"TripleEMAMomentum conditions: ema_5={df['ema_5'].iloc[-1]:.2f}, "
                            f"ema_13={df['ema_13'].iloc[-1]:.2f}, "
                            f"ema_34={df['ema_34'].iloc[-1]:.2f}, "
                            f"rsi_14={df['rsi_14'].iloc[-1]:.2f}, "
                            f"close={df['close'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except KeyError as e:
            logger.error(f"Missing column in TripleEMAMomentum: {e}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
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

class EMARollingADXStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            adx_col = f"adx_{self.params.get('adx_period', 14)}"
            df['ema_up'] = df['ema_9'] > df['ema_21']
            df['ema_down'] = df['ema_9'] < df['ema_21']
            df['trend_strong'] = df[adx_col] > 25
            df['buy'] = df['ema_up'] & df['trend_strong']
            df['sell'] = df['ema_down'] & df['trend_strong']
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = df['close'] * 0.995
            df['tp'] = df['close'] * 1.01
            df['tsl'] = df['close'] - df['atr_14']
        except Exception as e:
            logger.error(f"EMARollingADXStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

# class BollingerStochasticCrossoverStrategy(BaseStrategy):
#     def generate_signals(self, data):
#         df = data.copy()
#         df['signal'] = 0
#         try:
#             df['buy'] = (
#                 (df['close'] < df['bb_lower']) &
#                 (df['stoch_k_14'] > df['stoch_d_14'])
#             )
#             df['sell'] = (
#                 (df['close'] > df['bb_upper']) &
#                 (df['stoch_k_14'] < df['stoch_d_14'])
#             )
#             df.loc[df['buy'], 'signal'] = 1
#             df.loc[df['sell'], 'signal'] = -1
#             df['sl'] = df['close'] * 0.995
#             df['tp'] = df['close'] * 1.01
#             df['tsl'] = df['close'] - df['atr_14']
#         except Exception as e:
#             logger.error(f"BollingerStochasticCrossoverStrategy error: {e}")
#             df[['signal', 'sl', 'tp', 'tsl']] = np.nan
#         return df

class BollingerStochasticCrossoverStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            # Relaxed Bollinger Band conditions with a multiplier
            bb_multiplier = 0.5  # Adjustable: 0.5 means halfway to bb_lower/bb_upper
            df['buy'] = (
                (df['close'] < (df['bb_lower'] + bb_multiplier * (df['bb_middle'] - df['bb_lower']))) &
                (df['stoch_k_14'] > df['stoch_d_14']) &
                (df['stoch_k_14'] < 80)  # Avoid overbought
            )
            df['sell'] = (
                (df['close'] > (df['bb_upper'] - bb_multiplier * (df['bb_upper'] - df['bb_middle']))) &
                (df['stoch_k_14'] < df['stoch_d_14']) &
                (df['stoch_k_14'] > 20)  # Avoid oversold
            )
            # Optional RSI filter to increase signals in trending markets
            df['rsi_filter'] = (df['rsi_14'] > 30) & (df['rsi_14'] < 70)
            df.loc[df['buy'] & df['rsi_filter'], 'signal'] = 1
            df.loc[df['sell'] & df['rsi_filter'], 'signal'] = -1
            # Adjusted SL, TP, TSL
            df['sl'] = df['close'] * 0.99  # Wider SL: 1% below entry
            df['tp'] = df['close'] * 1.015  # Wider TP: 1.5% above entry
            df['tsl'] = df['close'] - 0.5 * df['atr_14']  # Tighter TSL
        except Exception as e:
            logger.error(f"BollingerStochasticCrossoverStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df


class RSIATRReversalStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['buy'] = (df['rsi_14'] < 30) & (df['atr_14'] > df['atr_14'].rolling(14).mean())
            df['sell'] = (df['rsi_14'] > 70) & (df['atr_14'] > df['atr_14'].rolling(14).mean())
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
            df['volume_breakout'] = df['volume'] > 1.5 * avg_volume
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
            df['orb_high'] = df['high'].rolling(window=3, min_periods=1).max()
            df['orb_low'] = df['low'].rolling(window=3, min_periods=1).min()
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

# === HERO-ZERO 3PM STRATEGIES === #

class ThreePMBreakoutPowerBarStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['is_3pm'] = df.index.map(lambda ts: ts.hour == 14 and ts.minute >= 50)
            df['body'] = abs(df['close'] - df['open'])
            df['avg_volume'] = df['volume'].rolling(20).mean()
            df['strong_bull'] = (df['close'] > df['open']) & (df['body'] > df['body'].rolling(20).mean())
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

class VWAPRSIMACD3PMSpikeStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['is_post_255'] = df.index.map(lambda ts: ts.hour == 14 and ts.minute >= 55)
            df['rsi_neutral'] = (df['rsi_14'] > 45) & (df['rsi_14'] < 55)
            df['macd_cross'] = df['macd_12_26_9'] > df['macds_12_26_9']
            df['buy'] = df['is_post_255'] & df['rsi_neutral'] & df['macd_cross'] & (df['close'] > df['vwap'])
            df['sell'] = df['is_post_255'] & (~df['macd_cross']) & (df['close'] < df['vwap'])
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = df['close'] * 0.995
            df['tp'] = df['close'] * 1.01
            df['tsl'] = df['close'] - df['atr_14']
        except Exception as e:
            logger.error(f"VWAPRSIMACD3PMSpikeStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class SupertrendReversal3PMStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            st_col = 'supertrend_10_3.0'
            df['is_3pm'] = df.index.map(lambda ts: ts.hour == 14 and ts.minute >= 50)
            df['adx_ok'] = df['adx_14'] > 20
            df['wick_large'] = abs(df['high'] - df['close']) > df['atr_14'] * 0.5
            df['supertrend_reversal'] = (df['close'] > df[st_col]) & (df['close'].shift(1) < df[st_col].shift(1))
            df['buy'] = df['is_3pm'] & df['supertrend_reversal'] & df['adx_ok'] & df['wick_large']
            df['sell'] = df['is_3pm'] & (~df['supertrend_reversal']) & df['adx_ok'] & df['wick_large']
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = df['close'] * 0.995
            df['tp'] = df['close'] * 1.01
            df['tsl'] = df['close'] - df['atr_14']
        except Exception as e:
            logger.error(f"SupertrendReversal3PMStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class BollingerSqueezeSpikeStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['is_3pm'] = df.index.map(lambda ts: ts.hour == 14 and ts.minute >= 50)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.7
            df['volume_spike'] = df['volume'] > df['volume'].rolling(20).mean() * 1.5
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

# --- ENHANCED STRATEGY EXAMPLE ---
class AlphaTrendStrategy_EnhancedV1(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            # Ensure necessary base indicators are present/calculated
            # For this example, we assume 'rsi_10', 'supertrend_10_3.0', 'ema_50', 'volume_avg', 'adx_14' are in df
            # If not, you'd calculate them here or ensure they are pre-calculated.
            # Example: df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            #          df['volume_avg'] = df['volume'].rolling(window=20).mean()
            #          df['adx_14'] = calculate_adx(df, period=14) # Assuming a function or library for ADX

            required_cols = ['rsi_10', 'supertrend_10_3.0', 'ema_50', 'volume', 'volume_avg', 'adx_14', 'atr_14']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing column for AlphaTrendStrategy_EnhancedV1: {col}. Check data preparation.")
                    # For critical columns, might be best to return df with no signals
                    # or use fallback default values if appropriate (though less ideal for core logic)
                    # Example: if col == 'ema_50': df['ema_50'] = df['close'] # Very crude fallback
                    return df # Exit if critical data is missing after logging

            # Regime Filter
            is_uptrend_regime = df['close'] > df['ema_50']
            is_downtrend_regime = df['close'] < df['ema_50']

            # Volume Confirmation
            has_volume_confirmation = df['volume'] > (df['volume_avg'] * self.params.get('volume_conf_mult', 1.2))

            # ADX Trend Strength
            has_trend_strength = df['adx_14'] > self.params.get('adx_min_strength', 20)

            # Original Buy Conditions
            original_buy_cond = (
                (df['rsi_10'] > self.params.get('rsi_buy_lower', 35)) &
                (df['rsi_10'] < self.params.get('rsi_buy_upper', 65)) &
                (df['supertrend_10_3.0'] < df['close'])
            )

            # Enhanced Buy Conditions
            df['buy_cond_enhanced'] = (
                original_buy_cond &
                is_uptrend_regime &      # Regime filter: Only buy in uptrend
                has_volume_confirmation & # Volume filter
                has_trend_strength       # ADX filter
            )

            # Original Sell Conditions
            original_sell_cond = (
                (df['rsi_10'] > self.params.get('rsi_sell_lower', 55)) &
                (df['rsi_10'] < self.params.get('rsi_sell_upper', 75)) &
                (df['supertrend_10_3.0'] > df['close'])
            )

            # Enhanced Sell Conditions
            df['sell_cond_enhanced'] = (
                original_sell_cond &
                is_downtrend_regime &    # Regime filter: Only sell in downtrend
                has_volume_confirmation & # Volume filter
                has_trend_strength       # ADX filter
            )

            df.loc[df['buy_cond_enhanced'], 'signal'] = 1
            df.loc[df['sell_cond_enhanced'], 'signal'] = -1

            # SL/TP - Can also be made adaptive, e.g., using ATR
            atr_multiplier_sl = self.params.get('atr_sl_mult', 1.5)
            atr_multiplier_tp = self.params.get('atr_tp_mult', 2.0)

            df['sl'] = np.where(df['signal'] == 1, df['close'] - (df['atr_14'] * atr_multiplier_sl),
                                np.where(df['signal'] == -1, df['close'] + (df['atr_14'] * atr_multiplier_sl), np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] + (df['atr_14'] * atr_multiplier_tp),
                                np.where(df['signal'] == -1, df['close'] - (df['atr_14'] * atr_multiplier_tp), np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan) # Original TSL

            buy_signals = df['buy_cond_enhanced'].sum()
            sell_signals = df['sell_cond_enhanced'].sum()
            logger.debug(f"AlphaTrend_EnhancedV1: Buy signals: {buy_signals}, Sell signals: {sell_signals}")

        except KeyError as e: # Should be caught by individual checks now, but good as a fallback
            logger.error(f"Missing column in AlphaTrendStrategy_EnhancedV1: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = 0 # Use 0 for signal, np.nan for SL/TP/TSL
            df['sl'] = np.nan; df['tp'] = np.nan; df['tsl'] = np.nan;

        except Exception as e:
            logger.error(f"Error in AlphaTrendStrategy_EnhancedV1: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = 0
            df['sl'] = np.nan; df['tp'] = np.nan; df['tsl'] = np.nan;
        return df

class AlphaTrendStrategy(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        
        if df.index.has_duplicates:
            duplicate_count = df.index.duplicated().sum()
            logger.warning(f"AlphaTrendStrategy received DF with {duplicate_count} duplicate timestamps")
            logger.warning(f"Sample duplicates: {df.index[df.index.duplicated(keep=False)].unique()[:10].tolist()}")
            df = df[~df.index.duplicated(keep='last')]
            logger.info(f"DataFrame shape after removing duplicates: {df.shape}")
        
        df['signal'] = 0
        required_columns = ['rsi_10', 'supertrend_10_3.0', 'close', 'atr_14']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in AlphaTrendStrategy: {missing_cols}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
            return df
        
        try:
            df['buy_cond'] = (
                (df['rsi_10'] > self.params.get('rsi_buy_lower', 35)) &
                (df['rsi_10'] < self.params.get('rsi_buy_upper', 65)) &
                (df['supertrend_10_3.0'] < df['close'])
            )
            df['sell_cond'] = (
                (df['rsi_10'] > self.params.get('rsi_sell_lower', 55)) &
                (df['rsi_10'] < self.params.get('rsi_sell_upper', 75)) &
                (df['supertrend_10_3.0'] > df['close'])
            )
            df.loc[df['buy_cond'], 'signal'] = 1
            df.loc[df['sell_cond'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995,
                               np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01,
                               np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            buy_signals = df['buy_cond'].sum()
            sell_signals = df['sell_cond'].sum()
            logger.debug(f"AlphaTrend: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"AlphaTrend conditions: rsi_10={df['rsi_10'].iloc[-1]:.2f}, "
                            f"supertrend_10_3.0={df['supertrend_10_3.0'].iloc[-1]:.2f}, "
                            f"close={df['close'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except Exception as e:
            logger.error(f"Error in AlphaTrendStrategy: {e}")
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
            # Ensure index is timezone-aware (IST)
            if not df.index.tz:
                logger.warning("DataFrame index is not timezone-aware. Localizing to IST.")
                df.index = df.index.tz_localize('Asia/Kolkata')
            
            # Widen time window: 2:30 PM to 3:30 PM IST
            # df['is_timeslot'] = df.index.to_series().apply(
            #     lambda ts: (ts.time() >= time(14, 30) and ts.time() <= time(15, 30))
            # )
            def in_timeslot(ts):
                if pd.isna(ts):
                    return False
                t = ts.time()
                return (t >= time(14, 30)) and (t <= time(15, 30))
            df['is_timeslot'] = df.index.to_series().apply(in_timeslot)
            timeslot_count = df['is_timeslot'].sum()
            logger.debug(f"ClosingBellBreakoutScalp: {timeslot_count} rows in time window 2:30-3:30 PM")

            df['body'] = abs(df['close'] - df['open'])
            df['body_avg'] = df['body'].rolling(window=20, min_periods=5).mean()
            df['volume_avg'] = df['volume'].rolling(window=20, min_periods=5).mean()

            # Relaxed conditions
            df['strong_bull_candle'] = (df['close'] > df['open']) & (df['body'] > df['body_avg'] * 1.0)  # From 1.1
            df['strong_bear_candle'] = (df['open'] > df['close']) & (df['body'] > df['body_avg'] * 1.0)
            df['volume_spike'] = df['volume'] > df['volume_avg'] * 1.2  # From 1.5

            # VWAP check
            if 'vwap' not in df.columns or df['vwap'].isna().all():
                logger.error("VWAP not found or all NaN in data. Cannot generate signals.")
                return df
            df['vwap_buy_condition'] = df['close'] > df['vwap']
            df['vwap_sell_condition'] = df['close'] < df['vwap']

            # ATR check
            if 'atr_14' not in df.columns:
                logger.warning("atr_14 not found. Using 0.3% of close for SL/TP.")
                atr_for_sl_tp = df['close'] * 0.003
            else:
                atr_for_sl_tp = df['atr_14']

            # Buy Signal
            buy_conditions = (
                df['is_timeslot'] &
                df['strong_bull_candle'] &
                df['volume_spike'] &
                df['vwap_buy_condition']
            )
            df.loc[buy_conditions, 'signal'] = 1

            # Sell Signal
            sell_conditions = (
                df['is_timeslot'] &
                df['strong_bear_candle'] &
                df['volume_spike'] &
                df['vwap_sell_condition']
            )
            df.loc[sell_conditions, 'signal'] = -1

            # SL and TP
            df.loc[df['signal'] == 1, 'sl'] = df['close'] - (atr_for_sl_tp * 1.2)  # Relaxed from 1.5
            #df.loc[df['signal'] == 1, 'tp'] = df['close'] + (atr_for_sl_tp * 2.5)  # Relaxed from 3.0
            df.loc[df['signal'] == 1, 'tp'] = df['close'] + (df['atr_14'] * 2.5)

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
            # Ensure index is timezone-aware (IST)
            if not df.index.tz:
                logger.warning("DataFrame index is not timezone-aware. Localizing to IST.")
                df.index = df.index.tz_localize('Asia/Kolkata')
            
            # Widen time window: 2:30 PM to 3:30 PM IST
            df['is_timeslot'] = df.index.to_series().apply(
                lambda ts: (ts.time() >= time(14, 30) and ts.time() <= time(15, 30))
            )
            timeslot_count = df['is_timeslot'].sum()
            logger.debug(f"ExpiryDayVolatilitySpike: {timeslot_count} rows in time window 2:30-3:30 PM")

            # Check for is_expiry_day, allow non-expiry days for testing
            if 'is_expiry_day' not in df.columns:
                logger.warning("is_expiry_day not found. Assuming all days are non-expiry.")
                df['is_expiry_day'] = False
            expiry_count = df['is_expiry_day'].sum()
            logger.debug(f"ExpiryDayVolatilitySpike: {expiry_count} expiry day rows")

            # Bollinger Bands
            df['bb_mid'] = df['close'].rolling(window=self.bb_window).mean()
            df['bb_std'] = df['close'].rolling(window=self.bb_window).std()
            df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * self.bb_std_dev)
            df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * self.bb_std_dev)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']

            # ATR for squeeze threshold
            if 'atr_14' not in df.columns:
                logger.warning("atr_14 not found. Using IQR-based fallback.")
                df['squeeze_ref_atr'] = df['close'].rolling(window=self.bb_window).apply(
                    lambda x: np.percentile(x, 75) - np.percentile(x, 25), raw=True
                ) / 2
                df['squeeze_ref_atr'] = df['squeeze_ref_atr'].fillna(method='bfill').fillna(method='ffill')
            else:
                df['squeeze_ref_atr'] = df['atr_14']

            # Squeeze condition
            df['is_squeeze'] = df['bb_width'] < (df['squeeze_ref_atr'] * self.squeeze_threshold_multiplier)

            df['volume_avg'] = df['volume'].rolling(window=20, min_periods=5).mean()
            df['volume_spike'] = df['volume'] > df['volume_avg'] * 1.2  # Relaxed from 1.7

            # Buy Signal (allow non-expiry days for testing)
            buy_conditions = (
                df['is_timeslot'] &
                df['is_squeeze'].shift(1) &
                (df['close'] > df['bb_upper'].shift(1)) &
                df['volume_spike']
            )
            df.loc[buy_conditions, 'signal'] = 1

            # Sell Signal
            sell_conditions = (
                df['is_timeslot'] &
                df['is_squeeze'].shift(1) &
                (df['close'] < df['bb_lower'].shift(1)) &
                df['volume_spike']
            )
            df.loc[sell_conditions, 'signal'] = -1

            # SL/TP
            atr_for_sl_tp = df['squeeze_ref_atr']
            df.loc[df['signal'] == 1, 'sl'] = df['close'] - (atr_for_sl_tp * 1.5)  # Relaxed from 2.0
            df.loc[df['signal'] == 1, 'tp'] = df['close'] + (atr_for_sl_tp * 3.0)  # Relaxed from 4.0
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
    """
    Strategy 1: Expiry-Day OTM Options Scalp
    Targets OTM options on expiry day, entering on Bollinger Band breakout at ~3:00 PM.
    """
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['sl'] = np.nan
        df['tp'] = np.nan
        df['tsl'] = np.nan
        try:
            # Parameters
            bb_period = self.params.get('bb_period', 20)
            bb_std = self.params.get('bb_std', 2.0)
            volume_multiplier = self.params.get('volume_multiplier', 1.5)
            sl_atr_mult = self.params.get('sl_atr_mult', 1.0)
            tp_atr_mult = self.params.get('tp_atr_mult', 0.5)
            start_hour = self.params.get('start_hour', 14)
            start_minute = self.params.get('start_minute', 55)
            end_hour = self.params.get('end_hour', 15)
            end_minute = self.params.get('end_minute', 20)

            # Time window (2:55 PM to 3:20 PM)
            df['is_3pm'] = df.index.map(lambda ts: (ts.hour == start_hour and ts.minute >= start_minute) or 
                                         (ts.hour == end_hour and ts.minute <= end_minute))
            
            # Volume condition
            df['avg_volume'] = df['volume'].rolling(window=5).mean()
            df['volume_spike'] = df['volume'] > volume_multiplier * df['avg_volume']
            
            # Breakout conditions
            df['buy'] = (df['is_3pm'] & 
                        (df['close'] > df['bollinger_hband']) & 
                        df['volume_spike'])
            df['sell'] = (df['is_3pm'] & 
                         (df['close'] < df['bollinger_lband']) & 
                         df['volume_spike'])
            
            # Set signals
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            
            # Set SL, TP, TSL
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
    """
    Strategy 2: Momentum Breakout Scalp with RSI Confirmation
    Enters ATM options on EMA breakout with RSI confirmation at ~3:00 PM.
    """
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['sl'] = np.nan
        df['tp'] = np.nan
        df['tsl'] = np.nan
        try:
            # Parameters
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

            # Time window (2:55 PM to 3:20 PM)
            df['is_3pm'] = df.index.map(lambda ts: (ts.hour == start_hour and ts.minute >= start_minute) or 
                                         (ts.hour == end_hour and ts.minute <= end_minute))
            
            # Volume condition
            df['avg_volume'] = df['volume'].rolling(window=5).mean()
            df['volume_spike'] = df['volume'] > volume_multiplier * df['avg_volume']
            
            # Breakout conditions
            df['buy'] = (df['is_3pm'] & 
                        (df['close'] > df[f'ema_{ema_period}']) & 
                        (df['rsi'] > rsi_overbought) & 
                        df['volume_spike'])
            df['sell'] = (df['is_3pm'] & 
                         (df['close'] < df[f'ema_{ema_period}']) & 
                         (df['rsi'] < rsi_oversold) & 
                         df['volume_spike'])
            
            # Set signals
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            
            # Set SL, TP, TSL
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
    """
    Strategy 3: VWAP Reversal Scalp
    Enters ATM/ITM options on VWAP reversal with Stochastic confirmation at ~3:00 PM.
    """
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['sl'] = np.nan
        df['tp'] = np.nan
        df['tsl'] = np.nan
        try:
            # Parameters
            stoch_k_period = self.params.get('stoch_k_period', 14)
            stoch_d_period = self.params.get('stoch_d_period', 3)
            stoch_overbought = self.params.get('stoch_overbought', 70)
            stoch_oversold = self.params.get('stoch_oversold', 30)
            vwap_tolerance = self.params.get('vwap_tolerance', 0.001)
            volume_multiplier = self.params.get('volume_multiplier', 1.5)
            sl_atr_mult = self.params.get('sl_atr_mult', 1.0)
            tp_atr_mult = self.params.get('tp_atr_mult', 0.5)
            start_hour = self.params.get('start_hour', 14)
            start_minute = self.params.get('start_minute', 50)
            end_hour = self.params.get('end_hour', 15)
            end_minute = self.params.get('end_minute', 20)

            # Time window (2:50 PM to 3:20 PM)
            df['is_3pm'] = df.index.map(lambda ts: (ts.hour == start_hour and ts.minute >= start_minute) or 
                                         (ts.hour == end_hour and ts.minute <= end_minute))
            
            # Volume condition
            df['avg_volume'] = df['volume'].rolling(window=5).mean()
            df['volume_spike'] = df['volume'] > volume_multiplier * df['avg_volume']
            
            # VWAP proximity and reversal
            df['vwap_proximity'] = (df['close'].shift(1) - df['vwap']).abs() / df['vwap'] <= vwap_tolerance
            df['stoch_k_cross_d'] = df['stoch_k'] > df['stoch_d']
            df['stoch_k_cross_d_reverse'] = df['stoch_k'] < df['stoch_d']
            
            # Reversal conditions
            df['buy'] = (df['is_3pm'] & 
                        df['vwap_proximity'] & 
                        (df['close'] > df['vwap']) & 
                        df['stoch_k_cross_d'] & 
                        (df['stoch_k'] < stoch_oversold) & 
                        df['volume_spike'])
            df['sell'] = (df['is_3pm'] & 
                         df['vwap_proximity'] & 
                         (df['close'] < df['vwap']) & 
                         df['stoch_k_cross_d_reverse'] & 
                         (df['stoch_k'] > stoch_overbought) & 
                         df['volume_spike'])
            
            # Set signals
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            
            # Set SL, TP, TSL
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
    """
    Strategy 4: Straddle Scalp on High Volatility
    Buys ATM call + put on high volatility, expecting a 3:00 PM spike in either direction.
    """
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['sl'] = np.nan
        df['tp'] = np.nan
        df['tsl'] = np.nan
        try:
            # Parameters
            vix_threshold = self.params.get('vix_threshold', 15.0)
            atr_threshold = self.params.get('atr_threshold', 0.002)
            range_threshold = self.params.get('range_threshold', 0.001)
            sl_atr_mult = self.params.get('sl_atr_mult', 1.0)
            tp_atr_mult = self.params.get('tp_atr_mult', 0.5)
            start_hour = self.params.get('start_hour', 14)
            start_minute = self.params.get('start_minute', 50)
            end_hour = self.params.get('end_hour', 15)
            end_minute = self.params.get('end_minute', 20)

            # Check required columns
            required_cols = ['close', 'high', 'low', 'atr_14', 'volatility_20']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in StraddleScalpHighVolStrategy: {missing_cols}")
                return df

            # Time window (2:50 PM to 3:20 PM)
            df['is_3pm'] = df.index.map(lambda ts: (ts.hour == start_hour and ts.minute >= start_minute) or 
                                         (ts.hour == end_hour and ts.minute <= end_minute))
            
            # Consolidation and volatility
            df['recent_range'] = (df['high'].rolling(window=5).max() - 
                                df['low'].rolling(window=5).min()) / df['close']
            # Use volatility_20 instead of vix, scaled to approximate VIX
            df['high_vol'] = (df['volatility_20'] * 100 > vix_threshold) & (df['atr_14'] / df['close'] > atr_threshold)
            
            # Straddle condition
            df['straddle'] = (df['is_3pm'] & 
                            (df['recent_range'] <= range_threshold) & 
                            df['high_vol'])
            
            # Set signals (1 for call, -1 for put, handled in backtester)
            df.loc[df['straddle'], 'signal'] = 1  # Buy call
            # Note: Backtester should handle dual signals for put
            df.loc[df['straddle'], 'signal'] = -1  # Buy put (overwrites call; see note)
            
            # Set SL, TP, TSL (for call; put mirrors in backtester)
            df.loc[df['straddle'], 'sl'] = df['close'] - sl_atr_mult * df['atr_14']
            df.loc[df['straddle'], 'tp'] = df['close'] + tp_atr_mult * df['atr_14']
            df.loc[df['straddle'], 'tsl'] = df['close'] - sl_atr_mult * df['atr_14'] * 0.5
            
        except Exception as e:
            logger.error(f"StraddleScalpHighVolStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        
        return df


class MeanReversionSnapBackStrategy(BaseStrategy):
    def __init__(self, **params):
        super().__init__(**params)
        self.rsi_period = self.params.get('rsi_period', 7)
        self.rsi_ob = self.params.get('rsi_ob', 80)  # Relaxed from 75
        self.rsi_os = self.params.get('rsi_os', 20)  # Relaxed from 25
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
            # Ensure index is timezone-aware (IST)
            if not df.index.tz:
                logger.warning("DataFrame index is not timezone-aware. Localizing to IST.")
                df.index = df.index.tz_localize('Asia/Kolkata')
            
            # Check required columns
            required_cols = ['close', 'high', 'low']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in MeanReversionSnapBackStrategy: {missing_cols}")
                return df

            # Widen time window: 2:30 PM to 3:30 PM IST
            df['is_timeslot'] = df.index.to_series().apply(
                lambda ts: (ts.time() >= time(14, 30) and ts.time() <= time(15, 30))
            )
            timeslot_count = df['is_timeslot'].sum()
            logger.debug(f"MeanReversionSnapBack: {timeslot_count} rows in time window 2:30-3:30 PM")

            # Calculate RSI
            df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
            df['ema_short'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()

            # Use atr_14 if available, else fallback
            if 'atr_14' not in df.columns:
                logger.warning("atr_14 not found. Using 0.2% of close for SL/TP.")
                atr_for_sl_tp = df['close'] * 0.002
            else:
                atr_for_sl_tp = df['atr_14']

            # Buy Signal (relaxed RSI condition)
            df['was_oversold'] = (df['rsi'].shift(1) < self.rsi_os) | (df['rsi'].shift(2) < self.rsi_os)
            buy_reversal_confirmation = (df['close'] > df['low'].shift(1))
            buy_conditions = (
                df['is_timeslot'] &
                df['was_oversold'] &
                buy_reversal_confirmation
            )
            df.loc[buy_conditions, 'signal'] = 1

            # Sell Signal
            df['was_overbought'] = (df['rsi'].shift(1) > self.rsi_ob) | (df['rsi'].shift(2) > self.rsi_ob)
            sell_reversal_confirmation = (df['close'] < df['high'].shift(1))
            sell_conditions = (
                df['is_timeslot'] &
                df['was_overbought'] &
                sell_reversal_confirmation
            )
            df.loc[sell_conditions, 'signal'] = -1

            # SL/TP with NaN checks
            df.loc[df['signal'] == 1, 'sl'] = df['low'].shift(1) - atr_for_sl_tp * 0.3
            df.loc[df['signal'] == 1, 'tp'] = df['close'] + (atr_for_sl_tp * 1.5)
            df.loc[df['signal'] == -1, 'sl'] = df['high'].shift(1) + atr_for_sl_tp * 0.3
            df.loc[df['signal'] == -1, 'tp'] = df['close'] - (atr_for_sl_tp * 1.5)

            # Debug logging with NaN checks
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
    """
    Strategy: 3:00 PM Bollinger Volatility Breakout
    Concept: Capitalize on volatility breakouts from a Bollinger Band squeeze
             around 3:00 PM IST, confirmed by volume spike and neutral RSI.
    Signal: Triggers on price breaking above/below Bollinger Bands during
            2:50 PM–3:20 PM, with high volume and RSI between 40–60.
    Hero Zero Application: Buy OTM calls on buy signal, OTM puts on sell signal.
                          Risk is premium paid, targeting 2x–5x premium.
    """
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        df['sl'] = np.nan
        df['tp'] = np.nan
        df['tsl'] = np.nan
        try:
            # Parameters
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

            # Time window (2:50 PM to 3:20 PM IST)
            df['is_3pm'] = df.index.map(lambda ts: (ts.hour == start_hour and ts.minute >= start_minute) or
                                        (ts.hour == end_hour and ts.minute <= end_minute))

            # Bollinger Band squeeze
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(bb_period).mean() * squeeze_threshold

            # Volume spike
            df['avg_volume'] = df['volume'].rolling(window=5).mean()
            df['volume_spike'] = df['volume'] > volume_multiplier * df['avg_volume']

            # RSI condition
            df['rsi_neutral'] = (df['rsi_14'] >= rsi_lower) & (df['rsi_14'] <= rsi_upper)

            # Breakout conditions
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

            # Set signals
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1

            # Set SL, TP, TSL
            df.loc[df['buy'], 'sl'] = df['close'] - sl_atr_mult * df['atr_14']
            df.loc[df['buy'], 'tp'] = df['close'] + tp_atr_mult * df['atr_14']
            df.loc[df['buy'], 'tsl'] = df['close'] - sl_atr_mult * df['atr_14'] * 0.5
            df.loc[df['sell'], 'sl'] = df['close'] + sl_atr_mult * df['atr_14']
            df.loc[df['sell'], 'tp'] = df['close'] - tp_atr_mult * df['atr_14']
            df.loc[df['sell'], 'tsl'] = df['close'] + sl_atr_mult * df['atr_14'] * 0.5

            # Debug logging
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
            # Ensure base indicator columns are present (from your data prep)
            # ema_9, ema_21, adx_14 (or your chosen ADX period), volume, volume_avg, atr_14
            # Optional: ema_200 for regime filter
            required_cols = ['ema_9', 'ema_21', 'adx_14', 'volume', 'volume_avg', 'atr_14', 'close']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing column for EMACrossoverStrategy_EnhancedV1: {col}.")
                    return df # Exit if critical data is missing

            # --- Base Crossover Logic ---
            # (Assuming this part is now working and ema_9, ema_21 are valid)
            basic_buy_crossover = (df['ema_9'] > df['ema_21']) & (df['ema_9'].shift(1) <= df['ema_21'].shift(1))
            basic_sell_crossover = (df['ema_9'] < df['ema_21']) & (df['ema_9'].shift(1) >= df['ema_21'].shift(1))

            # --- Enhancements ---
            # 1. ADX Trend Strength Filter
            adx_threshold = self.params.get('adx_threshold', 20)
            is_trending = df['adx_14'] > adx_threshold

            # 2. Volume Confirmation
            volume_multiplier = self.params.get('volume_multiplier', 1.2)
            has_volume_confirmation = df['volume'] > (df['volume_avg'] * volume_multiplier)
            
            # 3. Regime Filter (Optional - using ema_200)
            use_regime_filter = self.params.get('use_regime_filter', False)
            if use_regime_filter:
                if 'ema_200' not in df.columns:
                    logger.warning("ema_200 not found for regime filter in EMACrossover_EnhancedV1. Skipping regime filter.")
                    is_bullish_regime = pd.Series([True] * len(df), index=df.index) # Default to allow all
                    is_bearish_regime = pd.Series([True] * len(df), index=df.index)
                else:
                    is_bullish_regime = df['close'] > df['ema_200']
                    is_bearish_regime = df['close'] < df['ema_200']
            else:
                is_bullish_regime = pd.Series([True] * len(df), index=df.index) # Allow if filter not used
                is_bearish_regime = pd.Series([True] * len(df), index=df.index)

            # --- Combined Conditions ---
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

            # SL/TP using ATR
            sl_atr_mult = self.params.get('sl_atr_mult', 1.5)
            tp_atr_mult = self.params.get('tp_atr_mult', 2.0)

            df['sl'] = np.where(df['signal'] == 1, df['close'] - (df['atr_14'] * sl_atr_mult),
                                np.where(df['signal'] == -1, df['close'] + (df['atr_14'] * sl_atr_mult), np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] + (df['atr_14'] * tp_atr_mult),
                                np.where(df['signal'] == -1, df['close'] - (df['atr_14'] * tp_atr_mult), np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan) # Your original TSL logic

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
            required_cols = ['close', 'open', 'volume', 'vwap', 'atr_14', 'rsi_14'] # Added rsi_14 for optional momentum
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing column for ThreePMBreakoutPowerBarStrategy_EnhancedV1: {col}.")
                    # df['volume_avg'] and df['body_avg'] will be calculated
                    return df

            # Time window (original: 14:50 onwards)
            df['is_3pm_window'] = df.index.map(
                lambda ts: (ts.hour == 14 and ts.minute >= self.params.get('start_minute', 50)) or \
                           (ts.hour == 15 and ts.minute <= self.params.get('end_minute', 15))
            )

            df['body'] = abs(df['close'] - df['open'])
            df['body_avg'] = df['body'].rolling(window=self.params.get('body_avg_period', 20), min_periods=5).mean()
            df['volume_avg'] = df['volume'].rolling(window=self.params.get('vol_avg_period', 20), min_periods=5).mean()

            # Original conditions
            is_strong_bull_base = (df['close'] > df['open']) & (df['body'] > df['body_avg'])
            is_strong_bear_base = (df['open'] > df['close']) & (df['body'] > df['body_avg'])
            has_volume_spike = df['volume'] > (df['volume_avg'] * self.params.get('vol_spike_mult', 1.5))

            # --- Enhancements ---
            # 1. ATR-Relative Power Bar
            atr_body_mult = self.params.get('atr_body_mult', 1.0) # Body must be at least 1x ATR
            is_atr_power_bar = df['body'] > (df['atr_14'] * atr_body_mult)

            # 2. VWAP Decisive Cross/Position
            # For buy: closed above VWAP, and Open was below or near VWAP (shows VWAP broken upwards)
            closed_above_vwap = df['close'] > df['vwap']
            # For sell: closed below VWAP, and Open was above or near VWAP
            closed_below_vwap = df['close'] < df['vwap']
            
            # Optional: Opened near/other side of VWAP for stronger signal
            opened_near_below_vwap = (df['open'] <= df['vwap']) # For buy
            opened_near_above_vwap = (df['open'] >= df['vwap']) # For sell
            
            vwap_buy_confirm = closed_above_vwap # & opened_near_below_vwap (can make it stricter)
            vwap_sell_confirm = closed_below_vwap # & opened_near_above_vwap (can make it stricter)

            # 3. Optional: RSI Momentum Confirmation (e.g. RSI moving up for buy)
            rsi_momentum_buy = df['rsi_14'] > self.params.get('rsi_buy_thresh', 55) # Example: RSI confirming upward push
            # rsi_momentum_buy = df['rsi_14'] > df['rsi_14'].shift(1) # Simpler: RSI increasing
            rsi_momentum_sell = df['rsi_14'] < self.params.get('rsi_sell_thresh', 45)
            # rsi_momentum_sell = df['rsi_14'] < df['rsi_14'].shift(1)

            use_rsi_confirm = self.params.get('use_rsi_confirm_3pm', False)


            # --- Combined Conditions ---
            df['buy'] = (
                df['is_3pm_window'] &
                is_strong_bull_base &
                has_volume_spike &
                is_atr_power_bar &      # Enhancement
                vwap_buy_confirm        # Enhancement (potentially stricter)
            )
            if use_rsi_confirm:
                 df['buy'] = df['buy'] & rsi_momentum_buy


            df['sell'] = (
                df['is_3pm_window'] &
                is_strong_bear_base &
                has_volume_spike &
                is_atr_power_bar &      # Enhancement
                vwap_sell_confirm       # Enhancement (potentially stricter)
            )
            if use_rsi_confirm:
                 df['sell'] = df['sell'] & rsi_momentum_sell

            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1

            # SL/TP can be tighter for 3PM scalps or based on the Power Bar's low/high
            sl_atr_mult_3pm = self.params.get('sl_atr_mult_3pm', 1.0)
            tp_atr_mult_3pm = self.params.get('tp_atr_mult_3pm', 1.5) # Quick targets

            df.loc[df['signal'] == 1, 'sl'] = df['open'] - (df['atr_14'] * 0.25) # Example: SL below open of power bar
            # df.loc[df['signal'] == 1, 'sl'] = df['low'] # Alternative: SL at low of power bar
            df.loc[df['signal'] == 1, 'tp'] = df['close'] + (df['atr_14'] * tp_atr_mult_3pm)

            df.loc[df['signal'] == -1, 'sl'] = df['open'] + (df['atr_14'] * 0.25) # Example: SL above open of power bar
            # df.loc[df['signal'] == -1, 'sl'] = df['high'] # Alternative: SL at high of power bar
            df.loc[df['signal'] == -1, 'tp'] = df['close'] - (df['atr_14'] * tp_atr_mult_3pm)
            
            # Using original TSL logic for consistency with your file
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)


            buy_signals = df['buy'].sum()
            sell_signals = df['sell'].sum()
            logger.debug(f"ThreePMBreakoutPowerBar_EnhancedV1: Buy: {buy_signals}, Sell: {sell_signals}")

        except KeyError as e:
            logger.error(f"KeyError in ThreePMBreakoutPowerBarStrategy_EnhancedV1: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = 0; df['sl'] = np.nan; df['tp'] = np.nan; df['tsl'] = np.nan
        except Exception as e:
            logger.error(f"General error in ThreePMBreakoutPowerBarStrategy_EnhancedV1: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = 0; df['sl'] = np.nan; df['tp'] = np.nan; df['tsl'] = np.nan
        return df
class MeanReversionSnapBackStrategy_EnhancedV1(BaseStrategy):
    # (This class would inherit from your BaseStrategy)
    def __init__(self, **kwargs): # Using **kwargs to match your BaseStrategy
        super().__init__(**kwargs) # Pass kwargs to parent
        # Default parameters specific to this strategy, can be overridden by self.params
        self.rsi_period = self.params.get('rsi_period', 7)
        self.rsi_ob = self.params.get('rsi_ob', 75)
        self.rsi_os = self.params.get('rsi_os', 25)
        self.ema_period = self.params.get('ema_period', 9)
        self.atr_period_for_vol = self.params.get('atr_period_for_vol', 14)
        self.max_atr_multiplier_for_entry = self.params.get('max_atr_multiplier_for_entry', 2.5) # e.g. ATR shouldn't be > 2.5x its own recent average
        self.reversal_vol_spike_mult = self.params.get('reversal_vol_spike_mult', 1.5)

    def _calculate_rsi(self, series, period): # Helper, assuming not globally available
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0) # Handle division by zero if loss is 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            required_cols = ['close', 'open', 'high', 'low', 'volume', 'atr_14'] # Assuming atr_14 is main ATR
            # 'volume_avg' will be calculated
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing column for MeanReversionSnapBackStrategy_EnhancedV1: {col}.")
                    return df
            
            df['volume_avg'] = df['volume'].rolling(window=20, min_periods=5).mean()

            # Time window (original: 2:55 PM - 3:15 PM)
            df['is_timeslot'] = df.index.to_series().apply(
                lambda ts: (ts.hour == 14 and ts.minute >= self.params.get('start_minute_mr', 55)) or \
                           (ts.hour == 15 and ts.minute <= self.params.get('end_minute_mr', 15))
            )

            df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
            df['ema_short'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
            
            # --- Enhancements ---
            # 1. Volatility Condition (using atr_14)
            # Optional: Only trade if ATR is not excessively high compared to its own average
            # df['avg_atr'] = df['atr_14'].rolling(window=self.atr_period_for_vol).mean()
            # is_not_extreme_volatility = df['atr_14'] < (df['avg_atr'] * self.max_atr_multiplier_for_entry)
            # For simplicity here, we'll focus on reversal volume spike
            has_reversal_volume_spike = df['volume'] > (df['volume_avg'] * self.reversal_vol_spike_mult)

            # 2. Confirmation Candle Proxy (significant close into prior bar's range)
            # For Buy (oversold): Current close is above previous bar's midpoint AND previous bar was bearish
            prev_bar_midpoint = (df['high'].shift(1) + df['low'].shift(1)) / 2
            prev_bar_was_bearish = df['close'].shift(1) < df['open'].shift(1)
            buy_reversal_candle_confirm = (df['close'] > prev_bar_midpoint) & prev_bar_was_bearish & (df['close'] > df['open']) # current bar is bullish

            # For Sell (overbought): Current close is below previous bar's midpoint AND previous bar was bullish
            prev_bar_was_bullish = df['close'].shift(1) > df['open'].shift(1)
            sell_reversal_candle_confirm = (df['close'] < prev_bar_midpoint) & prev_bar_was_bullish & (df['close'] < df['open']) # current bar is bearish

            # Original conditions
            was_oversold_shifted = (df['rsi'].shift(1) < self.rsi_os) | (df['rsi'].shift(2) < self.rsi_os) # Check previous 1 or 2 bars
            was_overbought_shifted = (df['rsi'].shift(1) > self.rsi_ob) | (df['rsi'].shift(2) > self.rsi_ob)

            # --- Combined Conditions ---
            df['buy'] = (
                df['is_timeslot'] &
                was_oversold_shifted &
                # is_not_extreme_volatility & # Optional volatility filter
                has_reversal_volume_spike & # Volume on reversal
                buy_reversal_candle_confirm # Candle confirmation
                # (df['close'] > df['ema_short']) # Original EMA condition, can be kept or made optional
            )

            df['sell'] = (
                df['is_timeslot'] &
                was_overbought_shifted &
                # is_not_extreme_volatility & # Optional
                has_reversal_volume_spike &
                sell_reversal_candle_confirm
                # (df['close'] < df['ema_short']) # Original EMA condition
            )

            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1

            # SL/TP for quick snap-back
            sl_atr_mult_mr = self.params.get('sl_atr_mult_mr', 1.0)
            tp_atr_mult_mr = self.params.get('tp_atr_mult_mr', 1.5)

            # SL just beyond the low/high of the signal bar (or previous bar if more extreme)
            df.loc[df['signal'] == 1, 'sl'] = df['low'] - (df['atr_14'] * 0.25) # Slightly below signal bar low
            df.loc[df['signal'] == 1, 'tp'] = df['close'] + (df['atr_14'] * tp_atr_mult_mr)

            df.loc[df['signal'] == -1, 'sl'] = df['high'] + (df['atr_14'] * 0.25) # Slightly above signal bar high
            df.loc[df['signal'] == -1, 'tp'] = df['close'] - (df['atr_14'] * tp_atr_mult_mr)
            
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
class EMACrossover_enhancedV1:
    def __init__(self, **params):
        self.params = params

    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['ema_fast'] = df['ema_9']
            df['ema_slow'] = df['ema_21']
            df['adx_ok'] = df['adx_14'] > 20
            df['macd_conf'] = df['macd_12_26_9'] > df['macds_12_26_9']

            df['buy'] = (
                (df['ema_fast'] > df['ema_slow']) &
                (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)) &
                df['adx_ok'] & df['macd_conf']
            )
            df['sell'] = (
                (df['ema_fast'] < df['ema_slow']) &
                (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1)) &
                df['adx_ok'] & (~df['macd_conf'])
            )
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = df['close'] * 0.995
            df['tp'] = df['close'] * 1.01
            df['tsl'] = df['close'] - df['atr_14']
        except Exception as e:
            logger.error(f"EMACrossover_enhancedV1 error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class ATRBreakout_enhancedV1:
    def __init__(self, **params):
        self.params = params

    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            atr_mult = self.params.get('atr_multiplier', 2.0)
            df['upper_band'] = df['close'].shift(1) + atr_mult * df['atr_14']
            df['lower_band'] = df['close'].shift(1) - atr_mult * df['atr_14']
            df['rsi_filter'] = (df['rsi_14'] > 45) & (df['rsi_14'] < 70)
            df['macd_filter'] = df['macd_12_26_9'] > df['macds_12_26_9']
            df['adx_strong'] = df['adx_14'] > 25

            df['buy'] = (df['close'] > df['upper_band']) & df['rsi_filter'] & df['macd_filter'] & df['adx_strong']
            df['sell'] = (df['close'] < df['lower_band']) & df['rsi_filter'] & (~df['macd_filter']) & df['adx_strong']
            df.loc[df['buy'], 'signal'] = 1
            df.loc[df['sell'], 'signal'] = -1
            df['sl'] = df['close'] * 0.995
            df['tp'] = df['close'] * 1.01
            df['tsl'] = df['close'] - df['atr_14']
        except Exception as e:
            logger.error(f"ATRBreakout_enhancedV1 error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class MomentumBreakout_enhancedV1:
    def __init__(self, **params):
        self.params = params

    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
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

class MeanReversion_enhancedV1:
    def __init__(self, **params):
        self.params = params

    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
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
# === NEW STRATEGIES === #

class IchimokuCloudStrategy_v1(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
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

class KeltnerChannelBreakoutStrategy_v1(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
            df['ema'] = df['close'].ewm(span=20).mean()
            df['upper_band'] = df['ema'] + 2 * df['atr_14']
            df['lower_band'] = df['ema'] - 2 * df['atr_14']

            buy = df['close'] > df['upper_band']
            sell = df['close'] < df['lower_band']

            df.loc[buy, 'signal'] = 1
            df.loc[sell, 'signal'] = -1

            df['sl'] = df['close'] * 0.995
            df['tp'] = df['close'] * 1.01
            df['tsl'] = df['close'] - df['atr_14']
        except Exception as e:
            logger.error(f"KeltnerChannelBreakoutStrategy error: {e}")
            df[['signal', 'sl', 'tp', 'tsl']] = np.nan
        return df

class DonchianBreakoutStrategy_v1(BaseStrategy):
    def generate_signals(self, data):
        df = data.copy()
        df['signal'] = 0
        try:
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
        
        # Check for duplicate indices
        if df.index.has_duplicates:
            duplicate_count = df.index.duplicated().sum()
            logger.warning(f"AlphaTrendStrategy received DF with {duplicate_count} duplicate timestamps")
            logger.warning(f"Sample duplicates: {df.index[df.index.duplicated(keep=False)].unique()[:10].tolist()}")
            df = df[~df.index.duplicated(keep='last')]
            logger.info(f"DataFrame shape after removing duplicates: {df.shape}")
        
        df['signal'] = 0
        required_columns = ['rsi_10', 'supertrend_10_3.0', 'close', 'atr_14']
        
        # Verify required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in AlphaTrendStrategy: {missing_cols}")
            df['signal'] = 0
            df['sl'] = np.nan
            df['tp'] = np.nan
            df['tsl'] = np.nan
            return df
        
        try:
            df['buy_cond'] = (
                (df['rsi_10'] > self.params.get('rsi_buy_lower', 35)) &
                (df['rsi_10'] < self.params.get('rsi_buy_upper', 65)) &
                (df['supertrend_10_3.0'] < df['close'])
            )
            df['sell_cond'] = (
                (df['rsi_10'] > self.params.get('rsi_sell_lower', 55)) &
                (df['rsi_10'] < self.params.get('rsi_sell_upper', 75)) &
                (df['supertrend_10_3.0'] > df['close'])
            )
            df.loc[df['buy_cond'], 'signal'] = 1
            df.loc[df['sell_cond'], 'signal'] = -1
            df['sl'] = np.where(df['signal'] == 1, df['close'] * 0.995,
                               np.where(df['signal'] == -1, df['close'] * 1.005, np.nan))
            df['tp'] = np.where(df['signal'] == 1, df['close'] * 1.01,
                               np.where(df['signal'] == -1, df['close'] * 0.99, np.nan))
            df['tsl'] = np.where(df['signal'] != 0, df['close'] - df['atr_14'], np.nan)
            
            # Debug logging
            buy_signals = df['buy_cond'].sum()
            sell_signals = df['sell_cond'].sum()
            logger.debug(f"AlphaTrend: Buy signals: {buy_signals}, Sell signals: {sell_signals}")
            if buy_signals == 0 and sell_signals == 0:
                logger.debug(f"AlphaTrend conditions: rsi_10={df['rsi_10'].iloc[-1]:.2f}, "
                            f"supertrend_10_3.0={df['supertrend_10_3.0'].iloc[-1]:.2f}, "
                            f"close={df['close'].iloc[-1]:.2f}, "
                            f"params={self.params}")
        except Exception as e:
            logger.error(f"Error in AlphaTrendStrategy: {e}")
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
    'AlphaTrend': make_strategy_func(AlphaTrendStrategy),
    'EMACrossover': make_strategy_func(EMACrossoverStrategy),
    'RSIMACD': make_strategy_func(RSIMACDStrategy),
    'SuperTrend_ADX': make_strategy_func(SuperTrendADXStrategy),
    'Bollinger_Bands': make_strategy_func(BollingerMeanReversionStrategy),
     
    'ATR_Breakout': make_strategy_func(ATRBreakoutStrategy),
    'RSIMACDSuperTrend': make_strategy_func(RSIMACDSuperTrendStrategy),
    'ADXMACDEMA': make_strategy_func(ADXMACDEMAStrategy),
    'RSIBollingerMACD': make_strategy_func(RSIBollingerMACDStrategy),

    'SuperMomentum': make_strategy_func(SuperMomentumStrategy),
    'Stochastic_Oscillator': make_strategy_func(StochasticOscillatorStrategy),
    
    'RSI': make_strategy_func(RSIStrategy),
    'MACD_Crossover': make_strategy_func(MACDCrossoverStrategy),
    'ATRSuperAlphaTrend': make_strategy_func(ATRSuperAlphaTrendStrategy),
    'RSIBollingerConfluence': make_strategy_func(RSIBollingerConfluenceStrategy),
    'MACDStochasticTrend': make_strategy_func(MACDStochasticTrendStrategy),
    'ADXVolatilityBreakout': make_strategy_func(ADXVolatilityBreakoutStrategy),
    'TripleEMAMomentum': make_strategy_func(TripleEMAMomentumStrategy),
    'VWAPMomentum': make_strategy_func(VWAPMomentumStrategy),
    'EMARollingADX': make_strategy_func(EMARollingADXStrategy),
    'BollingerStochX': make_strategy_func(BollingerStochasticCrossoverStrategy),
    'BollingerStochasticCrossoverStrategy': make_strategy_func(BollingerStochasticCrossoverStrategy),

    'VWAPRSIMACD3PMSpike': make_strategy_func(VWAPRSIMACD3PMSpikeStrategy),     
    'SupertrendReversal3PM': make_strategy_func(SupertrendReversal3PMStrategy),
    'KeltnerChannelBreakoutStrategy_v1': make_strategy_func(KeltnerChannelBreakoutStrategy_v1),
    'EMACrossover_enhancedV1': make_strategy_func(EMACrossover_enhancedV1),
    'ATRBreakout_enhancedV1': make_strategy_func(ATRBreakout_enhancedV1),
 


 #=========================
 'Support_Resistance': make_strategy_func(SupportResistanceStrategy),
     'Mean_Reversion': make_strategy_func(MeanReversionStrategy),
    'Momentum': make_strategy_func(MomentumStrategy),
     'BollingerSqueezeSpike': make_strategy_func(BollingerSqueezeSpikeStrategy),
    'ClosingBellBreakoutScalp': make_strategy_func(ClosingBellBreakoutScalpStrategy),
     'ExpiryDayVolatilitySpike': make_strategy_func(ExpiryDayVolatilitySpikeStrategy),
    'MeanReversionSnapBack': make_strategy_func(MeanReversionSnapBackStrategy),
     'ExpiryDayOTMScalp': make_strategy_func(ExpiryDayOTMScalpStrategy),
     'MomentumBreakoutRSI': make_strategy_func(MomentumBreakoutRSIStrategy),
     'VWAPReversalScalp': make_strategy_func(VWAPReversalScalpStrategy),
     'StraddleScalpHighVol': make_strategy_func(StraddleScalpHighVolStrategy),
     'ThreePMBollingerVolBreakout': make_strategy_func(ThreePMBollingerVolBreakoutStrategy), 
     'AlphaTrendStrategy_EnhancedV1': make_strategy_func(AlphaTrendStrategy_EnhancedV1),
    'EMACrossoverStrategy_EnhancedV1': make_strategy_func(EMACrossoverStrategy_EnhancedV1),
     'ThreePMBreakoutPowerBarStrategy_EnhancedV1': make_strategy_func(ThreePMBreakoutPowerBarStrategy_EnhancedV1),
     'MeanReversionSnapBackStrategy_EnhancedV1': make_strategy_func(MeanReversionSnapBackStrategy_EnhancedV1),
   'Momentum_Breakout': make_strategy_func(MomentumBreakoutStrategy),
     'MomentumBreakout_enhancedV1': make_strategy_func(MomentumBreakout_enhancedV1),
     'MeanReversion_enhancedV1': make_strategy_func(MeanReversion_enhancedV1),
     'IchimokuCloudStrategy_v1': make_strategy_func(IchimokuCloudStrategy_v1),
        'DonchianBreakoutStrategy_v1': make_strategy_func(DonchianBreakoutStrategy_v1),
             'RSIATRReversal': make_strategy_func(RSIATRReversalStrategy),
     'MACDTrendVolume': make_strategy_func(MACDTrendVolumeStrategy),
     'ORBVolumeMACD': make_strategy_func(ORBVolumeMACDStrategy),
    'ThreePMBreakoutPowerBar': make_strategy_func(ThreePMBreakoutPowerBarStrategy),
         'Breakout_ATR': make_strategy_func(BreakoutATRStrategy),
   

    

}