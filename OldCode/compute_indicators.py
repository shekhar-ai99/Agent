
import pandas as pd
import numpy as np
import logging
import traceback
from datetime import time

try:
    import pandas_ta as ta
    talib = ta
except ImportError:
    talib = None

logger = logging.getLogger(__name__)

def compute_indicators(df, output_path):
    df = df.copy()
    
    if df.index.has_duplicates:
        duplicate_count = df.index.duplicated().sum()
        logger.error(f"Input DataFrame has {duplicate_count} duplicate timestamps!")
        logger.error(f"Sample duplicated timestamps: {df.index[df.index.duplicated(keep=False)].unique()[:10].tolist()}")
        df = df[~df.index.duplicated(keep='first')]
        logger.info(f"DataFrame shape after removing input duplicates: {df.shape}")
    
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        logger.error(f"Missing required columns: {required}")
        return df

    if not df.index.tz:
        logger.warning("DataFrame index is not timezone-aware. Localizing to IST.")
        df.index = df.index.tz_localize('Asia/Kolkata')

    # df['is_timeslot'] = df.index.to_series().apply(
    #     lambda ts: (ts.time() >= time(9, 15) and ts.time() <= time(15, 30))
    # )
    def is_timeslot(ts):
        if pd.isna(ts):
            return False
        try:
            t = ts.time()
            return (t >= time(9, 15)) and (t <= time(15, 30))
        except Exception:
            return False

    df['is_timeslot'] = df.index.to_series().apply(is_timeslot)

    timeslot_count = df['is_timeslot'].sum()
    logger.info(f"Data has {timeslot_count} rows in 9:15 AM-3:30 PM IST window")
    df = df.drop(columns=['is_timeslot'])

    logger.info(f"Volume stats: min={df['volume'].min()}, max={df['volume'].max()}, mean={df['volume'].mean():.2f}")
    zero_volume_count = (df['volume'] == 0).sum()
    if zero_volume_count > 0:
        logger.warning(f"Found {zero_volume_count} rows with zero volume. Replacing with 1.")
        df['volume'] = df['volume'].replace(0, 1)

    df['ema_3'] = df['close'].ewm(span=3, adjust=False).mean()
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_13'] = df['close'].ewm(span=13, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_34'] = df['close'].ewm(span=34, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

    df['sma_20'] = df['close'].rolling(window=20).mean()

    if talib:
        df['rsi_10'] = talib.rsi(df['close'], length=10)
        df['rsi_14'] = talib.rsi(df['close'], length=14)
    else:
        def compute_rsi(series, period):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        df['rsi_10'] = compute_rsi(df['close'], 10)
        df['rsi_14'] = compute_rsi(df['close'], 14)

    if talib:
        macd = talib.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd_12_26_9'] = macd.iloc[:, 0]
        df['macds_12_26_9'] = macd.iloc[:, 1]
    else:
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_12_26_9'] = ema_12 - ema_26
        df['macds_12_26_9'] = df['macd_12_26_9'].ewm(span=9, adjust=False).mean()

    if talib:
        bb = talib.bbands(df['close'], length=20, std=2)
        df['bb_upper'] = bb.iloc[:, 0]
        df['bb_middle'] = bb.iloc[:, 1]
        df['bb_lower'] = bb.iloc[:, 2]
    else:
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * std
        df['bb_lower'] = df['bb_middle'] - 2 * std

    df['bb_width'] = df['bb_upper'] - df['bb_lower']

    if talib:
        df['atr_14'] = talib.atr(df['high'], df['low'], df['close'], length=14)
    else:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr_14'] = tr.rolling(window=14).mean()

    if talib:
        stoch = talib.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        df['stoch_k_14'] = stoch.iloc[:, 0]
        df['stoch_d_14'] = stoch.iloc[:, 1]
    else:
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k_14'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d_14'] = df['stoch_k_14'].rolling(window=3).mean()

    df['stoch_k'] = df['stoch_k_14']
    df['stoch_d'] = df['stoch_d_14']

    if talib:
        adx = talib.adx(df['high'], df['low'], df['close'], length=14)
        df['adx_14'] = adx['ADX_14']
    else:
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        tr = df['atr_14']
        plus_di = 100 * (plus_dm.rolling(14).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / tr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx_14'] = dx.rolling(14).mean()

    if talib:
        st = talib.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
        df['supertrend_10_3.0'] = st.iloc[:, 0]
        st2 = talib.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=2.0)
        df['supertrend_10_2.0'] = st2.iloc[:, 0]
    else:
        def compute_supertrend(df, period=10, multiplier=3.0):
            hl2 = (df['high'] + df['low']) / 2
            atr = df['atr_14']
            upperband = hl2 + (multiplier * atr)
            lowerband = hl2 - (multiplier * atr)
            supertrend = np.full(len(df), np.nan)
            for i in range(1, len(df)):
                if np.isnan(supertrend[i - 1]):
                    supertrend[i] = upperband.iloc[i]
                elif df['close'].iloc[i - 1] > supertrend[i - 1]:
                    supertrend[i] = lowerband.iloc[i]
                else:
                    supertrend[i] = upperband.iloc[i]
            return pd.Series(supertrend, index=df.index)
        df['supertrend_10_3.0'] = compute_supertrend(df, period=10, multiplier=3.0)
        df['supertrend_10_2.0'] = compute_supertrend(df, period=10, multiplier=2.0)

    df['roc_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("DataFrame index is not DatetimeIndex. Cannot calculate VWAP.")
        df['vwap'] = df['sma_20']
    else:
        try:
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['typical_price_vol'] = df['typical_price'] * df['volume']
            # Rolling VWAP (20 periods) for robustness
            df['cum_vol'] = df['volume'].rolling(window=20, min_periods=5).sum()
            df['cum_typical_price_vol'] = df['typical_price_vol'].rolling(window=20, min_periods=5).sum()
            df['vwap'] = df['cum_typical_price_vol'] / df['cum_vol'].replace(0, np.nan)
            #df['vwap'] = df['vwap'].fillna(method='ffill').fillna(method='bfill')
            df['vwap'] = df['vwap'].ffill().bfill()

            df['vwap'] = df['vwap'].fillna(df['sma_20'])
            vwap_nan_count = df['vwap'].isna().sum()
            if vwap_nan_count > 0:
                logger.warning(f"VWAP has {vwap_nan_count} NaN values after calculation.")
            else:
                logger.info("VWAP calculated successfully with no NaNs.")
            df = df.drop(columns=['typical_price', 'typical_price_vol', 'cum_vol', 'cum_typical_price_vol'])
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            df['vwap'] = df['sma_20']

    df['is_expiry_day'] = df.index.weekday == 3
    expiry_count = df['is_expiry_day'].sum()
    logger.info(f"Data has {expiry_count} expiry day rows (Thursdays)")

    df['volume_avg'] = df['volume'].rolling(window=20).mean()

    df['body'] = abs(df['close'] - df['open'])
    df['hl_range'] = df['high'] - df['low']

    df['boll_dist'] = (df['close'] - df['sma_20']) / df['sma_20']

    df['recent_range'] = (df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()) / df['close']
    df['volatility_20'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    df['rsi'] = df['rsi_14']
    df['atr'] = df['atr_14']
    df['avg_volume'] = df['volume_avg']
    df['bollinger_hband'] = df['bb_upper']
    df['bollinger_lband'] = df['bb_lower']

    logger.info(f"Rows before dropna: {len(df)}")
    nan_summary = df.isna().sum()
    nan_summary = nan_summary[nan_summary > 0]
    if not nan_summary.empty:
        #logger.warning(f"NaNs found in columns:\n{nan_summary}")
        logger.warning(f"NaNs found in columns")
    else:
        logger.info("No NaNs found before dropna.")

    critical_cols = ['close', 'ema_5', 'ema_13', 'rsi_14', 'macd_12_26_9', 'supertrend_10_2.0', 'atr_14', 'vwap']
    try:
        df = df.dropna(subset=critical_cols)
        logger.info(f"Rows after dropna on critical indicators: {len(df)}")
    except Exception as e:
        logger.error(f"Error during dropna: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

    try:
        df.to_csv(output_path, index=True)
        logger.info(f"Saved {len(df)} rows with indicators to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save DataFrame to {output_path}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

    logger.info(f"Computed indicators: {list(df.columns)}")
    return df
def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    if not all(col in data.columns for col in ['high', 'low', 'close']):
        return pd.Series(np.nan, index=data.index)
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr