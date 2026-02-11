"""
Strategy Utilities - Common Helper Functions

Provides utility functions for strategy development:
- Indicator calculations
- Crossover detection
- Price action patterns
- Time-based filters
- Common validation logic

These utilities help avoid code duplication across strategies.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import time as dt_time, datetime, timedelta
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# INDICATOR CALCULATIONS
# ============================================================================

def calculate_ema(data: pd.Series, period: int) -> Optional[pd.Series]:
    """
    Calculate EMA safely.
    
    Args:
        data: Price series (usually 'close')
        period: EMA period
    
    Returns:
        EMA series or None if calculation fails
    """
    try:
        return ta.ema(data, length=period)
    except Exception as e:
        logger.error(f"EMA calculation failed: {e}")
        return None


def calculate_rsi(data: pd.Series, period: int = 14) -> Optional[pd.Series]:
    """
    Calculate RSI safely.
    
    Args:
        data: Price series (usually 'close')
        period: RSI period (default: 14)
    
    Returns:
        RSI series or None if calculation fails
    """
    try:
        return ta.rsi(data, length=period)
    except Exception as e:
        logger.error(f"RSI calculation failed: {e}")
        return None


def calculate_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0
) -> Optional[pd.DataFrame]:
    """
    Calculate SuperTrend indicator.
    
    Args:
        high, low, close: OHLC series
        period: ATR period
        multiplier: ATR multiplier
    
    Returns:
        DataFrame with SuperTrend columns or None
    """
    try:
        return ta.supertrend(high, low, close, length=period, multiplier=multiplier)
    except Exception as e:
        logger.error(f"SuperTrend calculation failed: {e}")
        return None


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> Optional[pd.DataFrame]:
    """
    Calculate ADX (Average Directional Index).
    
    Args:
        high, low, close: OHLC series
        period: ADX period
    
    Returns:
        DataFrame with ADX columns or None
    """
    try:
        return ta.adx(high, low, close, length=period)
    except Exception as e:
        logger.error(f"ADX calculation failed: {e}")
        return None


def calculate_bollinger_bands(
    data: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Optional[pd.DataFrame]:
    """
    Calculate Bollinger Bands.
    
    Args:
        data: Price series (usually 'close')
        period: Moving average period
        std_dev: Standard deviation multiplier
    
    Returns:
        DataFrame with BBL, BBM, BBU, BBB columns or None
    """
    try:
        return ta.bbands(data, length=period, std=std_dev)
    except Exception as e:
        logger.error(f"Bollinger Bands calculation failed: {e}")
        return None


# ============================================================================
# CROSSOVER DETECTION
# ============================================================================

def detect_crossover(
    fast_series: pd.Series,
    slow_series: pd.Series,
    current_idx: int
) -> Optional[str]:
    """
    Detect crossover between two series.
    
    Args:
        fast_series: Faster moving series (e.g., short EMA)
        slow_series: Slower moving series (e.g., long EMA)
        current_idx: Index position to check
    
    Returns:
        'bullish_cross': fast crossed above slow
        'bearish_cross': fast crossed below slow
        None: no crossover
    """
    if current_idx < 1:
        return None
    
    try:
        current_fast = fast_series.iloc[current_idx]
        current_slow = slow_series.iloc[current_idx]
        prev_fast = fast_series.iloc[current_idx - 1]
        prev_slow = slow_series.iloc[current_idx - 1]
        
        if pd.isna(current_fast) or pd.isna(current_slow):
            return None
        if pd.isna(prev_fast) or pd.isna(prev_slow):
            return None
        
        # Bullish crossover
        if prev_fast <= prev_slow and current_fast > current_slow:
            return 'bullish_cross'
        
        # Bearish crossover
        if prev_fast >= prev_slow and current_fast < current_slow:
            return 'bearish_cross'
        
        return None
        
    except Exception as e:
        logger.error(f"Crossover detection failed: {e}")
        return None


def is_above(series1: pd.Series, series2: pd.Series, idx: int) -> bool:
    """Check if series1 > series2 at index."""
    try:
        return series1.iloc[idx] > series2.iloc[idx]
    except:
        return False


def is_below(series1: pd.Series, series2: pd.Series, idx: int) -> bool:
    """Check if series1 < series2 at index."""
    try:
        return series1.iloc[idx] < series2.iloc[idx]
    except:
        return False


# ============================================================================
# PRICE ACTION PATTERNS
# ============================================================================

def is_bullish_candle(row: pd.Series) -> bool:
    """Check if candle is bullish (close > open)."""
    return row['close'] > row['open']


def is_bearish_candle(row: pd.Series) -> bool:
    """Check if candle is bearish (close < open)."""
    return row['close'] < row['open']


def candle_body_size(row: pd.Series) -> float:
    """Get candle body size (absolute difference between open and close)."""
    return abs(row['close'] - row['open'])


def candle_range(row: pd.Series) -> float:
    """Get candle total range (high - low)."""
    return row['high'] - row['low']


def is_doji(row: pd.Series, threshold: float = 0.1) -> bool:
    """
    Check if candle is a doji (small body relative to range).
    
    Args:
        row: OHLC bar
        threshold: Max body/range ratio to be considered doji
    """
    range_val = candle_range(row)
    if range_val == 0:
        return True
    body = candle_body_size(row)
    return (body / range_val) < threshold


def touches_level(price: float, level: float, tolerance_pct: float = 0.001) -> bool:
    """
    Check if price touches a level within tolerance.
    
    Args:
        price: Current price
        level: Target level
        tolerance_pct: Tolerance as percentage (default 0.1%)
    """
    tolerance = level * tolerance_pct
    return abs(price - level) <= tolerance


# ============================================================================
# SLOPE CALCULATIONS
# ============================================================================

def calculate_slope(series: pd.Series, lookback: int) -> Optional[float]:
    """
    Calculate slope of a series over lookback periods.
    
    Args:
        series: Data series (e.g., RSI values)
        lookback: Number of periods to calculate slope over
    
    Returns:
        Slope value (rise per period) or None
    """
    if len(series) < lookback:
        return None
    
    try:
        start_val = series.iloc[-lookback]
        end_val = series.iloc[-1]
        
        if pd.isna(start_val) or pd.isna(end_val):
            return None
        
        # Slope = (end - start) / (periods - 1)
        if lookback > 1:
            return (end_val - start_val) / (lookback - 1)
        else:
            return end_val - start_val
            
    except Exception as e:
        logger.error(f"Slope calculation failed: {e}")
        return None


def is_rising(series: pd.Series, lookback: int = 3, threshold: float = 0) -> bool:
    """Check if series is rising (positive slope above threshold)."""
    slope = calculate_slope(series, lookback)
    return slope is not None and slope > threshold


def is_falling(series: pd.Series, lookback: int = 3, threshold: float = 0) -> bool:
    """Check if series is falling (negative slope below threshold)."""
    slope = calculate_slope(series, lookback)
    return slope is not None and slope < -threshold


# ============================================================================
# TIME-BASED FILTERS
# ============================================================================

def is_in_time_range(
    timestamp: datetime,
    start_time: dt_time,
    end_time: dt_time
) -> bool:
    """
    Check if timestamp falls within time range.
    
    Args:
        timestamp: Timestamp to check
        start_time: Range start (datetime.time object)
        end_time: Range end (datetime.time object)
    
    Returns:
        True if in range
    """
    current_time = timestamp.time()
    
    if start_time <= end_time:
        # Normal range (e.g., 9:00 to 15:00)
        return start_time <= current_time <= end_time
    else:
        # Overnight range (e.g., 22:00 to 06:00)
        return current_time >= start_time or current_time <= end_time


def get_opening_range(
    data: pd.DataFrame,
    current_timestamp: datetime,
    orb_start_hour: int = 9,
    orb_start_minute: int = 15,
    orb_duration_minutes: int = 15
) -> Optional[Tuple[float, float]]:
    """
    Calculate opening range high and low.
    
    Args:
        data: Historical OHLC data with DatetimeIndex
        current_timestamp: Current bar timestamp
        orb_start_hour: ORB start hour
        orb_start_minute: ORB start minute
        orb_duration_minutes: ORB duration
    
    Returns:
        (orb_high, orb_low) tuple or None
    """
    try:
        # Define ORB window
        orb_start = current_timestamp.replace(
            hour=orb_start_hour,
            minute=orb_start_minute,
            second=0,
            microsecond=0
        )
        orb_end = orb_start + timedelta(minutes=orb_duration_minutes)
        
        # Filter data for ORB window (same day only)
        orb_mask = (
            (data.index >= orb_start) &
            (data.index <= orb_end) &
            (data.index.date == current_timestamp.date())
        )
        
        orb_data = data.loc[orb_mask]
        
        if orb_data.empty:
            return None
        
        orb_high = orb_data['high'].max()
        orb_low = orb_data['low'].min()
        
        if pd.isna(orb_high) or pd.isna(orb_low):
            return None
        
        return (orb_high, orb_low)
        
    except Exception as e:
        logger.error(f"Opening range calculation failed: {e}")
        return None


def minutes_since_session_start(
    timestamp: datetime,
    session_start_hour: int = 9,
    session_start_minute: int = 15
) -> int:
    """
    Calculate minutes elapsed since session start.
    
    Args:
        timestamp: Current timestamp
        session_start_hour: Session start hour
        session_start_minute: Session start minute
    
    Returns:
        Minutes elapsed
    """
    session_start = timestamp.replace(
        hour=session_start_hour,
        minute=session_start_minute,
        second=0,
        microsecond=0
    )
    
    if timestamp < session_start:
        return 0
    
    delta = timestamp - session_start
    return int(delta.total_seconds() / 60)


# ============================================================================
# SQUEEZE DETECTION
# ============================================================================

def detect_volatility_squeeze(
    bandwidth_series: pd.Series,
    lookback: int = 20,
    tolerance: float = 1.05
) -> bool:
    """
    Detect Bollinger Band squeeze condition.
    
    A squeeze occurs when current bandwidth is near the lowest
    in the lookback period (indicating contraction before expansion).
    
    Args:
        bandwidth_series: Bollinger Band bandwidth series
        lookback: Lookback period
        tolerance: Tolerance multiplier (e.g., 1.05 = within 5% of lowest)
    
    Returns:
        True if in squeeze
    """
    if len(bandwidth_series) < lookback:
        return False
    
    try:
        recent_bandwidth = bandwidth_series.tail(lookback)
        lowest_bandwidth = recent_bandwidth.min()
        current_bandwidth = bandwidth_series.iloc[-1]
        
        if pd.isna(lowest_bandwidth) or pd.isna(current_bandwidth):
            return False
        
        return current_bandwidth <= (lowest_bandwidth * tolerance)
        
    except Exception as e:
        logger.error(f"Squeeze detection failed: {e}")
        return False


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_indicator_exists(
    row: pd.Series,
    indicator_name: str,
    default_value: Optional[float] = None
) -> Tuple[bool, float]:
    """
    Validate that indicator exists and has valid value.
    
    Args:
        row: Current bar
        indicator_name: Indicator to check
        default_value: Value to return if not found
    
    Returns:
        (is_valid: bool, value: float)
    """
    if indicator_name not in row.index:
        return (False, default_value or 0.0)
    
    value = row[indicator_name]
    
    if pd.isna(value):
        return (False, default_value or 0.0)
    
    return (True, value)


def get_safe_indicator(
    row: pd.Series,
    indicator_name: str,
    default: float = 0.0
) -> float:
    """
    Safely get indicator value with fallback.
    
    Args:
        row: Current bar
        indicator_name: Indicator name
        default: Default value if not found/invalid
    
    Returns:
        Indicator value or default
    """
    _, value = validate_indicator_exists(row, indicator_name, default)
    return value


def ensure_min_history(
    data: pd.DataFrame,
    min_bars: int
) -> bool:
    """
    Check if sufficient historical data available.
    
    Args:
        data: Historical data
        min_bars: Minimum bars required
    
    Returns:
        True if sufficient data
    """
    return len(data) >= min_bars
