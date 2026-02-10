#!/usr/bin/env python3
"""
Quick Start: Migrate Next Strategy (SuperTrend_ADX)

This script provides a template for migrating SuperTrend_ADX
to the new framework. Copy and modify as needed.

INSTRUCTIONS:
1. Copy this file to: strategies/supertrend_adx.py
2. Replace TODO sections with actual logic
3. Test with: python test_strategy_framework.py
4. Verify registry discovers the new strategy
"""

import pandas as pd
from typing import List
import logging

from strategies.base_strategy import (
    BaseStrategy,
    Signal,
    StrategyContext,
    MarketRegime,
    VolatilityBucket
)
from strategies.strategy_utils import (
    calculate_supertrend,
    calculate_adx,
    ensure_min_history,
    get_safe_indicator
)

logger = logging.getLogger(__name__)


class SuperTrendADX(BaseStrategy):
    """
    SuperTrend + ADX Filter Strategy
    
    Trend-following strategy using SuperTrend direction confirmed by ADX.
    
    Entry Logic:
    - BUY: SuperTrend direction = 1 (bullish) AND ADX >= threshold
    - SELL: SuperTrend direction = -1 (bearish) AND ADX >= threshold
    
    Exit Logic: Controlled by risk module (SL/TP/TSL)
    
    Optimal Conditions:
    - Trending markets (ADX > 25 required)
    - Medium to high volatility
    - All timeframes
    
    Parameters:
    - st_period: SuperTrend ATR period (default: 10)
    - st_multiplier: SuperTrend multiplier (default: 3.0)
    - adx_period: ADX calculation period (default: 14)
    - adx_threshold: Minimum ADX for signals (default: 25)
    
    Historical Performance (NIFTY 5min, Jan-May 2025):
    - 1,454 trades
    - 84.53% win rate ⭐ (2nd best)
    - ₹48,747 P&L ⭐ (2nd best)
    
    MIGRATED FROM: Common/strategies/strategies.py
    ORIGINAL: create_supertrend_adx_strategy()
    STATUS: ✅ READY TO MIGRATE
    """
    
    def __init__(
        self,
        st_period: int = 10,
        st_multiplier: float = 3.0,
        adx_period: int = 14,
        adx_threshold: int = 25,
        name: str = "SuperTrend_ADX",
        version: str = "2.0"
    ):
        """Initialize SuperTrend + ADX strategy."""
        super().__init__(name=name, version=version)
        
        self.st_period = st_period
        self.st_multiplier = st_multiplier
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.required_lookback = max(st_period, adx_period) + 50  # ADX needs stability
        
        # Update name with parameters
        self.name = f"{name}_{st_period}_{st_multiplier:.1f}_{adx_period}_{adx_threshold}"
        
        self.logger.info(
            f"Initialized {self.name} with ST({st_period},{st_multiplier}), "
            f"ADX({adx_period},{adx_threshold})"
        )
    
    def required_indicators(self) -> List[str]:
        """
        Required indicators: Will calculate dynamically.
        Need high, low, close for SuperTrend and ADX.
        """
        return ['high', 'low', 'close']
    
    def supports_market(self, market_type: str) -> bool:
        """Supports all markets."""
        return True
    
    def supports_regime(self, regime: MarketRegime) -> bool:
        """
        ONLY works in TRENDING markets.
        ADX threshold filter ensures trend strength.
        """
        return regime == MarketRegime.TRENDING
    
    def supports_volatility(self, volatility: VolatilityBucket) -> bool:
        """Works in medium to high volatility."""
        return volatility in [VolatilityBucket.MEDIUM, VolatilityBucket.HIGH]
    
    def generate_signal(self, context: StrategyContext) -> Signal:
        """
        Generate trading signal based on SuperTrend + ADX.
        
        TODO: Implement the following logic from Common/strategies/strategies.py:
        
        1. Check sufficient history (required_lookback bars)
        
        2. Calculate SuperTrend:
           supertrend_df = ta.supertrend(
               history['high'], history['low'], history['close'],
               length=st_period, multiplier=st_multiplier
           )
        
        3. Calculate ADX:
           adx_df = ta.adx(
               history['high'], history['low'], history['close'],
               length=adx_period
           )
        
        4. Get current values:
           - SuperTrend direction column: f'SUPERTd_{st_period}_{st_multiplier}'
           - ADX value column: f'ADX_{adx_period}'
        
        5. Check signal conditions:
           if st_direction == 1 and adx >= adx_threshold:
               return Signal(direction="BUY", ...)
           elif st_direction == -1 and adx >= adx_threshold:
               return Signal(direction="SELL", ...)
        
        6. Scale confidence by ADX strength:
           confidence = min(0.5 + ((adx - threshold) / 100), 1.0)
        
        HINT: Use calculate_supertrend() and calculate_adx() from strategy_utils
        HINT: Follow exact pattern from ema_crossover.py
        """
        
        # TODO: Implement logic here
        
        return Signal(
            direction="HOLD",
            confidence=0.0,
            strategy_name=self.name,
            reasoning="TODO: Implement SuperTrend + ADX logic"
        )
    
    def __str__(self) -> str:
        return (
            f"{self.name} (ST: {self.st_period}/{self.st_multiplier}, "
            f"ADX: {self.adx_period}/{self.adx_threshold})"
        )


# Backward compatibility factory
def create_supertrend_adx_strategy(
    st_period: int = 10,
    st_multiplier: float = 3.0,
    adx_period: int = 14,
    adx_threshold: int = 25
) -> SuperTrendADX:
    """Factory for backward compatibility."""
    return SuperTrendADX(
        st_period=st_period,
        st_multiplier=st_multiplier,
        adx_period=adx_period,
        adx_threshold=adx_threshold
    )


# ============================================================================
# REFERENCE: Original Logic from Common/strategies/strategies.py
# ============================================================================
"""
ORIGINAL PRODUCTION CODE (PRESERVE THIS EXACTLY):

def create_supertrend_adx_strategy(st_period=10, st_multiplier=3.0, adx_period=14, adx_threshold=25):
    required_lookback = max(st_period, adx_period) + 50

    def strategy(row, data_history=None):
        signal = 'hold'
        
        if data_history is None or len(data_history) < required_lookback:
            return signal

        try:
            # Get history slice
            current_data_idx = data_history.index.get_loc(row.name)
            start_slice_idx = max(0, current_data_idx - required_lookback + 1)
            history_slice = data_history.iloc[start_slice_idx : current_data_idx + 1]

            if len(history_slice) < max(st_period, adx_period) + 1:
                return signal

            # Calculate indicators
            supertrend_df = ta.supertrend(
                history_slice['high'], 
                history_slice['low'], 
                history_slice['close'], 
                length=st_period, 
                multiplier=st_multiplier
            )
            adx_df = ta.adx(
                history_slice['high'], 
                history_slice['low'], 
                history_slice['close'], 
                length=adx_period
            )

            if supertrend_df is None or adx_df is None:
                return signal

            # Get column names
            st_direction_col = f'SUPERTd_{st_period}_{st_multiplier}'
            adx_val_col = f'ADX_{adx_period}'

            if st_direction_col not in supertrend_df.columns:
                return signal
            if adx_val_col not in adx_df.columns:
                return signal

            # Get current values
            current_st_direction = supertrend_df[st_direction_col].iloc[-1]
            current_adx = adx_df[adx_val_col].iloc[-1]

            # Generate signals
            if pd.notna(current_st_direction) and pd.notna(current_adx):
                if current_adx >= adx_threshold:
                    if current_st_direction == 1:
                        signal = 'buy_potential'
                    elif current_st_direction == -1:
                        signal = 'sell_potential'

        except Exception as e:
            logger.error(f"Error: {e}")
            signal = 'hold'
            
        return signal
    
    return strategy
"""
