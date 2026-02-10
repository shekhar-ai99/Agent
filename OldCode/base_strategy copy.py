"""
Base Strategy Abstract Class - Unified Framework

This base class consolidates best practices from:
1. TradingPlatform/core/base_strategy.py (clean OOP design)
2. Common/strategies/strategies.py (proven production logic)
3. IndianMarket backtest requirements (Indian market compatibility)

All strategies inherit from this. Strategies are market-agnostic and contain ONLY signal logic.
Risk parameters (SL, TP, TSL) are injected by the risk management layer.

VERSION: 2.0 (Unified)
CREATED: 2026-02-06
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS - Market Context Classification
# ============================================================================

class MarketRegime(Enum):
    """Market conditions that strategies respond to"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CHOPPY = "choppy"  # Added from Indian market backtest
    UNKNOWN = "unknown"


class VolatilityBucket(Enum):
    """Volatility classification for context-aware adjustments"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SessionType(Enum):
    """Trading session classification (Indian market specific but extensible)"""
    PRE_MARKET = "pre_market"
    SESSION_1 = "session_1"  # 9:15-11:30
    SESSION_2 = "session_2"  # 11:30-13:45
    SESSION_3 = "session_3"  # 13:45-15:30
    POST_MARKET = "post_market"
    OVERNIGHT = "overnight"  # For crypto/24hr markets


# ============================================================================
# DATA STRUCTURES - Signal & Context
# ============================================================================

@dataclass
class Signal:
    """
    Output from strategy.generate_signal()
    
    This is the ONLY thing a strategy returns. No SL/TP/TSL logic here.
    """
    direction: str  # "BUY", "SELL", "HOLD", "EXIT" (new: explicit exit signal)
    confidence: float  # 0.0 to 1.0 (0.0 = weak, 1.0 = strong conviction)
    strategy_name: str
    reasoning: str  # Human-readable explanation
    
    # Optional context for advanced use
    additional_context: Optional[Dict[str, Any]] = None
    
    # Optional: Strategy-suggested risk params (risk module can override)
    suggested_sl_mult: Optional[float] = None
    suggested_tp_mult: Optional[float] = None
    suggested_tsl_mult: Optional[float] = None
    
    def is_valid(self) -> bool:
        """Validate signal integrity"""
        return (
            self.direction in ["BUY", "SELL", "HOLD", "EXIT"]
            and 0.0 <= self.confidence <= 1.0
            and self.strategy_name
            and self.reasoning
        )
    
    def to_legacy_format(self) -> str:
        """
        Convert to legacy string format for backward compatibility
        Returns: 'buy_potential', 'sell_potential', 'hold', 'none'
        """
        mapping = {
            "BUY": "buy_potential",
            "SELL": "sell_potential",
            "HOLD": "hold",
            "EXIT": "none"
        }
        return mapping.get(self.direction, "hold")


@dataclass
class StrategyContext:
    """
    Input context for strategy decision-making.
    
    Contains everything a strategy needs to make intelligent decisions.
    Strategies should NOT access anything outside this context.
    """
    # Core market data
    symbol: str
    market_type: str  # "indian", "crypto", "us_equity", etc.
    timeframe: str  # "5min", "15min", "1h", etc.
    current_bar: pd.Series  # Latest candle OHLCV + indicators
    historical_data: pd.DataFrame  # Last N candles for lookback
    
    # Market conditions
    regime: MarketRegime
    volatility: VolatilityBucket
    
    # Time context
    session: SessionType
    timestamp: pd.Timestamp
    is_expiry_day: bool = False
    
    # Optional: Market-specific data
    additional_info: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def get_indicator(self, name: str, default: Any = None) -> Any:
        """
        Safely get indicator value from current bar.
        
        Args:
            name: Indicator name (e.g., 'rsi_14', 'atr_14', 'ema_9')
            default: Value to return if indicator not found
        """
        return self.current_bar.get(name, default)
    
    def get_indicator_history(self, indicator_name: str, lookback: int = 5) -> Optional[List[float]]:
        """
        Get historical values of an indicator.
        
        Args:
            indicator_name: Name of the indicator column
            lookback: Number of periods to look back
            
        Returns:
            List of indicator values (or None if not available)
        """
        if indicator_name not in self.historical_data.columns:
            return None
        
        values = self.historical_data[indicator_name].tail(lookback).tolist()
        return values if len(values) > 0 else None
    
    def get_bar_history(self, column: str, lookback: int = 10) -> Optional[pd.Series]:
        """
        Get historical OHLCV column values.
        
        Args:
            column: 'open', 'high', 'low', 'close', 'volume'
            lookback: Number of bars
        """
        if column not in self.historical_data.columns:
            return None
        return self.historical_data[column].tail(lookback)


# ============================================================================
# BASE STRATEGY CLASS
# ============================================================================

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    REQUIRED METHODS (must implement):
    - required_indicators(): List indicators needed
    - supports_market(): Declare market compatibility
    - supports_regime(): Declare regime compatibility  
    - supports_volatility(): Declare volatility compatibility
    - generate_signal(): Core strategy logic
    
    OPTIONAL METHODS (can override):
    - compute_dynamic_sl(): Custom stop loss logic
    - compute_dynamic_tp(): Custom take profit logic
    - compute_dynamic_tsl(): Custom trailing stop logic
    - should_force_exit(): Early exit conditions
    - validate_entry(): Additional entry filters
    
    STRATEGIES MUST NOT CONTAIN:
    - Broker API calls
    - Position sizing logic
    - Data fetching code
    - Market execution logic
    
    STRATEGIES = SIGNAL GENERATION + LOGIC ONLY
    """
    
    def __init__(self, name: str, version: str = "1.0", enabled: bool = True):
        """
        Initialize strategy.
        
        Args:
            name: Strategy identifier (e.g., "EMA_Crossover")
            version: Strategy version for tracking changes
            enabled: Whether strategy is active
        """
        self.name = name
        self.version = version
        self.enabled = enabled
        self.logger = logging.getLogger(f"Strategy.{name}")
        
        # Statistics (optional tracking)
        self.signals_generated = 0
        self.buy_signals = 0
        self.sell_signals = 0
        self.hold_signals = 0

    # ========================================================================
    # REQUIRED ABSTRACT METHODS
    # ========================================================================
    
    @abstractmethod
    def required_indicators(self) -> List[str]:
        """
        Return list of indicator names needed from data layer.
        
        Examples: 
            ["rsi_14", "atr_14", "ema_9", "ema_21"]
            ["supertrend_10_3.0", "adx_14"]
            ["bb_upper_20_2.0", "bb_lower_20_2.0"]
        
        The data layer ensures these are available in context.current_bar
        """
        pass
    
    @abstractmethod
    def supports_market(self, market_type: str) -> bool:
        """
        Declare if strategy can run on this market type.
        
        Args:
            market_type: "indian", "crypto", "us_equity", etc.
        
        Returns:
            True if supported, False otherwise
        
        Example:
            return market_type in ["indian", "crypto"]
        """
        pass
    
    @abstractmethod
    def supports_regime(self, regime: MarketRegime) -> bool:
        """
        Declare if strategy works in this market regime.
        
        Args:
            regime: TRENDING, RANGING, VOLATILE, CHOPPY
        
        Returns:
            True if supported
        
        Example:
            return regime in [MarketRegime.TRENDING, MarketRegime.RANGING]
        """
        pass
    
    @abstractmethod
    def supports_volatility(self, volatility: VolatilityBucket) -> bool:
        """
        Declare if strategy handles this volatility level.
        
        Args:
            volatility: LOW, MEDIUM, HIGH
        
        Returns:
            True if supported
        
        Example:
            return volatility in [VolatilityBucket.MEDIUM, VolatilityBucket.HIGH]
        """
        pass
    
    @abstractmethod
    def generate_signal(self, context: StrategyContext) -> Signal:
        """
        🔥 CORE STRATEGY LOGIC 🔥
        
        This is where the magic happens. Given market context, decide:
        - Should I enter long? (BUY)
        - Should I enter short? (SELL)
        - Should I stay out? (HOLD)
        - Should I exit? (EXIT) [for position-aware strategies]
        
        Args:
            context: StrategyContext with all market data
        
        Returns:
            Signal with direction, confidence, and reasoning
        
        Example:
            rsi = context.get_indicator('rsi_14', 50)
            if rsi < 30:
                return Signal(
                    direction="BUY",
                    confidence=0.8,
                    strategy_name=self.name,
                    reasoning=f"RSI {rsi:.2f} oversold, potential reversal"
                )
        """
        pass
    
    # ========================================================================
    # OPTIONAL METHODS - Can override for advanced behavior
    # ========================================================================
    
    def compute_dynamic_sl(self, entry_price: float, context: StrategyContext, 
                          direction: str) -> Optional[float]:
        """
        OPTIONAL: Compute strategy-specific stop loss level.
        
        If not implemented, risk module uses default ATR-based SL.
        
        Args:
            entry_price: Entry price of trade
            context: Current market context
            direction: "BUY" or "SELL"
        
        Returns:
            Stop loss price level (or None to use default)
        
        Example:
            # Use previous swing low for long, swing high for short
            if direction == "BUY":
                swing_low = context.historical_data['low'].tail(20).min()
                return swing_low * 0.995  # 0.5% buffer
        """
        return None  # None = use default from risk module
    
    def compute_dynamic_tp(self, entry_price: float, context: StrategyContext,
                          direction: str) -> Optional[float]:
        """
        OPTIONAL: Compute strategy-specific take profit level.
        
        Args:
            entry_price: Entry price of trade
            context: Current market context
            direction: "BUY" or "SELL"
        
        Returns:
            Take profit price level (or None to use default)
        """
        return None
    
    def compute_dynamic_tsl(self, entry_price: float, current_price: float, 
                           context: StrategyContext, direction: str) -> Optional[float]:
        """
        OPTIONAL: Compute trailing stop loss level.
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            context: Current market context
            direction: "BUY" or "SELL"
        
        Returns:
            New trailing stop level (or None to use default)
        """
        return None
    
    def should_force_exit(self, context: StrategyContext, position: Dict) -> Tuple[bool, str]:
        """
        OPTIONAL: Define strategy-specific forced exit conditions.
        
        Examples:
        - Time-based: Exit at end of session
        - Event-based: Exit on news release
        - Condition-based: Exit if indicator crosses threshold
        
        Args:
            context: Current market context
            position: Current position info (entry_price, direction, timestamp)
        
        Returns:
            (should_exit: bool, reason: str)
        
        Example:
            # Exit at 3:15 PM for Indian market
            if context.market_type == "indian":
                if context.timestamp.time() >= datetime.time(15, 15):
                    return (True, "session_end_approach")
            return (False, "")
        """
        return (False, "")
    
    def validate_entry(self, signal: Signal, context: StrategyContext) -> Tuple[bool, str]:
        """
        OPTIONAL: Additional entry validation filters.
        
        Use this for:
        - Volume filters
        - Spread checks
        - Time-of-day restrictions
        - Max positions limits
        
        Args:
            signal: Generated signal
            context: Current market context
        
        Returns:
            (is_valid: bool, reason: str)
        
        Example:
            # Don't trade in first 15 minutes
            if context.session == SessionType.SESSION_1:
                if (context.timestamp.time() < datetime.time(9, 30)):
                    return (False, "waiting_for_opening_range")
            return (True, "")
        """
        return (True, "")
    
    # ========================================================================
    # INTERNAL VALIDATION & EXECUTION
    # ========================================================================
    
    def validate_context(self, context: StrategyContext) -> bool:
        """
        Verify that required indicators are present in context.
        """
        required = self.required_indicators()
        missing = []
        
        for ind in required:
            if ind not in context.current_bar.index:
                missing.append(ind)
        
        if missing:
            self.logger.warning(
                f"Strategy {self.name}: Missing indicators: {missing}"
            )
            return False
        
        return True
    
    def pre_signal_checks(self, context: StrategyContext) -> Optional[Signal]:
        """
        Perform pre-checks before generating signal.
        Returns HOLD signal if conditions don't allow trading.
        """
        # Check if strategy is enabled
        if not self.enabled:
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning="Strategy disabled"
            )
        
        # Check market compatibility
        if not self.supports_market(context.market_type):
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=f"Market {context.market_type} not supported"
            )
        
        # Check regime compatibility
        if not self.supports_regime(context.regime):
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=f"Regime {context.regime.value} not supported"
            )
        
        # Check volatility compatibility
        if not self.supports_volatility(context.volatility):
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=f"Volatility {context.volatility.value} not supported"
            )
        
        # Check required indicators
        if not self.validate_context(context):
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning="Required indicators missing"
            )
        
        return None  # All checks passed
    
    def execute_signal_generation(self, context: StrategyContext) -> Signal:
        """
        Main execution wrapper. Runs pre-checks, generates signal, validates.
        
        This is the method called by the execution engine.
        """
        # Allow relaxed entry for testing (skip compatibility checks)
        relax_entry = (
            context.additional_info 
            and context.additional_info.get("relax_entry", False)
        )
        
        # Pre-checks
        if not relax_entry:
            pre_check_signal = self.pre_signal_checks(context)
            if pre_check_signal:
                self.hold_signals += 1
                return pre_check_signal
        
        try:
            # Generate signal
            signal = self.generate_signal(context)
            
            # Validate signal
            if not signal.is_valid():
                self.logger.error(f"Invalid signal from {self.name}: {signal}")
                return Signal(
                    direction="HOLD",
                    confidence=0.0,
                    strategy_name=self.name,
                    reasoning="Signal validation failed"
                )
            
            # Track statistics
            self.signals_generated += 1
            if signal.direction == "BUY":
                self.buy_signals += 1
            elif signal.direction == "SELL":
                self.sell_signals += 1
            else:
                self.hold_signals += 1
            
            return signal
            
        except Exception as e:
            self.logger.error(
                f"Error generating signal in {self.name}: {e}", 
                exc_info=True
            )
            return Signal(
                direction="HOLD",
                confidence=0.0,
                strategy_name=self.name,
                reasoning=f"Exception: {str(e)}"
            )
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled,
            "signals_generated": self.signals_generated,
            "buy_signals": self.buy_signals,
            "sell_signals": self.sell_signals,
            "hold_signals": self.hold_signals,
            "buy_rate": (self.buy_signals / self.signals_generated * 100) 
                       if self.signals_generated > 0 else 0,
            "sell_rate": (self.sell_signals / self.signals_generated * 100) 
                        if self.signals_generated > 0 else 0,
        }
    
    def reset_stats(self):
        """Reset signal statistics."""
        self.signals_generated = 0
        self.buy_signals = 0
        self.sell_signals = 0
        self.hold_signals = 0
    
    def __str__(self) -> str:
        return f"{self.name} (v{self.version}) [{'enabled' if self.enabled else 'disabled'}]"
    
    def __repr__(self) -> str:
        return f"<Strategy: {self.name} v{self.version}>"
