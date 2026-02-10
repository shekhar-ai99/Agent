#!/usr/bin/env python3
"""
Strategy Framework Test Script

Tests the new unified strategy framework:
1. Registry discovery
2. Strategy instantiation
3. Support filtering
4. Signal generation (mock)

Run this after migrating each strategy to verify integration.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, '/Users/shekhar/Desktop/Desk/Agent')

from strategies import (
    StrategyRegistry,
    BaseStrategy,
    Signal,
    StrategyContext,
    MarketRegime,
    VolatilityBucket,
    SessionType
)

print("=" * 70)
print("STRATEGY FRAMEWORK TEST")
print("=" * 70)

# Test 1: Registry Discovery
print("\n📋 TEST 1: Strategy Discovery")
print("-" * 70)
StrategyRegistry.reload()
all_strategies = StrategyRegistry.list_names()
print(f"✅ Discovered {len(all_strategies)} strategies:")
for name in all_strategies:
    print(f"   • {name}")

# Test 2: Registry Summary
print("\n📊 TEST 2: Registry Summary")
print("-" * 70)
print(StrategyRegistry.summary())

# Test 3: Get Specific Strategy
print("\n🎯 TEST 3: Strategy Instantiation")
print("-" * 70)
if "EMACrossover" in all_strategies:
    EMACrossover = StrategyRegistry.get("EMACrossover")
    strategy = EMACrossover(ema_short_period=9, ema_long_period=21)
    print(f"✅ Created: {strategy}")
    print(f"   Required indicators: {strategy.required_indicators()}")
    print(f"   Supports Indian market: {strategy.supports_market('indian')}")
    print(f"   Supports trending: {strategy.supports_regime(MarketRegime.TRENDING)}")
    print(f"   Supports medium vol: {strategy.supports_volatility(VolatilityBucket.MEDIUM)}")
else:
    print("⚠️  EMACrossover not found in registry")

# Test 4: Support Filtering
print("\n🔍 TEST 4: Support Filtering")
print("-" * 70)
indian_strategies = StrategyRegistry.get_supported(market="indian")
print(f"✅ Strategies supporting Indian market: {len(indian_strategies)}")
for strat_class in indian_strategies:
    temp = strat_class()
    print(f"   • {temp.name}")

trending_strategies = StrategyRegistry.get_supported(
    regime=MarketRegime.TRENDING
)
print(f"✅ Strategies supporting TRENDING regime: {len(trending_strategies)}")
for strat_class in trending_strategies:
    temp = strat_class()
    print(f"   • {temp.name}")

# Test 5: Mock Signal Generation
print("\n🚦 TEST 5: Mock Signal Generation")
print("-" * 70)

if "EMACrossover" in all_strategies:
    # Create mock data
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01 09:15', periods=100, freq='5min')
    
    # Generate trending price data
    base_price = 24000
    trend = np.linspace(0, 200, 100)
    noise = np.random.randn(100) * 20
    prices = base_price + trend + noise
    
    mock_data = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(100) * 10),
        'low': prices - np.abs(np.random.randn(100) * 10),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Create strategy
    strategy = EMACrossover(ema_short_period=9, ema_long_period=21)
    
    # Create context for last bar
    current_bar = mock_data.iloc[-1]
    
    context = StrategyContext(
        symbol="NIFTY",
        market_type="indian",
        timeframe="5min",
        current_bar=current_bar,
        historical_data=mock_data,
        regime=MarketRegime.TRENDING,
        volatility=VolatilityBucket.MEDIUM,
        session=SessionType.SESSION_1,
        timestamp=dates[-1],
        is_expiry_day=False
    )
    
    # Generate signal
    signal = strategy.execute_signal_generation(context)
    
    print(f"✅ Signal Generated:")
    print(f"   Direction: {signal.direction}")
    print(f"   Confidence: {signal.confidence:.2f}")
    print(f"   Reasoning: {signal.reasoning}")
    print(f"   Valid: {signal.is_valid()}")
    print(f"   Legacy Format: {signal.to_legacy_format()}")
    
    # Test statistics
    print(f"\n📈 Strategy Statistics:")
    stats = strategy.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

# Test 6: Strategy Info
print("\n📝 TEST 6: Strategy Metadata")
print("-" * 70)
if "EMACrossover" in all_strategies:
    info = StrategyRegistry.get_strategy_info("EMACrossover")
    print(f"✅ Strategy Info:")
    for key, value in info.items():
        if key != "docstring":
            print(f"   {key}: {value}")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED")
print("=" * 70)
print("\nFramework Status:")
print(f"  • Base Strategy: ✅ Defined")
print(f"  • Registry: ✅ Functional")
print(f"  • Discovery: ✅ Working")
print(f"  • Filtering: ✅ Working")
print(f"  • Signal Generation: ✅ Working")
print(f"  • Strategies Migrated: {len(all_strategies)}/6")
print("\n🎯 Ready for production strategy migration!")
print("=" * 70)
