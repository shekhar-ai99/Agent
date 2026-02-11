"""
Scan all generic strategies and collect required indicators.
"""

import os
import re
from pathlib import Path
from typing import Set, Dict, List
import importlib.util
import sys

def load_module_from_path(file_path: Path):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location("module", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module"] = module
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"  ERROR loading {file_path.name}: {e}")
        return None

def extract_indicators_from_strategy(file_path: Path) -> Set[str]:
    """Extract indicator requirements from a strategy file."""
    indicators = set()
    
    # Parse source code for patterns
    with open(file_path, 'r') as f:
        content = f.read()
        
        # Find return statements in required_indicators()
        # Match: return ['ind1', 'ind2', ...]
        req_ind_pattern = r"def required_indicators.*?return\s+\[(.*?)\]"
        req_matches = re.findall(req_ind_pattern, content, re.DOTALL)
        for match in req_matches:
            # Extract quoted strings
            ind_names = re.findall(r"['\"]([^'\"]+)['\"]", match)
            indicators.update(ind_names)
        
        # Find patterns like: context.data['indicator_name']
        pattern1 = r"context\.data\['([^']+)'\]"
        matches1 = re.findall(pattern1, content)
        indicators.update(matches1)
        
        # Find patterns like: row['indicator_name']
        pattern2 = r"row\['([^']+)'\]"
        matches2 = re.findall(pattern2, content)
        indicators.update(matches2)
        
        # Find patterns like: df['indicator_name']
        pattern3 = r"df\['([^']+)'\]"
        matches3 = re.findall(pattern3, content)
        indicators.update(matches3)
        
        # Find patterns like: data['indicator_name']
        pattern4 = r"data\['([^']+)'\]"
        matches4 = re.findall(pattern4, content)
        indicators.update(matches4)
    
    # Filter out OHLCV columns and common non-indicator fields
    ohlcv = {'open', 'high', 'low', 'close', 'volume', 'timestamp', 'symbol', 'date', 'time'}
    indicators = indicators - ohlcv
    
    return indicators

def scan_all_strategies():
    """Scan all generic strategies and collect indicators."""
    strategies_dir = Path(__file__).parent / "strategies" / "generic"
    
    all_indicators = set()
    strategy_indicators = {}
    
    print("=" * 80)
    print("SCANNING GENERIC STRATEGIES FOR INDICATOR REQUIREMENTS")
    print("=" * 80)
    print()
    
    strategy_files = sorted([f for f in strategies_dir.glob("*.py") if f.name != "__init__.py"])
    
    for file_path in strategy_files:
        print(f"📊 {file_path.name}")
        indicators = extract_indicators_from_strategy(file_path)
        
        if indicators:
            strategy_indicators[file_path.name] = sorted(indicators)
            all_indicators.update(indicators)
            print(f"   Indicators: {', '.join(sorted(indicators))}")
        else:
            print(f"   No indicators found")
        print()
    
    print("=" * 80)
    print(f"SUMMARY: {len(strategy_files)} strategies scanned")
    print("=" * 80)
    print()
    print("ALL UNIQUE INDICATORS REQUIRED:")
    print("-" * 80)
    for indicator in sorted(all_indicators):
        print(f"  • {indicator}")
    print()
    print(f"Total unique indicators: {len(all_indicators)}")
    print("=" * 80)
    
    return all_indicators, strategy_indicators

if __name__ == "__main__":
    all_indicators, strategy_indicators = scan_all_strategies()
    
    # Save to file for reference
    output_file = Path(__file__).parent / "INDICATOR_REQUIREMENTS.txt"
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("INDICATOR REQUIREMENTS FOR ALL GENERIC STRATEGIES\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("ALL UNIQUE INDICATORS:\n")
        f.write("-" * 80 + "\n")
        for indicator in sorted(all_indicators):
            f.write(f"  • {indicator}\n")
        
        f.write(f"\nTotal: {len(all_indicators)} unique indicators\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("PER-STRATEGY BREAKDOWN\n")
        f.write("=" * 80 + "\n\n")
        
        for strategy_file, indicators in sorted(strategy_indicators.items()):
            f.write(f"{strategy_file}:\n")
            for ind in indicators:
                f.write(f"  • {ind}\n")
            f.write("\n")
    
    print(f"\n✅ Results saved to: {output_file}")
