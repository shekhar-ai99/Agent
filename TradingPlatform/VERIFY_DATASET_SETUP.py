#!/usr/bin/env python3
"""
VERIFY DATASET SETUP

Quick verification that all dataset components are working correctly.
"""

import sys
from pathlib import Path

def check_structure():
    """Verify directory structure"""
    print("\n" + "="*70)
    print("1️⃣  CHECKING DIRECTORY STRUCTURE")
    print("="*70)
    
    base = Path(__file__).parent
    required_dirs = [
        "datasets",
        "datasets/nifty",
        "datasets/nifty/1min",
        "datasets/nifty/5min",
        "datasets/nifty/15min",
        "data",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = base / dir_path
        exists = full_path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {dir_path:40s} {'[OK]' if exists else '[MISSING]'}")
        if not exists:
            all_exist = False
    
    return all_exist

def check_data_files():
    """Verify CSV data files exist"""
    print("\n" + "="*70)
    print("2️⃣  CHECKING DATA FILES")
    print("="*70)
    
    base = Path(__file__).parent
    files_to_check = [
        ("NIFTY 1min", "datasets/nifty/1min/nifty_historical_data_1min.csv"),
        ("NIFTY 5min", "datasets/nifty/5min/nifty_historical_data_5min.csv"),
        ("NIFTY 15min", "datasets/nifty/15min/nifty_historical_data_15min.csv"),
    ]
    
    all_exist = True
    for name, file_path in files_to_check:
        full_path = base / file_path
        exists = full_path.exists()
        status = "✅" if exists else "❌"
        
        size_info = ""
        if exists:
            size_mb = full_path.stat().st_size / (1024*1024)
            size_info = f" [{size_mb:.2f} MB]"
        
        print(f"{status} {name:20s} {size_info}")
        if not exists:
            all_exist = False
    
    return all_exist

def check_module_imports():
    """Verify Python modules can be imported"""
    print("\n" + "="*70)
    print("3️⃣  CHECKING MODULE IMPORTS")
    print("="*70)
    
    modules_to_check = [
        ("data", "DatasetLoader"),
        ("simulation", "PaperTradingEngine"),
        ("strategies.example_strategies", "RSIStrategy"),
        ("core", "BaseStrategy"),
    ]
    
    all_ok = True
    for module_name, class_name in modules_to_check:
        try:
            if class_name:
                exec(f"from {module_name} import {class_name}")
                print(f"✅ from {module_name} import {class_name}")
            else:
                exec(f"import {module_name}")
                print(f"✅ import {module_name}")
        except Exception as e:
            print(f"❌ from {module_name} import {class_name} - Error: {e}")
            all_ok = False
    
    return all_ok

def check_dataloader_functionality():
    """Test DatasetLoader methods"""
    print("\n" + "="*70)
    print("4️⃣  TESTING DATASETLOADER FUNCTIONALITY")
    print("="*70)
    
    try:
        from data import DatasetLoader
        
        # Test list_available_datasets
        print("Testing DatasetLoader.list_available_datasets()...")
        DatasetLoader.list_available_datasets()
        print("✅ list_available_datasets() works")
        
        # Test load_nifty
        print("\nTesting DatasetLoader.load_nifty('5min')...")
        df = DatasetLoader.load_nifty("5min")
        print(f"✅ Loaded {len(df)} rows")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        
        # Test add_indicators
        print("\nTesting DatasetLoader.add_indicators()...")
        df_with_indicators = DatasetLoader.add_indicators(df)
        print(f"✅ Indicators added")
        print(f"   New columns: sma20, sma50, rsi, atr, bb_upper, bb_middle, bb_lower")
        print(f"   Total columns now: {len(df_with_indicators.columns)}")
        
        # Test load_for_backtest
        print("\nTesting DatasetLoader.load_for_backtest()...")
        data_dict = DatasetLoader.load_for_backtest("NIFTY50", "5min")
        print(f"✅ Backtest data prepared")
        print(f"   Symbols: {list(data_dict.keys())}")
        print(f"   NIFTY50 rows: {len(data_dict['NIFTY50'])}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error testing DatasetLoader: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_engine_integration():
    """Test integration with PaperTradingEngine"""
    print("\n" + "="*70)
    print("5️⃣  TESTING ENGINE INTEGRATION")
    print("="*70)
    
    try:
        from data import DatasetLoader
        from simulation import PaperTradingEngine
        from strategies.example_strategies import RSIStrategy
        
        # Load data
        print("Loading data...")
        data_source = DatasetLoader.load_for_backtest("NIFTY50", "5min")
        data_source["NIFTY50"] = DatasetLoader.add_indicators(data_source["NIFTY50"])
        print(f"✅ Data prepared: {len(data_source['NIFTY50'])} rows")
        
        # Create engine
        print("\nCreating PaperTradingEngine...")
        engine = PaperTradingEngine(
            data_source=data_source,
            initial_capital=100000
        )
        print(f"✅ Engine created with capital: ₹100,000")
        
        # Create strategy
        print("\nCreating strategy...")
        strategy = RSIStrategy()
        print(f"✅ Strategy created: {strategy.name}")
        
        print("\n✅ All integrations working!")
        return True
    
    except Exception as e:
        print(f"❌ Error testing integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all checks"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  TRADINGPLATFORM - DATASET SETUP VERIFICATION".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    checks = [
        ("Directory Structure", check_structure),
        ("Data Files", check_data_files),
        ("Module Imports", check_module_imports),
        ("DatasetLoader Functionality", check_dataloader_functionality),
        ("Engine Integration", check_engine_integration),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            result = check_func()
            results[name] = result
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL CHECKS PASSED - SYSTEM IS READY FOR BACKTESTING!")
        print("="*70)
        print("\nYou can now:")
        print("  1. Load data: df = DatasetLoader.load_nifty('5min')")
        print("  2. Add indicators: df = DatasetLoader.add_indicators(df)")
        print("  3. Run backtest: engine.run(strategy)")
        print("\nSee DATASET_READY.md for quick start guide")
        print("See DATASET_INTEGRATION_GUIDE.md for complete documentation")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - PLEASE REVIEW ERRORS ABOVE")
        print("="*70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
