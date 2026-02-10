# analysis/context_data_analyzer.py
import pandas as pd
from pathlib import Path
import sys
import logging

# Adjust path to import from your app
current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from app.config import config # Your app's config
from app.optuna_tuner import filter_data_for_context # Import the actual filter
# from pipeline.run_simulation_step import load_data # Or your data loading utility

# Setup basic logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
analysis_logger = logging.getLogger("ContextDataAnalyzer")

def analyze_context_data_distribution(symbol: str, timeframes: list):
    analysis_logger.info(f"Starting data distribution analysis for {symbol} across timeframes: {timeframes}")

    # Define the context iterators (mirroring optuna_tuner.py's run_contextual_tuning)
    days_iter = getattr(config, "DAYS_FOR_TUNING", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", None])
    sessions_iter = getattr(config, "SESSIONS_FOR_TUNING", ["Morning", "Midday", "Afternoon", None])
    expiry_flags_iter = getattr(config, "EXPIRY_STATUS_FOR_TUNING", [True, False, None])
    market_regimes_iter = getattr(config, "MARKET_REGIMES_FOR_TUNING", ["Trending", "Ranging", "Volatile", "Choppy", None])
    volatility_statuses_iter = getattr(config, "VOLATILITY_STATUSES_FOR_TUNING", ["High", "Normal", "Low", None])
    
    min_bars_threshold = int(getattr(config, "MIN_BARS_FOR_CONTEXT_TUNING", 10))


    for tf in timeframes:
        analysis_logger.info(f"\n--- Analyzing Timeframe: {tf} for Symbol: {symbol} ---")
        
        # Load your feature-engineered data
        # Adjust path as per your structure
        # Make sure PROJECT_ROOT is defined if config uses it for paths
        if not hasattr(config, 'PROJECT_ROOT'): # Simple way to add if not already there
            config.PROJECT_ROOT = Path(__file__).resolve().parent.parent

        data_file_path = Path(getattr(config, "DATA_DIR_PROCESSED", config.PROJECT_ROOT / "data" / "datawithindicator")) / f"{symbol.lower()}__{tf}_with_indicators.csv"
        
        if not data_file_path.exists():
            analysis_logger.warning(f"Data file not found for {tf}: {data_file_path}. Skipping.")
            continue
        
        try:
            # Using a simplified load_data, assuming 'datetime' index and lowercase columns from feature_engine
            df_full_instrument = pd.read_csv(data_file_path, parse_dates=['datetime'], index_col='datetime')
            df_full_instrument.columns = df_full_instrument.columns.str.lower() # Ensure lowercase
            if df_full_instrument.empty:
                 analysis_logger.warning(f"Loaded data for {tf} is empty. Skipping.")
                 continue
            analysis_logger.info(f"Loaded data for {tf} from {data_file_path}, shape: {df_full_instrument.shape}")

        except Exception as e:
            analysis_logger.error(f"Could not load data for {tf} from {data_file_path}: {e}")
            continue
            
        total_contexts_for_tf = 0
        skipped_contexts_for_tf = 0

        for day in days_iter:
            for session in sessions_iter:
                for expiry_flag in expiry_flags_iter:
                    for regime in market_regimes_iter:
                        for vol_status in volatility_statuses_iter:
                            total_contexts_for_tf += 1
                            # Call your actual filter_data_for_context function
                            # Ensure symbol and exchange are correctly passed; exchange might come from config
                            exchange_market = getattr(config, "DEFAULT_MARKET", "NSE") # Example
                            
                            # Make a fresh copy for each context filtering to avoid side-effects if df_full_instrument is modified
                            df_to_filter = df_full_instrument.copy()

                            filtered_df = filter_data_for_context(
                                df=df_to_filter,
                                day_of_week=day,
                                session=session,
                                expiry_status_filter=expiry_flag,
                                symbol=symbol,
                                exchange=exchange_market, 
                                market_regime_filter=regime,
                                volatility_status_filter=vol_status
                            )
                            bar_count = len(filtered_df)
                            status = "OK" if bar_count >= min_bars_threshold else f"SKIPPED (Needs {min_bars_threshold})"
                            
                            if status != "OK":
                                skipped_contexts_for_tf +=1
                            
                            context_desc = f"Day={day}, Sess={session}, Exp={expiry_flag}, Reg={regime}, Vol={vol_status}"
                            analysis_logger.info(f"Context: [{context_desc}] -> Bars: {bar_count} ({status})")
        
        analysis_logger.info(f"--- Timeframe {tf} Summary ---")
        analysis_logger.info(f"Total contexts checked: {total_contexts_for_tf}")
        analysis_logger.info(f"Contexts skipped (less than {min_bars_threshold} bars): {skipped_contexts_for_tf}")
        analysis_logger.info(f"Contexts with enough data: {total_contexts_for_tf - skipped_contexts_for_tf}")


if __name__ == "__main__":
    # Example usage:
    test_symbol = getattr(config, "DEFAULT_SYMBOL", "nifty") 
    test_timeframes = list(getattr(config, "RAW_DATA_FILES", {"5min": ""}).keys())
    if not test_timeframes: test_timeframes = ["5min"]
    
    analyze_context_data_distribution(symbol=test_symbol, timeframes=test_timeframes)