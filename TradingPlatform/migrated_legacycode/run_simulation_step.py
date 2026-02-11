# # pipeline/run_simulation_step.py

# import argparse
# import logging
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import json
# import sys # For sys.exit and path manipulation
# from datetime import datetime # For default run_id

# # --- Add to your existing imports ---
# from typing import Optional, Dict, Any 

# # --- Ensure correct import paths for your app modules ---
# try:
#     from app.config import config
#     from app.simulation_engine import SimpleBacktester
#     from app.strategies import strategy_factories # For single strategy mode
#     from app.agentic_core import RuleBasedAgent # For agent mode
#     # Import MongoManager if performance_logger_mongo is used directly here,
#     # or ensure it's closed by pipeline_manager
#     from app.mongo_manager import MongoManager, close_mongo_connection_on_exit # For clean shutdown
#     import atexit # To register the cleanup function

# except ImportError:
#     # Fallback if run directly and app module is not in PYTHONPATH
#     current_dir = Path(__file__).resolve().parent.parent # Assuming this script is in 'pipeline' subdir
#     if str(current_dir) not in sys.path:
#         sys.path.insert(0, str(current_dir))
#     from app.config import config
#     from app.simulation_engine import SimpleBacktester
#     from app.strategies import strategy_factories
#     from app.agentic_core import RuleBasedAgent
#     from app.mongo_manager import MongoManager, close_mongo_connection_on_exit
#     import atexit

# # --- Logger Setup (ensure it's consistent with your project) ---
# logger = logging.getLogger("RunSimStep") # Specific logger for this script
# if not logger.hasHandlers():
#     log_level = getattr(config, "LOG_LEVEL", "INFO")
#     log_format = getattr(config, "LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     logging.basicConfig(level=log_level, format=log_format, handlers=[logging.StreamHandler(sys.stdout)])
#     # If you want file logging for this script itself (distinct from simulation trace logs):
#     # pipeline_manager_run_id = os.getenv("PIPELINE_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S_simstep"))
#     # script_log_file = Path(config.LOG_DIR or "./logs") / f"run_simulation_step_{pipeline_manager_run_id}.log"
#     # script_log_file.parent.mkdir(parents=True, exist_ok=True)
#     # logger.addHandler(logging.FileHandler(script_log_file))


# def load_data(file_path: Path) -> Optional[pd.DataFrame]:
#     """Loads data from CSV, ensuring datetime index."""
#     if not file_path.exists():
#         logger.error(f"Data file not found: {file_path}")
#         return None
#     try:
#         df = pd.read_csv(file_path, parse_dates=['datetime'])
#         if 'datetime' not in df.columns:
#             logger.error(f"File {file_path} missing 'datetime' column.")
#             return None
#         df.set_index('datetime', inplace=True)
#         df.columns = df.columns.str.lower() # Standardize column names
        
#         required_ohlcv = ['open', 'high', 'low', 'close']
#         if not all(col in df.columns for col in required_ohlcv):
#             logger.error(f"Data file {file_path} is missing one or more OHLC columns.")
#             return None
#         df.dropna(subset=required_ohlcv, inplace=True)
#         if df.empty:
#             logger.warning(f"Data at {file_path} became empty after OHLCV NaN drop.")
#             return None
#         logger.info(f"Successfully loaded data from {file_path}, shape: {df.shape}")
#         return df
#     except Exception as e:
#         logger.error(f"Error loading data from {file_path}: {e}", exc_info=True)
#         return None

# def run_simulation_task(
#     input_file_path: Path,
#     output_json_path: Path,
#     simulation_log_dir: Path, # Base directory for SimpleBacktester's trace logs
#     strategy_name_arg: Optional[str], # From --strategy-name
#     symbol_arg: Optional[str],
#     market_arg: Optional[str],
#     run_id_arg: Optional[str] # MODIFIED (2025-05-09): Added run_id_arg
# ):
#     """
#     Main task function to run a single simulation (either strategy or agent).
#     """
#     logger.info(f"--- Starting Simulation Task ---")
#     logger.info(f"Input data: {input_file_path}")
#     logger.info(f"Output JSON: {output_json_path}")
#     logger.info(f"Simulation Log Dir: {simulation_log_dir}")
#     logger.info(f"Strategy Name (arg): {strategy_name_arg}")
#     logger.info(f"Symbol (arg): {symbol_arg}")
#     logger.info(f"Market (arg): {market_arg}")
#     logger.info(f"Run ID (arg): {run_id_arg}") # MODIFIED (2025-05-09): Log run_id

#     df_instrument = load_data(input_file_path)
#     if df_instrument is None or df_instrument.empty:
#         logger.error("Failed to load data or data is empty. Aborting simulation task.")
#         # Create a dummy error result if needed by pipeline_manager for consistent failure handling
#         error_result = {"error": "Data loading failed or empty data.", "performance_score": -np.inf}
#         with open(output_json_path, 'w') as f:
#             json.dump(error_result, f, indent=4)
#         return # Or raise an exception

#     # Determine mode: Agent or Single Strategy
#     backtester: Optional[SimpleBacktester] = None
#     effective_strategy_name_for_run = "UnknownRun"

#     if strategy_name_arg: # Single Strategy Mode
#         logger.info(f"Mode: Single Strategy ('{strategy_name_arg}')")
#         if strategy_name_arg not in strategy_factories:
#             logger.error(f"Strategy '{strategy_name_arg}' not found in strategy_factories.")
#             # Create dummy error result
#             error_result = {"error": f"Strategy '{strategy_name_arg}' not found.", "performance_score": -np.inf}
#             with open(output_json_path, 'w') as f:
#                 json.dump(error_result, f, indent=4)
#             return
        
#         # Get strategy factory and create instance (usually with default params for independent runs)
#         # Optuna tuner would pass specific params to the factory if this script was adapted for it.
#         # For now, assuming factory() gives a default instance.
#         strategy_factory_func = strategy_factories[strategy_name_arg]
#         try:
#             # MODIFIED (2025-05-09): Pass default or configured initial params if needed by factory
#             # For now, assuming factory can be called with no args for default.
#             # If your factories require initial params (e.g. from config), fetch them here.
#             # Example: initial_params = config.INITIAL_PARAMS.get(strategy_name_arg, {})
#             # strategy_logic_func = strategy_factory_func(**initial_params)
#             strategy_logic_func = strategy_factory_func() 
#         except Exception as e_strat:
#             logger.error(f"Error creating strategy '{strategy_name_arg}' from factory: {e_strat}", exc_info=True)
#             error_result = {"error": f"Strategy creation failed for {strategy_name_arg}.", "performance_score": -np.inf}
#             with open(output_json_path, 'w') as f: json.dump(error_result, f, indent=4)
#             return

#         backtester = SimpleBacktester(strategy_func=strategy_logic_func, strategy_name=strategy_name_arg)
#         effective_strategy_name_for_run = strategy_name_arg

#     else: # Agent Mode
#         logger.info("Mode: Agent-driven Simulation")
#         try:
#             agent_instance = RuleBasedAgent() # Initialize your agent
#             backtester = SimpleBacktester(agent=agent_instance, strategy_name="AgentRun") # strategy_name is for logging
#             effective_strategy_name_for_run = "AgentRun" # Or agent_instance.name
#         except Exception as e_agent:
#             logger.error(f"Error initializing RuleBasedAgent: {e_agent}", exc_info=True)
#             error_result = {"error": "Agent initialization failed.", "performance_score": -np.inf}
#             with open(output_json_path, 'w') as f: json.dump(error_result, f, indent=4)
#             return

#     if backtester is None: # Should not happen if logic above is correct
#         logger.error("Backtester could not be initialized. Aborting.")
#         error_result = {"error": "Backtester init failed.", "performance_score": -np.inf}
#         with open(output_json_path, 'w') as f: json.dump(error_result, f, indent=4)
#         return

#     # --- Run the simulation ---
#     # The `timeframe` argument for run_simulation is the original data timeframe (e.g., "5min")
#     # The log file name will be constructed by SimpleBacktester using this and other info.
#     # `simulation_log_dir` is the specific directory for this run's trace log.
    
#     # MODIFIED (2025-05-09): Extract timeframe from input_file_path for clarity,
#     # or use a passed argument if available and more reliable.
#     # Assuming filename format like: nifty__5min_with_indicators.csv
#     try:
#         parts = input_file_path.stem.split('__')
#         data_timeframe = parts[1].split('_')[0] if len(parts) > 1 else "unknown_tf"
#     except Exception:
#         data_timeframe = "unknown_tf"
#     logger.info(f"Derived data timeframe for simulation run: {data_timeframe}")


#     results_dict = backtester.run_simulation(
#         df=df_instrument,
#         log_dir=simulation_log_dir, # This is where SimpleBacktester will create its log file
#         timeframe=data_timeframe,   # Original timeframe of the data
#         run_id=run_id_arg,          # MODIFIED (2025-05-09): Pass run_id
#         # optuna_trial_params, optuna_study_name, optuna_trial_number are not directly relevant
#         # for independent backtests or agent runs called by this script, unless this script
#         # is also adapted to be the core of Optuna's objective function.
#         # For now, they are None.
#         optuna_trial_params=None,
#         optuna_study_name=None,
#         optuna_trial_number=None
#     )

#     # --- Process and Save Results ---
#     if "error" in results_dict:
#         logger.error(f"Simulation for '{effective_strategy_name_for_run}' failed: {results_dict['error']}")
#         # Optionally, re-raise to make pipeline_manager aware of a hard failure
#         # For now, just saving the error in JSON.
#     else:
#         logger.info(f"Simulation for '{effective_strategy_name_for_run}' completed.")
#         logger.info(f"Performance Score: {results_dict.get('performance_score', 'N/A')}")
#         logger.info(f"Total PnL: {results_dict.get('total_pnl', 'N/A')}, Trades: {results_dict.get('trade_count', 'N/A')}")

#     # Add metadata to the results
#     results_dict["metadata"] = {
#         "run_id": run_id_arg, # MODIFIED (2025-05-09): Include run_id
#         "simulation_mode": "Agent" if not strategy_name_arg else "SingleStrategy",
#         "strategy_or_agent_name": effective_strategy_name_for_run,
#         "input_file": str(input_file_path.name),
#         "symbol": symbol_arg or getattr(config, "DEFAULT_SYMBOL", "Unknown"), # Use arg or default
#         "market": market_arg or getattr(config, "DEFAULT_MARKET", "Unknown"),
#         "timeframe_data": data_timeframe,
#         "simulation_timestamp": datetime.now().isoformat(),
#         "log_trace_file": str(Path(backtester.log_file_path).name) if backtester.log_file_path else "N/A"
#     }
#     # Ensure parameters used are also part of the results if it's a single strategy run
#     # For agent runs, the "parameters" are more complex (agent's internal state/rules)
#     if strategy_name_arg and "params_used_this_run" not in results_dict:
#         # This part might need refinement based on how SimpleBacktester returns params
#         # For now, adding a placeholder if not returned by run_simulation explicitly for single strategy
#         results_dict["params_used_this_run"] = results_dict.get("indicator_config", "default_or_not_specified")


#     try:
#         output_json_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(output_json_path, 'w') as f:
#             # MODIFIED (2025-05-09): Custom encoder for numpy types if not handled by _convert_types_for_mongo
#             # However, SimpleBacktester should ideally return Python-native types.
#             # For now, assuming results_dict contains mostly Python natives.
#             json.dump(results_dict, f, indent=4, default=str) # default=str for non-serializable
#         logger.info(f"Results saved to {output_json_path}")
#     except Exception as e:
#         logger.error(f"Failed to save results JSON to {output_json_path}: {e}", exc_info=True)
#         # If saving JSON fails, we might still want to signal error if results_dict had an error
#         if "error" in results_dict:
#              raise ValueError(f"Simulation failed: {results_dict['error']} (and JSON save also failed)")


#     # MODIFIED (2025-05-09): If simulation itself had an error, raise it to fail the subprocess
#     if "error" in results_dict:
#         logger.error(f"Raising ValueError because simulation reported an error: {results_dict['error']}")
#         raise ValueError(f"Simulation failed: {results_dict['error']}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run a backtest simulation for a strategy or agent.")
#     parser.add_argument("--input", type=Path, required=True, help="Path to the input CSV data file with indicators.")
#     parser.add_argument("--output-json", type=Path, required=True, help="Path to save the output results JSON file.")
#     parser.add_argument("--log-dir", type=Path, required=True, help="Directory to store the detailed simulation trace log file.")
#     parser.add_argument("--strategy-name", type=str, help="Name of the strategy to run (from strategy_factories). If not provided, runs in Agent mode.")
#     parser.add_argument("--symbol", type=str, help="Symbol being traded (e.g., NIFTY, BANKNIFTY). Used for metadata.")
#     parser.add_argument("--market", type=str, help="Market/Exchange (e.g., NSE, BSE). Used for metadata.")
#     # MODIFIED (2025-05-09): Added --run-id argument
#     parser.add_argument("--run-id", type=str, help="Pipeline Run ID for linking results and logs.", default=f"sim_step_{datetime.now().strftime('%Y%m%d%H%M%S')}")

#     args = parser.parse_args()

#     # Register MongoDB connection cleanup
#     atexit.register(close_mongo_connection_on_exit)

#     try:
#         run_simulation_task(
#             input_file_path=args.input,
#             output_json_path=args.output_json,
#             simulation_log_dir=args.log_dir,
#             strategy_name_arg=args.strategy_name,
#             symbol_arg=args.symbol,
#             market_arg=args.market,
#             run_id_arg=args.run_id # MODIFIED (2025-05-09): Pass it here
#         )
#         logger.info("run_simulation_step.py finished successfully.")
#         sys.exit(0) # Explicit success exit
#     except ValueError as ve: # Catch the ValueError raised on simulation error
#         logger.error(f"Simulation task failed with ValueError: {ve}")
#         sys.exit(1) # Exit with error code
#     except Exception as e:
#         logger.error(f"An unhandled error occurred in run_simulation_step.py: {e}", exc_info=True)
#         sys.exit(1) # Exit with error code

# pipeline/run_simulation_step.py

import argparse
import logging
from pathlib import Path
import numpy as np # For -np.inf
import pandas as pd
import json
import sys # For sys.exit and path manipulation
from datetime import datetime # For default run_id

# --- Add to your existing imports ---
from typing import Optional, Dict, Any, List # Added List

# --- Ensure correct import paths for your app modules ---
try:
    from app.config import config
    from app.simulation_engine import SimpleBacktester
    from app.strategies import strategy_factories # For single strategy mode
    from app.agentic_core import RuleBasedAgent # For agent mode
    from app.mongo_manager import MongoManager, close_mongo_connection_on_exit # For clean shutdown
    import atexit # To register the cleanup function
    from pymongo import MongoClient # For type hinting, though MongoManager might abstract it

except ImportError:
    # Fallback if run directly and app module is not in PYTHONPATH
    current_dir = Path(__file__).resolve().parent.parent # Assuming this script is in 'pipeline' subdir
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from app.config import config
    from app.simulation_engine import SimpleBacktester
    from app.strategies import strategy_factories
    from app.agentic_core import RuleBasedAgent
    from app.mongo_manager import MongoManager, close_mongo_connection_on_exit
    import atexit
    from pymongo import MongoClient


# --- Logger Setup (ensure it's consistent with your project) ---
logger = logging.getLogger("RunSimStep") # Specific logger for this script
if not logger.hasHandlers():
    log_level = getattr(config, "LOG_LEVEL", "INFO")
    log_format = getattr(config, "LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Configure basicConfig if this script can be run standalone and needs its own root logging setup
    # If pipeline_manager always calls it, pipeline_manager's logging might suffice if this logger propagates.
    # For robustness, giving it a default handler:
    logging.basicConfig(level=log_level, format=log_format, handlers=[logging.StreamHandler(sys.stdout)])
    logger.setLevel(log_level) # Ensure this logger instance respects the level
    # logger.propagate = False # Set to True if you want messages to also go to root logger (if configured by pipeline_manager)


def load_data(file_path: Path) -> Optional[pd.DataFrame]:
    """Loads data from CSV, ensuring datetime index and lowercase columns."""
    if not file_path.exists():
        logger.error(f"Data file not found: {file_path}")
        return None
    try:
        # Try to infer datetime format, or specify if known
        df = pd.read_csv(file_path, parse_dates=['datetime']) # Assuming 'datetime' is the column
        if 'datetime' not in df.columns:
            # Attempt with a common alternative like 'timestamp'
            if 'timestamp' in df.columns:
                logger.warning("Column 'datetime' not found, trying 'timestamp' for date parsing.")
                df = pd.read_csv(file_path, parse_dates=['timestamp'])
                df.rename(columns={'timestamp': 'datetime'}, inplace=True)
            else:
                logger.error(f"File {file_path} missing 'datetime' or 'timestamp' column for index.")
                return None

        df.set_index('datetime', inplace=True)
        df.columns = df.columns.str.lower() # Standardize column names
        
        required_ohlcv = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_ohlcv):
            missing_cols = [col for col in required_ohlcv if col not in df.columns]
            logger.error(f"Data file {file_path} is missing one or more OHLC columns: {missing_cols}")
            return None
            
        # Convert OHLCV to numeric, coercing errors to NaN, then drop rows with NaN in these critical columns
        for col in required_ohlcv + ['volume']: # Include volume if it's critical and should be numeric
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=required_ohlcv, inplace=True)
        if df.empty:
            logger.warning(f"Data at {file_path} became empty after OHLC NaN drop or type coercion.")
            return None
            
        logger.info(f"Successfully loaded data from {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}", exc_info=True)
        return None

def run_simulation_task(
    input_file_path: Path,
    output_json_path: Path,
    simulation_log_dir: Path,
    strategy_name_arg: Optional[str],
    symbol_arg: Optional[str],
    market_arg: Optional[str],
    run_id_arg: Optional[str]
):
    """
    Main task function to run a single simulation (either strategy or agent).
    """
    logger.info(f"--- Starting Simulation Task ---")
    logger.info(f"Input data: {input_file_path}")
    logger.info(f"Output JSON: {output_json_path}")
    logger.info(f"Simulation Log Dir (for SimpleBacktester traces): {simulation_log_dir}")
    logger.info(f"Strategy Name (from arg): {strategy_name_arg}")
    logger.info(f"Symbol (from arg): {symbol_arg}")
    logger.info(f"Market (from arg): {market_arg}")
    logger.info(f"Run ID (from arg): {run_id_arg}")

    # Ensure output directory exists for error case JSONs too
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    df_instrument = load_data(input_file_path)
    if df_instrument is None or df_instrument.empty:
        logger.error("Failed to load data or data is empty. Aborting simulation task.")
        error_result = {"error": "Data loading failed or data is empty.", "performance_score": -np.inf}
        with open(output_json_path, 'w') as f:
            json.dump(error_result, f, indent=4)
        raise ValueError("Data loading failed for simulation task.") # Raise error to signal failure

    backtester: Optional[SimpleBacktester] = None
    effective_strategy_name_for_run = "UnknownRun" # Used for logging and metadata

    if strategy_name_arg: # Single Strategy Mode
        logger.info(f"Mode: Single Strategy ('{strategy_name_arg}')")
        if strategy_name_arg not in strategy_factories:
            logger.error(f"Strategy '{strategy_name_arg}' not found in strategy_factories.")
            error_result = {"error": f"Strategy '{strategy_name_arg}' not found.", "performance_score": -np.inf}
            with open(output_json_path, 'w') as f: json.dump(error_result, f, indent=4)
            raise ValueError(f"Strategy '{strategy_name_arg}' not found.")

        strategy_factory_func = strategy_factories[strategy_name_arg]
        try:
            strategy_logic_func = strategy_factory_func() 
            logger.info(f"Strategy '{strategy_name_arg}' instantiated from factory.")
        except Exception as e_strat:
            logger.error(f"Error creating strategy '{strategy_name_arg}' from factory: {e_strat}", exc_info=True)
            error_result = {"error": f"Strategy creation failed for {strategy_name_arg}: {str(e_strat)}", "performance_score": -np.inf}
            with open(output_json_path, 'w') as f: json.dump(error_result, f, indent=4)
            raise ValueError(f"Strategy creation failed for {strategy_name_arg}.")

        backtester = SimpleBacktester(strategy_func=strategy_logic_func, strategy_name=strategy_name_arg)
        effective_strategy_name_for_run = strategy_name_arg

    else: # Agent Mode
        logger.info("Mode: Agent-driven Simulation")
        mongo_client_for_agent: Optional[MongoClient] = None # Define here for visibility in finally
        try:
            logger.info("Attempting to get MongoDB client from MongoManager for RuleBasedAgent...")
            mongo_client_for_agent = MongoManager.get_client() 
            
            if not mongo_client_for_agent:
                logger.critical("Failed to get MongoDB client from MongoManager. Agent cannot be initialized.")
                error_result = {"error": "MongoDB client acquisition failed for Agent.", "performance_score": -np.inf}
                with open(output_json_path, 'w') as f: json.dump(error_result, f, indent=4)
                raise ConnectionError("MongoDB client acquisition failed for Agent.")

            logger.info("MongoDB client obtained. Initializing RuleBasedAgent...")
            agent_instance = RuleBasedAgent(mongo_client=mongo_client_for_agent)
            
            agent_run_log_name = "AgentRun"
            if hasattr(config, 'AGENT_USE_TUNED_PARAMS'):
                current_agent_mode = "TunedSLTP" if config.AGENT_USE_TUNED_PARAMS else "DefaultSLTP"
                agent_run_log_name += f"_{current_agent_mode}"
            
            backtester = SimpleBacktester(agent=agent_instance, strategy_name=agent_run_log_name)
            effective_strategy_name_for_run = agent_run_log_name
            logger.info(f"RuleBasedAgent initialized and passed to SimpleBacktester. Agent's SL/TP/TSL mode: '{agent_instance.parameter_mode}' (from config).")

        except ConnectionError as ce: # Catch connection errors specifically
            logger.error(f"MongoDB ConnectionError during Agent setup: {ce}", exc_info=True)
            error_result = {"error": f"Agent setup MongoDB ConnectionError: {str(ce)}", "performance_score": -np.inf}
            with open(output_json_path, 'w') as f: json.dump(error_result, f, indent=4)
            raise # Re-raise to signal failure
        except Exception as e_agent:
            logger.error(f"Error initializing RuleBasedAgent or its dependencies: {e_agent}", exc_info=True)
            error_result = {"error": f"Agent initialization or dependency failed: {str(e_agent)}", "performance_score": -np.inf}
            with open(output_json_path, 'w') as f: json.dump(error_result, f, indent=4)
            raise # Re-raise to signal failure

    if backtester is None: # Should be caught by earlier returns/raises
        logger.critical("Backtester could not be initialized due to prior errors. Aborting.")
        error_result = {"error": "Backtester init failed (should have been caught earlier).", "performance_score": -np.inf}
        with open(output_json_path, 'w') as f: json.dump(error_result, f, indent=4)
        raise ValueError("Backtester initialization failed critically.")

    try:
        parts = input_file_path.stem.split('__')
        data_timeframe = parts[1].split('_')[0] if len(parts) > 1 else "unknown_tf"
    except Exception:
        data_timeframe = "unknown_tf"
        logger.warning(f"Could not reliably derive timeframe from filename {input_file_path.stem}, using '{data_timeframe}'. Consider passing timeframe as an argument.")
    logger.info(f"Derived data timeframe for simulation run: {data_timeframe}")

    results_dict = backtester.run_simulation(
        df=df_instrument,
        log_dir=simulation_log_dir,
        timeframe=data_timeframe,
        run_id=run_id_arg,
        optuna_trial_params=None, # This script handles direct runs, not Optuna trials itself
        optuna_study_name=None,
        optuna_trial_number=None
    )

    if "error" in results_dict:
        logger.error(f"Simulation for '{effective_strategy_name_for_run}' failed with error: {results_dict['error']}")
    else:
        logger.info(f"Simulation for '{effective_strategy_name_for_run}' completed.")
        logger.info(f"Performance Score: {results_dict.get('performance_score', 'N/A')}")
        logger.info(f"Total PnL: {results_dict.get('total_pnl', 'N/A')}, Trades: {results_dict.get('trade_count', 'N/A')}")

    results_dict["metadata"] = {
        "pipeline_run_id": run_id_arg, # Changed from "run_id" to avoid conflict with Optuna's run_id if any
        "simulation_mode": "Agent" if not strategy_name_arg else "SingleStrategy",
        "strategy_or_agent_name_used": effective_strategy_name_for_run, # Name used for SimpleBacktester
        "input_file": str(input_file_path.name),
        "symbol": symbol_arg or getattr(config, "DEFAULT_SYMBOL", "UnknownSymbol"),
        "market": market_arg or getattr(config, "DEFAULT_MARKET", "UnknownMarket"),
        "timeframe_data_used": data_timeframe,
        "simulation_script_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "simulation_trace_log_filename": str(Path(backtester.log_file_path).name) if backtester and backtester.log_file_path else "N/A"
    }
    # Ensure 'params_used_this_run' from results_dict is the one saved, not a placeholder
    # SimpleBacktester._log_and_prepare_final_results now populates this comprehensively
    # So, we don't need to add a placeholder here.

    try:
        with open(output_json_path, 'w') as f:
            json.dump(results_dict, f, indent=4, default=str)
        logger.info(f"Results saved to {output_json_path}")
    except Exception as e:
        logger.error(f"Failed to save results JSON to {output_json_path}: {e}", exc_info=True)
        if "error" in results_dict:
             raise ValueError(f"Simulation failed: {results_dict['error']} (and JSON save also failed: {str(e)})")
        else:
             raise IOError(f"Failed to save results JSON: {str(e)}")

    if "error" in results_dict and results_dict["error"]: # Check if error has a value
        logger.error(f"Raising ValueError because simulation reported an error: {results_dict['error']}")
        raise ValueError(f"Simulation failed: {results_dict['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a backtest simulation for a strategy or agent.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input CSV data file with indicators.")
    parser.add_argument("--output-json", type=Path, required=True, help="Path to save the output results JSON file.")
    parser.add_argument("--log-dir", type=Path, required=True, help="Directory to store the detailed simulation trace log file (for SimpleBacktester).")
    parser.add_argument("--strategy-name", type=str, help="Name of the strategy to run (from strategy_factories). If not provided, runs in Agent mode.")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol being traded (e.g., NIFTY, BANKNIFTY).") # Made symbol required
    parser.add_argument("--market", type=str, required=True, help="Market/Exchange (e.g., NSE, BSE).") # Made market required
    parser.add_argument("--run-id", type=str, help="Pipeline Run ID for linking results and logs.", default=f"sim_step_direct_{datetime.now().strftime('%Y%m%d%H%M%S')}")

    args = parser.parse_args()

    # Register MongoDB connection cleanup (ensure MongoManager is set up to handle this)
    # This should ideally be managed by MongoManager itself if it provides a global/singleton client.
    atexit.register(close_mongo_connection_on_exit)
    logger.info("Registered MongoManager's connection closer via atexit for script exit.")

    try:
        run_simulation_task(
            input_file_path=args.input,
            output_json_path=args.output_json,
            simulation_log_dir=args.log_dir,
            strategy_name_arg=args.strategy_name,
            symbol_arg=args.symbol,
            market_arg=args.market,
            run_id_arg=args.run_id
        )
        logger.info(f"{Path(__file__).name} finished successfully for output: {args.output_json}")
        sys.exit(0)
    except (ValueError, ConnectionError, IOError) as ve: # Catch specific errors raised by the task
        logger.error(f"Terminating {Path(__file__).name} due to error: {ve}")
        sys.exit(1)
    except Exception as e: # Catch any other unexpected errors
        logger.critical(f"An unhandled critical error occurred in {Path(__file__).name}: {e}", exc_info=True)
        sys.exit(2)


    parser = argparse.ArgumentParser(description="Run a backtest simulation for a strategy or agent.")
    parser.add_argument("--input", type=Path, required=True, help="Path to input CSV or directory for multiple timeframes.")
    parser.add_argument("--output-json", type=Path, required=True, help="Path for results JSON or output directory.")
    parser.add_argument("--log-dir", type=Path, required=True, help="Directory for simulation trace logs.")
    parser.add_argument("--strategy-name", type=str, help="Strategy name (else Agent mode).")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol (e.g., NIFTY).")
    parser.add_argument("--market", type=str, required=True, help="Market (e.g., NSE).")
    parser.add_argument("--run-id", type=str, default=f"sim_step_{datetime.now().strftime('%Y%m%d%H%M%S')}", help="Run ID.")
    parser.add_argument("--parallel", action="store_true", help="Run simulations for multiple timeframes in parallel.")

    args = parser.parse_args()
    atexit.register(close_mongo_connection_on_exit)

    try:
        if args.parallel:
            input_dir = args.input if args.input.is_dir() else args.input.parent
            output_dir = args.output_json if args.output_json.is_dir() else args.output_json.parent
            input_files = [input_dir / config.RAW_DATA_FILES[tf] for tf in config.BACKTEST_TIMEFRAMES]
            results = run_parallel_simulation(
                input_files=input_files,
                output_dir=output_dir,
                log_dir=args.log_dir,
                strategy_name=args.strategy_name,
                symbol=args.symbol,
                market=args.market,
                run_id=args.run_id
            )
            logger.info(f"Parallel simulation completed: {results.keys()}")
        else:
            results = run_simulation_task(
                input_file_path=args.input,
                output_json_path=args.output_json,
                simulation_log_dir=args.log_dir,
                strategy_name_arg=args.strategy_name,
                symbol_arg=args.symbol,
                market_arg=args.market,
                run_id_arg=args.run_id
            )
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Script failed: {e}", exc_info=True)
        sys.exit(1)