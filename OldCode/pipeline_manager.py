
# # pipeline_manager.py
import subprocess
import sys
import logging
from pathlib import Path
import os
import json # Added for run_manifest.json
import argparse # Added for command-line arguments
from datetime import datetime
from typing import Optional, List, Dict, Any # Added Dict, Any for type hinting

# --- App Imports ---
from app.config import config
# app.optuna_tuner is imported inside main_orchestration to avoid potential circular issues
# if optuna_tuner itself might (even indirectly) cause an import of pipeline_manager
from app.strategies import strategy_factories

# --- Constants and Paths (defined but not acted upon at module level) ---
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_BASE_DIR = PROJECT_ROOT / "runs"
APP_DIR = PROJECT_ROOT / "app"
PIPELINE_SCRIPTS_DIR = PROJECT_ROOT / "pipeline"
FEATURE_ENGINE_SCRIPT_PATH = PIPELINE_SCRIPTS_DIR / "run_feature_engine.py"
SIMULATION_SCRIPT_PATH = PIPELINE_SCRIPTS_DIR / "run_simulation_step.py"

# --- Global variables to store run-specific paths and ID ---
# These will be initialized by setup_run_environment() when script is run as __main__
RUN_ID_GLOBAL: Optional[str] = None
RUN_DIR_GLOBAL: Optional[Path] = None
LOGS_DIR_GLOBAL: Optional[Path] = None
RESULTS_DIR_GLOBAL: Optional[Path] = None
STRATEGY_RESULTS_DIR_GLOBAL: Optional[Path] = None
AGENT_RESULTS_DIR_GLOBAL: Optional[Path] = None
EXECUTION_MODE_GLOBAL: Optional[str] = "unknown_init" # Stores how this pipeline run was initiated

# Get a logger instance for this module.
# Its handlers and formatting will be set in setup_run_environment().
logger = logging.getLogger(__name__)

def generate_run_id_internal() -> str:
    """Generates a timestamp-based Run ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_run_environment(execution_mode: str = "direct_cli_run") -> str:
    """
    Generates RUN_ID, creates run-specific directories, configures logging
    for the main pipeline manager, and saves a run_manifest.json.
    This function should be called ONLY ONCE when the pipeline manager is the main script.
    """
    global RUN_ID_GLOBAL, RUN_DIR_GLOBAL, LOGS_DIR_GLOBAL, RESULTS_DIR_GLOBAL, \
           STRATEGY_RESULTS_DIR_GLOBAL, AGENT_RESULTS_DIR_GLOBAL, EXECUTION_MODE_GLOBAL

    # Prevent re-initialization if called multiple times within the same process
    if RUN_ID_GLOBAL is not None:
        # Using print for early warning as logger might not be fully configured if this is a true re-entry
        print(
            f"WARNING: setup_run_environment called again for RUN_ID: {RUN_ID_GLOBAL}. "
            "Skipping re-initialization of run environment.", file=sys.stderr
        )
        return

    RUN_ID_GLOBAL = generate_run_id_internal()
    EXECUTION_MODE_GLOBAL = execution_mode
    RUN_DIR_GLOBAL = RUNS_BASE_DIR / RUN_ID_GLOBAL
    LOGS_DIR_GLOBAL = RUN_DIR_GLOBAL / "logs"
    RESULTS_DIR_GLOBAL = RUN_DIR_GLOBAL / "results"
    STRATEGY_RESULTS_DIR_GLOBAL = RESULTS_DIR_GLOBAL / "strategy_results"
    AGENT_RESULTS_DIR_GLOBAL = RESULTS_DIR_GLOBAL / "agent_results"

    paths_to_create = [
        RUN_DIR_GLOBAL, LOGS_DIR_GLOBAL, RESULTS_DIR_GLOBAL,
        STRATEGY_RESULTS_DIR_GLOBAL, AGENT_RESULTS_DIR_GLOBAL
    ]
    for path_to_create in paths_to_create:
        path_to_create.mkdir(parents=True, exist_ok=True)

    # Save run_manifest.json
    manifest_data = {
        "run_id": RUN_ID_GLOBAL,
        "project_root": str(PROJECT_ROOT.resolve()),
        "execution_mode": EXECUTION_MODE_GLOBAL,
        "start_time_utc": datetime.utcnow().isoformat() + "Z", # UTC time for standardization
        "log_directory": str(LOGS_DIR_GLOBAL.resolve()),
        "results_directory": str(RESULTS_DIR_GLOBAL.resolve())
    }
    manifest_file_path = RUN_DIR_GLOBAL / "run_manifest.json"
    try:
        with open(manifest_file_path, 'w', encoding='utf-8') as f_manifest:
            json.dump(manifest_data, f_manifest, indent=4)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to write run_manifest.json for RUN_ID {RUN_ID_GLOBAL}: {e}", file=sys.stderr)
        # Consider if pipeline should halt if manifest cannot be written

    # Configure logging for the pipeline_manager.py script itself
    log_file_path = LOGS_DIR_GLOBAL / f'pipeline_manager_{RUN_ID_GLOBAL}.log'

    # Clear any existing handlers from the root logger AND this module's logger
    # to ensure a clean setup for this specific run instance.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    for handler in logger.handlers[:]: # Clear handlers for this module's logger instance
        logger.removeHandler(handler)

    # Configure basicConfig: This sets up the root logger.
    # All other loggers (including the module-level `logger` instance for this file)
    # will use these handlers by default, unless they have propagate=False or their own handlers.
    logging.basicConfig(
        level=config.LOG_LEVEL if hasattr(config, "LOG_LEVEL") else logging.INFO,
        format=config.LOG_FORMAT if hasattr(config, "LOG_FORMAT") else '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout) # Keep console output
        ]
    )
    # Explicitly set the level for this module's logger after basicConfig.
    logger.setLevel(config.LOG_LEVEL if hasattr(config, "LOG_LEVEL") else logging.INFO)
    # logger.propagate = False # Usually not needed if basicConfig is the sole configuration point.
                               # Set to False only if you experience duplicate logs from this specific logger
                               # due to other root configurations not being fully cleared.

    logger.info(f"Run environment setup complete. RUN_ID: {RUN_ID_GLOBAL}, Execution Mode: {EXECUTION_MODE_GLOBAL}")
    logger.info(f"Main logs for pipeline_manager will be in: {log_file_path}")
    if manifest_file_path.exists():
        logger.info(f"Run manifest saved to: {manifest_file_path}")
    else:
        logger.warning(f"Run manifest file was NOT created at: {manifest_file_path}. Check permissions or earlier errors.")


def run_script(script_path: Path, args_list: Optional[List[str]] = None, timeout: Optional[int] = None, log_suffix: str = "") -> bool:
    """Executes a given script as a subprocess and manages its logging."""
    if LOGS_DIR_GLOBAL is None or RUN_ID_GLOBAL is None or EXECUTION_MODE_GLOBAL is None:
        # This implies setup_run_environment was not called. This is a critical failure.
        print("CRITICAL ERROR: Run environment not fully initialized in run_script. Call setup_run_environment() first.", file=sys.stderr)
        return False

    if args_list is None:
        args_list = []

    if not script_path.is_file():
        logger.error(f"Script not found: {script_path}")
        return False

    safe_log_suffix = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in log_suffix)
    log_filename = f"{script_path.stem}{safe_log_suffix}.log"
    script_capture_log_file = LOGS_DIR_GLOBAL / log_filename # Subprocess stdout/stderr capture

    command = [sys.executable, str(script_path)] + [str(arg) for arg in args_list]
    logger.info(f"Running command: {' '.join(command)}")
    logger.info(f"Redirecting script stdout/stderr to: {script_capture_log_file}")

    process_env = os.environ.copy()
    project_root_str = str(PROJECT_ROOT)
    current_pythonpath = process_env.get('PYTHONPATH', '')
    if project_root_str not in current_pythonpath.split(os.pathsep):
        process_env['PYTHONPATH'] = project_root_str + os.pathsep + current_pythonpath
    
    # Pass main run context to subprocesses
    process_env['PIPELINE_MAIN_RUN_ID'] = RUN_ID_GLOBAL
    process_env['PIPELINE_MAIN_LOGS_DIR'] = str(LOGS_DIR_GLOBAL.resolve())
    process_env['PIPELINE_EXECUTION_MODE'] = EXECUTION_MODE_GLOBAL

    logger.debug(f"Effective PYTHONPATH for subprocess: {process_env.get('PYTHONPATH', 'Not Set')}")
    logger.debug(f"Env PIPELINE_MAIN_RUN_ID for subprocess: {process_env['PIPELINE_MAIN_RUN_ID']}")
    logger.debug(f"Env PIPELINE_MAIN_LOGS_DIR for subprocess: {process_env['PIPELINE_MAIN_LOGS_DIR']}")
    logger.debug(f"Env PIPELINE_EXECUTION_MODE for subprocess: {process_env['PIPELINE_EXECUTION_MODE']}")

    try:
        with open(script_capture_log_file, 'w', encoding='utf-8') as f_log:
            subprocess.run(
                command, check=True, timeout=timeout, stdout=f_log,
                stderr=subprocess.STDOUT, text=True, cwd=PROJECT_ROOT, env=process_env
            )
        logger.info(f"--- Script {script_path.name} OK (Log Suffix: '{log_suffix}') ---")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Script {script_path.name} FAIL (Return Code: {e.returncode}) - Check log: {script_capture_log_file}")
    except subprocess.TimeoutExpired:
        logger.error(f"Script {script_path.name} TIMEOUT - Check log: {script_capture_log_file}")
    except Exception as e:
        logger.error(f"Script {script_path.name} ERROR: {e}", exc_info=True)
    return False


def main_orchestration(strategies_to_run_this_time: Optional[List[str]] = None):
    """Main pipeline orchestration logic."""
    if RUN_ID_GLOBAL is None or LOGS_DIR_GLOBAL is None or EXECUTION_MODE_GLOBAL is None:
        logger.critical("Run environment is not initialized. Aborting orchestration.")
        sys.exit("Critical error: Run environment not set up for main_orchestration.")
        
    from app.optuna_tuner import run_contextual_tuning

    logger.info(f"--- Starting Pipeline Manager Orchestration --- RUN ID: {RUN_ID_GLOBAL} (Mode: {EXECUTION_MODE_GLOBAL}) ---")
    logger.info(f"Run directory: {RUN_DIR_GLOBAL}")
    logger.info(f"Run Logs directory (all logs for this run should be here or in subdirs): {LOGS_DIR_GLOBAL}")
    logger.info(f"Project Root interpreted as: {PROJECT_ROOT}")
    logger.info(f"Feature Engine Script Path: {FEATURE_ENGINE_SCRIPT_PATH}")
    logger.info(f"Simulation Script Path: {SIMULATION_SCRIPT_PATH}")

    timeframes_to_process = list(config.RAW_DATA_FILES.keys())
    
    if strategies_to_run_this_time is None:
        strategies_to_process = list(strategy_factories.keys())
    else:
        strategies_to_process = strategies_to_run_this_time

    if not strategies_to_process:
        logger.error("No strategies selected or available to process. Aborting.")
        return False # Indicate failure

    target_symbol = getattr(config, "DEFAULT_SYMBOL", "nifty")
    target_market = getattr(config, "DEFAULT_MARKET", "NSE")
    target_segment = getattr(config, "DEFAULT_SEGMENT", "Index")

    logger.info(f"Target Instrument: {target_symbol} ({target_market}/{target_segment})")
    logger.info(f"Processing timeframes: {timeframes_to_process}")
    logger.info(f"Processing strategies for this run: {strategies_to_process}")
    overall_success = True

    processed_data_dir = Path(getattr(config, "DATA_DIR_PROCESSED", PROJECT_ROOT / "data" / "datawithindicator"))
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # === Phase 1: Feature Generation ===
    for timeframe in timeframes_to_process:
        logger.info(f"\n===== Phase: Feature Generation for {target_symbol} ({timeframe}) =====")
        raw_file_name = config.RAW_DATA_FILES.get(timeframe)
        if not raw_file_name:
            logger.error(f"Missing raw data file configuration for timeframe '{timeframe}'. Skipping.")
            overall_success = False; continue
        data_folder_path = Path(getattr(config, "DATA_FOLDER", PROJECT_ROOT / "data" / "raw"))
        raw_path = data_folder_path / raw_file_name
        indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
        indicator_output_path = processed_data_dir / indicator_file_name
        if not raw_path.exists():
            logger.error(f"Raw data file not found: {raw_path}. Skipping feature generation for {timeframe}.")
            overall_success = False; continue
        
        feature_args = [
            "--input", str(raw_path), "--output", str(indicator_output_path),
            "--symbol", target_symbol, "--exchange", target_market
            # Subprocesses (like run_feature_engine.py) will get PIPELINE_EXECUTION_MODE via environment variable
        ]
        if not run_script(FEATURE_ENGINE_SCRIPT_PATH, feature_args, timeout=300, log_suffix=f"_features_{target_symbol}_{timeframe}"):
            logger.error(f"Feature generation failed for {target_symbol} - {timeframe}.")
            logger.critical(f"❌ Feature generation failed for {target_symbol} - {timeframe}. Aborting pipeline.")
            sys.exit(1)

    # === Phase 2: Independent Strategy Backtests ===
    run_independent_backtests = getattr(config, "RUN_INDEPENDENT_BACKTESTS", True)
    if run_independent_backtests:
        logger.info(f"\n===== Phase: Independent Strategy Simulation for {target_symbol} =====")
        independent_sim_trace_logs_dir = LOGS_DIR_GLOBAL / "independent_sim_traces"
        independent_sim_trace_logs_dir.mkdir(parents=True, exist_ok=True)

        for timeframe in timeframes_to_process:
            indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
            indicator_input_path = processed_data_dir / indicator_file_name
            if not indicator_input_path.is_file():
                logger.critical(f"❌ Expected feature file not found: {indicator_input_path} for independent backtests. Aborting.")
                sys.exit(1)
            for strategy_name in strategies_to_process:
                logger.info(f"--- Running Independent Backtest for Strategy: {strategy_name} ({target_symbol} - {timeframe}) ---")
                result_path = STRATEGY_RESULTS_DIR_GLOBAL / f"{strategy_name}_{target_symbol}_{timeframe}_{RUN_ID_GLOBAL}.json"
                sim_args = [
                    "--input", str(indicator_input_path),
                    "--output-json", str(result_path),
                    "--log-dir", str(independent_sim_trace_logs_dir),
                    "--strategy-name", strategy_name,
                    "--symbol", target_symbol,
                    "--market", target_market,
                    "--run-id", RUN_ID_GLOBAL
                    # Subprocesses (like run_simulation_step.py) will get PIPELINE_EXECUTION_MODE via environment variable
                ]
                if not run_script(SIMULATION_SCRIPT_PATH, sim_args, timeout=600, log_suffix=f"_indep_sim_{strategy_name}_{target_symbol}_{timeframe}"):
                    logger.error(f"Independent simulation failed for {strategy_name} on {target_symbol} - {timeframe}.")
                    logger.critical(f"❌ Independent simulation failed for {strategy_name}. Aborting pipeline.")
                    sys.exit(1)
    else:
        logger.info("Skipping Phase 2: Independent Strategy Backtests as per configuration.")

    # === Phase 3: Contextual Strategy Tuning (Optuna) ===
    run_optuna_phase = getattr(config, "RUN_OPTUNA_TUNING", True)
    if run_optuna_phase:
        logger.info(f"\n===== Phase: Contextual Strategy Tuning (Optuna) for {target_symbol} =====")
        try:
            # Modify app/optuna_tuner.py's run_contextual_tuning to accept 'execution_mode'
            # If it doesn't, it can still access it via os.environ.get('PIPELINE_EXECUTION_MODE')
            run_contextual_tuning(
                symbol=target_symbol, market=target_market, segment=target_segment,
                timeframes=timeframes_to_process, strategies=strategies_to_process,
                n_trials_per_study=config.OPTUNA_TRIALS_PER_CONTEXT,
                max_workers=config.MAX_OPTUNA_WORKERS,
                run_id=RUN_ID_GLOBAL,
                run_specific_logs_dir=LOGS_DIR_GLOBAL,
                execution_mode=EXECUTION_MODE_GLOBAL, # Pass it directly
                context_filter_override=None
            )
            logger.info("✅ Contextual Tuning Phase completed.")
        except TypeError as te: 
            if "execution_mode" in str(te): # Check if the error is about the unexpected keyword
                logger.warning(f"Optuna tuner function might not accept 'execution_mode' parameter yet. Running without passing it directly. Error: {te}")
                # Fallback call without execution_mode direct parameter
                run_contextual_tuning(
                    symbol=target_symbol, market=target_market, segment=target_segment,
                    timeframes=timeframes_to_process, strategies=strategies_to_process,
                    n_trials_per_study=config.OPTUNA_TRIALS_PER_CONTEXT,
                    max_workers=config.MAX_OPTUNA_WORKERS,
                    run_id=RUN_ID_GLOBAL,
                    run_specific_logs_dir=LOGS_DIR_GLOBAL,
                    context_filter_override=None
                )
                logger.info("✅ Contextual Tuning Phase completed (using fallback call for optuna_tuner).")
            else:
                raise # Re-raise other TypeErrors
        except Exception as e:
            logger.error(f"❌ Contextual Strategy Tuning Phase failed: {e}", exc_info=True)
            logger.critical(f"❌ Optuna tuning failed. Aborting pipeline.")
            sys.exit(1)
    else:
        logger.info("Skipping Phase 3: Contextual Strategy Tuning (Optuna) as per configuration.")

    # === Phase 4: Optional Agent Simulation ===
    run_agent_simulation = getattr(config, "RUN_AGENT_SIMULATION", False)
    if run_agent_simulation:
        logger.info(f"\n===== Phase: Agent Simulation for {target_symbol} =====")
        agent_sim_trace_logs_dir = LOGS_DIR_GLOBAL / "agent_sim_traces"
        agent_sim_trace_logs_dir.mkdir(parents=True, exist_ok=True)
        for timeframe in timeframes_to_process:
            indicator_file_name = f"{target_symbol.lower()}__{timeframe}_with_indicators.csv"
            indicator_input_path = processed_data_dir / indicator_file_name
            if not indicator_input_path.is_file():
                logger.warning(f"Skipping agent simulation for {timeframe} due to missing feature file: {indicator_input_path}")
                continue
            logger.info(f"--- Running Simulation for Agent ({target_symbol} - {timeframe}) ---")
            results_filename = f"Agent_{target_symbol}_{timeframe}_{RUN_ID_GLOBAL}.json"
            results_json_path = AGENT_RESULTS_DIR_GLOBAL / results_filename
            agent_sim_args = [
                "--input", str(indicator_input_path),
                "--output-json", str(results_json_path),
                "--log-dir", str(agent_sim_trace_logs_dir),
                "--symbol", target_symbol,
                "--market", target_market,
                "--run-id", RUN_ID_GLOBAL
                # Subprocesses get PIPELINE_EXECUTION_MODE via environment variable
            ]
            if not run_script(SIMULATION_SCRIPT_PATH, agent_sim_args, timeout=600, log_suffix=f"_Agent_{target_symbol}_{timeframe}"):
                 logger.error(f"Agent simulation failed for {target_symbol} - {timeframe}.")
                 overall_success = False
                 logger.warning(f"⚠️ Agent simulation failed for {target_symbol} - {timeframe}. Continuing if other tasks exist.")
    else:
        logger.info("Skipping Phase 4: Agent Simulation as per configuration.")

    logger.info("\n--- Pipeline Manager Finished ---")
    if overall_success:
        logger.info(f"✅ Pipeline completed. Check results in {RESULTS_DIR_GLOBAL} and logs in {LOGS_DIR_GLOBAL}. RUN ID: {RUN_ID_GLOBAL}")
    else:
        logger.warning(f"⚠️ Pipeline completed with errors. Check script logs in {LOGS_DIR_GLOBAL} for details. RUN ID: {RUN_ID_GLOBAL}")
    
    return overall_success


if __name__ == "__main__":
    # --- Argument Parsing for Execution Mode ---
    parser = argparse.ArgumentParser(description="Main Pipeline Manager for Trading Bot")
    parser.add_argument(
        "--execution-mode",
        type=str,
        default="direct_cli_run", # Default if script is run without this argument
        help="Describes how the pipeline was initiated (e.g., 'direct_cli_run', 'ui_triggered_run', 'scheduled_run')."
    )
    cli_args = parser.parse_args() # Parse command-line arguments

    # --- Setup Run Environment (Call this ONCE at the beginning of direct execution) ---
    # Pass the execution_mode from command line (or default)
    setup_run_environment(execution_mode=cli_args.execution_mode)
    
    # Now that logging is configured by setup_run_environment, subsequent logger calls will work.
    try:
        from app.mongo_manager import close_mongo_connection_on_exit
        import atexit
        atexit.register(close_mongo_connection_on_exit)
        logger.info("Registered MongoDB connection closer on atexit.")
    except ImportError:
        logger.warning("Could not import or register MongoDB connection closer. MongoDB connection might not be closed automatically.")

    # --- Test Overrides (apply after setup) ---
    config.RAW_DATA_FILES = {"5min": "nifty_historical_data_5min.csv"} # Example override
    config.OPTUNA_TRIALS_PER_CONTEXT = getattr(config, "OPTUNA_TRIALS_PER_CONTEXT_TEST", 5) # Use test-specific or default
    config.MAX_OPTUNA_WORKERS = getattr(config, "MAX_OPTUNA_WORKERS_TEST", 1)       

    # Determine which strategies to run for this specific execution
    _original_strategy_factories_backup = strategy_factories.copy() # Backup for restoration in finally
    strategy_to_test_override_name = "SuperTrend_ADX"  # Set to None or comment out to run all configured strategies
    #strategy_to_test_override_name = "None"  # Set to None or comment out to run all configured strategies
    

    
    strategies_for_this_run: List[str] # Explicitly type
    if strategy_to_test_override_name:
        if strategy_to_test_override_name in _original_strategy_factories_backup: # Check against backup
            strategies_for_this_run = [strategy_to_test_override_name]
            logger.info(f"--- OVERRIDE: Testing only strategy: {strategy_to_test_override_name} ---")
        else:
            logger.warning(f"--- OVERRIDE WARNING: Strategy '{strategy_to_test_override_name}' not found. Will test all available strategies from factory. ---")
            strategies_for_this_run = list(_original_strategy_factories_backup.keys())
    else:
        strategies_for_this_run = list(_original_strategy_factories_backup.keys())
        logger.info(f"--- No strategy override. Testing all available strategies: {strategies_for_this_run} ---")
    
    exit_code = 1 # Default to error exit code
    try:
        # Pass the determined list of strategies to the orchestration function
        success = main_orchestration(strategies_to_run_this_time=strategies_for_this_run)
        exit_code = 0 if success else 1
    except Exception as e:
        logger.critical(f"Pipeline Manager CRASHED during main_orchestration: {e}", exc_info=True)
        # exit_code remains 1 or can be a specific error code
    finally:
        # Restore original strategy_factories if it was potentially modified for the override.
        # This check ensures we only attempt to restore if the backup exists and is different
        if '_original_strategy_factories_backup' in locals() and \
           strategy_factories is not _original_strategy_factories_backup: # Check if actually changed
             strategy_factories.clear()
             strategy_factories.update(_original_strategy_factories_backup)
             logger.debug("Restored original strategy_factories.")
        logger.info(f"Pipeline manager exiting with code {exit_code}.")


    parser = argparse.ArgumentParser(description="Main Pipeline Manager for Trading Bot")
    parser.add_argument(
        "--execution-mode",
        type=str,
        default="direct_cli_run",
        help="Execution mode (e.g., 'direct_cli_run', 'ui_triggered_run', 'scheduled_run')."
    )
    cli_args = parser.parse_args()

    setup_run_environment(execution_mode=cli_args.execution_mode)
    
    try:
        from app.mongo_manager import close_mongo_connection_on_exit
        import atexit
        atexit.register(close_mongo_connection_on_exit)
        logger.info("Registered MongoDB connection closer on atexit.")
    except ImportError:
        logger.warning("Could not import MongoDB connection closer.")

    # --- Test Overrides ---
    config.RAW_DATA_FILES = {"5min": "nifty_historical_data_5min.csv"}
    config.OPTUNA_TRIALS_PER_CONTEXT = getattr(config, "OPTUNA_TRIALS_PER_CONTEXT_TEST", 5)
    config.MAX_OPTUNA_WORKERS = getattr(config, "MAX_OPTUNA_WORKERS_TEST", 1)

    # Determine which strategies to run
    _original_strategy_factories_backup = strategy_factories.copy()
    strategy_to_test_override_name = "SuperTrend_ADX"  # Run only SuperTrend_ADX for all phases, including tuning

    strategies_for_this_run: List[str]
    if strategy_to_test_override_name:
        if strategy_to_test_override_name in _original_strategy_factories_backup:
            strategies_for_this_run = [strategy_to_test_override_name]
            logger.info(f"--- OVERRIDE: Running only strategy: {strategy_to_test_override_name} for tuning and other phases ---")
        else:
            logger.error(f"--- OVERRIDE ERROR: Strategy '{strategy_to_test_override_name}' not found in strategy_factories. Aborting. ---")
            sys.exit(1)
    else:
        strategies_for_this_run = list(_original_strategy_factories_backup.keys())
        logger.info(f"--- No strategy override. Running all strategies: {strategies_for_this_run} ---")

    exit_code = 1
    try:
        success = main_orchestration(strategies_to_run_this_time=strategies_for_this_run)
        exit_code = 0 if success else 1
    except Exception as e:
        logger.critical(f"Pipeline Manager CRASHED: {e}", exc_info=True)
    finally:
        if '_original_strategy_factories_backup' in locals() and strategy_factories is not _original_strategy_factories_backup:
            strategy_factories.clear()
            strategy_factories.update(_original_strategy_factories_backup)
            logger.debug("Restored original strategy_factories.")
        logger.info(f"Pipeline manager exiting with code {exit_code}.")

    sys.exit(exit_code)