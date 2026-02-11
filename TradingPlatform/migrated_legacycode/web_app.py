
# app/web_app.py

import base64
import logging
import json
import os
import sys
import time
# from venv import logger # REMOVE THIS - Incorrect import
# import pandas as pd # pandas might not be needed here anymore if /results/<run_id> is simplified or removed
from flask import Flask, Response, request, jsonify, render_template
from typing import Dict, List, Optional 
from pathlib import Path
from datetime import datetime
import threading
import subprocess
from collections import defaultdict

# Corrected and consolidated imports from app.reporting
from app.reporting import (
    get_distinct_optuna_study_names,
    get_optuna_trials_data_from_mongo,
    load_contextual_strategy_summary_from_mongo,
    get_specific_run_details_from_mongo,
    get_full_pipeline_run_summary_from_mongo # <<< ENSURE THIS IS IMPORTED
    # calculate_detailed_metrics # This should be removed if not in reporting.py
)
from app.mongo_manager import MongoManager
from bson import ObjectId

try:
    from app.config import config
except ImportError as e:
    # Using print as logger might not be configured yet if config fails
    print(f"CRITICAL ERROR: Failed to import 'app.config'. Ensure config.py exists and is importable. Error: {e}", file=sys.stderr)
    # Depending on how critical config is at startup, you might sys.exit(1)
    # For now, we'll let Flask try to start, but many things might fail.
    class MockConfig: # Minimal mock to prevent further immediate crashes if config is used early
        LOG_LEVEL = "INFO"
        LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        DEFAULT_SYMBOL = "NIFTY" # Add other essential defaults if web_app.py uses them directly
        MONGO_DB_NAME = "trading_bot_default"
        MONGO_COLLECTION_BACKTEST_RESULTS = "strategy_backtest_results_default"

    config = MockConfig()
    print("WARNING: Using mock config due to import error.", file=sys.stderr)


# --- Flask App Setup ---
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

PROJECT_ROOT_WEBAPP = Path(__file__).resolve().parent
app.template_folder = str(PROJECT_ROOT_WEBAPP / 'templates')
app.static_folder = str(PROJECT_ROOT_WEBAPP / 'static')

# Logging setup
log_level_str = getattr(config, "LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
if not app.logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(getattr(config, "LOG_FORMAT", '%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    stream_handler.setFormatter(formatter)
    app.logger.addHandler(stream_handler)
app.logger.setLevel(log_level)
app.logger.propagate = False


# --- Global Status & Lock (Keep as is) ---
backtest_status = {
    "running": False, "run_id": None, "message": "Idle", "error": None,
    "pid": None, "log_file": None
}
status_lock = threading.Lock()

# --- Background Thread for pipeline_manager.py (Keep as is from your latest) ---
def run_pipeline_manager_thread(run_id):
    global backtest_status
    pipeline_successful = False
    process = None
    project_root = Path(__file__).resolve().parent
    pipeline_script_path = project_root / "pipeline_manager.py"
    run_log_dir = project_root / "runs" / run_id / "logs"
    pipeline_manager_log_path = run_log_dir / f"pipeline_manager_{run_id}.log"
    try:
        run_log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
         app.logger.error(f"Failed to create log directory {run_log_dir}: {e}")
         with status_lock:
              backtest_status.update({"message": f"Run {run_id} failed! Cannot create log dir.", "running": False, "error": str(e), "pid": None, "log_file": None})
         return
    try:
        app.logger.info(f"Background thread starting pipeline_manager.py for RUN_ID: {run_id}")
        command = [sys.executable, str(pipeline_script_path), f"--execution-mode=ui_triggered_run_{run_id}"]
        app.logger.info(f"Executing command: {' '.join(command)}")
        app.logger.info(f"Pipeline manager log file will be: {pipeline_manager_log_path}")
        with status_lock:
            backtest_status["log_file"] = str(pipeline_manager_log_path)
            backtest_status["pid"] = None 
        with open(pipeline_manager_log_path, 'w', encoding='utf-8') as log_fp:
            process = subprocess.Popen(
                command, stdout=log_fp, stderr=subprocess.STDOUT,
                cwd=project_root, text=True, env=os.environ.copy()
            )
        with status_lock:
             backtest_status["pid"] = process.pid
        while True:
            with status_lock:
                 should_be_running = backtest_status["running"]
                 current_thread_run_id = backtest_status["run_id"]
            if not should_be_running or current_thread_run_id != run_id:
                 app.logger.info(f"Run {run_id} aborted by status flag. Terminating process {process.pid if process else 'N/A'}.")
                 if process:
                     try:
                         process.terminate(); process.wait(timeout=5)
                     except: process.kill()
                 with status_lock:
                      if backtest_status["run_id"] == run_id:
                           backtest_status.update({"message": f"Run {run_id} aborted by user.", "running": False, "pid": None})
                 return
            if process:
                return_code = process.poll()
                if return_code is not None:
                     pipeline_successful = (return_code == 0)
                     app.logger.info(f"pipeline_manager.py for {run_id} finished with code: {return_code}. Success: {pipeline_successful}")
                     break
            else:
                app.logger.error(f"Process object is None for run {run_id} during monitoring loop.")
                pipeline_successful = False; break
            time.sleep(1.0)
        final_message = f"Run {run_id} completed."
        error_message = None
        current_return_code = process.poll() if process else "N/A"
        if not pipeline_successful:
             final_message = f"Run {run_id} completed with errors (Code: {current_return_code}). Check logs."
             error_message = f"Pipeline failed with code {current_return_code}."
        with status_lock:
             if backtest_status["run_id"] == run_id:
                  backtest_status.update({"message": final_message, "running": False, "error": error_message, "pid": None})
    except Exception as e:
        app.logger.error(f"Error launching/monitoring pipeline_manager.py for run {run_id}: {e}", exc_info=True)
        if process and process.poll() is None: process.kill()
        with status_lock:
            if backtest_status["run_id"] == run_id:
                 backtest_status.update({"message": f"Run {run_id} failed unexpectedly!", "running": False, "error": str(e), "pid": None})

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_backtest', methods=['POST'])
def start_backtest():
    global backtest_status
    with status_lock:
        if backtest_status["running"]:
            return jsonify({"error": "Backtest already running."}), 409
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_placeholder = str(Path(__file__).parent / "runs" / run_id / "logs" / f"pipeline_manager_{run_id}.log")
        pipeline_thread = threading.Thread(target=run_pipeline_manager_thread, args=(run_id,))
        pipeline_thread.daemon = True
        pipeline_thread.start()
        backtest_status.update({
            "running": True, "run_id": run_id,
            "message": f"Run {run_id} starting (Executing pipeline_manager.py)...",
            "error": None, "pid": None, "log_file": log_file_placeholder
        })
        return jsonify({"message": "Backtest run triggered via pipeline_manager.", "run_id": run_id})

@app.route('/status')
def get_status():
    with status_lock:
        return jsonify(backtest_status.copy())

@app.route('/stop_backtest', methods=['POST'])
def stop_backtest():
    global backtest_status
    stopped_pid = None
    current_run_id_for_stop = None
    with status_lock:
        current_run_id_for_stop = backtest_status.get('run_id')
        current_pid = backtest_status.get('pid')
        if not backtest_status["running"]:
            return jsonify({"error": "No backtest running."}), 409
        app.logger.info(f"Stop requested for run {current_run_id_for_stop} (PID: {current_pid})")
        backtest_status["running"] = False
        backtest_status["message"] = f"Stop requested by user for run {current_run_id_for_stop}..."
        stopped_pid = current_pid 
    if stopped_pid:
         app.logger.info(f"Signaled monitoring thread to stop process {stopped_pid} for run {current_run_id_for_stop}.")
    else:
         app.logger.warning(f"Stop requested for run {current_run_id_for_stop}, but PID was not found in status.")
    return jsonify({"message": "Stop signal sent to running process."})

@app.route('/stream_logs/<run_id>')
def stream_logs(run_id):
    def generate_log_updates():
        log_file_path_str = None
        for _ in range(10): # Retry for a few seconds to get log file path
            with status_lock:
                if backtest_status.get("run_id") == run_id:
                    log_file_path_str = backtest_status.get("log_file")
            if log_file_path_str and Path(log_file_path_str).is_file():
                break
            time.sleep(0.5)
        
        if not log_file_path_str or not Path(log_file_path_str).is_file():
            yield f"data: Log file for run {run_id} not found or not ready. Path: {log_file_path_str}\n\n"
            return

        log_file_path = Path(log_file_path_str)
        app.logger.info(f"Starting log stream for: {log_file_path}")
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                while True:
                    line = f.readline()
                    if line:
                        yield f"data: {line.rstrip()}\n\n"
                    else:
                        with status_lock:
                             is_running = backtest_status.get("running", False)
                             current_run_in_status = backtest_status.get("run_id")
                        if not is_running and current_run_in_status == run_id:
                             yield f"data: --- End of logs for run {run_id} (process terminated) ---\n\n"
                             break 
                        time.sleep(0.2)
        except Exception as e:
            app.logger.error(f"Error streaming log file {log_file_path}: {e}")
            yield f"data: ERROR reading log file: {e}\n\n"
        finally:
             app.logger.info(f"Log stream stopped for {run_id}")
    return Response(generate_log_updates(), mimetype='text/event-stream')

# --- OLD Results Endpoint (File-based) - Kept for potential direct file viewing, but not primary for UI ---
@app.route('/results/<pipeline_run_id>')
def get_results_json_from_files(pipeline_run_id):
    app.logger.info(f"Serving file-based results for /results/{pipeline_run_id}")
    results_dir = Path(__file__).parent / "runs" / pipeline_run_id / "results" / "strategy_results"
    if not results_dir.is_dir():
         return jsonify({"error": f"Strategy results directory not found for run ID: {pipeline_run_id}"}), 404
    json_files = sorted(list(results_dir.glob("*.json")))
    if not json_files:
        return jsonify({"error": f"No strategy result JSON files found in {results_dir}"}), 404
    
    result_data = {} 
    for f in json_files:
        try:
            stem_parts = f.stem.split('_')
            timeframe = "UnknownTF"
            strategy_name_key = f.stem 
            if len(stem_parts) >= 2:
                timeframe = stem_parts[-1]
                strategy_name_key = "_".join(stem_parts[:-1])
            if timeframe not in result_data: result_data[timeframe] = {}
            with open(f, 'r', encoding='utf-8') as fp:
                result_data[timeframe][strategy_name_key] = json.load(fp)
        except Exception as e:
            app.logger.error(f"Error processing result file {f.name} for bundle view: {e}", exc_info=True)
            tf_for_error = locals().get('timeframe', 'unknown_tf')
            key_for_error = locals().get('strategy_name_key', f.stem)
            if tf_for_error not in result_data: result_data[tf_for_error] = {}
            result_data[tf_for_error][key_for_error] = {"error": f"Failed to parse: {str(e)}"}
    # The 'summary_table' part is removed as it relied on calculate_detailed_metrics
    return jsonify({"structured_report": result_data})


@app.route('/runs')
def list_runs():
    runs_dir = Path(__file__).parent / "runs"
    if not runs_dir.exists(): return jsonify([])
    try:
        # Improved filtering for run_id format YYYYMMDD_HHMMSS
        run_ids = sorted([d.name for d in runs_dir.iterdir() if d.is_dir() and len(d.name) == 15 and d.name[:8].isdigit() and d.name[9:].isdigit()], reverse=True)
        return jsonify(run_ids)
    except Exception as e:
        app.logger.error(f"Error scanning runs directory: {e}", exc_info=True)
        return jsonify([])

@app.route('/results_html/<pipeline_run_id>') # For the simpler template, if used
def view_html_report(pipeline_run_id):
    return render_template("report_template.html", run_id=pipeline_run_id)

@app.route('/results_html/<path_placeholder_run_id>/detailed') # For the detailed MongoDB-driven report
def view_detailed_mongo_report(path_placeholder_run_id: str):
    # The JS in "report_template_detailed.html" will get 'doc_id' from query params
    return render_template("report_template_detailed.html", run_id_from_path=path_placeholder_run_id)


# --- API Endpoints for MongoDB Data ---
@app.route("/api/report/contextual", methods=["GET"])
def api_contextual_report():
    client = MongoManager.get_client()
    if not client: return jsonify({"error": "Database connection failed"}), 500
    try:
        symbol = request.args.get("symbol", getattr(config, "DEFAULT_SYMBOL", "NIFTY"))
        timeframe = request.args.get("timeframe", "5min")
        session = request.args.get("session", None)
        day = request.args.get("day", None)
        is_expiry_raw = request.args.get("is_expiry", None)
        is_expiry = is_expiry_raw.lower() == "true" if is_expiry_raw is not None else None
        market_condition = request.args.get("market_condition", None)
        volatility_status = request.args.get("volatility_status", None)
        limit = request.args.get("limit", 10, type=int)
        
        results = load_contextual_strategy_summary_from_mongo(
            client=client, symbol=symbol, timeframe=timeframe, session=session, day=day,
            is_expiry=is_expiry, market_condition=market_condition, 
            volatility_status=volatility_status, limit=limit
        )
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Error in /api/report/contextual: {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

@app.route("/api/mongo/optuna_studies", methods=["GET"])
def api_get_optuna_study_names_list():
    client = MongoManager.get_client()
    if not client: return jsonify({"error": "Database connection failed"}), 500
    try:
        symbol = request.args.get("symbol", None)
        timeframe = request.args.get("timeframe", None)
        study_names = get_distinct_optuna_study_names(client=client, symbol=symbol, timeframe=timeframe)
        return jsonify(study_names)
    except Exception as e:
        app.logger.error(f"Error in /api/mongo/optuna_studies: {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

@app.route("/api/mongo/optuna_study_trials/<path:optuna_study_name_b64>", methods=["GET"])
def api_get_optuna_study_trials(optuna_study_name_b64: str):
    client = MongoManager.get_client()
    if not client: return jsonify({"error": "Database connection failed"}), 500
    try:
        optuna_study_name = optuna_study_name_b64 
        try:
            missing_padding = len(optuna_study_name_b64) % 4
            if missing_padding: optuna_study_name_b64 += '=' * (4 - missing_padding)
            optuna_study_name = base64.urlsafe_b64decode(optuna_study_name_b64.encode('ascii')).decode('utf-8')
        except Exception:
            app.logger.warning(f"Not base64 or decode error, using as is: {optuna_study_name_b64}")
        
        symbol = request.args.get("symbol", None)
        timeframe = request.args.get("timeframe", None)
        trials_data = get_optuna_trials_data_from_mongo(
            client=client, optuna_study_name=optuna_study_name, symbol=symbol, timeframe=timeframe
        )
        return jsonify(trials_data if trials_data else []) # Return empty list if no trials
    except Exception as e:
        app.logger.error(f"Error in /api/mongo/optuna_study_trials: {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

@app.route("/api/mongo/run_details/<string:doc_id>", methods=["GET"])
def api_get_specific_run_details(doc_id: str):
    client = MongoManager.get_client()
    if not client: return jsonify({"error": "Database connection failed"}), 500
    if not doc_id or not ObjectId.is_valid(doc_id):
        return jsonify({"error": "Valid MongoDB document ID ('doc_id') parameter is required"}), 400
    try:
        run_details = get_specific_run_details_from_mongo(client=client, run_document_id=doc_id)
        if run_details: return jsonify(run_details)
        else: return jsonify({"error": f"No details found for document ID: {doc_id}"}), 404
    except Exception as e:
        app.logger.error(f"Error in /api/mongo/run_details/{doc_id}: {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

# --- NEW API ENDPOINT for Full Pipeline Run Summary from MongoDB ---
@app.route("/api/mongo/pipeline_run_summary/<string:pipeline_run_id>", methods=["GET"])
def api_get_pipeline_run_summary(pipeline_run_id: str):
    client = MongoManager.get_client()
    if not client:
        app.logger.error("Failed to get MongoDB client for /api/mongo/pipeline_run_summary")
        return jsonify({"error": "Database connection failed"}), 500
    
    if not pipeline_run_id: 
        return jsonify({"error": "pipeline_run_id parameter is required"}), 400
        
    try:
        summary_data = get_full_pipeline_run_summary_from_mongo(
            client=client,
            pipeline_run_id=pipeline_run_id
        )
        if summary_data.get("error"):
            return jsonify(summary_data), 500 
        if summary_data.get("message") and "No data found" in summary_data.get("message","").lower():
            return jsonify(summary_data), 200 # Return 200 with message, let JS handle display
            
        return jsonify(summary_data)
    except Exception as e:
        app.logger.error(f"Error in /api/mongo/pipeline_run_summary/{pipeline_run_id}: {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

# --- REMOVED OLD/AMBIGUOUS ENDPOINT ---
# @app.route("/api/mongo/doc_id_from_run_id/<string:run_id>", methods=["GET"]) ...

if __name__ == '__main__':
    app.logger.info(f"Starting Flask server ({Path(__file__).name})...")
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
