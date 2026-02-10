
# app/reporting.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import argparse
import sys
from collections import defaultdict # Ensure this is imported

# MongoDB imports
from pymongo import MongoClient, DESCENDING, ASCENDING
from bson import ObjectId # Ensure ObjectId is imported

try:
    from app.config import config
    from app.mongo_manager import MongoManager # Import MongoManager
except ImportError:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from app.config import config
    from app.mongo_manager import MongoManager

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    log_level_from_config = getattr(config, "LOG_LEVEL", "INFO")
    log_format_from_config = getattr(config, "LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=log_level_from_config, format=log_format_from_config, handlers=[logging.StreamHandler(sys.stdout)])

# --- Jinja2 Environment Setup ---
TEMPLATE_DIR_PATHS_TO_TRY = [
    Path(__file__).resolve().parent.parent / "templates",
    Path(__file__).resolve().parent / "templates",
]
if hasattr(config, "TEMPLATES_DIR") and config.TEMPLATES_DIR and Path(config.TEMPLATES_DIR).exists():
    TEMPLATE_DIR_PATHS_TO_TRY.insert(0, Path(config.TEMPLATES_DIR))

LOADED_TEMPLATE_DIR = None
for p_dir in TEMPLATE_DIR_PATHS_TO_TRY:
    if p_dir.exists() and p_dir.is_dir():
        LOADED_TEMPLATE_DIR = p_dir
        break
if LOADED_TEMPLATE_DIR:
    env = Environment(loader=FileSystemLoader(str(LOADED_TEMPLATE_DIR)))
    logger.info(f"Jinja2 Environment loaded from: {LOADED_TEMPLATE_DIR}")
else:
    logger.error(f"Could not find templates directory. Tried: {TEMPLATE_DIR_PATHS_TO_TRY}.")
    class DummyEnv:
        def get_template(self, name): raise FileNotFoundError(f"Template {name} not found, Jinja2 env failed to load.")
    env = DummyEnv()

# ========= Functions for Reporting from JSON Files (Kept for compatibility/other uses if any) =========
def load_simulation_results(json_path: Path) -> dict:
    """Loads simulation results from a JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        logger.debug(f"Successfully loaded simulation results from {json_path}") # Changed to debug
        return results
    except FileNotFoundError:
        logger.error(f"JSON results file not found: {json_path}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {json_path}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {json_path}: {e}")
    return {}

def create_single_backtest_report_html(results: dict, output_html_path: Path, template_name: str = "report_template.html"):
    """
    Generates an HTML performance report from a single backtest's simulation results (JSON) using a Jinja2 template.
    """
    if not results:
        logger.warning("No results data to generate report. Skipping HTML generation.")
        return
    try:
        template = env.get_template(template_name)
    except Exception as e:
        logger.error(f"Failed to load template '{template_name}': {e}. Ensure template exists in {LOADED_TEMPLATE_DIR}.")
        return

    # Adapt to use keys directly from simulation_engine output if overall_metrics is not present
    metrics_source = results.get("overall_metrics", results)
    
    # Ensure metrics_source is a dict before creating DataFrame
    if not isinstance(metrics_source, dict):
        logger.error(f"Metrics source is not a dictionary for template {template_name}. Data: {metrics_source}")
        metrics_html = "<p>Error: Metrics data is not in the expected format.</p>"
    else:
        metrics_df = pd.DataFrame(list(metrics_source.items()), columns=['Metric', 'Value'])
        # Formatting for display
        for col in metrics_df.columns:
            if not metrics_df.empty and len(metrics_df[col]) > 0 and metrics_df[col].iloc[0] is not None:
                if metrics_df[col].dtype == 'float64' or isinstance(metrics_df[col].iloc[0], float) :
                    metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int,float)) and pd.notnull(x) else x)
        metrics_html = metrics_df.to_html(classes='table table-striped table-hover table-sm', index=False, border=0)

    # Check for equity_curve or pnl_curve (prefer equity_curve if simulation_engine provides it)
    pnl_curve_data = results.get("equity_curve", results.get("pnl_curve", []))
    equity_curve_html = "<p>No P&L/Equity curve data available or data format incorrect.</p>"
    if pnl_curve_data and isinstance(pnl_curve_data, list) and len(pnl_curve_data) > 0:
        pnl_df = pd.DataFrame(pnl_curve_data)
        # Check for expected columns, e.g., 'timestamp' and 'equity'
        if not pnl_df.empty and 'timestamp' in pnl_df.columns and 'equity' in pnl_df.columns:
            try:
                pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pnl_df['timestamp'], y=pnl_df['equity'], mode='lines', name='Equity Curve'))
                fig.update_layout(title='Equity Curve', xaxis_title='Time', yaxis_title='Equity')
                equity_curve_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            except Exception as plot_e:
                logger.error(f"Error generating P&L curve plot: {plot_e}")
                equity_curve_html = f"<p>Error generating P&L plot: {plot_e}</p>"
        else:
            logger.warning("PNL/Equity curve data in JSON is missing 'timestamp' or 'equity' columns or is empty.")

    trades_log = results.get("trades_details", results.get("trade_log", []))
    trades_html = "<p>No trade log data available.</p>"
    if trades_log and isinstance(trades_log, list) and len(trades_log) > 0:
        trades_df = pd.DataFrame(trades_log)
        if not trades_df.empty:
            # Standardize column names for formatting if they vary (e.g., PnL_Net vs pnl)
            cols_to_format_numeric = ['EntryPrice', 'ExitPrice', 'PnL_Net', 'PnL_Gross', 'pnl', 'profit_loss', 'qty', 'commission', 'sl_price', 'tp_price', 'atr_at_entry']
            for col_name in cols_to_format_numeric:
                if col_name in trades_df.columns:
                    trades_df[col_name] = trades_df[col_name].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) and pd.notnull(x) else x)
            trades_html = trades_df.to_html(classes='table table-striped table-hover table-sm trades-table', index=False, border=0, justify='right')

    # Consolidate metadata/info access
    strategy_name_val = results.get("strategy_name", results.get("strategy_info", {}).get("name", results.get("metadata", {}).get("strategy_name", "N/A")))
    params_val = results.get("params_used_this_run", results.get("parameters_used", results.get("strategy_info", {}).get("params", results.get("metadata", {}).get("parameters_used", {}))))

    context_params = {
        "report_title": f"{strategy_name_val} Performance Report",
        "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics_table": metrics_html,
        "equity_curve_plot": equity_curve_html,
        "trades_table": trades_html,
        "strategy_name": strategy_name_val,
        "parameters": params_val,
        "input_file": results.get("metadata", {}).get("input_file", "N/A"), # Assuming metadata might still be a sub-dict
        "symbol": results.get("symbol", "N/A"),
        "timeframe": results.get("timeframe", "N/A"),
        "run_id": results.get("run_id", "N/A"), # This is the pipeline RUN_ID_GLOBAL
        "custom_summary": results.get("custom_summary", "")
    }
    try:
        template = env.get_template(template_name) # Ensure template_name is correct
        html_content = template.render(context_params)
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"HTML report (single backtest) generated successfully: {output_html_path}")
    except Exception as e:
        logger.error(f"Error rendering or writing HTML report for single backtest: {e}", exc_info=True)


# ========= MongoDB Analytics Reporting Functions =========

def get_mongo_client_for_reporting() -> Optional[MongoClient]:
    client = MongoManager.get_client()
    if client: return client
    logger.warning("MongoManager client not available, attempting direct connection for reporting script.")
    try:
        client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=config.MONGO_TIMEOUT_MS)
        client.admin.command('ping')
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB directly for reporting: {e}", exc_info=True)
        return None

def load_contextual_strategy_summary_from_mongo(
    client: MongoClient, symbol: str, timeframe: str, session: Optional[str] = None,
    day: Optional[str] = None, is_expiry: Optional[bool] = None,
    market_condition: Optional[str] = None, volatility_status: Optional[str] = None,
    limit: int = 10, db_name: str = config.MONGO_DB_NAME,
    collection_name: str = config.MONGO_COLLECTION_BACKTEST_RESULTS
) -> List[Dict[str, Any]]:
    if not client:
        logger.error("MongoDB client not provided to load_contextual_strategy_summary_from_mongo.")
        return []
    db = client[db_name]
    collection = db[collection_name]
    query_filter = {
        "symbol": symbol.upper(), "timeframe": timeframe,
        "performance_score": {"$ne": None, "$exists": True},
        "parameters_used_this_run": {"$ne": None, "$exists": True} # Prefer this key
    }
    if session and session.lower() != "allday": query_filter["session"] = session
    if day and day.lower() != "anyday": query_filter["day"] = day
    if is_expiry is not None: query_filter["is_expiry"] = is_expiry
    if market_condition and market_condition.lower() != "anyregime": query_filter["market_condition"] = market_condition
    if volatility_status and volatility_status.lower() != "anyvolatility":
        query_filter["custom_data.volatility_status_from_data"] = volatility_status
    
    logger.debug(f"Contextual summary query: {query_filter} on {collection_name}")
    try:
        pipeline = [
            {"$match": query_filter},
            {"$sort": {"performance_score": -1}},
            {"$group": {
                "_id": { # Group by strategy and its specific parameters for uniqueness
                    "strategy_name": "$strategy_name",
                    "params_str": {"$toString": "$params_used_this_run"} # Use params_used_this_run
                },
                "doc": {"$first": "$$ROOT"}
            }},
            {"$replaceRoot": {"newRoot": "$doc"}},
            {"$sort": {"performance_score": -1}},
            {"$limit": limit},
            {"$project": { # Ensure all desired fields are projected
                "_id": 0, "doc_id": "$_id", "strategy_name": 1, 
                "parameters_used": "$params_used_this_run", # Use the correct field name
                "performance_score": 1, "total_pnl": 1, "win_rate": 1, "trade_count": 1,
                "profit_factor": 1, "max_drawdown": 1, "symbol": 1, "timeframe": 1,
                "market_condition": 1, "session": 1, "day": 1, "is_expiry": 1,
                "custom_data": 1, "optuna_study_name": 1, "optuna_trial_number":1, "run_id": 1
            }}
        ]
        results_list = list(collection.aggregate(pipeline))
        # Convert MongoDB _id in doc_id to string if not already
        for item in results_list:
            if isinstance(item.get("doc_id"), ObjectId):
                item["doc_id"] = str(item["doc_id"])
        logger.info(f"Found {len(results_list)} contextual summary entries (limit {limit}).")
        return results_list
    except Exception as e:
        logger.error(f"Error in load_contextual_strategy_summary_from_mongo: {e}", exc_info=True)
        return []

def get_specific_run_details_from_mongo(
    client: MongoClient, run_document_id: str, db_name: str = config.MONGO_DB_NAME,
    collection_name: str = config.MONGO_COLLECTION_BACKTEST_RESULTS
) -> Optional[Dict[str, Any]]:
    if not client: return None
    if not run_document_id: return None
    db = client[db_name]
    collection = db[collection_name]
    try:
        object_id = ObjectId(run_document_id)
        query_filter = {"_id": object_id}
        run_details = collection.find_one(query_filter)
        if run_details:
            if '_id' in run_details and isinstance(run_details['_id'], ObjectId):
                run_details['_id'] = str(run_details['_id'])
        return run_details
    except Exception as e: # Catch specific bson.errors.InvalidId if needed
        logger.error(f"Error getting specific run details (ID: {run_document_id}): {e}", exc_info=True)
        return None

def get_optuna_trials_data_from_mongo(
    client: MongoClient, optuna_study_name: str, symbol: Optional[str] = None,
    timeframe: Optional[str] = None, db_name: str = config.MONGO_DB_NAME,
    collection_name: str = config.MONGO_COLLECTION_BACKTEST_RESULTS
) -> List[Dict[str, Any]]:
    if not client or not optuna_study_name: return []
    db = client[db_name]
    collection = db[collection_name]
    query_filter = {
        "optuna_study_name": optuna_study_name,
        "optuna_trial_number": {"$exists": True, "$ne": None}
    }
    if symbol: query_filter["symbol"] = symbol.upper()
    if timeframe: query_filter["timeframe"] = timeframe
    try:
        results_cursor = collection.find(query_filter).sort("optuna_trial_number", ASCENDING)
        trials_data = [{
            "trial_number": doc.get("optuna_trial_number"),
            "parameters_used": doc.get("params_used_this_run", doc.get("parameters_used")), # Check both
            "performance_score": doc.get("performance_score"),
            "total_pnl": doc.get("total_pnl"),
            "win_rate": doc.get("win_rate"),
            "trade_count": doc.get("trade_count"),
            "_id": str(doc.get("_id")) ,
            "run_id": doc.get("run_id") # Include run_id if needed
        } for doc in results_cursor]
        return trials_data
    except Exception as e:
        logger.error(f"Error getting Optuna trials (Study: {optuna_study_name}): {e}", exc_info=True)
        return []

def get_distinct_optuna_study_names(
    client: MongoClient, symbol: Optional[str] = None, timeframe: Optional[str] = None,
    db_name: str = config.MONGO_DB_NAME, collection_name: str = config.MONGO_COLLECTION_BACKTEST_RESULTS
) -> List[str]:
    if not client: return []
    db = client[db_name]
    collection = db[collection_name]
    query_filter = {"optuna_study_name": {"$exists": True, "$ne": None}}
    if symbol: query_filter["symbol"] = symbol.upper()
    if timeframe: query_filter["timeframe"] = timeframe
    try:
        return sorted(collection.distinct("optuna_study_name", query_filter))
    except Exception as e:
        logger.error(f"Error getting distinct Optuna study names: {e}", exc_info=True)
        return []

# # --- NEW FUNCTION FOR PIPELINE RUN SUMMARY ---
# def get_full_pipeline_run_summary_from_mongo(
#     client: MongoClient,
#     pipeline_run_id: str, 
#     db_name: str = config.MONGO_DB_NAME,
#     collection_name: str = config.MONGO_COLLECTION_BACKTEST_RESULTS
# ) -> Dict[str, Any]:
#     if not client:
#         logger.error("MongoDB client not provided for get_full_pipeline_run_summary_from_mongo.")
#         return {"error": "DB client missing", "pipeline_run_id": pipeline_run_id}
#     if not pipeline_run_id:
#         logger.error("pipeline_run_id not provided for get_full_pipeline_run_summary_from_mongo.")
#         return {"error": "Pipeline Run ID is required"}

#     db = client[db_name]
#     collection = db[collection_name]
#     query_filter = {"run_id": pipeline_run_id} 
    
#     logger.info(f"Querying for full pipeline run summary (pipeline_run_id: '{pipeline_run_id}') on collection '{collection_name}'")
    
#     all_docs_for_run = list(collection.find(query_filter))
#     logger.info(f"Found {len(all_docs_for_run)} MongoDB documents for pipeline_run_id '{pipeline_run_id}'.")

#     if not all_docs_for_run:
#         return {
#             "pipeline_run_id": pipeline_run_id, 
#             "message": "No data found for this pipeline run in MongoDB.", 
#             "individual_strategy_results": [], 
#             "optuna_studies_overview": [] 
#         }

#     summary_output = {
#         "pipeline_run_id": pipeline_run_id,
#         "total_db_documents_for_run": len(all_docs_for_run),
#         "individual_strategy_results": [], 
#         "optuna_studies_overview_map": defaultdict(lambda: {
#             "study_name": "", "trial_count": 0, "best_score": -float('inf'), 
#             "worst_score": float('inf'), "best_trial_mongodb_id": None, 
#             "best_trial_params": None, "best_trial_number": None,
#             "symbol": None, "timeframe": None
#         })
#     }

#     for doc in all_docs_for_run:
#         doc_mongodb_id = str(doc.get("_id"))
#         strategy_name = doc.get("strategy_name")
#         timeframe = doc.get("timeframe")
#         symbol = doc.get("symbol")
#         performance_score = doc.get("performance_score")
        
#         is_optuna_trial = doc.get("optuna_study_name") and doc.get("optuna_trial_number") is not None

#         if not is_optuna_trial:
#             summary_output["individual_strategy_results"].append({
#                 "doc_id": doc_mongodb_id, # This is the MongoDB _id
#                 "strategy_name": strategy_name,
#                 "symbol": symbol,
#                 "timeframe": timeframe,
#                 "performance_score": performance_score,
#                 "total_pnl": doc.get("total_pnl"),
#                 "trade_count": doc.get("trade_count"),
#                 "win_rate": doc.get("win_rate"),
#                 "exit_reasons": doc.get("custom_data", {}).get("exit_reasons_summary", {}),
#                 "parameters_used": doc.get("params_used_this_run", doc.get("parameters_used"))
#             })
#         else: 
#             optuna_study_name = doc.get("optuna_study_name")
#             trial_score = performance_score if isinstance(performance_score, (int, float)) else -float('inf')
#             trial_number = doc.get("optuna_trial_number")

#             study_summary = summary_output["optuna_studies_overview_map"][optuna_study_name]
#             study_summary["study_name"] = optuna_study_name 
#             study_summary["trial_count"] += 1
#             if study_summary["symbol"] is None and symbol: study_summary["symbol"] = symbol
#             if study_summary["timeframe"] is None and timeframe: study_summary["timeframe"] = timeframe

#             if trial_score > study_summary["best_score"]:
#                 study_summary["best_score"] = trial_score
#                 study_summary["best_trial_mongodb_id"] = doc_mongodb_id
#                 study_summary["best_trial_params"] = doc.get("params_used_this_run", doc.get("parameters_used"))
#                 study_summary["best_trial_number"] = trial_number
            
#             current_worst = study_summary.get("worst_score", float('inf'))
#             study_summary["worst_score"] = min(current_worst, trial_score)
    
#     optuna_list_final = []
#     for overview_data in summary_output["optuna_studies_overview_map"].values():
#         if overview_data["worst_score"] == float('inf'): 
#             overview_data["worst_score"] = None if overview_data["best_score"] == -float('inf') else overview_data["best_score"]
#         optuna_list_final.append(overview_data)
    
#     summary_output["optuna_studies_overview"] = sorted(optuna_list_final, key=lambda x: x.get('best_score', -float('inf')), reverse=True)
#     del summary_output["optuna_studies_overview_map"]

#     logger.info(f"Generated summary for pipeline_run_id '{pipeline_run_id}'. Individual strategies: {len(summary_output['individual_strategy_results'])}, Optuna studies: {len(summary_output['optuna_studies_overview'])}")
#     return summary_output

def get_full_pipeline_run_summary_from_mongo(
    client: MongoClient,
    pipeline_run_id: str,
    db_name: str = config.MONGO_DB_NAME,
    collection_name: str = config.MONGO_COLLECTION_BACKTEST_RESULTS
) -> Dict[str, Any]:
    if not client:
        logger.error("MongoDB client not provided for get_full_pipeline_run_summary_from_mongo.")
        return {"error": "DB client missing", "pipeline_run_id": pipeline_run_id}
    if not pipeline_run_id:
        logger.error("pipeline_run_id not provided for get_full_pipeline_run_summary_from_mongo.")
        return {"error": "Pipeline Run ID is required"}

    db = client[db_name]
    collection = db[collection_name]
    query_filter = {"run_id": pipeline_run_id}

    all_docs_for_run = list(collection.find(query_filter))

    if not all_docs_for_run:
        return {
            "pipeline_run_id": pipeline_run_id,
            "message": "No data found for this pipeline run in MongoDB.",
            "individual_strategy_results": [],
            "optuna_studies_overview": []
        }

    summary_output = {
        "pipeline_run_id": pipeline_run_id,
        "total_db_documents_for_run": len(all_docs_for_run),
        "individual_strategy_results": [],
        "optuna_studies_overview_map": defaultdict(lambda: {
            "study_name": "", "trial_count": 0, 
            "best_score": -float('inf'),
            "worst_score": float('inf'), 
            "best_trial_mongodb_id": None,
            "best_trial_params": None, 
            "best_trial_number": None,
            "symbol": None, 
            "timeframe": None,
            "session": None, 
            "day": None, 
            "is_expiry": None,
            "market_condition": None, 
            "volatility_status": None
        })
    }

    for doc in all_docs_for_run:
        doc_mongodb_id = str(doc.get("_id"))
        strategy_name = doc.get("strategy_name")
        timeframe = doc.get("timeframe")
        symbol = doc.get("symbol")
        performance_score = doc.get("performance_score")

        is_optuna_trial = doc.get("optuna_study_name") and doc.get("optuna_trial_number") is not None

        if not is_optuna_trial:
            summary_output["individual_strategy_results"].append({
                "doc_id": doc_mongodb_id,
                "strategy_name": strategy_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "performance_score": performance_score,
                "total_pnl": doc.get("total_pnl"),
                "trade_count": doc.get("trade_count"),
                "win_rate": doc.get("win_rate"),
                "exit_reasons": doc.get("custom_data", {}).get("exit_reasons_summary", {}),
                "parameters_used": doc.get("params_used_this_run", doc.get("parameters_used")),
                "run_id": doc.get("run_id")
            })
        else:
            optuna_study_name = doc.get("optuna_study_name")
            trial_score = performance_score if isinstance(performance_score, (int, float)) else -float('inf')
            trial_number = doc.get("optuna_trial_number")

            study_summary = summary_output["optuna_studies_overview_map"][optuna_study_name]
            study_summary["study_name"] = optuna_study_name
            study_summary["trial_count"] += 1
            if study_summary["symbol"] is None and symbol: study_summary["symbol"] = symbol
            if study_summary["timeframe"] is None and timeframe: study_summary["timeframe"] = timeframe
            if study_summary["session"] is None and doc.get("session"): study_summary["session"] = doc.get("session")
            if study_summary["day"] is None and doc.get("day"): study_summary["day"] = doc.get("day")
            if study_summary["is_expiry"] is None and "is_expiry" in doc: study_summary["is_expiry"] = doc.get("is_expiry")
            if study_summary["market_condition"] is None and doc.get("market_condition"): study_summary["market_condition"] = doc.get("market_condition")
            if study_summary["volatility_status"] is None:
                vs = doc.get("custom_data", {}).get("volatility_status_from_data")
                if vs: study_summary["volatility_status"] = vs

            if trial_score > study_summary["best_score"]:
                study_summary["best_score"] = trial_score
                study_summary["best_trial_mongodb_id"] = doc_mongodb_id
                study_summary["best_trial_params"] = doc.get("params_used_this_run", doc.get("parameters_used"))
                study_summary["best_trial_number"] = trial_number

            current_worst = study_summary.get("worst_score", float('inf'))
            study_summary["worst_score"] = min(current_worst, trial_score)

    optuna_list_final = []
    for overview_data in summary_output["optuna_studies_overview_map"].values():
        if overview_data["worst_score"] == float('inf'):
            overview_data["worst_score"] = None if overview_data["best_score"] == -float('inf') else overview_data["best_score"]
        optuna_list_final.append(overview_data)

    summary_output["optuna_studies_overview"] = sorted(optuna_list_final, key=lambda x: x.get('best_score', -float('inf')), reverse=True)
    del summary_output["optuna_studies_overview_map"]

    return summary_output

# CLI functionality
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reporting module CLI")
    parser.add_argument("--test-pipeline-summary", type=str, metavar="PIPELINE_RUN_ID",
                        help="Test get_full_pipeline_run_summary_from_mongo with a specific pipeline_run_id.")
    # Add other CLI commands and arguments as in your original file if needed for query/generate_report
    
    args = parser.parse_args()
    client = None
    try:
        client = get_mongo_client_for_reporting()
        if not client:
            logger.error("Failed to get MongoDB client for CLI execution.")
            sys.exit(1)

        if args.test_pipeline_summary:
            logger.info(f"CLI: Testing pipeline summary for RUN_ID: {args.test_pipeline_summary}")
            summary = get_full_pipeline_run_summary_from_mongo(client, args.test_pipeline_summary)
            print(json.dumps(summary, indent=2, default=str)) # Use default=str for any non-serializable types
        else:
            logger.info("Reporting module CLI. Use --help for options. Example: --test-pipeline-summary <ID>")
            # You can add your other CLI command handling (query, generate_report) here
            # For example:
            # if args.command == "query": ...
            # elif args.command == "generate_report": ...

    except Exception as e:
        logger.error(f"Error in reporting.py CLI: {e}", exc_info=True)
    finally:
        if client:
            client.close()
            logger.info("CLI: MongoDB connection closed.")
