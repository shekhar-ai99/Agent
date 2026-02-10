# # # app/agentic_core.py

# # import pandas as pd
# # import logging
# # from typing import Dict, Tuple, Callable, Optional
# # from app.config import config
# # from app.strategies import strategy_factories
# # from datetime import datetime


# # from pymongo import MongoClient


# # logger = logging.getLogger(__name__)
# # if not logger.hasHandlers():
# #     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # class RuleBasedAgent:
# #     def __init__(self):
# #         self.strategy_mapping: Dict[str, Optional[Tuple[str, Callable]]] = {
# #     "Trending": ("SuperTrend_ADX", strategy_factories.get("SuperTrend_ADX")),
# #     # --- Use BB strategy for Ranging ---
# #     "Ranging": ("BB_MeanReversion", strategy_factories.get("BB_MeanReversion")),
# #     # --- Decide what to do with Momentum ---
# #     "Momentum": ("EMA_Crossover", strategy_factories.get("EMA_Crossover")), # Keep EMA? Or use another? Or None?
# #     "Unknown": None, # Or assign a default strategy?
# # }

# #         self.parameter_mapping: Dict[str, Dict[str, float]] = {
# #             "Trending": {
# #                 "sl_mult": getattr(config, "TREND_SL_ATR_MULT", config.DEFAULT_SL_ATR_MULT * 1.2),
# #                 "tp_mult": getattr(config, "TREND_TP_ATR_MULT", config.DEFAULT_TP_ATR_MULT * 1.5),
# #             },
# #             "Ranging": {
# #                 "sl_mult": getattr(config, "RANGE_SL_ATR_MULT", config.DEFAULT_SL_ATR_MULT * 0.8),
# #                 "tp_mult": getattr(config, "RANGE_TP_ATR_MULT", config.DEFAULT_TP_ATR_MULT * 0.8),
# #             },
# #             "Momentum": {
# #                 "sl_mult": config.DEFAULT_SL_ATR_MULT,
# #                 "tp_mult": config.DEFAULT_TP_ATR_MULT,
# #             },
# #             "Unknown": {
# #                 "sl_mult": config.DEFAULT_SL_ATR_MULT,
# #                 "tp_mult": config.DEFAULT_TP_ATR_MULT,
# #             }
# #         }

# #         for regime, pair in self.strategy_mapping.items():
# #             if pair is None:
# #                 logger.warning(f"No strategy for regime '{regime}'.")
# #             else:
# #                 name, func = pair
# #                 if not callable(func):
# #                     logger.error(f"Strategy '{name}' for regime '{regime}' is not callable.")
# #                     self.strategy_mapping[regime] = None

# #         logger.info("RuleBasedAgent initialized.")

# #     def decide(self, current_row: pd.Series, data_history: pd.DataFrame = None) -> Tuple[str, float, float, float, Optional[str]]:
# #         regime = current_row.get('regime', 'Unknown')
# #         symbol = current_row.get('symbol', 'nifty').lower()
# #         timeframe = current_row.get('timeframe', '5min')
# #         dt: datetime = current_row.name if isinstance(current_row.name, datetime) else datetime.now()
        
# #         context = {
# #             "day": dt.strftime("%A"),
# #             "session": self._infer_session(dt),
# #             "is_expiry": current_row.get('is_expiry', False),
# #             "timeframe": timeframe,
# #             "symbol": symbol,
# #             "market_condition": regime
# #         }

# #         strategy_name, best_params = RuleBasedAgent.get_top_strategy_for_context(context)
# #         sl_mult = best_params.get("sl_mult", config.DEFAULT_SL_ATR_MULT)
# #         tp_mult = best_params.get("tp_mult", config.DEFAULT_TP_ATR_MULT)
# #         tsl_mult = best_params.get("tsl_mult", config.DEFAULT_TSL_ATR_MULT)
# #         signal = "hold"

# #         if strategy_name and strategy_name in strategy_factories:
# #             strategy_func = strategy_factories[strategy_name](**best_params)
# #             try:
# #                 signal = strategy_func(current_row, data_history)
# #                 if signal not in ['buy_potential', 'sell_potential', 'hold']:
# #                     logger.warning(f"Strategy {strategy_name} returned invalid signal: '{signal}'")
# #                     signal = "hold"
# #             except Exception as e:
# #                 logger.error(f"Error in strategy '{strategy_name}': {e}")
# #                 signal = "hold"
# #         else:
# #             logger.warning(f"No valid strategy found for context: {context}")

# #         return signal, sl_mult, tp_mult, tsl_mult, strategy_name

# #     def _infer_session(self, ts: datetime) -> str:
# #         if ts.time() <= datetime.strptime("10:59", "%H:%M").time():
# #             return "Morning"
# #         elif ts.time() <= datetime.strptime("13:29", "%H:%M").time():
# #             return "Midday"
# #         return "Afternoon"

# #     def get_top_strategy_for_context(context: Dict, limit: int = 1) -> Tuple[Optional[str], Dict]:
# #         """
# #         Retrieves the top-performing strategy and its tuned parameters based on the given context,
# #         including market regime (trending, ranging, choppy).

# #         Args:
# #             context (Dict): {
# #                 "day": "Monday",
# #                 "session": "Morning",
# #                 "is_expiry": True,
# #                 "timeframe": "5min",
# #                 "symbol": "nifty",
# #                 "market_condition": "trending"
# #             }
# #             limit (int): How many top strategies to retrieve

# #         Returns:
# #             Tuple[str | None, Dict]: (strategy_name, best_params) or (None, {})
# #         """
# #         client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)

# #         try:
# #             db = client[config.MONGO_DB_NAME]
# #             perf_collection = db[config.MONGO_COLLECTION_BACKTEST_RESULTS]
# #             param_collection = db[config.MONGO_COLLECTION_TUNED_PARAMS]

# #             # === Build Filter Query for Performance Ranking ===
# #             filter_query = {
# #                 "day": context.get("day"),
# #                 "session": context.get("session"),
# #                 "is_expiry": context.get("is_expiry"),
# #                 "timeframe": context.get("timeframe"),
# #                 "symbol": context.get("symbol"),
# #                 "market_condition": context.get("market_condition")
# #             }

# #             filter_query = {k: v for k, v in filter_query.items() if v is not None}

# #             top_strats = perf_collection.find(filter_query).sort("performance_score", -1).limit(limit)
# #             top_strat_docs = list(top_strats)

# #             if not top_strat_docs:
# #                 logger.warning(f"⚠️ No top strategies found for context: {context}")
# #                 return None, {}

# #             best_doc = top_strat_docs[0]
# #             strategy_name = best_doc.get("strategy")

# #             if not strategy_name:
# #                 logger.error("❌ Strategy field missing in top strategy document.")
# #                 return None, {}

# #             # === Retrieve Best Tuned Parameters for This Context ===
# #             param_query = filter_query.copy()
# #             param_query["strategy"] = strategy_name
# #             param_doc = param_collection.find_one(param_query)

# #             best_params = param_doc.get("best_params", {}) if param_doc else {}

# #             logger.info(f"✅ Selected strategy: {strategy_name} | Params: {best_params} | Context: {context}")
# #             return strategy_name, best_params

# #         except Exception as e:
# #             logger.error(f"❌ Error in get_top_strategy_for_context: {e}", exc_info=True)
# #             return None, {}

# #         finally:
# #             client.close()
# # ##############################################
# # app/agentic_core.py
# # app/agentic_core.py

# import pandas as pd
# import logging
# from typing import Dict, Tuple, Callable, Optional, Any
# from app.config import config # Ensure config is imported
# from app.strategies import strategy_factories
# from datetime import datetime, time

# from pymongo import MongoClient, DESCENDING # Import DESCENDING for sorting

# # load_contextual_strategy_summary_from_mongo is no longer needed here
# # as the agent will directly query the tuned_parameters collection.

# logger = logging.getLogger(__name__)
# if not logger.hasHandlers():
#     log_level_from_config = getattr(config, "LOG_LEVEL", "INFO")
#     log_format_from_config = getattr(config, "LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     logging.basicConfig(level=log_level_from_config, format=log_format_from_config)

# class RuleBasedAgent:
#     def __init__(self, mongo_client: MongoClient):
#         """
#         Initializes the RuleBasedAgent.
#         Parameter mode for SL/TP/TSL is determined by config.AGENT_USE_TUNED_PARAMS.
#         Args:
#             mongo_client: An active MongoClient instance.
#         """
#         self.mongo_client = mongo_client
#         try:
#             self.db = self.mongo_client[config.MONGO_DB_NAME]
#             self.db.command('ping') # Test connection
#             logger.info(f"Successfully connected to MongoDB database: {config.MONGO_DB_NAME}")
#         except Exception as e:
#             logger.error(f"Failed to connect to MongoDB or database '{config.MONGO_DB_NAME}': {e}", exc_info=True)
#             raise ConnectionError(f"RuleBasedAgent: Failed to connect to MongoDB: {e}") from e

#         # Set parameter_mode based on the global config flag
#         if getattr(config, 'AGENT_USE_TUNED_PARAMS', True): # Default to True (tuned) if flag isn't in config
#             self.parameter_mode = "tuned"
#         else:
#             self.parameter_mode = "default"
        
#         logger.info(f"RuleBasedAgent initialized. SL/TP/TSL parameter_mode: '{self.parameter_mode}' (derived from config.AGENT_USE_TUNED_PARAMS = {getattr(config, 'AGENT_USE_TUNED_PARAMS', True)})")
            
#         # These mappings are currently not directly used by the primary MongoDB-driven `decide` logic
#         # but could be used for an ultimate fallback if the DB lookups yield nothing and
#         # such fallback logic is implemented in get_top_strategy_for_context or decide.
#         self.default_strategy_mapping_by_regime: Dict[str, Optional[Tuple[str, Callable]]] = {
#             "Trending": ("SuperTrend_ADX", strategy_factories.get("SuperTrend_ADX")),
#             "Ranging": ("BB_MeanReversion", strategy_factories.get("BB_MeanReversion")),
#             "Momentum": ("EMA_Crossover", strategy_factories.get("EMA_Crossover")),
#             "Unknown": None,
#         }
#         self.default_parameter_mapping_by_regime: Dict[str, Dict[str, float]] = {
#             "Trending": {
#                 "sl_mult": getattr(config, "TREND_SL_ATR_MULT", config.DEFAULT_SL_ATR_MULT * 1.2),
#                 "tp_mult": getattr(config, "TREND_TP_ATR_MULT", config.DEFAULT_TP_ATR_MULT * 1.5),
#             },
#             "Ranging": {
#                 "sl_mult": getattr(config, "RANGE_SL_ATR_MULT", config.DEFAULT_SL_ATR_MULT * 0.8),
#                 "tp_mult": getattr(config, "RANGE_TP_ATR_MULT", config.DEFAULT_TP_ATR_MULT * 0.8),
#             },
#             "Momentum": {
#                 "sl_mult": config.DEFAULT_SL_ATR_MULT,
#                 "tp_mult": config.DEFAULT_TP_ATR_MULT,
#             },
#             "Unknown": {
#                 "sl_mult": config.DEFAULT_SL_ATR_MULT,
#                 "tp_mult": config.DEFAULT_TP_ATR_MULT,
#             }
#         }

#         # Validate default strategy factories on init
#         for regime, pair in self.default_strategy_mapping_by_regime.items():
#             if pair is None:
#                 logger.debug(f"No default strategy defined for regime '{regime}'.")
#             else:
#                 name, func = pair
#                 if not callable(func):
#                     logger.error(f"Default strategy '{name}' for regime '{regime}' is not callable. Disabling default for this regime.")
#                     self.default_strategy_mapping_by_regime[regime] = None
#         logger.info("RuleBasedAgent initialization complete.")


#     def _infer_session(self, ts: datetime) -> str:
#         """Infers trading session from timestamp."""
#         current_time = ts.time()
#         if current_time >= time(9, 15) and current_time <= time(10, 59):
#             return "Morning"
#         elif current_time >= time(11, 0) and current_time <= time(13, 29):
#             return "Midday"
#         elif current_time >= time(13, 30) and current_time <= time(15, 30): 
#             return "Afternoon"
#         return "OffMarket"

#     def get_top_strategy_for_context(self, context: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
#         """
#         Retrieves the top-performing strategy and its comprehensive best_params 
#         directly from the MONGO_COLLECTION_TUNED_PARAMS based on the given context.
#         The collection should store documents representing the single best strategy_name
#         and its full best_params (including SL/TP/TSL and core logic params)
#         for each unique context, ranked by 'achieved_performance_score'.
#         """
#         strategy_name = None
#         comprehensive_best_params: Dict[str, Any] = {} # Holds all params for the strategy

#         try:
#             # Ensure MONGO_COLLECTION_TUNED_PARAMS is defined in your config
#             tuned_params_collection_name = getattr(config, "MONGO_COLLECTION_TUNED_PARAMS", "tuned_parameters")
#             tuned_params_collection = self.db[tuned_params_collection_name]

#             # Build the filter query directly from the context.
#             # These keys must exactly match the field names in your tuned_parameters documents.
#             filter_query: Dict[str, Any] = {
#                 "symbol": str(context.get("symbol")).upper(), # Ensure consistency
#                 "timeframe": str(context.get("timeframe"))
#             }
#             # Add other context fields that are part of the unique key for a tuned entry
#             if context.get("day") is not None: filter_query["day"] = str(context.get("day"))
#             if context.get("session") is not None: filter_query["session"] = str(context.get("session"))
#             if context.get("is_expiry") is not None: filter_query["is_expiry"] = context.get("is_expiry") # bool
#             if context.get("market_condition") is not None: filter_query["market_condition"] = str(context.get("market_condition"))
#             if context.get("volatility_status_from_data") is not None:
#                 # Assuming this field is directly available in tuned_parameters docs, not nested under 'custom_data'
#                 filter_query["volatility_status_from_data"] = str(context.get("volatility_status_from_data"))
            
#             # Ensure we are looking for documents that have a score and params
#             filter_query["achieved_performance_score"] = {"$exists": True, "$ne": None}
#             filter_query["best_params"] = {"$exists": True, "$ne": None} # The object containing all params

#             logger.debug(f"Querying '{tuned_params_collection_name}' with filter: {filter_query} to find best strategy config for context.")
            
#             # Find the single document with the highest 'achieved_performance_score' matching the context.
#             # This document itself contains the strategy_name and its full best_params.
#             top_entry_cursor = tuned_params_collection.find(filter_query).sort("achieved_performance_score", DESCENDING).limit(1)
            
#             top_entry_list = list(top_entry_cursor) # Evaluate cursor to get the document

#             if top_entry_list:
#                 top_entry_doc = top_entry_list[0]
#                 strategy_name = top_entry_doc.get("strategy_name")
#                 comprehensive_best_params = top_entry_doc.get("best_params", {}) # This should be a dict
                
#                 if not strategy_name:
#                     logger.warning(f"⚠️ Document found in '{tuned_params_collection_name}' for context {context} is missing 'strategy_name'. Doc ID: {top_entry_doc.get('_id')}")
#                     return None, {} # Cannot proceed without strategy name
#                 if not comprehensive_best_params or not isinstance(comprehensive_best_params, dict):
#                     logger.warning(f"⚠️ Document found for strategy '{strategy_name}' in context {context} is missing 'best_params' field, it's empty, or not a dictionary. Doc ID: {top_entry_doc.get('_id')}")
#                     comprehensive_best_params = {} # Ensure it's a dict for downstream safety
                
#                 logger.info(f"Found top strategy config in '{tuned_params_collection_name}' for context: '{strategy_name}' with score {top_entry_doc.get('achieved_performance_score')}. All Params: {comprehensive_best_params}")
#             else:
#                 logger.warning(f"⚠️ No matching & complete entry found in '{tuned_params_collection_name}' for the exact context: {context}. No specific strategy/params determined by this method.")
#                 return None, {} # No specific tuned strategy found for this exact context

#             return strategy_name, comprehensive_best_params

#         except Exception as e:
#             logger.error(f"❌ Error querying '{tuned_params_collection_name}' for context {context}: {e}", exc_info=True)
#             return None, {}

#     def decide(self, current_row: pd.Series, data_history: pd.DataFrame = None) -> Tuple[str, float, float, Optional[float], Optional[str]]:
#         regime = current_row.get('regime', 'Unknown')
#         symbol = current_row.get('symbol', getattr(config, "DEFAULT_SYMBOL", "NIFTY")).lower()
#         timeframe = current_row.get('timeframe', getattr(config, "DEFAULT_TIMEFRAME", "5min"))
        
#         dt_index = current_row.name
#         if not isinstance(dt_index, datetime):
#             try: dt_index = pd.to_datetime(dt_index)
#             except (ValueError, TypeError):
#                 logger.warning(f"current_row.name '{dt_index}' is not a datetime. Using current time for session inference.")
#                 dt_index = datetime.now()
        
#         context: Dict[str, Any] = {
#             "symbol": symbol.upper(),
#             "timeframe": timeframe,
#             "day": dt_index.strftime("%A"),
#             "session": self._infer_session(dt_index),
#             "is_expiry": bool(current_row.get('is_expiry', False)),
#             "market_condition": str(regime),
#             "volatility_status_from_data": str(current_row.get('volatility_status', 'Unknown'))
#         }
#         logger.info(f"Decision context: {context}, Agent SL/TP/TSL Mode: '{self.parameter_mode}'")

#         # Fetches strategy_name and its comprehensive set of best_params (including tuned SL/TP/TSL etc.)
#         strategy_name, comprehensive_best_params = self.get_top_strategy_for_context(context)
        
#         signal = "hold"
#         final_strategy_name_used = None
        
#         sl_mult: float
#         tp_mult: float
#         tsl_mult: Optional[float] = None

#         # Determine final SL/TP/TSL multipliers based on agent's parameter_mode
#         if self.parameter_mode == "tuned":
#             # Use SL/TP/TSL from the comprehensive_best_params if available, else global defaults
#             sl_mult = comprehensive_best_params.get("sl_mult", config.DEFAULT_SL_ATR_MULT)
#             tp_mult = comprehensive_best_params.get("tp_mult", config.DEFAULT_TP_ATR_MULT)
#             enable_tsl = comprehensive_best_params.get("enable_tsl_mult", getattr(config, "DEFAULT_ENABLE_TSL", False))
#             if enable_tsl:
#                 tsl_mult = comprehensive_best_params.get("tsl_mult", config.DEFAULT_TSL_ATR_MULT)
#             logger.debug(f"Using 'tuned' SL/TP/TSL settings for strategy '{strategy_name or 'N/A'}'. Fetched values: SL={comprehensive_best_params.get('sl_mult')}, TP={comprehensive_best_params.get('tp_mult')}, EnableTSL={comprehensive_best_params.get('enable_tsl_mult')}, TSL_Val={comprehensive_best_params.get('tsl_mult')}. Final: SL={sl_mult}, TP={tp_mult}, TSL={tsl_mult}")
        
#         elif self.parameter_mode == "default":
#             # Use global defaults from config for SL/TP/TSL
#             sl_mult = config.DEFAULT_SL_ATR_MULT
#             tp_mult = config.DEFAULT_TP_ATR_MULT
#             if getattr(config, "DEFAULT_ENABLE_TSL", False):
#                 tsl_mult = config.DEFAULT_TSL_ATR_MULT
#             logger.debug(f"Using global 'default' SL/TP/TSL settings for strategy '{strategy_name or 'N/A'}': SL={sl_mult}, TP={tp_mult}, EnableTSL={getattr(config, 'DEFAULT_ENABLE_TSL', False)}, TSL={tsl_mult}")
        
#         else: # Should not be reached if __init__ validated parameter_mode
#             logger.error(f"Invalid self.parameter_mode: '{self.parameter_mode}'. Defaulting SL/TP/TSL to global config values.")
#             sl_mult = config.DEFAULT_SL_ATR_MULT
#             tp_mult = config.DEFAULT_TP_ATR_MULT
#             if getattr(config, "DEFAULT_ENABLE_TSL", False):
#                 tsl_mult = config.DEFAULT_TSL_ATR_MULT
        
#         # Strategy Execution Part
#         if strategy_name and strategy_name in strategy_factories:
#             strategy_func_factory = strategy_factories[strategy_name]
#             if not callable(strategy_func_factory):
#                 logger.error(f"Strategy factory for '{strategy_name}' is not callable.")
#             else:
#                 try:
#                     # Instantiate strategy with its core parameters from comprehensive_best_params.
#                     # This dict should contain all necessary strategy-specific inputs (e.g., periods, factors).
#                     # If comprehensive_best_params is empty (e.g., DB lookup failed), strategy relies on its own defaults.
#                     strategy_instance = strategy_func_factory(**comprehensive_best_params) 
#                     current_signal = strategy_instance(current_row, data_history)

#                     if current_signal in ['buy_potential', 'sell_potential', 'hold']:
#                         signal = current_signal
#                         final_strategy_name_used = strategy_name
#                         logger.info(f"Strategy '{strategy_name}' (Core params from DB: {bool(comprehensive_best_params)}) generated signal: '{signal}'")
#                     else:
#                         logger.warning(f"Strategy '{strategy_name}' returned invalid signal: '{current_signal}'. Defaulting to 'hold'.")
#                         signal = "hold"
#                 except Exception as e:
#                     logger.error(f"Error executing strategy '{strategy_name}': {e}", exc_info=True)
#                     signal = "hold"
#         else:
#             logger.warning(f"No valid strategy identified from DB for context: {context}. Defaulting to 'hold'.")
#             signal = "hold"
        
#         sl_tp_tsl_mode_info = f"'{self.parameter_mode}' mode"
#         logger.info(f"Final decision: Signal='{signal}', SL_Mult={sl_mult}, TP_Mult={tp_mult}, TSL_Mult={tsl_mult}, Strategy='{final_strategy_name_used or 'N/A'}' (SL/TP/TSL Source: {sl_tp_tsl_mode_info})")
#         return signal, float(sl_mult), float(tp_mult), float(tsl_mult) if tsl_mult is not None else None, final_strategy_name_used

# # Example usage (for testing purposes)
# if __name__ == '__main__':
#     logger.info("Testing RuleBasedAgent (requires MongoDB connection and data)...")
#     logger.info(f"Agent will use SL/TP/TSL settings based on config.AGENT_USE_TUNED_PARAMS.")
#     logger.info(f"Current config.AGENT_USE_TUNED_PARAMS: {getattr(config, 'AGENT_USE_TUNED_PARAMS', 'Not Set (agent defaults to True/tuned)')}")
    
#     # Ensure MONGO_COLLECTION_TUNED_PARAMS is set in your config.py
#     # e.g., MONGO_COLLECTION_TUNED_PARAMS = "my_tuned_strategy_configs"
#     if not hasattr(config, "MONGO_COLLECTION_TUNED_PARAMS"):
#         logger.error("config.MONGO_COLLECTION_TUNED_PARAMS is not defined! Agent will fail to find tuned params.")
#         # sys.exit(1) # Optional: exit if config is missing

#     mock_mongo_client = None
#     try:
#         mock_mongo_client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)
#         mock_mongo_client.admin.command('ping') 
#         logger.info("MongoDB connection successful for test.")

#         agent = RuleBasedAgent(mongo_client=mock_mongo_client)

#         mock_data = {
#             'timestamp': datetime(2025, 5, 15, 10, 30, 0),
#             'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0, 'volume': 1000,
#             'atr': 1.5, 
#             'regime': 'Trending', 
#             'symbol': 'NIFTY',
#             'timeframe': '5min',
#             'is_expiry': False,
#             'volatility_status': 'Normal' # This will be mapped to context['volatility_status_from_data']
#         }
#         mock_current_row = pd.Series(mock_data, name=mock_data['timestamp'])
#         mock_data_history = pd.DataFrame() # Empty for this simple test

#         logger.info(f"\n--- Calling agent.decide() with mock data ---")
#         signal, sl, tp, tsl, strat_name = agent.decide(mock_current_row, mock_data_history)
#         logger.info(f"--- Agent decision ---")
#         logger.info(f"Signal: {signal}")
#         logger.info(f"SL Multiplier: {sl}")
#         logger.info(f"TP Multiplier: {tp}")
#         logger.info(f"TSL Multiplier: {tsl}")
#         logger.info(f"Strategy Name Used: {strat_name}")
#         logger.info(f"Agent's parameter_mode used (for SL/TP/TSL): {agent.parameter_mode}")
#         logger.info(f"-----------------------\n")

#     except ConnectionError as ce:
#         logger.error(f"Test failed due to MongoDB connection error: {ce}")
#     except Exception as e:
#         logger.error(f"An error occurred during RuleBasedAgent test: {e}", exc_info=True)
#     finally:
#         if mock_mongo_client:
#             mock_mongo_client.close()
#             logger.info("Test: MongoDB connection closed.")

# app/agentic_core.py
import pandas as pd
import logging
from typing import Dict, Tuple, Callable, Optional, Any
from datetime import datetime, time as dt_time
import pytz
import random
import time
from pymongo import DESCENDING
from pymongo.errors import PyMongoError
from builtins import ConnectionError  # or simply remove if not needed here


try:
    from app.config import config
    from app.strategies import strategy_factories
    from app.mongo_manager import MongoManager
except ImportError:
    import sys
    from pathlib import Path
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from app.config import config
    from app.strategies import strategy_factories
    from app.mongo_manager import MongoManager

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    log_level_from_config = getattr(config, "LOG_LEVEL", "INFO")
    log_format_from_config = getattr(config, "LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.basicConfig(level=log_level_from_config, format=log_format_from_config)

class RuleBasedAgent:
    def __init__(self):
        """Initializes the RuleBasedAgent using MongoManager for DB connection."""
        self.mongo_db = MongoManager.get_database()
        if self.mongo_db is None:
            logger.error("Failed to connect to MongoDB via MongoManager.")
            raise ConnectionError("RuleBasedAgent: Failed to connect to MongoDB.")

        # Set parameter_mode based on config
        self.parameter_mode = "tuned" if getattr(config, 'AGENT_USE_TUNED_PARAMS', True) else "default"
        logger.info(f"RuleBasedAgent initialized. SL/TP/TSL parameter_mode: '{self.parameter_mode}'")

        # Default mappings for ultimate fallback
        self.default_strategy_mapping_by_regime: Dict[str, Optional[Tuple[str, Callable]]] = {
            "Trending": ("SuperTrend_ADX", strategy_factories.get("SuperTrend_ADX")),
            "Ranging": ("BB_MeanReversion", strategy_factories.get("BB_MeanReversion")),
            "Momentum": ("EMA_Crossover", strategy_factories.get("EMA_Crossover")),
            "Unknown": None,
        }
        self.default_parameter_mapping_by_regime: Dict[str, Dict[str, float]] = {
            "Trending": {
                "sl_mult": getattr(config, "TREND_SL_ATR_MULT", config.DEFAULT_SL_ATR_MULT * 1.2),
                "tp_mult": getattr(config, "TREND_TP_ATR_MULT", config.DEFAULT_TP_ATR_MULT * 1.5),
            },
            "Ranging": {
                "sl_mult": getattr(config, "RANGE_SL_ATR_MULT", config.DEFAULT_SL_ATR_MULT * 0.8),
                "tp_mult": getattr(config, "RANGE_TP_ATR_MULT", config.DEFAULT_TP_ATR_MULT * 0.8),
            },
            "Momentum": {
                "sl_mult": config.DEFAULT_SL_ATR_MULT,
                "tp_mult": config.DEFAULT_TP_ATR_MULT,
            },
            "Unknown": {
                "sl_mult": config.DEFAULT_SL_ATR_MULT,
                "tp_mult": config.DEFAULT_TP_ATR_MULT,
            }
        }

        # Validate default strategy factories
        for regime, pair in self.default_strategy_mapping_by_regime.items():
            if pair is None:
                logger.debug(f"No default strategy defined for regime '{regime}'.")
            else:
                name, func = pair
                if not callable(func):
                    logger.error(f"Default strategy '{name}' for regime '{regime}' is not callable. Disabling default.")
                    self.default_strategy_mapping_by_regime[regime] = None
        logger.info("RuleBasedAgent initialization complete.")

        # Cache for query results
        self._context_cache: Dict[tuple, Tuple[Optional[str], Dict[str, Any]]] = {}

    # Inside infer_session function in simulation_engine.py
    def infer_session(ts: pd.Timestamp) -> str:
        if not isinstance(ts, pd.Timestamp):
            return "Unknown"
        tm = ts.time()
        if dt_time(9, 15) <= tm <= dt_time(11, 30): # Use dt_time()
            return "Morning"
        elif dt_time(11, 30) < tm <= dt_time(13, 45): # Use dt_time()
            return "Midday"
        elif dt_time(13, 45) < tm <= dt_time(15, 30): # Use dt_time()
            return "Afternoon"
        return "OffMarket"
    def get_top_strategy_for_context(self, context: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Retrieves the top-performing strategy and parameters from strategy_best_params,
        falling back to strategy_tuned_parameters, then default mappings.
        """
        # Validate context
        required_fields = ["symbol", "timeframe", "market_condition"]
        if not all(field in context for field in required_fields):
            logger.error(f"Missing required context fields: {required_fields}. Got: {context}")
            return None, {}

        valid_market_conditions = {"Trending", "Ranging", "Momentum", "Unknown"}
        valid_volatility_statuses = {"high", "normal", "low", "anyvolatility", "Unknown"}
        market_condition = context.get("market_condition", "Unknown")
        volatility_status = context.get("volatility_status_from_data", "Unknown")
        if market_condition not in valid_market_conditions:
            logger.warning(f"Invalid market_condition: {market_condition}. Defaulting to 'Unknown'.")
            market_condition = "Unknown"
        if volatility_status not in valid_volatility_statuses:
            logger.warning(f"Invalid volatility_status_from_data: {volatility_status}. Defaulting to 'Unknown'.")
            volatility_status = "Unknown"

        # Create cache key
        cache_key = (
            context.get("symbol", "").upper(),
            context.get("timeframe", ""),
            context.get("day", ""),
            context.get("session", ""),
            context.get("is_expiry", False),
            market_condition,
            volatility_status
        )
        if cache_key in self._context_cache:
            logger.debug(f"Cache hit for context: {context}")
            return self._context_cache[cache_key]

        # Build query
        filter_query = {
            "symbol": context.get("symbol", "").upper(),
            "timeframe": context.get("timeframe", ""),
            "market_condition": market_condition,
            "volatility_status_from_data": volatility_status
        }
        if context.get("day") is not None:
            filter_query["day"] = str(context.get("day"))
        if context.get("session") is not None:
            filter_query["session"] = str(context.get("session"))
        if context.get("is_expiry") is not None:
            filter_query["is_expiry"] = bool(context.get("is_expiry"))
        filter_query["achieved_performance_score"] = {"$exists": True, "$ne": None}
        filter_query["best_params"] = {"$exists": True, "$ne": None}

        strategy_name = None
        comprehensive_best_params: Dict[str, Any] = {}

        # Try strategy_best_params first
        best_params_collection_name = getattr(config, "MONGO_COLLECTION_BEST_PARAMS", "strategy_best_params")
        tuned_params_collection_name = getattr(config, "MONGO_COLLECTION_TUNED_PARAMS", "strategy_tuned_parameters")

        for collection_name in [best_params_collection_name, tuned_params_collection_name]:
            collection = self.mongo_db[collection_name]
            for attempt in range(3):
                try:
                    logger.debug(f"Querying '{collection_name}' with filter: {filter_query}")
                    top_entry = collection.find(filter_query).sort("achieved_performance_score", DESCENDING).limit(1)
                    top_entry_list = list(top_entry)
                    if top_entry_list:
                        top_entry_doc = top_entry_list[0]
                        strategy_name = top_entry_doc.get("strategy_name")
                        comprehensive_best_params = top_entry_doc.get("best_params", {})
                        if not strategy_name:
                            logger.warning(f"Document in '{collection_name}' missing 'strategy_name'. Doc ID: {top_entry_doc.get('_id')}")
                            continue
                        if not comprehensive_best_params or not isinstance(comprehensive_best_params, dict):
                            logger.warning(f"Document for '{strategy_name}' in '{collection_name}' has invalid 'best_params'. Doc ID: {top_entry_doc.get('_id')}")
                            comprehensive_best_params = {}
                        logger.info(f"Found strategy in '{collection_name}': '{strategy_name}' with score {top_entry_doc.get('achieved_performance_score')}. Params: {comprehensive_best_params}")
                        break
                    else:
                        logger.debug(f"No matching entry in '{collection_name}' for context: {context}")
                except ConnectionError as e:
                    logger.error(f"Connection error querying '{collection_name}', attempt {attempt + 1}: {e}", exc_info=True)
                    if attempt == 2:
                        logger.critical(f"Max retries reached for '{collection_name}'.")
                        break
                    time.sleep(random.uniform(0.5, 1.5))
                except PyMongoError as e:
                    logger.error(f"PyMongo error querying '{collection_name}': {e}", exc_info=True)
                    break
                except Exception as e:
                    logger.error(f"Unexpected error querying '{collection_name}': {e}", exc_info=True)
                    break
            if strategy_name:
                break

        # Fallback to default mappings
        if not strategy_name:
            regime = context.get("market_condition", "Unknown")
            default_pair = self.default_strategy_mapping_by_regime.get(regime)
            if default_pair:
                strategy_name, _ = default_pair
                comprehensive_best_params = self.default_parameter_mapping_by_regime.get(regime, {})
                logger.info(f"Fallback to default strategy '{strategy_name}' for regime '{regime}'. Params: {comprehensive_best_params}")
            else:
                logger.warning(f"No strategy found in DB or defaults for context: {context}")
                return None, {}

        # Cache result
        self._context_cache[cache_key] = (strategy_name, comprehensive_best_params)
        return strategy_name, comprehensive_best_params

    def decide(self, current_row: pd.Series, data_history: pd.DataFrame = None) -> Tuple[str, float, float, Optional[float], Optional[str]]:
        """Generates a trading decision based on context and tuned parameters."""
        regime = current_row.get('regime', 'Unknown')
        symbol = current_row.get('symbol', getattr(config, "DEFAULT_SYMBOL", "NIFTY")).lower()
        timeframe = current_row.get('timeframe', getattr(config, "DEFAULT_TIMEFRAME", "5min"))

        dt_index = current_row.name
        if not isinstance(dt_index, datetime):
            try:
                dt_index = pd.to_datetime(dt_index)
            except (ValueError, TypeError):
                logger.warning(f"current_row.name '{dt_index}' is not a datetime. Using current time.")
                dt_index = datetime.now(pytz.timezone('Asia/Kolkata'))

        context: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "day": dt_index.strftime("%A"),
            "session": self._infer_session(dt_index),
            "is_expiry": bool(current_row.get('is_expiry', False)),
            "market_condition": str(regime),
            "volatility_status_from_data": str(current_row.get('volatility_status', 'Unknown'))
        }
        logger.info(f"Decision context: {context}, Parameter Mode: '{self.parameter_mode}'")

        strategy_name, comprehensive_best_params = self.get_top_strategy_for_context(context)
        signal = "hold"
        final_strategy_name_used = None
        sl_mult: float = config.DEFAULT_SL_ATR_MULT
        tp_mult: float = config.DEFAULT_TP_ATR_MULT
        tsl_mult: Optional[float] = None

        if self.parameter_mode == "tuned":
            sl_mult = comprehensive_best_params.get("sl_mult", config.DEFAULT_SL_ATR_MULT)
            tp_mult = comprehensive_best_params.get("tp_mult", config.DEFAULT_TP_ATR_MULT)
            enable_tsl = comprehensive_best_params.get("enable_tsl_mult", getattr(config, "DEFAULT_ENABLE_TSL", False))
            if enable_tsl:
                tsl_mult = comprehensive_best_params.get("tsl_mult", config.DEFAULT_TSL_ATR_MULT)
            logger.debug(f"Tuned params: SL={sl_mult}, TP={tp_mult}, EnableTSL={enable_tsl}, TSL={tsl_mult}")
        elif self.parameter_mode == "default":
            sl_mult = config.DEFAULT_SL_ATR_MULT
            tp_mult = config.DEFAULT_TP_ATR_MULT
            if getattr(config, "DEFAULT_ENABLE_TSL", False):
                tsl_mult = config.DEFAULT_TSL_ATR_MULT
            logger.debug(f"Default params: SL={sl_mult}, TP={tp_mult}, EnableTSL={getattr(config, 'DEFAULT_ENABLE_TSL', False)}, TSL={tsl_mult}")

        if strategy_name and strategy_name in strategy_factories:
            strategy_func_factory = strategy_factories[strategy_name]
            if not callable(strategy_func_factory):
                logger.error(f"Strategy factory for '{strategy_name}' is not callable.")
            else:
                try:
                    strategy_instance = strategy_func_factory(**comprehensive_best_params)
                    current_signal = strategy_instance(current_row, data_history)
                    if current_signal in ['buy_potential', 'sell_potential', 'hold']:
                        signal = current_signal
                        final_strategy_name_used = strategy_name
                        logger.info(f"Strategy '{strategy_name}' generated signal: '{signal}'")
                    else:
                        logger.warning(f"Strategy '{strategy_name}' returned invalid signal: '{current_signal}'. Defaulting to 'hold'.")
                        signal = "hold"
                except Exception as e:
                    logger.error(f"Error executing strategy '{strategy_name}': {e}", exc_info=True)
                    signal = "hold"
        else:
            logger.warning(f"No valid strategy for context: {context}. Defaulting to 'hold'.")

        logger.info(f"Final decision: Signal='{signal}', SL_Mult={sl_mult}, TP_Mult={tp_mult}, TSL_Mult={tsl_mult}, Strategy='{final_strategy_name_used or 'N/A'}'")
        return signal, float(sl_mult), float(tp_mult), float(tsl_mult) if tsl_mult is not None else None, final_strategy_name_used

if __name__ == '__main__':
    logger.info("Testing RuleBasedAgent...")
    try:
        agent = RuleBasedAgent()
        mock_data = {
            'timestamp': datetime(2025, 5, 15, 10, 30, 0, tzinfo=pytz.timezone('Asia/Kolkata')),
            'open': 100.0, 'high': 102.0, 'low': 99.0, 'close': 101.0, 'volume': 1000,
            'atr': 1.5,
            'regime': 'Trending',
            'symbol': 'NIFTY',
            'timeframe': '5min',
            'is_expiry': False,
            'volatility_status': 'normal'
        }
        mock_current_row = pd.Series(mock_data, name=mock_data['timestamp'])
        mock_data_history = pd.DataFrame()

        logger.info("\n--- Calling agent.decide() with mock data ---")
        signal, sl, tp, tsl, strat_name = agent.decide(mock_current_row, mock_data_history)
        logger.info(f"--- Agent decision ---")
        logger.info(f"Signal: {signal}")
        logger.info(f"SL Multiplier: {sl}")
        logger.info(f"TP Multiplier: {tp}")
        logger.info(f"TSL Multiplier: {tsl}")
        logger.info(f"Strategy Name: {strat_name}")
        logger.info(f"Parameter Mode: {agent.parameter_mode}")
        logger.info("-----------------------\n")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        MongoManager.close_client()
        logger.info("Test: MongoDB connection closed.")