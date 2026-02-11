# In strategy_tester/option_trade_executor_wrapper.py
import os
import logging # Ensure logging is imported if you use logger_ote_wrapper
from .option_trade_executor import OptionTradeExecutor # Make sure path is correct

logger_ote_wrapper = logging.getLogger(__name__) # Define logger if not already

def run_option_trade_executor(signal_df, strategy_name, timeframe,
                              strategy_params, # <--- CHANGED HERE
                              run_id, base_output_dir):
    """
    Initializes and runs the OptionTradeExecutor.
    'strategy_params' are the parameters from the main index-level strategy.
    """
    json_log_subfolder = "json_trade_logs"
    json_log_dir = os.path.join(base_output_dir, str(run_id), json_log_subfolder)
    os.makedirs(json_log_dir, exist_ok=True)
    json_log_filename = f"{strategy_name}_{timeframe}_option_trades.json"
    json_log_path = os.path.join(json_log_dir, json_log_filename)

    logger_ote_wrapper.info(f"OTE Wrapper: Initializing OptionTradeExecutor for {strategy_name} on {timeframe}. JSON log: {json_log_path}")

    executor = OptionTradeExecutor(
        lot_size=strategy_params.get('lot_size', 75),
        entry_premium=strategy_params.get('option_entry_premium', 100), # Assumes these keys might be in strategy_params
        premium_change_rate=strategy_params.get('option_premium_change_rate', 0.1),
        points_per_change=strategy_params.get('option_points_per_change', 10),
        strike_interval=strategy_params.get('strike_interval', 50),
        json_log_path=json_log_path,
        index_strategy_params=strategy_params # <--- CHANGED HERE (used for passing full dict if OTE needs it)
    )

    # The 'params_from_main_backtester' argument in OptionTradeExecutor.execute_trades
    # is specifically for the ATR multipliers.
    # 'strategy_params' (the dict itself) contains these multipliers.
    option_trades_df = executor.execute_trades(
        signal_df=signal_df,
        strategy_name=strategy_name,
        timeframe=timeframe,
        params_from_main_backtester=strategy_params # <--- CHANGED HERE (this dict should contain sl_atr_mult, etc.)
    )
    return option_trades_df