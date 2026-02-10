from flask import Flask, jsonify, render_template, request
import os
from datetime import datetime
import logging
from pathlib import Path
import pandas as pd
from data.data_fetcher import fetch_data
# Assuming indicators.py contains IndicatorCalculator
from data.indicators import IndicatorCalculator
# Assuming sim_engine.py contains run_simulation
from simulator.sim_engine import run_simulation
# Assuming base_strategy.py contains get_strategy
from strategy.base_strategy import get_strategy
from config import Config

app = Flask(__name__)
project_root = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = Config()
# Assuming IndicatorCalculator is available and working
indicator_calculator_available = True

@app.route("/")
@app.route("/", methods=["GET"])
def index():
    # Renders the main page template
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    """
    Main endpoint to fetch data, calculate indicators, generate signals,
    run simulation, and return results.
    """
    req = request.get_json(force=True) or {}
    logger.info("Received payload: %s", req)
    try:
        # --- Extract Request Parameters ---
        source = req.get("source", "offline")
        exchange = req.get("exchange", "NSE").upper()
        ticker = req.get("ticker", "NIFTY 50")
        interval = req.get("interval", "5minute")
        start_date = req.get("start_date")
        end_date = req.get("end_date")
        sl_pct = float(req.get("sl_pct", config.DEFAULT_SL_PCT))
        trail_pct = float(req.get("trail_pct", config.DEFAULT_TRAIL_PCT))
        strat_key = req.get("strategy", "ema").lower() # Use lowercase for consistency
        commission_per_trade = float(req.get('commission_per_trade', config.COMMISSION_PER_TRADE))
        slippage_per_trade = float(req.get('slippage_per_trade', config.SLIPPAGE_PER_TRADE))

        # --- Fetch Data ---
        df = fetch_data(source=source, interval=interval, start_date=start_date, end_date=end_date, ticker=ticker, exchange=exchange)
        if df is None or df.empty:
            logger.error(f"No data fetched for {ticker} ({exchange})")
            return jsonify(error=f"No data for {ticker} ({exchange})"), 404
        logger.info("Data fetched → %s", df.shape)

        # --- Fetch Previous Day's Close (for potential gap calculations) ---
        prev_close = None
        # Only fetch if online source and start_date is provided
        if source.lower() == "online" and start_date:
            try:
                # Calculate previous trading day (needs more robust logic for holidays/weekends)
                # Simple subtraction might land on a non-trading day
                prev_date = (pd.to_datetime(start_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d") # Basic previous day
                logger.info(f"Attempting to fetch previous day data for date: {prev_date}")
                prev_df = fetch_data(
                    source=source, interval="1day", start_date=prev_date, end_date=prev_date,
                    ticker=ticker, exchange=exchange
                )
                if prev_df is not None and not prev_df.empty:
                    prev_close = prev_df['close'].iloc[-1]
                    logger.info(f"Previous day's close fetched: {prev_close}")
                else:
                    logger.warning(f"No previous day data found for {prev_date}")
            except Exception as e:
                logger.warning(f"Failed to fetch previous day's close: {e}", exc_info=True)

        # --- Calculate Indicators (Optional, if needed before strategy) ---
        df_processed = df.copy()
        if indicator_calculator_available:
            try:
                logger.info("Calculating indicators using IndicatorCalculator...")
                # Define parameters for IndicatorCalculator
                calc_params = {
                    'sma_periods': config.INDICATOR_SMA_PERIODS, # Use config values
                    'ema_periods': config.INDICATOR_EMA_PERIODS,
                    'rsi_period': int(req.get('rsi_period', 14)), # Default RSI
                    'atr_period': int(req.get('atr_period', 14)), # Default ATR
                    'bollinger_period': 20,
                    'stochastic_period': 14,
                    'dmi_length': 14,
                    'adx_smoothing': 14,
                    'vol_ma_len': 20,
                    'macd_params': (
                        int(req.get('macd_fast', 12)),
                        int(req.get('macd_slow', 26)),
                        int(req.get('macd_signal', 9))
                    ),
                    'vwap_type': config.VWAP_TYPE, # Use config
                    'vwap_enabled': config.VWAP_ENABLED
                }
                # Adjust specific periods if AlphaTrend is selected
                if strat_key == "alphatrend":
                    at_ap = int(req.get('at_ap', 10)) # Get AlphaTrend period
                    calc_params['atr_period'] = at_ap # Use AT period for ATR calc here
                    calc_params['rsi_period'] = at_ap # Use AT period for RSI calc here
                    logger.info(f"Adjusted IndicatorCalculator params for AlphaTrend: atr_period={at_ap}, rsi_period={at_ap}")

                ic = IndicatorCalculator(params=calc_params)
                # Ensure df_processed has DatetimeIndex before passing
                if not isinstance(df_processed.index, pd.DatetimeIndex):
                     if 'datetime' in df_processed.columns:
                         df_processed = df_processed.set_index(pd.to_datetime(df_processed['datetime']))
                     else:
                          raise ValueError("DataFrame needs 'datetime' column or DatetimeIndex for IndicatorCalculator")

                df_processed = ic.calculate_all_indicators(df_processed)
                logger.info(f"Indicators calculated via IndicatorCalculator. Shape: {df_processed.shape}")
            except Exception as e:
                logger.error(f"IndicatorCalculator failed: {e}", exc_info=True)
                # Decide if this is fatal or if strategy can proceed without these indicators
                # return jsonify(error=f"Indicator calculation failed: {e}"), 500
                logger.warning("Proceeding without indicators from IndicatorCalculator due to error.")
                df_processed = df.copy() # Fallback to original data if calc fails
        else:
            logger.info("Skipping IndicatorCalculator.")
            df_processed = df.copy() # Ensure df_processed is assigned

        # --- Get Strategy & Generate Signals ---
        strategy_params = {}
        # Populate strategy-specific parameters from the request
        if strat_key == "alphatrend":
            # Only include parameters actually used by the current AlphaTrendStrategy __init__
            strategy_params['coeff'] = float(req.get('at_coeff', 0.6))
            strategy_params['ap'] = int(req.get('at_ap', 10))
            strategy_params['macd_fast'] = int(req.get('macd_fast', 12))
            strategy_params['macd_slow'] = int(req.get('macd_slow', 26))
            strategy_params['macd_signal'] = int(req.get('macd_signal', 9))
            strategy_params['supertrend_period'] = int(req.get('supertrend_period', 10))
            strategy_params['supertrend_multiplier'] = float(req.get('supertrend_multiplier', 3.0))
            # Add other configurable params from AlphaTrendStrategy.__init__ if needed
            # e.g., strategy_params['rsi_buy_lower'] = int(req.get('rsi_buy_lower', 50)) ...
            logger.info("Cleaned AlphaTrend params for get_strategy.")
        elif strat_key == "ema":
             strategy_params['short_window'] = int(req.get('short_window', 9))
             strategy_params['long_window'] = int(req.get('long_window', 21))
        # Add other strategies here...

        strategy = get_strategy(strat_key, params=strategy_params)
        logger.info(f"Using strategy: {strat_key} with effective params used by __init__.")

        logger.info("Generating signals...")
        try:
            # Pass prev_close only if the strategy's generate_signals accepts it
            # (Requires inspecting the strategy method or using try/except)
            # For simplicity, using try/except as before
            df_with_signals = strategy.generate_signals(df_processed, prev_close=prev_close)
            logger.info("Strategy accepted prev_close.")
        except TypeError:
            logger.info("Strategy does not accept prev_close, calling without it.")
            df_with_signals = strategy.generate_signals(df_processed)
        except Exception as e:
             logger.error(f"Error during signal generation: {e}", exc_info=True)
             return jsonify(error=f"Signal generation failed: {e}"), 500

        logger.info(f"Signals generated. Shape: {df_with_signals.shape}")

        # --- Validation Before Simulation ---
        if df_with_signals is None or df_with_signals.empty:
             logger.error("Signal generation returned empty or None DataFrame.")
             return jsonify(error="Signal generation failed to produce results."), 500

        required_sim_cols = ['close', 'position', 'high', 'low'] # Minimal required by most sims
        # Add strategy-specific required columns if known (e.g., AlphaTrend needs specific ATR, RSI, MACD)
        if strat_key == "alphatrend":
            # These are the names expected by the simulation according to the previous error log
            required_sim_cols.extend(['rsi_10', 'rsi_slope', 'macd', 'macd_signal', 'atr_14'])

        missing_cols = [col for col in required_sim_cols if col not in df_with_signals.columns]
        if missing_cols:
            logger.error(f"DataFrame missing required columns for simulation: {missing_cols}. Available: {df_with_signals.columns.tolist()}")
            # Attempting to run sim anyway might lead to errors downstream
            # return jsonify(error=f"Signal generation output missing columns: {missing_cols}"), 500
            logger.warning("Proceeding to simulation despite missing columns...")
        else:
             logger.info("All required columns for simulation are present.")

        # --- Remove Redundant Column Mapping ---
        # The strategy (e.g., AlphaTrendStrategy) should now output the correctly named columns.
        # logger.debug(f"Columns before sim: {df_with_signals.columns.tolist()}")

        # --- Run Simulation ---
        logger.info(f"Starting simulation for {strat_key}...")
        # *** ERROR LIKELY INSIDE run_simulation in sim_engine.py ***
        # *** Check run_simulation for assignment to 'macd_prev' ***
        sim_result = run_simulation(
            data=df_with_signals, # Pass the dataframe with signals and indicators
            strategy=strategy, # Pass the strategy object (might be needed for params inside sim)
            sl_pct=sl_pct,
            trail_pct=trail_pct,
            commission_per_trade=commission_per_trade,
            slippage_per_trade=slippage_per_trade,
        )
        logger.info(f"Simulation complete. Pnl: {sim_result.get('pnl')}")

        # --- Export Results ---
        try:
            results_dir = Path(project_root) / "results"
            results_dir.mkdir(parents=True, exist_ok=True) # Use Path object
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Sanitize ticker name for filename
            safe_ticker = "".join(c for c in ticker if c.isalnum() or c in (' ', '_', '-')).replace(" ", "_").lower() or "unknown"
            filename = f"result_{safe_ticker}_{exchange.lower()}_{strat_key}_{interval}_{timestamp_str}.csv"
            filepath = results_dir / filename # Use Path object
            # Save simulation input data (which includes signals/indicators)
            df_with_signals.to_csv(filepath)
            logger.info(f"Full results data exported to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to export results to CSV: {e}", exc_info=True)
            # Don't fail the request just because export failed

        # Return simulation results (e.g., PnL, trades)
        return jsonify(sim_result)

    except (ValueError, FileNotFoundError, ConnectionError) as e:
        # Handle specific known errors gracefully
        logger.error(f"Handled Error: {e}", exc_info=True) # Log traceback for handled errors too
        status_code = 503 if isinstance(e, ConnectionError) else 400
        return jsonify(error=str(e)), status_code
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error("Unhandled exception during /start request", exc_info=True)
        return jsonify(error=f"An unexpected error occurred: {e}"), 500

if __name__ == "__main__":
    # Run the Flask development server
    # Use debug=False for production or when debugger PIN is not needed
    app.run(host="0.0.0.0", port=5001, debug=config.FLASK_DEBUG) # Use config for debug mode

