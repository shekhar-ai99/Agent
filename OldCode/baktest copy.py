import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from typing import Dict, Callable, Optional, Tuple, Any

# Configure logging (Consider adding strategy name to format later if needed)
logging.basicConfig(
    level=logging.INFO, # Set to DEBUG for more verbose entry/exit logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest_enhanced.log'), # New log file name
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedMultiStrategyBacktester:
    """Enhanced backtester with improved performance metrics, visualization,
       trailing stops, and detailed logging."""

    def __init__(self, strategies: Dict[str, Callable], initial_capital: float = 100000,
                 commission_pct: float = 0.0005, slippage_pct: float = 0.0002,
                 risk_per_trade_pct: Optional[float] = 0.01): # Added optional risk % for position sizing
        """
        Initialize backtester with strategies and trading parameters

        Args:
            strategies: Dictionary of strategy names to functions
            initial_capital: Starting capital (used for position sizing if risk_per_trade_pct is set)
            commission_pct: Percentage commission per trade side (applied on entry and exit)
            slippage_pct: Percentage slippage assumption per side
            risk_per_trade_pct: Optional - Percentage of capital to risk per trade for position sizing.
                                If None, PnL is calculated in points.
        """
        self.strategies = strategies
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.risk_per_trade_pct = risk_per_trade_pct # Store risk parameter
        self.results_df = None
        self.performance_metrics = None

        # Initialize indicator calculator
        try:
            # IMPORTANT: Ensure this import works in your project structure
            from indicators import IndicatorCalculator
            self.indicator_calculator = IndicatorCalculator()
            logger.info("Successfully imported IndicatorCalculator.")
        except ImportError:
            logger.warning("Could not import IndicatorCalculator. Using simplified dummy calculator.")
            self.indicator_calculator = self.DummyIndicatorCalculator()

    class DummyIndicatorCalculator:
        """Fallback calculator if main one isn't available"""
        def calculate_all_indicators(self, df):
            logger.warning("Using dummy indicator calculator - ONLY calculates 'atr' for stops.")
            # Calculate basic ATR needed for stops if missing
            if 'atr' not in df.columns:
                 high_low = df['high'] - df['low']
                 high_close = abs(df['high'] - df['close'].shift())
                 low_close = abs(df['low'] - df['close'].shift())
                 tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                 df['atr'] = tr.rolling(window=14).mean() # Default ATR period
            df['returns'] = df['close'].pct_change() # Keep returns for metrics
            return df

    def preprocess_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Enhance data with indicators and validation"""
        logger.info("Preprocessing data...")

        # Ensure lowercase columns for consistency
        raw_data.columns = [col.lower() for col in raw_data.columns]

        # Validate input
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in raw_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Convert index to datetime if needed
        if not isinstance(raw_data.index, pd.DatetimeIndex):
            raw_data.index = pd.to_datetime(raw_data.index)
            logger.info("Converted index to DatetimeIndex.")

        # Calculate indicators
        logger.info("Calculating indicators...")
        processed_data = self.indicator_calculator.calculate_all_indicators(raw_data.copy())
        logger.info("Indicator calculation finished.")

        # Ensure ATR is present (essential for stops)
        if 'atr' not in processed_data.columns or processed_data['atr'].isnull().all():
             logger.warning("ATR indicator missing or all NaN after calculation. Attempting basic calculation.")
             high_low = processed_data['high'] - processed_data['low']
             high_close = abs(processed_data['high'] - processed_data['close'].shift())
             low_close = abs(processed_data['low'] - processed_data['close'].shift())
             tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
             processed_data['atr'] = tr.rolling(window=14).mean() # Default ATR period
             if processed_data['atr'].isnull().all():
                 raise ValueError("Failed to calculate ATR. Cannot proceed without ATR for stops.")

        # Additional derived features (optional)
        processed_data['volatility'] = processed_data['close'].pct_change().rolling(20).std() * np.sqrt(252) # Annualized
        processed_data['intraday_range_pct'] = (processed_data['high'] - processed_data['low']) / processed_data['low']

        processed_data = processed_data.dropna(subset=['atr']) # Ensure ATR is not NaN for subsequent steps
        logger.info(f"Data preprocessing complete. Final shape after dropping NaNs (esp. ATR): {processed_data.shape}")
        return processed_data

    def run_backtest(self, raw_data: pd.DataFrame, atr_stop_multiplier: float = 1.5,
                     atr_target_multiplier: float = 2.0, use_trailing_stop: bool = True) -> Optional[pd.DataFrame]:
        """
        Run backtest across all strategies with enhanced position management and logging.

        Args:
            raw_data: DataFrame with price data (datetime index, ohlcv columns).
            atr_stop_multiplier: ATR multiple for initial stop loss.
            atr_target_multiplier: ATR multiple for take profit.
            use_trailing_stop: Whether to use the ATR trailing stop logic.

        Returns:
            DataFrame with backtest results or None if failed.
        """
        try:
            data = self.preprocess_data(raw_data)
            if data.empty:
                logger.error("No valid data after preprocessing. Aborting backtest.")
                return None

            # Initialize results storage
            results = data.copy()
            strategy_states = self._initialize_strategy_columns(results) # Also gets initial state dict

            logger.info(f"Starting backtest loop ({len(results)} bars)...")
            # Loop through data - start index depends on indicator warm-up,
            # preprocess_data dropping NaNs handles this implicitly now.
            for i in range(1, len(results)):
                current_idx = results.index[i]
                prev_idx = results.index[i-1]
                current_row = results.iloc[i]

                for name, strategy_func in self.strategies.items():
                    # Process one strategy for the current bar
                    self._process_strategy(
                        results=results,
                        strategy_states=strategy_states,
                        name=name,
                        strategy_func=strategy_func,
                        current_idx=current_idx,
                        prev_idx=prev_idx,
                        current_row=current_row,
                        atr_stop_multiplier=atr_stop_multiplier,
                        atr_target_multiplier=atr_target_multiplier,
                        use_trailing_stop=use_trailing_stop
                    )

            logger.info("Backtest loop finished.")
            self.results_df = results
            logger.info("Calculating performance metrics...")
            self._calculate_performance_metrics() # Calculate metrics after loop
            logger.info("Performance metrics calculation finished.")
            return results

        except Exception as e:
            logger.error(f"Backtest run failed: {str(e)}", exc_info=True)
            self.results_df = None # Ensure results are None on failure
            return None


    def _initialize_strategy_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Initialize results DataFrame columns and return initial state dictionary."""
        strategy_states = {}
        trade_id_counters = {name: 0 for name in self.strategies.keys()} # Track trade IDs per strategy

        for name in self.strategies.keys():
            # Trade signals and execution
            df[f'{name}_signal'] = 'hold'
            df[f'{name}_position'] = '' # 'long', 'short', ''
            df[f'{name}_entry_price'] = np.nan
            df[f'{name}_exit_price'] = np.nan
            df[f'{name}_pnl_points'] = 0.0 # PnL in price points per trade
            df[f'{name}_cumulative_pnl_points'] = 0.0 # Cumulative points PnL

            # Risk management
            df[f'{name}_sl'] = np.nan # Initial Stop Loss
            df[f'{name}_tp'] = np.nan # Take Profit
            df[f'{name}_atr_at_entry'] = np.nan # Store ATR used for SL/TP calculation
            df[f'{name}_trailing_sl'] = np.nan # Trailing Stop Loss

            # Trade tracking / State
            df[f'{name}_trade_id'] = 0 # ID for each trade entry-exit pair
            df[f'{name}_active_trade'] = False # If currently in a trade

            # Runtime state dictionary (not stored in DataFrame)
            strategy_states[name] = {
                'active_trade': False,
                'position': '', # 'long' or 'short'
                'entry_price': np.nan,
                'initial_sl': np.nan,
                'take_profit': np.nan,
                'trailing_sl': np.nan,
                'highest_high_since_entry': np.nan, # For trailing SL (long)
                'lowest_low_since_entry': np.nan,  # For trailing SL (short)
                'trade_id': 0,
                 'cumulative_pnl_points': 0.0
            }
        return strategy_states


    def _process_strategy(self, results: pd.DataFrame, strategy_states: Dict[str, Dict[str, Any]],
                           name: str, strategy_func: Callable, current_idx, prev_idx, current_row,
                           atr_stop_multiplier: float, atr_target_multiplier: float, use_trailing_stop: bool) -> None:
        """Process a single strategy for the current bar, updating results DataFrame and state dictionary."""

        state = strategy_states[name] # Get current runtime state for this strategy
        signal = strategy_func(current_row, results.iloc[:current_row.name]) # Get signal

        # --- 1. Handle Active Trade ---
        if state['active_trade']:
            # Update Trailing Stop Loss (if enabled)
            if use_trailing_stop:
                self._update_trailing_stop(state, current_row, name, current_idx)

            # Check for Exits
            exit_triggered, exit_price, exit_reason = self._check_exits(state, current_row, signal, name, current_idx)

            if exit_triggered:
                # Calculate PnL for this trade
                pnl_points = self._calculate_pnl_points(state['entry_price'], exit_price, state['position'])

                # Update DataFrame at current_idx
                results.loc[current_idx, f'{name}_exit_price'] = exit_price
                results.loc[current_idx, f'{name}_pnl_points'] = pnl_points
                state['cumulative_pnl_points'] += pnl_points
                results.loc[current_idx, f'{name}_cumulative_pnl_points'] = state['cumulative_pnl_points']
                results.loc[current_idx, f'{name}_trade_id'] = state['trade_id'] # Mark exit with current trade ID
                results.loc[current_idx, f'{name}_active_trade'] = False # Mark trade as inactive *for the next bar*
                results.loc[current_idx, f'{name}_position'] = '' # Clear position *for the next bar*

                # Update state dictionary (resetting for next bar)
                state['active_trade'] = False
                state['position'] = ''
                state['entry_price'] = np.nan
                state['initial_sl'] = np.nan
                state['take_profit'] = np.nan
                state['trailing_sl'] = np.nan
                state['highest_high_since_entry'] = np.nan
                state['lowest_low_since_entry'] = np.nan
                # Keep trade_id unchanged until next entry

            else:
                # If no exit, carry forward state in DataFrame
                results.loc[current_idx, f'{name}_active_trade'] = True
                results.loc[current_idx, f'{name}_position'] = state['position']
                results.loc[current_idx, f'{name}_entry_price'] = state['entry_price']
                results.loc[current_idx, f'{name}_sl'] = state['initial_sl'] # Store initial SL
                results.loc[current_idx, f'{name}_tp'] = state['take_profit']
                results.loc[current_idx, f'{name}_trailing_sl'] = state['trailing_sl'] # Store updated trailing SL
                results.loc[current_idx, f'{name}_trade_id'] = state['trade_id']
                results.loc[current_idx, f'{name}_cumulative_pnl_points'] = state['cumulative_pnl_points'] # Carry forward cumulative PnL


        # --- 2. Check for New Entries (Only if not currently active) ---
        elif signal in ['buy', 'sell']: # Check signal from strategy function
            entry_price_slippage, sl, tp, atr_at_entry = self._calculate_entry_sl_tp(current_row, signal, atr_stop_multiplier, atr_target_multiplier)

            if pd.notna(entry_price_slippage): # Ensure entry is possible
                # Update DataFrame at current_idx
                results.loc[current_idx, f'{name}_signal'] = signal
                results.loc[current_idx, f'{name}_position'] = 'long' if signal == 'buy' else 'short'
                results.loc[current_idx, f'{name}_entry_price'] = entry_price_slippage
                results.loc[current_idx, f'{name}_sl'] = sl
                results.loc[current_idx, f'{name}_tp'] = tp
                results.loc[current_idx, f'{name}_atr_at_entry'] = atr_at_entry
                results.loc[current_idx, f'{name}_active_trade'] = True
                state['trade_id'] += 1 # Increment trade ID for the new trade
                results.loc[current_idx, f'{name}_trade_id'] = state['trade_id']
                results.loc[current_idx, f'{name}_cumulative_pnl_points'] = state['cumulative_pnl_points'] # Carry forward cumulative PnL

                # Update state dictionary
                state['active_trade'] = True
                state['position'] = 'long' if signal == 'buy' else 'short'
                state['entry_price'] = entry_price_slippage
                state['initial_sl'] = sl
                state['take_profit'] = tp
                state['trailing_sl'] = sl # Initial trailing SL starts at initial SL
                state['highest_high_since_entry'] = current_row['high'] if signal == 'buy' else np.nan
                state['lowest_low_since_entry'] = current_row['low'] if signal == 'short' else np.nan

                # Log Entry
                logger.info(f"🚀 ENTRY [{name}]: {state['position'].upper()} at {entry_price_slippage:.2f} "
                            f"(SL: {sl:.2f}, TP: {tp:.2f}) on {current_idx.strftime('%Y-%m-%d %H:%M')}")

            else:
                 # If entry calculation failed (e.g., no ATR), log hold and carry state
                 results.loc[current_idx, f'{name}_signal'] = 'hold'
                 results.loc[current_idx, f'{name}_cumulative_pnl_points'] = state['cumulative_pnl_points']

        # --- 3. No Active Trade and No Entry Signal ---
        else:
            results.loc[current_idx, f'{name}_signal'] = 'hold'
            results.loc[current_idx, f'{name}_cumulative_pnl_points'] = state['cumulative_pnl_points'] # Carry forward cumulative PnL
            # Ensure other state columns are NaN or default for this bar if not active
            results.loc[current_idx, f'{name}_active_trade'] = False
            results.loc[current_idx, f'{name}_position'] = ''


    def _update_trailing_stop(self, state: Dict[str, Any], current_row: pd.Series, name: str, current_idx) -> None:
        """Update the trailing stop loss based on price movement."""
        if not state['active_trade'] or pd.isna(state['trailing_sl']):
            return

        atr = current_row.get('atr', np.nan)
        if pd.isna(atr):
            # logger.warning(f"ATR NaN during trailing stop update for {name} at {current_idx}. Stop not updated.") # Can be noisy
            return

        # Use same ATR multiplier as initial stop for simplicity, could be parameterized
        # TODO: Parameterize trailing stop ATR multiplier separately if needed
        atr_multiplier = (state['initial_sl'] - state['entry_price']) / state['atr_at_entry'] if state['position']=='short' else (state['entry_price'] - state['initial_sl']) / state['atr_at_entry']
        if pd.isna(atr_multiplier) or state['atr_at_entry'] == 0:
             atr_multiplier = 1.5 # Fallback multiplier


        new_trailing_sl = state['trailing_sl'] # Start with the current value

        if state['position'] == 'long':
            # Update highest high since entry
            if pd.isna(state['highest_high_since_entry']) or current_row['high'] > state['highest_high_since_entry']:
                state['highest_high_since_entry'] = current_row['high']

            # Calculate potential new stop based on highest high
            potential_stop = state['highest_high_since_entry'] - atr * atr_multiplier
            # Only move stop up (never down)
            if potential_stop > new_trailing_sl:
                new_trailing_sl = potential_stop
                # Log Trailing SL Update
                logger.debug(f"🔁 Trailing SL Update [{name}]: LONG stop moved to {new_trailing_sl:.2f} on {current_idx.strftime('%Y-%m-%d %H:%M')}")


        elif state['position'] == 'short':
            # Update lowest low since entry
            if pd.isna(state['lowest_low_since_entry']) or current_row['low'] < state['lowest_low_since_entry']:
                state['lowest_low_since_entry'] = current_row['low']

            # Calculate potential new stop based on lowest low
            potential_stop = state['lowest_low_since_entry'] + atr * atr_multiplier
            # Only move stop down (never up)
            if potential_stop < new_trailing_sl:
                new_trailing_sl = potential_stop
                 # Log Trailing SL Update
                logger.debug(f"🔁 Trailing SL Update [{name}]: SHORT stop moved to {new_trailing_sl:.2f} on {current_idx.strftime('%Y-%m-%d %H:%M')}")


        # Update the state
        state['trailing_sl'] = new_trailing_sl


    def _check_exits(self, state: Dict[str, Any], current_row: pd.Series, signal: str, name: str, current_idx) -> Tuple[bool, float, str]:
        """Check SL, TP, Trailing SL, and Signal Reversal for exit. Returns (triggered, exit_price, reason)."""
        if not state['active_trade']:
            return False, np.nan, ""

        position_type = state['position']
        initial_stop_loss = state['initial_sl']
        trailing_stop_loss = state['trailing_sl']
        take_profit = state['take_profit']

        exit_triggered = False
        exit_price = np.nan
        exit_reason = ""

        # --- Priority: 1. Stop Losses (Initial or Trailing), 2. Take Profit, 3. Signal Reversal ---

        # Check Trailing Stop Loss first (if different from initial)
        if pd.notna(trailing_stop_loss) and trailing_stop_loss != initial_stop_loss:
             if position_type == 'long' and current_row['low'] <= trailing_stop_loss:
                 exit_triggered = True
                 exit_price = trailing_stop_loss # Exit at stop price
                 exit_reason = "Trailing SL"
             elif position_type == 'short' and current_row['high'] >= trailing_stop_loss:
                 exit_triggered = True
                 exit_price = trailing_stop_loss # Exit at stop price
                 exit_reason = "Trailing SL"

        # Check Initial Stop Loss (if not already triggered by trailing)
        if not exit_triggered and pd.notna(initial_stop_loss):
            if position_type == 'long' and current_row['low'] <= initial_stop_loss:
                exit_triggered = True
                exit_price = initial_stop_loss
                exit_reason = "Initial SL"
            elif position_type == 'short' and current_row['high'] >= initial_stop_loss:
                exit_triggered = True
                exit_price = initial_stop_loss
                exit_reason = "Initial SL"

        # Check Take Profit (if not already stopped out)
        if not exit_triggered and pd.notna(take_profit):
             if position_type == 'long' and current_row['high'] >= take_profit:
                 exit_triggered = True
                 exit_price = take_profit # Exit at TP price
                 exit_reason = "Take Profit"
             elif position_type == 'short' and current_row['low'] <= take_profit:
                 exit_triggered = True
                 exit_price = take_profit # Exit at TP price
                 exit_reason = "Take Profit"

        # Check for Signal Reversal (if not already exited)
        if not exit_triggered:
            if (position_type == 'long' and signal == 'sell') or \
               (position_type == 'short' and signal == 'buy'):
                exit_triggered = True
                exit_price = self._apply_slippage(current_row['close'], 'sell' if position_type == 'long' else 'buy')
                exit_reason = "Signal Reversal"

        # Log Exit
        if exit_triggered:
             pnl_points = self._calculate_pnl_points(state['entry_price'], exit_price, state['position']) # Calculate PnL for logging
             logger.info(f"🏁 EXIT [{name}]: {position_type.upper()} via {exit_reason} at {exit_price:.2f}. "
                         f"PnL Points: {pnl_points:.2f} on {current_idx.strftime('%Y-%m-%d %H:%M')}")

        return exit_triggered, exit_price, exit_reason


    def _calculate_entry_sl_tp(self, current_row: pd.Series, signal: str,
                              atr_stop_multiplier: float, atr_target_multiplier: float) -> Tuple[float, float, float, float]:
        """Calculate entry price, stop loss, and take profit."""
        atr = current_row.get('atr', np.nan)
        if pd.isna(atr):
            return np.nan, np.nan, np.nan, np.nan # Cannot calculate without ATR

        entry_price_slippage = self._apply_slippage(current_row['close'], signal)
        atr_at_entry = atr # Store the ATR value used

        if signal == 'buy':
            sl = entry_price_slippage - atr * atr_stop_multiplier
            tp = entry_price_slippage + atr * atr_target_multiplier
        else: # sell
            sl = entry_price_slippage + atr * atr_stop_multiplier
            tp = entry_price_slippage - atr * atr_target_multiplier

        return entry_price_slippage, sl, tp, atr_at_entry


    def _calculate_pnl_points(self, entry_price: float, exit_price: float, position_type: str) -> float:
        """Calculate P&L for a trade in price points, including simulated commission."""
        if pd.isna(entry_price) or pd.isna(exit_price):
             return 0.0

        # Calculate gross PnL in points
        if position_type == 'long':
            gross_pnl_points = exit_price - entry_price
        elif position_type == 'short':
            gross_pnl_points = entry_price - exit_price
        else:
            return 0.0

        # Simulate commission cost in points (approximate)
        # This assumes commission is a % of the entry/exit value
        commission_points_entry = entry_price * self.commission_pct
        commission_points_exit = exit_price * self.commission_pct
        total_commission_points = commission_points_entry + commission_points_exit

        net_pnl_points = gross_pnl_points - total_commission_points
        return net_pnl_points

    # --- Slippage ---
    def _apply_slippage(self, price: float, signal: str) -> float:
        """Apply slippage to trade execution price"""
        if pd.isna(price): return np.nan
        slippage_amount = price * self.slippage_pct
        if signal == 'buy':
            return price + slippage_amount
        elif signal == 'sell':
            return price - slippage_amount
        return price

    # --- Performance Metrics ---
    def _calculate_performance_metrics(self) -> None:
        """Calculate comprehensive performance metrics for all strategies"""
        if self.results_df is None:
            logger.error("Cannot calculate metrics: No backtest results available.")
            self.performance_metrics = {}
            return

        logger.info("Calculating performance metrics for all strategies...")
        metrics = {}
        for name in self.strategies.keys():
            try:
                trade_results = self._extract_trades(name)
                if trade_results.empty:
                    logger.warning(f"No trades extracted for strategy '{name}'. Skipping metrics.")
                    metrics[name] = {'total_trades': 0}
                    continue

                # Use PnL points for calculations
                pnl_points = trade_results['pnl_points']
                wins = pnl_points[pnl_points > 0]
                losses = pnl_points[pnl_points <= 0]

                # Calculate equity curve based on cumulative points PnL
                equity_curve_points = self.results_df[f'{name}_cumulative_pnl_points'].dropna()

                total_trades = len(trade_results)
                metrics[name] = {
                    'total_trades': total_trades,
                    'win_rate': (len(wins) / total_trades * 100) if total_trades > 0 else 0,
                    'avg_win_points': wins.mean() if not wins.empty else 0,
                    'avg_loss_points': losses.mean() if not losses.empty else 0,
                    'total_pnl_points': pnl_points.sum(),
                    'profit_factor': self._calculate_profit_factor(pnl_points),
                    'max_drawdown_points': self._calculate_max_drawdown_points(equity_curve_points),
                    # Add more metrics as needed (e.g., Sharpe based on returns if position sizing was implemented)
                }
                # Example: Expectancy in points
                avg_win = metrics[name]['avg_win_points']
                avg_loss = metrics[name]['avg_loss_points'] # Note: avg_loss is typically negative
                win_rate_dec = metrics[name]['win_rate'] / 100.0
                metrics[name]['expectancy_points'] = (avg_win * win_rate_dec) + (avg_loss * (1 - win_rate_dec))


            except Exception as e:
                logger.error(f"Failed to calculate metrics for strategy '{name}': {e}", exc_info=True)
                metrics[name] = {'total_trades': 'Error'}

        self.performance_metrics = metrics
        logger.info("Finished calculating performance metrics.")
        return metrics

    def _extract_trades(self, strategy_name: str) -> pd.DataFrame:
        """Extract completed trades for a strategy, using trade_id."""
        df = self.results_df
        entry_col = f'{strategy_name}_entry_price'
        exit_col = f'{strategy_name}_exit_price'
        id_col = f'{strategy_name}_trade_id'
        pnl_col = f'{strategy_name}_pnl_points'
        pos_col = f'{strategy_name}_position'
        cum_pnl_col = f'{strategy_name}_cumulative_pnl_points'

        # Find rows with exits
        exit_rows = df[df[exit_col].notna()].copy()
        if exit_rows.empty:
            return pd.DataFrame() # No trades completed

        trades = []
        processed_trade_ids = set()

        for exit_idx, exit_row in exit_rows.iterrows():
            trade_id = exit_row[id_col]
            # Ensure we process each trade_id only once (in case of multiple rows per exit?)
            if trade_id in processed_trade_ids or trade_id == 0:
                continue

            # Find the corresponding entry row(s) for this trade_id
            entry_rows = df[(df[id_col] == trade_id) & df[entry_col].notna()]

            if entry_rows.empty:
                logger.warning(f"Could not find entry for trade_id {trade_id} of strategy '{strategy_name}' ending at {exit_idx}. Skipping trade.")
                continue

            # Assume the first row with this trade_id and a valid entry price is the entry
            entry_row = entry_rows.iloc[0]
            entry_idx = entry_row.name

            trade = {
                'trade_id': trade_id,
                'entry_time': entry_idx,
                'exit_time': exit_idx,
                'position': entry_row[pos_col], # Position type at entry
                'entry_price': entry_row[entry_col],
                'exit_price': exit_row[exit_col],
                'duration_bars': df.index.get_loc(exit_idx) - df.index.get_loc(entry_idx),
                'pnl_points': exit_row[pnl_col], # PnL recorded at exit bar
                'cumulative_pnl_points': exit_row[cum_pnl_col] # Cum PnL at exit
            }
            trades.append(trade)
            processed_trade_ids.add(trade_id)

        return pd.DataFrame(trades)


    def _calculate_profit_factor(self, pnl_points: pd.Series) -> float:
        """Calculate profit factor (gross wins / gross losses in points)"""
        wins = pnl_points[pnl_points > 0].sum()
        losses = abs(pnl_points[pnl_points <= 0].sum())
        if losses == 0:
            return np.inf if wins > 0 else 1.0 # Avoid division by zero
        return wins / losses

    def _calculate_max_drawdown_points(self, cumulative_pnl_points: pd.Series) -> float:
        """Calculate maximum drawdown from peak in points."""
        if cumulative_pnl_points.empty:
            return 0.0
        peak = cumulative_pnl_points.cummax()
        drawdown = peak - cumulative_pnl_points
        return drawdown.max()


    # --- Reporting ---
    def generate_report(self, output_dir: str = 'backtest_results') -> Optional[Dict[str, Path]]:
        """Generate comprehensive report with metrics and visualizations"""
        if self.results_df is None:
             logger.error("Cannot generate report: No backtest results available.")
             return None
        if self.performance_metrics is None:
             logger.warning("Performance metrics not calculated. Calculating now.")
             self._calculate_performance_metrics()
             if self.performance_metrics is None: # Check again if calculation failed
                  logger.error("Cannot generate report: Metric calculation failed.")
                  return None

        logger.info(f"Generating backtest report in directory: {output_dir}")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_files = {}

        try:
            # Save raw results
            results_path = output_path / 'backtest_results_detailed.csv'
            self.results_df.to_csv(results_path)
            report_files['results_csv'] = results_path
            logger.info(f"Saved detailed results to {results_path}")

            # Save performance metrics summary
            metrics_path = output_path / 'performance_metrics_summary.csv'
            pd.DataFrame(self.performance_metrics).T.to_csv(metrics_path)
            report_files['metrics_csv'] = metrics_path
            logger.info(f"Saved performance metrics summary to {metrics_path}")

            # Generate visualizations
            logger.info("Generating equity curve plots...")
            equity_plot_path = self._generate_equity_curves(output_path)
            if equity_plot_path:
                report_files['equity_plot'] = equity_plot_path
            logger.info("Generating trade analysis plots...")
            trade_plot_paths = self._generate_trade_analysis(output_path)
            report_files.update(trade_plot_paths) # Add individual plot paths


            # Generate HTML report
            logger.info("Generating HTML report...")
            html_report_content = self._generate_html_report()
            html_path = output_path / 'backtest_summary_report.html'
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_report_content)
            report_files['html_report'] = html_path
            logger.info(f"HTML report saved to {html_path}")

            logger.info(f"Backtest report generation successful in {output_path}")
            return report_files

        except Exception as e:
            logger.error(f"Failed to generate report components: {e}", exc_info=True)
            return None


    def _generate_equity_curves(self, output_path: Path) -> Optional[Path]:
        """Plot equity curves (cumulative PnL points) for all strategies"""
        try:
            fig, ax = plt.subplots(figsize=(14, 7))
            has_data = False
            for name in self.strategies.keys():
                cum_pnl_col = f'{name}_cumulative_pnl_points'
                if cum_pnl_col in self.results_df.columns:
                    # Plot only where trades were potentially active or PnL changed
                    equity = self.results_df[cum_pnl_col].ffill().fillna(0) # Forward fill and fill initial NaNs
                    if not equity.empty:
                         ax.plot(equity.index, equity, label=name, linewidth=1.5)
                         has_data = True

            if not has_data:
                logger.warning("No cumulative PnL data found to plot equity curves.")
                plt.close(fig)
                return None

            ax.set_title('Strategy Equity Curves (Cumulative PnL Points)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative PnL (Points)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            equity_path = output_path / 'equity_curves.png'
            plt.savefig(equity_path, dpi=150)
            plt.close(fig)
            logger.info(f"Saved equity curves plot to {equity_path}")
            return equity_path
        except Exception as e:
             logger.error(f"Failed to generate equity curve plot: {e}", exc_info=True)
             plt.close() # Ensure plot is closed on error
             return None


    def _generate_trade_analysis(self, output_path: Path) -> Dict[str, Path]:
        """Generate PnL distribution and cumulative PnL plots per strategy."""
        plot_paths = {}
        for name in self.strategies.keys():
            try:
                trades = self._extract_trades(name)
                if trades.empty:
                    logger.warning(f"No trades found for '{name}', skipping trade analysis plots.")
                    continue

                # PnL distribution
                fig1, ax1 = plt.subplots(figsize=(10, 5))
                ax1.hist(trades['pnl_points'], bins=30, edgecolor='black')
                ax1.set_title(f'{name} - Trade PnL (Points) Distribution')
                ax1.set_xlabel('PnL (Points)')
                ax1.set_ylabel('Frequency')
                ax1.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                pnl_dist_path = output_path / f'{name}_pnl_distribution.png'
                plt.savefig(pnl_dist_path, dpi=100)
                plt.close(fig1)
                plot_paths[f'{name}_pnl_distribution_plot'] = pnl_dist_path

                # Cumulative PnL over time (using trade exit times)
                fig2, ax2 = plt.subplots(figsize=(12, 5))
                ax2.plot(trades['exit_time'], trades['cumulative_pnl_points'], marker='.', linestyle='-', markersize=4)
                ax2.set_title(f'{name} - Cumulative PnL (Points) Over Time')
                ax2.set_xlabel('Exit Date')
                ax2.set_ylabel('Cumulative PnL (Points)')
                ax2.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                cum_pnl_path = output_path / f'{name}_cumulative_pnl_plot.png'
                plt.savefig(cum_pnl_path, dpi=100)
                plt.close(fig2)
                plot_paths[f'{name}_cumulative_pnl_plot'] = cum_pnl_path

                logger.info(f"Generated trade analysis plots for '{name}'.")

            except Exception as e:
                 logger.error(f"Failed to generate trade analysis plots for '{name}': {e}", exc_info=True)
                 plt.close() # Ensure plot is closed on error

        return plot_paths

    def _generate_html_report(self) -> str:
        """Generate comprehensive HTML report string."""
        if not self.performance_metrics:
            return "<html><body><h2>Error: No performance metrics calculated.</h2></body></html>"

        # --- Metrics Table ---
        try:
            metrics_df = pd.DataFrame(self.performance_metrics).T.fillna(0) # Fill NaN metrics with 0 for display
            # Format numerical columns for better readability
            metrics_to_format = {
                'total_pnl_points': '{:,.2f}',
                'avg_win_points': '{:,.2f}',
                'avg_loss_points': '{:,.2f}',
                'win_rate': '{:.2f}%',
                'profit_factor': '{:.2f}',
                'expectancy_points': '{:.2f}',
                'max_drawdown_points': '{:,.2f}'
            }
            metrics_display_df = metrics_df.copy()
            for col, fmt in metrics_to_format.items():
                 if col in metrics_display_df.columns:
                      metrics_display_df[col] = metrics_display_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')
            if 'total_trades' in metrics_display_df.columns:
                 metrics_display_df['total_trades'] = metrics_display_df['total_trades'].astype(int)

            metrics_html = metrics_display_df.to_html(classes='performance-table', border=1, justify='right')
        except Exception as e:
             logger.error(f"Error formatting metrics table for HTML: {e}", exc_info=True)
             metrics_html = "<p>Error displaying metrics table.</p>"


        # --- HTML Structure ---
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Multi-Strategy Backtest Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
                .container {{ max-width: 1200px; margin: auto; background-color: #fff; padding: 25px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 8px; }}
                h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; margin-top: 30px; }}
                h1 {{ text-align: center; margin-bottom: 20px; }}
                table.performance-table {{ border-collapse: collapse; width: 100%; margin-bottom: 25px; font-size: 0.9em; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: right; }}
                th {{ background-color: #3498db; color: white; text-align: center; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric-card {{ border: 1px solid #e0e0e0; border-radius: 6px; padding: 15px; margin-bottom: 20px; background-color: #fdfdfd; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
                img {{ max-width: 100%; height: auto; border-radius: 4px; margin-top: 10px; border: 1px solid #eee; }}
                .positive {{ color: #27ae60; font-weight: bold; }}
                .negative {{ color: #c0392b; font-weight: bold; }}
                .neutral {{ color: #7f8c8d; }}
                .timestamp {{ text-align: center; color: #7f8c8d; margin-bottom: 30px; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Multi-Strategy Backtest Report</h1>
                <p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

                <h2>Performance Summary</h2>
                <div class="metric-card">
                    {metrics_html}
                    <p style="font-size: 0.8em; color: #555;">Note: PnL and Drawdown are in price points unless position sizing was implemented.</p>
                </div>

                <h2>Equity Curves (Cumulative PnL Points)</h2>
                <div class="metric-card">
                    <img src="equity_curves.png" alt="Equity Curves Comparison">
                </div>

                <h2>Individual Strategy Analysis</h2>
                <div class="metric-grid">
        """

        # Add individual strategy cards
        for name in self.strategies.keys():
            if name not in self.performance_metrics or self.performance_metrics[name]['total_trades'] == 0:
                html += f"""
                    <div class="metric-card">
                        <h3>{name}</h3>
                        <p>No trades executed or error during metric calculation.</p>
                    </div>"""
                continue

            metrics = self.performance_metrics[name]
            win_rate_str = f"{metrics.get('win_rate', 0):.2f}%" if pd.notna(metrics.get('win_rate')) else 'N/A'
            pf_str = f"{metrics.get('profit_factor', 0):.2f}" if pd.notna(metrics.get('profit_factor')) else 'N/A'
            pnl_str = f"{metrics.get('total_pnl_points', 0):,.2f}" if pd.notna(metrics.get('total_pnl_points')) else 'N/A'

            html += f"""
                    <div class="metric-card">
                        <h3>{name}</h3>
                        <p><strong>Total Trades:</strong> {metrics.get('total_trades', 0)}</p>
                        <p><strong>Win Rate:</strong> {win_rate_str}</p>
                        <p><strong>Profit Factor:</strong> {pf_str}</p>
                        <p><strong>Total PnL (Points):</strong> {pnl_str}</p>
                        <details>
                            <summary>Show Plots</summary>
                            <img src="{name}_pnl_distribution.png" alt="{name} PnL Distribution">
                            <img src="{name}_cumulative_pnl_plot.png" alt="{name} Cumulative PnL">
                        </details>
                    </div>
            """

        html += """
                </div> </div> </body>
        </html>
        """
        return html


# --- Example Strategies (Keep your well-defined strategies here) ---
# Placeholder strategies matching the names used in run_full_backtest
def strategy_ema_crossover(current_row: pd.Series, data: pd.DataFrame = None) -> str:
    """Simple EMA 9/21 Crossover"""
    if pd.isna(current_row['ema_9']) or pd.isna(current_row['ema_21']):
        return 'hold'
    idx = data.index.get_loc(current_row.name)
    if idx == 0: return 'hold'
    prev_row = data.iloc[idx-1] # Use iloc for safety

    # Signal only on the bar the cross happens
    if current_row['ema_9'] > current_row['ema_21'] and prev_row['ema_9'] <= prev_row['ema_21']:
        return 'buy'
    elif current_row['ema_9'] < current_row['ema_21'] and prev_row['ema_9'] >= prev_row['ema_21']:
        return 'sell'
    return 'hold'

def strategy_rsi_threshold(current_row: pd.Series, data: pd.DataFrame = None) -> str:
    """RSI Overbought/Oversold on Crossover"""
    rsi_oversold = 35 # Adjusted threshold example
    rsi_overbought = 65 # Adjusted threshold example
    if pd.isna(current_row['rsi']):
        return 'hold'
    idx = data.index.get_loc(current_row.name)
    if idx == 0: return 'hold'
    prev_row = data.iloc[idx-1]

    # Signal only on the bar the threshold is crossed
    if current_row['rsi'] > rsi_oversold and prev_row['rsi'] <= rsi_oversold:
         return 'buy'
    elif current_row['rsi'] < rsi_overbought and prev_row['rsi'] >= rsi_overbought:
         return 'sell'
    return 'hold'

def strategy_macd_cross(current_row: pd.Series, data: pd.DataFrame = None) -> str:
    """ MACD Line crosses Signal Line """
    if pd.isna(current_row['macd']) or pd.isna(current_row['macd_signal']):
        return 'hold'

    idx = data.index.get_loc(current_row.name)
    if idx == 0: return 'hold'
    prev_row = data.iloc[idx-1]

    if current_row['macd'] > current_row['macd_signal'] and prev_row['macd'] <= prev_row['macd_signal']:
        return 'buy'
    elif current_row['macd'] < current_row['macd_signal'] and prev_row['macd'] >= prev_row['macd_signal']:
        return 'sell'
    return 'hold'

def strategy_bb_squeeze_breakout(current_row: pd.Series, data: pd.DataFrame = None) -> str:
    """ Bollinger Band Squeeze Breakout (simplified) """
    if pd.isna(current_row['bollinger_bandwidth']) or pd.isna(current_row['close']) or pd.isna(current_row['bollinger_upper']) or pd.isna(current_row['bollinger_lower']):
       return 'hold'

    idx = data.index.get_loc(current_row.name)
    squeeze_lookback = 20 # How far back to check for low bandwidth
    if idx < squeeze_lookback: return 'hold'
    prev_rows = data.iloc[idx-squeeze_lookback:idx]
    # Define squeeze: bandwidth is in the lower quantile of recent values
    bandwidth_threshold = prev_rows['bollinger_bandwidth'].quantile(0.15) # Example: Bottom 15%

    # Check if the *previous* bar was in a squeeze
    prev_bandwidth = data.iloc[idx-1]['bollinger_bandwidth']
    if pd.isna(prev_bandwidth): return 'hold'

    in_squeeze_prev = prev_bandwidth < bandwidth_threshold
    breaking_out_up = current_row['close'] > current_row['bollinger_upper']
    breaking_out_down = current_row['close'] < current_row['bollinger_lower']

    if in_squeeze_prev and breaking_out_up:
        return 'buy'
    elif in_squeeze_prev and breaking_out_down:
        return 'sell'
    return 'hold'

def strategy_combined_momentum(current_row: pd.Series, data: pd.DataFrame = None) -> str:
    """ Combine EMA trend and RSI confirmation on Cross """
    if pd.isna(current_row['ema_9']) or pd.isna(current_row['ema_21']) or pd.isna(current_row['rsi']):
        return 'hold'

    # EMA Trend Check
    ema_crossed_up = False
    ema_crossed_down = False
    idx = data.index.get_loc(current_row.name)
    if idx > 0:
        prev_row = data.iloc[idx-1]
        ema_crossed_up = current_row['ema_9'] > current_row['ema_21'] and prev_row['ema_9'] <= prev_row['ema_21']
        ema_crossed_down = current_row['ema_9'] < current_row['ema_21'] and prev_row['ema_9'] >= prev_row['ema_21']

    # RSI Confirmation Check
    rsi_confirm_buy = current_row['rsi'] > 50 # RSI above neutral
    rsi_confirm_sell = current_row['rsi'] < 50 # RSI below neutral

    if ema_crossed_up and rsi_confirm_buy:
        return 'buy'
    elif ema_crossed_down and rsi_confirm_sell:
        return 'sell'
    return 'hold'

# --- Main Execution Function ---
def run_full_backtest_workflow(data_paths: Dict[str, str], output_base_dir: str = 'backtest_runs'):
    """
    Runs the complete backtest workflow for multiple data files (timeframes).

    Args:
        data_paths: Dictionary mapping timeframe description (e.g., "3min", "5min")
                    to the corresponding CSV file path.
        output_base_dir: Base directory to store results for each timeframe.
    """
    base_path = Path(output_base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    all_run_metrics = {}

    for timeframe, data_path_str in data_paths.items():
        logger.info(f"\n===== Starting Backtest for Timeframe: {timeframe} =====")
        data_path = Path(data_path_str)
        timeframe_output_dir = base_path / timeframe # Create subfolder for each timeframe

        if not data_path.exists():
            logger.error(f"Data file not found for {timeframe}: {data_path}. Skipping.")
            continue

        try:
            # Load data
            logger.info(f"Loading data from {data_path}")
            # Ensure datetime index is correctly parsed
            raw_data = pd.read_csv(data_path, index_col=0, parse_dates=True) # Assuming first col is datetime index
            if raw_data.index.name != 'datetime': # Basic check if index name is not datetime
                 raw_data.index = pd.to_datetime(raw_data.index) # Attempt conversion
                 raw_data.index.name = 'datetime'


            # Define strategies (could be defined outside loop if always the same)
            strategies = {
                'EMA_Crossover': strategy_ema_crossover,
                'RSI_Threshold': strategy_rsi_threshold,
                'MACD_Cross': strategy_macd_cross,
                'BB_Breakout': strategy_bb_squeeze_breakout,
                'Combined_Mom': strategy_combined_momentum,
            }

            # Initialize and run backtest
            logger.info(f"Initializing backtester for {timeframe}...")
            backtester = EnhancedMultiStrategyBacktester(strategies) # Use default capital/commission/slippage for now

            logger.info(f"Running backtest for {timeframe}...")
            backtester.run_backtest(raw_data, use_trailing_stop=True) # Enable trailing stop

            # Generate report
            logger.info(f"Generating report for {timeframe}...")
            report_files = backtester.generate_report(output_dir=str(timeframe_output_dir))

            if report_files:
                logger.info(f"Backtest for {timeframe} complete. Report generated in {timeframe_output_dir}")
                all_run_metrics[timeframe] = backtester.performance_metrics # Store metrics for overall summary
            else:
                 logger.error(f"Report generation failed for {timeframe}.")

        except Exception as e:
            logger.error(f"Backtest failed for timeframe {timeframe}: {str(e)}", exc_info=True)

    # --- Optional: Generate an overall summary comparing timeframes ---
    if all_run_metrics:
         logger.info("\n===== Overall Timeframe Comparison =====")
         # Create a DataFrame from the collected metrics
         summary_df = pd.DataFrame({tf: pd.Series(metrics.get('total_pnl_points', 0) for metrics in tf_metrics.values())
                                     for tf, tf_metrics in all_run_metrics.items()}) # Example: Compare total PnL points
         # Need better aggregation if metrics structure is nested
         # This part needs refinement based on the exact structure of performance_metrics

         # For a simpler text summary:
         for timeframe, metrics_dict in all_run_metrics.items():
              print(f"\n--- {timeframe} Summary ---")
              if metrics_dict:
                   for strategy_name, strategy_metrics in metrics_dict.items():
                        pnl = strategy_metrics.get('total_pnl_points', 'N/A')
                        trades = strategy_metrics.get('total_trades', 'N/A')
                        win_rate = strategy_metrics.get('win_rate', 'N/A')
                        pf = strategy_metrics.get('profit_factor', 'N/A')
                        print(f"  Strategy: {strategy_name:<15} | Trades: {trades:<5} | Win Rate: {win_rate:<6.2f}% | Profit Factor: {pf:<5.2f} | PnL Points: {pnl:<.2f}")
              else:
                   print(f"  No metrics calculated for {timeframe}.")

         # Save the aggregated metrics if desired (requires better structuring above)
         # overall_summary_path = base_path / "overall_timeframe_summary.csv"
         # summary_df.to_csv(overall_summary_path)
         # logger.info(f"Overall timeframe comparison saved to {overall_summary_path}")


if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Define the paths to your different timeframe datasets
    # The keys ("3min", "5min", etc.) will be used for folder names and reports.
    # Ensure these CSV files have a datetime index and ohlcv columns (lowercase).
    data_files_by_timeframe = {
        "3min": "path/to/your/nifty_data_3min.csv", # <--- CHANGE THIS PATH
        "5min": "path/to/your/nifty_data_5min.csv", # <--- CHANGE THIS PATH
        "15min": "path/to/your/nifty_data_15min.csv", # <--- CHANGE THIS PATH
        # Add more timeframes as needed
    }

    # --- Check if files exist ---
    valid_data_files = {}
    for tf, path_str in data_files_by_timeframe.items():
         if Path(path_str).is_file():
              valid_data_files[tf] = path_str
         else:
              logger.warning(f"Data file for timeframe '{tf}' not found at: {path_str}. It will be skipped.")

    if not valid_data_files:
         logger.error("No valid data files found. Please check the paths in 'data_files_by_timeframe'. Exiting.")
         sys.exit(1)

    # --- Execution ---
    run_full_backtest_workflow(
        data_paths=valid_data_files,
        output_base_dir='multi_timeframe_backtest_results' # Main output folder
    )