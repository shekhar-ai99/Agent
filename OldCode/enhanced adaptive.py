from io import StringIO
import pandas as pd
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# --- Strategy Implementation ---
class AdaptiveIntradayStrategy(Strategy):
    # Indicator Parameters (as per description)
    ema_fast_period = 9
    ema_slow_period = 21
    rsi_period = 10
    atr_period = 14
    st_atr_period = 10 # SuperTrend ATR period
    st_multiplier = 3.0
    bb_period = 20
    bb_std_dev = 2.0
    atr_sma_period = 5 # For ATR stability check

    # --- Adaptive SL/TP Multipliers (PLACEHOLDERS - Adjust based on regime logic) ---
    # These are examples; link them properly to your regime detection logic
    sl_multiplier_low_vol = 0.5
    sl_multiplier_mod_vol = 1.0 # Example value, description was inconsistent
    sl_multiplier_high_vol = 1.5

    tp_multiplier_low_vol = 1.5
    tp_multiplier_mod_vol = 2.0
    tp_multiplier_high_vol = 2.5 # Using lower end of 2.5-3 range

    # Trailing Stop Parameters
    trailing_sl_activation_atr = 1.0
    trailing_sl_tighten_atr = 0.7 # As per example

    def init(self):
        # Calculate Indicators using pandas_ta
        self.ema_fast = self.I(ta.ema, pd.Series(self.data.Close), length=self.ema_fast_period)
        self.ema_slow = self.I(ta.ema, pd.Series(self.data.Close), length=self.ema_slow_period)
        self.rsi = self.I(ta.rsi, pd.Series(self.data.Close), length=self.rsi_period)
        self.atr = self.I(ta.atr, pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), length=self.atr_period)
        
        # Calculate SuperTrend manually or using pandas_ta if available and matches logic
        # pandas_ta.supertrend might need verification against the exact logic if critical
        # Using a simplified manual calculation for clarity:
        hl2 = (self.data.High + self.data.Low) / 2
        atr_st = ta.atr(pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), length=self.st_atr_period)
        matr = self.st_multiplier * atr_st
        upper_band = hl2 + matr
        lower_band = hl2 - matr
        
        # Placeholder for SuperTrend line calculation (requires iterative logic)
        # For backtesting.py, it's often easier to pre-calculate complex indicators like SuperTrend
        # and add them to the input DataFrame. Here we use pandas_ta's version.
        st_df = ta.supertrend(pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), length=self.st_atr_period, multiplier=self.st_multiplier)
        # Extract the SuperTrend line column (adjust column name based on pandas_ta version)
        st_col_name = f'SUPERT_{self.st_atr_period}_{self.st_multiplier}' # Common name format
        if st_col_name in st_df.columns:
             self.supertrend = self.I(lambda x: st_df[st_col_name], pd.Series(self.data.Close)) # Pass series for I function
        else:
             # Handle case where column name might differ or calculation failed
             print(f"Warning: SuperTrend column '{st_col_name}' not found in pandas_ta output. Using NaN.")
             self.supertrend = self.I(lambda x: pd.Series([float('nan')] * len(x)), pd.Series(self.data.Close))


        # Calculate Bollinger Bands
        bbands = self.I(ta.bbands, pd.Series(self.data.Close), length=self.bb_period, std=self.bb_std_dev)
        self.bb_upper = bbands
        self.bb_middle = bbands
        self.bb_lower = bbands
        self.bb_width = (self.bb_upper - self.bb_lower) / self.bb_middle # For regime detection

        # Calculate RSI Slope
        self.rsi_slope = self.I(lambda x: x.diff(), self.rsi) # Simple difference

        # Calculate ATR SMA for stability check
        self.atr_sma = self.I(ta.sma, self.atr, length=self.atr_sma_period)

    def _detect_regime(self):
        """
        Placeholder function to detect market regime.
        *** Replace this with your specific regime detection logic. ***
        This example uses ATR and Bollinger Band Width.
        """
        current_atr = self.atr[-1]
        avg_atr = self.atr[-20:].mean() # Example: Compare current ATR to recent average
        current_bbw = self.bb_width[-1]
        avg_bbw = self.bb_width[-20:].mean() # Example: Compare current BBW to recent average

        # --- EXAMPLE LOGIC (Replace with yours) ---
        if current_atr > avg_atr * 1.2 and current_bbw > avg_bbw * 1.1: # High ATR and widening bands
            return "Volatile"
        elif current_atr < avg_atr * 0.8 and current_bbw < avg_bbw * 0.9: # Low ATR and narrowing bands
            return "Range-bound"
        else: # Otherwise, assume trending (needs better logic)
            # A proper trending check might involve ADX or sustained EMA separation
            return "Trending"
        # --- END EXAMPLE LOGIC ---

    def next(self):
        price = self.data.Close[-1]
        current_atr = self.atr[-1]
        
        # Ensure we have enough data for indicators
        if len(self.data.Close) < max(self.ema_slow_period, self.rsi_period, self.atr_period, self.bb_period, self.st_atr_period, self.atr_sma_period) + 2:
             return
             
        # --- Regime Detection ---
        regime = self._detect_regime()

        # --- Adaptive SL/TP based on Regime ---
        # *** Link these multipliers to the detected regime ***
        if regime == "Volatile":
            sl_mult = self.sl_multiplier_high_vol
            tp_mult = self.tp_multiplier_high_vol # Or potentially smaller as per description
        elif regime == "Range-bound":
            sl_mult = self.sl_multiplier_low_vol
            tp_mult = self.tp_multiplier_low_vol
        else: # Trending (Default)
            sl_mult = self.sl_multiplier_mod_vol
            tp_mult = self.tp_multiplier_mod_vol

        stop_loss_dist = sl_mult * current_atr
        take_profit_dist = tp_mult * current_atr

        # --- Entry Conditions ---
        # ATR Stability Check
        atr_stable = current_atr >= self.atr_sma[-1]

        # Long Entry Conditions
        long_ema_cross = crossover(self.ema_fast, self.ema_slow)
        long_rsi_slope = self.rsi_slope[-1] > 0
        long_supertrend = price > self.supertrend[-1]
        long_bb_mid = price > self.bb_middle[-1]

        # Short Entry Conditions
        short_ema_cross = crossover(self.ema_slow, self.ema_fast)
        short_rsi_slope = self.rsi_slope[-1] < 0
        short_supertrend = price < self.supertrend[-1]
        short_bb_mid = price < self.bb_middle[-1]

        # --- Trade Execution ---
        # Exit existing position if reverse EMA cross occurs
        if self.position.is_long and short_ema_cross:
            self.position.close()
        elif self.position.is_short and long_ema_cross:
            self.position.close()

        # Check for new entries only if not already in a position
        if not self.position:
            if long_ema_cross and long_rsi_slope and long_supertrend and long_bb_mid and atr_stable:
                # --- Regime-Specific Entry Logic (Placeholder) ---
                # Add checks here if entry rules change based on regime
                # e.g., if regime == "Range-bound": skip trend entry
                if regime!= "Range-bound": # Example: Don't take trend trades in range
                    sl = price - stop_loss_dist
                    tp = price + take_profit_dist
                    self.buy(sl=sl, tp=tp)

            elif short_ema_cross and short_rsi_slope and short_supertrend and short_bb_mid and atr_stable:
                 # --- Regime-Specific Entry Logic (Placeholder) ---
                 if regime!= "Range-bound": # Example: Don't take trend trades in range
                    sl = price + stop_loss_dist
                    tp = price - take_profit_dist
                    self.sell(sl=sl, tp=tp)

        # --- Trailing Stop Logic ---
        if self.position:
            trade = self.trades[-1] # Get the last trade associated with the current position
            
            # Check if trailing stop needs activation
            if not hasattr(trade, 'trailing_sl_active') or not trade.trailing_sl_active:
                 if self.position.is_long and price >= trade.entry_price + (self.trailing_sl_activation_atr * current_atr):
                     new_sl = trade.entry_price - (self.trailing_sl_tighten_atr * current_atr)
                     # Ensure new SL is profitable or at least better than original SL
                     if new_sl > trade.sl:
                         trade.sl = new_sl
                         trade.trailing_sl_active = True # Mark as active
                         # print(f"Trailing SL Activated (Long): New SL = {new_sl}") # Optional debug print
                 elif self.position.is_short and price <= trade.entry_price - (self.trailing_sl_activation_atr * current_atr):
                     new_sl = trade.entry_price + (self.trailing_sl_tighten_atr * current_atr)
                     # Ensure new SL is profitable or at least better than original SL
                     if new_sl < trade.sl:
                         trade.sl = new_sl
                         trade.trailing_sl_active = True # Mark as active
                         # print(f"Trailing SL Activated (Short): New SL = {new_sl}") # Optional debug print
            
            # --- Add further trailing logic here if needed ---
            # e.g., continuously trail by X * ATR below highest high / above lowest low
            # This part requires careful state management within backtesting.py


# --- Backtesting Execution ---

# 1. Load Your Data
# Replace 'your_banknifty_5min_data.csv' with your actual data file path
# Ensure data has 'Open', 'High', 'Low', 'Close', 'Volume' columns and is indexed by DateTime
try:
    # Example: Load data from CSV

    # --- Hardcoded Data ---
    data_string = """datetime,open,high,low,close,volume
    2025-04-29 09:15:00+05:30,24370.7,24442.25,24364.35,24439.25,0
    2025-04-29 09:20:00+05:30,24438.5,24455.05,24424.5,24453.7,0
    2025-04-29 09:25:00+05:30,24453.6,24457.65,24413.3,24417.1,0
    2025-04-29 09:30:00+05:30,24418.25,24452.9,24393.55,24440.2,0
    2025-04-29 09:35:00+05:30,24440.85,24442.5,24359.2,24369.45,0
    2025-04-29 09:40:00+05:30,24370.1,24387.05,24303.9,24316.6,0
    2025-04-29 09:45:00+05:30,24317.85,24364.5,24308.1,24315.55,0
    2025-04-29 09:50:00+05:30,24316.75,24340.05,24290.75,24329.1,0
    2025-04-29 09:55:00+05:30,24330.15,24395,24326.5,24390.25,0
    2025-04-29 10:00:00+05:30,24390,24396.15,24362.2,24378.3,0
    2025-04-29 10:05:00+05:30,24379.15,24379.75,24333.8,24344.65,0
    2025-04-29 10:10:00+05:30,24345.2,24365.1,24338.55,24351.55,0
    2025-04-29 10:15:00+05:30,24351.3,24354.1,24319,24319,0
    2025-04-29 10:20:00+05:30,24318.1,24347.1,24317.4,24346.4,0
    2025-04-29 10:25:00+05:30,24346.45,24348.15,24311.3,24317.8,0
    2025-04-29 10:30:00+05:30,24318.85,24329.7,24305.5,24314.95,0
    2025-04-29 10:35:00+05:30,24316.2,24335.25,24314.55,24331.5,0
    2025-04-29 10:40:00+05:30,24331.55,24333.9,24312.8,24321.5,0
    2025-04-29 10:45:00+05:30,24321.85,24352.35,24317.95,24350.7,0
    2025-04-29 10:50:00+05:30,24349.85,24353.7,24328.45,24330.85,0
    2025-04-29 10:55:00+05:30,24330.25,24349.95,24330.25,24333.8,0
    2025-04-29 11:00:00+05:30,24334.7,24351.9,24332.8,24351.9,0
    2025-04-29 11:05:00+05:30,24353.05,24369.85,24351.9,24367.1,0
    2025-04-29 11:10:00+05:30,24367.25,24369.7,24350.6,24356.05,0
    2025-04-29 11:15:00+05:30,24356.55,24358.5,24338.95,24348.8,0
    2025-04-29 11:20:00+05:30,24348.3,24360.35,24347.05,24348.95,0
    2025-04-29 11:25:00+05:30,24349.05,24358.75,24347.6,24349.25,0
    2025-04-29 11:30:00+05:30,24348.65,24351.6,24334.95,24347.65,0
    2025-04-29 11:35:00+05:30,24347.95,24353.6,24341.6,24346.45,0
    2025-04-29 11:40:00+05:30,24346.7,24356.8,24342.2,24354.4,0
    2025-04-29 11:45:00+05:30,24354.65,24355.8,24335.3,24340.15,0
    2025-04-29 11:50:00+05:30,24340.8,24340.85,24321,24322.95,0
    2025-04-29 11:55:00+05:30,24323.4,24329.1,24311.95,24313.25,0
    2025-04-29 12:00:00+05:30,24313.4,24322.5,24302.45,24322.05,0
    2025-04-29 12:05:00+05:30,24322.15,24323.75,24310.65,24315.5,0
    2025-04-29 12:10:00+05:30,24315.3,24322.3,24309.2,24316.6,0
    2025-04-29 12:15:00+05:30,24316.4,24321.6,24310.65,24313.1,0
    2025-04-29 12:20:00+05:30,24315.1,24315.85,24304.1,24313.65,0
    2025-04-29 12:25:00+05:30,24312.95,24317.25,24306.6,24309.75,0
    2025-04-29 12:30:00+05:30,24309.25,24330.8,24309.15,24329.95,0
    2025-04-29 12:35:00+05:30,24328.75,24347.95,24325.4,24341.35,0
    2025-04-29 12:40:00+05:30,24340.6,24346.8,24336.4,24341.85,0
    2025-04-29 12:45:00+05:30,24338.8,24344.85,24330.6,24339.55,0
    2025-04-29 12:50:00+05:30,24339.15,24358.35,24336.45,24357.9,0
    2025-04-29 12:55:00+05:30,24357.75,24363.55,24353,24358.95,0
    2025-04-29 13:00:00+05:30,24358.25,24369.5,24352.2,24368.4,0
    2025-04-29 13:05:00+05:30,24368.05,24368.5,24350.6,24360.1,0
    2025-04-29 13:10:00+05:30,24359.4,24363.55,24353.5,24356.85,0
    2025-04-29 13:15:00+05:30,24355.45,24359.35,24341.35,24354.1,0
    2025-04-29 13:20:00+05:30,24352.2,24363.15,24346.55,24362.9,0
    2025-04-29 13:25:00+05:30,24362.1,24376.05,24356,24366.05,0
    2025-04-29 13:30:00+05:30,24366.65,24371.7,24361.45,24366.8,0
    2025-04-29 13:35:00+05:30,24366.5,24368.8,24357.4,24366.15,0
    2025-04-29 13:40:00+05:30,24367.15,24372.7,24351,24353.9,0
    2025-04-29 13:45:00+05:30,24352.7,24366.95,24351.75,24363.45,0
    2025-04-29 13:50:00+05:30,24364,24368.75,24355.1,24355.65,0
    2025-04-29 13:55:00+05:30,24356.65,24363.4,24347.65,24358.25,0
    2025-04-29 14:00:00+05:30,24360.15,24368.3,24356.85,24363.55,0
    2025-04-29 14:05:00+05:30,24363.8,24375.5,24360.6,24365.2,0
    2025-04-29 14:10:00+05:30,24365.35,24374.8,24360.25,24365.95,0
    2025-04-29 14:15:00+05:30,24366.9,24368.3,24355.05,24361.65,0
    2025-04-29 14:20:00+05:30,24361.9,24367.55,24356.25,24357.15,0
    2025-04-29 14:25:00+05:30,24357.75,24359.7,24342.4,24346,0
    2025-04-29 14:30:00+05:30,24346.45,24350.5,24339,24343.45,0
    2025-04-29 14:35:00+05:30,24343.75,24361.95,24343.75,24348.1,0
    2025-04-29 14:40:00+05:30,24348.8,24360.7,24347.2,24356.45,0
    2025-04-29 14:45:00+05:30,24356.2,24357.95,24347.85,24352.95,0
    2025-04-29 14:50:00+05:30,24353.35,24358,24344.85,24346.85,0
    2025-04-29 14:55:00+05:30,24346.8,24355.7,24339.45,24340.2,0
    2025-04-29 15:00:00+05:30,24340.55,24347.8,24325.2,24331.4,0
    2025-04-29 15:05:00+05:30,24331.55,24336.5,24324.9,24332.7,0
    2025-04-29 15:10:00+05:30,24332.6,24335.6,24324.45,24326.45,0
    2025-04-29 15:15:00+05:30,24326.2,24341.45,24310.6,24339.65,0
    2025-04-29 15:20:00+05:30,24338.85,24348.5,24325.85,24333.1,0
    2025-04-29 15:25:00+05:30,24331.55,24340.85,24317.6,24325.45,0"""
    # data = pd.read_csv(StringIO(data_string))
    
    data = pd.read_csv('results/result_nifty_50_nse_alphatrend_5minute_20250501_001639.csv')
    data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
    data = data.dropna(subset=['datetime'])
    data = data.set_index('datetime')

    # Ensure column names match exactly (case-sensitive)
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume'] # Adjust if needed
    print("Data loaded successfully.")
    print(data.head())
    print(f"Data shape: {data.shape}")

    # Optional: Filter data for a specific period if needed
    # data = data['2023-01-01':'2023-12-31']

    # Check for sufficient data
    if len(data) < 100: # Need enough data for indicator calculations + backtest
        raise ValueError("Insufficient data for backtesting after loading/filtering.")

    # 2. Run the Backtest
    # bt = Backtest(data, AdaptiveIntradayStrategy, cash=100_000, commission=.002) # Example commission
    bt = Backtest(data, AdaptiveIntradayStrategy, cash=1_00_000, commission=.0002) # Lower commission example


    stats = bt.run()
    print("\n--- Backtest Results ---")
    print(stats)

    # Print individual trades (optional)
    # print("\n--- Trades ---")
    # print(stats['_trades'])

    # 3. Plot the Results
    print("\nGenerating plot...")
    bt.plot()
    print("Plot generated.")

except FileNotFoundError:
    print("Error: Data file not found. Please ensure 'your_banknifty_5min_data.csv' exists.")
except ValueError as ve:
     print(f"Data Error: {ve}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    # Potentially print more details during debugging
    # import traceback
    # traceback.print_exc()
