// @version=6
indicator (title="CS_HK Score Optimized (User Version Corrected)",shorttitle="CS Score v5+",overlay=true,max_lines_count=50,max_labels_count=50)

// === INPUTS ===

// --- Core Entry & Exit ---
grp_core = "Core Strategy Settings"
entry_score_threshold_long = input.float(6.0, "Entry Score Threshold (Long >)", minval=5.1, maxval=9.9, step=0.1, group=grp_core)
entry_score_threshold_short = input.float(4.0, "Entry Score Threshold (Short <)", minval=0.1, maxval=4.9, step=0.1, group=grp_core)
exit_score_drop_threshold = input.float(1.5, "Exit on Score Drop Threshold", minval=0.1, maxval=5.0, step=0.1, group=grp_core,tooltip="Exit if score drops below (5.0 - Threshold) for longs or above (5.0 + Threshold) for shorts")
use_fib_bounce_entry = input.bool(true, "Use Fib 0.5-0.618 Bounce Entry (Long)", group=grp_core,tooltip="Generate BUY signal on upward bounce from 0.5-0.618 Fib in uptrend")
use_fib_bounce_sell = input.bool(true, "Use Fib 0.5-0.382 Bounce Entry (Short)", group=grp_core,tooltip="Generate SELL signal on downward bounce from 0.5-0.382 Fib in downtrend")
fib_bounce_lookback = input.int(3, "Fib Bounce Price Lookback Bars", minval=1, maxval=10, group=grp_core,tooltip="Bars back to check for Fib zone touch")
fib_volume_mult = input.float(1.5, "Fib Bounce Volume Multiplier", minval=1.0, maxval=3.0, step=0.1, group=grp_core,tooltip="Require volume to exceed SMA(20) by this multiplier for Fibonacci bounce confirmation")
use_ema_bounce_buy = input.bool(true, title="Use EMA Bounce Entry (Long)?", group=grp_core)
use_ema_bounce_sell = input.bool(true, title="Use EMA Bounce Entry (Short)?", group=grp_core)
ema_bounce_lookback = input.int(2, title="EMA Bounce Price Lookback Bars", minval=1, maxval=5, group=grp_core)
ema_bounce_source = input.string("Fast EMA", title="EMA Source for Bounce", options=["Fast EMA", "Medium EMA"], group=grp_core)
use_bb_mid_bounce_buy = input.bool(true, title="Use BB Mid Bounce Entry (Long)?", group=grp_core)
use_bb_mid_bounce_sell = input.bool(true, title="Use BB Mid Bounce Entry (Short)?", group=grp_core)
bb_bounce_lookback = input.int(2, title="BB Mid Bounce Price Lookback Bars", minval=1, maxval=5, group=grp_core)
use_vol_breakout_buy = input.bool(true, title="Use Volume Breakout Entry (Long)?", group=grp_core)
use_vol_breakout_sell = input.bool(true, title="Use Volume Breakout Entry (Short)?", group=grp_core)
cooldown_bars = input.int(5, "Trade Cooldown Bars", minval=0, group=grp_core, tooltip="Minimum bars between the exit of one trade and the entry of the next.")
min_atr_factor = input.float(0.5, "Min ATR Factor for Entries", minval=0.1, maxval=2.0, step=0.1, group=grp_core,tooltip="Require ATR to be at least this multiple of its 20-period SMA for entry signals")
initial_capital = input.float(10000, "Initial Capital (for hypothetical size)", group=grp_core)
risk_percent = input.float(1.0, "Risk % per Trade", minval=0.1, maxval=5.0, step=0.1, group=grp_core,tooltip="Percentage of hypothetical equity to risk per trade, used for position sizing")

// --- Weak Signal Filter Settings ---
grp_skip = "Weak Signal Filter Settings"
use_skip_trade_filter = input.bool(true, "Enable Weak Signal Filter?", group=grp_skip,tooltip="If enabled, signals on bars matching the criteria below will be skipped.")
skip_atr_factor = input.float(0.3, "Min Price Move (ATR Factor)", minval=0.0, step=0.05, group=grp_skip,tooltip="Skip signal if |close - open| is less than ATR * this factor. Set to 0 to disable.")
skip_vol_ma_len = input.int(20, "Volume MA Length for Filter", minval=1, group=grp_skip,tooltip="The lookback period for the Simple Moving Average of volume used in the filter.")
skip_vol_multiplier = input.float(1.2, "Min Volume Spike (x MA)", minval=0.0, step=0.1, group=grp_skip,tooltip="Skip signal if volume is less than its MA * this multiplier. Set to 0 to disable.")

// --- EMAs ---
grp_ema = "EMA Settings"
ema_fast_len = input.int(9, "Fast EMA Length", minval=1, group=grp_ema)
ema_med_len = input.int(14, "Medium EMA Length", minval=1, group=grp_ema)
ema_slow_len = input.int(21, "Slow EMA Length", minval=1, group=grp_ema)
use_ema_exit = input.bool(true, "Use Fast/Med EMA Cross for Exit", group=grp_ema)

// --- Bollinger Bands ---
grp_bb = "Bollinger Bands Settings"
show_bb = input.bool(true, "Show Bollinger Bands", group=grp_bb)
bb_len = input.int(20, "BB Length", minval=1, group=grp_bb)
bb_std_dev = input.float(2.0, "BB StdDev Multiplier", minval=0.1, group=grp_bb)
bb_color = input.color(color.new(color.gray, 70), "BB Color", group=grp_bb)
use_bb_return_exit = input.bool(true, "Use BB Return to Mean for Exit", group=grp_bb)

// --- RSI ---
grp_rsi = "RSI Settings"
rsi_len = input.int(14, "RSI Length", minval=1, group=grp_rsi)
rsi_buy_level = input.float(55.0, "RSI Buy Threshold (>)", group=grp_rsi)
rsi_sell_level = input.float(45.0, "RSI Sell Threshold (<)", group=grp_rsi)
use_rsi_div_exit = input.bool(false, "Use RSI Divergence Exit", group=grp_rsi,tooltip="Exit on RSI divergence (lower high/higher low)")
rsi_confirm_fib_bounce = input.bool(true, "Require RSI Confirmation for Fib Bounce", group=grp_rsi,tooltip="Long: RSI>40 Rising. Short: RSI<60 Falling")
rsi_confirm_ema_bounce = input.bool(false, title="Require RSI Confirmation for EMA Bounce?", group=grp_rsi)
rsi_confirm_bb_bounce = input.bool(false, title="Require RSI Confirmation for BB Bounce?", group=grp_rsi)

// --- MACD ---
grp_macd = "MACD Settings"
macd_fast_len = input.int(12, "MACD Fast Length", group=grp_macd)
macd_slow_len = input.int(26, "MACD Slow Length", group=grp_macd)
macd_signal_len = input.int(9, "MACD Signal Length", group=grp_macd)

// --- Volume ---
grp_vol = "Volume Settings"
vol_ma_len = input.int(50, "Volume MA Length", minval=1, group=grp_vol)
vol_multiplier = input.float(1.5, "Volume Breakout Multiplier (> MA)", group=grp_vol)
use_vol_fade_exit = input.bool(true, "Use Low Volume Pullback Exit", group=grp_vol)

// --- ATR Stop Loss ---
grp_atr = "ATR Stop Loss"
use_atr_stop = input.bool(true, "Use ATR Stop Loss for Exit", group=grp_atr)
atr_len = input.int(14, "ATR Length", minval=1, group=grp_atr)
atr_mult = input.float(2.0, "ATR Multiplier", minval=0.1, group=grp_atr)

// --- Fibonacci Exit Target ---
grp_fib_exit = "Fibonacci Exit Target"
use_fib_exit = input.bool(true, "Use Fib Extension Exit Target", group=grp_fib_exit)
fib_lookback_exit = input.int(30, "Fib Exit Swing Lookback", minval=5, group=grp_fib_exit,tooltip="Bars back from entry to find swing point")
fib_extension_level = input.float(1.618, "Fib Extension Target", minval=0.1, group=grp_fib_exit)

// --- Risk/Reward Management ---
grp_rr = "Risk/Reward Management"
use_rr_tp = input.bool(false, "Use R:R Take Profit?", group=grp_rr)
rr_tp_level = input.float(2.0, "R:R Level for TP", minval=0.5, step=0.1, group=grp_rr)
use_rr_be = input.bool(false, "Move SL to Breakeven at R:R?", group=grp_rr)
rr_be_level = input.float(1.0, "R:R Level to Move SL to BE", minval=0.5, step=0.1, group=grp_rr)

// --- Auto Fibonacci Retracement ---
grp_fib_ret = "Auto Fib Retracement"
show_auto_fib = input.bool(true, "Show Auto Fib Retracement", group=grp_fib_ret)
fib_pivot_lookback = input.int(15, "Pivot Lookback (Left/Right Bars)", minval=2, group=grp_fib_ret)
fib_max_bars = input.int(200, "Max Bars Back for Pivots", minval=20, group=grp_fib_ret)
fib_line_color = input.color(color.new(color.gray, 50), "Fib Line Color", group=grp_fib_ret)
fib_line_width = input.int(1, "Fib Line Width", minval=1, group=grp_fib_ret)
extend_fib_lines = input.string(extend.right, title="Extend Fib Lines", options=[extend.none, extend.left, extend.right, extend.both], group=grp_fib_ret)

// --- Trend / Market Condition Filters ---
grp_filter = "Trend Filters"
use_ema_trend_filter = input.bool(true, "Require EMA Trend (Med > Slow)", group=grp_filter)
use_adx_filter = input.bool(true, "Use ADX Filter", group=grp_filter)
adx_len = input.int(14, "ADX Length", minval=1, group=grp_filter)
adx_threshold = input.float(20.0, "ADX Trend Strength Threshold (>)", minval=0, group=grp_filter)
use_adx_direction_filter = input.bool(true, "Require ADX Direction (DI+ vs DI-)", group=grp_filter)
use_htf_filter = input.bool(false, "Use Higher Timeframe EMA Filter", group=grp_filter)
htf = input.timeframe("60", "Higher Timeframe for Filter", group=grp_filter)
htf_ema_len = input.int(50, "HTF EMA Length", group=grp_filter)

// --- Score Calculation Weights ---
grp_score = "Confidence Score Weights"
w_ema_trend = input.int(2, "Weight: EMA Trend", group=grp_score, minval=0)
w_ema_signal = input.int(1, "Weight: EMA Cross Signal", group=grp_score, minval=0)
w_rsi_thresh = input.int(1, "Weight: RSI Threshold", group=grp_score, minval=0)
w_macd_signal = input.int(1, "Weight: MACD Signal Cross", group=grp_score, minval=0)
w_macd_zero = input.int(1, "Weight: MACD Zero Cross", group=grp_score, minval=0)
w_vol_break = input.int(1, "Weight: Volume Breakout", group=grp_score, minval=0)
w_adx_strength = input.int(1, "Weight: ADX Strength", group=grp_score, minval=0)
w_adx_direction = input.int(1, "Weight: ADX Direction", group=grp_score, minval=0)
w_htf_trend = input.int(2, "Weight: HTF Trend", group=grp_score, minval=0)
w_fib_bounce = input.int(2, "Weight: Fib Bounce Signal", group=grp_score, minval=0)
w_ema_bounce = input.int(1, "Weight: EMA Bounce Signal", group=grp_score, minval=0)
w_bb_bounce = input.int(1, "Weight: BB Mid Bounce Signal", group=grp_score, minval=0)

// --- Supply/Demand Zone Inputs ---
grp_sd = "Supply/Demand Zones"
show_sd_zones = input.bool(true, "Show Supply/Demand Zones", group=grp_sd)
sd_lookback = input.int(50, "S/D Zone Lookback Bars", minval=10, group=grp_sd, tooltip="How far back to look for highs/lows to define zones.")
sd_zone_height_atr_mult = input.float(0.5, "S/D Zone Height (ATR Multiplier)", minval=0.1, step=0.1, group=grp_sd, tooltip="Height of the zone based on ATR at the time the high/low was formed.")
sd_color_supply = input.color(color.new(color.red, 80), "Supply Zone Color", group=grp_sd)
sd_color_demand = input.color(color.new(color.green, 80), "Demand Zone Color", group=grp_sd)
sd_max_bars = input.int(100, "Max Bars for S/D Zones", minval=20, group=grp_sd, tooltip="Maximum bars back for valid supply/demand zones.")

// --- Exit Settings ---
grp_exit = "Exit Settings"
max_hold_bars = input.int(30, "Max Bars to Hold", minval=1, group=grp_exit, tooltip="Maximum number of bars to hold a trade if no other exit condition is met.")

// === FUNCTIONS ===

// Function to update/draw Fibonacci lines
f_update_fib_line(level, line_ref, x1, x2, _color, _width, _extend) =>
    var line return_line = na
    if not na(level)
        if not na(line_ref)
            line.set_xy1(line_ref, x1, level)
            line.set_xy2(line_ref, x2, level)
            line.set_color(line_ref, _color)
            line.set_width(line_ref, _width)
            line.set_extend(line_ref, _extend)
            return_line := line_ref
        else
            return_line := line.new(x1, level, x2, level, color=_color, width=_width, extend=_extend)
    return_line

// Function to update/draw Fibonacci labels
f_update_fib_label(level, label_ref, label_text, x, _color) =>
    var label return_label = na
    if not na(level)
        if not na(label_ref)
            label.set_xy(label_ref, x, level)
            label.set_text(label_ref, label_text)
            label.set_textcolor(label_ref, _color)
            return_label := label_ref
        else
            return_label := label.new(x, level, label_text, style=label.style_none, textcolor=_color, size=size.small)
    return_label

// Function to delete Fibonacci objects - **CORRECTED** (Removed return)
f_delete_fib_objects(line0, line236, line382, line500, line618, line786, line100, label0, label236, label382, label500, label618, label786, label100) =>
    if not na(line0)
        line.delete(line0)
        label.delete(label0)
    if not na(line236)
        line.delete(line236)
        label.delete(label236)
    if not na(line382)
        line.delete(line382)
        label.delete(label382)
    if not na(line500)
        line.delete(line500)
        label.delete(label500)
    if not na(line618)
        line.delete(line618)
        label.delete(label618)
    if not na(line786)
        line.delete(line786)
        label.delete(label786)
    if not na(line100)
        line.delete(line100)
        label.delete(label100)
    // Removed return [na,...] line

// Function to update/draw S/D boxes
f_update_sd_box(box_ref, left_bar, top, right_bar, bottom, _color, bar_ref, ref_bar) =>
    var box return_box = na
    var int return_bar = na
    if not na(top) and not na(bottom) and not na(left_bar) and left_bar > 0 // Ensure left_bar is valid
        // If no box exists OR the new pivot is more recent than the current box's pivot
        if na(box_ref) or left_bar > nz(bar_ref, -1)
            if not na(box_ref)
                box.delete(box_ref) // Delete the old box
            return_box := box.new(left_bar, top, right_bar, bottom, border_color=na, bgcolor=_color, extend=extend.right)
            return_bar := left_bar
        // If a box exists AND it's based on the same pivot bar, just update its right boundary
        else if not na(box_ref) and box.get_left(box_ref) == left_bar
            box.set_right(box_ref, right_bar)
            return_box := box_ref
            return_bar := left_bar // Keep the original bar reference
        // Else (new pivot is older or invalid), keep the existing box and bar reference
        else
            return_box := box_ref
            return_bar := bar_ref
    else // If input data is invalid, keep the existing box and bar reference
        return_box := box_ref
        return_bar := bar_ref

    [return_box, return_bar]


// Function to compute potential entry signals (before score/filter checks)
f_compute_potential_signals(ema_cross_buy, fib_buy, ema_bounce_buy, bb_bounce_buy, vol_buy, ema_cross_sell, fib_sell, ema_bounce_sell, bb_bounce_sell, vol_sell) =>
    bool potential_buy = ema_cross_buy or fib_buy or ema_bounce_buy or bb_bounce_buy or vol_buy
    bool potential_sell = ema_cross_sell or fib_sell or ema_bounce_sell or bb_bounce_sell or vol_sell
    string buy_reason = fib_buy ? "Fib" : ema_cross_buy ? "EMA Cross" : vol_buy ? "Vol Breakout" : bb_bounce_buy ? "BB Bounce" : ema_bounce_buy ? "EMA Bounce" : ""
    string sell_reason = fib_sell ? "Fib" : ema_cross_sell ? "EMA Cross" : vol_sell ? "Vol Breakout" : bb_bounce_sell ? "BB Bounce" : ema_bounce_sell ? "EMA Bounce" : ""
    [potential_buy, potential_sell, buy_reason, sell_reason]

// === GLOBAL VARIABLE DECLARATIONS ===
// --- Fibonacci Drawing Objects ---
var line fib_line_0 = na
var line fib_line_236 = na
var line fib_line_382 = na
var line fib_line_500 = na
var line fib_line_618 = na
var line fib_line_786 = na
var line fib_line_100 = na
var label fib_label_0 = na
var label fib_label_236 = na
var label fib_label_382 = na
var label fib_label_500 = na
var label fib_label_618 = na
var label fib_label_786 = na
var label fib_label_100 = na
var float fib_swing_high = na
var int fib_swing_high_bar = na
var float fib_swing_low = na
var int fib_swing_low_bar = na

// --- Supply/Demand Drawing Objects ---
var box supply_box = na
var box demand_box = na
var int supply_bar = na // Bar index where the supply zone high occurred
var int demand_bar = na // Bar index where the demand zone low occurred

// --- Trade State Tracking ---
var bool in_long = false
var bool in_short = false
var float stop_loss_level = na
var float fib_target_level_exit = na
var float entry_price = na
var string exit_reason = "" // Stores the exit reason for the last closed trade
var float last_trade_pnl = na // Stores PnL of the last closed trade
var int bars_held = 0 // Bars held in the current trade
var int last_trade_bar = na // Bar index of the last entry
var float cumulative_pnl = 0.0 // Cumulative PnL percentage (hypothetical)
var float current_equity = initial_capital // Hypothetical equity tracking
var float position_size = na // Hypothetical position size in units
var float stop_distance_points = na // Store stop distance for plotting

// --- RR Management State ---
var float rr_target_price = na
var float rr_breakeven_trigger_price = na
var bool sl_moved_to_be = false
// varip int last_cleaned_bar = na  // Corrected below
var int last_cleaned_bar = na  // For periodic cleanup - CORRECTED `varip` to `var`, but this logic is removed later
// var int max_sd_zones = 10        // Removed - Unused

// === CALCULATIONS ===

// --- Indicators ---
ema_fast = ta.ema(close, ema_fast_len)
ema_med = ta.ema(close, ema_med_len)
ema_slow = ta.ema(close, ema_slow_len)
[bb_basis, bb_upper, bb_lower] = ta.bb(close, bb_len, bb_std_dev) // Renamed bb_middle to bb_basis
price_rsi = ta.rsi(close, rsi_len)
is_rsi_rising = ta.rising(price_rsi, 1) // Calculate unconditionally
is_rsi_falling = ta.falling(price_rsi, 1) // Calculate unconditionally
[macd_line, signal_line, hist_line] = ta.macd(close, macd_fast_len, macd_slow_len, macd_signal_len)
vol_ma = ta.sma(volume, vol_ma_len)
atr_value = ta.atr(atr_len) // Renamed atr_val to atr_value
[di_pos, di_neg, adx_val] = ta.dmi(adx_len, adx_len)
htf_ema = request.security(syminfo.tickerid, htf, ta.ema(close, htf_ema_len), lookahead=barmerge.lookahead_off)
lowest_for_fib_exit = ta.lowest(low, fib_lookback_exit)[1] // Lookback excluding current bar
highest_for_fib_exit = ta.highest(high, fib_lookback_exit)[1] // Lookback excluding current bar

// --- Weak Signal Filter Calculations ---
expected_move = atr_value * skip_atr_factor
price_move = math.abs(close - open)
momentum_ok = skip_atr_factor == 0 or price_move > expected_move
skip_vol_ma = ta.sma(volume, skip_vol_ma_len)
volume_ok_for_filter = skip_vol_multiplier == 0 or volume > skip_vol_ma * skip_vol_multiplier
// Skip trade if filter is enabled AND (momentum is weak OR volume is weak)
skip_trade = use_skip_trade_filter and not (momentum_ok and volume_ok_for_filter)

// --- Basic Conditions ---
cond_ema_fast_slow_cross_buy = ta.crossover(ema_fast, ema_slow)
cond_ema_fast_slow_cross_sell = ta.crossunder(ema_fast, ema_slow)
cond_ema_fast_med_cross_buy = ta.crossover(ema_fast, ema_med)
cond_ema_fast_med_cross_sell = ta.crossunder(ema_fast, ema_med)
cond_bb_return_mean_buy = ta.crossover(close, bb_basis) // Used for exit condition
cond_bb_return_mean_sell = ta.crossunder(close, bb_basis) // Used for exit condition
cond_rsi_buy = price_rsi > rsi_buy_level
cond_rsi_sell = price_rsi < rsi_sell_level
cond_macd_signal_cross_buy = ta.crossover(macd_line, signal_line)
cond_macd_signal_cross_sell = ta.crossunder(macd_line, signal_line)
cond_macd_zero_cross_buy = ta.crossover(macd_line, 0)
cond_macd_zero_cross_sell = ta.crossunder(macd_line, 0)
cond_high_vol = volume > vol_ma * vol_multiplier
cond_vol_breakout_buy_raw = cond_high_vol and close > open // Basic vol breakout up
cond_vol_breakout_sell_raw = cond_high_vol and close < open // Basic vol breakout down
cond_vol_fade_long = close < ema_fast and volume < vol_ma // Used for exit
cond_vol_fade_short = close > ema_fast and volume < vol_ma // Used for exit

// --- Filter Conditions ---
cond_ema_trend_ok_buy = not use_ema_trend_filter or ema_med > ema_slow
cond_ema_trend_ok_sell = not use_ema_trend_filter or ema_med < ema_slow
cond_adx_strength_ok = not use_adx_filter or adx_val > adx_threshold
cond_adx_direction_ok_buy = not use_adx_direction_filter or di_pos > di_neg
cond_adx_direction_ok_sell = not use_adx_direction_filter or di_neg > di_pos
cond_adx_filter_ok_buy = cond_adx_strength_ok and cond_adx_direction_ok_buy
cond_adx_filter_ok_sell = cond_adx_strength_ok and cond_adx_direction_ok_sell
cond_htf_filter_ok_buy = not use_htf_filter or close > htf_ema
cond_htf_filter_ok_sell = not use_htf_filter or close < htf_ema
bb_width = bb_upper - bb_lower
min_bb_width = ta.sma(bb_width, 10) * 0.3 // Example threshold for ranging market based on BB width
is_in_range = bb_width < min_bb_width or (use_adx_filter and adx_val < 12) // Consider low ADX as range indicator too
// Combine all filters for buy/sell signals
all_filters_ok_buy = cond_ema_trend_ok_buy and cond_adx_filter_ok_buy and cond_htf_filter_ok_buy and not is_in_range
all_filters_ok_sell = cond_ema_trend_ok_sell and cond_adx_filter_ok_sell and cond_htf_filter_ok_sell and not is_in_range

// --- Auto Fibonacci Retracement Calculation ---
var float last_pivot_high_price = na
var int last_pivot_high_bar = na
var float last_pivot_low_price = na
var int last_pivot_low_bar = na

pivot_high_val = ta.pivothigh(high, fib_pivot_lookback, fib_pivot_lookback)
pivot_low_val = ta.pivotlow(low, fib_pivot_lookback, fib_pivot_lookback)

if not na(pivot_high_val)
    pivot_high_bar_index = bar_index[fib_pivot_lookback]
    // Update if no previous high, or new pivot is later, or new pivot is same bar but higher price
    if na(last_pivot_high_bar) or pivot_high_bar_index > last_pivot_high_bar or (pivot_high_bar_index == last_pivot_high_bar and pivot_high_val > last_pivot_high_price)
        last_pivot_high_price := pivot_high_val
        last_pivot_high_bar := pivot_high_bar_index
if not na(pivot_low_val)
    pivot_low_bar_index = bar_index[fib_pivot_lookback]
    // Update if no previous low, or new pivot is later, or new pivot is same bar but lower price
    if na(last_pivot_low_bar) or pivot_low_bar_index > last_pivot_low_bar or (pivot_low_bar_index == last_pivot_low_bar and pivot_low_val < last_pivot_low_price)
        last_pivot_low_price := pivot_low_val
        last_pivot_low_bar := pivot_low_bar_index

// Invalidate pivots if they are too old
if bar_index - nz(last_pivot_high_bar, -fib_max_bars - 1) > fib_max_bars
    last_pivot_high_price := na
    last_pivot_high_bar := na
if bar_index - nz(last_pivot_low_bar, -fib_max_bars - 1) > fib_max_bars
    last_pivot_low_price := na
    last_pivot_low_bar := na

// Determine the swing points for Fibonacci calculation
fib_swing_high := na
fib_swing_high_bar := na
fib_swing_low := na
fib_swing_low_bar := na
if not na(last_pivot_high_price) and not na(last_pivot_low_price)
    // If high pivot is more recent than low pivot -> downtrend swing
    if nz(last_pivot_high_bar, -1) > nz(last_pivot_low_bar, -1)
        fib_swing_high := last_pivot_high_price
        fib_swing_high_bar := last_pivot_high_bar
        fib_swing_low := last_pivot_low_price
        fib_swing_low_bar := last_pivot_low_bar
    // If low pivot is more recent than high pivot -> uptrend swing
    else
        fib_swing_low := last_pivot_low_price
        fib_swing_low_bar := last_pivot_low_bar
        fib_swing_high := last_pivot_high_price
        fib_swing_high_bar := last_pivot_high_bar

// Calculate Fibonacci levels
var float level_0 = na
var float level_236 = na
var float level_382 = na
var float level_500 = na
var float level_618 = na
var float level_786 = na
var float level_100 = na
bool valid_fib_range = not na(fib_swing_high) and not na(fib_swing_low) and fib_swing_high > fib_swing_low and not na(fib_swing_high_bar) and not na(fib_swing_low_bar)
bool is_uptrend_fib = valid_fib_range and fib_swing_low_bar < fib_swing_high_bar // Low precedes High

if valid_fib_range
    fib_range = fib_swing_high - fib_swing_low
    // Determine 0% and 100% levels based on trend direction
    if is_uptrend_fib
        level_0 := fib_swing_low
        level_100 := fib_swing_high
    else // Downtrend
        level_0 := fib_swing_high
        level_100 := fib_swing_low
    // Calculate standard levels
    level_236 := level_0 + (level_100 - level_0) * 0.236
    level_382 := level_0 + (level_100 - level_0) * 0.382
    level_500 := level_0 + (level_100 - level_0) * 0.500
    level_618 := level_0 + (level_100 - level_0) * 0.618
    level_786 := level_0 + (level_100 - level_0) * 0.786
else
    // Invalidate levels if range is not valid
    level_0 := na
    level_236 := na
    level_382 := na
    level_500 := na
    level_618 := na
    level_786 := na
    level_100 := na


// --- Fibonacci Bounce Condition (Buy & Sell) --- Using Loops for accuracy
bool touched_fib_zone_buy = false
if use_fib_bounce_entry and is_uptrend_fib and not na(level_618) and not na(level_500)
    for i = 0 to fib_bounce_lookback - 1 // Check current and previous bars
        if low[i] <= level_618 and low[i] >= level_500 // Touched the 0.5-0.618 zone
            touched_fib_zone_buy := true
            break // Exit loop once touch is confirmed
bool bounced_above_50_buy = close > level_500 // Price closed above the 50% level
bool rsi_confirms_bounce_buy = not rsi_confirm_fib_bounce or (price_rsi > 40 and is_rsi_rising) // RSI > 40 and rising
bool fib_volume_spike_buy = volume > ta.sma(volume, 20)[1] * fib_volume_mult // Volume spike confirmation
cond_fib_bounce_buy_raw = use_fib_bounce_entry and is_uptrend_fib and touched_fib_zone_buy and bounced_above_50_buy and rsi_confirms_bounce_buy and fib_volume_spike_buy

bool touched_fib_zone_sell = false
if use_fib_bounce_sell and not is_uptrend_fib and valid_fib_range and not na(level_382) and not na(level_500)
    for i = 0 to fib_bounce_lookback - 1
        if high[i] >= level_382 and high[i] <= level_500 // Touched the 0.382-0.5 zone (reversed for downtrend)
            touched_fib_zone_sell := true
            break
// **REMOVED** Incorrect overwrite: touched_fib_zone_buy = ta.lowest(low, fib_bounce_lookback) <= level_618 and ta.highest(low, fib_bounce_lookback) >= level_500
// **REMOVED** Incorrect overwrite: touched_fib_zone_sell = ta.highest(high, fib_bounce_lookback) >= level_382 and ta.lowest(high, fib_bounce_lookback) <= level_500

bool rejected_below_50_sell = close < level_500 // Price closed below the 50% level
bool rsi_confirms_bounce_sell = not rsi_confirm_fib_bounce or (price_rsi < 60 and is_rsi_falling) // RSI < 60 and falling
bool fib_volume_spike_sell = volume > ta.sma(volume, 20)[1] * fib_volume_mult // Volume spike confirmation
cond_fib_bounce_sell_raw = use_fib_bounce_sell and not is_uptrend_fib and valid_fib_range and touched_fib_zone_sell and rejected_below_50_sell and rsi_confirms_bounce_sell and fib_volume_spike_sell

// --- EMA Bounce Condition (Buy & Sell) --- Using Loops for accuracy
float ema_source_val = ema_bounce_source == "Fast EMA" ? ema_fast : ema_med
bool touched_ema_buy = false
if use_ema_bounce_buy
    for i = 1 to ema_bounce_lookback // Check previous bars only
        if low[i] <= ema_source_val[i]
            touched_ema_buy := true
            break
bool rsi_confirms_ema_buy = not rsi_confirm_ema_bounce or (price_rsi > 40 and is_rsi_rising)
cond_ema_bounce_buy_raw = use_ema_bounce_buy and touched_ema_buy and close > ema_source_val and close > open and rsi_confirms_ema_buy

bool touched_ema_sell = false
if use_ema_bounce_sell
    for i = 1 to ema_bounce_lookback
        if high[i] >= ema_source_val[i]
            touched_ema_sell := true
            break
bool rsi_confirms_ema_sell = not rsi_confirm_ema_bounce or (price_rsi < 60 and is_rsi_falling)
cond_ema_bounce_sell_raw = use_ema_bounce_sell and touched_ema_sell and close < ema_source_val and close < open and rsi_confirms_ema_sell

// **REMOVED** Incorrect overwrites for EMA and BB bounces
// touched_ema_buy = ta.lowest(low, ema_bounce_lookback) <= ema_source_val
// touched_ema_sell = ta.highest(high, ema_bounce_lookback) >= ema_source_val
// touched_bb_mid_buy = ta.lowest(low, bb_bounce_lookback) <= bb_basis
// touched_bb_mid_sell = ta.highest(high, bb_bounce_lookback) >= bb_basis

// --- BB Middle Bounce Condition (Buy & Sell) --- Using Loops for accuracy
bool touched_bb_mid_buy = false // Re-declare to ensure scope if needed, though loops below set it
if use_bb_mid_bounce_buy
    for i = 1 to bb_bounce_lookback
        if low[i] <= bb_basis[i]
            touched_bb_mid_buy := true
            break
bool rsi_confirms_bb_buy = not rsi_confirm_bb_bounce or (price_rsi > 40 and is_rsi_rising)
cond_bb_mid_bounce_buy_raw = use_bb_mid_bounce_buy and touched_bb_mid_buy and close > bb_basis and close > open and rsi_confirms_bb_buy

bool touched_bb_mid_sell = false // Re-declare to ensure scope if needed
if use_bb_mid_bounce_sell
    for i = 1 to bb_bounce_lookback
        if high[i] >= bb_basis[i]
            touched_bb_mid_sell := true
            break
bool rsi_confirms_bb_sell = not rsi_confirm_bb_bounce or (price_rsi < 60 and is_rsi_falling)
cond_bb_mid_bounce_sell_raw = use_bb_mid_bounce_sell and touched_bb_mid_sell and close < bb_basis and close < open and rsi_confirms_bb_sell

// --- Confidence Score Calculation ---
float buy_score_bar = 0.0
float sell_score_bar = 0.0
float total_possible_score = 0.0 // Recalculate based on active conditions

// EMA Trend
if use_ema_trend_filter
    total_possible_score += w_ema_trend
    if cond_ema_trend_ok_buy
        buy_score_bar += w_ema_trend
    if cond_ema_trend_ok_sell
        sell_score_bar += w_ema_trend
// EMA Signal (Fast/Slow Cross) - Note: Only adds score on the crossover bar
total_possible_score += w_ema_signal
if cond_ema_fast_slow_cross_buy
    buy_score_bar += w_ema_signal
if cond_ema_fast_slow_cross_sell
    sell_score_bar += w_ema_signal
// RSI Threshold
total_possible_score += w_rsi_thresh
if cond_rsi_buy
    buy_score_bar += w_rsi_thresh
if cond_rsi_sell
    sell_score_bar += w_rsi_thresh
// MACD Signal Cross
total_possible_score += w_macd_signal
if cond_macd_signal_cross_buy
    buy_score_bar += w_macd_signal
if cond_macd_signal_cross_sell
    sell_score_bar += w_macd_signal
// MACD Zero Cross
total_possible_score += w_macd_zero
if cond_macd_zero_cross_buy
    buy_score_bar += w_macd_zero
if cond_macd_zero_cross_sell
    sell_score_bar += w_macd_zero
// Volume Breakout
if use_vol_breakout_buy or use_vol_breakout_sell
    total_possible_score += w_vol_break
    if use_vol_breakout_buy and cond_vol_breakout_buy_raw
        buy_score_bar += w_vol_break // Use raw condition before filter
    if use_vol_breakout_sell and cond_vol_breakout_sell_raw
        sell_score_bar += w_vol_break // Use raw condition before filter
// ADX Strength
if use_adx_filter
    total_possible_score += w_adx_strength
    if cond_adx_strength_ok
    // Add strength score if ADX is high, regardless of direction here
        buy_score_bar += w_adx_strength
        sell_score_bar += w_adx_strength
// ADX Direction
if use_adx_direction_filter
    total_possible_score += w_adx_direction
    if cond_adx_direction_ok_buy
        buy_score_bar += w_adx_direction
    if cond_adx_direction_ok_sell
        sell_score_bar += w_adx_direction
// HTF Trend
if use_htf_filter
    total_possible_score += w_htf_trend
    if cond_htf_filter_ok_buy
        buy_score_bar += w_htf_trend
    if cond_htf_filter_ok_sell
        sell_score_bar += w_htf_trend
// Fib Bounce Signal
if use_fib_bounce_entry or use_fib_bounce_sell
    total_possible_score += w_fib_bounce
    if use_fib_bounce_entry and cond_fib_bounce_buy_raw
        buy_score_bar += w_fib_bounce
    if use_fib_bounce_sell and cond_fib_bounce_sell_raw
        sell_score_bar += w_fib_bounce
// EMA Bounce Signal
if use_ema_bounce_buy or use_ema_bounce_sell
    total_possible_score += w_ema_bounce
    if use_ema_bounce_buy and cond_ema_bounce_buy_raw
        buy_score_bar += w_ema_bounce
    if use_ema_bounce_sell and cond_ema_bounce_sell_raw
        sell_score_bar += w_ema_bounce
// BB Mid Bounce Signal
if use_bb_mid_bounce_buy or use_bb_mid_bounce_sell
    total_possible_score += w_bb_bounce
    if use_bb_mid_bounce_buy and cond_bb_mid_bounce_buy_raw
        buy_score_bar += w_bb_bounce
    if use_bb_mid_bounce_sell and cond_bb_mid_bounce_sell_raw
        sell_score_bar += w_bb_bounce

// Calculate Scaled Score (0-10)
total_possible_score := math.max(1.0, total_possible_score) // Avoid division by zero
net_score = buy_score_bar - sell_score_bar
// Scale net score (which ranges roughly from -total to +total) to 0-10 range
scaled_score = (net_score / total_possible_score) * 5.0 + 5.0
scaled_score := math.max(0.0, math.min(10.0, scaled_score)) // Clamp between 0 and 10

// --- RSI Divergence ---
rsi_peak_val = ta.pivothigh(price_rsi, 5, 2) // Look 5 bars left, 2 bars right for pivot high
rsi_trough_val = ta.pivotlow(price_rsi, 5, 2)  // Look 5 bars left, 2 bars right for pivot low
price_peak_val = ta.pivothigh(high, 5, 2)      // Price pivot high corresponding to RSI pivot search window
price_trough_val = ta.pivotlow(low, 5, 2)       // Price pivot low corresponding to RSI pivot search window

// Bullish Divergence: Lower low in price, higher low in RSI
// Check if a price trough occurred 2 bars ago, and an RSI trough occurred 2 bars ago
// Then check if current low is lower than that price trough, and current RSI is higher than that RSI trough
bullish_rsi_div = not na(price_trough_val[2]) and not na(rsi_trough_val[2]) and low < price_trough_val[2] and price_rsi > rsi_trough_val[2]

// Bearish Divergence: Higher high in price, lower high in RSI
// Check if a price peak occurred 2 bars ago, and an RSI peak occurred 2 bars ago
// Then check if current high is higher than that price peak, and current RSI is lower than that RSI peak
bearish_rsi_div = not na(price_peak_val[2]) and not na(rsi_peak_val[2]) and high > price_peak_val[2] and price_rsi < rsi_peak_val[2]

cond_rsi_bull_div_exit = use_rsi_div_exit and bearish_rsi_div // Exit LONG on BEARISH divergence
cond_rsi_bear_div_exit = use_rsi_div_exit and bullish_rsi_div // Exit SHORT on BULLISH divergence


// === STATE MANAGEMENT & TRADE LOGIC ===

// Capture previous state *before* any logic runs for the current bar
bool was_in_long_prev = nz(in_long[1]) // Use nz() for safety on first bar
bool was_in_short_prev = nz(in_short[1]) // Use nz() for safety on first bar

// Determine if a new trade can be entered (cooldown period) - CORRECTED SYNTAX
can_reenter = na(last_trade_bar) or (bar_index - last_trade_bar >= cooldown_bars) // Added missing parenthesis

// --- Potential Entry Signals ---
// Combine raw signals with filters
is_potential_ema_buy = cond_ema_fast_slow_cross_buy and all_filters_ok_buy
is_potential_fib_buy = cond_fib_bounce_buy_raw and all_filters_ok_buy
is_potential_ema_bounce_buy = cond_ema_bounce_buy_raw and all_filters_ok_buy
is_potential_bb_bounce_buy = cond_bb_mid_bounce_buy_raw and all_filters_ok_buy
is_potential_vol_buy = use_vol_breakout_buy and cond_vol_breakout_buy_raw and all_filters_ok_buy

is_potential_ema_sell = cond_ema_fast_slow_cross_sell and all_filters_ok_sell
is_potential_fib_sell = cond_fib_bounce_sell_raw and all_filters_ok_sell
is_potential_ema_bounce_sell = cond_ema_bounce_sell_raw and all_filters_ok_sell
is_potential_bb_bounce_sell = cond_bb_mid_bounce_sell_raw and all_filters_ok_sell
is_potential_vol_sell = use_vol_breakout_sell and cond_vol_breakout_sell_raw and all_filters_ok_sell

// Compute final entry signals based on score, potential signals, filters, and ATR minimum
min_atr_ok = atr_value > ta.sma(atr_value, 20) * min_atr_factor
[potential_buy_signal, potential_sell_signal, buy_reason_text, sell_reason_text] = f_compute_potential_signals(
     is_potential_ema_buy, is_potential_fib_buy, is_potential_ema_bounce_buy, is_potential_bb_bounce_buy, is_potential_vol_buy,
     is_potential_ema_sell, is_potential_fib_sell, is_potential_ema_bounce_sell, is_potential_bb_bounce_sell, is_potential_vol_sell)

// Final Entry Conditions
entry_signal_buy = potential_buy_signal and scaled_score >= entry_score_threshold_long and not skip_trade and min_atr_ok
entry_signal_sell = potential_sell_signal and scaled_score <= entry_score_threshold_short and not skip_trade and min_atr_ok

// Determine if it's a NEW entry (could be initial or re-entry after cooldown)
enter_long_condition = entry_signal_buy and not in_long and can_reenter
enter_short_condition = entry_signal_sell and not in_short and can_reenter

// --- Trade Execution Simulation ---
// Reset exit reason at the start of the bar
exit_reason_bar = ""
exited_long_this_bar = false
exited_short_this_bar = false
float exit_price = na // Price at which the exit occurred this bar

// Increment bars held if in a position from the previous bar
if was_in_long_prev or was_in_short_prev
    bars_held += 1

// --- Exit Logic (Check based on PREVIOUS bar's state) ---
// Priority: 1. Time, 2. ATR SL, 3. TP (RR/Fib), 4. Signal Exits
if was_in_long_prev
    // 1. Time-based Exit
    if bars_held >= max_hold_bars
        exited_long_this_bar := true
        exit_reason_bar := "Time Exit"
        exit_price := close
    // 2. ATR Stop Loss
    else if use_atr_stop and not na(stop_loss_level[1]) and low <= stop_loss_level[1]
        exited_long_this_bar := true
        exit_reason_bar := "ATR SL"
        exit_price := stop_loss_level[1]
    // 3. Take Profit (RR or Fib) - **MODIFIED LABEL**
    else if use_rr_tp and not na(rr_target_price[1]) and high >= rr_target_price[1]
        exited_long_this_bar := true
        exit_reason_bar := "Long TP" // Changed label
        exit_price := rr_target_price[1]
    else if use_fib_exit and not na(fib_target_level_exit[1]) and high >= fib_target_level_exit[1]
        exited_long_this_bar := true
        exit_reason_bar := "Long TP" // Changed label
        exit_price := fib_target_level_exit[1]
    // 4. Signal-Based Exits
    else if scaled_score < (5.0 - exit_score_drop_threshold)
        exited_long_this_bar := true
        exit_reason_bar := "Score Drop (" + str.tostring(scaled_score, "#.#") + ")"
        exit_price := close
    else if use_ema_exit and cond_ema_fast_med_cross_sell
        exited_long_this_bar := true
        exit_reason_bar := "EMA Cross"
        exit_price := close
    else if use_bb_return_exit and cond_bb_return_mean_sell
        exited_long_this_bar := true
        exit_reason_bar := "BB Mid Exit"
        exit_price := close
    else if use_vol_fade_exit and cond_vol_fade_long
        exited_long_this_bar := true
        exit_reason_bar := "Vol Fade"
        exit_price := close
    else if cond_rsi_bull_div_exit
        exited_long_this_bar := true
        exit_reason_bar := "RSI Div"
        exit_price := close

if was_in_short_prev
    // 1. Time-based Exit
    if bars_held >= max_hold_bars
        exited_short_this_bar := true
        exit_reason_bar := "Time Exit"
        exit_price := close
    // 2. ATR Stop Loss
    else if use_atr_stop and not na(stop_loss_level[1]) and high >= stop_loss_level[1]
        exited_short_this_bar := true
        exit_reason_bar := "ATR SL"
        exit_price := stop_loss_level[1]
    // 3. Take Profit (RR or Fib) - **MODIFIED LABEL**
    else if use_rr_tp and not na(rr_target_price[1]) and low <= rr_target_price[1]
        exited_short_this_bar := true
        exit_reason_bar := "Short TP" // Changed label
        exit_price := rr_target_price[1]
    else if use_fib_exit and not na(fib_target_level_exit[1]) and low <= fib_target_level_exit[1]
        exited_short_this_bar := true
        exit_reason_bar := "Short TP" // Changed label
        exit_price := fib_target_level_exit[1]
    // 4. Signal-Based Exits
    else if scaled_score > (5.0 + exit_score_drop_threshold)
        exited_short_this_bar := true
        exit_reason_bar := "Score Drop (" + str.tostring(scaled_score, "#.#") + ")"
        exit_price := close
    else if use_ema_exit and cond_ema_fast_med_cross_buy
        exited_short_this_bar := true
        exit_reason_bar := "EMA Cross"
        exit_price := close
    else if use_bb_return_exit and cond_bb_return_mean_buy
        exited_short_this_bar := true
        exit_reason_bar := "BB Mid Exit"
        exit_price := close
    else if use_vol_fade_exit and cond_vol_fade_short
        exited_short_this_bar := true
        exit_reason_bar := "Vol Fade"
        exit_price := close
    else if cond_rsi_bear_div_exit
        exited_short_this_bar := true
        exit_reason_bar := "RSI Div"
        exit_price := close

// --- Reset State on Exit ---
if exited_long_this_bar or exited_short_this_bar
    // Calculate PnL for the closed trade (Hypothetical - using exit_price)
    float pnl_points = exited_long_this_bar ? exit_price - entry_price[1] : entry_price[1] - exit_price
    float pnl_percent = (pnl_points / entry_price[1]) * 100
    last_trade_pnl := pnl_percent
    cumulative_pnl += pnl_percent
    // Update hypothetical equity (Note: Doesnt account for commission/slippage)
    float pnl_currency = pnl_points * nz(position_size[1], 1) // Use position size from previous bar, default to 1 if na
    current_equity += pnl_currency

    // Reset trade state variables
    in_long := false
    in_short := false
    stop_loss_level := na
    fib_target_level_exit := na
    entry_price := na
    exit_reason := exit_reason_bar // Store the reason for the closed trade
    bars_held := 0 // Reset bars held counter
    position_size := na
    rr_target_price := na
    rr_breakeven_trigger_price := na
    sl_moved_to_be := false
    stop_distance_points := na // Reset plot variable
    // Keep last_trade_bar as is, it marks the entry bar for cooldown calculation


// --- Entry Logic ---
if enter_long_condition
    in_long := true
    in_short := false
    entry_price := close // Enter at close price
    stop_loss_level := entry_price - atr_value * atr_mult // Set initial SL
    // Calculate Fib Exit Target
    swing_low_price_exit = lowest_for_fib_exit
    swing_range_fib_exit = entry_price - swing_low_price_exit
    fib_target_level_exit := swing_range_fib_exit > 0 and use_fib_exit ? entry_price + swing_range_fib_exit * fib_extension_level : na
    // Calculate Position Size (Hypothetical) - Integrated safer calculation
    float risk_per_trade = current_equity * (risk_percent / 100)
    stop_distance_points := math.max(math.abs(entry_price - stop_loss_level), syminfo.mintick) // Use max with mintick
    position_size := stop_distance_points > 0 ? risk_per_trade / stop_distance_points : na // Set to na if stop distance is zero
    // Calculate RR levels
    if not na(position_size) and stop_distance_points > 0 // Check if size could be calculated
        rr_target_price := use_rr_tp ? entry_price + stop_distance_points * rr_tp_level : na
        rr_breakeven_trigger_price := use_rr_be ? entry_price + stop_distance_points * rr_be_level : na
    else // Cannot calculate RR if position size is na
        rr_target_price := na
        rr_breakeven_trigger_price := na
        position_size := na // Ensure size is na if stop distance was zero

    sl_moved_to_be := false // Reset BE flag
    last_trade_bar := bar_index // Mark entry bar for cooldown
    bars_held := 1 // Start bars held count
    last_trade_pnl := na // Reset PnL for new trade
    exit_reason := "" // Clear exit reason for new trade

if enter_short_condition
    in_short := true
    in_long := false
    entry_price := close
    stop_loss_level := entry_price + atr_value * atr_mult
    // Calculate Fib Exit Target
    swing_high_price_exit = highest_for_fib_exit
    swing_range_fib_exit = swing_high_price_exit - entry_price
    fib_target_level_exit := swing_range_fib_exit > 0 and use_fib_exit ? entry_price - swing_range_fib_exit * fib_extension_level : na
    // Calculate Position Size (Hypothetical) - Integrated safer calculation
    float risk_per_trade = current_equity * (risk_percent / 100)
    stop_distance_points := math.max(math.abs(stop_loss_level - entry_price), syminfo.mintick) // Use max with mintick
    position_size := stop_distance_points > 0 ? risk_per_trade / stop_distance_points : na // Set to na if stop distance is zero
    // Calculate RR levels
    if not na(position_size) and stop_distance_points > 0 // Check if size could be calculated
        rr_target_price := use_rr_tp ? entry_price - stop_distance_points * rr_tp_level : na
        rr_breakeven_trigger_price := use_rr_be ? entry_price - stop_distance_points * rr_be_level : na
    else // Cannot calculate RR if position size is na
        rr_target_price := na
        rr_breakeven_trigger_price := na
        position_size := na // Ensure size is na if stop distance was zero

    sl_moved_to_be := false // Reset BE flag
    last_trade_bar := bar_index
    bars_held := 1
    last_trade_pnl := na
    exit_reason := ""

// **REMOVED** Misplaced position sizing block
// // --- Safer Position Sizing ---
// stop_distance_points = math.max(
//     in_long ? entry_price - stop_loss_level : stop_loss_level - entry_price,
//     syminfo.mintick
// )
// position_size := stop_distance_points > 0 ?
//     (current_equity * (risk_percent / 100)) / stop_distance_points :
//     na

// --- Stop Loss Management (Trailing and Break-Even) ---
// Check if still in position AFTER potential exit this bar
if in_long
    // Move SL to Break-Even
    if use_rr_be and not sl_moved_to_be and not na(rr_breakeven_trigger_price) and high >= rr_breakeven_trigger_price and entry_price < rr_breakeven_trigger_price
        if stop_loss_level < entry_price // Only move if SL is below entry
            stop_loss_level := entry_price // Move SL to BE
            sl_moved_to_be := true
            label.new(bar_index, stop_loss_level, "SL->BE", style=label.style_label_down, color=color.new(color.orange, 30), textcolor=color.black, size=size.tiny)
    // Trail ATR Stop (only if not moved to BE or if new SL is higher than BE)
    if use_atr_stop
        new_stop_long = close - atr_value * atr_mult // Trail based on close
        // Only trail up, never down. Ensure it respects BE level if activated.
        stop_loss_level := math.max(nz(stop_loss_level[1], new_stop_long), new_stop_long)
        if sl_moved_to_be // If already at BE, ensure trailing stop doesn't go below entry
            stop_loss_level := math.max(stop_loss_level, entry_price)

if in_short
    // Move SL to Break-Even
    if use_rr_be and not sl_moved_to_be and not na(rr_breakeven_trigger_price) and low <= rr_breakeven_trigger_price and entry_price > rr_breakeven_trigger_price
        if stop_loss_level > entry_price // Only move if SL is above entry
            stop_loss_level := entry_price // Move SL to BE
            sl_moved_to_be := true
            label.new(bar_index, stop_loss_level, "SL->BE", style=label.style_label_up, color=color.new(color.orange, 30), textcolor=color.black, size=size.tiny)
    // Trail ATR Stop (only if not moved to BE or if new SL is lower than BE)
    if use_atr_stop
        new_stop_short = close + atr_value * atr_mult // Trail based on close
        // Only trail down, never up. Ensure it respects BE level if activated.
        stop_loss_level := math.min(nz(stop_loss_level[1], new_stop_short), new_stop_short)
        if sl_moved_to_be // If already at BE, ensure trailing stop doesn't go above entry
            stop_loss_level := math.min(stop_loss_level, entry_price)

// === PLOTTING ===

// --- Score Strength Visualization ---
score_increasing = scaled_score > scaled_score[1]
score_decreasing = scaled_score < scaled_score[1]
is_bullish_bias = scaled_score > 5.0
is_bearish_bias = scaled_score < 5.0
plotshape(is_bullish_bias and score_increasing, title="Bullish Score Boost", location=location.bottom, color=color.new(color.green, 0), style=shape.arrowup, size=size.tiny)
plotshape(is_bearish_bias and score_decreasing, title="Bearish Score Boost", location=location.top, color=color.new(color.red, 0), style=shape.arrowdown, size=size.tiny)

// --- Plot Auto Fibonacci Retracement Lines ---
if show_auto_fib and valid_fib_range
    line_start_x = math.min(nz(fib_swing_low_bar, bar_index), nz(fib_swing_high_bar, bar_index))
    line_end_x = bar_index
    label_x = line_end_x + 3 // Position labels slightly to the right

    fib_line_0    := f_update_fib_line(level_0,   fib_line_0,   line_start_x, line_end_x, fib_line_color, fib_line_width, extend_fib_lines)
    fib_label_0   := f_update_fib_label(level_0,  fib_label_0,  "0.0 (" + str.tostring(level_0, format.mintick) + ")",   label_x, fib_line_color)
    fib_line_236  := f_update_fib_line(level_236, fib_line_236, line_start_x, line_end_x, fib_line_color, fib_line_width, extend_fib_lines)
    fib_label_236 := f_update_fib_label(level_236,fib_label_236,"0.236 (" + str.tostring(level_236, format.mintick) + ")", label_x, fib_line_color)
    fib_line_382  := f_update_fib_line(level_382, fib_line_382, line_start_x, line_end_x, fib_line_color, fib_line_width, extend_fib_lines)
    fib_label_382 := f_update_fib_label(level_382,fib_label_382,"0.382 (" + str.tostring(level_382, format.mintick) + ")", label_x, fib_line_color)
    fib_line_500  := f_update_fib_line(level_500, fib_line_500, line_start_x, line_end_x, fib_line_color, fib_line_width, extend_fib_lines)
    fib_label_500 := f_update_fib_label(level_500,fib_label_500,"0.5 (" + str.tostring(level_500, format.mintick) + ")",   label_x, fib_line_color)
    fib_line_618  := f_update_fib_line(level_618, fib_line_618, line_start_x, line_end_x, fib_line_color, fib_line_width, extend_fib_lines)
    fib_label_618 := f_update_fib_label(level_618,fib_label_618,"0.618 (" + str.tostring(level_618, format.mintick) + ")", label_x, fib_line_color)
    fib_line_786  := f_update_fib_line(level_786, fib_line_786, line_start_x, line_end_x, fib_line_color, fib_line_width, extend_fib_lines)
    fib_label_786 := f_update_fib_label(level_786,fib_label_786,"0.786 (" + str.tostring(level_786, format.mintick) + ")", label_x, fib_line_color)
    fib_line_100  := f_update_fib_line(level_100, fib_line_100, line_start_x, line_end_x, fib_line_color, fib_line_width, extend_fib_lines)
    fib_label_100 := f_update_fib_label(level_100,fib_label_100,"1.0 (" + str.tostring(level_100, format.mintick) + ")",   label_x, fib_line_color)
else if not show_auto_fib or not valid_fib_range // Delete if turned off or invalid range
    f_delete_fib_objects(fib_line_0, fib_line_236, fib_line_382, fib_line_500, fib_line_618, fib_line_786, fib_line_100,
                         fib_label_0, fib_label_236, fib_label_382, fib_label_500, fib_label_618, fib_label_786, fib_label_100) // **CORRECTED CALL**


// --- Supply / Demand Zone Calculations & Plotting ---
float supply_top = na
float supply_bottom = na
float demand_top = na
float demand_bottom = na
int recent_high_bar_idx = na
int recent_low_bar_idx = na

if show_sd_zones
    float recent_high = ta.highest(high, sd_lookback)[1] // Lookback excluding current bar
    float recent_low = ta.lowest(low, sd_lookback)[1]   // Lookback excluding current bar
    recent_high_bar_idx := bar_index - ta.highestbars(high, sd_lookback)[1] // Bar index where high occurred
    recent_low_bar_idx := bar_index - ta.lowestbars(low, sd_lookback)[1]   // Bar index where low occurred

    // Get ATR at the time the pivot occurred
    float atr_at_high = ta.valuewhen(bar_index == recent_high_bar_idx, atr_value, 0) // Use 0 index to get value on that bar
    float atr_at_low = ta.valuewhen(bar_index == recent_low_bar_idx, atr_value, 0)

    // Define zone boundaries if pivots are valid and within max bars
    if not na(recent_high) and not na(atr_at_high) and atr_at_high > 0 and (bar_index - recent_high_bar_idx) <= sd_max_bars
        supply_top := recent_high + atr_at_high * sd_zone_height_atr_mult * 0.2
        supply_bottom := recent_high - atr_at_high * sd_zone_height_atr_mult * 0.8
    if not na(recent_low) and not na(atr_at_low) and atr_at_low > 0 and (bar_index - recent_low_bar_idx) <= sd_max_bars
        demand_top := recent_low + atr_at_low * sd_zone_height_atr_mult * 0.8
        demand_bottom := recent_low - atr_at_low * sd_zone_height_atr_mult * 0.2

    // Update or create boxes - **CORRECTED ASSIGNMENT using intermediate variables**
    //var box temp_supply_box_id = na
    //var int temp_supply_bar_idx = na
    [temp_supply_box_id, temp_supply_bar_idx] = f_update_sd_box(supply_box, recent_high_bar_idx, supply_top, bar_index + 1, supply_bottom, sd_color_supply, supply_bar, recent_high_bar_idx)
    supply_box := temp_supply_box_id
    supply_bar := temp_supply_bar_idx

    //var box temp_demand_box_id = na
    //var int temp_demand_bar_idx = na
    [temp_demand_box_id, temp_demand_bar_idx] = f_update_sd_box(demand_box, recent_low_bar_idx, demand_top, bar_index + 1, demand_bottom, sd_color_demand, demand_bar, recent_low_bar_idx)
    demand_box := temp_demand_box_id
    demand_bar := temp_demand_bar_idx
else // Delete boxes if turned off
    if not na(supply_box)
        box.delete(supply_box)
        supply_box := na
        supply_bar := na
    if not na(demand_box)
        box.delete(demand_box)
        demand_box := na
        demand_bar := na

// **REMOVED** S/D Cleanup block
// if bar_index - nz(last_cleaned_bar, 0) >= 100
//    ...

// --- Candle Colors ---
candle_color = close > open ? color.new(color.green, 0) : color.new(color.red, 0)
plotcandle(open, high, low, close, title="Candles", color=candle_color, wickcolor=candle_color, bordercolor=candle_color)

// --- Indicators ---
plot(ema_fast, "Fast EMA", color=color.new(color.blue, 0))
plot(ema_med, "Medium EMA", color=color.new(color.orange, 0))
plot(ema_slow, "Slow EMA", color=color.new(color.red, 0))
bb_mid_plot = plot(show_bb ? bb_basis : na, "BB Basis", color=bb_color, linewidth=1)
bb_upper_plot = plot(show_bb ? bb_upper : na, "BB Upper", color=bb_color, linewidth=1)
bb_lower_plot = plot(show_bb ? bb_lower : na, "BB Lower", color=bb_color, linewidth=1)
fill(bb_upper_plot, bb_lower_plot, color=color.new(bb_color, 90), title="BB Fill")

// --- SL and Target Lines ---
// Plot current SL only if in a trade
plot(in_long and use_atr_stop and not na(stop_loss_level) ? stop_loss_level : na, "Long SL", color=color.new(color.maroon, 0), style=plot.style_linebr, linewidth=2)
plot(in_short and use_atr_stop and not na(stop_loss_level) ? stop_loss_level : na, "Short SL", color=color.new(color.teal, 0), style=plot.style_linebr, linewidth=2)
// Plot Fib Target only if in a trade and enabled
plot(in_long and use_fib_exit and not na(fib_target_level_exit) ? fib_target_level_exit : na, "Long Fib Tgt", color=color.new(color.fuchsia, 0), style=plot.style_linebr, linewidth=1)
plot(in_short and use_fib_exit and not na(fib_target_level_exit) ? fib_target_level_exit : na, "Short Fib Tgt", color=color.new(color.fuchsia, 0), style=plot.style_linebr, linewidth=1)
// Plot RR Target only if in a trade and enabled
plot(in_long and use_rr_tp and not na(rr_target_price) ? rr_target_price : na, "Long RR Tgt", color=color.new(color.aqua, 0), style=plot.style_linebr, linewidth=1)
plot(in_short and use_rr_tp and not na(rr_target_price) ? rr_target_price : na, "Short RR Tgt", color=color.new(color.aqua, 0), style=plot.style_linebr, linewidth=1)


// --- Entry/Exit Shapes and Labels ---
plotshape(exited_long_this_bar, "Exit Long Shape", location=location.abovebar, color=color.new(color.maroon, 50), style=shape.xcross, size=size.small) // Changed to xcross
plotshape(exited_short_this_bar, "Exit Short Shape", location=location.belowbar, color=color.new(color.teal, 50), style=shape.xcross, size=size.small) // Changed to xcross
plotshape(use_skip_trade_filter and skip_trade, title="Skipped Weak Signal", location=location.bottom, color=color.new(color.orange, 50), style=shape.xcross, size=size.tiny)

// --- Entry Labels ---
if enter_long_condition
    bool is_first_trade = na(last_trade_bar[1]) // Check if the last trade bar on the *previous* bar was NA
    label_text = is_first_trade ? "Buy" : "Re-Buy"
    label_color = is_first_trade ? color.new(color.green, 20) : color.new(color.lime, 20)
    text_color = color.white
    label.new(bar_index, low - atr_value * 0.2, label_text,
              color=label_color, textcolor=text_color,
              style=label.style_label_up, size=size.small,
              textalign=text.align_center,
              tooltip="Entry: " + buy_reason_text + "\nScore: " + str.tostring(scaled_score, "#.##") + "\nSize: " + str.tostring(position_size, "#.##"))

if enter_short_condition
    bool is_first_trade = na(last_trade_bar[1])
    label_text = is_first_trade ? "Sell" : "Re-Sell"
    label_color = is_first_trade ? color.new(color.red, 20) : color.new(color.maroon, 20)
    text_color = color.white
    label.new(bar_index, high + atr_value * 0.2, label_text,
              color=label_color, textcolor=text_color,
              style=label.style_label_down, size=size.small,
              textalign=text.align_center,
              tooltip="Entry: " + sell_reason_text + "\nScore: " + str.tostring(scaled_score, "#.##") + "\nSize: " + str.tostring(position_size, "#.##"))

// --- Exit Reason Labels ---
// Only plot exit labels if the reason contains "TP"
if exited_long_this_bar and not na(exit_reason) and str.contains(exit_reason, "TP")
    label_color_long = last_trade_pnl > 0 ? color.new(color.green, 30) : color.new(color.red, 30)
    label_ypos_long = high + atr_value * 0.3 // Position TP labels slightly higher
    label.new(bar_index, label_ypos_long, exit_reason,
              color=label_color_long, textcolor=color.white,
              style=label.style_label_down, size=size.tiny, // Made smaller
              tooltip="Exit Reason: " + exit_reason + "\nPnL: " + str.tostring(last_trade_pnl, "#.##") + "%")

if exited_short_this_bar and not na(exit_reason) and str.contains(exit_reason, "TP")
    label_color_short = last_trade_pnl > 0 ? color.new(color.green, 30) : color.new(color.red, 30)
    label_ypos_short = low - atr_value * 0.3 // Position TP labels slightly lower
    label.new(bar_index, label_ypos_short, exit_reason,
              color=label_color_short, textcolor=color.white,
              style=label.style_label_up, size=size.tiny, // Made smaller
              tooltip="Exit Reason: " + exit_reason + "\nPnL: " + str.tostring(last_trade_pnl, "#.##") + "%")

// --- Background and Data Window ---
bgcolor(in_long ? color.new(color.green, 85) : in_short ? color.new(color.red, 85) : na)
plot(scaled_score, "Confidence Score", color=color.new(color.gray,0), display=display.data_window)
// Calculate live PnL for display if in trade
float live_pnl = na // Use float here
if in_long and not na(entry_price)
    live_pnl := (close - entry_price) / entry_price * 100
if in_short and not na(entry_price)
    live_pnl := (entry_price - close) / entry_price * 100
plot(live_pnl, "Live Trade PnL (%)", color=live_pnl > 0 ? color.green : color.red, display=display.data_window)
plot(in_long or in_short ? bars_held : na, "Bars Held", color=color.new(color.orange, 0), display=display.data_window)
plot(cumulative_pnl, "Cumulative PnL % (Hypothetical)", color=color.new(color.purple, 0), display=display.data_window)
plot(in_long or in_short ? position_size : na, "Position Size (Hypothetical)", color=color.new(color.blue, 0), display=display.data_window)
plot(current_equity, "Equity (Hypothetical)", color=color.new(color.black, 0), display=display.data_window)

// --- Added Plots for Debugging/Monitoring ---
// Plot the stop distance calculated at entry
plot(in_long or in_short ? stop_distance_points : na, "Stop Distance Pts", color=color.orange, display=display.data_window)
// Plot the position size calculated at entry
plot(in_long or in_short ? position_size : na, "Calculated Size", color=color.blue, display=display.data_window)

// **REMOVED** Duplicate plot: plot(scaled_score, "Optimized Score", ...)

// === ALERTS ===
// **CORRECTED**: Remove dynamic reason strings from entry alert messages

// Entry Alerts
alertcondition(enter_long_condition, title="Enter Buy", message="BUY ALERT: {{ticker}} at {{close}}. Score: {{plot('Confidence Score')}}. Size: {{plot('Calculated Size')}}") // Removed dynamic reason
alertcondition(enter_short_condition, title="Enter Sell", message="SELL ALERT: {{ticker}} at {{close}}. Score: {{plot('Confidence Score')}}. Size: {{plot('Calculated Size')}}") // Removed dynamic reason

// Exit Alerts (Use the stored 'exit_reason' which now includes "Long TP" / "Short TP")
alertcondition(exited_long_this_bar, title="Exit Long Signal", message="EXIT LONG: {{ticker}} at {{str.tostring(nz(exit_price, close))}}. Reason: {{exit_reason}}. PnL: {{str.tostring(last_trade_pnl, '#.##')}}%")
alertcondition(exited_short_this_bar, title="Exit Short Signal", message="EXIT SHORT: {{ticker}} at {{str.tostring(nz(exit_price, close))}}. Reason: {{exit_reason}}. PnL: {{str.tostring(last_trade_pnl, '#.##')}}%")

// Potential Signal Alerts (Fires when score/filters align, but before cooldown/position check)
alertcondition(entry_signal_buy and not entry_signal_buy[1], title="Potential Buy Setup", message="{{ticker}}: Potential Buy Signal forming. Score: {{plot('Confidence Score')}}. Price: {{close}}")
alertcondition(entry_signal_sell and not entry_signal_sell[1], title="Potential Sell Setup", message="{{ticker}}: Potential Sell Signal forming. Score: {{plot('Confidence Score')}}. Price: {{close}}")

// Specific TP Hit Alerts (Check the exit_reason variable)
alertcondition(exited_long_this_bar and str.contains(exit_reason, "Long TP"), title="Long TP Hit", message="LONG TP HIT: {{ticker}} at {{str.tostring(nz(exit_price, close))}}. PnL: {{str.tostring(last_trade_pnl, '#.##')}}%")
alertcondition(exited_short_this_bar and str.contains(exit_reason, "Short TP"), title="Short TP Hit", message="SHORT TP HIT: {{ticker}} at {{str.tostring(nz(exit_price, close))}}. PnL: {{str.tostring(last_trade_pnl, '#.##')}}%")

// Specific ATR Stop Alerts
alertcondition(exited_long_this_bar and str.contains(exit_reason, "ATR SL"), title="ATR Stop Long Hit", message="ATR STOP LONG: {{ticker}} at {{str.tostring(nz(exit_price, close))}}. Exit triggered.")
alertcondition(exited_short_this_bar and str.contains(exit_reason, "ATR SL"), title="ATR Stop Short Hit", message="ATR STOP SHORT: {{ticker}} at {{str.tostring(nz(exit_price, close))}}. Exit triggered.")

