//@version=6
indicator(
     title="CS5", // Keep title as CS5
     shorttitle="CS5",
     overlay=true,
     max_lines_count=50,
     max_labels_count=50,
     max_boxes_count=10
     )

// === INPUTS ===

// --- Core Entry & Exit ---
grpCore = "Core Strategy Settings"
exitScoreDropThreshold = input.float(1.5, "Exit on Score Drop Threshold", minval=0.1, maxval=5.0, step=0.1, group=grpCore,
  tooltip="Exit if score drops below (5.0 - Threshold) for longs or above (5.0 + Threshold) for shorts")
useFibBounceEntry = input.bool(true, "Use Fib 0.5-0.618 Bounce Entry (Long)", group=grpCore,
  tooltip="Generate BUY signal on upward bounce from 0.5-0.618 Fib in uptrend")
useFibBounceSell = input.bool(true, "Use Fib 0.5-0.382 Bounce Entry (Short)", group=grpCore,
  tooltip="Generate SELL signal on downward bounce from 0.5-0.382 Fib in downtrend")
fibBounceLookback = input.int(3, "Fib Bounce Price Lookback Bars", minval=1, maxval=10, group=grpCore,
  tooltip="Bars back to check for Fib zone touch")
useEmaBounceBuy = input.bool(true, title="Use EMA Bounce Entry (Long)?", group=grpCore)
useEmaBounceSell = input.bool(true, title="Use EMA Bounce Entry (Short)?", group=grpCore)
emaBounceLookback = input.int(2, title="EMA Bounce Price Lookback Bars", minval=1, maxval=5, group=grpCore)
emaBounceSource = input.string("Fast EMA", title="EMA Source for Bounce", options=["Fast EMA", "Medium EMA"], group=grpCore)
useBbMidBounceBuy = input.bool(true, title="Use BB Mid Bounce Entry (Long)?", group=grpCore)
useBbMidBounceSell = input.bool(true, title="Use BB Mid Bounce Entry (Short)?", group=grpCore)
bbBounceLookback = input.int(2, title="BB Mid Bounce Price Lookback Bars", minval=1, maxval=5, group=grpCore)
useVolBreakoutBuy = input.bool(true, title="Use Volume Breakout Entry (Long)?", group=grpCore)
useVolBreakoutSell = input.bool(true, title="Use Volume Breakout Entry (Short)?", group=grpCore)
cooldownBars = input.int(5, "Trade Cooldown Bars", minval=0, group=grpCore)

// --- Weak Signal Filter Settings ---
grpSkip = "Weak Signal Filter Settings"
useSkipTradeFilter = input.bool(true, "Enable Weak Signal Filter?", group=grpSkip,
  tooltip="If enabled, signals on bars matching the criteria below will be skipped.")
skipAtrFactor = input.float(0.3, "Min Price Move (ATR Factor)", minval=0.0, step=0.05, group=grpSkip,
  tooltip="Skip signal if |close - open| is less than ATR * this factor. Set to 0 to disable this part of the check.")
skipVolMALen = input.int(20, "Volume MA Length for Filter", minval=1, group=grpSkip,
  tooltip="The lookback period for the Simple Moving Average of volume used in the filter.")
skipVolMultiplier = input.float(1.2, "Min Volume Spike (x MA)", minval=0.0, step=0.1, group=grpSkip,
  tooltip="Skip signal if volume is less than its MA * this multiplier. Set to 0 to disable this part of the check.")

// --- EMAs ---
grpEMA = "EMA Settings"
emaFastLen = input.int(9, "Fast EMA Length", minval=1, group=grpEMA)
emaMedLen = input.int(14, "Medium EMA Length", minval=1, group=grpEMA)
emaSlowLen = input.int(21, "Slow EMA Length", minval=1, group=grpEMA)
useEmaExit = input.bool(true, "Use Fast/Med EMA Cross for Exit", group=grpEMA)

// --- Bollinger Bands ---
grpBB = "Bollinger Bands Settings"
showBB = input.bool(true, "Show Bollinger Bands", group=grpBB)
bbLen = input.int(20, "BB Length", minval=1, group=grpBB)
bbStdDev = input.float(2.0, "BB StdDev Multiplier", minval=0.1, group=grpBB)
bbColor = input.color(color.new(color.gray, 70), "BB Color", group=grpBB)
useBBReturnExit = input.bool(true, "Use BB Return to Mean for Exit", group=grpBB)

// --- RSI ---
grpRSI = "RSI Settings"
rsiLen = input.int(14, "RSI Length", minval=1, group=grpRSI)
rsiBuyLevel = input.float(55.0, "RSI Buy Threshold (>)", group=grpRSI)
rsiSellLevel = input.float(45.0, "RSI Sell Threshold (<)", group=grpRSI)
useRsiDivExit = input.bool(false, "Use RSI Divergence Exit", group=grpRSI,
  tooltip="Exit on RSI divergence (lower high/higher low)")
rsiConfirmFibBounce = input.bool(true, "Require RSI Confirmation for Fib Bounce", group=grpRSI,
  tooltip="Long: RSI>40 Rising. Short: RSI<60 Falling")
rsiConfirmEmaBounce = input.bool(false, title="Require RSI Confirmation for EMA Bounce?", group=grpRSI)
rsiConfirmBbBounce = input.bool(false, title="Require RSI Confirmation for BB Bounce?", group=grpRSI)

// --- MACD ---
grpMACD = "MACD Settings"
macdFastLen = input.int(12, "MACD Fast Length", group=grpMACD)
macdSlowLen = input.int(26, "MACD Slow Length", group=grpMACD)
macdSignalLen = input.int(9, "MACD Signal Length", group=grpMACD)

// --- Volume ---
grpVol = "Volume Settings"
volMALen = input.int(50, "Volume MA Length", minval=1, group=grpVol)
volMultiplier = input.float(1.5, "Volume Breakout Multiplier (> MA)", group=grpVol)
useVolFadeExit = input.bool(true, "Use Low Volume Pullback Exit", group=grpVol)

// --- ATR Stop Loss ---
grpATR = "ATR Stop Loss"
useAtrStop = input.bool(true, "Use ATR Stop Loss for Exit", group=grpATR)
atrLen = input.int(14, "ATR Length", minval=1, group=grpATR)
atrMult = input.float(2.0, "ATR Multiplier", minval=0.1, group=grpATR)

// --- Fibonacci EXIT Target ---
grpFibExit = "Fibonacci Exit Target"
useFibExit = input.bool(true, "Use Fib Extension Exit Target", group=grpFibExit)
fibLookbackExit = input.int(30, "Fib Exit Swing Lookback", minval=5, group=grpFibExit,
  tooltip="Bars back from entry to find swing point")
fibExtensionLevel = input.float(1.618, "Fib Extension Target", minval=0.1, group=grpFibExit)

// --- Auto Fibonacci RETRACEMENT ---
grpFibRet = "Auto Fib Retracement"
showAutoFib = input.bool(true, "Show Auto Fib Retracement", group=grpFibRet)
fibPivotLookback = input.int(15, "Pivot Lookback (Left/Right Bars)", minval=2, group=grpFibRet)
fibMaxBars = input.int(200, "Max Bars Back for Pivots", minval=20, group=grpFibRet)
fibLineColor = input.color(color.new(color.gray, 50), "Fib Line Color", group=grpFibRet)
fibLineWidth = input.int(1, "Fib Line Width", minval=1, group=grpFibRet)
extend_fib_lines = input(extend.right, title="Extend Fib Lines", group = grpFibRet) // Input to control extension

// --- Trend / Market Condition Filters ---
grpFilter = "Trend Filters"
useEmaTrendFilter = input.bool(true, "Require EMA Trend (Med > Slow)", group=grpFilter)
useAdxFilter = input.bool(true, "Use ADX Filter", group=grpFilter)
adxLen = input.int(14, "ADX Length", minval=1, group=grpFilter)
adxThreshold = input.float(20.0, "ADX Trend Strength Threshold (>)", minval=0, group=grpFilter)
useAdxDirectionFilter = input.bool(true, "Require ADX Direction (DI+ vs DI-)", group=grpFilter)
useHtfFilter = input.bool(false, "Use Higher Timeframe EMA Filter", group=grpFilter)
htf = input.timeframe("60", "Higher Timeframe for Filter", group=grpFilter)
htfEmaLen = input.int(50, "HTF EMA Length", group=grpFilter)

// --- SCORE CALCULATION WEIGHTS ---
grpScore = "Confidence Score Weights"
wEmaTrend = input.int(2, "Weight: EMA Trend", group=grpScore, minval=0)
wEmaSignal = input.int(1, "Weight: EMA Cross Signal", group=grpScore, minval=0)
wRsiThresh = input.int(1, "Weight: RSI Threshold", group=grpScore, minval=0)
wMacdSignal = input.int(1, "Weight: MACD Signal Cross", group=grpScore, minval=0)
wMacdZero = input.int(1, "Weight: MACD Zero Cross", group=grpScore, minval=0)
wVolBreak = input.int(1, "Weight: Volume Breakout", group=grpScore, minval=0)
wAdxStrength = input.int(1, "Weight: ADX Strength", group=grpScore, minval=0)
wAdxDirection = input.int(1, "Weight: ADX Direction", group=grpScore, minval=0)
wHtfTrend = input.int(2, "Weight: HTF Trend", group=grpScore, minval=0)
wFibBounce = input.int(2, "Weight: Fib Bounce Signal", group=grpScore, minval=0)
wEmaBounce = input.int(1, "Weight: EMA Bounce Signal", group=grpScore, minval=0)
wBbBounce = input.int(1, "Weight: BB Mid Bounce Signal", group=grpScore, minval=0)

// --- Supply/Demand Zone Inputs ---
grpSD = "Supply/Demand Zones"
showSDZones = input.bool(true, "Show Supply/Demand Zones", group=grpSD)
sdLookback = input.int(50, "S/D Zone Lookback Bars", minval=10, group=grpSD, tooltip="How far back to look for highs/lows to define zones.")
sdZoneHeightAtrMult = input.float(0.5, "S/D Zone Height (ATR Multiplier)", minval=0.1, step=0.1, group=grpSD, tooltip="Height of the zone based on ATR at the time the high/low was formed.")
sdColorSupply = input.color(color.new(color.red, 80), "Supply Zone Color", group=grpSD)
sdColorDemand = input.color(color.new(color.green, 80), "Demand Zone Color", group=grpSD)
// ADDED: Inputs for S/D Exit Logic
useSupplyZoneExit = input.bool(true, "Use Supply Zone + Green Candle Exit (Long)?", group=grpSD)
useDemandZoneExit = input.bool(true, "Use Demand Zone + Green Candle Exit (Short)?", group=grpSD)


// === CALCULATIONS ===

// --- Indicators ---
emaFast = ta.ema(close, emaFastLen)
emaMed = ta.ema(close, emaMedLen)
emaSlow = ta.ema(close, emaSlowLen)
[bbMiddle, bbUpper, bbLower] = ta.bb(close, bbLen, bbStdDev)
priceRsi = ta.rsi(close, rsiLen)
[macdLine, signalLine, histLine] = ta.macd(close, macdFastLen, macdSlowLen, macdSignalLen)
volMA = ta.sma(volume, volMALen)
atrVal = ta.atr(atrLen)
[diPos, diNeg, adxVal] = ta.dmi(adxLen, adxLen)
htfEma = request.security(syminfo.tickerid, htf, ta.ema(close, htfEmaLen), lookahead=barmerge.lookahead_off)
lowestForFibExit = ta.lowest(low, fibLookbackExit)[1]
highestForFibExit = ta.highest(high, fibLookbackExit)[1]

// --- Weak Signal Filter Calculations ---
expectedMove = atrVal * skipAtrFactor
priceMove = math.abs(close - open)
momentumOk = skipAtrFactor <= 0 or (priceMove > expectedMove)
skipVolMA = ta.sma(volume, skipVolMALen)
volumeOkForFilter = skipVolMultiplier <= 0 or (volume > skipVolMA * skipVolMultiplier)
skipTrade = not (momentumOk and volumeOkForFilter)

// --- Supply / Demand Zone Calculations (Moved Here) ---
float supplyTop = na
float supplyBottom = na
float demandTop = na
float demandBottom = na
int recentHighBar = na // Keep track of the bar index
int recentLowBar = na  // Keep track of the bar index

if showSDZones
    recentHigh = ta.highest(high, sdLookback)[1]
    recentHighBar := bar_index - ta.highestbars(high, sdLookback)[1] // Assign bar index
    recentLow = ta.lowest(low, sdLookback)[1]
    recentLowBar := bar_index - ta.lowestbars(low, sdLookback)[1]   // Assign bar index
    atrAtHigh = ta.valuewhen(bar_index[1] == recentHighBar, atrVal[1], 0) // Use previous bar values for stability
    atrAtLow = ta.valuewhen(bar_index[1] == recentLowBar, atrVal[1], 0)   // Use previous bar values for stability

    if not na(recentHigh) and not na(atrAtHigh) and atrAtHigh > 0
        supplyTop := recentHigh + atrAtHigh * sdZoneHeightAtrMult * 0.2
        supplyBottom := recentHigh - atrAtHigh * sdZoneHeightAtrMult * 0.8
    if not na(recentLow) and not na(atrAtLow) and atrAtLow > 0
        demandTop := recentLow + atrAtLow * sdZoneHeightAtrMult * 0.8
        demandBottom := recentLow - atrAtLow * sdZoneHeightAtrMult * 0.2

// --- Basic Conditions ---
condEmaFastSlowCrossBuy = ta.crossover(emaFast, emaSlow)
condEmaFastSlowCrossSell = ta.crossunder(emaFast, emaSlow)
condEmaFastMedCrossBuy = ta.crossover(emaFast, emaMed)
condEmaFastMedCrossSell = ta.crossunder(emaFast, emaMed)
condBbReturnMeanBuy = ta.crossover(close, bbMiddle)
condBbReturnMeanSell = ta.crossunder(close, bbMiddle)
condRsiBuy = priceRsi > rsiBuyLevel
condRsiSell = priceRsi < rsiSellLevel
condMacdSignalCrossBuy = ta.crossover(macdLine, signalLine)
condMacdSignalCrossSell = ta.crossunder(macdLine, signalLine)
condMacdZeroCrossBuy = ta.crossover(macdLine, 0)
condMacdZeroCrossSell = ta.crossunder(macdLine, 0)
condHighVol = volume > volMA * volMultiplier
condVolBreakoutBuy = condHighVol and close > open and close > emaSlow
condVolBreakoutSell = condHighVol and close < open and close < emaSlow
condVolFadeLong = close < emaFast and volume < volMA
condVolFadeShort = close > emaFast and volume < volMA

// --- Filter Conditions ---
condEmaTrendOkBuy = not useEmaTrendFilter or emaMed > emaSlow
condEmaTrendOkSell = not useEmaTrendFilter or emaMed < emaSlow
condAdxStrengthOk = not useAdxFilter or adxVal > adxThreshold
condAdxDirectionOkBuy = not useAdxDirectionFilter or diPos > diNeg
condAdxDirectionOkSell = not useAdxDirectionFilter or diNeg > diPos
condAdxFilterOkBuy = condAdxStrengthOk and condAdxDirectionOkBuy
condAdxFilterOkSell = condAdxStrengthOk and condAdxDirectionOkSell
condHtfFilterOkBuy = not useHtfFilter or close > htfEma
condHtfFilterOkSell = not useHtfFilter or close < htfEma
bbWidth = bbUpper - bbLower
minBBWidth = ta.sma(bbWidth, 10) * 0.3
isInRange = bbWidth < minBBWidth or (useAdxFilter and adxVal < 12)
allFiltersOkBuy = condEmaTrendOkBuy and condAdxFilterOkBuy and condHtfFilterOkBuy and not isInRange
allFiltersOkSell = condEmaTrendOkSell and condAdxFilterOkSell and condHtfFilterOkSell and not isInRange

// --- Auto Fibonacci Retracement Calculation ---
var float lastPivotHighPrice = na, var int lastPivotHighBar = na
var float lastPivotLowPrice = na, var int lastPivotLowBar = na
float swingHighCheck = ta.highest(high, fibPivotLookback)
int swingHighCheckBar = bar_index - ta.highestbars(high, fibPivotLookback)
float swingLowCheck = ta.lowest(low, fibPivotLookback)
int swingLowCheckBar = bar_index - ta.lowestbars(low, fibPivotLookback)
if not na(swingHighCheck) and (na(lastPivotHighBar) or swingHighCheckBar > lastPivotHighBar or swingHighCheck >= nz(lastPivotHighPrice, -1.0e10))
    if bar_index - swingHighCheckBar <= fibMaxBars
        lastPivotHighPrice := swingHighCheck
        lastPivotHighBar := swingHighCheckBar
if not na(swingLowCheck) and (na(lastPivotLowBar) or swingLowCheckBar > lastPivotLowBar or swingLowCheck <= nz(lastPivotLowPrice, 1.0e10))
    if bar_index - swingLowCheckBar <= fibMaxBars
        lastPivotLowPrice := swingLowCheck
        lastPivotLowBar := swingLowCheckBar
if not na(lastPivotHighBar) and bar_index - lastPivotHighBar > fibMaxBars
    lastPivotHighPrice := na
    lastPivotHighBar := na
if not na(lastPivotLowBar) and bar_index - lastPivotLowBar > fibMaxBars
    lastPivotLowPrice := na
    lastPivotLowBar := na
float fibSwingHigh = na, int fibSwingHighBar = na
float fibSwingLow = na, int fibSwingLowBar = na
if not na(lastPivotHighPrice) and not na(lastPivotLowPrice)
    if nz(lastPivotHighBar, -1) > nz(lastPivotLowBar, -1)
        fibSwingHigh := lastPivotHighPrice
        fibSwingHighBar := lastPivotHighBar
        fibSwingLow := lastPivotLowPrice
        fibSwingLowBar := lastPivotLowBar
    else
        fibSwingLow := lastPivotLowPrice
        fibSwingLowBar := lastPivotLowBar
        fibSwingHigh := lastPivotHighPrice
        fibSwingHighBar := lastPivotHighBar
var line fibLine0 = na, var line fibLine236 = na, var line fibLine382 = na, var line fibLine500 = na
var line fibLine618 = na, var line fibLine786 = na, var line fibLine100 = na
var label fibLabel0 = na, var label fibLabel236 = na, var label fibLabel382 = na, var label fibLabel500 = na
var label fibLabel618 = na, var label fibLabel786 = na, var label fibLabel100 = na
float level_0 = na, float level_236 = na, float level_382 = na, float level_500 = na
float level_618 = na, float level_786 = na, float level_100 = na
bool validFibRange = false, bool isUptrendFib = false
if not na(fibSwingHigh) and not na(fibSwingLow) and fibSwingHigh > fibSwingLow and not na(fibSwingHighBar) and not na(fibSwingLowBar)
    validFibRange := true
    float fibRange = fibSwingHigh - fibSwingLow
    isUptrendFib := fibSwingHighBar > fibSwingLowBar
    if isUptrendFib 
        level_0 := fibSwingLow
        level_100 := fibSwingHigh
    else 
        level_0 := fibSwingHigh
        level_100 := fibSwingLow
    level_236 := level_0 + (level_100 - level_0) * 0.236
    level_382 := level_0 + (level_100 - level_0) * 0.382
    level_500 := level_0 + (level_100 - level_0) * 0.500
    level_618 := level_0 + (level_100 - level_0) * 0.618
    level_786 := level_0 + (level_100 - level_0) * 0.786

// --- Fibonacci Bounce Condition (Buy & Sell) ---
bool touchedFibZoneBuy = false
if useFibBounceEntry and isUptrendFib and not na(level_618) and not na(level_500)
    for i = 0 to fibBounceLookback - 1
        if low[i] <= level_618 and low[i] >= level_500
            touchedFibZoneBuy := true
            break
bool bouncedAbove50Buy = close > level_500
bool rsiConfirmsBounceBuy = not rsiConfirmFibBounce or (priceRsi > 40 and priceRsi > priceRsi[1])
condFibBounceBuy = useFibBounceEntry and isUptrendFib and touchedFibZoneBuy and bouncedAbove50Buy and rsiConfirmsBounceBuy
bool touchedFibZoneSell = false
if useFibBounceSell and not isUptrendFib and validFibRange and not na(level_382) and not na(level_500)
    for i = 0 to fibBounceLookback - 1
        if high[i] >= level_382 and high[i] <= level_500 
            touchedFibZoneSell := true
            break
bool rejectedBelow50Sell = close < level_500
bool rsiConfirmsBounceSell = not rsiConfirmFibBounce or (priceRsi < 60 and priceRsi < priceRsi[1])
condFibBounceSell = useFibBounceSell and not isUptrendFib and validFibRange and touchedFibZoneSell and rejectedBelow50Sell and rsiConfirmsBounceSell

// --- EMA Bounce Condition (Buy & Sell) ---
float emaSource = emaBounceSource == "Fast EMA" ? emaFast : emaMed
bool touchedEmaBuy = false
if useEmaBounceBuy
    for i = 1 to emaBounceLookback
        if low[i] <= emaSource[i]
            touchedEmaBuy := true
            break
bool touchedEmaSell = false
if useEmaBounceSell
    for i = 1 to emaBounceLookback
        if high[i] >= emaSource[i]
            touchedEmaSell := true
            break
bool rsiConfirmsEmaBuy = not rsiConfirmEmaBounce or (priceRsi > 40 and priceRsi > priceRsi[1])
bool rsiConfirmsEmaSell = not rsiConfirmEmaBounce or (priceRsi < 60 and priceRsi < priceRsi[1])
condEmaBounceBuy = useEmaBounceBuy and touchedEmaBuy and close > emaSource and close > open and rsiConfirmsEmaBuy
condEmaBounceSell = useEmaBounceSell and touchedEmaSell and close < emaSource and close < open and rsiConfirmsEmaSell

// --- BB Middle Bounce Condition (Buy & Sell) ---
bool touchedBbMidBuy = false
if useBbMidBounceBuy
    for i = 1 to bbBounceLookback
        if low[i] <= bbMiddle[i]
            touchedBbMidBuy := true
            break
bool touchedBbMidSell = false
if useBbMidBounceSell
    for i = 1 to bbBounceLookback
        if high[i] >= bbMiddle[i]
            touchedBbMidSell := true
            break
bool rsiConfirmsBbBuy = not rsiConfirmBbBounce or (priceRsi > 40 and priceRsi > priceRsi[1])
bool rsiConfirmsBbSell = not rsiConfirmBbBounce or (priceRsi < 60 and priceRsi < priceRsi[1])
condBbMidBounceBuy = useBbMidBounceBuy and touchedBbMidBuy and close > bbMiddle and close > open and rsiConfirmsBbBuy
condBbMidBounceSell = useBbMidBounceSell and touchedBbMidSell and close < bbMiddle and close < open and rsiConfirmsBbSell

// --- Confidence Score Calculation ---
float buyScore_bar = 0.0, sellScore_bar = 0.0
if condEmaTrendOkBuy
    buyScore_bar += wEmaTrend
if condEmaTrendOkSell
    sellScore_bar += wEmaTrend
if condEmaFastSlowCrossBuy
    buyScore_bar += wEmaSignal
if condEmaFastSlowCrossSell
    sellScore_bar += wEmaSignal
if condRsiBuy
    buyScore_bar += wRsiThresh
if condRsiSell
    sellScore_bar += wRsiThresh
if condMacdSignalCrossBuy
    buyScore_bar += wMacdSignal
if condMacdSignalCrossSell
    sellScore_bar += wMacdSignal
if condMacdZeroCrossBuy
    buyScore_bar += wMacdZero
if condMacdZeroCrossSell
    sellScore_bar += wMacdZero
if useVolBreakoutBuy and condVolBreakoutBuy
    buyScore_bar += wVolBreak
if useVolBreakoutSell and condVolBreakoutSell
    sellScore_bar += wVolBreak
if condAdxStrengthOk and condAdxDirectionOkBuy
    buyScore_bar += wAdxStrength
if condAdxStrengthOk and condAdxDirectionOkSell
    sellScore_bar += wAdxStrength
if condAdxDirectionOkBuy
    buyScore_bar += wAdxDirection
if condAdxDirectionOkSell
    sellScore_bar += wAdxDirection
if condHtfFilterOkBuy
    buyScore_bar += wHtfTrend
if condHtfFilterOkSell
    sellScore_bar += wHtfTrend
if useFibBounceEntry and condFibBounceBuy
    buyScore_bar += wFibBounce
if useFibBounceSell and condFibBounceSell
    sellScore_bar += wFibBounce
if useEmaBounceBuy and condEmaBounceBuy
    buyScore_bar += wEmaBounce
if useEmaBounceSell and condEmaBounceSell
    sellScore_bar += wEmaBounce
if useBbMidBounceBuy and condBbMidBounceBuy
    buyScore_bar += wBbBounce
if useBbMidBounceSell and condBbMidBounceSell
    sellScore_bar += wBbBounce

totalPossibleScore = 0.0
totalPossibleScore += useEmaTrendFilter ? wEmaTrend : 0
totalPossibleScore += wEmaSignal
totalPossibleScore += wRsiThresh
totalPossibleScore += wMacdSignal
totalPossibleScore += wMacdZero
totalPossibleScore += useVolBreakoutBuy or useVolBreakoutSell ? wVolBreak : 0
totalPossibleScore += useAdxFilter ? wAdxStrength : 0
totalPossibleScore += useAdxDirectionFilter ? wAdxDirection : 0
totalPossibleScore += useHtfFilter ? wHtfTrend : 0
totalPossibleScore += useFibBounceEntry or useFibBounceSell ? wFibBounce : 0
totalPossibleScore += useEmaBounceBuy or useEmaBounceSell ? wEmaBounce : 0
totalPossibleScore += useBbMidBounceBuy or useBbMidBounceSell ? wBbBounce : 0
totalPossibleScore := math.max(1.0, totalPossibleScore)
netScore = buyScore_bar - sellScore_bar
scaledScore = (netScore / totalPossibleScore) * 5.0 + 5.0
scaledScore := math.max(0.0, math.min(10.0, scaledScore))

// --- RSI Divergence ---
rsiPeak = ta.pivothigh(priceRsi, 5, 2)
rsiTrough = ta.pivotlow(priceRsi, 5, 2)
pricePeak = ta.pivothigh(high, 5, 2)
priceTrough = ta.pivotlow(low, 5, 2)
validRsiTrough = not na(rsiTrough[1]) and not na(priceTrough[1])
validRsiPeak = not na(rsiPeak[1]) and not na(pricePeak[1])
bullishRsiDiv = validRsiTrough and priceRsi > rsiTrough[1] and low < priceTrough[1]
bearishRsiDiv = validRsiPeak and priceRsi < rsiPeak[1] and high > pricePeak[1]
condRsiBullDivExit = useRsiDivExit and bearishRsiDiv
condRsiBearDivExit = useRsiDivExit and bullishRsiDiv

// --- State Tracking ---
var bool inLong = false, var bool inShort = false
var float stopLossLevel = na, var float fibTargetLevelExit = na, var float entryPrice = na
var string exitReason = ""
var bool wasInLong = false, var bool wasInShort = false
wasInLong := inLong
wasInShort := inShort
var float tradePnL = na, var int barsHeld = 0
var int atrStopHitLongCount = 0, var int atrStopHitShortCount = 0
var int lastTradeBar = na
canReEnter = na(lastTradeBar) or (bar_index - lastTradeBar > cooldownBars)
// ADDED: State Vars for S/D Zone Exit Logic
var bool enteredSupplyZoneLong = false
var bool enteredDemandZoneShort = false
var bool armedToExitLongOnGreen = false
var bool armedToExitShortOnGreen = false


// --- Update S/D Zone Entry & Arming Flags ---
if showSDZones
    if inLong and not enteredSupplyZoneLong and not na(supplyBottom) and high >= supplyBottom
        enteredSupplyZoneLong := true // Entered supply zone while long

    if enteredSupplyZoneLong and not armedToExitLongOnGreen // Only arm once after entering
        armedToExitLongOnGreen := true // Arm the exit check for the next green candle

    if inShort and not enteredDemandZoneShort and not na(demandTop) and low <= demandTop
        enteredDemandZoneShort := true // Entered demand zone while short

    if enteredDemandZoneShort and not armedToExitShortOnGreen // Only arm once after entering
        armedToExitShortOnGreen := true // Arm the exit check for the next green candle


// --- Entry Logic ---
isPotentialEmaBuy = condEmaFastSlowCrossBuy and allFiltersOkBuy
isPotentialFibBuy = condFibBounceBuy and allFiltersOkBuy
isPotentialEmaBounceBuy = condEmaBounceBuy and allFiltersOkBuy
isPotentialBbMidBounceBuy = condBbMidBounceBuy and allFiltersOkBuy
isPotentialVolBuy = useVolBreakoutBuy and condVolBreakoutBuy and allFiltersOkBuy
isPotentialEmaSell = condEmaFastSlowCrossSell and allFiltersOkSell
isPotentialFibSell = condFibBounceSell and allFiltersOkSell
isPotentialEmaBounceSell = condEmaBounceSell and allFiltersOkSell
isPotentialBbMidBounceSell = condBbMidBounceSell and allFiltersOkSell
isPotentialVolSell = useVolBreakoutSell and condVolBreakoutSell and allFiltersOkSell
plotSignalBuy = (isPotentialEmaBuy or isPotentialFibBuy or isPotentialEmaBounceBuy or isPotentialBbMidBounceBuy or isPotentialVolBuy) and (not useSkipTradeFilter or not skipTrade) and canReEnter
plotSignalSell = (isPotentialEmaSell or isPotentialFibSell or isPotentialEmaBounceSell or isPotentialBbMidBounceSell or isPotentialVolSell) and (not useSkipTradeFilter or not skipTrade) and canReEnter
isNewTradeBuy = plotSignalBuy and not wasInLong
isNewTradeSell = plotSignalSell and not wasInShort

// Update Trade State
if isNewTradeBuy
    inLong := true
    inShort := false
    entryPrice := close
    stopLossLevel := low - atrVal * atrMult
    swingLowPriceExit = lowestForFibExit
    swingRangeFibExit = entryPrice - swingLowPriceExit
    fibTargetLevelExit := swingRangeFibExit > 0 and useFibExit ? entryPrice + swingRangeFibExit * fibExtensionLevel : na
    exitReason := ""
    lastTradeBar := bar_index
    tradePnL := 0.0
    barsHeld := 0
    // Reset S/D flags on new long entry
    enteredSupplyZoneLong := false
    armedToExitLongOnGreen := false
    enteredDemandZoneShort := false
    armedToExitShortOnGreen := false
if isNewTradeSell
    inShort := true
    inLong := false
    entryPrice := close
    stopLossLevel := high + atrVal * atrMult
    swingHighPriceExit = highestForFibExit
    swingRangeFibExit = swingHighPriceExit - entryPrice
    fibTargetLevelExit := swingRangeFibExit > 0 and useFibExit ? entryPrice - swingRangeFibExit * fibExtensionLevel : na
    exitReason := ""
    lastTradeBar := bar_index
    tradePnL := 0.0
    barsHeld := 0
    // Reset S/D flags on new short entry
    enteredSupplyZoneLong := false
    armedToExitLongOnGreen := false
    enteredDemandZoneShort := false
    armedToExitShortOnGreen := false

// Update Trailing SL
if useAtrStop and inLong and not isNewTradeBuy
    stopLossLevel := math.max(nz(stopLossLevel, low - atrVal * atrMult), low - atrVal * atrMult)
if useAtrStop and inShort and not isNewTradeSell
    stopLossLevel := math.min(nz(stopLossLevel, high + atrVal * atrMult), high + atrVal * atrMult)

// Update PnL and Bars Held
if inLong or inShort
    barsHeld += 1
    if not na(entryPrice)
        tradePnL := inLong ? (close - entryPrice) / entryPrice * 100 : (entryPrice - close) / entryPrice * 100

// --- Exit Logic ---
exitedLongThisBar = false, exitedShortThisBar = false
atrStopFiredLong = false, atrStopFiredShort = false
fibTargetHitLong = false, fibTargetHitShort = false // Keep flags if needed elsewhere
exitReason_bar = ""

// ADDED: S/D Zone Exit Conditions
exitLongSupplyZoneGreen = useSupplyZoneExit and inLong and armedToExitLongOnGreen and close > open
exitShortDemandZoneGreen = useDemandZoneExit and inShort and armedToExitShortOnGreen and close > open

if wasInLong
    atrStopHitLong = useAtrStop and not na(stopLossLevel) and close < stopLossLevel
    fibHitLong = useFibExit and not na(fibTargetLevelExit) and high >= fibTargetLevelExit
    scoreDropExitLong = scaledScore < (5.0 - exitScoreDropThreshold)
    emaExitLong = useEmaExit and condEmaFastMedCrossSell
    bbExitLong = useBBReturnExit and condBbReturnMeanSell
    volExitLong = useVolFadeExit and condVolFadeLong
    rsiDivExitLong = condRsiBullDivExit

    if atrStopHitLong // Highest priority exit
        exitedLongThisBar := true
        exitReason_bar := "ATR SL"
        atrStopFiredLong := true
        atrStopHitLongCount += 1
    else if exitLongSupplyZoneGreen // ADDED: Supply Zone Exit
        exitedLongThisBar := true
        exitReason_bar := "Supply Zone Exit"
    else if fibHitLong
        exitedLongThisBar := true
        exitReason_bar := "Fib Tgt"
        fibTargetHitLong := true
    else if scoreDropExitLong
        exitedLongThisBar := true
        exitReason_bar := "Score Drop (" + str.tostring(scaledScore, "#.#") + ")"
    else if emaExitLong
        exitedLongThisBar := true
        exitReason_bar := "EMA Cross"
    else if bbExitLong
        exitedLongThisBar := true
        exitReason_bar := "BB Mid Exit"
    else if volExitLong
        exitedLongThisBar := true
        exitReason_bar := "Vol Fade"
    else if rsiDivExitLong
        exitedLongThisBar := true
        exitReason_bar := "RSI Div"

if wasInShort
    atrStopHitShort = useAtrStop and not na(stopLossLevel) and close > stopLossLevel
    fibHitShort = useFibExit and not na(fibTargetLevelExit) and low <= fibTargetLevelExit
    scoreDropExitShort = scaledScore > (5.0 + exitScoreDropThreshold)
    emaExitShort = useEmaExit and condEmaFastMedCrossBuy
    bbExitShort = useBBReturnExit and condBbReturnMeanBuy
    volExitShort = useVolFadeExit and condVolFadeShort
    rsiDivExitShort = condRsiBearDivExit

    if atrStopHitShort // Highest priority exit
        exitedShortThisBar := true
        exitReason_bar := "ATR SL"
        atrStopFiredShort := true
        atrStopHitShortCount += 1
    else if exitShortDemandZoneGreen // ADDED: Demand Zone Exit
        exitedShortThisBar := true
        exitReason_bar := "Demand Zone Exit"
    else if fibHitShort
        exitedShortThisBar := true
        exitReason_bar := "Fib Tgt"
        fibTargetHitShort := true
    else if scoreDropExitShort
        exitedShortThisBar := true
        exitReason_bar := "Score Drop (" + str.tostring(scaledScore, "#.#") + ")"
    else if emaExitShort
        exitedShortThisBar := true
        exitReason_bar := "EMA Cross"
    else if bbExitShort
        exitedShortThisBar := true
        exitReason_bar := "BB Mid Exit"
    else if volExitShort
        exitedShortThisBar := true
        exitReason_bar := "Vol Fade"
    else if rsiDivExitShort
        exitedShortThisBar := true
        exitReason_bar := "RSI Div"

// --- Reset State on Exit ---
if exitedLongThisBar or exitedShortThisBar // Reset on ANY exit
    // ADDED: Reset S/D flags
    enteredSupplyZoneLong := false
    armedToExitLongOnGreen := false
    enteredDemandZoneShort := false
    armedToExitShortOnGreen := false

    // Reset trade state
    if exitedLongThisBar
        inLong := false
    if exitedShortThisBar
        inShort := false
    stopLossLevel := na
    fibTargetLevelExit := na
    entryPrice := na
    exitReason := exitReason_bar
    tradePnL := na
    barsHeld := na


// === PLOTTING ===

// Candle Colors
candleColor = close > open ? color.new(color.green, 0) : color.new(color.red, 0)
plotcandle(open, high, low, close, title="Candles", color=candleColor, wickcolor=candleColor, bordercolor=candleColor)

// Indicators
plot(emaFast, "Fast EMA", color=color.new(color.blue, 0))
plot(emaMed, "Medium EMA", color=color.new(color.orange, 0))
plot(emaSlow, "Slow EMA", color=color.new(color.red, 0))
bbMidPlot = plot(showBB ? bbMiddle : na, "BB Middle", color=bbColor, linewidth=1)
bbUpperPlot = plot(showBB ? bbUpper : na, "BB Upper", color=bbColor, linewidth=1)
bbLowerPlot = plot(showBB ? bbLower : na, "BB Lower", color=bbColor, linewidth=1)
fill(bbUpperPlot, bbLowerPlot, color=color.new(bbColor, 90), title="BB Fill")

// SL and Target Lines
plot(inLong and useAtrStop and not na(stopLossLevel) ? stopLossLevel : na, "Long SL", color=color.new(color.maroon, 0), style=plot.style_linebr, linewidth=2)
plot(inShort and useAtrStop and not na(stopLossLevel) ? stopLossLevel : na, "Short SL", color=color.new(color.teal, 0), style=plot.style_linebr, linewidth=2)
plot(inLong and useFibExit and not na(fibTargetLevelExit) ? fibTargetLevelExit : na, "Long Tgt Line", color=color.new(color.fuchsia, 0), style=plot.style_linebr, linewidth=1)
plot(inShort and useFibExit and not na(fibTargetLevelExit) ? fibTargetLevelExit : na, "Short Tgt Line", color=color.new(color.fuchsia, 0), style=plot.style_linebr, linewidth=1)


// --- Plot Auto Fibonacci Retracement Lines ---
// Function to draw or update fib lines and labels
drawFibLineAndLabel(level, _line, _label, levelText, x1, x2, _color, _width, _extend) =>
    line newLine = na
    label newLabel = na
    if not na(level)
        newLine := line.new(x1, level, x2, level, color=_color, width=_width, extend=_extend)
        newLabel := label.new(x2 + 2, level, levelText, style=label.style_none, textcolor=_color, size=size.small)
        if not na(_line)
            line.set_xy1(_line, x1, level)
            line.set_xy2(_line, x2, level)
            line.set_color(_line, _color)
            line.set_width(_line, _width)
            line.set_extend(_line, _extend)
            line.delete(newLine)
            newLine := _line
        if not na(_label)
            label.set_xy(_label, x2 + 2, level)
            label.set_text(_label, levelText)
            label.set_textcolor(_label, _color)
            label.delete(newLabel)
            newLabel := _label
    [newLine, newLabel]

// Delete lines/labels function
deleteFibObjectsOnly() =>
    if not na(fibLine0) 
        line.delete(fibLine0)
    if not na(fibLine236)
        line.delete(fibLine236)
    if not na(fibLine382)
        line.delete(fibLine382)
    if not na(fibLine500)
        line.delete(fibLine500)
    if not na(fibLine618)
        line.delete(fibLine618)
    if not na(fibLine786)
        line.delete(fibLine786)
    if not na(fibLine100)
        line.delete(fibLine100)
    if not na(fibLabel0)
        label.delete(fibLabel0)
    if not na(fibLabel236)
        label.delete(fibLabel236)
    if not na(fibLabel382)
        label.delete(fibLabel382)
    if not na(fibLabel500)
        label.delete(fibLabel500)
    if not na(fibLabel618)
        label.delete(fibLabel618)
    if not na(fibLabel786)
        label.delete(fibLabel786)
    if not na(fibLabel100)
        label.delete(fibLabel100)
    ok = true

// Manage Fib drawing based on conditions
if showAutoFib
    if validFibRange
        lineStartX = math.min(nz(fibSwingLowBar, bar_index), nz(fibSwingHighBar, bar_index))
        lineEndX = bar_index
        [fibLine0, fibLabel0] = drawFibLineAndLabel(level_0, fibLine0, fibLabel0, "0.0", lineStartX, lineEndX, fibLineColor, fibLineWidth, extend_fib_lines)
        [fibLine236, fibLabel236] = drawFibLineAndLabel(level_236, fibLine236, fibLabel236, "0.236", lineStartX, lineEndX, fibLineColor, fibLineWidth, extend_fib_lines)
        [fibLine382, fibLabel382] = drawFibLineAndLabel(level_382, fibLine382, fibLabel382, "0.382", lineStartX, lineEndX, fibLineColor, fibLineWidth, extend_fib_lines)
        [fibLine500, fibLabel500] = drawFibLineAndLabel(level_500, fibLine500, fibLabel500, "0.5", lineStartX, lineEndX, fibLineColor, fibLineWidth, extend_fib_lines)
        [fibLine618, fibLabel618] = drawFibLineAndLabel(level_618, fibLine618, fibLabel618, "0.618", lineStartX, lineEndX, fibLineColor, fibLineWidth, extend_fib_lines)
        [fibLine786, fibLabel786] = drawFibLineAndLabel(level_786, fibLine786, fibLabel786, "0.786", lineStartX, lineEndX, fibLineColor, fibLineWidth, extend_fib_lines)
        [fibLine100, fibLabel100] = drawFibLineAndLabel(level_100, fibLine100, fibLabel100, "1.0", lineStartX, lineEndX, fibLineColor, fibLineWidth, extend_fib_lines)
    else
        deleteFibObjectsOnly()
        fibLine0 := na
        fibLabel0 := na
        fibLine236 := na
        fibLabel236 := na
        fibLine382 := na
        fibLabel382 := na
        fibLine500 := na
        fibLabel500 := na
        fibLine618 := na
        fibLabel618 := na
        fibLine786 := na
        fibLabel786 := na
        fibLine100 := na
        fibLabel100 := na
else
    deleteFibObjectsOnly()
    fibLine0 := na
    fibLabel0 := na
    fibLine236 := na
    fibLabel236 := na
    fibLine382 := na
    fibLabel382 := na
    fibLine500 := na
    fibLabel500 := na
    fibLine618 := na
    fibLabel618 := na
    fibLine786 := na
    fibLabel786 := na
    fibLine100 := na
    fibLabel100 := na


// --- Supply / Demand Zone Plotting ---
var box supplyBox = na, var box demandBox = na
var float plottedSupplyLevel = na, var int plottedSupplyBar = na
var float plottedDemandLevel = na, var int plottedDemandBar = na

if showSDZones
    // Use calculated levels from CALCULATION section
    if not na(supplyBottom) and not na(supplyTop)
        if na(supplyBox) or recentHighBar > nz(plottedSupplyBar, -1)
            if not na(supplyBox)
                box.delete(supplyBox)
            supplyBox := box.new(recentHighBar, supplyTop, bar_index + 1, supplyBottom, border_color=na, bgcolor=sdColorSupply, extend=extend.right)
            plottedSupplyLevel := supplyTop
            plottedSupplyBar := recentHighBar
        else if not na(supplyBox) and recentHighBar == plottedSupplyBar
            box.set_right(supplyBox, bar_index + 1)

    if not na(demandBottom) and not na(demandTop)
        if na(demandBox) or recentLowBar > nz(plottedDemandBar, -1)
            if not na(demandBox)
                box.delete(demandBox)
            demandBox := box.new(recentLowBar, demandTop, bar_index + 1, demandBottom, border_color=na, bgcolor=sdColorDemand, extend=extend.right)
            plottedDemandLevel := demandBottom
            plottedDemandBar := recentLowBar
        else if not na(demandBox) and recentLowBar == plottedDemandBar
            box.set_right(demandBox, bar_index + 1)

if not showSDZones
    if not na(supplyBox)
        box.delete(supplyBox)
        supplyBox := na
        plottedSupplyBar := na
    if not na(demandBox)
        box.delete(demandBox)
        demandBox := na
        plottedDemandBar := na


// Entry Signal Shapes
plotshape(series=isPotentialEmaBuy and not isNewTradeBuy, title="EMA Buy Shape", location=location.belowbar, color=color.new(color.green, 70), style=shape.triangleup, size=size.tiny)
plotshape(series=isPotentialFibBuy and not isNewTradeBuy, title="Fib Buy Shape", location=location.belowbar, color=color.new(color.green, 70), style=shape.diamond, size=size.tiny)
plotshape(series=isPotentialEmaBounceBuy and not isNewTradeBuy, title="EMA Bounce Buy Shape", location=location.belowbar, color=color.new(color.green, 70), style=shape.circle, size=size.tiny)
plotshape(series=isPotentialBbMidBounceBuy and not isNewTradeBuy, title="BB Mid Bounce Buy Shape", location=location.belowbar, color=color.new(color.green, 70), style=shape.square, size=size.tiny)
plotshape(series=isPotentialVolBuy and not isNewTradeBuy, title="Volume Buy Shape", location=location.belowbar, color=color.new(color.green, 70), style=shape.arrowup, size=size.tiny)
plotshape(series=isPotentialEmaSell and not isNewTradeSell, title="EMA Sell Shape", location=location.abovebar, color=color.new(color.red, 70), style=shape.triangledown, size=size.tiny)
plotshape(series=isPotentialFibSell and not isNewTradeSell, title="Fib Sell Shape", location=location.abovebar, color=color.new(color.red, 70), style=shape.diamond, size=size.tiny)
plotshape(series=isPotentialEmaBounceSell and not isNewTradeSell, title="EMA Bounce Sell Shape", location=location.abovebar, color=color.new(color.red, 70), style=shape.circle, size=size.tiny)
plotshape(series=isPotentialBbMidBounceSell and not isNewTradeSell, title="BB Mid Bounce Sell Shape", location=location.abovebar, color=color.new(color.red, 70), style=shape.square, size=size.tiny)
plotshape(series=isPotentialVolSell and not isNewTradeSell, title="Volume Sell Shape", location=location.abovebar, color=color.new(color.red, 70), style=shape.arrowdown, size=size.tiny)

// Exit Shapes
plotshape(exitedLongThisBar, "Exit Long Shape", location=location.abovebar, color=color.new(color.maroon, 50), style=shape.cross, size=size.small)
plotshape(exitedShortThisBar, "Exit Short Shape", location=location.belowbar, color=color.new(color.teal, 50), style=shape.cross, size=size.small)
plotshape(useSkipTradeFilter and skipTrade, title="Skipped Weak Signal", location=location.bottom, color=color.new(color.orange, 50), style=shape.xcross, size=size.tiny)

// Simple Entry Labels
if isNewTradeBuy
    label.new(bar_index, low - atrVal * 0.2, "Buy Signal", color=color.new(color.green, 20), textcolor=color.white, style=label.style_label_up, size=size.small, textalign=text.align_center)
if isNewTradeSell
    label.new(bar_index, high + atrVal * 0.2, "Sell Signal", color=color.new(color.red, 20), textcolor=color.white, style=label.style_label_down, size=size.small, textalign=text.align_center)

// Take Profit Labels - MODIFIED: Show if exit occurred and it wasn't ATR Stop
if exitedLongThisBar and not atrStopFiredLong
    label.new(bar_index, high + atrVal * 0.1, "Take Profit",
              color=color.new(color.gray, 40), textcolor=color.white,
              style=label.style_label_down, size=size.small, textalign=text.align_center)

if exitedShortThisBar and not atrStopFiredShort
    label.new(bar_index, low - atrVal * 0.1, "Take Profit",
              color=color.new(color.gray, 40), textcolor=color.white,
              style=label.style_label_up, size=size.small, textalign=text.align_center)

// Background color when in trade
bgcolor(inLong ? color.new(color.green, 85) : inShort ? color.new(color.red, 85) : na)

// Data Window Plotting
plot(scaledScore, "Confidence Score", color=color.new(color.gray,0), display=display.data_window)
plot(inLong or inShort ? tradePnL : na, "Live Trade PnL (%)", color=tradePnL > 0 ? color.green : color.red, display=display.data_window)
plot(inLong or inShort ? barsHeld : na, "Bars Held", color=color.new(color.orange, 0), display=display.data_window)

// === ALERTS ===
alertcondition(isNewTradeBuy, title="Enter Buy", message="{{ticker}}: ENTER BUY state. Score: {{plot('Confidence Score')}}. Price: {{close}}")
alertcondition(isNewTradeSell, title="Enter Sell", message="{{ticker}}: ENTER SELL state. Score: {{plot('Confidence Score')}}. Price: {{close}}")
// MODIFIED: Added reason back to alert message using the 'exitReason' variable (which holds the text from exitReason_bar)
alertcondition(exitedLongThisBar and not atrStopFiredLong, title="Exit Long Signal (Non-SL)", message="{{ticker}}: EXIT LONG signal (" + exitReason + "). Price: {{close}}")
alertcondition(exitedShortThisBar and not atrStopFiredShort, title="Exit Short Signal (Non-SL)", message="{{ticker}}: EXIT SHORT signal (" + exitReason + "). Price: {{close}}")
alertcondition(atrStopFiredLong, title="ATR Stop Long", message="{{ticker}}: ATR STOP LONG hit. Price: {{close}}")
alertcondition(atrStopFiredShort, title="ATR Stop Short", message="{{ticker}}: ATR STOP SHORT hit. Price: {{close}}")
alertcondition(plotSignalBuy and not plotSignalBuy[1], title="Potential Buy", message="{{ticker}}: Potential Buy Signal. Score: {{plot('Confidence Score')}}. Price: {{close}}")
alertcondition(plotSignalSell and not plotSignalSell[1], title="Potential Sell", message="{{ticker}}: Potential Sell Signal. Score: {{plot('Confidence Score')}}. Price: {{close}}")

