//@version=6
indicator("EMA Volume RSI ATR Strategy ", overlay=true)

// Define EMA lengths as user inputs
ema9Length = input.int(9, title="EMA 9 Length", minval=1)
ema14Length = input.int(14, title="EMA 14 Length", minval=1)
ema21Length = input.int(21, title="EMA 21 Length", minval=1)

// Define EMAs
ema9 = ta.ema(close, ema9Length)
ema14 = ta.ema(close, ema14Length)
ema21 = ta.ema(close, ema21Length)

// RSI Condition
rsiLength = input.int(14, title="RSI Length", minval=1)
rsi = ta.rsi(close, rsiLength)

// ATR Stop Loss
atrLength = input.int(14, title="ATR Length", minval=1)
atrMultiplier = input.float(2.0, title="ATR Multiplier", minval=0.1)
atr = ta.atr(atrLength)
stopLossLong = low - atr * atrMultiplier
stopLossShort = high + atr * atrMultiplier

// Buy Condition: 9 EMA crosses above 21 EMA (Volume removed)
buySignal = ta.crossover(ema9, ema21)

// Sell Condition: 9 EMA crosses below 21 EMA (Volume removed)
sellSignal = ta.crossunder(ema9, ema21)

// Entry Variables
var float buyPrice = na
var float sellPrice = na

// Buy/Sell Logic and Stop Loss Tracking
if (buySignal)
    buyPrice := close
if (sellSignal)
    sellPrice := close

// Volume Based Exit (Optional: Comment out if not needed)
volAvgLength = input.int(50, title="Volume SMA Length", minval=1)
vol_avg = ta.sma(volume, volAvgLength)
volumeExitLong = close < ema9 and volume < vol_avg and not na(buyPrice[1])
volumeExitShort = close > ema9 and volume < vol_avg and not na(sellPrice[1])

// RSI based Exit
rsiExitLong = rsi < 50 and not na(buyPrice[1])
rsiExitShort = rsi > 50 and not na(sellPrice[1])

// ATR Stop Loss Exits
atrStopLongHit = close < stopLossLong and not na(buyPrice[1])
atrStopShortHit = close > stopLossShort and not na(sellPrice[1])

// Combined Exit Signals
exitLong = volumeExitLong or rsiExitLong or atrStopLongHit
exitShort = volumeExitShort or rsiExitShort or atrStopShortHit

// Reset Entry Variables on Exit
if (exitLong)
    buyPrice := na
if (exitShort)
    sellPrice := na

// Plot EMAs
plot(ema9, color=color.blue, title="EMA 9")
plot(ema14, color=color.orange, title="EMA 14")
plot(ema21, color=color.red, title="EMA 21")

// Plot Buy Arrows (Bright Green ▲)
plotshape(series=buySignal, location=location.belowbar, color=color.lime, style=shape.triangleup, title="BUY", size=size.small)

// Plot Sell Arrows (Bright Red ▼)
plotshape(series=sellSignal, location=location.abovebar, color=color.red, style=shape.triangledown, title="SELL", size=size.small)

// Plot Exit Long (Blue ✖ above bars)
plotshape(series=exitLong, location=location.abovebar, color=color.blue, style=shape.cross, title="Exit Long", size=size.tiny)

// Plot Exit Short (Orange ✖ below bars)
plotshape(series=exitShort, location=location.belowbar, color=color.orange, style=shape.cross, title="Exit Short", size=size.tiny)

// Plot Stop Loss Lines
plot(not na(buyPrice) ? stopLossLong : na, color=color.red, style=plot.style_linebr, title="Long Stop Loss")
plot(not na(sellPrice) ? stopLossShort : na, color=color.green, style=plot.style_linebr, title="Short Stop Loss")



//@version=6
indicator("EMA Volume Strategy", overlay=true)

// Define EMAs
ema9 = ta.ema(close, 9)
ema14 = ta.ema(close, 14)
ema21 = ta.ema(close, 21)

// Volume Condition (Check if volume is above 50-period average)
vol_avg = ta.sma(volume, 50)
high_vol = volume > vol_avg

// Buy Condition: 9 EMA crosses above 21 EMA & high volume
buySignal = ta.crossover(ema9, ema21) and high_vol

// Sell Condition: 9 EMA crosses below 21 EMA & high volume
sellSignal = ta.crossunder(ema9, ema21) and high_vol

// Plot EMAs
plot(ema9, color=color.blue, title="EMA 9")
plot(ema14, color=color.orange, title="EMA 14")
plot(ema21, color=color.red, title="EMA 21")

// Plot Buy Arrows
plotshape(series=buySignal, location=location.belowbar, color=color.green, style=shape.labelup, title="BUY", text="BUY")

// Plot Sell Arrows
plotshape(series=sellSignal, location=location.abovebar, color=color.red, style=shape.labeldown, title="SELL", text="SELL")


//@version=6
indicator("EMA Volume RSI ATR Strategy ", overlay=true)

// Define EMA lengths as user inputs
ema9Length = input.int(9, title="EMA 9 Length", minval=1)
ema14Length = input.int(14, title="EMA 14 Length", minval=1)
ema21Length = input.int(21, title="EMA 21 Length", minval=1)

// Define EMAs
ema9 = ta.ema(close, ema9Length)
ema14 = ta.ema(close, ema14Length)
ema21 = ta.ema(close, ema21Length)

// RSI Condition
rsiLength = input.int(14, title="RSI Length", minval=1)
rsi = ta.rsi(close, rsiLength)

// ATR Stop Loss
atrLength = input.int(14, title="ATR Length", minval=1)
atrMultiplier = input.float(2.0, title="ATR Multiplier", minval=0.1)
atr = ta.atr(atrLength)
stopLossLong = low - atr * atrMultiplier
stopLossShort = high + atr * atrMultiplier

// Buy Condition: 9 EMA crosses above 21 EMA (Volume removed)
buySignal = ta.crossover(ema9, ema21)

// Sell Condition: 9 EMA crosses below 21 EMA (Volume removed)
sellSignal = ta.crossunder(ema9, ema21)

// Entry Variables
var float buyPrice = na
var float sellPrice = na

// Buy/Sell Logic and Stop Loss Tracking
if (buySignal)
    buyPrice := close
if (sellSignal)
    sellPrice := close

// Volume Based Exit (Optional: Comment out if not needed)
volAvgLength = input.int(50, title="Volume SMA Length", minval=1)
vol_avg = ta.sma(volume, volAvgLength)
volumeExitLong = close < ema9 and volume < vol_avg and not na(buyPrice[1])
volumeExitShort = close > ema9 and volume < vol_avg and not na(sellPrice[1])

// RSI based Exit
rsiExitLong = rsi < 50 and not na(buyPrice[1])
rsiExitShort = rsi > 50 and not na(sellPrice[1])

// ATR Stop Loss Exits
atrStopLongHit = close < stopLossLong and not na(buyPrice[1])
atrStopShortHit = close > stopLossShort and not na(sellPrice[1])

// Combined Exit Signals
exitLong = volumeExitLong or rsiExitLong or atrStopLongHit
exitShort = volumeExitShort or rsiExitShort or atrStopShortHit

// Reset Entry Variables on Exit
if (exitLong)
    buyPrice := na
if (exitShort)
    sellPrice := na

// Plot EMAs
plot(ema9, color=color.blue, title="EMA 9")
plot(ema14, color=color.orange, title="EMA 14")
plot(ema21, color=color.red, title="EMA 21")

// Plot Buy Arrows (Bright Green ▲)
plotshape(series=buySignal, location=location.belowbar, color=color.lime, style=shape.triangleup, title="BUY", size=size.small)

// Plot Sell Arrows (Bright Red ▼)
plotshape(series=sellSignal, location=location.abovebar, color=color.red, style=shape.triangledown, title="SELL", size=size.small)

// Plot Exit Long (Blue ✖ above bars)
plotshape(series=exitLong, location=location.abovebar, color=color.blue, style=shape.cross, title="Exit Long", size=size.tiny)

// Plot Exit Short (Orange ✖ below bars)
plotshape(series=exitShort, location=location.belowbar, color=color.orange, style=shape.cross, title="Exit Short", size=size.tiny)

// Plot Stop Loss Lines
plot(not na(buyPrice) ? stopLossLong : na, color=color.red, style=plot.style_linebr, title="Long Stop Loss")
plot(not na(sellPrice) ? stopLossShort : na, color=color.green, style=plot.style_linebr, title="Short Stop Loss")

//@version=6
indicator("Multi-EMA + Volume RSI Strategy", overlay=true)

// Inputs
ema9 = ta.ema(close, 9)
ema12 = ta.ema(close, 12)
ema14 = ta.ema(close, 14)

// Volume RSI (14-period)
volume_rsi = ta.rsi(volume, 14)
rsi_bullish = volume_rsi > 50  // Custom threshold
rsi_bearish = volume_rsi < 50

// Conditions
ema9_above_12 = ta.crossover(ema9, ema12)
ema9_above_14 = ta.crossover(ema9, ema14)
bullish_signal = ema9_above_12 and ema9_above_14 and rsi_bullish

ema9_below_12 = ta.crossunder(ema9, ema12)
ema9_below_14 = ta.crossunder(ema9, ema14)
bearish_signal = ema9_below_12 and ema9_below_14 and rsi_bearish

// Plot EMAs
plot(ema9, "EMA 9", color=color.new(color.blue, 0))
plot(ema12, "EMA 12", color=color.new(color.orange, 0))
plot(ema14, "EMA 14", color=color.new(color.purple, 0))

// Plot Signals
plotshape(bullish_signal, style=shape.triangleup, location=location.belowbar, color=color.green, size=size.small, text="BUY")
plotshape(bearish_signal, style=shape.triangledown, location=location.abovebar, color=color.red, size=size.small, text="SELL")

// Volume RSI Subplot
plot(volume_rsi, "Volume RSI", color=color.new(color.fuchsia, 0), style=plot.style_linebr)
hline(50, "RSI Midline", color=color.gray)

//@version=6
indicator("SS", overlay=true)

// Inputs
ema9 = ta.ema(close, 9)
ema12 = ta.ema(close, 12)
ema14 = ta.ema(close, 14)

// Volume RSI (14-period)
volume_rsi = ta.rsi(volume, 14)
rsi_bullish = volume_rsi > 50  // Custom threshold
rsi_bearish = volume_rsi < 50

// New RSI Condition for Long
rsi_low_and_rising = ta.crossover(volume_rsi, 20) 

// Conditions
ema9_above_12 = ta.crossover(ema9, ema12)
ema9_above_14 = ta.crossover(ema9, ema14)
bullish_signal = ema9_above_12 and ema9_above_14 and rsi_bullish

ema9_below_12 = ta.crossunder(ema9, ema12)
ema9_below_14 = ta.crossunder(ema9, ema14)
bearish_signal = ema9_below_12 and ema9_below_14 and rsi_bearish

// Plot EMAs
plot(ema9, "EMA 9", color=color.new(color.blue, 0))
plot(ema12, "EMA 12", color=color.new(color.orange, 0))
plot(ema14, "EMA 14", color=color.new(color.purple, 0))

// Plot Signals
plotshape(bullish_signal, style=shape.triangleup, location=location.belowbar, color=color.green, size=size.small, text="BUY")
plotshape(bearish_signal, style=shape.triangledown, location=location.abovebar, color=color.red, size=size.small, text="SELL")
plotshape(rsi_low_and_rising, style=shape.triangleup, location=location.belowbar, color=color.orange, size=size.small, text="RSI LONG")

// Volume RSI Subplot
plot(volume_rsi, "Volume RSI", color=color.new(color.fuchsia, 0), style=plot.style_linebr)
hline(50, "RSI Midline", color=color.gray)