# Crypto Pattern Recognition - Backend Data Analysis

## Overview
A Python-based system for detecting chart patterns in cryptocurrency data using historical price, volume, and order book information. No frontend - pure data analysis and pattern detection.

## CCXT Public API Data Available
### OHCLV Open, High, Low, Close, Volume candlestick data
#### Timeframe
```
| Key     | Meaning    |
| ------- | ---------- |
| `'1m'`  | 1 minute   |
| `'3m'`  | 3 minutes  |
| `'5m'`  | 5 minutes  |
| `'15m'` | 15 minutes |
| `'30m'` | 30 minutes |
| `'1h'`  | 1 hour     |
| `'2h'`  | 2 hours    |
| `'4h'`  | 4 hours    |
| `'6h'`  | 6 hours    |
| `'12h'` | 12 hours   |
| `'1d'`  | 1 day      |
| `'3d'`  | 3 days     |
| `'1w'`  | 1 week     |
| `'1M'`  | 1 month    |

```
#### Candlestick Limit
```
4 Hour timeframe
2025-07-24 04:00:00,1753329600000,3.192,3.1965,2.9555,3.075,114510469.6
2025-07-24 08:00:00,1753344000000,3.075,3.1788,3.055,3.1685,64732542.0

1 Days timeframe
2025-06-30,1751241600000,2.2062,2.3271,2.165,2.2362,151525906.2
2025-07-01,1751328000000,2.2362,2.2537,2.1475,2.172,123558690.1

3 Days timeframe
2025-05-03,1746230400000,2.2093,2.22,2.1067,2.1306,236650634.9
2025-05-06,1746489600000,2.1306,2.3297,2.0777,2.3272,422235051.4
```

### Order Book: Bid/ask prices and volumes at different levels
### Ticker Data: Current market prices, 24h volume, price changes
### Trades: Recent trade history with price, volume, timestamp
### Markets: Available trading pairs and exchange information


## Phase 1: EMA 9/21
1. create files /trend/ema_9_21.py
2. Use only Close price from OHLCV data (via `collector.py → fetch_ohlcv_data`)
3. Require at 50–100 closes for EMA calculation.
4. bullish trend start if EMA 9 crosses above EMA 21, Close is above both EMAs, with Volume spike
5. bearish trend start if EMA 9 crosses below EMA 21, Close is below both EMAs, with Volume spike
6. Confirmation = crossover + follow-through over next candles
### Input in terminal
> ema_9_21 s=XRP/USDT t=1d/1h/4h... l=30
- t = timeframe
- s = symbol
- l = limit to 30 candles
- data fetch is defaulting to current days
### Output example in terminal
[2025-07-29 09:00:00] CLOSE: 103.25 | EMA9: 102.80 | EMA21: 102.50 | ⬆️ BUY | Trend: BULLISH | ✔️ Confirmed
[2025-07-29 09:05:00] CLOSE: 103.10 | EMA9: 102.85 | EMA21: 102.65 | ➖ NONE | Trend: NEUTRAL | ⏳ Waiting
### CLI
Make sure to add new handler to the cli. /cli/ema_9_21_handler.py


## Phase 2
create interactive terminal cli, example:
```
> ema_9_21 s=XRP/USDT t=1d l=30
[2025-07-29 09:00:00] CLOSE: 103.25 | EMA9: 102.80 | EMA21: 102.50 | ⬆️ BUY | Trend: BULLISH | ✔️ Confirmed
[2025-07-29 09:05:00] CLOSE: 103.10 | EMA9: 102.85 | EMA21: 102.65 | ➖ NONE | Trend: NEUTRAL | ⏳ Waiting
> ema_9_21 s=XRP/USDT t=4h l=30
[2025-07-29 09:00:00] CLOSE: 103.25 | EMA9: 102.80 | EMA21: 102.50 | ⬆️ BUY | Trend: BULLISH | ✔️ Confirmed
[2025-07-29 13:00:00] CLOSE: 103.10 | EMA9: 102.85 | EMA21: 102.65 | ➖ NONE | Trend: NEUTRAL | ⏳ Waiting

```


## Phase 3: RSI 14
- Create file `/trend/rsi_14.py`
- Use only **Close** price from OHLCV data (via `collector.py → fetch_ohlcv_data`)
- Require at least **20–30 closes** for RSI(14) calculation
- **Overbought zone:** RSI > 70 → signal potential reversal or trend weakness
- **Oversold zone:** RSI < 30 → signal potential bounce or trend exhaustion
- Trend continuation is **confirmed** if RSI stays in a strong region (e.g. > 50 in bull or < 50 in bear)
- Use RSI in combination with EMA 9/21 crossover to validate momentum and filter noise
- **BUY bias:** When RSI < 30 and rising (bullish reversal)  
- **SELL bias:** When RSI > 70 and dropping (bearish reversal)  
- **NEUTRAL:** When RSI between 40–60 or flat (sideways market)  
### Input in terminal
> rsi_14 s=XRP/USDT t=1d l=30
### Output example in terminal
[2025-07-29 09:00:00] CLOSE: 103.25 | RSI(14): 72.50 | ⚠️ OVERBOUGHT | Signal: SELL | ⏳ Waiting
[2025-07-29 09:05:00] CLOSE: 101.80 | RSI(14): 68.90 | ✅ Confirmed Signal: SELL
[2025-07-29 10:00:00] CLOSE: 98.75  | RSI(14): 29.60 | 🔽 OVERSOLD | Signal: BUY | ⏳ Waiting
### CLI
Make sure to add new handler to the cli. /cli/rsi_14_handler.py


## Phase 4: MACD
- Create file `/trend/macd.py`
- Use only **Close** price from OHLCV data (via `collector.py → fetch_ohlcv_data`)
- Require at least **50–100 closes** to generate smooth MACD values
- **MACD Line = EMA(12) − EMA(26)**
- **Signal Line = EMA(9)** of MACD Line
- **Histogram = MACD − Signal**
- **BUY signal:** When MACD Line crosses above Signal Line  
- **SELL signal:** When MACD Line crosses below Signal Line  
- **STRONG trend:** When histogram grows in direction of crossover  
- Use MACD to confirm EMA 9/21 crossovers and RSI direction  
- Effective in scalping (15m+), swing (4h+), or position (1d+)

### Input in terminal
> macd s=XRP/USDT t=4h l=100
### Output example in terminal
[2025-07-29 09:00:00] MACD: 0.034 | SIGNAL: 0.029 | HIST: +0.005 | ⬆️ Crossover | Signal: BUY | 🔄 Confirming Uptrend  
[2025-07-29 13:00:00] MACD: -0.018 | SIGNAL: -0.016 | HIST: -0.002 | ⬇️ Crossover | Signal: SELL | 🧨 Weak Momentum
### CLI
Make sure to add new handler to the cli. /cli/macd_handler.py


## Phase 5: All trend analysis
- command "all_trend" analysis every method in the trend analysis
- call all trend analysis one by one and show the output on the terminal
### Input in terminal
> all_trend s=XRP/USDT t=4h l=100
### Output example in terminal
[2025-07-29 09:00:00] CLOSE: 103.25 | EMA9: 102.80 | EMA21: 102.50 | ⬆️ BUY | Trend: BULLISH | ✔️ Confirmed
[2025-07-29 09:00:00] MACD: 0.034 | SIGNAL: 0.029 | HIST: +0.005 | ⬆️ Crossover | Signal: BUY | 🔄 Confirming Uptrend  
[2025-07-29 09:00:00] CLOSE: 103.25 | RSI(14): 72.50 | ⚠️ OVERBOUGHT | Signal: SELL | ⏳ Waiting
[2025-07-29 09:05:00] CLOSE: 101.80 | RSI(14): 68.90 | ✅ Confirmed Signal: SELL
[2025-07-29 09:05:00] CLOSE: 103.10 | EMA9: 102.85 | EMA21: 102.65 | ➖ NONE | Trend: NEUTRAL | ⏳ Waiting
[2025-07-29 10:00:00] CLOSE: 98.75  | RSI(14): 29.60 | 🔽 OVERSOLD | Signal: BUY | ⏳ Waiting
[2025-07-29 13:00:00] MACD: -0.018 | SIGNAL: -0.016 | HIST: -0.002 | ⬇️ Crossover | Signal: SELL | 🧨 Weak Momentum
### CLI
Make sure to add new handler to the cli. /cli/all_trend_handler.py


## Phase 6: OBV (On-Balance Volume)
- Create file `/trend/obv.py`
- Use **Close** price and **Volume** from OHLCV data (via `collector.py → fetch_ohlcv_data`)
- Require at least **50–100 closes and volumes** for stable OBV calculation
- **OBV Calculation:**  
  - If today's Close > yesterday's Close → OBV = previous OBV + today's Volume  
  - If today's Close < yesterday's Close → OBV = previous OBV − today's Volume  
  - If today's Close = yesterday's Close → OBV = previous OBV (no change)  
- **BUY signal:** OBV rising, confirming price uptrend (especially when price breaks resistance)  
- **SELL signal:** OBV falling, confirming price downtrend or divergence with price  
- Use OBV to confirm momentum and validate breakouts or reversals indicated by EMA, MACD, or RSI  
- Effective in all timeframes, but especially useful in swing (4h+) and position (1d+) trading to avoid fake breakouts  
### Input in terminal
> obv s=BTC/USDT t=1d l=100
### Output example in terminal
[2025-07-29 00:00:00] OBV: 1,234,567,890 | Price: 30,500 | Signal: BUY | 📈 Confirmed Uptrend  
[2025-07-30 00:00:00] OBV: 1,220,000,000 | Price: 29,800 | Signal: SELL | ⚠️ Divergence Detected
### CLI
Make sure to add new handler to the cli. `/cli/obv_handler.py`
### All trend
add obv analyzer to all_trend.py too


## Phase 7: ATR + ADX (Average True Range & Average Directional Index)
- Create file `/trend/atr_adx.py`
- Use **High, Low, Close** price from OHLCV data (via `collector.py → fetch_ohlcv_data`)
- Require at least **14–30 candles** for ATR and ADX smoothing
- **ATR Calculation:**  
  - Measures average true range (volatility) over specified period (default 14)  
  - True Range = max(High−Low, abs(High−Previous Close), abs(Low−Previous Close))  
- **ADX Calculation:**  
  - Measures trend strength from directional movement indicators (+DI and -DI)  
  - ADX = smoothed average of absolute difference between +DI and -DI divided by sum of +DI and -DI  
- **BUY signal:** ADX > 25 and +DI crosses above -DI, ATR confirms volatility suitable for entry  
- **SELL signal:** ADX > 25 and -DI crosses above +DI, ATR indicates elevated volatility for exits or stops  
- Use ATR for dynamic stop loss placement and position sizing based on current volatility  
- Combine ADX with EMA/MACD/RSI/OBV to confirm strong trending conditions and avoid sideways noise  
- Highly effective for swing (4h+) and position (1d+) trading to improve risk management and signal accuracy  
### Input in terminal
> atr_adx s=ETH/USDT t=4h l=14
### Output example in terminal
[2025-07-29 12:00:00] ATR: 250 | ADX: 32 | +DI: 28 | -DI: 15 | Signal: BUY | 📊 Strong Trend Confirmed  
[2025-07-30 00:00:00] ATR: 200 | ADX: 22 | +DI: 18 | -DI: 22 | Signal: HOLD | ⚠️ Weak Trend - Avoid New Positions
### CLI
Make sure to add new handler to the cli. `/cli/atr_adx_handler.py`


## Phase 8: Bollinger Bands (BB)
- Create file `/trend/bollinger_bands.py`
- Use **Close** price from OHLCV data (via `collector.py → fetch_ohlcv_data`)
- Require at least **50–100 closes** for reliable SMA and standard deviation
- **Calculation:**  
  - **Middle Band (MB) = SMA(n)** of Close (default n = 20)  
  - **Standard Deviation (SD) = stdev** of Close over n periods  
  - **Upper Band (UB) = MB + (k × SD)** (default k = 2)  
  - **Lower Band (LB) = MB − (k × SD)**  
- **BUY signal:** When Close crosses above Lower Band (oversold bounce)  
- **SELL signal:** When Close crosses below Upper Band (overbought reversal)  
- **STRONG trend confirmation:** When price rides the Upper Band (uptrend) or Lower Band (downtrend)  
- Use BB to spot volatility squeezes (narrow bands) and breakout entries/exits  
- Pair with RSI/MACD for better entry timing
### Input in terminal
> bb s=ETH/USDT t=4h l=100
### Output example in terminal
[2025-07-29 12:00:00] Price: 1,850.00 | MB: 1,820.50 | UB: 1,880.75 | LB: 1,760.25 | Signal: BUY | 🔄 Squeeze Breakout  
[2025-07-29 16:00:00] Price: 1,890.00 | MB: 1,825.00 | UB: 1,885.00 | LB: 1,765.00 | Signal: SELL | ⚠️ Overbought
### CLI
Make sure to add new handler to the cli: `/cli/bollinger_bands_handler.py`


## Phase 9: Divergence Detection Engine
- Create file `/trend/divergence.py`
- Use **Close** price from OHLCV and indicator values from existing analysis modules: `rsi_14.py`, `macd.py`, and `obv.py`
- Require at least **30–50 candles** to establish valid swing highs/lows
- **Bullish Divergence:**  
  - Price makes a **lower low**, but indicator (RSI, MACD, or OBV) makes a **higher low**  
  - Signal: BUY — Potential reversal from downtrend  
- **Bearish Divergence:**  
  - Price makes a **higher high**, but indicator makes a **lower high**  
  - Signal: SELL — Potential reversal from uptrend  
- **Supported Indicators:**  
  - RSI(14) → momentum divergence  
  - MACD Line → trend momentum divergence  
  - OBV → volume divergence confirmation  
- **Classification:**  
  - Weak, Moderate, Strong (based on distance between swings and confluence across indicators)  
- Use divergence detection to **pre-warn trend shifts** and filter false breakouts  
- Works best in swing (4h) and position (1d) timeframes
### Input in terminal
> divergence s=SOL/USDT t=4h l=100
### Output example in terminal
[2025-07-30 00:00:00] RSI Divergence: Bullish | Price LL: 27.50 → 26.80 | RSI HL: 30.2 → 32.6 | Signal: BUY | 🧠 Early Reversal  
[2025-07-30 04:00:00] MACD Divergence: Bearish | Price HH: 28.90 → 29.25 | MACD LH: 0.034 → 0.022 | Signal: SELL | ⚠️ Weak Momentum  
[2025-07-30 08:00:00] OBV Divergence: Bullish | OBV Rising vs Price Drop | Signal: BUY | 🔋 Accumulation Phase
### CLI
Make sure to add new handler to the cli: `/cli/divergence_handler.py`
### All trend
Add divergence check to `/trend/all_trend.py` for full confluence analysis


## Phase 10: Supertrend Indicator
- Create file `/trend/supertrend.py`
- Use **High, Low, Close** prices from OHLCV data (via `collector.py → fetch_ohlcv_data`)
- Require at least **20–50 candles** for reliable trend generation
- **Calculation:**  
  - **ATR(n)** → use existing ATR logic (default n = 10 or 14)  
  - **Basic Upper Band = (High + Low) / 2 + Multiplier × ATR**  
  - **Basic Lower Band = (High + Low) / 2 − Multiplier × ATR**  
  - **Final Upper/Lower Band = Dynamic, based on price direction**  
  - **Supertrend:**  
    - If Close > Final Upper Band → Trend = Bullish  
    - If Close < Final Lower Band → Trend = Bearish  
    - Else → Continue previous trend  
- **BUY signal:** Price crosses above Supertrend band → trend shift to bullish  
- **SELL signal:** Price crosses below Supertrend band → trend shift to bearish  
- **Utility:**  
  - Provides **clear directional bias**  
  - Acts as **dynamic stop-loss** and **re-entry guide**  
  - Works extremely well in combination with EMA crossovers and ADX for confirmation

### Input in terminal
> supertrend s=BTC/USDT t=1h l=100
### Output example in terminal
[2025-07-30 10:00:00] Price: 29,800 | Supertrend: 29,600 | Signal: BUY | ✅ Trend Reversal Confirmed  
[2025-07-30 11:00:00] Price: 29,350 | Supertrend: 29,550 | Signal: SELL | 🔻 Bearish Trend Shift
### CLI
Make sure to add new handler to the CLI: `/cli/supertrend_handler.py`
### All trend
Add Supertrend output to `/trend/all_trend.py` for unified signal reporting


## Phase 11: VWAP (Volume Weighted Average Price)
- Create file `/trend/vwap.py`
- Use **Close** price, **Volume**, and **Typical Price** (H+L+C)/3 from OHLCV data (via `collector.py → fetch_ohlcv_data`)
- Require intraday or daily candles; more effective in **1m to 4h** timeframes  
- **VWAP Calculation (standard):**  
  - **Cumulative TP×Volume / Cumulative Volume**, where **TP = (High + Low + Close) / 3**  
  - Reset at the start of each trading day or anchor manually  
- **Signal logic:**  
  - If **Close > VWAP** → Bullish Bias  
  - If **Close < VWAP** → Bearish Bias  
  - Use crossover as a trigger or confirmation for trend shifts  
- **BUY signal:** Price crosses above VWAP with volume confirmation  
- **SELL signal:** Price crosses below VWAP with downward momentum  
- **Bonus:**  
  - Implement **Anchored VWAP** support by setting a custom index/time as the anchor  
  - Useful for analyzing breakout points or post-event reactions
### Input in terminal
> vwap s=ETH/USDT t=15m l=200  
> vwap s=BTC/USDT t=1h l=100 anchor="2025-07-29T04:00:00"
### Output example in terminal
[2025-07-30 10:15:00] Price: 2,950 | VWAP: 2,940 | Signal: BUY | 🟢 Price Above VWAP  
[2025-07-30 10:45:00] Price: 2,915 | VWAP: 2,925 | Signal: SELL | 🔻 Bearish Bias
### CLI
Make sure to add new handler to the CLI: `/cli/vwap_handler.py`
### All trend
Add VWAP output to `/trend/all_trend.py` to support intraday bias analysis and breakout confirmation


## Phase 12: Double Bottom Pattern
- Create file `/patterns/double_bottom.py`
- Use **Low**, **Close**, and **Volume** from OHLCV data (via `collector.py → fetch_ohlcv_data`)
- Require at least **50–100 candles** to identify two distinct swing lows
- **Detection Logic:**  
  - Identify two swing lows at approximately the same price level, separated by a peak (intervening high)  
  - **Support Level:** Price lows within a tolerance band (e.g. ±1–2% of each other)  
  - **Neckline:** The intervening peak’s price level  
  - Confirm pattern when:  
    1. Price falls to first low → rebounds to peak → retraces to second low (≈ first low)  
    2. Volume on second low is lower or equal to first low (showing weakening sell pressure)  
    3. Breakout above neckline on increased volume  
- **BUY signal:** Close breaks and holds above the neckline after the second low, with volume spike  
- **Invalidation:** Price drops below the lower of the two lows after breakout attempt  
- **Utility:**  
  - Forecasts trend reversal from downtrend to uptrend  
  - Provides clear entry level (neckline) and stop-loss (below second low)  
  - Pair with RSI rising from oversold (<30) for extra confirmation
### Input in terminal
> double_bottom s=ADA/USDT t=4h l=200
### Output example in terminal
[2025-07-30 08:00:00] Low1: 0.420 | Peak: 0.460 | Low2: 0.422 | Neckline: 0.460 | Signal: NONE | ⏳ Pattern Forming  
[2025-07-30 12:00:00] Price: 0.465 | Volume: +25% | Signal: BUY | 🚀 Breakout Confirmed  
### CLI
Make sure to add new handler to the CLI: `/cli/pattern/double_bottom_handler.py`
### All patterns
Add Double Bottom detection to `/patterns/all_patterns.py`


## Phase 13: Double Top Pattern
- Create file `/patterns/double_top.py`
- Use **Close** price from OHLCV data (via `collector.py → fetch_ohlcv_data`)
- Require at least **50–100 closes** to detect valid swing highs
- **Double Top Definition:**  
  - Two prominent swing highs at **similar price levels**  
  - A **swing low (valley)** between them  
  - Confirmation when price **breaks below** the valley low
- **Signal Conditions:**  
  - First peak formed  
  - Price retraces (valley)  
  - Second peak forms **within ~1–3% of the first**  
  - Price then **breaks below valley** → confirms pattern  
- **SELL signal:** Pattern confirmed → potential bearish reversal  
- Use pattern with RSI divergence, MACD crossover, or OBV drop for confluence  
- Best suited for swing (4h) and position (1d) timeframes
### Input in terminal
> double_top s=ETH/USDT t=4h l=100
### Output example in terminal
[2025-07-30 08:00:00] Double Top Detected | Peaks: 3,200 & 3,180 | Valley: 3,050 | Confirmed: Yes  
Signal: SELL | ⚠️ Bearish Reversal | 📉 Price breaking support zone
### CLI
Make sure to add new handler to the CLI: `/cli/double_top_handler.py`
### All patterns
Add `double_top` detection to `/patterns/all_patterns.py`
