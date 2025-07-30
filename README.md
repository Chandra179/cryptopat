# Crypto Pattern Recognition - Backend Data Analysis

## Overview
A Python-based system for detecting chart patterns in cryptocurrency data using historical price, volume, and order book information. No frontend - pure data analysis and pattern detection.

## Current implementation 


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
[TIMESTAMP] <METRIC_1>: value | <METRIC_2>: value | ... | Signal: ACTION | 📈/📉/➖ Trend Label or Emoji
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
[TIMESTAMP] <METRIC_1>: value | <METRIC_2>: value | ... | Signal: ACTION | 📈/📉/➖ Trend Label or Emoji
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
[TIMESTAMP] <METRIC_1>: value | <METRIC_2>: value | ... | Signal: ACTION | 📈/📉/➖ Trend Label or Emoji
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
[TIMESTAMP] <METRIC_1>: value | <METRIC_2>: value | ... | Signal: ACTION | 📈/📉/➖ Trend Label or Emoji
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
[TIMESTAMP] <METRIC_1>: value | <METRIC_2>: value | ... | Signal: ACTION | 📈/📉/➖ Trend Label or Emoji
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
[TIMESTAMP] <METRIC_1>: value | <METRIC_2>: value | ... | Signal: ACTION | 📈/📉/➖ Trend Label or Emoji
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
[TIMESTAMP] <METRIC_1>: value | <METRIC_2>: value | ... | Signal: ACTION | 📈/📉/➖ Trend Label or Emoji
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
[TIMESTAMP] <METRIC_1>: value | <METRIC_2>: value | ... | Signal: ACTION | 📈/📉/➖ Trend Label or Emoji
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
[TIMESTAMP] <METRIC_1>: value | <METRIC_2>: value | ... | Signal: ACTION | 📈/📉/➖ Trend Label or Emoji
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
XRPUSDT (4h) - Double bottom
Price: 3.1482 | Signal: BUY 🚀 | Neckline: 2.5312
Target: — | Confidence: 95%
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
XRPUSDT (4h) - Double top
Price: 3.1482 | Signal: BUY 🚀 | Neckline: 2.5312
Target: — | Confidence: 95%
### CLI
Make sure to add new handler to the CLI: `/cli/double_top_handler.py`
### All patterns
Add `double_top` detection to `/patterns/all_patterns.py`


## Phase 14: Head and Shoulders (H&S) Pattern
- Create file `/patterns/head_and_shoulders.py`
- Use **Close** price from OHLCV data (via `collector.py → fetch_ohlcv_data`)
- Require at least **80–150 candles** to reliably detect three major swing highs/lows
- **Pattern Definition (Bearish Reversal):**  
  - **Left Shoulder (LS):** swing high  
  - **Head:** higher swing high  
  - **Right Shoulder (RS):** lower high near LS level  
  - **Neckline:** support line connecting LS-RS valleys  
  - **Confirmation:** Close below neckline after RS
- **Signal Conditions:**  
  - Head peak is **above** both shoulders  
  - Both shoulders are **near-equal height** (within ~3%)  
  - Neckline break confirms the setup
- **SELL signal:** Price breaks neckline after RS → bearish reversal  
- Use with RSI divergence, MACD bearish crossover, or OBV drop for confluence  
- Effective in swing (4h) and position (1d) timeframes
### Input in terminal
> head_and_shoulders s=BTC/USDT t=4h l=150
### Output example in terminal
XRPUSDT (4h) - Head and shoulder
Price: 3.1482 | Signal: BUY 🚀 | Neckline: 2.5312
Target: — | Confidence: 95%
### CLI
Make sure to add new handler to the CLI: `/cli/head_and_shoulders_handler.py`
### All patterns
Add H&S to `/patterns/all_patterns.py` for full-pattern confluence scanning


## Phase 15: Inverse Head and Shoulders (iH&S) Pattern
- Create file `/patterns/inverse_head_and_shoulders.py`
- Use **Close** price from OHLCV data (via `collector.py → fetch_ohlcv_data`)
- Require at least **80–150 candles** to reliably detect three major swing lows/highs
- **Pattern Definition (Bullish Reversal):**  
  - **Left Shoulder (LS):** swing low  
  - **Head:** lower swing low  
  - **Right Shoulder (RS):** higher low near LS level  
  - **Neckline:** resistance line connecting LS-RS highs  
  - **Confirmation:** Close above neckline after RS
- **Signal Conditions:**  
  - Head dip is **lower** than both shoulders  
  - Shoulders are **roughly equal** (within ~3%)  
  - Breakout above neckline confirms the pattern
- **BUY signal:** Price breaks above neckline after RS → bullish reversal  
- Combine with RSI bullish divergence, MACD crossover, or OBV rise for confluence  
- Effective in swing (4h) and position (1d) timeframe
### Input in terminal
> inverse_head_and_shoulders s=SOL/USDT t=4h l=150
### Output example in terminal
XRPUSDT (4h) - Inverse head and shoulder
Price: 3.1482 | Signal: BUY 🚀 | Neckline: 2.5312
Target: — | Confidence: 95%
### CLI
Make sure to add new handler to the CLI: `/cli/inverse_head_and_shoulders_handler.py`
### All patterns
Add iH&S to `/patterns/all_patterns.py` for consolidated reversal detection


## Phase 16: Ascending Triangle Pattern
- Create file `/patterns/ascending_triangle.py`
- Use **Close** price from OHLCV data (via `collector.py → fetch_ohlcv_data`)
- Require at least **80–120 candles** to identify consolidation and breakout
- **Pattern Definition (Bullish Continuation):**  
  - **Flat Resistance Line (Top):** multiple swing highs near the same level  
  - **Rising Support Line (Bottom):** ascending swing lows  
  - **Converging Structure**: price squeezed between flat top and rising bottom  
  - **Breakout:** confirmed when price **closes above** resistance line with volume
- **BUY signal:** Close above resistance line + volume spike → breakout confirmed  
- **Failed Breakout / Fakeout Detection:**  
  - Price closes above but quickly returns below resistance → no signal  
- Use with MACD/RSI/Volume confluence to filter noise  
- Works best in **15m–4h** for breakout trading or **1d** for position entries
### Input in terminal
> ascending_triangle s=ETH/USDT t=1h l=120
### Output example in terminal
XRPUSDT (4h) - Triangle
Price: 3.1621 | Signal: NONE ⏳ | Neckline: —
Target: — | Confidence: —
### CLI
Make sure to add new handler to the CLI: `/cli/ascending_triangle_handler.py`
### All patterns
Add this pattern to `/patterns/all_patterns.py` to integrate with full breakout scanner


## Phase 17: Market Regime Detection (Trend vs Range)
- Create file `/trend/regime.py`
- Use **OHLCV** data (via `collector.py → fetch_ohlcv_data`)
- Works best on **1h to 1D** timeframes for reliable classification  
- **Core Indicators Used:**  
  - **ATR** (Average True Range) → Volatility gauge  
  - **ADX** (Average Directional Index) → Trend strength  
  - **EMA Slope** or EMA distance → Trend direction  
  - **Bollinger Band Width** → Compression/expansion signal  
- **Rules for Market Regime Classification:**  
  - **TRENDING** if:
    - **ADX > 25**
    - **EMA-9 > EMA-21** (Uptrend) or **EMA-9 < EMA-21** (Downtrend)
    - **ATR % of Close > 1.5%**
  - **RANGING** if:
    - **ADX < 20**
    - **EMA-9 ≈ EMA-21** (Flat or crossing repeatedly)
    - **Bollinger Band Width** is below 20-period average (compression)
  - **NEUTRAL / UNDEFINED** if mixed signals (e.g. high ADX but flat EMAs)
- **Regime Output:**  
  - `Regime: TRENDING ↑` or `Regime: TRENDING ↓`  
  - `Regime: RANGING ⏸️`  
  - `Regime: NEUTRAL ❓`  
- **Optional Enhancements:**  
  - Add **trend strength score** (e.g. scale from 0–100)  
  - Add **volatility level classification**: Low, Moderate, High  
  - Add recent **BOS/CHOCH detection** for breakout confirmation  
### Input in terminal
> regime s=BTC/USDT t=4h l=100
### Output example in terminal
[2025-07-29 04:00:00] ADX: 32.1 | ATR%: 2.3% | EMA Angle: Positive | BB Width: Expanding |  
Signal: TRENDING ↑ | 📈 Strong Bullish Trend
### CLI
Make sure to add handler: `/cli/regime_handler.py`
### All trend
Add regime output to `/trend/all_trend.py`


## Phase 18: Multi-Timeframe Confluence
- Create file `/trend/multi_tf_confluence.py`
- Use OHLCV data across at least 2-3 timeframes (e.g., Daily, 4H, 1H)
- Core idea: Confirm signals on higher timeframe before taking action on lower timeframe  
- **Rules:**
  - **EMA Alignment:**
    - Calculate EMA 9 and EMA 21 on each timeframe  
    - Confirm trend only if EMA 9 > EMA 21 on *all* selected timeframes for bullish trend  
    - Confirm trend only if EMA 9 < EMA 21 on *all* selected timeframes for bearish trend  
  - **MACD Histogram Confluence:**
    - MACD histogram should be positive on higher timeframe to support bullish trades on lower timeframe  
    - MACD histogram should be negative on higher timeframe to support bearish trades on lower timeframe  
  - **RSI Divergence Match:**
    - Check for RSI bullish or bearish divergence on the higher timeframe  
    - Only consider lower timeframe signals if divergence confirms potential reversal or continuation  
- **Signal Logic:**
  - Enter trade on lower timeframe only if higher timeframe confirms trend or divergence  
  - Reject or avoid trades if higher timeframe trend contradicts lower timeframe signals  
  - Use higher timeframe trend as bias filter, lower timeframe for entry precision  
- **Timeframe Examples:**
  - Daily (trend bias) + 4H (confirmation) + 1H (entry)  
  - 4H (trend) + 1H (confirmation) + 15m (entry)  
- **Bonus:**
  - Add strength scoring by counting how many timeframes align in the same direction  
  - Use cross-timeframe RSI and MACD divergence to detect early reversals  
### Input in terminal
> multi_tf s=BTC/USDT t1=1d t2=4h t3=1h indicators=ema9/21,macd,rsi14 l=200  
### Output example in terminal
[TIMESTAMP] EMA: Bullish (all TF) | MACD: Bullish Confirmed | RSI Div: Bullish HTF | Signal: BUY | 📈 Trend Strong
### CLI
Add new handler `/cli/multi_tf_handler.py` for signal processing and alerts


## Phase 19: Smart Money Concepts (SMC)
- Create file /trend/smc.py
- Use **OHLCV** data from `collector.py → fetch_ohlcv_data`
- Require **sufficient historical data** (≥200 candles) for structure context  
- Focus timeframes: **15m, 1h, 4h, 1D** for cleaner structure detection  
### 🔍 Core Concepts
#### 1. **Liquidity Zones**
- Identify **equal highs/lows**, recent swing highs/lows
- Mark zones where stop hunts are likely (retail SL clusters)
- Use wick rejections + volume spikes to confirm raid attempts  
- Zone logic:
  - **Above recent highs** = Buy-side liquidity
  - **Below recent lows** = Sell-side liquidity
#### 2. **Order Blocks**
- Find **last bullish/bearish candle before a strong move**
- Confirm with:
  - **Large-bodied impulse candle** following the block
  - **Volume surge** or **imbalance gap**
- Use zone ± buffer range for entry triggers
#### 3. **Break of Structure (BOS)**
- Detect when price **closes above/below previous swing high/low**
- Confirms trend continuation or trend shift initiation
#### 4. **Change of Character (CHOCH)**
- Signals **early reversal**
- Happens when price breaks the most recent **minor swing** opposite to current trend
- Use in confluence with OB or Liquidity sweep
### ✅ Signal Logic
- **BUY signal:**
  - Liquidity sweep + CHOCH + bullish OB retest  
- **SELL signal:**
  - Liquidity sweep + CHOCH + bearish OB retest  
### 📥 Input in terminal
> smc s=BTC/USDT t=1h l=300  
> smc s=ETH/USDT t=4h l=500 zones=true choch=true  
### 📤 Output example in terminal
[2025-07-29 16:00:00] BOS: YES | CHOCH: YES | OB Zone Hit: YES | Signal: BUY | 🟢 Bullish Reversal  
[2025-07-29 20:00:00] BOS: NO | CHOCH: YES | Signal: SELL | 🔻 Potential Liquidity Sweep  
### 🧠 Bonus
- Highlight **FVG (Fair Value Gaps)** or imbalance zones for sniper entries  
- Add option to export marked zones as plot or JSON for chart overlay  
- Optional: Detect **internal BOS (iBOS)** for LTF entry precision  
### CLI
Make sure to add new handler to the CLI: /cli/smc_handler.py


## Phase 20: Statistical Pattern Validation
- Create file /trend/stat_pattern_validation.py  
- Use **OHLCV** data and output from existing signal modules (e.g., RSI, MACD, Double Top, etc.)
- Require labeled historical data or run inference using existing strategies over past candles  
- Focus timeframes: **15m to 1D** for robust pattern outcome sampling
### 📊 Validation Engine
#### 1. **Pattern Outcome Tracking**
- Collect occurrences for the following key patterns:  
  - Double Bottom  
  - Double Top  
  - Head and Shoulders (regular and inverse)  
  - Triangles (ascending, descending, symmetrical)  
  - Wedges (rising and falling)   
  - MACD Crossovers  
  - RSI Divergence  
  - Volume Breakouts  
  - Supertrend Signals  
  - VWAP Crosses  
  - ATR + ADX Trend Regime Signals  
  - Smart Money Concepts Zones & BOS/CHOCH  
  - On-Balance Volume (OBV) Breakouts  
- Record subsequent price movement (e.g., +X% within N candles)  
- Label each occurrence as: `Success`, `Fail`, or `Neutral`
#### 2. **Metrics to Calculate**
- **Win Rate:** % of pattern signals that hit profit target before stop loss  
- **Expectancy:** Avg. gain/loss per trade = (WinRate × AvgWin) - (LossRate × AvgLoss)  
- **Signal Delay:** Bars between pattern trigger and move  
- **Frequency:** How often each pattern appears (per 100 candles)  
#### 3. **Confusion Matrix (Binary Classifier Style)**
- TP = Correctly predicted move  
- FP = False breakout  
- FN = Missed move  
- TN = Correctly skipped  
- Output classification matrix per pattern type  
### ✅ Signal Logic for Backtest
- BUY signal:
  - Pattern = "Double Bottom"  
  - After signal, check if `Close[n] > Entry × (1 + TP%)` before SL is hit  
- SELL signal:
  - Pattern = "Head and Shoulders"  
  - Check if `Close[n] < Entry × (1 - TP%)`
- You define TP/SL rules globally or per pattern  
### Bonus Features
- Calculate **Sharpe Ratio** and **Max Drawdown** per pattern  
- Add histogram of **outcome distribution** (e.g., % gain after 5 bars)  
- Generate a leaderboard of **top-performing patterns** by asset
### Input in terminal
> statval s=BTC/USDT t=1h l=500 pattern=double_bottom tp=2% sl=1.5%  
> statval s=ETH/USDT t=4h pattern=macd_cross timeframe_outcome=5
### Output example in terminal
[2025-07-29] Pattern: Double Bottom | Win Rate: 64.3% | Expectancy: +0.82R | Count: 28 | Signal Quality: ✅  
[2025-07-29] Pattern: RSI Divergence | Win Rate: 45.1% | Expectancy: -0.12R | Count: 41 | Signal Quality: ❌
### CLI
Make sure to add new handler to the CLI: /cli/stat_pattern_handler.py