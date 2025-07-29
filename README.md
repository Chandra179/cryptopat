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
