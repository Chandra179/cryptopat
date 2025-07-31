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


## Phase 2: Elliott Wave Rules
1. Create file: `/trend/elliott_wave.py`  
2. Use High, Low, Close from OHLCV data (`collector.py → fetch_ohlcv_data`)  
3. Detect significant swings using ZigZag threshold within `l` candles for pattern identification  
4. **Core Elliott Wave Rules** (Industry Standard):
   - **Wave 3 is NEVER the shortest** of waves 1, 3, and 5
   - **Wave 2 cannot retrace more than 100%** of Wave 1
   - **Wave 4 cannot overlap** into Wave 1 price territory (except in diagonal triangles)
5. **Impulse wave rules (Waves 1–5)**:
   - **Wave 2** retraces **38.2–78.6%** of Wave 1 (most common: 50-61.8%)
   - **Wave 3** typically **1.618×** Wave 1 or **2.618×** Wave 1 (extended wave)
   - **Wave 4** retraces **23.6–50%** of Wave 3 (alternation with Wave 2)
   - **Wave 5** = **0.618×**, **1.0×**, or **1.618×** Wave 1 (truncation possible)
6. **Corrective wave rules (Waves A–B–C)**:
   - **Wave B** retraces **38.2–78.6%** of Wave A (rarely exceeds 100%)
   - **Wave C** = **1.0×**, **1.618×**, or **2.618×** Wave A length
7. **Wave relationships & alternation**:
   - **Alternation principle**: Wave 2 vs Wave 4 differ in complexity/time
   - **Extension rule**: One of waves 1, 3, or 5 extends (typically Wave 3)
   - **Time relationships**: Wave patterns often show Fibonacci time ratios
8. **Confirmation criteria**:
   - All core rules satisfied + valid swing structure
   - Volume confirmation (Wave 3 highest, Wave 5 divergence possible)
   - Fibonacci confluence at key reversal points
   - RSI divergence at Wave 5 completion (optional)  

### Input in terminal
> elliott_wave s=BTC/USDT t=1h/4h/1d l=150 zz=5  
- **s** = symbol  
- **t** = timeframe  
- **l** = candle limit  
- **zz** = ZigZag threshold (%) for swing detection  
### Output example in terminal
[ELLIOTT WAVE STRUCTURE]
Symbol: BTC/USDT | TF: 4h | Pattern: Impulse Wave (Bullish)
Core Rules: ✅ VALID (Wave 3 longest, no overlaps)
Waves:
1: 30000 → 31200 (1200pts)
2: 31200 → 30500 (58.3% retrace) ✅
3: 30500 → 33000 (2500pts, 2.08× W1) ✅ EXTENDED
4: 33000 → 32400 (24% retrace) ✅ ALT
5: 32400 → 34000 (1600pts, 1.33× W1) ✅
Alternation: Wave 2 sharp, Wave 4 sideways ✅
Volume: W3 highest, W5 divergence ⚠️
Signal: Wave 5 completion ↗️ → ABC correction expected
🎯 Target Zone: 35000-35500 (1.618-2.618× W1 from W4 low)
⚠️ Watch for: RSI divergence, volume decline
### CLI
Make sure to add new handler: `/cli/elliott_wave_handler.py`