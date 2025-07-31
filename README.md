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


## Phase 2: Butterfly Pattern Detection
1. Create file: `/trend/butterfly_pattern.py`
2. Use OHLCV data: High, Low, and Close prices are required
3. Detect potential Butterfly pattern using the X-A-B-C-D leg structure
   - Identify swing points using ZigZag algorithm or fractal pivot detection
4. Butterfly Leg Ratio Rules:
   - AB = 0.786 retracement of XA ✅
   - BC = 0.382 to 0.886 retracement of AB ✅
   - CD = 1.618 to 2.618 extension of BC ✅
   - AD = 1.27 extension of XA ✅
5. Entry Signal:
   - At point D, if pattern completes within tight Fibonacci confluence zone
   - Additional confirmation: Volume spike + rejection candle at D
6. Target Zones:
   - TP1 = 38.2% retracement of CD
   - TP2 = 61.8% retracement of CD
   - SL = slightly beyond point X
### Input in terminal
> butterfly s=XRP/USDT t=4h l=150 zz=5
- `s` = symbol  
- `t` = timeframe  
- `l` = limit (candles to load)  
- `zz` = ZigZag threshold (% swing sensitivity)
### Output example in terminal
[HARMONIC STRUCTURE: BUTTERFLY]
Symbol: XRP/USDT | Timeframe: 4h
Pattern Status: ✅ VALID | Bias: 📈 Bullish
• X: 0.500
• A: 0.610
• B: 0.534 (AB retrace: 0.786) ✅
• C: 0.585 (BC retrace: 0.618) ✅
• D: 0.450 (CD ext: 2.240) ✅ → 📍 Entry
Fibonacci Confluence ✅ | Volume Spike ✅ | Rejection Candle ✅
🎯 Target 1: 0.494 (TP1)
🎯 Target 2: 0.517 (TP2)
🛑 Stop Loss: 0.438
🚦 Signal: BUY | Confidence: HIGH
### CLI
Make sure to add a handler `/cli/butterfly_pattern_handler.py`