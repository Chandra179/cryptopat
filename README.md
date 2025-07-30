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


## Phase 2: Elliott Wave + Fibonacci Confluence
1. Create file: .../trend/elliott_fibonacci.py
2. Use High, Low, Close from OHLCV data (`collector.py → fetch_ohlcv_data`)
3. Elliott Wave Cardinal Rules:
    - ✅ Wave 3 cannot be the shortest
    - ✅ Wave 2 cannot retrace more than 100% of Wave 1
    - ✅ Wave 4 cannot overlap Wave 1 (except diagonals)
4. Standard Fibonacci Ratios:
    - ✅ Wave 2: 50%-78.6% retracement (0.5-0.786)
    - ✅ Wave 3: 1.618× Wave 1 extension (most common)
    - ✅ Wave 4: 23.6%-38.2% retracement (0.236-0.382)
    - ✅ Wave 5: 0.618× Wave 1 or equal to Wave 1
    - ✅ Wave C: 1.0-1.618× Wave A extension
5. Advanced Features:
    - ✅ ZigZag detection for swing point identification
    - ✅ Confidence scoring based on Fibonacci confluence
    - ✅ Pattern validation with Elliott Wave rules
    - ✅ Target projections for incomplete waves
### Input in terminal
> elliott_fibonacci s=SOL/USDT t=4h l=150 zz=4
- s = symbol
- t = timeframe
- l = candle limit
- zz = ZigZag threshold or fractal depth
### Output example in terminal
[ELLIOTT + FIBONACCI STRUCTURE]
- Pattern: Impulse Wave (5-wave)
- Wave 1: 42.00 → 50.00
- Wave 2: 50.00 → 46.50 (0.618 retracement)
- Wave 3: 46.50 → 61.80 (1.618 extension of W1)
- Wave 4: 61.80 → 58.80 (0.382 retracement)
- Wave 5: Projected to 66.00 (0.618 of Wave 1)
- Status: Wave 5 in progress
- Confluence: Strong — multiple Fib + structure alignment
### CLI
Add a new handler:
- `/cli/elliott_fibonacci_handler.py`
