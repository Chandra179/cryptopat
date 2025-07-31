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


## Phase 2: Advanced Order Book Heatmap & Market Microstructure Analysis
1. Create file `/orderflow/orderbook_heatmap.py`
2. Use **real-time Level 2 (L2) market depth data** (via `collector.py → fetch_orderbook_stream`)
3. **Enhanced Data Collection:**
   - Capture snapshots of top 20–50 bid/ask levels **every 100–500ms** for high-frequency analysis
   - Correlate with **Time & Sales execution data** for order flow validation
   - Track **order additions, cancellations, and modifications** (if supported by exchange)
   - Store order book imbalance ratios at each snapshot
4. **Industry-Standard Market Microstructure Analytics:**
   - **Volume-Weighted Average Price (VWAP)** integration for institutional reference levels
   - **Cumulative Volume Delta (CVD)** tracking buyer vs seller aggression
   - **Order Flow Imbalance (OFI)** calculation: `(Bid_Volume - Ask_Volume) / (Bid_Volume + Ask_Volume)`
   - **Liquidity depth metrics**: Average depth, depth volatility, liquidity concentration
   - **Market maker vs taker classification** using trade direction inference
5. Store historical depth as **multi-dimensional time-series matrix:**
   - Rows = timestamps (100-500ms intervals)
   - Columns = normalized price levels (% from mid-price)
   - Values = [quantity, order_count, avg_order_size, time_in_book]
   - Additional layers: CVD, OFI, VWAP deviation, liquidity scores
6. **Advanced Signal Detection:**
   - **Institutional Liquidity Patterns:**
     - **Iceberg Orders**: Detect hidden liquidity through order refill patterns
     - **Algorithmic Spoofing**: Statistical detection of fake walls (rapid place/cancel cycles)
     - **Liquidity Hunting**: Track price movements toward large orders
     - **Hidden Liquidity**: Identify dark pool activity through execution vs visible depth mismatches
   - **Market Regime Classification:**
     - **Normal Trading**: Balanced order flow, stable spreads
     - **Stress Conditions**: Widening spreads, liquidity gaps, unusual imbalances
     - **Manipulation Detection**: Coordinated spoofing, wash trading patterns
   - **Predictive Signals:**
     - **Absorption vs Rejection**: How price reacts when hitting liquidity walls
     - **Liquidity Magnets**: Large orders attracting price movement
     - **Support/Resistance Validation**: Order flow confirmation of technical levels
7. **Professional Visualization & Analytics:**
   - Multi-layer heatmap with configurable depth and time horizons
   - Real-time CVD and OFI overlay on price chart
   - Statistical significance testing for detected patterns
   - Confidence intervals for liquidity wall predictions
   - Anomaly detection alerts for unusual order flow behavior

### Enhanced Input Parameters
> orderbook_heatmap s=XRP/USDT t=250ms d=1200 depth=30 analytics=full vwap=true
- `s` = symbol (required)
- `t` = snapshot interval (100ms, 250ms, 500ms, 1s)
- `d` = duration (number of snapshots or time period: 300s, 5m, 1h)
- `depth` = order book depth levels to analyze (10-50, default: 20)
- `analytics` = analysis level (basic, standard, full, institutional)
- `vwap` = enable VWAP tracking and deviation analysis
- `cvd` = track Cumulative Volume Delta
- `ofi` = calculate Order Flow Imbalance
- `spoof` = enable spoofing detection algorithms
- `regime` = market regime classification
- `alerts` = real-time anomaly detection thresholds

### Enhanced Output with Professional Metrics
```
[2025-07-31 14:00:15.250] XRP/USDT | 250ms Snapshots | Depth: 30 levels
═══════════════════════════════════════════════════════════════════════════════
💹 MARKET DATA
Price: 0.6428 (+0.0003, +0.05%) | Mid: 0.64275 | Spread: 0.0005 (0.08%)
Best Bid: 0.6425 (1,847 XRP, 3 orders) | Best Ask: 0.6430 (2,103 XRP, 5 orders)
VWAP: 0.6422 (-0.0006 from current) | Volume Profile: Heavy @ 0.6420-0.6430

📊 ORDER FLOW ANALYTICS
CVD (5m): +245.2K XRP (Buyer Aggression) | OFI: +0.34 (Bid Heavy)
Imbalance Ratio: 67% Bids / 33% Asks | Liquidity Score: 8.7/10 (Excellent)
Market Regime: NORMAL_TRENDING | Volatility: LOW (0.12%)

🔍 INSTITUTIONAL LIQUIDITY ZONES
• MAJOR SUPPORT WALL @ 0.6400 (15.7K XRP, 12 orders) - 92% confidence
  └─ Depth Quality: HIGH | Time in Book: 4m 23s | Iceberg Detected: NO
• RESISTANCE CLUSTER @ 0.6450-0.6455 (23.1K XRP, 18 orders)
  └─ Depth Quality: MEDIUM | Absorption Rate: 45% | Spoof Risk: LOW

⚠️  REAL-TIME ALERTS
• SPOOF DETECTED @ 0.6435: 6.5K wall vanished in 1.2s (High Confidence: 87%)
• ICEBERG ACTIVITY @ 0.6428: 500 XRP orders refilling every 15s
• WHALES ACCUMULATING: 3 large buyers absorbed 12K at 0.6425 support

🎯 TRADING SIGNALS
Primary Bias: 📈 BULLISH (Strength: 7.2/10)
└─ Rationale: Price climbing into resistance + Strong bid support + Positive CVD
Signal: WATCH 0.6450 breakout | Stop: Below 0.6400 | Target: 0.6480-0.6500
Confidence: 74% | Risk/Reward: 1:2.8 | Expected Move: +1.2% / -0.7%

📈 MICROSTRUCTURE HEALTH
Spread Stability: GOOD (σ=0.0002) | Order Arrival Rate: 47/min
Market Impact Cost: 0.03% (1K trade) | Liquidity Replenishment: 23s avg
Algo Activity Level: MODERATE (34% of volume) | Market Maker Presence: ACTIVE
═══════════════════════════════════════════════════════════════════════════════
```

### CLI Integration
add Handler: `/cli/orderbook_heatmap_handler.py`
