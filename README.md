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


## Phase 3: Enhanced Order Flow Imbalance Detection
1. Create file `/orderflow/imbalance.py`
2. Use **real-time Level 2 (L2) market depth data** (via `collector.py → fetch_orderbook_stream`) **and** **tick trade data** (`collector.py → fetch_trades_stream`)
3. **Industry-Standard Data Processing:**
   - **Multi-Timeframe Snapshots**: 100ms, 500ms, 1s, 5s, 30s intervals
   - **Volume-Weighted Calculations**: Weight by trade size and execution quality
   - **Latency-Aware Processing**: Sub-100ms processing for relevance
   - **Exchange-Specific Logic**: Handle different tick rules and trade classifications
4. **Enhanced Imbalance Calculations:**
   - **Basic Formula**: `imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)`
   - **Volume-Weighted Imbalance**: `vw_imbalance = Σ(trade_size * direction) / Σ(trade_size)`
   - **Time-Decay Weighted**: Apply exponential decay to older snapshots (λ=0.95)
   - **Statistical Significance**: Calculate z-scores and confidence intervals
   - **Market Impact Adjustment**: Normalize by average trade size and liquidity depth
5. **Professional Order Flow Analytics:**
   - **Trade Classification**:
     - Aggressive buyers (market orders lifting ask)
     - Aggressive sellers (market orders hitting bid)  
     - Passive fills (limit orders being hit)
     - Self-trade and wash trade filtering
   - **Liquidity Context Integration**:
     - Total available liquidity at each level
     - Liquidity concentration ratios
     - Order book imbalance correlation
     - VWAP deviation impact on signals
   - **Market Regime Awareness**:
     - Trending vs ranging market adjustments
     - Volatility-based threshold scaling
     - Session-specific behavior patterns
6. **Advanced Signal Detection:**
   - **Smart Thresholds**: Dynamic thresholds based on volatility and liquidity
   - **Multi-Level Analysis**: Analyze imbalances across price levels simultaneously  
   - **Pattern Recognition**:
     - Absorption vs rejection patterns
     - Hidden liquidity detection
     - Institutional footprint identification
     - Algorithmic trading pattern detection
   - **False Signal Filtering**:
     - Statistical significance testing (p-value < 0.05)
     - Minimum volume requirements
     - Cross-validation with order book data
7. **Comprehensive Data Storage:**
   - Store as enhanced time-series: `(timestamp, price_level, imbalance_value, vw_imbalance, buy_volume, sell_volume, trade_count, avg_trade_size, z_score, confidence_level, liquidity_context)`
   - Historical pattern database for machine learning
   - Real-time anomaly detection metrics
8. **Professional Signal Interpretation:**
   - **Bullish Signals**:
     - Strong positive imbalance with statistical significance (z > 2.0)
     - Buyers absorbing offers at or below current price
     - Increasing aggressive buy volume with stable/rising price
   - **Bearish Signals**:
     - Strong negative imbalance with statistical significance (z < -2.0)
     - Sellers absorbing bids at or above current price  
     - Increasing aggressive sell volume with stable/falling price
   - **Advanced Patterns**:
     - **Hidden Imbalance**: High imbalance without price movement (trap detection)
     - **Exhaustion Signals**: Decreasing imbalance despite continued directional pressure
     - **Institutional Footprints**: Large block trades creating temporary imbalances
     - **Spoofing Detection**: Rapid order placement/cancellation affecting imbalance

### Enhanced Input Parameters
> imbalance s=BTC/USDT t=500ms d=600 th=0.6 analytics=full vw=true decay=0.95 significance=0.05
- `s` = symbol (required)
- `t` = snapshot interval (100ms, 500ms, 1s, 5s, 30s)
- `d` = duration (number of snapshots or time: 300s, 5m, 1h)
- `th` = imbalance threshold (0-1, auto-scaling available)
- `analytics` = analysis level (basic, standard, full, institutional)
- `vw` = enable volume-weighted calculations
- `decay` = time decay factor (0.9-0.99, default: 0.95)
- `significance` = statistical significance level (0.01-0.1, default: 0.05)
- `regime` = market regime adjustment (trending, ranging, auto)
- `min_volume` = minimum volume threshold for signals
- `liquidity_context` = include order book liquidity analysis

### Enhanced Professional Output
```
[2025-07-31 14:00:05.500] BTC/USDT | Interval: 500ms | Analytics: FULL
═══════════════════════════════════════════════════════════════════════════════
💹 MARKET CONTEXT
Price: 43,200.50 (+12.30, +0.028%) | Spread: $0.50 | Volume: 247.3 BTC (5m)
VWAP: 43,195.20 | Session: LONDON_OPEN | Regime: TRENDING_UP | Vol: NORMAL

📊 ORDER FLOW IMBALANCE ANALYSIS
Multi-Timeframe View:
 • 100ms → +0.45 (emerging)     • 5s  → +0.71 (strong)
 • 500ms → +0.68 (significant)  • 30s → +0.58 (sustained) 

🎯 TOP STATISTICAL IMBALANCES (Z-Score > 2.0)
 • 43,195.00 → +0.84 ⭐ (Buy: 1,247 | Sell: 156) | Z: 3.2 | p<0.001
   └─ Volume-Weighted: +0.91 | Confidence: 99.9% | INSTITUTIONAL ABSORPTION
   └─ Context: Major support + 15.7K liquidity wall | Absorption Rate: 87%
   
 • 43,210.00 → -0.72 ❌ (Buy: 203 | Sell: 1,089) | Z: -2.7 | p<0.01  
   └─ Volume-Weighted: -0.78 | Confidence: 99.0% | BEARISH REJECTION
   └─ Context: Resistance cluster + weak liquidity | Rejection Rate: 74%
   
 • 43,205.00 → +0.66 🔍 (Buy: 834 | Sell: 421) | Z: 2.1 | p<0.05
   └─ Volume-Weighted: +0.59 | Confidence: 95.0% | HIDDEN BULLISH TRAP
   └─ Context: No price movement despite buying pressure | Algo Activity: HIGH

⚡ REAL-TIME SIGNALS & ALERTS
🔥 PRIMARY SIGNAL: STRONG BUY | Confidence: 87.3% (HIGH)
└─ Rationale: Sustained absorption at key support + Statistical significance
└─ Entry: Above 43,201 | Stop: 43,190 | Target1: 43,225 | Target2: 43,250
└─ Risk/Reward: 1:2.2 | Expected Success Rate: 73% (historical backtest)

⚠️  CRITICAL ALERTS:
• IMBALANCE REGIME SHIFT: Flip from -0.23 → +0.84 in 2.1s (BULLISH REVERSAL)
• INSTITUTIONAL FOOTPRINT: 3 block trades (>50 BTC) absorbed at 43,195 support
• SPOOFING DETECTED: 125 BTC fake wall removed at 43,212 (Confidence: 91%)
• EXHAUSTION WARNING: Selling pressure decreasing despite failed breakout

📈 ADVANCED ANALYTICS
Order Flow Quality Score: 8.4/10 (Excellent institutional participation)
Market Impact Analysis: Large trades showing 0.02% average impact (healthy)
Liquidity Depth Ratio: 2.3:1 (Bid favored) | Concentration: 67% in top 5 levels
Algo Activity Level: 43% of volume | Pattern: ACCUMULATION_PHASE
Statistical Reliability: 94% (600+ data points, robust significance)

🧠 PATTERN RECOGNITION
Detected Pattern: INSTITUTIONAL_ACCUMULATION_WITH_SUPPORT_TEST
Historical Success Rate: 78% (274/351 occurrences)
Similar Setups: 2025-07-28 09:23 (+1.4%), 2025-07-25 15:47 (+2.1%)
Key Risk: Failed absorption below 43,190 → Potential -0.8% correction
═══════════════════════════════════════════════════════════════════════════════
```

### CLI
Add handler to `/cli/imbalance_handler.py`
- Implement `handle_imbalance_command`  
- Integrate into main CLI dispatcher  
