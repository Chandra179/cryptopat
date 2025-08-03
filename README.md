# Crypto Pattern Recognition - Backend Data Analysis

## Overview
A Python-based system for detecting chart patterns in cryptocurrency data using historical price, volume, and order book information. No frontend - pure data analysis and pattern detection.

## Current implementation 
atr_adx, bollinger bands, ema 9/21, macd, obv, rsi14, smart money concept, supertrend, Volume Weighted Average Price,
butterfly pattern, double bottom, double top, elliot wave, flag, head and shoulder, inverse head and shoulder, shark pattern,
triangle pattern, wedge pattern

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

## Phase 2: CLI
```
> s=BTC/USDT t=1h l=100

s = coin symbol
t = timeframe (15m, 1h, 4h, 1d, 3d, 1w, 1m)
l = number of candles
```

## Phase 3: Market summary
{
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "analysis_time": "2025-08-02 18:00:00",
  "overall_trend_summary": "BTC/USDT on the 1H timeframe shows strong bullish alignment. EMA crossover is confirmed with volume support, Supertrend is in bullish mode, RSI is rising, and MACD histogram is expanding. Multiple indicators agree on trend and momentum direction with high confidence.",
  "bias": {
    "direction": "bullish",
    "confidence": 83
  },
  "combined_signals": {
    "bullish_count": 6,
    "bearish_count": 2,
    "neutral_count": 1,
    "total_signals": 9
  },
  "indicator_reports": {
    "vwap": "Price is currently above the VWAP, indicating short-term bullish bias. Anchored VWAP from July 31 shows a sustained uptrend with more buy signals than sell signals. Signal confidence is high at 78%.",
    
    "supertrend": "Price is above the Supertrend line (bullish). ATR value is 87.2, suggesting moderate volatility. Trend is consistent, with a risk-reward ratio of 3.4.",
    
    "rsi": "RSI is at 64.3, which is considered neutral but leaning bullish. Signal is confirmed by price structure. Current momentum is building, and entry zone is favorable.",
    
    "macd": "MACD line is 0.0035, signal line is 0.0021, histogram is 0.0014. This suggests growing momentum with bullish bias. Risk-reward is 2.8, confidence is 76%.",
    
    "ema_9_21": "EMA-9 is 27300.5 and EMA-21 is 27020.8, indicating a bullish crossover. Volume is confirmed, and price has changed +2.4%. Entry zone is 27300–27450, RR is 3.1.",
    
    "bollinger_bands": "Price is currently 88% within the Bollinger Band range. A squeeze is occurring, indicating potential breakout. Signal suggests bullish move with context: price nearing upper band after volatility compression.",
    
    "atr_adx": "ATR is 73.5 and ADX is 29.2, indicating a strengthening trend. DI+ is 26.1, DI- is 14.7, suggesting bullish bias. Entry zone: 27450–27580, RR: 3.6.",

    "obv": "OBV is currently {obv_state}, {divergence_text} relative to price movement. This indicates {volume_bias}. Recent volume flow {volume_direction} suggests {momentum_assessment}."
  },
  "multi_indicator_alignment": {
    "trend_agreement": true,
    "momentum_agreement": true,
    "entry_signal": true,
    "entry_window_score": 9.1,
    "risk_score": 6.8,
    "overall_rr_ratio": 3.2
  },
  "trade_recommendation": {
    "action": "buy",
    "entry_zone": [27380.0, 27550.0],
    "stop_loss": 26820.0,
    "take_profit": [28100.0, 28900.0],
    "max_drawdown": 2.1
  }
}
