"""
## Data structures
**OHLCV Format:**
[
    [
        1504541580000, // UTC timestamp in milliseconds, integer
        4235.4,        // (O)pen price, float
        4240.6,        // (H)ighest price, float
        4230.0,        // (L)owest price, float
        4230.7,        // (C)losing price, float
        37.72941911    // (V)olume float (usually in terms of the base currency, the exchanges docstring may list whether quote or base units are used)
    ],
    ...
]

**Order Book Format:**
{
    'bids': [
        [ price, amount ], // [ float, float ]
        [ price, amount ],
        ...
    ],
    'asks': [
        [ price, amount ],
        [ price, amount ],
        ...
    ],
    'symbol': 'ETH/BTC', // a unified market symbol
    'timestamp': 1499280391811, // Unix Timestamp in milliseconds (seconds * 1000)
    'datetime': '2017-07-05T18:47:14.692Z', // ISO8601 datetime string with milliseconds
    'nonce': 1499280391811, // an increasing unique identifier of the orderbook snapshot
}


**Ticker Format:**
{
    'symbol':        string symbol of the market ('BTC/USD', 'ETH/BTC', ...)
    'info':        { the original non-modified unparsed reply from exchange API },
    'timestamp':     int (64-bit Unix Timestamp in milliseconds since Epoch 1 Jan 1970)
    'datetime':      ISO8601 datetime string with milliseconds
    'high':          float, // highest price
    'low':           float, // lowest price
    'bid':           float, // current best bid (buy) price
    'bidVolume':     float, // current best bid (buy) amount (may be missing or undefined)
    'ask':           float, // current best ask (sell) price
    'askVolume':     float, // current best ask (sell) amount (may be missing or undefined)
    'vwap':          float, // volume weighed average price
    'open':          float, // opening price
    'close':         float, // price of last trade (closing price for current period)
    'last':          float, // same as `close`, duplicated for convenience
    'previousClose': float, // closing price for the previous period
    'change':        float, // absolute change, `last - open`
    'percentage':    float, // relative change, `(change/open) * 100`
    'average':       float, // average price, `(last + open) / 2`
    'baseVolume':    float, // volume of base currency traded for last 24 hours
    'quoteVolume':   float, // volume of quote currency traded for last 24 hours
}


**Trades Format:**
[
    {
        'info':          { ... },                  // the original decoded JSON as is
        'id':           '12345-67890:09876/54321', // string trade id
        'timestamp':     1502962946216,            // Unix timestamp in milliseconds
        'datetime':     '2017-08-17 12:42:48.000', // ISO8601 datetime with milliseconds
        'symbol':       'ETH/BTC',                 // symbol
        'order':        '12345-67890:09876/54321', // string order id or undefined/None/null
        'type':         'limit',                   // order type, 'market', 'limit' or undefined/None/null
        'side':         'buy',                     // direction of the trade, 'buy' or 'sell'
        'takerOrMaker': 'taker',                   // string, 'taker' or 'maker'
        'price':         0.06917684,               // float price in quote currency
        'amount':        1.5,                      // amount of base currency
        'cost':          0.10376526,               // total cost, `price * amount`,
        'fee':           {                         // if provided by exchange or calculated by ccxt
            'cost':  0.0015,                       // float
            'currency': 'ETH',                     // usually base currency for buys, quote currency for sells
            'rate': 0.002,                         // the fee rate (if available)
        },
        'fees': [                                  // an array of fees if paid in multiple currencies
            {                                      // if provided by exchange or calculated by ccxt
                'cost':  0.0015,                   // float
                'currency': 'ETH',                 // usually base currency for buys, quote currency for sells
                'rate': 0.002,                     // the fee rate (if available)
            },
        ]
    },
    ...
]
"""

from typing import List, Dict
import pandas as pd
from analysis_summary import add_indicator_result, IndicatorResult


class OBV:
    
    def __init__(self, 
                 symbol: str,
                 timeframe: str,
                 limit: int,
                 ob: dict,
                 ticker: dict,            
                 ohlcv: List[List],       
                 trades: List[Dict]):
        self.param = {
            # On-Balance Volume Standard Parameters (Source: Joseph Granville, TradingView, Wikipedia)
            "smoothing_period": 0,           # Optional smoothing period (0 = no smoothing)
            "smoothing_type": "sma",         # Type of smoothing: sma, ema
            "signal_line_period": 10,        # Period for signal line moving average
            "divergence_period": 14,         # Lookback period for divergence detection
            "trend_confirmation_period": 5,  # Period to confirm trend changes
            "volume_threshold": 1.0,         # Minimum volume multiplier to consider significant
            "price_change_threshold": 0.001, # Minimum price change % to avoid noise
            
            # Signal Generation Parameters
            "breakout_threshold": 0.02,      # OBV % change threshold for breakout signals
            "divergence_min_periods": 5,     # Minimum periods for valid divergence
            "trend_strength_period": 20,     # Period for trend strength calculation
            "momentum_period": 10,           # Period for OBV momentum calculation
        }
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.ticker = ticker
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
    
    def calculate(self):
        """
        Calculate On-Balance Volume (OBV) according to Joseph Granville's methodology.
        
        Formula (Source: Joseph Granville, 1963):
        OBV = OBV_prev + {
            + volume    if close > close_prev
            + 0         if close = close_prev  
            - volume    if close < close_prev
        }
        
        The calculation assigns the entire day's volume a positive or negative value
        depending on whether the closing price moved up or down from the previous day.
        
        Core Concept:
        "Volume is higher on days where the price move is in the dominant direction"
        - Rising OBV indicates buying pressure and potential upward price movement
        - Falling OBV indicates selling pressure and potential downward price movement
        
        References:
        - Created by Joseph Granville in 1963 in "Granville's New Key to Stock Market Profits"
        - Originally called "continuous volume" by Woods and Vignola
        - Wikipedia: https://en.wikipedia.org/wiki/On-balance_volume
        - TradingView: On-Balance Volume technical indicator documentation
        
        Applications:
        - Confirms price moves by tracking volume direction
        - Detects potential trend strength or weakness
        - Identifies bullish/bearish divergences
        - Volume-based momentum analysis
        """
        if not self.ohlcv or len(self.ohlcv) < 2:
            result = {
                "error": f"Insufficient data: need at least 2 candles, got {len(self.ohlcv) if self.ohlcv else 0}"
            }
            return result
            
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # Calculate price change threshold
        price_change_threshold = self.param["price_change_threshold"]
        
        # Calculate OBV using vectorized operations
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['price_change'] / df['close'].shift(1)
        
        # Apply volume threshold filter
        volume_threshold = self.param["volume_threshold"]
        avg_volume = df['volume'].rolling(window=10).mean()
        significant_volume = df['volume'] >= (avg_volume * volume_threshold)
        
        # OBV calculation with Granville's formula
        obv_delta = pd.Series(0.0, index=df.index, dtype='float64')
        
        # Positive volume when close > close_prev and significant price change
        up_condition = (df['price_change'] > 0) & (abs(df['price_change_pct']) >= price_change_threshold)
        obv_delta.loc[up_condition] = df.loc[up_condition, 'volume']
        
        # Negative volume when close < close_prev and significant price change  
        down_condition = (df['price_change'] < 0) & (abs(df['price_change_pct']) >= price_change_threshold)
        obv_delta.loc[down_condition] = -df.loc[down_condition, 'volume']
        
        # Apply volume significance filter
        obv_delta = obv_delta.where(significant_volume, 0)
        
        # Calculate cumulative OBV
        df['obv'] = obv_delta.cumsum()
        
        # Optional smoothing
        if self.param["smoothing_period"] > 0:
            period = self.param["smoothing_period"]
            if self.param["smoothing_type"] == "ema":
                df['obv_smooth'] = df['obv'].ewm(span=period).mean()
            else:  # sma
                df['obv_smooth'] = df['obv'].rolling(window=period).mean()
            obv_series = df['obv_smooth']
        else:
            obv_series = df['obv']
            
        # Signal line
        signal_period = self.param["signal_line_period"]
        signal_line = obv_series.rolling(window=signal_period).mean()
        
        # OBV momentum
        momentum_period = self.param["momentum_period"]
        obv_momentum = obv_series.pct_change(periods=momentum_period)
        
        # Trend analysis
        trend_period = self.param["trend_confirmation_period"]
        obv_trend = obv_series.rolling(window=trend_period).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else (-1 if x.iloc[-1] < x.iloc[0] else 0)
        )
        
        # Current values
        current_obv = float(obv_series.iloc[-1])
        current_signal = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else current_obv
        current_momentum = float(obv_momentum.iloc[-1]) if not pd.isna(obv_momentum.iloc[-1]) else 0.0
        current_trend = int(obv_trend.iloc[-1]) if not pd.isna(obv_trend.iloc[-1]) else 0
        current_price = float(df['close'].iloc[-1])
        current_volume = float(df['volume'].iloc[-1])
        
        # Signal generation
        signal = "neutral"
        breakout_threshold = self.param["breakout_threshold"]
        
        if current_momentum > breakout_threshold:
            signal = "bullish_volume"
        elif current_momentum < -breakout_threshold:
            signal = "bearish_volume"
        elif current_obv > current_signal and current_trend > 0:
            signal = "accumulation"
        elif current_obv < current_signal and current_trend < 0:
            signal = "distribution"
            
        # Divergence detection (simplified)
        price_trend = df['close'].rolling(window=trend_period).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else (-1 if x.iloc[-1] < x.iloc[0] else 0)
        ).iloc[-1]
        
        divergence = "none"
        if current_trend > 0 and price_trend < 0:
            divergence = "bullish"
        elif current_trend < 0 and price_trend > 0:
            divergence = "bearish"

        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": current_price,
            "current_volume": current_volume,
            "obv": current_obv,
            "obv_signal_line": current_signal,
            "obv_momentum": current_momentum,
            "obv_trend": current_trend,
            "signal": signal,
            "divergence": divergence,
            "volume_significant": bool(significant_volume.iloc[-1]),
            "parameters": {
                "smoothing_period": self.param["smoothing_period"],
                "signal_line_period": signal_period,
                "momentum_period": momentum_period,
                "trend_confirmation_period": trend_period,
                "volume_threshold": volume_threshold,
                "breakout_threshold": breakout_threshold
            }
        }
        
        # Add result to analysis summary
        indicator_result = IndicatorResult(
            name="OBV",
            signal=result["signal"],
            value=result["obv"],
            strength="strong" if "strong" in result["signal"] else "medium",
            metadata={
                "flow_direction": result.get("flow_direction", "unknown"),
                "obv_ma": result.get("obv_ma", result["obv_signal_line"]),
                "obv_divergence": result.get("obv_divergence", result["divergence"]),
                "volume_acceleration": result.get("volume_acceleration", result["obv_momentum"]),
                "trend_strength": result.get("trend_strength", result["obv_trend"]),
                "volume_breakout": result.get("volume_breakout", False),
                "parameters": result["parameters"]
            }
        )
        add_indicator_result(indicator_result)
        
        return result
