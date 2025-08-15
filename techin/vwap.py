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
import numpy as np


class VWAP:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.param = {
            # VWAP Standard Parameters (Source: TradingView, Investopedia, Wikipedia)
            "price_source": "hlc3",              # Typical Price (High + Low + Close) / 3 (standard)
            "session_reset": "daily",           # Reset VWAP calculation (daily, weekly, monthly)
            "volume_threshold": 0,              # Minimum volume threshold for inclusion
            "periods_lookback": None,           # Number of periods for rolling VWAP (None = session-based)
            
            # Alternative Price Sources
            "use_ohlc4": False,                 # Use (Open + High + Low + Close) / 4
            "use_close": False,                 # Use Close price only
            "use_hl2": False,                   # Use (High + Low) / 2
            
            # Analysis Parameters
            "deviation_bands": True,            # Calculate VWAP deviation bands
            "std_dev_multiplier": [1.0, 2.0],  # Standard deviation multipliers for bands
            "volume_profile": False,            # Include volume profile analysis
            "session_high_low": True,           # Track session high/low vs VWAP
            
            # Signal Parameters
            "trend_confirmation_periods": 5,    # Periods for trend confirmation
            "volume_spike_threshold": 1.5,     # Volume spike multiplier
            "price_deviation_threshold": 0.02  # Price deviation threshold (2%)
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
        Calculate VWAP (Volume Weighted Average Price) according to standard methodology.
        
        Formula (Source: Investopedia, TradingView, Wikipedia):
        VWAP = Σ(Typical Price × Volume) / Σ(Volume)
        
        Where:
        - Typical Price = (High + Low + Close) / 3 (standard)
        - Alternative: OHLC4 = (Open + High + Low + Close) / 4
        - Summation is typically done from session start (market open)
        
        Key Characteristics:
        - Acts as both support and resistance level
        - Used by institutional traders for execution benchmarking
        - Resets at the beginning of each trading session
        - Higher timeframes provide stronger VWAP levels
        
        References:
        - Investopedia: https://www.investopedia.com/terms/v/vwap.asp
        - TradingView: https://www.tradingview.com/support/solutions/43000501613-volume-weighted-average-price-vwap/
        - Wikipedia: https://en.wikipedia.org/wiki/Volume-weighted_average_price
        - CFA Institute: Volume-Weighted Average Price as execution benchmark
        
        Trading Applications:
        - Price above VWAP: Bullish bias (buyers in control)
        - Price below VWAP: Bearish bias (sellers in control)
        - VWAP acts as dynamic support/resistance
        - Volume spikes near VWAP indicate institutional interest
        """
        if not self.ohlcv or len(self.ohlcv) < 2:
            return {
                "error": f"Insufficient data: need at least 2 candles, got {len(self.ohlcv) if self.ohlcv else 0}"
            }
            
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # Calculate typical price based on selected source
        if self.param["use_ohlc4"]:
            typical_price = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            price_source_name = "OHLC4"
        elif self.param["use_close"]:
            typical_price = df['close']
            price_source_name = "Close"
        elif self.param["use_hl2"]:
            typical_price = (df['high'] + df['low']) / 2
            price_source_name = "HL2"
        else:  # HLC3 (default)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            price_source_name = "HLC3"
        
        # Filter by volume threshold if specified
        volume_mask = df['volume'] >= self.param["volume_threshold"]
        filtered_volume = df['volume'].where(volume_mask, 0)
        filtered_typical_price = typical_price.where(volume_mask, 0)
        
        # Calculate cumulative values for VWAP
        df['price_volume'] = filtered_typical_price * filtered_volume
        df['cumulative_pv'] = df['price_volume'].cumsum()
        df['cumulative_volume'] = filtered_volume.cumsum()
        
        # Calculate VWAP
        df['vwap'] = df['cumulative_pv'] / df['cumulative_volume']
        
        # Handle division by zero
        df['vwap'] = df['vwap'].fillna(typical_price)
        
        # Calculate rolling VWAP if periods_lookback is specified
        if self.param["periods_lookback"]:
            periods = self.param["periods_lookback"]
            rolling_pv = (filtered_typical_price * filtered_volume).rolling(window=periods).sum()
            rolling_volume = filtered_volume.rolling(window=periods).sum()
            df['rolling_vwap'] = rolling_pv / rolling_volume
            df['rolling_vwap'] = df['rolling_vwap'].fillna(typical_price)
        
        # Calculate VWAP deviation bands if enabled
        upper_bands = []
        lower_bands = []
        if self.param["deviation_bands"]:
            for multiplier in self.param["std_dev_multiplier"]:
                # Calculate price deviation from VWAP
                price_diff = typical_price - df['vwap']
                weighted_variance = ((price_diff ** 2) * filtered_volume).cumsum() / df['cumulative_volume']
                weighted_std = np.sqrt(weighted_variance.fillna(0))
                
                upper_band = df['vwap'] + (weighted_std * multiplier)
                lower_band = df['vwap'] - (weighted_std * multiplier)
                
                upper_bands.append(float(upper_band.iloc[-1]))
                lower_bands.append(float(lower_band.iloc[-1]))
        
        # Current values
        current_price = float(df['close'].iloc[-1])
        current_vwap = float(df['vwap'].iloc[-1])
        current_volume = float(df['volume'].iloc[-1])
        current_typical_price = float(typical_price.iloc[-1])
        
        # Calculate session statistics
        session_high = float(df['high'].max())
        session_low = float(df['low'].min())
        total_volume = float(df['volume'].sum())
        avg_volume = float(df['volume'].mean())
        
        # Price position relative to VWAP
        price_above_vwap = current_price > current_vwap
        price_deviation_pct = ((current_price - current_vwap) / current_vwap) * 100
        
        # Volume analysis
        volume_spike = current_volume > (avg_volume * self.param["volume_spike_threshold"])
        
        # Trend analysis
        trend_periods = min(self.param["trend_confirmation_periods"], len(df))
        if trend_periods > 1:
            recent_vwap = df['vwap'].iloc[-trend_periods:]
            vwap_slope = (recent_vwap.iloc[-1] - recent_vwap.iloc[0]) / (trend_periods - 1)
            vwap_trend = "bullish" if vwap_slope > 0 else "bearish" if vwap_slope < 0 else "neutral"
        else:
            vwap_slope = 0
            vwap_trend = "neutral"
        
        # Signal generation
        signal = "neutral"
        if abs(price_deviation_pct) > (self.param["price_deviation_threshold"] * 100):
            if price_above_vwap:
                signal = "bullish" if volume_spike else "overbought"
            else:
                signal = "bearish" if volume_spike else "oversold"
        elif price_above_vwap and vwap_trend == "bullish":
            signal = "bullish"
        elif not price_above_vwap and vwap_trend == "bearish":
            signal = "bearish"
        
        # Build result
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": current_price,
            "vwap": current_vwap,
            "typical_price": current_typical_price,
            "price_source": price_source_name,
            "price_above_vwap": price_above_vwap,
            "price_deviation_pct": price_deviation_pct,
            "volume_spike": volume_spike,
            "signal": signal,
            "trend": vwap_trend,
            "vwap_slope": float(vwap_slope),
            "session_stats": {
                "high": session_high,
                "low": session_low,
                "total_volume": total_volume,
                "avg_volume": avg_volume,
                "current_volume": current_volume
            },
            "parameters": {
                "price_source": self.param["price_source"],
                "volume_threshold": self.param["volume_threshold"],
                "deviation_bands": self.param["deviation_bands"],
                "periods_lookback": self.param["periods_lookback"]
            }
        }
        
        # Add rolling VWAP if calculated
        if self.param["periods_lookback"]:
            result["rolling_vwap"] = float(df['rolling_vwap'].iloc[-1])
        
        # Add deviation bands if calculated
        if self.param["deviation_bands"] and upper_bands and lower_bands:
            result["deviation_bands"] = {
                "upper_bands": upper_bands,
                "lower_bands": lower_bands,
                "multipliers": self.param["std_dev_multiplier"]
            }
        
        return result