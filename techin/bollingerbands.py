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


class BollingerBands:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.rules = {
            "period": 20,
            "std_dev_multiplier": 2.0,
            "price_source": "close"
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
        Calculate Bollinger Bands according to TradingView methodology.
        """
        if not self.ohlcv or len(self.ohlcv) < self.rules["period"]:
            result = {
                "error": f"Insufficient data: need at least {self.rules['period']} candles, got {len(self.ohlcv) if self.ohlcv else 0}"
            }
            self.print_output(result)
            return
            
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = pd.to_numeric(df['close'])
        
        period = self.rules["period"]
        multiplier = self.rules["std_dev_multiplier"]
        
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        upper_band = sma + (std * multiplier)
        lower_band = sma - (std * multiplier)
        
        current_price = float(df['close'].iloc[-1])
        current_sma = float(sma.iloc[-1])
        current_upper = float(upper_band.iloc[-1])
        current_lower = float(lower_band.iloc[-1])
        
        band_width = ((current_upper - current_lower) / current_sma) * 100
        position_in_bands = ((current_price - current_lower) / (current_upper - current_lower)) * 100
        
        squeeze = band_width < 10
        expansion = band_width > 25
        
        if current_price > current_upper:
            signal = "OVERBOUGHT"
        elif current_price < current_lower:
            signal = "OVERSOLD"
        elif current_price > current_sma:
            signal = "BULLISH"
        else:
            signal = "BEARISH"
        
        # Calculate additional metrics for the new structure
        sma_slope = 0
        if len(sma) > 1:
            sma_slope = float(sma.iloc[-1] - sma.iloc[-2])
        
        position_from_sma_pct = ((current_price - current_sma) / current_sma) * 100
        squeeze_level = 9.5  # threshold for squeeze detection
        expansion_level = 25.0  # threshold for expansion detection
        
        # Determine trend bias
        if current_price > current_sma and sma_slope > 0:
            trend_bias = "UPTREND"
        elif current_price < current_sma and sma_slope < 0:
            trend_bias = "DOWNTREND"
        else:
            trend_bias = "SIDEWAYS"
        
        # Calculate confidence percentage based on multiple factors
        confidence = 50  # base confidence
        if abs(position_in_bands - 50) > 20:  # strong position in bands
            confidence += 15
        if abs(position_from_sma_pct) > 2:  # significant deviation from SMA
            confidence += 10
        if expansion and band_width > expansion_level:  # strong expansion
            confidence += 25
        confidence = min(confidence, 100)
        
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": int(df['timestamp'].iloc[-1]),
            "parameters": {
                "lookback_period": self.rules["period"],
                "std_dev_multiplier": self.rules["std_dev_multiplier"]
            },
            "price_metrics": {
                "current_price": round(current_price, 2),
                "sma": round(current_sma, 2),
                "sma_slope": round(sma_slope, 1),
                "upper_band": round(current_upper, 2),
                "lower_band": round(current_lower, 2),
                "position_in_bands_pct": round(position_in_bands, 1),
                "position_from_sma_pct": round(position_from_sma_pct, 1)
            },
            "volatility_metrics": {
                "band_width_pct": round(band_width, 2),
                "squeeze": squeeze,
                "expansion": expansion,
                "squeeze_level": squeeze_level,
                "expansion_level": expansion_level
            },
            "signal_metrics": {
                "signal": signal,
                "trend_bias": trend_bias,
                "confidence_pct": round(confidence, 1)
            }
        }
        
        self.print_output(result)
    
    def print_output(self, result: dict):
        """Print the Bollinger Bands analysis output"""
        if "error" in result:
            print(f"❌ Bollinger Bands Error: {result['error']}")
            return
            
        pm = result['price_metrics']
        vm = result['volatility_metrics']
        sm = result['signal_metrics']
        
        print(f"\n🎯 Bollinger Bands Analysis - {result['symbol']} ({result['timeframe']})")
        print(f"📊 Current Price: ${pm['current_price']:.2f}")
        print(f"📈 Upper Band: ${pm['upper_band']:.2f}")
        print(f"📊 SMA ({result['parameters']['lookback_period']}): ${pm['sma']:.2f} (slope: {pm['sma_slope']:.1f})")
        print(f"📉 Lower Band: ${pm['lower_band']:.2f}")
        print(f"📏 Band Width: {vm['band_width_pct']:.2f}%")
        print(f"📍 Position in Bands: {pm['position_in_bands_pct']:.1f}%")
        print(f"📍 Position from SMA: {pm['position_from_sma_pct']:+.1f}%")
        print(f"🔄 Squeeze: {'Yes' if vm['squeeze'] else 'No'} (threshold: {vm['squeeze_level']}%)")
        print(f"💥 Expansion: {'Yes' if vm['expansion'] else 'No'} (threshold: {vm['expansion_level']}%)")
        print(f"🚦 Signal: {sm['signal']} | Trend: {sm['trend_bias']} | Confidence: {sm['confidence_pct']:.1f}%") 