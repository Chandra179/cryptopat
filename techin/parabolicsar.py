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

class ParabolicSAR:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.param = {
            "initial_af": 0.02,
            "af_increment": 0.02,
            "max_af": 0.20,
        }
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
    
    def calculate(self):
        """
        Calculate Parabolic SAR according to TradingView methodology.
        """
        if len(self.ohlcv) < 2:
            print("Insufficient data for Parabolic SAR calculation")
            return
        
        result = {}
        sar_values = []
        
        # Initialize variables
        af = self.param["initial_af"]
        trend = 1  # 1 for uptrend, -1 for downtrend
        
        # First candle - determine initial trend
        high_0 = self.ohlcv[0][2]  # High
        low_0 = self.ohlcv[0][3]   # Low
        close_0 = self.ohlcv[0][4] # Close
        
        high_1 = self.ohlcv[1][2]  # High
        low_1 = self.ohlcv[1][3]   # Low
        close_1 = self.ohlcv[1][4] # Close
        
        # Determine initial trend direction
        if close_1 > close_0:
            trend = 1  # Uptrend
            sar = low_0  # Start SAR at previous low
            ep = high_1  # Extreme point is current high
        else:
            trend = -1  # Downtrend
            sar = high_0  # Start SAR at previous high
            ep = low_1   # Extreme point is current low
        
        sar_values.append(sar)
        
        # Calculate SAR for remaining periods
        for i in range(1, len(self.ohlcv)):
            high = self.ohlcv[i][2]
            low = self.ohlcv[i][3]
            
            # Calculate new SAR
            new_sar = sar + af * (ep - sar)
            
            if trend == 1:  # Uptrend
                # SAR cannot be above the low of current or previous period
                new_sar = min(new_sar, low, self.ohlcv[i-1][3])
                
                # Check for trend reversal
                if low <= new_sar:
                    # Trend reversal - switch to downtrend
                    trend = -1
                    new_sar = ep  # SAR becomes the extreme point
                    ep = low      # New extreme point is current low
                    af = self.param["initial_af"]  # Reset acceleration factor
                else:
                    # Continue uptrend
                    if high > ep:
                        ep = high  # Update extreme point
                        af = min(af + self.param["af_increment"], self.param["max_af"])
            
            else:  # Downtrend
                # SAR cannot be below the high of current or previous period
                new_sar = max(new_sar, high, self.ohlcv[i-1][2])
                
                # Check for trend reversal
                if high >= new_sar:
                    # Trend reversal - switch to uptrend
                    trend = 1
                    new_sar = ep  # SAR becomes the extreme point
                    ep = high     # New extreme point is current high
                    af = self.param["initial_af"]  # Reset acceleration factor
                else:
                    # Continue downtrend
                    if low < ep:
                        ep = low   # Update extreme point
                        af = min(af + self.param["af_increment"], self.param["max_af"])
            
            sar = new_sar
            sar_values.append(sar)
        
        # Analyze current signals
        current_sar = sar_values[-1]
        current_price = self.ohlcv[-1][4]  # Current close price
        previous_sar = sar_values[-2] if len(sar_values) > 1 else current_sar
        
        # Determine signal
        if trend == 1:
            signal = "BUY" if current_price > current_sar else "NEUTRAL"
        else:
            signal = "SELL" if current_price < current_sar else "NEUTRAL"
        
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_sar": round(current_sar, 8),
            "current_price": round(current_price, 8),
            "trend": "UPTREND" if trend == 1 else "DOWNTREND",
            "signal": signal,
            "acceleration_factor": round(af, 4),
            "extreme_point": round(ep, 8),
            "sar_distance": round(abs(current_price - current_sar), 8),
            "sar_distance_percent": round(abs(current_price - current_sar) / current_price * 100, 4),
            "recent_sar_values": [round(val, 8) for val in sar_values[-5:]]
        }
        
        self.print_output(result)
        return result
    
    def print_output(self, result: dict):
        """ print the output """
        print("\n" + "="*50)
        print(f"PARABOLIC SAR ANALYSIS - {result['symbol']} ({result['timeframe']})")
        print("="*50)
        print(f"Current Price: ${result['current_price']}")
        print(f"Current SAR: ${result['current_sar']}")
        print(f"Trend: {result['trend']}")
        print(f"Signal: {result['signal']}")
        print(f"SAR Distance: ${result['sar_distance']} ({result['sar_distance_percent']}%)")
        print(f"Acceleration Factor: {result['acceleration_factor']}")
        print(f"Extreme Point: ${result['extreme_point']}")
        print(f"Recent SAR Values: {result['recent_sar_values']}")
        
        if result['signal'] == "BUY":
            print("\n📈 BULLISH: Price is above SAR - Consider long positions")
        elif result['signal'] == "SELL":
            print("\n📉 BEARISH: Price is below SAR - Consider short positions")
        else:
            print("\n⚠️  NEUTRAL: Price near SAR level - Wait for clear breakout")