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

class Supertrend:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.param = {
            "atr_period": 10,
            "multiplier": 3.0,
            "atr_formula": lambda high_low_close: sum(max(h-l, abs(h-c_prev), abs(l-c_prev)) for h, l, c_prev in high_low_close) / len(high_low_close)
        }
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
    
    def calculate(self):
        """
        Calculate Supertrend according to TradingView methodology.
        """
        if len(self.ohlcv) < self.param["atr_period"]:
            result = {"error": f"Not enough data points. Need at least {self.param['atr_period']}, got {len(self.ohlcv)}"}
            self.print_output(result)
            return

        atr_values = []
        supertrend_values = []
        trend_direction = []

        for i in range(len(self.ohlcv)):
            timestamp, open_price, high, low, close, volume = self.ohlcv[i]
            
            if i >= self.param["atr_period"] - 1:
                atr_data = []
                for j in range(i - self.param["atr_period"] + 1, i + 1):
                    if j == 0:
                        prev_close = self.ohlcv[j][1]  # Use open as prev close for first candle
                    else:
                        prev_close = self.ohlcv[j-1][4]
                    
                    current_high = self.ohlcv[j][2]
                    current_low = self.ohlcv[j][3]
                    
                    true_range = max(
                        current_high - current_low,
                        abs(current_high - prev_close),
                        abs(current_low - prev_close)
                    )
                    atr_data.append(true_range)
                
                atr = sum(atr_data) / len(atr_data)
                atr_values.append(atr)
                
                hl2 = (high + low) / 2
                upper_band = hl2 + (self.param["multiplier"] * atr)
                lower_band = hl2 - (self.param["multiplier"] * atr)
                
                if i == self.param["atr_period"] - 1:
                    supertrend = lower_band if close > hl2 else upper_band
                    direction = "up" if close > hl2 else "down"
                else:
                    prev_supertrend = supertrend_values[-1]
                    prev_direction = trend_direction[-1]
                    
                    if prev_direction == "up":
                        supertrend = lower_band if lower_band > prev_supertrend else prev_supertrend
                        if close < supertrend:
                            direction = "down"
                            supertrend = upper_band
                        else:
                            direction = "up"
                    else:
                        supertrend = upper_band if upper_band < prev_supertrend else prev_supertrend
                        if close > supertrend:
                            direction = "up"
                            supertrend = lower_band
                        else:
                            direction = "down"
                
                supertrend_values.append(supertrend)
                trend_direction.append(direction)

        current_price = self.ohlcv[-1][4]
        current_supertrend = supertrend_values[-1] if supertrend_values else None
        current_direction = trend_direction[-1] if trend_direction else None
        
        signal = "BUY" if current_direction == "up" and current_price > current_supertrend else "SELL"
        
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": current_price,
            "supertrend_value": current_supertrend,
            "trend_direction": current_direction,
            "signal": signal,
            "atr_period": self.param["atr_period"],
            "multiplier": self.param["multiplier"],
            "total_periods": len(supertrend_values)
        }
        
        self.print_output(result)
    
    def print_output(self, result: dict):
        """Print the output"""
        print(f"\n{'='*50}")
        print(f"SUPERTREND ANALYSIS - {result.get('symbol', 'N/A')} ({result.get('timeframe', 'N/A')})")
        print(f"{'='*50}")
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print(f"📊 Current Price: ${result['current_price']:.4f}")
        print(f"📈 Supertrend Value: ${result['supertrend_value']:.4f}")
        print(f"🎯 Trend Direction: {result['trend_direction'].upper()}")
        print(f"🚨 Signal: {result['signal']}")
        print(f"⚙️  ATR Period: {result['atr_period']}")
        print(f"⚙️  Multiplier: {result['multiplier']}")
        print(f"📋 Analysis Periods: {result['total_periods']}")
        
        if result['signal'] == 'BUY':
            print(f"🟢 Price is above Supertrend - Bullish trend")
        else:
            print(f"🔴 Price is below Supertrend - Bearish trend")