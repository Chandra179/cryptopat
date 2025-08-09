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

class KeltnerChannel:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.rules = {
            "ema_period": 20,
            "atr_period": 10,
            "multiplier": 2.0,
            "min_periods": max(20, 10)
        }
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.ticker = ticker
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
    
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        if len(prices) < period:
            return [None] * len(prices)
        
        ema_values = [None] * len(prices)
        multiplier = 2 / (period + 1)
        
        ema_values[period - 1] = sum(prices[:period]) / period
        
        for i in range(period, len(prices)):
            ema_values[i] = (prices[i] * multiplier) + (ema_values[i-1] * (1 - multiplier))
        
        return ema_values
    
    def calculate_true_range(self, high: float, low: float, prev_close: float) -> float:
        return max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
    
    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int) -> List[float]:
        if len(highs) < period + 1:
            return [None] * len(highs)
        
        true_ranges = [None]
        for i in range(1, len(highs)):
            tr = self.calculate_true_range(highs[i], lows[i], closes[i-1])
            true_ranges.append(tr)
        
        atr_values = [None] * len(highs)
        
        first_atr = sum(true_ranges[1:period+1]) / period
        atr_values[period] = first_atr
        
        multiplier = 1 / period
        for i in range(period + 1, len(highs)):
            atr_values[i] = (true_ranges[i] * multiplier) + (atr_values[i-1] * (1 - multiplier))
        
        return atr_values
    
    def calculate(self):
        if len(self.ohlcv) < self.rules["min_periods"]:
            result = {
                "error": f"Insufficient data. Need at least {self.rules['min_periods']} periods",
                "data_length": len(self.ohlcv)
            }
            self.print_output(result)
            return
        
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        closes = df['close'].tolist()
        highs = df['high'].tolist()
        lows = df['low'].tolist()
        
        ema_values = self.calculate_ema(closes, self.rules["ema_period"])
        atr_values = self.calculate_atr(highs, lows, closes, self.rules["atr_period"])
        
        upper_band = []
        lower_band = []
        middle_line = []
        
        for i in range(len(df)):
            if ema_values[i] is not None and atr_values[i] is not None:
                middle = ema_values[i]
                atr_offset = atr_values[i] * self.rules["multiplier"]
                
                middle_line.append(middle)
                upper_band.append(middle + atr_offset)
                lower_band.append(middle - atr_offset)
            else:
                middle_line.append(None)
                upper_band.append(None)
                lower_band.append(None)
        
        current_price = closes[-1] if closes else None
        current_upper = upper_band[-1] if upper_band[-1] is not None else None
        current_lower = lower_band[-1] if lower_band[-1] is not None else None
        current_middle = middle_line[-1] if middle_line[-1] is not None else None
        
        position = None
        if current_price and current_upper and current_lower:
            if current_price > current_upper:
                position = "Above Upper Band"
            elif current_price < current_lower:
                position = "Below Lower Band"
            else:
                position = "Within Channel"
        
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": current_price,
            "keltner_channel": {
                "upper_band": current_upper,
                "middle_line": current_middle,
                "lower_band": current_lower,
                "position": position,
                "band_width": current_upper - current_lower if current_upper and current_lower else None
            },
            "parameters": {
                "ema_period": self.rules["ema_period"],
                "atr_period": self.rules["atr_period"],
                "multiplier": self.rules["multiplier"]
            },
            "analysis": {
                "trend_direction": "Bullish" if current_price and current_middle and current_price > current_middle else "Bearish",
                "volatility": "High" if current_upper and current_lower and (current_upper - current_lower) > (current_middle * 0.1) else "Low"
            }
        }
        
        self.print_output(result)
    
    def print_output(self, result: dict):
        print("\n" + "="*50)
        print("KELTNER CHANNEL ANALYSIS")
        print("="*50)
        
        if "error" in result:
            print(f"ERROR: {result['error']}")
            print(f"Data Length: {result['data_length']}")
            return
        
        print(f"Symbol: {result['symbol']}")
        print(f"Timeframe: {result['timeframe']}")
        print(f"Current Price: ${result['current_price']:.6f}")
        
        kc = result['keltner_channel']
        print(f"\nKeltner Channel Values:")
        print(f"  Upper Band:  ${kc['upper_band']:.6f}")
        print(f"  Middle Line: ${kc['middle_line']:.6f}")
        print(f"  Lower Band:  ${kc['lower_band']:.6f}")
        print(f"  Position: {kc['position']}")
        print(f"  Band Width: ${kc['band_width']:.6f}")
        
        params = result['parameters']
        print(f"\nParameters:")
        print(f"  EMA Period: {params['ema_period']}")
        print(f"  ATR Period: {params['atr_period']}")
        print(f"  Multiplier: {params['multiplier']}")
        
        analysis = result['analysis']
        print(f"\nAnalysis:")
        print(f"  Trend Direction: {analysis['trend_direction']}")
        print(f"  Volatility: {analysis['volatility']}")