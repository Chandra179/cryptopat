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

class Renko:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.param = {
            "brick_size_percentage": 0.01,
            "atr_multiplier": 2.0,
            "min_bricks_for_trend": 3,
            "reversal_threshold": 2,
            "price_precision": 2
        }
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
    
    def calculate_atr(self, period=14):
        """Calculate Average True Range for dynamic brick sizing"""
        if len(self.ohlcv) < period:
            return None
        
        true_ranges = []
        for i in range(1, min(len(self.ohlcv), period + 1)):
            high = self.ohlcv[i][2]
            low = self.ohlcv[i][3]
            prev_close = self.ohlcv[i-1][4]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        return sum(true_ranges) / len(true_ranges) if true_ranges else None
    
    def calculate_brick_size(self):
        """Calculate optimal brick size using ATR or percentage method"""
        current_price = self.ohlcv[-1][4]
        
        atr = self.calculate_atr()
        if atr:
            return round(atr * self.param["atr_multiplier"], self.param["price_precision"])
        else:
            return round(current_price * self.param["brick_size_percentage"], self.param["price_precision"])
    
    def build_renko_bricks(self):
        """Convert OHLCV data to Renko bricks"""
        if len(self.ohlcv) < 2:
            return []
        
        brick_size = self.calculate_brick_size()
        bricks = []
        
        start_price = self.ohlcv[0][4]
        current_brick_level = round(start_price / brick_size) * brick_size
        
        for candle in self.ohlcv[1:]:
            high, low, close = candle[2], candle[3], candle[4]
            timestamp = candle[0]
            
            while close >= current_brick_level + brick_size:
                current_brick_level += brick_size
                bricks.append({
                    'timestamp': timestamp,
                    'open': current_brick_level - brick_size,
                    'close': current_brick_level,
                    'direction': 'up',
                    'brick_size': brick_size
                })
            
            while close <= current_brick_level - brick_size:
                current_brick_level -= brick_size
                bricks.append({
                    'timestamp': timestamp,
                    'open': current_brick_level + brick_size,
                    'close': current_brick_level,
                    'direction': 'down',
                    'brick_size': brick_size
                })
        
        return bricks
    
    def identify_trend(self, bricks):
        """Identify current trend based on consecutive brick direction"""
        if len(bricks) < self.param["min_bricks_for_trend"]:
            return "neutral"
        
        recent_bricks = bricks[-self.param["min_bricks_for_trend"]:]
        directions = [brick['direction'] for brick in recent_bricks]
        
        if all(d == 'up' for d in directions):
            return "uptrend"
        elif all(d == 'down' for d in directions):
            return "downtrend"
        else:
            return "sideways"
    
    def detect_reversals(self, bricks):
        """Detect potential trend reversals"""
        if len(bricks) < 4:
            return []
        
        reversals = []
        
        for i in range(self.param["reversal_threshold"], len(bricks)):
            current_direction = bricks[i]['direction']
            prev_directions = [bricks[j]['direction'] for j in range(i - self.param["reversal_threshold"], i)]
            
            if (current_direction == 'up' and all(d == 'down' for d in prev_directions)) or \
               (current_direction == 'down' and all(d == 'up' for d in prev_directions)):
                reversals.append({
                    'index': i,
                    'price': bricks[i]['close'],
                    'direction': current_direction,
                    'timestamp': bricks[i]['timestamp']
                })
        
        return reversals
    
    def calculate_support_resistance(self, bricks):
        """Calculate support and resistance levels from Renko bricks"""
        if not bricks:
            return {"support": [], "resistance": []}
        
        levels = {}
        for brick in bricks:
            price = brick['close']
            if price in levels:
                levels[price] += 1
            else:
                levels[price] = 1
        
        sorted_levels = sorted(levels.items(), key=lambda x: x[1], reverse=True)
        current_price = bricks[-1]['close']
        
        support = [level[0] for level in sorted_levels[:3] if level[0] < current_price]
        resistance = [level[0] for level in sorted_levels[:3] if level[0] > current_price]
        
        return {
            "support": sorted(support, reverse=True)[:2],
            "resistance": sorted(resistance)[:2]
        }
    
    def calculate(self):
        """
        Calculate Renko chart analysis according to TradingView methodology.
        """
        if len(self.ohlcv) < 2:
            result = {"error": "Insufficient data for Renko analysis"}
            self.print_output(result)
            return
        
        brick_size = self.calculate_brick_size()
        bricks = self.build_renko_bricks()
        
        if not bricks:
            result = {"error": "No Renko bricks generated"}
            self.print_output(result)
            return
        
        trend = self.identify_trend(bricks)
        reversals = self.detect_reversals(bricks)
        levels = self.calculate_support_resistance(bricks)
        
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": self.ohlcv[-1][4],
            "brick_size": brick_size,
            "total_bricks": len(bricks),
            "trend": trend,
            "last_brick_direction": bricks[-1]['direction'] if bricks else None,
            "recent_reversals": len(reversals),
            "support_levels": levels["support"],
            "resistance_levels": levels["resistance"],
            "trend_strength": {
                "consecutive_same_direction": self._count_consecutive_direction(bricks),
                "price_range": {
                    "high": max(brick['close'] for brick in bricks[-10:]) if len(bricks) >= 10 else None,
                    "low": min(brick['close'] for brick in bricks[-10:]) if len(bricks) >= 10 else None
                }
            },
            "signals": self._generate_signals(trend, bricks, reversals)
        }
        
        self.print_output(result)
    
    def _count_consecutive_direction(self, bricks):
        """Count consecutive bricks in the same direction"""
        if not bricks:
            return 0
        
        count = 1
        last_direction = bricks[-1]['direction']
        
        for i in range(len(bricks) - 2, -1, -1):
            if bricks[i]['direction'] == last_direction:
                count += 1
            else:
                break
        
        return count
    
    def _generate_signals(self, trend, bricks, reversals):
        """Generate trading signals based on Renko analysis"""
        signals = []
        
        if trend == "uptrend":
            signals.append("BULLISH - Strong uptrend detected")
        elif trend == "downtrend":
            signals.append("BEARISH - Strong downtrend detected")
        
        if reversals and len(reversals) > 0:
            latest_reversal = reversals[-1]
            if latest_reversal['direction'] == 'up':
                signals.append("REVERSAL - Potential bullish reversal")
            else:
                signals.append("REVERSAL - Potential bearish reversal")
        
        consecutive = self._count_consecutive_direction(bricks)
        if consecutive >= 5:
            signals.append(f"MOMENTUM - {consecutive} consecutive bricks in same direction")
        
        return signals
    
    def print_output(self, result: dict):
        """ print the output """
        print(f"\n{'='*50}")
        print(f"RENKO CHART ANALYSIS - {result.get('symbol', 'N/A')}")
        print(f"{'='*50}")
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"Current Price: ${result['current_price']:.2f}")
        print(f"Brick Size: ${result['brick_size']:.2f}")
        print(f"Total Bricks: {result['total_bricks']}")
        print(f"Trend: {result['trend'].upper()}")
        print(f"Last Brick: {result['last_brick_direction'].upper()}")
        print(f"Consecutive Same Direction: {result['trend_strength']['consecutive_same_direction']}")
        
        if result['support_levels']:
            print(f"Support Levels: {[f'${level:.2f}' for level in result['support_levels']]}")
        
        if result['resistance_levels']:
            print(f"Resistance Levels: {[f'${level:.2f}' for level in result['resistance_levels']]}")
        
        if result['signals']:
            print("\nSignals:")
            for signal in result['signals']:
                print(f"  • {signal}")