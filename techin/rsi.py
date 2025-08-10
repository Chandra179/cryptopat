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

import logging
from typing import List, Dict
import numpy as np

class RSI:
    
    def __init__(self, 
                 symbol: str,
                 timeframe: str,
                 limit: int,
                 ob: dict,
                 ticker: dict,            
                 ohlcv: List[List],       
                 trades: List[Dict]):    
        self.rules = {
            "rsi_period": 14,
            "overbought_level": 70,
            "oversold_level": 30,
            "midline": 50,
            "min_volume_confirmation": 1.2,
            "divergence_lookback": 5,
            "support_resistance_threshold": 0.02
        }
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.ticker = ticker
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.logger = logging.getLogger(__name__)
    
    def calculate(self):
        """
        Calculate RSI according to TradingView methodology.
        """
        if len(self.ohlcv) < self.rules["rsi_period"] + 1:
            self.logger.warning(f"Insufficient data for RSI calculation. Need at least {self.rules['rsi_period'] + 1} periods")
            return
        
        closes = [candle[4] for candle in self.ohlcv]
        volumes = [candle[5] for candle in self.ohlcv]
        
        rsi_values = self._calculate_rsi(closes)
        
        if not rsi_values:
            return
        
        current_rsi = rsi_values[-1]
        current_price = closes[-1]
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else np.mean(volumes)
        
        result = {
            "current_rsi": round(current_rsi, 2),
            "current_price": current_price,
            "market_condition": self._get_market_condition(current_rsi),
            "volume_confirmation": current_volume > avg_volume * self.rules["min_volume_confirmation"],
            "divergences": self._detect_divergences(closes, rsi_values),
            "support_resistance": self._check_support_resistance(closes),
            "midline_cross": self._check_midline_cross(rsi_values),
            "signal_strength": self._calculate_signal_strength(current_rsi, current_volume, avg_volume)
        }
        
        self.print_output(result)
        return result
    
    def _calculate_rsi(self, closes):
        """Calculate RSI using the standard formula"""
        if len(closes) < self.rules["rsi_period"] + 1:
            return []
        
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:self.rules["rsi_period"]])
        avg_loss = np.mean(losses[:self.rules["rsi_period"]])
        
        rsi_values = []
        
        for i in range(self.rules["rsi_period"], len(gains)):
            if i == self.rules["rsi_period"]:
                rs = avg_gain / avg_loss if avg_loss != 0 else 0
            else:
                avg_gain = (avg_gain * (self.rules["rsi_period"] - 1) + gains[i]) / self.rules["rsi_period"]
                avg_loss = (avg_loss * (self.rules["rsi_period"] - 1) + losses[i]) / self.rules["rsi_period"]
                rs = avg_gain / avg_loss if avg_loss != 0 else 0
            
            rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
        
        return rsi_values
    
    def _get_market_condition(self, rsi):
        """Determine market condition based on RSI level"""
        if rsi >= self.rules["overbought_level"]:
            return "OVERBOUGHT"
        elif rsi <= self.rules["oversold_level"]:
            return "OVERSOLD"
        elif rsi > self.rules["midline"]:
            return "BULLISH"
        else:
            return "BEARISH"
    
    def _detect_divergences(self, closes, rsi_values):
        """Detect bullish and bearish divergences"""
        if len(rsi_values) < self.rules["divergence_lookback"] * 2:
            return {"bullish": False, "bearish": False}
        
        lookback = self.rules["divergence_lookback"]
        
        recent_price_high = max(closes[-lookback:])
        recent_price_low = min(closes[-lookback:])
        previous_price_high = max(closes[-lookback*2:-lookback])
        previous_price_low = min(closes[-lookback*2:-lookback])
        
        recent_rsi_high = max(rsi_values[-lookback:])
        recent_rsi_low = min(rsi_values[-lookback:])
        previous_rsi_high = max(rsi_values[-lookback*2:-lookback])
        previous_rsi_low = min(rsi_values[-lookback*2:-lookback])
        
        bullish_divergence = (recent_price_low < previous_price_low and 
                            recent_rsi_low > previous_rsi_low)
        
        bearish_divergence = (recent_price_high > previous_price_high and 
                            recent_rsi_high < previous_rsi_high)
        
        return {
            "bullish": bullish_divergence,
            "bearish": bearish_divergence
        }
    
    def _check_support_resistance(self, closes):
        """Check if price is breaking support or resistance"""
        if len(closes) < 20:
            return {"support_break": False, "resistance_break": False}
        
        current_price = closes[-1]
        recent_high = max(closes[-20:])
        recent_low = min(closes[-20:])
        
        resistance_break = current_price > recent_high * (1 - self.rules["support_resistance_threshold"])
        support_break = current_price < recent_low * (1 + self.rules["support_resistance_threshold"])
        
        return {
            "support_break": support_break,
            "resistance_break": resistance_break,
            "support_level": recent_low,
            "resistance_level": recent_high
        }
    
    def _check_midline_cross(self, rsi_values):
        """Check for midline crosses for confirmation"""
        if len(rsi_values) < 2:
            return {"direction": "NONE", "confirmed": False}
        
        current_rsi = rsi_values[-1]
        previous_rsi = rsi_values[-2]
        midline = self.rules["midline"]
        
        if previous_rsi <= midline and current_rsi > midline:
            return {"direction": "BULLISH_CROSS", "confirmed": True}
        elif previous_rsi >= midline and current_rsi < midline:
            return {"direction": "BEARISH_CROSS", "confirmed": True}
        
        return {"direction": "NONE", "confirmed": False}
    
    def _calculate_signal_strength(self, rsi, current_volume, avg_volume):
        """Calculate overall signal strength"""
        strength = 0
        
        if rsi >= self.rules["overbought_level"] or rsi <= self.rules["oversold_level"]:
            strength += 3
        elif rsi > 60 or rsi < 40:
            strength += 2
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > self.rules["min_volume_confirmation"]:
            strength += 2
        elif volume_ratio > 1:
            strength += 1
        
        return min(strength, 5)
    
    def print_output(self, result: dict):
        """Print the RSI analysis output"""
        print(f"\n{'='*50}")
        print(f"RSI ANALYSIS - {self.symbol} ({self.timeframe})")
        print(f"{'='*50}")
        
        print(f"Current RSI: {result['current_rsi']}")
        print(f"Market Condition: {result['market_condition']}")
        print(f"Signal Strength: {'★' * result['signal_strength']} ({result['signal_strength']}/5)")
        
        if result['volume_confirmation']:
            print(f"✓ Volume Confirmation: HIGH VOLUME")
        else:
            print(f"✗ Volume Confirmation: LOW VOLUME")
        
        divergences = result['divergences']
        if divergences['bullish']:
            print(f"🔵 BULLISH DIVERGENCE DETECTED")
        if divergences['bearish']:
            print(f"🔴 BEARISH DIVERGENCE DETECTED")
        
        midline_cross = result['midline_cross']
        if midline_cross['confirmed']:
            print(f"📈 MIDLINE CROSS: {midline_cross['direction']}")
        
        sr = result['support_resistance']
        if sr['support_break']:
            print(f"⬇️ SUPPORT BREAK at {sr['support_level']}")
        if sr['resistance_break']:
            print(f"⬆️ RESISTANCE BREAK at {sr['resistance_level']}")
        
        if result['current_rsi'] >= self.rules['overbought_level']:
            print(f"⚠️ OVERBOUGHT: Consider selling opportunities")
        elif result['current_rsi'] <= self.rules['oversold_level']:
            print(f"⚠️ OVERSOLD: Consider buying opportunities")
        elif result['current_rsi'] > self.rules['midline']:
            print(f"📊 BULLISH TERRITORY: Above midline ({self.rules['midline']})")
        else:
            print(f"📊 BEARISH TERRITORY: Below midline ({self.rules['midline']})")