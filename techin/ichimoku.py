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

class IchimokuCloud:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.rules = {
            "tenkan_period": 9,
            "kijun_period": 26,
            "senkou_b_period": 52,
            "displacement": 26,
            "min_data_points": 52
        }
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
    
    def calculate(self):
        """
        Calculate Ichimoku Cloud according to TradingView methodology.
        """
        if len(self.ohlcv) < self.rules["min_data_points"]:
            result = {
                "error": f"Insufficient data points. Need at least {self.rules['min_data_points']}, got {len(self.ohlcv)}"
            }
            self.print_output(result)
            return
        
        # Extract high, low, close prices
        highs = [candle[2] for candle in self.ohlcv]
        lows = [candle[3] for candle in self.ohlcv]
        closes = [candle[4] for candle in self.ohlcv]
        
        # Calculate Ichimoku components
        tenkan_sen = self._calculate_tenkan_sen(highs, lows)
        kijun_sen = self._calculate_kijun_sen(highs, lows)
        senkou_span_a = self._calculate_senkou_span_a(tenkan_sen, kijun_sen)
        senkou_span_b = self._calculate_senkou_span_b(highs, lows)
        chikou_span = self._calculate_chikou_span(closes)
        
        # Current values (latest)
        current_price = closes[-1]
        current_tenkan = tenkan_sen[-1] if tenkan_sen else None
        current_kijun = kijun_sen[-1] if kijun_sen else None
        
        # Cloud analysis
        cloud_analysis = self._analyze_cloud(current_price, senkou_span_a, senkou_span_b)
        
        # Signal generation
        signals = self._generate_signals(current_price, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span)
        
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": current_price,
            "tenkan_sen": current_tenkan,
            "kijun_sen": current_kijun,
            "senkou_span_a": senkou_span_a[-1] if senkou_span_a else None,
            "senkou_span_b": senkou_span_b[-1] if senkou_span_b else None,
            "chikou_span": chikou_span[0] if chikou_span else None,
            "cloud_color": cloud_analysis["color"],
            "price_vs_cloud": cloud_analysis["position"],
            "cloud_thickness": cloud_analysis["thickness"],
            "signals": signals,
            "trend_strength": self._assess_trend_strength(tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b)
        }
        
        self.print_output(result)
    
    def _calculate_tenkan_sen(self, highs: List[float], lows: List[float]) -> List[float]:
        """Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2"""
        tenkan = []
        period = self.rules["tenkan_period"]
        
        for i in range(period - 1, len(highs)):
            period_high = max(highs[i - period + 1:i + 1])
            period_low = min(lows[i - period + 1:i + 1])
            tenkan.append((period_high + period_low) / 2)
        
        return tenkan
    
    def _calculate_kijun_sen(self, highs: List[float], lows: List[float]) -> List[float]:
        """Kijun-sen (Base Line): (26-period high + 26-period low) / 2"""
        kijun = []
        period = self.rules["kijun_period"]
        
        for i in range(period - 1, len(highs)):
            period_high = max(highs[i - period + 1:i + 1])
            period_low = min(lows[i - period + 1:i + 1])
            kijun.append((period_high + period_low) / 2)
        
        return kijun
    
    def _calculate_senkou_span_a(self, tenkan: List[float], kijun: List[float]) -> List[float]:
        """Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead"""
        if not tenkan or not kijun:
            return []
        
        min_len = min(len(tenkan), len(kijun))
        senkou_a = []
        
        for i in range(min_len):
            senkou_a.append((tenkan[i] + kijun[i]) / 2)
        
        return senkou_a
    
    def _calculate_senkou_span_b(self, highs: List[float], lows: List[float]) -> List[float]:
        """Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead"""
        senkou_b = []
        period = self.rules["senkou_b_period"]
        
        for i in range(period - 1, len(highs)):
            period_high = max(highs[i - period + 1:i + 1])
            period_low = min(lows[i - period + 1:i + 1])
            senkou_b.append((period_high + period_low) / 2)
        
        return senkou_b
    
    def _calculate_chikou_span(self, closes: List[float]) -> List[float]:
        """Chikou Span (Lagging Span): Current close price plotted 26 periods behind"""
        displacement = self.rules["displacement"]
        
        if len(closes) < displacement:
            return []
        
        return closes[-displacement:]
    
    def _analyze_cloud(self, current_price: float, senkou_a: List[float], senkou_b: List[float]) -> dict:
        """Analyze cloud color, thickness, and price position"""
        if not senkou_a or not senkou_b:
            return {"color": "unknown", "position": "unknown", "thickness": 0}
        
        current_a = senkou_a[-1]
        current_b = senkou_b[-1]
        
        # Cloud color (green if Senkou A > Senkou B, red otherwise)
        cloud_color = "green" if current_a > current_b else "red"
        
        # Price position relative to cloud
        cloud_top = max(current_a, current_b)
        cloud_bottom = min(current_a, current_b)
        
        if current_price > cloud_top:
            position = "above_cloud"
        elif current_price < cloud_bottom:
            position = "below_cloud"
        else:
            position = "inside_cloud"
        
        # Cloud thickness (distance between spans)
        thickness = abs(current_a - current_b)
        
        return {
            "color": cloud_color,
            "position": position,
            "thickness": thickness
        }
    
    def _generate_signals(self, current_price: float, tenkan: List[float], kijun: List[float], 
                         senkou_a: List[float], senkou_b: List[float], chikou: List[float]) -> dict:
        """Generate Ichimoku trading signals"""
        signals = {
            "tenkan_kijun_cross": "none",
            "price_cloud_breakout": "none",
            "chikou_confirmation": "none",
            "overall_signal": "none"
        }
        
        if not tenkan or not kijun or len(tenkan) < 2 or len(kijun) < 2:
            return signals
        
        # Tenkan-Kijun cross
        if tenkan[-1] > kijun[-1] and tenkan[-2] <= kijun[-2]:
            signals["tenkan_kijun_cross"] = "bullish"
        elif tenkan[-1] < kijun[-1] and tenkan[-2] >= kijun[-2]:
            signals["tenkan_kijun_cross"] = "bearish"
        
        # Price vs Cloud
        if senkou_a and senkou_b:
            cloud_top = max(senkou_a[-1], senkou_b[-1])
            cloud_bottom = min(senkou_a[-1], senkou_b[-1])
            
            if current_price > cloud_top:
                signals["price_cloud_breakout"] = "bullish"
            elif current_price < cloud_bottom:
                signals["price_cloud_breakout"] = "bearish"
        
        # Chikou confirmation
        if chikou and len(self.ohlcv) >= self.rules["displacement"]:
            chikou_price = chikou[0]
            historical_price = self.ohlcv[-(self.rules["displacement"] + 1)][4]  # Close price 26 periods ago
            
            if chikou_price > historical_price:
                signals["chikou_confirmation"] = "bullish"
            elif chikou_price < historical_price:
                signals["chikou_confirmation"] = "bearish"
        
        # Overall signal (all components must align)
        bullish_signals = sum(1 for signal in signals.values() if signal == "bullish")
        bearish_signals = sum(1 for signal in signals.values() if signal == "bearish")
        
        if bullish_signals >= 2 and bearish_signals == 0:
            signals["overall_signal"] = "strong_bullish"
        elif bullish_signals >= 1 and bearish_signals == 0:
            signals["overall_signal"] = "bullish"
        elif bearish_signals >= 2 and bullish_signals == 0:
            signals["overall_signal"] = "strong_bearish"
        elif bearish_signals >= 1 and bullish_signals == 0:
            signals["overall_signal"] = "bearish"
        else:
            signals["overall_signal"] = "neutral"
        
        return signals
    
    def _assess_trend_strength(self, tenkan: List[float], kijun: List[float], 
                              senkou_a: List[float], senkou_b: List[float]) -> str:
        """Assess overall trend strength based on line alignment"""
        if not all([tenkan, kijun, senkou_a, senkou_b]):
            return "unknown"
        
        current_tenkan = tenkan[-1]
        current_kijun = kijun[-1]
        current_senkou_a = senkou_a[-1]
        current_senkou_b = senkou_b[-1]
        
        # Perfect bullish alignment: Tenkan > Kijun > Senkou A > Senkou B
        if (current_tenkan > current_kijun > current_senkou_a > current_senkou_b):
            return "strong_bullish"
        
        # Perfect bearish alignment: Tenkan < Kijun < Senkou A < Senkou B
        if (current_tenkan < current_kijun < current_senkou_a < current_senkou_b):
            return "strong_bearish"
        
        # Partial alignments
        bullish_count = 0
        bearish_count = 0
        
        if current_tenkan > current_kijun:
            bullish_count += 1
        else:
            bearish_count += 1
        
        if current_senkou_a > current_senkou_b:
            bullish_count += 1
        else:
            bearish_count += 1
        
        if bullish_count > bearish_count:
            return "moderate_bullish"
        elif bearish_count > bullish_count:
            return "moderate_bearish"
        else:
            return "neutral"
    
    def print_output(self, result: dict):
        """Print the Ichimoku Cloud analysis output"""
        print("\n" + "="*60)
        print(f"ICHIMOKU CLOUD ANALYSIS - {result.get('symbol', 'N/A')} ({result.get('timeframe', 'N/A')})")
        print("="*60)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        # Current values
        print(f"Current Price: {result.get('current_price', 'N/A'):.4f}")
        print(f"Tenkan-sen (9): {result.get('tenkan_sen', 'N/A'):.4f}" if result.get('tenkan_sen') else "Tenkan-sen (9): N/A")
        print(f"Kijun-sen (26): {result.get('kijun_sen', 'N/A'):.4f}" if result.get('kijun_sen') else "Kijun-sen (26): N/A")
        print(f"Senkou Span A: {result.get('senkou_span_a', 'N/A'):.4f}" if result.get('senkou_span_a') else "Senkou Span A: N/A")
        print(f"Senkou Span B: {result.get('senkou_span_b', 'N/A'):.4f}" if result.get('senkou_span_b') else "Senkou Span B: N/A")
        print(f"Chikou Span: {result.get('chikou_span', 'N/A'):.4f}" if result.get('chikou_span') else "Chikou Span: N/A")
        
        # Cloud analysis
        print(f"\n🌤️  CLOUD ANALYSIS:")
        print(f"Cloud Color: {result.get('cloud_color', 'N/A').upper()}")
        print(f"Price Position: {result.get('price_vs_cloud', 'N/A').replace('_', ' ').upper()}")
        print(f"Cloud Thickness: {result.get('cloud_thickness', 'N/A'):.4f}" if result.get('cloud_thickness') else "Cloud Thickness: N/A")
        
        # Signals
        signals = result.get('signals', {})
        print(f"\n📊 TRADING SIGNALS:")
        print(f"Tenkan/Kijun Cross: {signals.get('tenkan_kijun_cross', 'N/A').replace('_', ' ').upper()}")
        print(f"Price/Cloud Breakout: {signals.get('price_cloud_breakout', 'N/A').replace('_', ' ').upper()}")
        print(f"Chikou Confirmation: {signals.get('chikou_confirmation', 'N/A').replace('_', ' ').upper()}")
        print(f"Overall Signal: {signals.get('overall_signal', 'N/A').replace('_', ' ').upper()}")
        
        # Trend strength
        trend_strength = result.get('trend_strength', 'N/A').replace('_', ' ').upper()
        print(f"\n📈 TREND STRENGTH: {trend_strength}")
        