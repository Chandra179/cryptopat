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


class IchimokuCloud:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.param = {
            # Goichi Hosoda's Standard Parameters (Source: Wikipedia, Fidelity, TradingView)
            "tenkan_period": 9,              # Conversion Line period (Tenkan-sen)
            "kijun_period": 26,              # Base Line period (Kijun-sen) 
            "senkou_span_b_period": 52,      # Leading Span B period (Senkou Span B)
            "displacement": 26,              # Cloud displacement forward/backward
            
            # Signal Generation Parameters
            "trend_confirmation": True,      # Require multiple line confirmations
            "cloud_thickness_threshold": 0.01,  # Minimum cloud thickness for valid signals
            "price_cloud_buffer": 0.002,     # Buffer zone for price-cloud interactions
            "lagging_span_periods": 26,      # Chikou Span lookback
            
            # Advanced Analysis Parameters  
            "breakout_confirmation_periods": 3,   # Periods to confirm cloud breakout
            "momentum_confirmation": True,    # Use Tenkan/Kijun crossover for momentum
            "support_resistance_levels": 5,  # Number of historical S/R levels to track
            "trend_strength_threshold": 0.5, # Threshold for trend strength assessment
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
        Calculate Ichimoku Cloud (Ichimoku Kinko Hyo) according to Goichi Hosoda's methodology.
        
        Formula (Source: Goichi Hosoda, 1930s; Wikipedia, Fidelity):
        - Tenkan-sen (Conversion Line) = (Highest High + Lowest Low) / 2 over 9 periods
        - Kijun-sen (Base Line) = (Highest High + Lowest Low) / 2 over 26 periods  
        - Senkou Span A (Leading Span A) = (Tenkan-sen + Kijun-sen) / 2, displaced +26 periods
        - Senkou Span B (Leading Span B) = (Highest High + Lowest Low) / 2 over 52 periods, displaced +26 periods
        - Chikou Span (Lagging Span) = Close price displaced -26 periods
        - Kumo (Cloud) = Area between Senkou Span A and Senkou Span B
        
        Standard Parameters:
        - Tenkan-sen: 9 periods (short-term trend)
        - Kijun-sen: 26 periods (medium-term trend) 
        - Senkou Span B: 52 periods (long-term trend)
        - Displacement: 26 periods (cloud projection)
        
        References:
        - Created by Goichi Hosoda (細田悟一) in the late 1930s, published 1960s
        - "Ichimoku Sanjin" - "what a man in the mountain sees"
        - Wikipedia: https://en.wikipedia.org/wiki/Ichimoku_Kink%C5%8D_Hy%C5%8D
        - Fidelity: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/ichimoku-cloud
        - TradingView: Multiple educational resources on Ichimoku analysis
        
        Interpretation:
        - Above cloud: Bullish trend
        - Below cloud: Bearish trend  
        - Inside cloud: Neutral/consolidation
        - Cloud color: Green (Span A > Span B), Red (Span A < Span B)
        - Thick cloud: Strong support/resistance
        - Thin cloud: Weak support/resistance
        """
        required_periods = max(self.param["senkou_span_b_period"], self.param["kijun_period"]) + self.param["displacement"]
        
        if not self.ohlcv or len(self.ohlcv) < required_periods:
            result = {
                "error": f"Insufficient data: need at least {required_periods} candles, got {len(self.ohlcv) if self.ohlcv else 0}"
            }
            return result
            
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        
        tenkan_period = self.param["tenkan_period"]
        kijun_period = self.param["kijun_period"]
        senkou_b_period = self.param["senkou_span_b_period"]
        displacement = self.param["displacement"]
        
        # Calculate Tenkan-sen (Conversion Line)
        tenkan_high = df['high'].rolling(window=tenkan_period).max()
        tenkan_low = df['low'].rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Calculate Kijun-sen (Base Line)
        kijun_high = df['high'].rolling(window=kijun_period).max()
        kijun_low = df['low'].rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        
        # Calculate Senkou Span B (Leading Span B)
        senkou_b_high = df['high'].rolling(window=senkou_b_period).max()
        senkou_b_low = df['low'].rolling(window=senkou_b_period).min()
        senkou_span_b = (senkou_b_high + senkou_b_low) / 2
        
        # Calculate Chikou Span (Lagging Span)
        chikou_span = df['close'].shift(-displacement)
        
        # Current values
        current_price = float(df['close'].iloc[-1])
        current_tenkan = float(tenkan_sen.iloc[-1])
        current_kijun = float(kijun_sen.iloc[-1])
        
        # Cloud values (current and future)
        current_span_a = float(senkou_span_a.iloc[-1])
        current_span_b = float(senkou_span_b.iloc[-1])
        
        # Future cloud values (displaced forward)
        if len(senkou_span_a) >= displacement:
            future_span_a = float(senkou_span_a.iloc[-displacement])
            future_span_b = float(senkou_span_b.iloc[-displacement])
        else:
            future_span_a = current_span_a
            future_span_b = current_span_b
        
        # Cloud characteristics
        cloud_top = max(future_span_a, future_span_b)
        cloud_bottom = min(future_span_a, future_span_b)
        cloud_thickness = abs(future_span_a - future_span_b)
        cloud_thickness_pct = cloud_thickness / current_price if current_price > 0 else 0
        
        # Cloud color (Bullish: green, Bearish: red)
        cloud_color = "green" if future_span_a > future_span_b else "red"
        
        # Price position relative to cloud
        if current_price > cloud_top:
            price_position = "above_cloud"
            trend = "bullish"
        elif current_price < cloud_bottom:
            price_position = "below_cloud" 
            trend = "bearish"
        else:
            price_position = "inside_cloud"
            trend = "neutral"
        
        # Tenkan/Kijun relationship
        tk_cross = "bullish" if current_tenkan > current_kijun else "bearish" if current_tenkan < current_kijun else "neutral"
        
        # Chikou Span analysis
        chikou_value = float(chikou_span.iloc[-displacement-1]) if len(chikou_span) > displacement else None
        chikou_vs_price = None
        if chikou_value is not None:
            historical_price = float(df['close'].iloc[-displacement-1])
            chikou_vs_price = "bullish" if chikou_value > historical_price else "bearish"
        
        # Signal generation
        signal = "neutral"
        if trend == "bullish" and tk_cross == "bullish" and cloud_color == "green":
            signal = "strong_bullish"
        elif trend == "bearish" and tk_cross == "bearish" and cloud_color == "red":
            signal = "strong_bearish"
        elif trend == "bullish" and cloud_thickness_pct > self.param["cloud_thickness_threshold"]:
            signal = "bullish"
        elif trend == "bearish" and cloud_thickness_pct > self.param["cloud_thickness_threshold"]:
            signal = "bearish"
        elif price_position == "inside_cloud":
            signal = "consolidation"
        
        # Confidence scoring
        confidence_factors = {
            "price_cloud_alignment": 1.0 if price_position != "inside_cloud" else 0.3,
            "tk_alignment": 1.0 if tk_cross != "neutral" else 0.5,
            "cloud_thickness": min(cloud_thickness_pct / self.param["cloud_thickness_threshold"], 1.0),
            "cloud_color_trend": 1.0 if (cloud_color == "green" and trend == "bullish") or (cloud_color == "red" and trend == "bearish") else 0.7
        }
        confidence = sum(confidence_factors.values()) / len(confidence_factors)

        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": current_price,
            "tenkan_sen": current_tenkan,
            "kijun_sen": current_kijun,
            "senkou_span_a": current_span_a,
            "senkou_span_b": current_span_b,
            "future_span_a": future_span_a,
            "future_span_b": future_span_b,
            "cloud_top": cloud_top,
            "cloud_bottom": cloud_bottom,
            "cloud_thickness": cloud_thickness,
            "cloud_thickness_pct": cloud_thickness_pct,
            "cloud_color": cloud_color,
            "price_position": price_position,
            "trend": trend,
            "tk_cross": tk_cross,
            "chikou_span": chikou_value,
            "chikou_vs_price": chikou_vs_price,
            "signal": signal,
            "confidence": confidence,
            "parameters": {
                "tenkan_period": tenkan_period,
                "kijun_period": kijun_period,
                "senkou_span_b_period": senkou_b_period,
                "displacement": displacement
            }
        }
        
        # Add result to analysis summary
        indicator_result = IndicatorResult(
            name="Ichimoku Cloud",
            signal=result["signal"],
            value=result.get("tenkan_sen"),
            strength="strong" if "strong" in result["signal"] else "medium",
            support=result["cloud_bottom"] if result["cloud_bottom"] < result["current_price"] else None,
            resistance=result["cloud_top"] if result["cloud_top"] > result["current_price"] else None,
            metadata={
                "cloud_position": result["price_position"],
                "cloud_color": result["cloud_color"],
                "trend": result["trend"],
                "tenkan_sen": result["tenkan_sen"],
                "kijun_sen": result["kijun_sen"],
                "senkou_span_a": result["senkou_span_a"],
                "senkou_span_b": result["senkou_span_b"],
                "chikou_span": result["chikou_span"],
                "cloud_bottom": result["cloud_bottom"],
                "cloud_top": result["cloud_top"],
                "parameters": result["parameters"]
            }
        )
        add_indicator_result(indicator_result)
        
        return result
