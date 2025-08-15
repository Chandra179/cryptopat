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


class DonchianChannel:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.param = {
            # Richard Donchian's Standard Parameters (Source: Wikipedia, TradingView)
            "period": 20,                    # N-period for high/low lookback (Donchian default)
            "middle_line": True,             # Calculate middle line (average of upper and lower)
            "price_source": "high_low",      # Uses high and low prices for channel calculation
            
            # Extended Analysis Parameters
            "breakout_confirmation": 1,      # Periods to confirm breakout signal
            "volume_confirmation": False,    # Optional volume confirmation for signals
            "channel_width_threshold": 0.05, # Threshold for narrow channel detection
            "trend_strength_period": 10,    # Period for trend strength calculation
            "volatility_percentile": 80,    # Percentile for high volatility detection
            "support_resistance_levels": 3, # Number of S/R levels to identify
            "position_threshold_pct": 1.0,  # Position threshold percentage
            "signal_sensitivity": "normal", # Signal sensitivity: conservative, normal, aggressive
            "false_breakout_filter": True,  # Filter false breakouts
            "min_breakout_distance": 0.001, # Minimum distance for valid breakout (%)
            "reversal_zone_pct": 0.1,      # Percentage zone near bands for reversal signals
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
        Calculate Donchian Channel according to Richard Donchian's methodology.
        
        Formula (Source: Richard Donchian, Wikipedia):
        - Upper Channel = Highest High over N periods
        - Lower Channel = Lowest Low over N periods
        - Middle Line = (Upper Channel + Lower Channel) / 2
        
        Standard Parameters:
        - N (period): 20 periods (default)
        - Price Source: High and Low prices
        
        References:
        - Created by Richard Donchian in the 1950s (Turtle Trading System)
        - Wikipedia: https://en.wikipedia.org/wiki/Donchian_channel
        - Known as "Price Channel" or "Trading Channel"
        - Used in the famous Turtle Trading system
        
        Trading Signals:
        - Buy Signal: Price breaks above upper channel (new N-period high)
        - Sell Signal: Price breaks below lower channel (new N-period low)
        - Reversal: Price touches channel and reverses
        
        Characteristics:
        - Trend-following indicator
        - Shows market volatility through channel width
        - Narrow channels suggest low volatility/consolidation
        - Wide channels suggest high volatility/trending markets
        
        Limitations:
        - Lagging indicator (based on historical extremes)
        - Can produce false signals in choppy markets
        - Works best in trending markets
        """
        if not self.ohlcv or len(self.ohlcv) < self.param["period"]:
            result = {
                "error": f"Insufficient data: need at least {self.param['period']} candles, got {len(self.ohlcv) if self.ohlcv else 0}"
            }
            self.print_output(result)
            return
            
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        period = self.param["period"]
        
        # Calculate Donchian Channel
        upper_channel = df['high'].rolling(window=period).max()
        lower_channel = df['low'].rolling(window=period).min()
        
        # Calculate middle line if enabled
        middle_line = None
        if self.param["middle_line"]:
            middle_line = (upper_channel + lower_channel) / 2
        
        # Channel width (volatility measure)
        channel_width = upper_channel - lower_channel
        channel_width_pct = (channel_width / df['close']) * 100
        
        # Breakout detection
        confirmation_period = self.param["breakout_confirmation"]
        upper_breakout = (df['close'] > upper_channel.shift(1)).rolling(window=confirmation_period).sum() >= confirmation_period
        lower_breakout = (df['close'] < lower_channel.shift(1)).rolling(window=confirmation_period).sum() >= confirmation_period
        
        # Support/Resistance levels
        recent_highs = df['high'].rolling(window=period//2).max()
        recent_lows = df['low'].rolling(window=period//2).min()
        
        # Trend strength
        trend_period = self.param["trend_strength_period"]
        price_position = (df['close'] - lower_channel) / channel_width
        trend_strength = price_position.rolling(window=trend_period).mean()
        
        # Current values
        current_price = float(df['close'].iloc[-1])
        current_upper = float(upper_channel.iloc[-1])
        current_lower = float(lower_channel.iloc[-1])
        current_middle = float(middle_line.iloc[-1]) if middle_line is not None else None
        current_width = float(channel_width.iloc[-1])
        current_width_pct = float(channel_width_pct.iloc[-1])
        current_position = float(price_position.iloc[-1])
        current_trend_strength = float(trend_strength.iloc[-1])
        current_upper_breakout = bool(upper_breakout.iloc[-1])
        current_lower_breakout = bool(lower_breakout.iloc[-1])
        
        # Volume confirmation if enabled
        volume_confirmed = True
        if self.param["volume_confirmation"]:
            avg_volume = df['volume'].rolling(window=period).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_confirmed = current_volume > avg_volume * 1.2  # 20% above average
        
        # Channel position analysis
        reversal_zone = self.param["reversal_zone_pct"]
        near_upper = current_position >= (1 - reversal_zone)
        near_lower = current_position <= reversal_zone
        
        # Signal generation
        signal = "neutral"
        confidence = 0.5
        
        if current_upper_breakout and volume_confirmed:
            signal = "bullish_breakout"
            confidence = 0.8
        elif current_lower_breakout and volume_confirmed:
            signal = "bearish_breakout"
            confidence = 0.8
        elif near_upper and current_trend_strength > 0.7:
            signal = "overbought_reversal"
            confidence = 0.6
        elif near_lower and current_trend_strength < 0.3:
            signal = "oversold_reversal"
            confidence = 0.6
        elif current_width_pct < self.param["channel_width_threshold"]:
            signal = "consolidation"
            confidence = 0.7
        elif current_trend_strength > 0.7:
            signal = "bullish_trend"
            confidence = 0.6
        elif current_trend_strength < 0.3:
            signal = "bearish_trend"
            confidence = 0.6
        
        # Adjust confidence based on volume
        if not volume_confirmed and signal in ["bullish_breakout", "bearish_breakout"]:
            confidence *= 0.7
        
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": current_price,
            "upper_channel": current_upper,
            "lower_channel": current_lower,
            "middle_line": current_middle,
            "channel_width": current_width,
            "channel_width_pct": current_width_pct,
            "price_position": current_position,
            "trend_strength": current_trend_strength,
            "signal": signal,
            "confidence": confidence,
            "upper_breakout": current_upper_breakout,
            "lower_breakout": current_lower_breakout,
            "volume_confirmed": volume_confirmed,
            "near_upper_band": near_upper,
            "near_lower_band": near_lower,
            "parameters": {
                "period": period,
                "middle_line": self.param["middle_line"],
                "breakout_confirmation": self.param["breakout_confirmation"],
                "channel_width_threshold": self.param["channel_width_threshold"],
                "reversal_zone_pct": self.param["reversal_zone_pct"]
            }
        }