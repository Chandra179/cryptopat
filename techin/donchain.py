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
import yaml
import os

class DonchianChannel:
    _config = None
    
    @classmethod
    def _load_config(cls):
        if cls._config is None:
            yaml_path = os.path.join(os.path.dirname(__file__), 'donchain.yaml')
            with open(yaml_path, 'r') as f:
                cls._config = yaml.safe_load(f)
        return cls._config
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        
        self.config = self._load_config()
        dc_config = self.config['donchian_channel']
        
        # Get timeframe-specific parameters or use default (1d)
        timeframe_params = dc_config['timeframes'].get(timeframe, dc_config['timeframes']['1d'])
        general_params = dc_config['params']
        
        # Combine parameters
        self.param = {**timeframe_params, **general_params}
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
            return result
            
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
        
        # Trend strength
        trend_period = self.param["trend_strength_period"]
        price_position = (df['close'] - lower_channel) / channel_width
        trend_strength = price_position.rolling(window=trend_period).mean()
        
        # Current values
        current_price = float(df['close'].iloc[-1])
        current_upper = float(upper_channel.iloc[-1])
        current_lower = float(lower_channel.iloc[-1])
        current_middle = float(middle_line.iloc[-1]) if middle_line is not None else None
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

        # Build result based on YAML output configuration  
        output_config = self.config['donchian_channel']['output']['fields']
        result = {}
        
        # Build result directly based on YAML fields
        for field_name in output_config:
            if field_name == "symbol":
                result[field_name] = self.symbol
            elif field_name == "timeframe":
                result[field_name] = self.timeframe
            elif field_name == "current_price":
                result[field_name] = current_price
            elif field_name == "upper_channel":
                result[field_name] = current_upper
            elif field_name == "lower_channel":
                result[field_name] = current_lower
            elif field_name == "middle_line":
                result[field_name] = current_middle
            elif field_name == "signal":
                result[field_name] = signal
            elif field_name == "channel_position":
                result[field_name] = current_position
            elif field_name == "channel_width":
                result[field_name] = current_width_pct
            elif field_name == "breakout_up":
                result[field_name] = current_upper_breakout
            elif field_name == "breakout_down":
                result[field_name] = current_lower_breakout
            elif field_name == "parameters":
                result[field_name] = {
                    "period": period,
                    "middle_line": self.param["middle_line"],
                    "breakout_confirmation": self.param["breakout_confirmation"],
                    "channel_width_threshold": self.param["channel_width_threshold"],
                    "reversal_zone_pct": self.param["reversal_zone_pct"]
                }
        
        self.print_output(result)
        return result
    
    def print_output(self, result):
        """Print analysis summary for Donchian Channel indicator"""
        if "error" in result:
            print(f"⚠️  Donchian Channel Error: {result['error']}")
            return
            
        symbol = result.get('symbol', 'N/A')
        timeframe = result.get('timeframe', 'N/A')
        signal = result.get('signal', 'neutral')
        current_price = result.get('current_price', 0)
        upper_band = result.get('upper_channel', 0)
        lower_band = result.get('lower_channel', 0)
        middle_line = result.get('middle_line', 0)
        position = result.get('position', 'neutral')
        channel_width = result.get('channel_width', 0)
        breakout_up = result.get('breakout_up', False)
        breakout_down = result.get('breakout_down', False)
        
        print("\n======================================")
        print(f"📊 Donchian Channel Analysis - {symbol} ({timeframe})")
        print("======================================")
        print(f"Current Price: ${current_price:.4f}")
        print(f"Upper Band: ${upper_band:.4f}")
        if middle_line:
            print(f"Middle Line: ${middle_line:.4f}")
        print(f"Lower Band: ${lower_band:.4f}")
        print(f"Channel Width: {channel_width:.2f}%")
        
        # Signal interpretation
        signal_emoji = {
            'bullish_breakout': '🚀',
            'bearish_breakout': '📉',
            'reversal_up': '🔄',
            'reversal_down': '🔄',
            'squeeze': '🔒',
            'neutral': '⚪'
        }
        
        print(f"Signal: {signal_emoji.get(signal, '⚪')} {signal.upper()}")
        
        # Breakout detection
        if breakout_up:
            print("🚀 Bullish breakout detected - price above 20-period high!")
        elif breakout_down:
            print("📉 Bearish breakout detected - price below 20-period low!")
        
        # Price position analysis
        if position == 'upper':
            distance_pct = (current_price - upper_band) / upper_band * 100 if breakout_up else (upper_band - current_price) / upper_band * 100
            if breakout_up:
                print(f"🚀 Price above upper band (+{abs(distance_pct):.2f}%) - strong momentum")
            else:
                print(f"📈 Price in upper channel area - bullish bias")
        elif position == 'lower':
            distance_pct = (lower_band - current_price) / lower_band * 100 if breakout_down else (current_price - lower_band) / lower_band * 100
            if breakout_down:
                print(f"📉 Price below lower band (-{abs(distance_pct):.2f}%) - strong selling")
            else:
                print(f"📉 Price in lower channel area - bearish bias")
        elif position == 'middle':
            print("⚖️  Price in middle of channel - neutral position")
        
        # Channel width analysis
        if channel_width > 8:
            print("Wide channel - high volatility, strong trending environment")
        elif channel_width < 2:
            print("Narrow channel - low volatility, potential breakout setup")
        else:
            print("Normal channel width - moderate volatility")
        
        # Signal-specific insights
        if signal == 'bullish_breakout':
            print("💡 Strong bullish signal - consider long positions")
            print("📈 Price breaking above resistance - trend continuation likely")
        elif signal == 'bearish_breakout':
            print("💡 Strong bearish signal - consider short positions")
            print("📉 Price breaking below support - trend continuation likely")
        elif signal == 'reversal_up':
            print("💡 Potential bullish reversal - watch for confirmation")
        elif signal == 'reversal_down':
            print("💡 Potential bearish reversal - watch for confirmation")
        elif signal == 'squeeze':
            print("💡 Channel squeeze - prepare for breakout in either direction")
        elif position == 'upper':
            print("💡 Price near resistance - watch for breakout or reversal")
        elif position == 'lower':
            print("💡 Price near support - watch for bounce or breakdown")
        else:
            print("💡 Price in neutral zone - wait for clearer signals")
        
        # Trend strength based on channel position
        if breakout_up or breakout_down:
            print("🔥 Strong trend momentum - follow the breakout direction")
        elif channel_width < 2:
            print("⏳ Consolidation phase - accumulate before next move")
