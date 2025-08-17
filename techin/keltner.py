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

class KeltnerChannel:
    _config = None
    
    @classmethod
    def _load_config(cls):
        if cls._config is None:
            yaml_path = os.path.join(os.path.dirname(__file__), 'keltner.yaml')
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
        kc_config = self.config['keltner_channel']
        
        # Get timeframe-specific parameters or use default (1d)
        timeframe_params = kc_config['timeframes'].get(timeframe, kc_config['timeframes']['1d'])
        general_params = kc_config['params']
        
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
        Calculate Keltner Channels according to Chester Keltner's original methodology.
        
        Formula (Source: Chester Keltner, TradingView, Fidelity):
        - Middle Line = N-period Exponential Moving Average (EMA) of typical price
        - Upper Channel = Middle Line + (Multiplier × Average True Range)
        - Lower Channel = Middle Line - (Multiplier × Average True Range)
        
        Where Typical Price = (High + Low + Close) / 3
        
        Standard Parameters:
        - EMA Period: 20 (default)
        - ATR Period: 10 (original Keltner specification)
        - ATR Multiplier: 2.0 (default)
        - Price Source: Typical Price ((H+L+C)/3)
        
        References:
        - Created by Chester Keltner in the 1960s
        - Modified by Linda Bradford Raschke (using ATR instead of simple range)
        - TradingView: https://www.tradingview.com/support/solutions/43000502266-keltner-channels-kc/
        - Fidelity: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/keltner-channels
        - Wikipedia: https://en.wikipedia.org/wiki/Keltner_channel
        
        Original vs Modified:
        - Original Keltner: Used simple moving average and simple range
        - Modern Keltner: Uses EMA and Average True Range (ATR) for better volatility measurement
        
        Usage:
        - Trend identification: Price above/below middle line
        - Breakout signals: Price moving outside the channels
        - Overbought/oversold: Price near upper/lower channels
        """
        if not self.ohlcv or len(self.ohlcv) < max(self.param["ema_period"], self.param["atr_period"]):
            min_required = max(self.param["ema_period"], self.param["atr_period"])
            result = {
                "error": f"Insufficient data: need at least {min_required} candles, got {len(self.ohlcv) if self.ohlcv else 0}"
            }
            return result
            
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        ema_period = self.param["ema_period"]
        atr_period = self.param["atr_period"]
        multiplier = self.param["atr_multiplier"]
        ma_type = self.param["ma_type"]
        price_source = self.param["price_source"]
        
        # Calculate price source
        if price_source == "typical":
            price = (df['high'] + df['low'] + df['close']) / 3
        elif price_source == "hl2":
            price = (df['high'] + df['low']) / 2
        else:  # close
            price = df['close']
            
        # Calculate moving average based on type
        if ma_type == "sma":
            middle_line = price.rolling(window=ema_period).mean()
        elif ma_type == "wma":
            weights = pd.Series(range(1, ema_period + 1))
            middle_line = price.rolling(window=ema_period).apply(lambda x: (x * weights).sum() / weights.sum(), raw=False)
        else:  # ema (default)
            middle_line = price.ewm(span=ema_period).mean()
            
        # Calculate Average True Range (ATR)
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        atr = df['true_range'].rolling(window=atr_period).mean()
        
        # Calculate Keltner Channels
        upper_channel = middle_line + (atr * multiplier)
        lower_channel = middle_line - (atr * multiplier)
        
        # Calculate channel position (similar to %B in Bollinger Bands)
        channel_position = (df['close'] - lower_channel) / (upper_channel - lower_channel)
        
        # Calculate channel width (volatility measure)
        channel_width = (upper_channel - lower_channel) / middle_line
        
        # Squeeze detection (low volatility)
        squeeze_detected = channel_width < self.param["squeeze_threshold"]
        
        # Breakout detection
        breakout_period = self.param["breakout_period"]
        price_above_upper = df['close'] > upper_channel
        price_below_lower = df['close'] < lower_channel
        breakout_up = price_above_upper.rolling(window=breakout_period).sum() > 0
        breakout_down = price_below_lower.rolling(window=breakout_period).sum() > 0
        
        current_price = float(df['close'].iloc[-1])
        current_middle = float(middle_line.iloc[-1])
        
        # Current values
        current_upper = float(upper_channel.iloc[-1])
        current_lower = float(lower_channel.iloc[-1])
        current_channel_position = float(channel_position.iloc[-1])
        current_channel_width = float(channel_width.iloc[-1])
        current_atr = float(atr.iloc[-1])
        current_squeeze = bool(squeeze_detected.iloc[-1])
        current_breakout_up = bool(breakout_up.iloc[-1])
        current_breakout_down = bool(breakout_down.iloc[-1])
        
        # Volume confirmation if enabled
        volume_confirmed = True
        if self.param["volume_confirmation"]:
            avg_volume = df['volume'].rolling(window=ema_period).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_confirmed = current_volume > avg_volume
        
        # Signal generation
        signal = "neutral"
        if current_channel_position >= self.param["channel_position_upper"]:
            signal = "overbought"
        elif current_channel_position <= self.param["channel_position_lower"]:
            signal = "oversold"
        elif current_breakout_up and volume_confirmed:
            signal = "bullish_breakout"
        elif current_breakout_down and volume_confirmed:
            signal = "bearish_breakout"
        elif current_squeeze:
            signal = "squeeze"

        # Determine position based on channel_position value
        position = "middle"
        if current_channel_position >= 0.8:
            position = "upper"
        elif current_channel_position <= 0.2:
            position = "lower"
        elif current_price > current_upper:
            position = "above"
        elif current_price < current_lower:
            position = "below"

        # Build result based on YAML output configuration  
        output_config = self.config['keltner_channel']['output']['fields']
        result = {}
        
        # Build result directly based on YAML fields
        for field_name in output_config:
            if field_name == "symbol":
                result[field_name] = self.symbol
            elif field_name == "timeframe":
                result[field_name] = self.timeframe
            elif field_name == "current_price":
                result[field_name] = current_price
            elif field_name == "middle_line":
                result[field_name] = current_middle
            elif field_name == "upper_band":
                result[field_name] = current_upper
            elif field_name == "lower_band":
                result[field_name] = current_lower
            elif field_name == "atr":
                result[field_name] = current_atr
            elif field_name == "signal":
                result[field_name] = signal
            elif field_name == "position":
                result[field_name] = position
            elif field_name == "squeeze":
                result[field_name] = current_squeeze
            elif field_name == "channel_width":
                result[field_name] = current_channel_width
            elif field_name == "parameters":
                result[field_name] = {
                    "ema_period": ema_period,
                    "atr_period": atr_period,
                    "atr_multiplier": multiplier,
                    "squeeze_threshold": self.param["squeeze_threshold"],
                    "channel_position_upper": self.param["channel_position_upper"],
                    "channel_position_lower": self.param["channel_position_lower"]
                }
        
        return result
