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


class KeltnerChannel:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.param = {
            # Chester Keltner's Original Parameters (Source: TradingView, Fidelity, Wikipedia)
            "ema_period": 20,                # N-period for Exponential Moving Average (typical default)
            "atr_period": 10,                # Period for Average True Range calculation (Keltner default)
            "atr_multiplier": 2.0,           # Multiplier for ATR bands (typical default)
            "price_source": "typical",       # Price source: "typical" (H+L+C)/3, "close", "hl2" (H+L)/2
            "ma_type": "ema",                # Moving average type: "ema" (original), "sma", "wma"
            
            # Extended Analysis Parameters
            "squeeze_threshold": 0.05,       # Channel width threshold for squeeze detection
            "breakout_period": 5,            # Periods to confirm breakout
            "volume_confirmation": False,    # Optional volume confirmation for signals
            "channel_position_upper": 0.8,   # Upper channel position for overbought (80%)
            "channel_position_lower": 0.2,   # Lower channel position for oversold (20%)
            "trend_strength_period": 10,     # Period for trend strength calculation
            "volatility_expansion_threshold": 1.5,  # Threshold for volatility expansion
            "momentum_period": 14,           # Period for momentum calculation
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
        
        # Trend direction based on price relative to middle line
        trend_direction = "neutral"
        current_price = float(df['close'].iloc[-1])
        current_middle = float(middle_line.iloc[-1])
        
        if current_price > current_middle:
            trend_direction = "bullish"
        elif current_price < current_middle:
            trend_direction = "bearish"
            
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

        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": current_price,
            "middle_line": current_middle,
            "upper_channel": current_upper,
            "lower_channel": current_lower,
            "channel_position": current_channel_position,
            "channel_width": current_channel_width,
            "atr": current_atr,
            "trend_direction": trend_direction,
            "squeeze": current_squeeze,
            "signal": signal,
            "breakout_up": current_breakout_up,
            "breakout_down": current_breakout_down,
            "volume_confirmed": volume_confirmed,
            "ma_type": ma_type,
            "price_source": price_source,
            "parameters": {
                "ema_period": ema_period,
                "atr_period": atr_period,
                "atr_multiplier": multiplier,
                "squeeze_threshold": self.param["squeeze_threshold"],
                "channel_position_upper": self.param["channel_position_upper"],
                "channel_position_lower": self.param["channel_position_lower"]
            }
        }
        
        # Add result to analysis summary
        indicator_result = IndicatorResult(
            name="Keltner Channel",
            signal=result["signal"],
            value=result["channel_position"],
            strength="strong" if abs(result["channel_position"] - 0.5) > 0.3 else "medium",
            support=result["lower_channel"],
            resistance=result["upper_channel"],
            metadata={
                "trend_direction": result["trend_direction"],
                "middle_line": result["middle_line"],
                "channel_width": result["channel_width"],
                "atr": result["atr"],
                "squeeze": result["squeeze"],
                "breakout_up": result["breakout_up"],
                "breakout_down": result["breakout_down"],
                "volume_confirmed": result["volume_confirmed"],
                "parameters": result["parameters"]
            }
        )
        add_indicator_result(indicator_result)
        
        return result
