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
from summary import add_indicator_result, IndicatorResult
from config import get_indicator_params


class BollingerBands:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        
        self.param = get_indicator_params('bollinger_bands', timeframe)
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.ticker = ticker
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
    
    def calculate(self):
        """
        Calculate Bollinger Bands according to John Bollinger's original methodology.
        
        Formula (Source: John Bollinger, TradingView, Fidelity):
        - Middle Band = N-period Simple Moving Average (SMA)
        - Upper Band = SMA + (K × N-period standard deviation)
        - Lower Band = SMA - (K × N-period standard deviation)
        
        Standard Parameters:
        - N (period): 20 days (default)
        - K (multiplier): 2.0 standard deviations (default)
        - Source: Close price
        
        References:
        - Created by John Bollinger in the 1980s
        - TradingView: https://www.tradingview.com/support/solutions/43000501840-bollinger-bands-bb/
        - Fidelity: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/bollinger-bands
        - Wikipedia: https://en.wikipedia.org/wiki/Bollinger_Bands
        
        Parameter Adjustments by John Bollinger:
        - 10-period SMA: Use 1.9 multiplier
        - 20-period SMA: Use 2.0 multiplier (standard)
        - 50-period SMA: Use 2.1 multiplier
        
        Coverage: ~88-89% of price action falls within the bands (not 95% as commonly assumed)
        """
        if not self.ohlcv or len(self.ohlcv) < self.param["period"]:
            return {
                "error": f"Insufficient data: need at least {self.param['period']} candles, got {len(self.ohlcv) if self.ohlcv else 0}"
            }
            
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        period = self.param["period"]
        multiplier = self.param["std_dev_multiplier"]
        ma_type = self.param["ma_type"]
        
        # Calculate moving average based on type
        if ma_type == "ema":
            ma = df['close'].ewm(span=period).mean()
        elif ma_type == "wma":
            weights = pd.Series(range(1, period + 1))
            ma = df['close'].rolling(window=period).apply(lambda x: (x * weights).sum() / weights.sum(), raw=False)
        else:  # sma (default)
            ma = df['close'].rolling(window=period).mean()
            
        std = df['close'].rolling(window=period).std()  
        
        upper_band = ma + (std * multiplier)
        lower_band = ma - (std * multiplier)
        
        # Calculate %B (Percent B)
        percent_b = (df['close'] - lower_band) / (upper_band - lower_band)
        
        # Calculate Bandwidth
        bandwidth = (upper_band - lower_band) / ma
        
        # Squeeze detection
        squeeze_detected = bandwidth < self.param["squeeze_threshold"]
        
        # Breakout detection
        breakout_period = self.param["breakout_period"]
        price_above_upper = df['close'] > upper_band
        price_below_lower = df['close'] < lower_band
        breakout_up = price_above_upper.rolling(window=breakout_period).sum() > 0
        breakout_down = price_below_lower.rolling(window=breakout_period).sum() > 0
        
        # Current values
        current_price = float(df['close'].iloc[-1])
        current_ma = float(ma.iloc[-1])
        current_upper = float(upper_band.iloc[-1])
        current_lower = float(lower_band.iloc[-1])
        current_percent_b = float(percent_b.iloc[-1])
        current_bandwidth = float(bandwidth.iloc[-1])
        current_squeeze = bool(squeeze_detected.iloc[-1])
        current_breakout_up = bool(breakout_up.iloc[-1])
        current_breakout_down = bool(breakout_down.iloc[-1])
        
        # Volume confirmation if enabled
        volume_confirmed = True
        if self.param["volume_confirmation"]:
            avg_volume = df['volume'].rolling(window=period).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_confirmed = current_volume > avg_volume
        
        # Signal generation
        signal = "neutral"
        if current_percent_b >= self.param["percent_b_overbought"]:
            signal = "overbought"
        elif current_percent_b <= self.param["percent_b_oversold"]:
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
            "middle_band": current_ma,
            "upper_band": current_upper,
            "lower_band": current_lower,
            "percent_b": current_percent_b,
            "bandwidth": current_bandwidth,
            "squeeze": current_squeeze,
            "signal": signal,
            "breakout_up": current_breakout_up,
            "breakout_down": current_breakout_down,
            "volume_confirmed": volume_confirmed,
            "ma_type": ma_type,
            "parameters": {
                "period": period,
                "multiplier": multiplier,
                "squeeze_threshold": self.param["squeeze_threshold"],
                "percent_b_overbought": self.param["percent_b_overbought"],
                "percent_b_oversold": self.param["percent_b_oversold"]
            }
        }
        
        # Add result to analysis summary
        position_desc = ("above upper band" if result["current_price"] > result["upper_band"] else
                        "below lower band" if result["current_price"] < result["lower_band"] else
                        "within bands")
        
        indicator_result = IndicatorResult(
            name="Bollinger Bands",
            signal=result["signal"],
            value=result["percent_b"],
            strength=result.get("strength", "medium"),
            support=result["lower_band"],
            resistance=result["upper_band"],
            metadata={
                "position": position_desc,
                "percent_b": result["percent_b"],
                "bandwidth": result["bandwidth"],
                "middle_band": result["middle_band"],
                "squeeze": result["squeeze"],
                "expansion": result.get("expansion", False),
                "parameters": result["parameters"]
            }
        )
        add_indicator_result(indicator_result)
        
        return result