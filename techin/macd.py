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
import numpy as np
from analysis_summary import add_indicator_result, IndicatorResult

class MACD:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.param = {
            # Gerald Appel's Standard Parameters (Source: TradingView, Investopedia, StockCharts)
            "fast_period": 12,               # Fast EMA period (Appel default)
            "slow_period": 26,               # Slow EMA period (Appel default)
            "signal_period": 9,              # Signal line EMA period (Appel default)
            "price_source": "close",         # Standard source for calculation
            
            # Signal Detection Parameters
            "zero_line_threshold": 0.0001,   # Threshold for zero line crossover detection
            "signal_threshold": 0.0001,      # Threshold for signal line crossover detection
            "divergence_lookback": 14,       # Periods to look back for divergence detection
            "histogram_threshold": 0.001,    # Threshold for histogram reversal signals
            "trend_confirmation_periods": 3, # Periods to confirm trend direction
            "overbought_threshold": None,    # No standard overbought level for MACD
            "oversold_threshold": None,      # No standard oversold level for MACD
            
            # Advanced Analysis Parameters
            "momentum_smoothing": False,     # Optional smoothing for momentum calculation
            "use_percentage": False,         # Use MACD percentage instead of absolute values
            "centerline_oscillation": True, # Track centerline oscillations
            "histogram_divergence": True,   # Enable histogram divergence detection
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
        Calculate MACD (Moving Average Convergence Divergence) according to Gerald Appel's original methodology.
        
        Formula (Source: Gerald Appel, TradingView, Investopedia, StockCharts):
        - MACD Line = EMA(12) - EMA(26)
        - Signal Line = EMA(9) of MACD Line
        - Histogram = MACD Line - Signal Line
        
        Standard Parameters:
        - Fast EMA: 12 periods (default)
        - Slow EMA: 26 periods (default)
        - Signal EMA: 9 periods (default)
        - Source: Close price
        
        References:
        - Created by Gerald Appel in the late 1970s
        - TradingView: https://www.tradingview.com/support/solutions/43000502344-macd-moving-average-convergence-divergence/
        - Investopedia: https://www.investopedia.com/terms/m/macd.asp
        - StockCharts: https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd
        
        Signal Interpretation:
        - MACD above zero: Upward momentum (fast EMA > slow EMA)
        - MACD below zero: Downward momentum (fast EMA < slow EMA)
        - MACD crosses above signal: Bullish signal
        - MACD crosses below signal: Bearish signal
        - Histogram expansion: Momentum increasing
        - Histogram contraction: Momentum decreasing
        
        Common Variations:
        - MACD(5,35,5): More sensitive for short-term trading
        - MACD(19,39,9): Less sensitive for longer-term analysis
        """
        if not self.ohlcv or len(self.ohlcv) < self.param["slow_period"]:
            return {
                "error": f"Insufficient data: need at least {self.param['slow_period']} candles, got {len(self.ohlcv) if self.ohlcv else 0}"
            }
            
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = pd.to_numeric(df['close'])
        
        fast_period = self.param["fast_period"]
        slow_period = self.param["slow_period"]
        signal_period = self.param["signal_period"]
        
        # Calculate EMAs
        fast_ema = df['close'].ewm(span=fast_period).mean()
        slow_ema = df['close'].ewm(span=slow_period).mean()
        
        # Calculate MACD Line
        macd_line = fast_ema - slow_ema
        
        # Calculate Signal Line (EMA of MACD)
        signal_line = macd_line.ewm(span=signal_period).mean()
        
        # Calculate Histogram
        histogram = macd_line - signal_line
        
        # Current values
        current_price = float(df['close'].iloc[-1])
        current_macd = float(macd_line.iloc[-1])
        current_signal = float(signal_line.iloc[-1])
        current_histogram = float(histogram.iloc[-1])
        current_fast_ema = float(fast_ema.iloc[-1])
        current_slow_ema = float(slow_ema.iloc[-1])
        
        # Signal detection
        signal = "neutral"
        
        # Zero line crossover
        if len(macd_line) >= 2:
            prev_macd = float(macd_line.iloc[-2])
            zero_crossover_up = prev_macd <= 0 and current_macd > self.param["zero_line_threshold"]
            zero_crossover_down = prev_macd >= 0 and current_macd < -self.param["zero_line_threshold"]
        else:
            zero_crossover_up = False
            zero_crossover_down = False
        
        # Signal line crossover
        if len(signal_line) >= 2:
            prev_signal = float(signal_line.iloc[-2])
            prev_macd_sig = float(macd_line.iloc[-2])
            signal_crossover_up = (prev_macd_sig <= prev_signal and 
                                 current_macd > current_signal + self.param["signal_threshold"])
            signal_crossover_down = (prev_macd_sig >= prev_signal and 
                                   current_macd < current_signal - self.param["signal_threshold"])
        else:
            signal_crossover_up = False
            signal_crossover_down = False
        
        # Histogram momentum
        if len(histogram) >= 2:
            prev_histogram = float(histogram.iloc[-2])
            histogram_increasing = current_histogram > prev_histogram
            histogram_decreasing = current_histogram < prev_histogram
        else:
            histogram_increasing = False
            histogram_decreasing = False
        
        # Overall signal determination
        if zero_crossover_up:
            signal = "bullish_zero_cross"
        elif zero_crossover_down:
            signal = "bearish_zero_cross"
        elif signal_crossover_up:
            signal = "bullish_signal_cross"
        elif signal_crossover_down:
            signal = "bearish_signal_cross"
        elif current_macd > 0 and current_macd > current_signal and histogram_increasing:
            signal = "bullish_momentum"
        elif current_macd < 0 and current_macd < current_signal and histogram_decreasing:
            signal = "bearish_momentum"
        elif current_macd > 0 and histogram_decreasing:
            signal = "weakening_bullish"
        elif current_macd < 0 and histogram_increasing:
            signal = "weakening_bearish"
        
        # Trend analysis
        trend = "neutral"
        if current_macd > 0:
            trend = "bullish"
        elif current_macd < 0:
            trend = "bearish"
        
        # Momentum strength
        momentum_strength = abs(current_histogram) / abs(current_macd) if current_macd != 0 else 0
        
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": current_price,
            "macd_line": current_macd,
            "signal_line": current_signal,
            "histogram": current_histogram,
            "fast_ema": current_fast_ema,
            "slow_ema": current_slow_ema,
            "signal": signal,
            "trend": trend,
            "momentum_strength": momentum_strength,
            "zero_crossover_up": zero_crossover_up,
            "zero_crossover_down": zero_crossover_down,
            "signal_crossover_up": signal_crossover_up,
            "signal_crossover_down": signal_crossover_down,
            "histogram_increasing": histogram_increasing,
            "histogram_decreasing": histogram_decreasing,
            "parameters": {
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
                "zero_line_threshold": self.param["zero_line_threshold"],
                "signal_threshold": self.param["signal_threshold"]
            }
        }
        
        # Add result to analysis summary
        position_desc = "above zero" if result["macd_line"] > 0 else "below zero"
        histogram_trend = ("increasing" if result["histogram_increasing"] else 
                          "decreasing" if result["histogram_decreasing"] else "stable")
        
        indicator_result = IndicatorResult(
            name="MACD",
            signal=result["signal"],
            value=result["macd_line"],
            strength=result.get("strength", "medium"),
            metadata={
                "macd_line": result["macd_line"],
                "signal_line": result["signal_line"],
                "histogram": result["histogram"],
                "position": position_desc,
                "histogram_trend": histogram_trend,
                "histogram_increasing": result["histogram_increasing"],
                "histogram_decreasing": result["histogram_decreasing"],
                "crossover_type": result.get("crossover_type", "none"),
                "divergence": result.get("divergence", "none"),
                "parameters": result["parameters"]
            }
        )
        add_indicator_result(indicator_result)
        
        return result