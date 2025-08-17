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

class MACD():
    _config = None
    
    @classmethod
    def _load_config(cls):
        if cls._config is None:
            yaml_path = os.path.join(os.path.dirname(__file__), 'macd.yaml')
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
        macd_config = self.config['macd']
        
        # Get timeframe-specific parameters or use default (1d)
        timeframe_params = macd_config['timeframes'].get(timeframe, macd_config['timeframes']['1d'])
        general_params = macd_config['params']
        
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
        
        # Build result based on YAML output configuration
        output_config = self.config['macd']['output']['fields']
        result = {}
        
        # Build result directly based on YAML fields
        for field_name in output_config:
            if field_name == "symbol":
                result[field_name] = self.symbol
            elif field_name == "timeframe":
                result[field_name] = self.timeframe
            elif field_name == "current_price":
                result[field_name] = current_price
            elif field_name == "macd_line":
                result[field_name] = current_macd
            elif field_name == "signal_line":
                result[field_name] = current_signal
            elif field_name == "histogram":
                result[field_name] = current_histogram
            elif field_name == "fast_ema":
                result[field_name] = current_fast_ema
            elif field_name == "slow_ema":
                result[field_name] = current_slow_ema
            elif field_name == "signal":
                result[field_name] = signal
            elif field_name == "trend":
                result[field_name] = trend
            elif field_name == "momentum_strength":
                result[field_name] = momentum_strength
            elif field_name == "zero_crossover_up":
                result[field_name] = zero_crossover_up
            elif field_name == "zero_crossover_down":
                result[field_name] = zero_crossover_down
            elif field_name == "signal_crossover_up":
                result[field_name] = signal_crossover_up
            elif field_name == "signal_crossover_down":
                result[field_name] = signal_crossover_down
            elif field_name == "histogram_increasing":
                result[field_name] = histogram_increasing
            elif field_name == "histogram_decreasing":
                result[field_name] = histogram_decreasing
            elif field_name == "parameters":
                result[field_name] = {
                    "fast_period": fast_period,
                    "slow_period": slow_period,
                    "signal_period": signal_period,
                    "zero_line_threshold": self.param["zero_line_threshold"],
                    "signal_threshold": self.param["signal_threshold"]
                }
        
        self.print_output(result)
        return result
    
    def print_output(self, result):
        """Print analysis summary for MACD indicator"""
        if "error" in result:
            print(f"⚠️  MACD Error: {result['error']}")
            return
            
        symbol = result.get('symbol', 'N/A')
        timeframe = result.get('timeframe', 'N/A')
        signal = result.get('signal', 'neutral')
        current_price = result.get('current_price', 0)
        macd_line = result.get('macd_line', 0)
        signal_line = result.get('signal_line', 0)
        histogram = result.get('histogram', 0)
        trend = result.get('trend', 'neutral')
        
        print(f"\n📈 MACD Analysis - {symbol} ({timeframe})")
        print(f"Current Price: ${current_price:.4f}")
        print(f"MACD Line: {macd_line:.6f}")
        print(f"Signal Line: {signal_line:.6f}")
        print(f"Histogram: {histogram:.6f}")
        
        # Signal interpretation
        signal_emoji = {
            'bullish_zero_cross': '🚀',
            'bearish_zero_cross': '📉',
            'bullish_signal_cross': '🟢',
            'bearish_signal_cross': '🔴',
            'bullish_momentum': '⬆️',
            'bearish_momentum': '⬇️',
            'weakening_bullish': '🟡',
            'weakening_bearish': '🟠',
            'neutral': '⚪'
        }
        
        trend_emoji = {
            'bullish': '🟢',
            'bearish': '🔴',
            'neutral': '⚪'
        }
        
        print(f"Signal: {signal_emoji.get(signal, '⚪')} {signal.replace('_', ' ').upper()}")
        print(f"Trend: {trend_emoji.get(trend, '⚪')} {trend.upper()}")
        
        # Key insights
        if macd_line > 0:
            print("📍 MACD above zero line - bullish momentum")
        else:
            print("📍 MACD below zero line - bearish momentum")
            
        if macd_line > signal_line:
            print("📊 MACD above signal line")
        else:
            print("📊 MACD below signal line")
            
        if abs(histogram) > 0.001:
            direction = "expanding" if abs(histogram) > abs(histogram * 0.9) else "contracting"
            print(f"📶 Histogram {direction} - momentum {'strengthening' if direction == 'expanding' else 'weakening'}")