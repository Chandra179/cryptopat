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

class EMA2050:
    _config = None
    
    @classmethod
    def _load_config(cls):
        if cls._config is None:
            yaml_path = os.path.join(os.path.dirname(__file__), 'ema_20_50.yaml')
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
        ema_config = self.config['ema_20_50']
        
        # Get timeframe-specific parameters or use default (1d)
        timeframe_params = ema_config['timeframes'].get(timeframe, ema_config['timeframes']['1d'])
        general_params = ema_config['params']
        
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
        Calculate EMA 20/50 crossover strategy signals.
        
        Formula (Source: TradingView, Investopedia, Fidelity):
        EMA = (Close - EMA_previous) × Smoothing Factor + EMA_previous
        
        Where Smoothing Factor = 2 / (Period + 1)
        
        Standard Parameters:
        - Fast EMA: 20 periods (Short-term trend)
        - Slow EMA: 50 periods (Medium-term trend)
        - Source: Close price
        
        Signals:
        - Golden Cross: Fast EMA crosses above Slow EMA (Bullish)
        - Death Cross: Fast EMA crosses below Slow EMA (Bearish)
        - Trend: Fast EMA above/below Slow EMA indicates trend direction
        
        References:
        - TradingView: https://www.tradingview.com/support/solutions/43000502589-exponential-moving-average-ema/
        - Investopedia: https://www.investopedia.com/terms/e/ema.asp
        - Fidelity: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/ema
        - Wikipedia: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average

        Advantages over SMA:
        - More responsive to recent price changes
        - Reduces lag in trend identification
        - Better for volatile markets
        """
        fast_period = self.param["ema_fast_period"]
        slow_period = self.param["ema_slow_period"]
        
        if not self.ohlcv or len(self.ohlcv) < slow_period:
            result = {
                "error": f"Insufficient data: need at least {slow_period} candles, got {len(self.ohlcv) if self.ohlcv else 0}"
            }
            return result
            
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # Calculate EMAs
        ema_fast = df['close'].ewm(span=fast_period).mean()
        ema_slow = df['close'].ewm(span=slow_period).mean()
        
        # Calculate crossover signals
        crossover_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        crossover_down = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
        
        # Trend identification
        trend_bullish = ema_fast > ema_slow
        trend_bearish = ema_fast < ema_slow
        
        # Calculate EMA distance (trend strength)
        ema_distance = (ema_fast - ema_slow) / ema_slow * 100  # Percentage difference
        
        # Crossover confirmation
        confirmation_periods = self.param["crossover_confirmation_periods"]
        confirmed_bullish = crossover_up.rolling(window=confirmation_periods).sum() > 0
        confirmed_bearish = crossover_down.rolling(window=confirmation_periods).sum() > 0
        
        # Volume confirmation if enabled
        volume_confirmed = True
        if self.param["volume_confirmation"]:
            volume_ma = df['volume'].rolling(window=self.param["volume_ma_period"]).mean()
            volume_confirmed = df['volume'].iloc[-1] > volume_ma.iloc[-1]
        
        # Current values
        current_price = float(df['close'].iloc[-1])
        current_ema_fast = float(ema_fast.iloc[-1])
        current_ema_slow = float(ema_slow.iloc[-1])
        current_ema_distance = float(ema_distance.iloc[-1])
        current_trend_bullish = bool(trend_bullish.iloc[-1])
        current_trend_bearish = bool(trend_bearish.iloc[-1])
        current_crossover_up = bool(crossover_up.iloc[-1])
        current_crossover_down = bool(crossover_down.iloc[-1])
        current_confirmed_bullish = bool(confirmed_bullish.iloc[-1])
        current_confirmed_bearish = bool(confirmed_bearish.iloc[-1])
        
        # Signal generation
        signal = "neutral"
        if current_crossover_up and volume_confirmed:
            signal = "golden_cross"
        elif current_crossover_down and volume_confirmed:
            signal = "death_cross"
        elif current_confirmed_bullish and abs(current_ema_distance) > self.param["trend_strength_threshold"]:
            signal = "bullish"
        elif current_confirmed_bearish and abs(current_ema_distance) > self.param["trend_strength_threshold"]:
            signal = "bearish"
        
        
        # Build result based on YAML output configuration  
        output_config = self.config['ema_20_50']['output']['fields']
        result = {}
        
        # Determine trend and crossover for YAML output format
        trend = "neutral"
        if current_trend_bullish:
            trend = "bullish"
        elif current_trend_bearish:
            trend = "bearish"
            
        crossover = "none"
        if current_crossover_up:
            crossover = "bullish"
        elif current_crossover_down:
            crossover = "bearish"
        
        # Build result directly based on YAML fields
        for field_name in output_config:
            if field_name == "symbol":
                result[field_name] = self.symbol
            elif field_name == "timeframe":
                result[field_name] = self.timeframe
            elif field_name == "current_price":
                result[field_name] = current_price
            elif field_name == "ema_20":
                result[field_name] = current_ema_fast
            elif field_name == "ema_50":
                result[field_name] = current_ema_slow
            elif field_name == "signal":
                result[field_name] = signal
            elif field_name == "trend":
                result[field_name] = trend
            elif field_name == "crossover":
                result[field_name] = crossover
            elif field_name == "price_above_ema20":
                result[field_name] = current_price > current_ema_fast
            elif field_name == "price_above_ema50":
                result[field_name] = current_price > current_ema_slow
            elif field_name == "ema20_above_ema50":
                result[field_name] = current_ema_fast > current_ema_slow
            elif field_name == "parameters":
                result[field_name] = {
                    "ema_fast_period": fast_period,
                    "ema_slow_period": slow_period,
                    "confirmation_periods": confirmation_periods,
                    "trend_strength_threshold": self.param["trend_strength_threshold"],
                    "volume_confirmation": self.param["volume_confirmation"]
                }
        
        self.print_output(result)
        return result
    
    def print_output(self, result):
        """Print analysis summary for EMA 20/50 crossover indicator"""
        if "error" in result:
            print(f"⚠️  EMA 20/50 Error: {result['error']}")
            return
            
        symbol = result.get('symbol', 'N/A')
        timeframe = result.get('timeframe', 'N/A')
        signal = result.get('signal', 'neutral')
        current_price = result.get('current_price', 0)
        ema_20 = result.get('ema_20', 0)
        ema_50 = result.get('ema_50', 0)
        trend = result.get('trend', 'neutral')
        crossover = result.get('crossover', 'none')
        
        print("\n======================================")
        print(f"📈 EMA 20/50 Analysis - {symbol} ({timeframe})")
        print("======================================")
        print(f"Current Price: ${current_price:.4f}")
        print(f"EMA 20: ${ema_20:.4f}")
        print(f"EMA 50: ${ema_50:.4f}")
        
        # Signal interpretation
        signal_emoji = {
            'golden_cross': '🟡',
            'death_cross': '⚫',
            'bullish': '🟢',
            'bearish': '🔴',
            'neutral': '⚪'
        }
        
        print(f"Signal: {signal_emoji.get(signal, '⚪')} {signal.upper()}")
        print(f"Trend: {trend.upper()}")
        
        # Crossover analysis
        if crossover == 'bullish':
            print("🚀 Golden Cross detected - EMA 20 crossed above EMA 50!")
        elif crossover == 'bearish':
            print("📉 Death Cross detected - EMA 20 crossed below EMA 50!")
        elif trend == 'bullish':
            print("📈 Price trending above both EMAs")
        elif trend == 'bearish':
            print("📉 Price trending below both EMAs")
        else:
            print("🔄 No clear trend - consolidation phase")
        
        # Price position analysis
        price_above_ema20 = result.get('price_above_ema20', False)
        price_above_ema50 = result.get('price_above_ema50', False)
        
        if price_above_ema20 and price_above_ema50:
            print("📍 Price above both EMAs - strong bullish momentum")
        elif not price_above_ema20 and not price_above_ema50:
            print("📍 Price below both EMAs - strong bearish momentum")
        elif price_above_ema20 and not price_above_ema50:
            print("📍 Price between EMAs - potential reversal zone")
        else:
            print("📍 Price position unclear")
