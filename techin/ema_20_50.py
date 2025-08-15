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


class EMA2050:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.param = {
            # Standard EMA Crossover Parameters (Source: TradingView, Investopedia, Fidelity)
            "ema_fast_period": 20,               # Fast EMA period (commonly used short-term)
            "ema_slow_period": 50,               # Slow EMA period (commonly used medium-term)
            "price_source": "close",             # Standard price source for calculation
            "smoothing_factor_fast": 2.0 / (20 + 1),  # Alpha for 20-period EMA
            "smoothing_factor_slow": 2.0 / (50 + 1),  # Alpha for 50-period EMA
            
            # Signal Parameters
            "crossover_confirmation_periods": 2,  # Periods to confirm crossover
            "trend_strength_threshold": 0.001,   # Minimum percentage difference for trend strength
            "volume_confirmation": False,        # Optional volume confirmation
            "volume_ma_period": 20,             # Volume moving average period
            "divergence_lookback": 10,          # Periods to look back for divergence
            
            # Position and Risk Management
            "position_threshold_pct": 0.5,      # Position threshold percentage
            "stop_loss_atr_multiplier": 2.0,    # ATR multiplier for stop loss
            "take_profit_ratio": 2.0,           # Risk/reward ratio for take profit
            "max_lookback_periods": 100,        # Maximum periods for historical analysis
            
            # Confidence Scoring Weights
            "confidence_weights": {
                "crossover_strength": 0.4,      # Weight for crossover momentum
                "trend_consistency": 0.3,       # Weight for trend direction consistency  
                "volume_confirmation": 0.2,     # Weight for volume support
                "divergence_absence": 0.1       # Weight for lack of divergence
            }
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
            self.print_output(result)
            return
            
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
            signal = "bullish_trend"
        elif current_confirmed_bearish and abs(current_ema_distance) > self.param["trend_strength_threshold"]:
            signal = "bearish_trend"
        elif abs(current_ema_distance) < self.param["trend_strength_threshold"]:
            signal = "consolidation"
        
        # Trend strength classification
        trend_strength = "weak"
        if abs(current_ema_distance) > 2.0:
            trend_strength = "strong"
        elif abs(current_ema_distance) > 1.0:
            trend_strength = "moderate"
        
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": current_price,
            "ema_fast": current_ema_fast,
            "ema_slow": current_ema_slow,
            "ema_distance_pct": current_ema_distance,
            "trend_bullish": current_trend_bullish,
            "trend_bearish": current_trend_bearish,
            "crossover_up": current_crossover_up,
            "crossover_down": current_crossover_down,
            "confirmed_bullish": current_confirmed_bullish,
            "confirmed_bearish": current_confirmed_bearish,
            "signal": signal,
            "trend_strength": trend_strength,
            "volume_confirmed": volume_confirmed,
            "parameters": {
                "ema_fast_period": fast_period,
                "ema_slow_period": slow_period,
                "confirmation_periods": confirmation_periods,
                "trend_strength_threshold": self.param["trend_strength_threshold"],
                "volume_confirmation": self.param["volume_confirmation"]
            }
        }
        
        self.print_output(result)
        
    def print_output(self, result):
        """Print EMA 20/50 analysis results with one-line summary"""
        if "error" in result:
            print(f"\nEMA 20/50 Error: {result['error']}")
            return
            
        # One-line summary
        ema_trend = "fast above slow" if result["ema_fast"] > result["ema_slow"] else "fast below slow"
        summary = f"\nEMA 20/50: {ema_trend} ({result['ema_distance_pct']:.2f}%), signal: {result['signal']}"
        print(summary)