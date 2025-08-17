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

class RSI():
    _config = None
    
    @classmethod
    def _load_config(cls):
        if cls._config is None:
            yaml_path = os.path.join(os.path.dirname(__file__), 'rsi.yaml')
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
        rsi_config = self.config['rsi']
        
        # Get timeframe-specific parameters or use default (1d)
        timeframe_params = rsi_config['timeframes'].get(timeframe, rsi_config['timeframes']['1d'])
        general_params = rsi_config['params']
        
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
        Calculate RSI (Relative Strength Index) according to J. Welles Wilder Jr.'s original methodology.
        
        Formula (Source: J. Welles Wilder Jr., TradingView, Fidelity):
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        
        Wilder's Smoothing (Original Method):
        - First RS = Simple Average of Gains / Simple Average of Losses (over N periods)
        - Subsequent RS = ((Previous Average Gain × (N-1)) + Current Gain) / N
                         / ((Previous Average Loss × (N-1)) + Current Loss) / N
        
        Standard Parameters:
        - Period: 14 days (Wilder's original)
        - Overbought: 70 (traditional threshold)
        - Oversold: 30 (traditional threshold)
        - Source: Close price
        
        Signal Interpretation:
        - RSI > 70: Potentially overbought (consider selling)
        - RSI < 30: Potentially oversold (consider buying)
        - RSI > 50: Bullish momentum
        - RSI < 50: Bearish momentum
        
        References:
        - Created by J. Welles Wilder Jr. in "New Concepts in Technical Trading Systems" (1978)
        - TradingView: https://www.tradingview.com/support/solutions/43000502338-relative-strength-index-rsi/
        - Fidelity: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/rsi
        - Wikipedia: https://en.wikipedia.org/wiki/Relative_strength_index
        
        Advanced Signals:
        - Failure Swings: RSI fails to exceed previous high/low while price makes new high/low
        - Divergence: Price and RSI move in opposite directions
        - Centerline Crossover: RSI crossing above/below 50 line
        """
        if not self.ohlcv or len(self.ohlcv) < self.param["period"] + 1:
            return {
                "error": f"Insufficient data: need at least {self.param['period'] + 1} candles, got {len(self.ohlcv) if self.ohlcv else 0}"
            }
            
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = pd.to_numeric(df['close'])
        
        period = self.param["period"]
        
        # Calculate price changes
        delta = df['close'].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Wilder's smoothing method (original RSI calculation)
        if self.param["smoothing_method"] == "wilder":
            # First average (simple average)
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            
            # Apply Wilder's smoothing for subsequent values
            for i in range(period, len(gains)):
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gains.iloc[i]) / period
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + losses.iloc[i]) / period
        else:
            # Alternative: EMA smoothing
            avg_gain = gains.ewm(alpha=1/period).mean()
            avg_loss = losses.ewm(alpha=1/period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Current values
        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        current_price = float(df['close'].iloc[-1])
        
        # Signal generation
        signal = "neutral"
        strength = "normal"
        
        if current_rsi >= self.param["extreme_overbought"]:
            signal = "extreme_overbought"
            strength = "strong"
        elif current_rsi >= self.param["overbought_level"]:
            signal = "overbought"
            strength = "moderate"
        elif current_rsi <= self.param["extreme_oversold"]:
            signal = "extreme_oversold"
            strength = "strong"
        elif current_rsi <= self.param["oversold_level"]:
            signal = "oversold"
            strength = "moderate"
        elif current_rsi > self.param["midline"]:
            signal = "bullish"
            strength = "weak"
        elif current_rsi < self.param["midline"]:
            signal = "bearish"
            strength = "weak"
        
        # Momentum analysis
        rsi_change = 0.0
        momentum_signal = "neutral"
        if len(rsi) >= 2 and not pd.isna(rsi.iloc[-2]):
            rsi_change = current_rsi - float(rsi.iloc[-2])
            if abs(rsi_change) >= self.param["momentum_threshold"]:
                momentum_signal = "bullish_momentum" if rsi_change > 0 else "bearish_momentum"
        
        # Trend filter (optional)
        trend_aligned = True
        trend_direction = "neutral"
        if self.param["use_trend_filter"] and len(df) >= self.param["trend_filter_period"]:
            trend_ma = df['close'].rolling(window=self.param["trend_filter_period"]).mean()
            if not pd.isna(trend_ma.iloc[-1]):
                trend_direction = "bullish" if current_price > trend_ma.iloc[-1] else "bearish"
                if signal in ["overbought", "extreme_overbought"] and trend_direction == "bullish":
                    trend_aligned = False
                elif signal in ["oversold", "extreme_oversold"] and trend_direction == "bearish":
                    trend_aligned = False
        
        # Failure swing detection
        failure_swing = None
        if self.param["failure_swing_enabled"] and len(rsi) >= self.param["divergence_lookback"] * 2:
            recent_rsi = rsi.iloc[-self.param["divergence_lookback"]:].values
            if len(recent_rsi) >= 3:
                # Bullish failure swing: RSI makes higher low while staying below 30
                if (current_rsi < self.param["oversold_level"] and 
                    current_rsi > min(recent_rsi[:-1]) and 
                    max(recent_rsi) < self.param["oversold_level"]):
                    failure_swing = "bullish_failure_swing"
                
                # Bearish failure swing: RSI makes lower high while staying above 70
                elif (current_rsi > self.param["overbought_level"] and 
                      current_rsi < max(recent_rsi[:-1]) and 
                      min(recent_rsi) > self.param["overbought_level"]):
                    failure_swing = "bearish_failure_swing"

        # Build result based on YAML output configuration  
        output_config = self.config['rsi']['output']['fields']
        result = {}
        
        # Build result directly based on YAML fields
        for field_name in output_config:
            if field_name == "symbol":
                result[field_name] = self.symbol
            elif field_name == "timeframe":
                result[field_name] = self.timeframe
            elif field_name == "current_price":
                result[field_name] = current_price
            elif field_name == "rsi":
                result[field_name] = current_rsi
            elif field_name == "signal":
                result[field_name] = signal
            elif field_name == "strength":
                result[field_name] = strength
            elif field_name == "momentum_signal":
                result[field_name] = momentum_signal
            elif field_name == "rsi_change":
                result[field_name] = rsi_change
            elif field_name == "trend_direction":
                result[field_name] = trend_direction
            elif field_name == "trend_aligned":
                result[field_name] = trend_aligned
            elif field_name == "failure_swing":
                result[field_name] = failure_swing
            elif field_name == "is_overbought":
                result[field_name] = current_rsi >= self.param["overbought_level"]
            elif field_name == "is_oversold":
                result[field_name] = current_rsi <= self.param["oversold_level"]
            elif field_name == "is_extreme_overbought":
                result[field_name] = current_rsi >= self.param["extreme_overbought"]
            elif field_name == "is_extreme_oversold":
                result[field_name] = current_rsi <= self.param["extreme_oversold"]
            elif field_name == "above_midline":
                result[field_name] = current_rsi > self.param["midline"]
            elif field_name == "parameters":
                result[field_name] = {
                    "period": period,
                    "overbought_level": self.param["overbought_level"],
                    "oversold_level": self.param["oversold_level"],
                    "extreme_overbought": self.param["extreme_overbought"],
                    "extreme_oversold": self.param["extreme_oversold"],
                    "smoothing_method": self.param["smoothing_method"]
                }
        
        self.print_output(result)
        return result
    
    def print_output(self, result):
        """Print analysis summary for RSI indicator"""
        if "error" in result:
            print(f"⚠️  RSI Error: {result['error']}")
            return
            
        symbol = result.get('symbol', 'N/A')
        timeframe = result.get('timeframe', 'N/A')
        signal = result.get('signal', 'neutral')
        current_price = result.get('current_price', 0)
        rsi = result.get('rsi', 50)
        strength = result.get('strength', 'normal')
        momentum_signal = result.get('momentum_signal', 'neutral')
        trend_direction = result.get('trend_direction', 'neutral')
        
        print(f"\n🎯 RSI Analysis - {symbol} ({timeframe})")
        print(f"Current Price: ${current_price:.4f}")
        print(f"RSI Value: {rsi:.2f}")
        
        # Signal interpretation
        signal_emoji = {
            'extreme_overbought': '🔴',
            'overbought': '🟠',
            'extreme_oversold': '🟢',
            'oversold': '🟡',
            'bullish': '⬆️',
            'bearish': '⬇️',
            'neutral': '⚪'
        }
        
        strength_emoji = {
            'strong': '💪',
            'moderate': '👍',
            'weak': '👌',
            'normal': '⚪'
        }
        
        print(f"Signal: {signal_emoji.get(signal, '⚪')} {signal.replace('_', ' ').upper()}")
        print(f"Strength: {strength_emoji.get(strength, '⚪')} {strength.upper()}")
        
        # RSI zones
        if rsi >= 80:
            print("📍 Extreme overbought zone (>80) - Strong sell signal")
        elif rsi >= 70:
            print("📍 Overbought zone (70-80) - Consider selling")
        elif rsi <= 20:
            print("📍 Extreme oversold zone (<20) - Strong buy signal")
        elif rsi <= 30:
            print("📍 Oversold zone (20-30) - Consider buying")
        elif rsi > 50:
            print("📍 Above midline - Bullish bias")
        else:
            print("📍 Below midline - Bearish bias")
            
        # Momentum analysis
        if momentum_signal != 'neutral':
            momentum_emoji = '🚀' if 'bullish' in momentum_signal else '📉'
            print(f"🔄 Momentum: {momentum_emoji} {momentum_signal.replace('_', ' ').upper()}")
            
        # Trend alignment
        if trend_direction != 'neutral':
            trend_emoji = '📈' if trend_direction == 'bullish' else '📉'
            print(f"📊 Trend: {trend_emoji} {trend_direction.upper()}")
            
        # Failure swing detection
        failure_swing = result.get('failure_swing')
        if failure_swing:
            swing_emoji = '🔄' if 'bullish' in failure_swing else '🔁'
            print(f"⚠️  {swing_emoji} {failure_swing.replace('_', ' ').upper()} detected!")