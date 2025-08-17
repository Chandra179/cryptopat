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
import yaml
import os

class Supertrend:
    _config = None
    
    @classmethod
    def _load_config(cls):
        if cls._config is None:
            yaml_path = os.path.join(os.path.dirname(__file__), 'supertrend.yaml')
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
        st_config = self.config['supertrend']
        
        # Get timeframe-specific parameters or use default (1d)
        timeframe_params = st_config['timeframes'].get(timeframe, st_config['timeframes']['1d'])
        general_params = st_config['params']
        
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
        Calculate Supertrend according to Olivier Seban's original methodology.
        
        Formula (Source: Olivier Seban, TradingView):
        1. HL2 = (High + Low) / 2
        2. ATR = Average True Range over N periods
        3. Basic Upper Band = HL2 + (Multiplier × ATR)
        4. Basic Lower Band = HL2 - (Multiplier × ATR)
        5. Final Upper Band = Basic Upper Band < Previous Final Upper Band OR Previous Close > Previous Final Upper Band ? Basic Upper Band : Previous Final Upper Band
        6. Final Lower Band = Basic Lower Band > Previous Final Lower Band OR Previous Close < Previous Final Lower Band ? Basic Lower Band : Previous Final Lower Band
        7. Supertrend = Close <= Final Lower Band ? Final Upper Band : Final Lower Band
        8. Trend = Close <= Final Lower Band ? Down : Up
        
        Standard Parameters:
        - ATR Period: 10 periods (default)
        - Multiplier: 3.0 (default)
        - Source: HL2 (High + Low) / 2
        
        References:
        - Created by Olivier Seban
        - TradingView: https://www.tradingview.com/support/solutions/43000502284-supertrend/
        - Fidelity: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/supertrend
        - Original Paper: Olivier Seban's "SuperTrend" methodology
        
        Key Features:
        - Trend-following indicator using ATR-based dynamic support/resistance
        - Adapts to market volatility through ATR calculation
        - Provides clear buy/sell signals with trend direction
        - Less prone to whipsaws compared to simple moving averages
        """
        if not self.ohlcv or len(self.ohlcv) < max(self.param["atr_period"], 14):
            result = {
                "error": f"Insufficient data: need at least {max(self.param['atr_period'], 14)} candles, got {len(self.ohlcv) if self.ohlcv else 0}"
            }
            return result
            
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        atr_period = self.param["atr_period"]
        multiplier = self.param["multiplier"]
        
        # Calculate HL2 (typical price)
        if self.param["use_hl2"]:
            hl2 = (df['high'] + df['low']) / 2
        else:
            hl2 = df['close']
        
        # Calculate ATR (Average True Range)
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=atr_period).mean()
        
        # Calculate basic bands
        df['basic_upper'] = hl2 + (multiplier * df['atr'])
        df['basic_lower'] = hl2 - (multiplier * df['atr'])
        
        # Initialize final bands
        df['final_upper'] = df['basic_upper'].copy()
        df['final_lower'] = df['basic_lower'].copy()
        
        # Calculate final bands with logic
        for i in range(1, len(df)):
            # Final Upper Band logic
            if (df.loc[i, 'basic_upper'] < df.loc[i-1, 'final_upper']) or (df.loc[i-1, 'close'] > df.loc[i-1, 'final_upper']):
                df.loc[i, 'final_upper'] = df.loc[i, 'basic_upper']
            else:
                df.loc[i, 'final_upper'] = df.loc[i-1, 'final_upper']
            
            # Final Lower Band logic
            if (df.loc[i, 'basic_lower'] > df.loc[i-1, 'final_lower']) or (df.loc[i-1, 'close'] < df.loc[i-1, 'final_lower']):
                df.loc[i, 'final_lower'] = df.loc[i, 'basic_lower']
            else:
                df.loc[i, 'final_lower'] = df.loc[i-1, 'final_lower']
        
        # Calculate Supertrend and Trend
        df['supertrend'] = np.where(df['close'] <= df['final_lower'], df['final_upper'], df['final_lower'])
        df['trend'] = np.where(df['close'] <= df['final_lower'], 'down', 'up')
        
        # Trend change detection
        df['prev_trend'] = df['trend'].shift(1)
        df['trend_change'] = df['trend'] != df['prev_trend']
        
        # Signal generation
        df['signal'] = 'hold'
        df.loc[(df['trend'] == 'up') & (df['prev_trend'] == 'down'), 'signal'] = 'buy'
        df.loc[(df['trend'] == 'down') & (df['prev_trend'] == 'up'), 'signal'] = 'sell'
        
        # Current values
        current_price = float(df['close'].iloc[-1])
        current_supertrend = float(df['supertrend'].iloc[-1])
        current_trend = df['trend'].iloc[-1]
        current_signal = df['signal'].iloc[-1]
        current_atr = float(df['atr'].iloc[-1])
        current_upper = float(df['final_upper'].iloc[-1])
        current_lower = float(df['final_lower'].iloc[-1])
        
        # Calculate additional metrics
        trend_changes = df['trend_change'].rolling(window=10).sum().iloc[-1] if len(df) >= 10 else 0
        distance_from_supertrend = abs(current_price - current_supertrend) / current_price * 100
        
        # Volume confirmation if enabled
        volume_confirmed = True
        if self.param["volume_confirmation"]:
            avg_volume = df['volume'].rolling(window=self.param["volume_ma_period"]).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_confirmed = current_volume > avg_volume
        
        # Signal strength calculation
        signal_strength = "weak"
        if distance_from_supertrend < 1.0:
            signal_strength = "strong"
        elif distance_from_supertrend < 2.0:
            signal_strength = "medium"
        
        # Market condition assessment
        recent_volatility = df['atr'].rolling(window=5).mean().iloc[-1] if len(df) >= 5 else current_atr
        volatility_ratio = current_atr / recent_volatility if recent_volatility > 0 else 1.0
        
        market_condition = "normal"
        if volatility_ratio > 1.5:
            market_condition = "high_volatility"
        elif volatility_ratio < 0.7:
            market_condition = "low_volatility"
        
        # Build result based on YAML output configuration  
        output_config = self.config['supertrend']['output']['fields']
        result = {}
        
        # Build result directly based on YAML fields
        for field_name in output_config:
            if field_name == "symbol":
                result[field_name] = self.symbol
            elif field_name == "timeframe":
                result[field_name] = self.timeframe
            elif field_name == "current_price":
                result[field_name] = current_price
            elif field_name == "supertrend":
                result[field_name] = current_supertrend
            elif field_name == "atr":
                result[field_name] = current_atr
            elif field_name == "hl2":
                hl2_value = (df['high'].iloc[-1] + df['low'].iloc[-1]) / 2 if self.param["use_hl2"] else current_price
                result[field_name] = float(hl2_value)
            elif field_name == "signal":
                result[field_name] = current_signal
            elif field_name == "trend":
                result[field_name] = current_trend
            elif field_name == "price_above_supertrend":
                result[field_name] = current_price > current_supertrend
            elif field_name == "trend_change":
                result[field_name] = df['trend_change'].iloc[-1] if len(df) > 1 else False
            elif field_name == "support_resistance":
                result[field_name] = current_supertrend
            elif field_name == "parameters":
                result[field_name] = {
                    "atr_period": atr_period,
                    "multiplier": multiplier,
                    "use_hl2": self.param["use_hl2"],
                    "volume_confirmation": self.param["volume_confirmation"]
                }
        
        self.print_output(result)
        return result
    
    def print_output(self, result):
        """Print analysis summary for Supertrend indicator"""
        if "error" in result:
            print(f"⚠️  Supertrend Error: {result['error']}")
            return
            
        symbol = result.get('symbol', 'N/A')
        timeframe = result.get('timeframe', 'N/A')
        signal = result.get('signal', 'hold')
        current_price = result.get('current_price', 0)
        supertrend = result.get('supertrend', 0)
        trend = result.get('trend', 'neutral')
        atr = result.get('atr', 0)
        price_above_supertrend = result.get('price_above_supertrend', False)
        trend_change = result.get('trend_change', False)
        support_resistance = result.get('support_resistance', 0)
        
        print(f"\n🎯 Supertrend Analysis - {symbol} ({timeframe})")
        print(f"Current Price: ${current_price:.4f}")
        print(f"Supertrend: ${supertrend:.4f}")
        print(f"ATR: ${atr:.4f}")
        
        # Signal interpretation
        signal_emoji = {
            'buy': '🟢',
            'sell': '🔴',
            'hold': '⚪'
        }
        
        trend_emoji = {
            'up': '📈',
            'down': '📉'
        }
        
        print(f"Signal: {signal_emoji.get(signal, '⚪')} {signal.upper()}")
        print(f"Trend: {trend_emoji.get(trend, '📊')} {trend.upper()}")
        
        # Trend change detection
        if trend_change:
            if trend == 'up':
                print("🚀 Trend change detected - Bullish reversal!")
            else:
                print("📉 Trend change detected - Bearish reversal!")
        
        # Price position analysis
        if price_above_supertrend:
            print("📈 Price above Supertrend - uptrend confirmed")
            print(f"🛡️  Support level: ${support_resistance:.4f}")
        else:
            print("📉 Price below Supertrend - downtrend confirmed")
            print(f"⚠️  Resistance level: ${support_resistance:.4f}")
        
        # Distance analysis
        distance_pct = abs(current_price - supertrend) / current_price * 100
        if distance_pct < 1.0:
            print(f"🎯 Price very close to Supertrend ({distance_pct:.2f}%) - potential reversal zone")
        elif distance_pct < 2.0:
            print(f"📍 Price moderately close to Supertrend ({distance_pct:.2f}%)")
        else:
            print(f"📏 Price distant from Supertrend ({distance_pct:.2f}%) - strong trend")
        
        # Signal-specific insights
        if signal == 'buy':
            print("💡 Consider long positions - trend turning bullish")
        elif signal == 'sell':
            print("💡 Consider short positions - trend turning bearish")
        elif trend == 'up':
            print("💡 Stay bullish - uptrend continues")
        elif trend == 'down':
            print("💡 Stay bearish - downtrend continues")
