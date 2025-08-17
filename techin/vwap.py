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

class VWAP:
    _config = None
    
    @classmethod
    def _load_config(cls):
        if cls._config is None:
            yaml_path = os.path.join(os.path.dirname(__file__), 'vwap.yaml')
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
        vwap_config = self.config['vwap']
        
        # Get parameters from YAML config
        self.param = vwap_config['params']
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.ticker = ticker
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
    
    def calculate(self):
        """
        Calculate VWAP (Volume Weighted Average Price) according to standard methodology.
        
        Formula (Source: Investopedia, TradingView, Wikipedia):
        VWAP = Σ(Typical Price × Volume) / Σ(Volume)
        
        Where:
        - Typical Price = (High + Low + Close) / 3 (standard)
        - Alternative: OHLC4 = (Open + High + Low + Close) / 4
        - Summation is typically done from session start (market open)
        
        Key Characteristics:
        - Acts as both support and resistance level
        - Used by institutional traders for execution benchmarking
        - Resets at the beginning of each trading session
        - Higher timeframes provide stronger VWAP levels
        
        References:
        - Investopedia: https://www.investopedia.com/terms/v/vwap.asp
        - TradingView: https://www.tradingview.com/support/solutions/43000501613-volume-weighted-average-price-vwap/
        - Wikipedia: https://en.wikipedia.org/wiki/Volume-weighted_average_price
        - CFA Institute: Volume-Weighted Average Price as execution benchmark
        
        Trading Applications:
        - Price above VWAP: Bullish bias (buyers in control)
        - Price below VWAP: Bearish bias (sellers in control)
        - VWAP acts as dynamic support/resistance
        - Volume spikes near VWAP indicate institutional interest
        """
        if not self.ohlcv or len(self.ohlcv) < 2:
            result = {
                "error": f"Insufficient data: need at least 2 candles, got {len(self.ohlcv) if self.ohlcv else 0}"
            }
            return result
            
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # Calculate typical price based on selected source
        if self.param["use_ohlc4"]:
            typical_price = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            price_source_name = "OHLC4"
        elif self.param["use_close"]:
            typical_price = df['close']
            price_source_name = "Close"
        elif self.param["use_hl2"]:
            typical_price = (df['high'] + df['low']) / 2
            price_source_name = "HL2"
        else:  # HLC3 (default)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            price_source_name = "HLC3"
        
        # Filter by volume threshold if specified
        volume_mask = df['volume'] >= self.param["volume_threshold"]
        filtered_volume = df['volume'].where(volume_mask, 0)
        filtered_typical_price = typical_price.where(volume_mask, 0)

        df['price_volume'] = filtered_typical_price * filtered_volume
        df['cumulative_pv'] = df['price_volume'].cumsum()
        df['cumulative_volume'] = filtered_volume.cumsum()
        df['vwap'] = df['cumulative_pv'] / df['cumulative_volume']
        df['vwap'] = df['vwap'].fillna(typical_price)
        
        # Calculate rolling VWAP if periods_lookback is specified
        if self.param["periods_lookback"]:
            periods = self.param["periods_lookback"]
            rolling_pv = (filtered_typical_price * filtered_volume).rolling(window=periods).sum()
            rolling_volume = filtered_volume.rolling(window=periods).sum()
            df['rolling_vwap'] = rolling_pv / rolling_volume
            df['rolling_vwap'] = df['rolling_vwap'].fillna(typical_price)
        
        # Calculate VWAP deviation bands if enabled
        upper_bands = []
        lower_bands = []
        if self.param["deviation_bands"]:
            for multiplier in self.param["std_dev_multiplier"]:
                # Calculate price deviation from VWAP
                price_diff = typical_price - df['vwap']
                weighted_variance = ((price_diff ** 2) * filtered_volume).cumsum() / df['cumulative_volume']
                weighted_std = np.sqrt(weighted_variance.fillna(0))
                
                upper_band = df['vwap'] + (weighted_std * multiplier)
                lower_band = df['vwap'] - (weighted_std * multiplier)
                
                upper_bands.append(float(upper_band.iloc[-1]))
                lower_bands.append(float(lower_band.iloc[-1]))
        
        # Current values
        current_price = float(df['close'].iloc[-1])
        current_vwap = float(df['vwap'].iloc[-1])
        current_volume = float(df['volume'].iloc[-1])
        current_typical_price = float(typical_price.iloc[-1])
        
        # Calculate session statistics
        session_high = float(df['high'].max())
        session_low = float(df['low'].min())
        total_volume = float(df['volume'].sum())
        avg_volume = float(df['volume'].mean())
        
        # Price position relative to VWAP
        price_above_vwap = current_price > current_vwap
        price_deviation_pct = ((current_price - current_vwap) / current_vwap) * 100
        
        # Cap extreme deviations (proven method: anything >100% suggests data quality issues)
        if abs(price_deviation_pct) > 100:
            # Log the extreme deviation for investigation
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"VWAP extreme deviation detected: {price_deviation_pct:.1f}% for {self.symbol} {self.timeframe}")
            # Cap at +/-100% but preserve sign
            price_deviation_pct = 100 if price_deviation_pct > 0 else -100
        
        # Volume analysis
        volume_spike = current_volume > (avg_volume * self.param["volume_spike_threshold"])
        
        # Trend analysis
        trend_periods = min(self.param["trend_confirmation_periods"], len(df))
        if trend_periods > 1:
            recent_vwap = df['vwap'].iloc[-trend_periods:]
            vwap_slope = (recent_vwap.iloc[-1] - recent_vwap.iloc[0]) / (trend_periods - 1)
            vwap_trend = "bullish" if vwap_slope > 0 else "bearish" if vwap_slope < 0 else "neutral"
        else:
            vwap_slope = 0
            vwap_trend = "neutral"
        
        # Signal generation
        signal = "neutral"
        if abs(price_deviation_pct) > (self.param["price_deviation_threshold"] * 100):
            if price_above_vwap:
                signal = "bullish" if volume_spike else "overbought"
            else:
                signal = "bearish" if volume_spike else "oversold"
        elif price_above_vwap and vwap_trend == "bullish":
            signal = "bullish"
        elif not price_above_vwap and vwap_trend == "bearish":
            signal = "bearish"
        
        # Build result based on YAML output configuration
        output_config = self.config['vwap']['output']['fields']
        result = {}
        
        # Build result directly based on YAML fields
        for field_name in output_config:
            if field_name == "symbol":
                result[field_name] = self.symbol
            elif field_name == "timeframe":
                result[field_name] = self.timeframe
            elif field_name == "current_price":
                result[field_name] = current_price
            elif field_name == "vwap":
                result[field_name] = current_vwap
            elif field_name == "typical_price":
                result[field_name] = current_typical_price
            elif field_name == "price_source":
                result[field_name] = price_source_name
            elif field_name == "price_above_vwap":
                result[field_name] = price_above_vwap
            elif field_name == "price_deviation_pct":
                result[field_name] = price_deviation_pct
            elif field_name == "volume_spike":
                result[field_name] = volume_spike
            elif field_name == "signal":
                result[field_name] = signal
            elif field_name == "trend":
                result[field_name] = vwap_trend
            elif field_name == "vwap_slope":
                result[field_name] = float(vwap_slope)
            elif field_name == "rolling_vwap" and self.param["periods_lookback"]:
                result[field_name] = float(df['rolling_vwap'].iloc[-1])
            elif field_name == "session_stats":
                result[field_name] = {
                    "high": session_high,
                    "low": session_low,
                    "total_volume": total_volume,
                    "avg_volume": avg_volume,
                    "current_volume": current_volume
                }
            elif field_name == "deviation_bands" and self.param["deviation_bands"] and upper_bands and lower_bands:
                result[field_name] = {
                    "upper_bands": upper_bands,
                    "lower_bands": lower_bands,
                    "multipliers": self.param["std_dev_multiplier"]
                }
            elif field_name == "parameters":
                result[field_name] = {
                    "price_source": self.param["price_source"],
                    "volume_threshold": self.param["volume_threshold"],
                    "deviation_bands": self.param["deviation_bands"],
                    "periods_lookback": self.param["periods_lookback"]
                }
        
        self.print_output(result)
        return result
    
    def print_output(self, result):
        """Print analysis summary for VWAP indicator"""
        if "error" in result:
            print(f"⚠️  VWAP Error: {result['error']}")
            return
            
        symbol = result.get('symbol', 'N/A')
        timeframe = result.get('timeframe', 'N/A')
        signal = result.get('signal', 'neutral')
        current_price = result.get('current_price', 0)
        vwap = result.get('vwap', 0)
        price_above_vwap = result.get('price_above_vwap', False)
        price_deviation_pct = result.get('price_deviation_pct', 0)
        volume_spike = result.get('volume_spike', False)
        trend = result.get('trend', 'neutral')
        
        print(f"\n💰 VWAP Analysis - {symbol} ({timeframe})")
        print(f"Current Price: ${current_price:.4f}")
        print(f"VWAP: ${vwap:.4f}")
        print(f"Deviation: {price_deviation_pct:+.2f}%")
        
        # Signal interpretation
        signal_emoji = {
            'bullish': '🟢',
            'bearish': '🔴',
            'overbought': '🔴',
            'oversold': '🟢',
            'neutral': '⚪'
        }
        
        print(f"Signal: {signal_emoji.get(signal, '⚪')} {signal.upper()}")
        print(f"Trend: {trend.upper()}")
        
        # Price position analysis
        if price_above_vwap:
            print("📈 Price trading ABOVE VWAP - bullish bias")
            if price_deviation_pct > 2:
                print("⚠️  Significant deviation above VWAP - potential reversal")
        else:
            print("📉 Price trading BELOW VWAP - bearish bias")
            if price_deviation_pct < -2:
                print("⚠️  Significant deviation below VWAP - potential reversal")
        
        # Volume analysis
        if volume_spike:
            print("📊 Volume spike detected - institutional interest!")
        
        # Session statistics
        session_stats = result.get('session_stats', {})
        if session_stats:
            print(f"📊 Session: High ${session_stats.get('high', 0):.4f} | Low ${session_stats.get('low', 0):.4f}")
            print(f"📊 Volume: Current {session_stats.get('current_volume', 0):.2f} | Avg {session_stats.get('avg_volume', 0):.2f}")
        
        # Deviation bands if available
        deviation_bands = result.get('deviation_bands', {})
        if deviation_bands:
            upper_bands = deviation_bands.get('upper_bands', [])
            lower_bands = deviation_bands.get('lower_bands', [])
            if upper_bands and lower_bands:
                print(f"📏 Bands: Upper ${upper_bands[0]:.4f} | Lower ${lower_bands[0]:.4f}")
