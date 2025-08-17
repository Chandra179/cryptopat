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


class ParabolicSAR:
    _config = None
    
    @classmethod
    def _load_config(cls):
        if cls._config is None:
            yaml_path = os.path.join(os.path.dirname(__file__), 'parabolic_sar.yaml')
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
        psar_config = self.config['parabolic_sar']
        
        # Get timeframe-specific parameters or use default (1d)
        timeframe_params = psar_config['timeframes'].get(timeframe, psar_config['timeframes']['1d'])
        general_params = psar_config['params']
        
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
        Calculate Parabolic SAR (Stop and Reverse) according to J. Welles Wilder Jr.'s methodology.
        
        Formula (Source: J. Welles Wilder Jr., "New Concepts in Technical Trading Systems", 1978):
        
        SAR Calculation:
        - SAR(t) = SAR(t-1) + AF × [EP - SAR(t-1)]
        
        Where:
        - SAR(t) = Current period's SAR value
        - SAR(t-1) = Previous period's SAR value
        - AF = Acceleration Factor (starts at 0.02, increases by 0.02 for each new extreme, max 0.20)
        - EP = Extreme Point (highest high during uptrend, lowest low during downtrend)
        
        Initialization:
        - First SAR = Previous period's low (for uptrend) or high (for downtrend)
        - Initial trend determined by comparing first two periods
        
        Rules:
        - Uptrend: SAR cannot exceed previous two periods' lows
        - Downtrend: SAR cannot exceed previous two periods' highs
        - Trend reversal occurs when price crosses SAR
        - AF resets to initial value (0.02) on trend reversal
        
        References:
        - Created by J. Welles Wilder Jr. (1978)
        - Book: "New Concepts in Technical Trading Systems"
        - TradingView: https://www.tradingview.com/support/solutions/43000502017-parabolic-sar/
        - Investopedia: https://www.investopedia.com/trading/introduction-to-parabolic-sar/
        - StockCharts: https://school.stockcharts.com/doku.php?id=technical_indicators:parabolic_sar
        
        Standard Parameters:
        - Initial AF: 0.02
        - AF Increment: 0.02
        - Maximum AF: 0.20
        
        Interpretation:
        - SAR below price: Uptrend (bullish)
        - SAR above price: Downtrend (bearish)
        - Price crosses SAR: Potential trend reversal
        """
        if not self.ohlcv or len(self.ohlcv) < 3:
            result = {
                "error": f"Insufficient data: need at least 3 candles, got {len(self.ohlcv) if self.ohlcv else 0}"
            }
            return result
            
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        
        n = len(df)
        sar = [0.0] * n
        af = [0.0] * n
        ep = [0.0] * n
        trend = [0] * n  # 1 for uptrend, -1 for downtrend
        
        af_initial = self.param["af_initial"]
        af_increment = self.param["af_increment"]
        af_maximum = self.param["af_maximum"]
        
        # Initialize first two periods
        if df['close'].iloc[1] > df['close'].iloc[0]:
            # Initial uptrend
            trend[0] = trend[1] = 1
            sar[0] = df['low'].iloc[0]
            sar[1] = df['low'].iloc[0]
            ep[0] = ep[1] = df['high'].iloc[1]
            af[0] = af[1] = af_initial
        else:
            # Initial downtrend
            trend[0] = trend[1] = -1
            sar[0] = df['high'].iloc[0]
            sar[1] = df['high'].iloc[0]
            ep[0] = ep[1] = df['low'].iloc[1]
            af[0] = af[1] = af_initial
        
        # Calculate SAR for remaining periods
        for i in range(2, n):
            prev_sar = sar[i-1]
            prev_af = af[i-1]
            prev_ep = ep[i-1]
            prev_trend = trend[i-1]
            
            # Calculate preliminary SAR
            sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
            trend[i] = prev_trend
            af[i] = prev_af
            ep[i] = prev_ep
            
            if prev_trend == 1:  # Uptrend
                # Check for trend reversal
                if df['low'].iloc[i] <= sar[i]:
                    # Trend reversal to downtrend
                    trend[i] = -1
                    sar[i] = prev_ep  # SAR becomes previous EP
                    af[i] = af_initial
                    ep[i] = df['low'].iloc[i]
                else:
                    # Continue uptrend
                    # SAR cannot exceed previous two lows
                    sar[i] = min(sar[i], df['low'].iloc[i-1], df['low'].iloc[i-2])
                    
                    # Update EP and AF if new high
                    if df['high'].iloc[i] > prev_ep:
                        ep[i] = df['high'].iloc[i]
                        af[i] = min(prev_af + af_increment, af_maximum)
            
            else:  # Downtrend
                # Check for trend reversal
                if df['high'].iloc[i] >= sar[i]:
                    # Trend reversal to uptrend
                    trend[i] = 1
                    sar[i] = prev_ep  # SAR becomes previous EP
                    af[i] = af_initial
                    ep[i] = df['high'].iloc[i]
                else:
                    # Continue downtrend
                    # SAR cannot exceed previous two highs
                    sar[i] = max(sar[i], df['high'].iloc[i-1], df['high'].iloc[i-2])
                    
                    # Update EP and AF if new low
                    if df['low'].iloc[i] < prev_ep:
                        ep[i] = df['low'].iloc[i]
                        af[i] = min(prev_af + af_increment, af_maximum)
        
        # Current values
        current_price = float(df['close'].iloc[-1])
        current_sar = float(sar[-1])
        current_trend = int(trend[-1])
        current_af = float(af[-1])
        current_ep = float(ep[-1])
        
        # Calculate trend strength
        trend_confirmation_periods = self.param["trend_confirmation_periods"]
        recent_trends = trend[-trend_confirmation_periods:]
        trend_strength = sum(1 for t in recent_trends if t == current_trend) / len(recent_trends)
        
        # Volume confirmation if enabled
        volume_confirmed = True
        if self.param["volume_confirmation"]:
            volume_ma_period = self.param["volume_ma_period"]
            if len(df) >= volume_ma_period:
                avg_volume = df['volume'].rolling(window=volume_ma_period).mean().iloc[-1]
                current_volume = df['volume'].iloc[-1]
                volume_confirmed = current_volume > avg_volume
        
        # Signal generation
        signal = "neutral"
        signal_strength = 0.0
        
        if current_trend == 1:
            signal = "bullish"
            signal_strength = trend_strength
            # Check for potential reversal
            distance_to_sar = (current_price - current_sar) / current_price
            if distance_to_sar < self.param["price_deviation_threshold"]:
                signal = "bullish_weakening"
                signal_strength *= 0.5
        else:
            signal = "bearish"
            signal_strength = trend_strength
            # Check for potential reversal
            distance_to_sar = (current_sar - current_price) / current_price
            if distance_to_sar < self.param["price_deviation_threshold"]:
                signal = "bearish_weakening"
                signal_strength *= 0.5
        
        # Check for recent trend change
        if len(trend) >= 2 and trend[-2] != current_trend:
            signal = "trend_reversal"
            signal_strength = 1.0
        
        # Apply volume confirmation
        if not volume_confirmed:
            signal_strength *= 0.7
        
        # Distance from SAR
        sar_distance_pct = abs(current_price - current_sar) / current_price * 100
        
        # Determine if price is above SAR and check for reversal
        price_above_sar = current_price > current_sar
        reversal_detected = len(trend) >= 2 and trend[-2] != current_trend
        
        # Map trend to simplified format
        trend_direction = "up" if current_trend == 1 else "down"
        
        # Build result based on YAML output configuration  
        output_config = self.config['parabolic_sar']['output']['fields']
        result = {}
        
        # Build result directly based on YAML fields
        for field_name in output_config:
            if field_name == "symbol":
                result[field_name] = self.symbol
            elif field_name == "timeframe":
                result[field_name] = self.timeframe
            elif field_name == "current_price":
                result[field_name] = current_price
            elif field_name == "sar":
                result[field_name] = current_sar
            elif field_name == "acceleration_factor":
                result[field_name] = current_af
            elif field_name == "signal":
                result[field_name] = signal
            elif field_name == "trend":
                result[field_name] = trend_direction
            elif field_name == "reversal":
                result[field_name] = reversal_detected
            elif field_name == "price_above_sar":
                result[field_name] = price_above_sar
            elif field_name == "parameters":
                result[field_name] = {
                    "af_initial": af_initial,
                    "af_increment": af_increment,
                    "af_maximum": af_maximum,
                    "trend_confirmation_periods": trend_confirmation_periods,
                    "volume_confirmation": self.param["volume_confirmation"]
                }
        
        self.print_output(result)
        return result
    
    def print_output(self, result):
        """Print analysis summary for Parabolic SAR indicator"""
        if "error" in result:
            print(f"⚠️  Parabolic SAR Error: {result['error']}")
            return
            
        symbol = result.get('symbol', 'N/A')
        timeframe = result.get('timeframe', 'N/A')
        signal = result.get('signal', 'neutral')
        current_price = result.get('current_price', 0)
        sar = result.get('sar', 0)
        trend = result.get('trend', 'neutral')
        acceleration_factor = result.get('acceleration_factor', 0)
        price_above_sar = result.get('price_above_sar', False)
        reversal = result.get('reversal', False)
        
        print(f"\n🌟 Parabolic SAR Analysis - {symbol} ({timeframe})")
        print(f"Current Price: ${current_price:.4f}")
        print(f"SAR: ${sar:.4f}")
        print(f"Acceleration Factor: {acceleration_factor:.4f}")
        
        # Signal interpretation
        signal_emoji = {
            'bullish': '🟢',
            'bearish': '🔴',
            'trend_reversal': '🔄',
            'bullish_weakening': '🟡',
            'bearish_weakening': '🟡',
            'neutral': '⚪'
        }
        
        trend_emoji = {
            'up': '📈',
            'down': '📉',
            'neutral': '📊'
        }
        
        print(f"Signal: {signal_emoji.get(signal, '⚪')} {signal.upper()}")
        print(f"Trend: {trend_emoji.get(trend, '📊')} {trend.upper()}")
        
        # Reversal detection
        if reversal:
            if trend == 'up':
                print("🚀 Bullish reversal detected - SAR switched below price!")
            else:
                print("📉 Bearish reversal detected - SAR switched above price!")
        
        # Price position analysis
        if price_above_sar:
            distance_pct = (current_price - sar) / current_price * 100
            print(f"📈 Price above SAR (+{distance_pct:.2f}%) - uptrend confirmed")
            print(f"🛡️  SAR acting as support at ${sar:.4f}")
        else:
            distance_pct = (sar - current_price) / current_price * 100
            print(f"📉 Price below SAR (-{distance_pct:.2f}%) - downtrend confirmed")
            print(f"⚠️  SAR acting as resistance at ${sar:.4f}")
        
        # Acceleration factor analysis
        if acceleration_factor > 0.15:
            print("🔥 High acceleration factor - strong momentum!")
        elif acceleration_factor > 0.10:
            print("⚡ Medium acceleration factor - building momentum")
        else:
            print("🐌 Low acceleration factor - slow trend development")
        
        # Signal-specific insights
        if signal == 'bullish':
            print("💡 Consider long positions - strong uptrend momentum")
        elif signal == 'bearish':
            print("💡 Consider short positions - strong downtrend momentum")
        elif signal == 'trend_reversal':
            print("💡 Major trend change - reassess position strategy")
        elif 'weakening' in signal:
            print("💡 Trend momentum weakening - prepare for potential reversal")
        elif trend == 'up':
            print("💡 Stay bullish - uptrend continues with SAR support")
        elif trend == 'down':
            print("💡 Stay bearish - downtrend continues with SAR resistance")
