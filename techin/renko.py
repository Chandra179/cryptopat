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

class Renko:
    _config = None
    
    @classmethod
    def _load_config(cls):
        if cls._config is None:
            yaml_path = os.path.join(os.path.dirname(__file__), 'renko.yaml')
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
        renko_config = self.config['renko']
        
        # Get timeframe-specific parameters or use default (1d)
        timeframe_params = renko_config['timeframes'].get(timeframe, renko_config['timeframes']['1d'])
        general_params = renko_config['params']
        
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
        Calculate Renko chart according to Steve Nison's methodology.
        
        Renko charts filter out time and focus purely on price movement,
        creating bricks only when price moves by a predetermined amount.
        
        Algorithm:
        1. Determine brick size (fixed, ATR-based, or percentage-based)
        2. Starting from first price, create bricks when price moves >= brick_size
        3. For upward movement: create white/green bricks
        4. For downward movement: create black/red bricks
        5. Reversal requires movement of 2x brick_size in opposite direction
        
        Formula (Source: Steve Nison, TradingView):
        - Brick Size (ATR): ATR(period) × multiplier
        - Brick Size (Percent): Current Price × (percent / 100)
        - New Brick Condition: |Price - Last Brick Close| >= Brick Size
        - Reversal Condition: |Price - Last Brick Close| >= (2 × Brick Size)
        
        References:
        - Created by Steve Nison, popularized in "Beyond Candlesticks" (1994)
        - Originally from Japanese rice trading (18th century)
        - TradingView: https://www.tradingview.com/support/solutions/43000502284-renko/
        - Fidelity: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/renko-chart
        - Wikipedia: https://en.wikipedia.org/wiki/Renko_chart
        
        Key Characteristics:
        - Time-independent: Only price movement matters
        - Trend clarity: Easier to spot trends without noise
        - Support/Resistance: Clear levels from brick patterns
        - Signal generation: Breakouts from brick patterns
        """
        if not self.ohlcv or len(self.ohlcv) < 20:
            result = {
                "error": f"Insufficient data: need at least 20 candles, got {len(self.ohlcv) if self.ohlcv else 0}"
            }
            return result
            
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['open'] = pd.to_numeric(df['open'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # Calculate price source
        if self.param["price_source"] == "typical":
            prices = (df['high'] + df['low'] + df['close']) / 3
        elif self.param["price_source"] == "hl2":
            prices = (df['high'] + df['low']) / 2
        else:  # close
            prices = df['close']
        
        # Calculate brick size
        brick_size = self._calculate_brick_size(df, prices)
        
        # Generate Renko bricks
        renko_data = self._generate_renko_bricks(df, prices, brick_size)
        
        if len(renko_data) < 2:
            result = {
                "error": "Insufficient price movement to generate Renko bricks"
            }
            return result
        
        # Analyze current trend and signals
        trend_analysis = self._analyze_trend(renko_data)
        signals = self._generate_signals(renko_data)
        
        # Current values
        current_price = float(prices.iloc[-1])
        last_brick = renko_data[-1]
        
        # Map signals to match YAML enum values
        signal_mapping = {
            "bullish": "bullish",
            "bearish": "bearish", 
            "reversal": "reversal_up" if last_brick['direction'] > 0 else "reversal_down",
            "breakout": "bullish" if last_brick['direction'] > 0 else "bearish",
            "neutral": "neutral"
        }
        mapped_signal = signal_mapping.get(signals["primary_signal"], "neutral")
        
        # Map trend to match YAML enum values
        trend_mapping = {
            "bullish": "up",
            "bearish": "down",
            "neutral": "neutral"
        }
        mapped_trend = trend_mapping.get(trend_analysis["current_trend"], "neutral")
        
        # Build result based on YAML output configuration
        output_config = self.config['renko']['output']['fields']
        result = {}
        
        # Build result directly based on YAML fields
        for field_name in output_config:
            if field_name == "symbol":
                result[field_name] = self.symbol
            elif field_name == "timeframe":
                result[field_name] = self.timeframe
            elif field_name == "current_price":
                result[field_name] = current_price
            elif field_name == "brick_size":
                result[field_name] = brick_size
            elif field_name == "last_brick_price":
                result[field_name] = last_brick['close']
            elif field_name == "signal":
                result[field_name] = mapped_signal
            elif field_name == "trend":
                result[field_name] = mapped_trend
            elif field_name == "new_brick":
                result[field_name] = True  # Always true if we have bricks
            elif field_name == "brick_direction":
                result[field_name] = "up" if last_brick['direction'] > 0 else "down"
            elif field_name == "consecutive_bricks":
                result[field_name] = trend_analysis["consecutive_bricks"]
            elif field_name == "reversal_detected":
                result[field_name] = signals["reversal_alert"]
            elif field_name == "parameters":
                result[field_name] = {
                    "brick_size": brick_size,
                    "auto_brick_method": self.param["auto_brick_method"],
                    "atr_period": self.param["atr_period"] if self.param["auto_brick_method"] == "atr" else None,
                    "atr_multiplier": self.param["atr_multiplier"] if self.param["auto_brick_method"] == "atr" else None,
                    "percent_size": self.param["percent_size"] if self.param["auto_brick_method"] == "percent" else None,
                    "price_source": self.param["price_source"],
                    "min_trend_bricks": self.param["min_trend_bricks"]
                }
        
        self.print_output(result)
        return result
    
    def _calculate_brick_size(self, df: pd.DataFrame, prices: pd.Series) -> float:
        """Calculate brick size based on the selected method."""
        if self.param["brick_size"] is not None:
            return float(self.param["brick_size"])
        
        method = self.param["auto_brick_method"]
        
        if method == "atr":
            # ATR-based brick size (Steve Nison's preferred method)
            atr_period = self.param["atr_period"]
            high_low = df['high'] - df['low']
            high_close_prev = abs(df['high'] - df['close'].shift(1))
            low_close_prev = abs(df['low'] - df['close'].shift(1))
            
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = true_range.rolling(window=atr_period).mean().iloc[-1]
            
            brick_size = atr * self.param["atr_multiplier"]
            
        elif method == "percent":
            # Percentage-based brick size
            current_price = prices.iloc[-1]
            brick_size = current_price * (self.param["percent_size"] / 100)
            
        else:  # fixed - use a reasonable default
            # Use 0.5% of current price as default
            current_price = prices.iloc[-1]
            brick_size = current_price * 0.005
        
        # Ensure brick size is reasonable (not too small or too large)
        current_price = prices.iloc[-1]
        min_brick_size = current_price * 0.0001  # 0.01%
        max_brick_size = current_price * 0.05    # 5%
        
        return max(min_brick_size, min(max_brick_size, brick_size))
    
    def _generate_renko_bricks(self, df: pd.DataFrame, prices: pd.Series, brick_size: float) -> List[Dict]:
        """Generate Renko bricks from price data."""
        bricks = []
        
        if len(prices) == 0:
            return bricks
        
        # Initialize with first price
        current_brick_close = float(prices.iloc[0])
        current_direction = 0  # 0 = unknown, 1 = up, -1 = down
        
        for i, (_, row) in enumerate(df.iterrows()):
            price = float(prices.iloc[i])
            timestamp = int(row['timestamp'])
            
            if self.param["wick_calculation"]:
                # Consider high and low wicks
                high = float(row['high'])
                low = float(row['low'])
                test_prices = [price, high, low]
            else:
                test_prices = [price]
            
            for test_price in test_prices:
                # Check for new bricks
                price_diff = test_price - current_brick_close
                
                if abs(price_diff) >= brick_size:
                    # Determine number of bricks to create
                    num_bricks = int(abs(price_diff) / brick_size)
                    brick_direction = 1 if price_diff > 0 else -1
                    
                    # Check for reversal (need 2x brick size move in opposite direction)
                    if current_direction != 0 and brick_direction != current_direction:
                        if abs(price_diff) < (brick_size * self.param["trend_reversal_bricks"]):
                            continue  # Not enough movement for reversal
                    
                    # Create bricks
                    for _ in range(num_bricks):
                        brick_open = current_brick_close
                        
                        if brick_direction > 0:
                            brick_close = current_brick_close + brick_size
                        else:
                            brick_close = current_brick_close - brick_size
                        
                        bricks.append({
                            'open': brick_open,
                            'close': brick_close,
                            'direction': brick_direction,
                            'timestamp': timestamp,
                            'brick_number': len(bricks) + 1
                        })
                        
                        current_brick_close = brick_close
                        current_direction = brick_direction
        
        return bricks
    
    def _analyze_trend(self, renko_data: List[Dict]) -> Dict:
        """Analyze trend from Renko bricks."""
        if len(renko_data) < 3:
            return {
                "current_trend": "neutral",
                "trend_strength": 0,
                "consecutive_bricks": 0,
                "support_level": None,
                "resistance_level": None
            }
        
        # Count consecutive bricks in same direction
        consecutive_bricks = 1
        current_direction = renko_data[-1]['direction']
        
        for i in range(len(renko_data) - 2, -1, -1):
            if renko_data[i]['direction'] == current_direction:
                consecutive_bricks += 1
            else:
                break
        
        # Determine trend
        if consecutive_bricks >= self.param["min_trend_bricks"]:
            if current_direction > 0:
                current_trend = "bullish"
            else:
                current_trend = "bearish"
        else:
            current_trend = "neutral"
        
        # Calculate trend strength (0-1 based on consecutive bricks)
        max_consecutive = min(consecutive_bricks, 10)  # Cap at 10 for calculation
        trend_strength = max_consecutive / 10
        
        # Find support and resistance levels
        lookback = min(len(renko_data), self.param["breakout_lookback"] * 2)
        recent_bricks = renko_data[-lookback:]
        
        lows = [brick['close'] if brick['direction'] < 0 else brick['open'] for brick in recent_bricks]
        highs = [brick['close'] if brick['direction'] > 0 else brick['open'] for brick in recent_bricks]
        
        support_level = min(lows) if lows else None
        resistance_level = max(highs) if highs else None
        
        return {
            "current_trend": current_trend,
            "trend_strength": trend_strength,
            "consecutive_bricks": consecutive_bricks,
            "support_level": support_level,
            "resistance_level": resistance_level
        }
    
    def _generate_signals(self, renko_data: List[Dict]) -> Dict:
        """Generate trading signals from Renko analysis."""
        if len(renko_data) < 3:
            return {
                "primary_signal": "neutral",
                "signal_strength": 0,
                "reversal_alert": False,
                "breakout_alert": False
            }
        
        last_brick = renko_data[-1]
        prev_bricks = renko_data[-3:]  # Last 3 bricks for pattern analysis
        
        # Check for trend reversal
        reversal_alert = False
        if len(prev_bricks) >= 2:
            # Look for direction change after sustained trend
            directions = [brick['direction'] for brick in prev_bricks]
            if directions[-1] != directions[-2] and abs(sum(directions[:-1])) >= 2:
                reversal_alert = True
        
        # Check for breakout
        breakout_alert = False
        if len(renko_data) >= self.param["breakout_lookback"]:
            recent_range = renko_data[-self.param["breakout_lookback"]:]
            range_high = max([brick['close'] if brick['direction'] > 0 else brick['open'] for brick in recent_range])
            range_low = min([brick['close'] if brick['direction'] < 0 else brick['open'] for brick in recent_range])
            
            current_price = last_brick['close']
            if current_price > range_high or current_price < range_low:
                breakout_alert = True
        
        # Primary signal generation
        consecutive_up = 0
        consecutive_down = 0
        for brick in reversed(renko_data[-5:]):  # Last 5 bricks
            if brick['direction'] > 0:
                consecutive_up += 1
                break
            elif brick['direction'] < 0:
                consecutive_down += 1
                break
        
        # Generate primary signal
        if consecutive_up >= self.param["min_trend_bricks"]:
            primary_signal = "bullish"
            signal_strength = min(consecutive_up / 5, 1.0)
        elif consecutive_down >= self.param["min_trend_bricks"]:
            primary_signal = "bearish"
            signal_strength = min(consecutive_down / 5, 1.0)
        elif reversal_alert:
            primary_signal = "reversal"
            signal_strength = 0.7
        elif breakout_alert:
            primary_signal = "breakout"
            signal_strength = 0.8
        else:
            primary_signal = "neutral"
            signal_strength = 0.0
        
        return {
            "primary_signal": primary_signal,
            "signal_strength": signal_strength,
            "reversal_alert": reversal_alert,
            "breakout_alert": breakout_alert
        }
    
    def print_output(self, result):
        """Print analysis summary for Renko indicator"""
        if "error" in result:
            print(f"⚠️  Renko Error: {result['error']}")
            return
            
        symbol = result.get('symbol', 'N/A')
        timeframe = result.get('timeframe', 'N/A')
        signal = result.get('signal', 'neutral')
        current_price = result.get('current_price', 0)
        brick_size = result.get('brick_size', 0)
        last_brick_price = result.get('last_brick_price', 0)
        trend = result.get('trend', 'neutral')
        brick_direction = result.get('brick_direction', 'neutral')
        consecutive_bricks = result.get('consecutive_bricks', 0)
        reversal_detected = result.get('reversal_detected', False)
        new_brick = result.get('new_brick', False)
        
        print(f"\n🧱 Renko Analysis - {symbol} ({timeframe})")
        print(f"Current Price: ${current_price:.4f}")
        print(f"Brick Size: ${brick_size:.4f}")
        print(f"Last Brick: ${last_brick_price:.4f}")
        
        # Signal interpretation
        signal_emoji = {
            'bullish': '🟢',
            'bearish': '🔴',
            'reversal_up': '🔄',
            'reversal_down': '🔄',
            'neutral': '⚪'
        }
        
        trend_emoji = {
            'up': '📈',
            'down': '📉',
            'neutral': '📊'
        }
        
        print(f"Signal: {signal_emoji.get(signal, '⚪')} {signal.upper()}")
        print(f"Trend: {trend_emoji.get(trend, '📊')} {trend.upper()}")
        print(f"Brick Direction: {brick_direction.upper()}")
        
        # Brick formation
        if new_brick:
            if brick_direction == 'up':
                print("🟢 New bullish brick formed!")
            elif brick_direction == 'down':
                print("🔴 New bearish brick formed!")
        
        # Consecutive analysis
        if consecutive_bricks > 1:
            direction_text = "bullish" if brick_direction == 'up' else "bearish"
            print(f"🔥 {consecutive_bricks} consecutive {direction_text} bricks - strong momentum!")
        
        # Reversal detection
        if reversal_detected:
            if signal == 'reversal_up':
                print("🚀 Bullish reversal detected - trend may be changing!")
            elif signal == 'reversal_down':
                print("📉 Bearish reversal detected - trend may be changing!")
            else:
                print("🔄 Trend reversal detected!")
        
        # Price distance from brick
        distance_pct = abs(current_price - last_brick_price) / last_brick_price * 100 if last_brick_price > 0 else 0
        
        # Signal-specific insights
        if signal == 'bullish':
            print("💡 Consider long positions - bullish brick pattern")
        elif signal == 'bearish':
            print("💡 Consider short positions - bearish brick pattern")
        elif 'reversal' in signal:
            print("💡 Watch for trend change confirmation")
        elif trend == 'up':
            print("💡 Uptrend continues - price above brick support")
        elif trend == 'down':
            print("💡 Downtrend continues - price below brick resistance")
        
        # Brick formation status
        remaining_distance = brick_size - (abs(current_price - last_brick_price) % brick_size)
        print(f"📏 Distance to next brick: ${remaining_distance:.4f} ({remaining_distance/brick_size*100:.1f}%)")
