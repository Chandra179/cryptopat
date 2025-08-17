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


class PivotPoint:
    _config = None
    
    @classmethod
    def _load_config(cls):
        if cls._config is None:
            yaml_path = os.path.join(os.path.dirname(__file__), 'pivot_point.yaml')
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
        pp_config = self.config['pivot_point']
        
        # Use parameters from YAML config
        self.param = pp_config['params']
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.ticker = ticker
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
    
    def calculate(self):
        """
        Calculate Pivot Points using standard methodology.
        
        Standard Formula (Source: TradingView, Wikipedia):
        - Pivot Point (P) = (High + Low + Close) / 3
        - Resistance 1 (R1) = P * 2 - Low
        - Resistance 2 (R2) = P + (High - Low)
        - Resistance 3 (R3) = P * 2 + (High - 2 * Low)
        - Support 1 (S1) = P * 2 - High
        - Support 2 (S2) = P - (High - Low)
        - Support 3 (S3) = P * 2 - (2 * High - Low)
        
        Alternative Methods:
        - Include Opening: P = (Open + High + Low + Close) / 4
        - Emphasize Close: P = (High + Low + Close + Close) / 4
        
        Central Pivot Range (CPR):
        - Top Central: TC = (P - BC) + P, where BC = (High + Low) / 2
        - Bottom Central: BC = (High + Low) / 2
        
        References:
        - TradingView: https://www.tradingview.com/support/solutions/43000521824-pivot-points-standard/
        - Wikipedia: https://en.wikipedia.org/wiki/Pivot_point_(technical_analysis)
        - Investopedia: Standard pivot point analysis for support and resistance
        
        Trading Interpretation:
        - Price above pivot point indicates bullish sentiment
        - Price below pivot point indicates bearish sentiment
        - Pivot points act as dynamic support and resistance levels
        - Higher timeframes provide stronger pivot levels
        """
        if not self.ohlcv or len(self.ohlcv) < 2:
            result = {
                "error": f"Insufficient data: need at least 2 candles, got {len(self.ohlcv) if self.ohlcv else 0}"
            }
            return result
            
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['open'] = pd.to_numeric(df['open'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # Use previous period's data for pivot calculation
        prev_high = float(df['high'].iloc[-2])
        prev_low = float(df['low'].iloc[-2])
        prev_close = float(df['close'].iloc[-2])
        prev_open = float(df['open'].iloc[-2])
        
        current_price = float(df['close'].iloc[-1])
        
        # Calculate pivot point based on method
        if self.param["include_opening"]:
            pivot = (prev_open + prev_high + prev_low + prev_close) / 4
        elif self.param["emphasize_close"]:
            pivot = (prev_high + prev_low + prev_close + prev_close) / 4
        else:  # Standard HLC method
            pivot = (prev_high + prev_low + prev_close) / 3
            
        # Calculate support and resistance levels
        levels = {"pivot": pivot}
        
        # Standard support and resistance calculations
        num_support = self.param["support_levels"]
        num_resistance = self.param["resistance_levels"]
        
        for i in range(1, max(num_support, num_resistance) + 1):
            if i <= num_resistance:
                if i == 1:
                    levels[f"r{i}"] = pivot * 2 - prev_low
                elif i == 2:
                    levels[f"r{i}"] = pivot + (prev_high - prev_low)
                elif i == 3:
                    levels[f"r{i}"] = pivot * 2 + (prev_high - 2 * prev_low)
                else:
                    # Extended resistance levels
                    levels[f"r{i}"] = pivot * i + (prev_high - i * prev_low)
                    
            if i <= num_support:
                if i == 1:
                    levels[f"s{i}"] = pivot * 2 - prev_high
                elif i == 2:
                    levels[f"s{i}"] = pivot - (prev_high - prev_low)
                elif i == 3:
                    levels[f"s{i}"] = pivot * 2 - (2 * prev_high - prev_low)
                else:
                    # Extended support levels
                    levels[f"s{i}"] = pivot * i - (i * prev_high - prev_low)
        
        # Central Pivot Range (CPR) if enabled
        cpr = {}
        if self.param["central_pivot_range"]:
            bc = (prev_high + prev_low) / 2  # Bottom Central
            tc = (pivot - bc) + pivot        # Top Central
            cpr = {"bc": bc, "tc": tc, "width": tc - bc}
        
        # Determine current position relative to pivot
        position = "neutral"
        distance_from_pivot = (current_price - pivot) / pivot
        
        if current_price > pivot:
            position = "bullish"
        elif current_price < pivot:
            position = "bearish"
            
        # Find nearest support/resistance levels
        nearest_support = None
        nearest_resistance = None
        
        for i in range(1, num_support + 1):
            if f"s{i}" in levels and current_price > levels[f"s{i}"]:
                nearest_support = {"level": f"s{i}", "price": levels[f"s{i}"]}
                break
                
        for i in range(1, num_resistance + 1):
            if f"r{i}" in levels and current_price < levels[f"r{i}"]:
                nearest_resistance = {"level": f"r{i}", "price": levels[f"r{i}"]}
                break
        
        # Signal generation based on YAML signal types
        signal = "neutral"
        
        # Check for strong resistance/support zones
        if price_zone == "above_r3":
            signal = "strong_resistance"
        elif price_zone == "below_s3":
            signal = "strong_support"
        elif position == "bullish" and distance_from_pivot > 0.01:
            signal = "bullish"
        elif position == "bearish" and distance_from_pivot < -0.01:
            signal = "bearish"
        else:
            signal = "neutral"
        
        # Volume confirmation if enabled
        volume_confirmed = True
        if self.param["volume_confirmation"] and len(df) >= self.param["volume_ma_period"]:
            avg_volume = df['volume'].rolling(window=self.param["volume_ma_period"]).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_confirmed = current_volume > avg_volume
            if not volume_confirmed:
                signal_strength *= 0.7
        
        # Determine price zone
        price_zone = "neutral"
        if current_price < levels.get("s3", float('-inf')):
            price_zone = "below_s3"
        elif current_price < levels.get("s2", float('-inf')):
            price_zone = "s2_s3"
        elif current_price < levels.get("s1", float('-inf')):
            price_zone = "s1_s2"
        elif current_price < pivot:
            price_zone = "pp_s1"
        elif current_price < levels.get("r1", float('inf')):
            price_zone = "pp_r1"
        elif current_price < levels.get("r2", float('inf')):
            price_zone = "r1_r2"
        elif current_price < levels.get("r3", float('inf')):
            price_zone = "r2_r3"
        else:
            price_zone = "above_r3"
        
        # Build result based on YAML output configuration  
        output_config = self.config['pivot_point']['output']['fields']
        result = {}
        
        # Build result directly based on YAML fields
        for field_name in output_config:
            if field_name == "symbol":
                result[field_name] = self.symbol
            elif field_name == "timeframe":
                result[field_name] = self.timeframe
            elif field_name == "current_price":
                result[field_name] = current_price
            elif field_name == "pivot_point":
                result[field_name] = pivot
            elif field_name == "resistance_1":
                result[field_name] = levels.get("r1")
            elif field_name == "resistance_2":
                result[field_name] = levels.get("r2")
            elif field_name == "resistance_3":
                result[field_name] = levels.get("r3")
            elif field_name == "support_1":
                result[field_name] = levels.get("s1")
            elif field_name == "support_2":
                result[field_name] = levels.get("s2")
            elif field_name == "support_3":
                result[field_name] = levels.get("s3")
            elif field_name == "signal":
                result[field_name] = signal
            elif field_name == "price_zone":
                result[field_name] = price_zone
            elif field_name == "nearest_support":
                result[field_name] = nearest_support["price"] if nearest_support else None
            elif field_name == "nearest_resistance":
                result[field_name] = nearest_resistance["price"] if nearest_resistance else None
            elif field_name == "parameters":
                result[field_name] = {
                    "calculation_method": self.param["calculation_method"],
                    "price_source": self.param["price_source"],
                    "support_levels": self.param["support_levels"],
                    "resistance_levels": self.param["resistance_levels"],
                    "breakout_threshold": self.param["breakout_threshold"]
                }
        
        return result
