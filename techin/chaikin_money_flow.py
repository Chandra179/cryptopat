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
import logging

logger = logging.getLogger(__name__)


class ChaikinMoneyFlow:
    _config = None
    
    @classmethod
    def _load_config(cls):
        if cls._config is None:
            yaml_path = os.path.join(os.path.dirname(__file__), 'chaikin_money_flow.yaml')
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
        cmf_config = self.config['chaikin_money_flow']
        
        # Get timeframe-specific parameters or use default (1d)
        timeframe_params = cmf_config['timeframes'].get(timeframe, cmf_config['timeframes']['1d'])
        general_params = cmf_config['params']
        
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
        Calculate Chaikin Money Flow (CMF) developed by Marc Chaikin.
        
        The Chaikin Money Flow indicator measures the flow of money into and out of a security
        over a specified period, combining price and volume to assess buying/selling pressure.
        
        Formula (Source: Marc Chaikin, TradingView, Fidelity, Investopedia):
        1. Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        2. Money Flow Volume = Money Flow Multiplier × Volume
        3. CMF = Sum(Money Flow Volume, N) / Sum(Volume, N)
        
        Where:
        - N = Period (typically 20 or 21 days)
        - High, Low, Close = Price data for each period
        - Volume = Trading volume for each period
        
        Interpretation:
        - CMF > 0: Buying pressure dominates (bullish)
        - CMF < 0: Selling pressure dominates (bearish)
        - CMF near 0: Balanced buying/selling pressure
        - CMF > +0.25: Strong buying pressure
        - CMF < -0.25: Strong selling pressure
        
        References:
        - Developed by Marc Chaikin in the 1980s
        - TradingView: https://www.tradingview.com/support/solutions/43000501970-chaikin-money-flow-cmf/
        - Fidelity: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/chaikin-money-flow
        - Investopedia: https://www.investopedia.com/terms/c/chaikinmoneyflow.asp
        - StockCharts: https://school.stockcharts.com/doku.php?id=technical_indicators:chaikin_money_flow_cmf
        
        Key Features:
        - Combines price and volume analysis
        - Leading indicator for price reversals
        - Effective for identifying accumulation/distribution phases
        - Works best in trending markets
        """
        if not self.ohlcv or len(self.ohlcv) < self.param["period"]:
            result = {
                "error": f"Insufficient data: need at least {self.param['period']} candles, got {len(self.ohlcv) if self.ohlcv else 0}"
            }
            return result
            
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        period = self.param["period"]
        
        # Calculate Money Flow Multiplier
        # Avoid division by zero when high == low
        hl_diff = df['high'] - df['low']
        hl_diff = hl_diff.replace(0, 0.0001)  # Small value to avoid division by zero
        
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / hl_diff
        
        # Calculate Money Flow Volume
        money_flow_volume = money_flow_multiplier * df['volume']
        
        # Calculate CMF using rolling sums
        mfv_sum = money_flow_volume.rolling(window=period).sum()
        volume_sum = df['volume'].rolling(window=period).sum()
        
        # Avoid division by zero
        volume_sum = volume_sum.replace(0, 1)  # Replace zero volume with 1 to avoid division by zero
        cmf = mfv_sum / volume_sum
        
        # Current values
        current_cmf = float(cmf.iloc[-1]) if not pd.isna(cmf.iloc[-1]) else 0.0
        current_price = float(df['close'].iloc[-1])
        current_mfm = float(money_flow_multiplier.iloc[-1])
        current_mfv = float(money_flow_volume.iloc[-1])
        
        # Signal generation
        signal = "neutral"
        strength = "normal"
        
        if current_cmf >= self.param["strong_bullish"]:
            signal = "strong_bullish"
            strength = "strong"
        elif current_cmf >= self.param["bullish_threshold"]:
            signal = "bullish"
            strength = "weak" if current_cmf < 0.15 else "normal"
        elif current_cmf <= self.param["strong_bearish"]:
            signal = "strong_bearish"
            strength = "strong"
        elif current_cmf <= self.param["bearish_threshold"]:
            signal = "bearish"
            strength = "weak" if current_cmf > -0.15 else "normal"
        
        # Money flow direction
        if current_mfm > 0.5:
            money_flow_trend = "accumulation"
        elif current_mfm < -0.5:
            money_flow_trend = "distribution" 
        else:
            money_flow_trend = "neutral"
        
        # Above zero line check
        above_zero = current_cmf > self.param["zero_line"]
        
        # Build result based on YAML output configuration  
        output_config = self.config['chaikin_money_flow']['output']['fields']
        result = {}
        
        # Build result directly based on YAML fields
        for field_name in output_config:
            if field_name == "symbol":
                result[field_name] = self.symbol
            elif field_name == "timeframe":
                result[field_name] = self.timeframe
            elif field_name == "current_price":
                result[field_name] = current_price
            elif field_name == "cmf":
                result[field_name] = current_cmf
            elif field_name == "money_flow_volume":
                result[field_name] = current_mfv
            elif field_name == "signal":
                result[field_name] = signal
            elif field_name == "money_flow_trend":
                result[field_name] = money_flow_trend
            elif field_name == "above_zero":
                result[field_name] = above_zero
            elif field_name == "strength":
                result[field_name] = strength
            elif field_name == "parameters":
                result[field_name] = {
                    "period": period,
                    "bullish_threshold": self.param["bullish_threshold"],
                    "bearish_threshold": self.param["bearish_threshold"],
                    "strong_bullish": self.param["strong_bullish"],
                    "strong_bearish": self.param["strong_bearish"]
                }
        
        return result
