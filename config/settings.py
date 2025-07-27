#!/usr/bin/env python3

# Default configuration settings for CryptoPat

DEFAULT_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT', 'PENGU/USDT']
DEFAULT_EXCHANGE = 'binance'
DEFAULT_TIMEFRAME = '1d'
DEFAULT_PREDICT_DAYS = 2
DEFAULT_ANALYSIS_DAYS = 7
DEFAULT_METHODS = ['sma', 'ema']
VALID_METHODS = ['sma', 'ema', 'ema_cross', 'macd', 'rsi']
VALID_TIMEFRAMES = ['1d', '4h', '1h']
RATE_LIMIT_DELAY = 1.2  # seconds between API calls

# Method-specific minimum requirements
METHOD_REQUIREMENTS = {
    'sma': {
        'min_analysis_days': 7,
        'min_timeframe': '1h',
        'description': 'Simple Moving Average - basic trend analysis'
    },
    'ema': {
        'min_analysis_days': 26,
        'min_timeframe': '1h', 
        'description': 'Exponential Moving Average - requires 26 days for 12/26 EMA comparison'
    },
    'ema_cross': {
        'min_analysis_days': 220,
        'min_timeframe': '1d',
        'description': 'EMA Golden/Death Cross - requires 200+ days for 50/200 EMA crossovers'
    },
    'macd': {
        'min_analysis_days': 35,
        'min_timeframe': '1h',
        'description': 'MACD - requires 26 slow EMA + 9 signal periods minimum'
    },
    'rsi': {
        'min_analysis_days': 15,
        'min_timeframe': '1h',
        'description': 'RSI - requires 14 period + buffer for reliable calculations'
    }
}

# Timeframe hierarchy for validation (lower = more granular)
TIMEFRAME_ORDER = ['1h', '4h', '1d']

# Timeframe multipliers for calculating correct data limits
TIMEFRAME_MULTIPLIERS = {
    '1h': 24,    # 24 hours per day
    '4h': 6,     # 6 four-hour periods per day
    '1d': 1      # 1 daily candle per day
}

def calculate_data_limit(analysis_days: int, timeframe: str) -> int:
    """
    Calculate the correct data limit based on analysis days and timeframe.
    
    Args:
        analysis_days: Number of days to analyze
        timeframe: Timeframe string ('1h', '4h', '1d')
        
    Returns:
        Number of candles needed for the specified period
    """
    multiplier = TIMEFRAME_MULTIPLIERS.get(timeframe, 1)
    return analysis_days * multiplier