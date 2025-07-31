"""
CLI handler for Butterfly pattern analysis.
Processes command line arguments and displays Butterfly harmonic pattern detection results.
"""

import logging
from typing import Dict
from trend.butterfly_pattern import analyze_butterfly_pattern

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handle_butterfly_pattern_command(args: Dict[str, str]) -> str:
    """
    Handle butterfly command with parsed arguments.
    
    Args:
        args: Dictionary containing parsed command arguments
              Expected keys: 's' (symbol), 't' (timeframe), 'l' (limit), 'zz' (zigzag threshold)
    
    Returns:
        Formatted analysis result string
    """
    try:
        # Extract parameters with defaults
        symbol = args.get('s', 'XRP/USDT')
        timeframe = args.get('t', '4h')
        limit = int(args.get('l', '150'))
        zigzag_threshold = float(args.get('zz', '5.0'))
        
        # Validate parameters
        if not symbol:
            return "❌ Error: Symbol parameter 's' is required"
        
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '3d', '1w', '1M']
        if timeframe not in valid_timeframes:
            return f"❌ Error: Invalid timeframe '{timeframe}'. Valid options: {', '.join(valid_timeframes)}"
        
        if limit < 50:
            return "❌ Error: Minimum limit is 50 candles for reliable pattern detection"
        
        if limit > 1000:
            return "❌ Error: Maximum limit is 1000 candles"
        
        if zigzag_threshold < 1.0 or zigzag_threshold > 20.0:
            return "❌ Error: ZigZag threshold must be between 1.0% and 20.0%"
        
        logger.info(f"Analyzing Butterfly pattern for {symbol} on {timeframe} with {limit} candles, ZZ={zigzag_threshold}%")
        
        # Perform analysis
        result = analyze_butterfly_pattern(symbol, timeframe, limit, zigzag_threshold)
        
        return result
        
    except ValueError as e:
        return f"❌ Error: Invalid parameter value - must be a number: {e}"
    except Exception as e:
        logger.error(f"Error in butterfly command: {e}")
        return f"❌ Error analyzing Butterfly pattern: {e}"


def parse_butterfly_pattern_args(command_parts: list) -> Dict[str, str]:
    """
    Parse command line arguments for butterfly command.
    
    Args:
        command_parts: List of command parts (e.g., ['butterfly', 's=XRP/USDT', 't=4h', 'l=150', 'zz=5'])
    
    Returns:
        Dictionary with parsed arguments
    """
    args = {}
    
    for part in command_parts[1:]:  # Skip the command name
        if '=' in part:
            key, value = part.split('=', 1)
            args[key] = value
        else:
            # Handle positional arguments if needed
            pass
    
    return args


def get_butterfly_pattern_help() -> str:
    """
    Get help text for the butterfly command.
    
    Returns:
        Help text string
    """
    return """
🦋 Butterfly Pattern Detection - Harmonic pattern with X-A-B-C-D structure
Usage examples:
  butterfly s=XRP/USDT t=4h l=150 zz=5
  butterfly s=BTC/USDT t=1d l=200 zz=3
  butterfly s=ETH/USDT t=1h l=100 zz=7

Parameters:
  s=SYMBOL    - Trading pair (e.g., XRP/USDT, BTC/USDT)
  t=TIMEFRAME - Chart timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, etc.)
  l=LIMIT     - Number of candles to analyze (50-1000, default: 150)
  zz=PERCENT  - ZigZag threshold percentage (1.0-20.0, default: 5.0)

Butterfly Leg Ratio Rules:
  • AB = 0.786 retracement of XA (±2% tolerance)
  • BC = 0.382 to 0.886 retracement of AB
  • CD = 1.618 to 2.618 extension of BC
  • AD = 1.27 extension of XA (±5% tolerance)
  • Entry at point D with volume spike + rejection candle confirmation

Target Zones:
  • TP1 = 38.2% retracement of CD
  • TP2 = 61.8% retracement of CD
  • SL = slightly beyond point X
"""


if __name__ == "__main__":
    # Test the handler
    test_args = {'s': 'XRP/USDT', 't': '4h', 'l': '150', 'zz': '5'}
    result = handle_butterfly_pattern_command(test_args)
    print(result)