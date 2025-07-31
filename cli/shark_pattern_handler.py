"""
CLI handler for Shark pattern analysis.
Processes command line arguments and displays Shark harmonic pattern detection results.
"""

import logging
from typing import Dict
from trend.shark_pattern import analyze_shark_pattern

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handle_shark_pattern_command(args: Dict[str, str]) -> str:
    """
    Handle shark_pattern command with parsed arguments.
    
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
        
        logger.info(f"Analyzing Shark pattern for {symbol} on {timeframe} with {limit} candles, ZZ={zigzag_threshold}%")
        
        # Perform analysis
        result = analyze_shark_pattern(symbol, timeframe, limit, zigzag_threshold)
        
        return result
        
    except ValueError as e:
        return f"❌ Error: Invalid parameter value - must be a number: {e}"
    except Exception as e:
        logger.error(f"Error in shark_pattern command: {e}")
        return f"❌ Error analyzing Shark pattern: {e}"


def parse_shark_pattern_args(command_parts: list) -> Dict[str, str]:
    """
    Parse command line arguments for shark_pattern command.
    
    Args:
        command_parts: List of command parts (e.g., ['shark_pattern', 's=XRP/USDT', 't=4h', 'l=150', 'zz=5'])
    
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


def get_shark_pattern_help() -> str:
    """
    Get help text for the shark_pattern command.
    
    Returns:
        Help text string
    """
    return """
🦈 Shark Pattern Detection - Harmonic pattern with Fibonacci validation
Usage examples:
  shark_pattern s=XRP/USDT t=4h l=150 zz=5
  shark_pattern s=BTC/USDT t=1d l=200 zz=3
  shark_pattern s=ETH/USDT t=1h l=100 zz=7

Parameters:
  s=SYMBOL    - Trading pair (e.g., XRP/USDT, BTC/USDT)
  t=TIMEFRAME - Chart timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, etc.)
  l=LIMIT     - Number of candles to analyze (50-1000, default: 150)
  zz=PERCENT  - ZigZag threshold percentage (1.0-20.0, default: 5.0)

Pattern Rules:
  • XA retracement = 0.886 (±2% tolerance)
  • AB extension = 1.13-1.618 of XA leg
  • BC extension = 1.13 of OX leg (±5% tolerance)
  • Entry at point C with volume spike confirmation
"""


if __name__ == "__main__":
    # Test the handler
    test_args = {'s': 'XRP/USDT', 't': '4h', 'l': '150', 'zz': '5'}
    result = handle_shark_pattern_command(test_args)
    print(result)