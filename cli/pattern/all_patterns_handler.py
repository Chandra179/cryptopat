"""
CLI handler for All Patterns analysis.
Processes command line arguments and displays consolidated pattern analysis results.
"""

import logging
from typing import Dict
from pattern.all_patterns import analyze_all_patterns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handle_all_patterns_command(args: Dict[str, str]) -> str:
    """
    Handle all_patterns command with parsed arguments.
    
    Args:
        args: Dictionary containing parsed command arguments
              Expected keys: 's' (symbol), 't' (timeframe), 'l' (limit)
    
    Returns:
        Formatted analysis result string
    """
    try:
        # Extract parameters with defaults
        symbol = args.get('s', 'ADA/USDT')
        timeframe = args.get('t', '4h')
        limit = int(args.get('l', '200'))
        
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
        
        logger.info(f"Running all patterns analysis for {symbol} on {timeframe} with {limit} candles")
        
        # Perform analysis
        result = analyze_all_patterns(symbol, timeframe, limit)
        
        return result
        
    except ValueError as e:
        return f"❌ Error: Invalid limit value - must be a number: {e}"
    except Exception as e:
        logger.error(f"Error in all_patterns command: {e}")
        return f"❌ Error analyzing patterns: {e}"


def parse_all_patterns_args(command_parts: list) -> Dict[str, str]:
    """
    Parse command line arguments for all_patterns command.
    
    Args:
        command_parts: List of command parts (e.g., ['all_patterns', 's=ADA/USDT', 't=4h', 'l=200'])
    
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


def get_all_patterns_help() -> str:
    """
    Get help text for the all_patterns command.
    
    Returns:
        Help text string
    """
    return """
  all_patterns s=XRP/USDT t=4h l=200
  all_patterns s=BTC/USDT t=1d l=150
  all_patterns s=ETH/USDT t=1h l=300
"""


if __name__ == "__main__":
    # Test the handler
    test_args = {'s': 'ADA/USDT', 't': '4h', 'l': '200'}
    result = handle_all_patterns_command(test_args)
    print(result)