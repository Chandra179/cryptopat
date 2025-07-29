"""
CLI handler for Double Top pattern analysis.
Processes command line arguments and displays pattern detection results.
"""

import logging
from typing import Dict
from patterns.double_top import analyze_double_top

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handle_double_top_command(args: Dict[str, str]) -> str:
    """
    Handle double_top command with parsed arguments.
    
    Args:
        args: Dictionary containing parsed command arguments
              Expected keys: 's' (symbol), 't' (timeframe), 'l' (limit)
    
    Returns:
        Formatted analysis result string
    """
    try:
        # Extract parameters with defaults
        symbol = args.get('s', 'ETH/USDT')
        timeframe = args.get('t', '4h')
        limit = int(args.get('l', '100'))
        
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
        
        logger.info(f"Analyzing double top pattern for {symbol} on {timeframe} with {limit} candles")
        
        # Perform analysis
        result = analyze_double_top(symbol, timeframe, limit)
        
        return result
        
    except ValueError as e:
        return f"❌ Error: Invalid limit value - must be a number: {e}"
    except Exception as e:
        logger.error(f"Error in double_top command: {e}")
        return f"❌ Error analyzing double top pattern: {e}"


def parse_double_top_args(command_parts: list) -> Dict[str, str]:
    """
    Parse command line arguments for double_top command.
    
    Args:
        command_parts: List of command parts (e.g., ['double_top', 's=ETH/USDT', 't=4h', 'l=100'])
    
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


def get_double_top_help() -> str:
    """
    Get help text for the double_top command.
    
    Returns:
        Help text string
    """
    return """
  double_top s=ETH/USDT t=4h l=100
  double_top s=BTC/USDT t=1d l=150
  double_top s=SOL/USDT t=1h l=200
"""


if __name__ == "__main__":
    # Test the handler
    test_args = {'s': 'ETH/USDT', 't': '4h', 'l': '100'}
    result = handle_double_top_command(test_args)
    print(result)