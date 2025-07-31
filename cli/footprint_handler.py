"""
CLI handler for Volume Footprint Chart analysis commands.
Provides command parsing and execution for footprint analysis.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def parse_footprint_command(command: str) -> Dict[str, Any]:
    """
    Parse footprint command arguments.
    
    Expected format: footprint s=XRP/USDT t=5m l=50 bins=40
    
    Args:
        command: The complete command string
        
    Returns:
        Dictionary with parsed arguments
    """
    # Default values
    args = {
        'symbol': 'XRP/USDT',
        'timeframe': '5m', 
        'limit': 50,
        'bins': 40
    }
    
    # Split command and extract parameters
    parts = command.split()
    
    for part in parts[1:]:  # Skip 'footprint'
        if '=' in part:
            key, value = part.split('=', 1)
            
            if key == 's':
                args['symbol'] = value
            elif key == 't':
                args['timeframe'] = value
            elif key == 'l':
                try:
                    args['limit'] = int(value)
                    if args['limit'] < 10:
                        logger.warning(f"Limit {args['limit']} is very low, minimum 10 recommended")
                    elif args['limit'] > 200:
                        logger.warning(f"Limit {args['limit']} is very high, may cause performance issues")
                except ValueError:
                    logger.error(f"Invalid limit value: {value}")
                    raise ValueError(f"Invalid limit value: {value}")
            elif key == 'bins':
                try:
                    args['bins'] = int(value)
                    if args['bins'] < 10:
                        logger.warning(f"Bins {args['bins']} is very low, minimum 10 recommended")
                    elif args['bins'] > 100:
                        logger.warning(f"Bins {args['bins']} is very high, may cause display issues")
                except ValueError:
                    logger.error(f"Invalid bins value: {value}")
                    raise ValueError(f"Invalid bins value: {value}")
            else:
                logger.warning(f"Unknown parameter: {key}={value}")
    
    return args

def validate_footprint_args(args: Dict[str, Any]) -> bool:
    """
    Validate footprint command arguments.
    
    Args:
        args: Parsed command arguments
        
    Returns:
        True if arguments are valid, False otherwise
    """
    # Validate symbol format
    symbol = args.get('symbol', '')
    if '/' not in symbol:
        logger.error(f"Invalid symbol format: {symbol}. Expected format: BTC/USDT")
        return False
    
    # Validate timeframe
    valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '3d', '1w', '1M']
    timeframe = args.get('timeframe', '')
    if timeframe not in valid_timeframes:
        logger.error(f"Invalid timeframe: {timeframe}. Valid options: {', '.join(valid_timeframes)}")
        return False
    
    # Validate limit
    limit = args.get('limit', 0)
    if not isinstance(limit, int) or limit <= 0:
        logger.error(f"Invalid limit: {limit}. Must be a positive integer")
        return False
    
    # Validate bins
    bins = args.get('bins', 0)
    if not isinstance(bins, int) or bins <= 0:
        logger.error(f"Invalid bins: {bins}. Must be a positive integer")
        return False
    
    return True

def handle_footprint_command(command: str) -> str:
    """
    Handle footprint analysis command execution.
    
    Args:
        command: The complete command string
        
    Returns:
        Analysis result string
    """
    try:
        # Parse command arguments
        args = parse_footprint_command(command)
        
        # Validate arguments
        if not validate_footprint_args(args):
            return get_footprint_help()
        
        # Import here to avoid circular imports
        from orderflow.footprint import analyze_footprint
        
        # Execute footprint analysis
        logger.info(f"Running footprint analysis: {args['symbol']} {args['timeframe']} {args['limit']} candles, {args['bins']} bins")
        
        result = analyze_footprint(
            symbol=args['symbol'],
            timeframe=args['timeframe'], 
            limit=args['limit'],
            bins=args['bins']
        )
        
        return result
        
    except ValueError as e:
        logger.error(f"Invalid footprint command arguments: {e}")
        return f"Error: {str(e)}\n\n{get_footprint_help()}"
    except Exception as e:
        logger.error(f"Error executing footprint command: {e}")
        return f"Error executing footprint analysis: {str(e)}"

def get_footprint_help() -> str:
    """
    Get help text for footprint command.
    
    Returns:
        Help text string
    """
    return """Volume Footprint Chart Analysis:
  footprint s=XRP/USDT t=5m l=50 bins=40
  
Parameters:
  s=     Trading symbol (required) - e.g., XRP/USDT, BTC/USDT
  t=     Timeframe (optional, default: 5m) - 1m, 5m, 15m, 30m, 1h, 4h
  l=     Number of candles (optional, default: 50) - minimum 10 recommended
  bins=  Price bins per candle (optional, default: 40) - 10-100 range
  
Examples:
  footprint s=XRP/USDT
  footprint s=BTC/USDT t=15m l=100 bins=50
  footprint s=ETH/USDT t=1h l=30 bins=30
  
Analysis shows:
  • Volume distribution across price levels within each candle
  • Buy/sell volume imbalances per price bin
  • Volume exhaustion signals at price extremes
  • Delta imbalance detection for hidden liquidity hunts
  • Potential reversal signals from volume patterns"""

class FootprintHandler:
    """Handler class for footprint commands with consistent interface."""
    
    def __init__(self):
        """Initialize FootprintHandler."""
        pass
    
    def handle(self, command: str) -> bool:
        """
        Handle footprint command with consistent CLI interface.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully
        """
        try:
            result = handle_footprint_command(command)
            print(result)
            return True
        except Exception as e:
            logger.error(f"Error in FootprintHandler: {e}")
            print(f"Error: {str(e)}")
            return True  # Return True to continue CLI
    
    def print_help(self):
        """Print help information for footprint command."""
        print(get_footprint_help())