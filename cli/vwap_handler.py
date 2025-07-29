"""
CLI Handler for VWAP (Volume Weighted Average Price) Analysis
Handles user input and executes VWAP analysis commands
"""

import re
from trend.vwap import run_vwap_analysis


def parse_vwap_command(command: str) -> dict:
    """
    Parse VWAP command and extract parameters
    
    Expected format: vwap s=ETH/USDT t=15m l=200 [anchor="2025-07-29T04:00:00"]
    
    Args:
        command: Raw command string
        
    Returns:
        Dictionary with parsed parameters
    """
    params = {
        'symbol': 'BTC/USDT',    # default
        'timeframe': '1h',       # default
        'limit': 100,            # default
        'anchor': None           # optional
    }
    
    # Extract symbol (s=)
    symbol_match = re.search(r's=([^\s]+)', command)
    if symbol_match:
        params['symbol'] = symbol_match.group(1)
    
    # Extract timeframe (t=)
    timeframe_match = re.search(r't=([^\s]+)', command)
    if timeframe_match:
        params['timeframe'] = timeframe_match.group(1)
    
    # Extract limit (l=)
    limit_match = re.search(r'l=(\d+)', command)
    if limit_match:
        params['limit'] = int(limit_match.group(1))
    
    # Extract anchor timestamp (anchor="...")
    anchor_match = re.search(r'anchor=["\']([^"\']+)["\']', command)
    if anchor_match:
        params['anchor'] = anchor_match.group(1)
    
    return params


def handle_vwap_command(command: str) -> str:
    """
    Handle VWAP analysis command
    
    Args:
        command: Full command string (e.g., "vwap s=ETH/USDT t=15m l=200")
        
    Returns:
        Analysis results or error message
    """
    try:
        # Parse command parameters
        params = parse_vwap_command(command)
        
        # Validate parameters
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '3d', '1w', '1M']
        if params['timeframe'] not in valid_timeframes:
            return f"Error: Invalid timeframe '{params['timeframe']}'. Valid options: {', '.join(valid_timeframes)}"
        
        if params['limit'] < 10 or params['limit'] > 1000:
            return "Error: Limit must be between 10 and 1000"
        
        # Run VWAP analysis
        result = run_vwap_analysis(
            symbol=params['symbol'],
            timeframe=params['timeframe'], 
            limit=params['limit'],
            anchor=params['anchor']
        )
        
        return result
        
    except Exception as e:
        return f"Error executing VWAP command: {str(e)}"


def get_vwap_help() -> str:
    """
    Get help text for VWAP commands
    
    Returns:
        Help text string
    """
    help_text = """
  vwap s=ETH/USDT t=15m l=200
  vwap s=BTC/USDT t=1h l=100 anchor="2025-07-29T04:00:00"
  vwap s=SOL/USDT t=4h l=50
  Note: VWAP is most effective in intraday timeframes (1m to 4h)
"""
    return help_text


if __name__ == "__main__":
    # Test the handler
    test_command = "vwap s=ETH/USDT t=15m l=50"
    result = handle_vwap_command(test_command)
    print(result)