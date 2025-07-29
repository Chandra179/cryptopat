"""
Triangle Pattern CLI Handler

Handles triangle pattern analysis commands in the interactive CLI.
"""

from typing import Dict, List
from patterns.triangle import analyze_triangle, format_triangle_output


def parse_triangle_args(command_parts: List[str]) -> Dict:
    """
    Parse triangle command arguments
    
    Args:
        command_parts: List of command parts
        
    Returns:
        Dictionary with parsed arguments
    """
    args = {
        'symbol': None,
        'timeframe': '4h',  # default
        'limit': 100  # default
    }
    
    for part in command_parts[1:]:  # Skip 'triangle'
        if part.startswith('s='):
            args['symbol'] = part[2:]
        elif part.startswith('t='):
            args['timeframe'] = part[2:]
        elif part.startswith('l='):
            try:
                args['limit'] = int(part[2:])
            except ValueError:
                raise ValueError(f"Invalid limit value: {part[2:]}")
    
    if args['symbol'] is None:
        raise ValueError("Symbol (s=) is required")
    
    return args


def handle_triangle_command(args: Dict) -> str:
    """
    Handle triangle pattern analysis command
    
    Args:
        args: Parsed command arguments
        
    Returns:
        Formatted analysis result
    """
    try:
        result = analyze_triangle(args['symbol'], args['timeframe'], args['limit'])
        return format_triangle_output(result)
    except Exception as e:
        return f"❌ Error in triangle analysis: {e}"


def get_triangle_help() -> str:
    """
    Get help text for triangle command
    
    Returns:
        Help text string
    """
    return """  triangle s=SYMBOL t=TIMEFRAME l=LIMIT - Analyze triangle patterns
    Examples:
      triangle s=BTC/USDT t=4h l=100
      triangle s=ETH/USDT t=1d l=150"""