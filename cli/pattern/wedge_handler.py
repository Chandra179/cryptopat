"""
Wedge Pattern CLI Handler

Handles wedge pattern analysis commands in the interactive CLI.
"""

from typing import Dict, List
from patterns.wedge import analyze_wedge, format_wedge_output


def parse_wedge_args(command_parts: List[str]) -> Dict:
    """
    Parse wedge command arguments
    
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
    
    for part in command_parts[1:]:  # Skip 'wedge'
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


def handle_wedge_command(args: Dict) -> str:
    """
    Handle wedge pattern analysis command
    
    Args:
        args: Parsed command arguments
        
    Returns:
        Formatted analysis result
    """
    try:
        result = analyze_wedge(args['symbol'], args['timeframe'], args['limit'])
        return format_wedge_output(result)
    except Exception as e:
        return f"❌ Error in wedge analysis: {e}"


def get_wedge_help() -> str:
    """
    Get help text for wedge command
    
    Returns:
        Help text string
    """
    return """  wedge s=SYMBOL t=TIMEFRAME l=LIMIT - Analyze wedge patterns (rising/falling)
    Examples:
      wedge s=BTC/USDT t=4h l=100
      wedge s=ADA/USDT t=1d l=120"""