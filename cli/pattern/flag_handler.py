"""
Flag Pattern CLI Handler

Handles flag pattern analysis commands in the interactive CLI.
"""

from typing import Dict, List
from patterns.flag import analyze_flag, format_flag_output


def parse_flag_args(command_parts: List[str]) -> Dict:
    """
    Parse flag command arguments
    
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
    
    for part in command_parts[1:]:  # Skip 'flag'
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


def handle_flag_command(args: Dict) -> str:
    """
    Handle flag pattern analysis command
    
    Args:
        args: Parsed command arguments
        
    Returns:
        Formatted analysis result
    """
    try:
        result = analyze_flag(args['symbol'], args['timeframe'], args['limit'])
        return format_flag_output(result)
    except Exception as e:
        return f"❌ Error in flag analysis: {e}"


def get_flag_help() -> str:
    """
    Get help text for flag command
    
    Returns:
        Help text string
    """
    return """  flag s=SYMBOL t=TIMEFRAME l=LIMIT - Analyze flag patterns (bull/bear flags)
    Examples:
      flag s=ETH/USDT t=4h l=100
      flag s=SOL/USDT t=1h l=80"""