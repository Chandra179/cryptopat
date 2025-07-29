"""
CLI Handler for Head and Shoulders Pattern Analysis

Handles head_and_shoulders command in the interactive CLI
Command format: head_and_shoulders s=BTC/USDT t=4h l=150
"""

import argparse
from patterns.head_and_shoulders import analyze_head_and_shoulders, format_head_and_shoulders_output


def parse_head_and_shoulders_command(command_args: str) -> dict:
    """
    Parse head_and_shoulders command arguments
    
    Args:
        command_args: Command arguments string (e.g., "s=BTC/USDT t=4h l=150")
        
    Returns:
        Dictionary with parsed arguments
    """
    # Default values
    symbol = "BTC/USDT"
    timeframe = "4h" 
    limit = 150
    
    if not command_args.strip():
        return {'symbol': symbol, 'timeframe': timeframe, 'limit': limit}
    
    # Parse arguments
    args = command_args.strip().split()
    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            if key.lower() == 's':
                symbol = value
            elif key.lower() == 't':
                timeframe = value
            elif key.lower() == 'l':
                try:
                    limit = int(value)
                except ValueError:
                    print(f"⚠️ Invalid limit value: {value}, using default: {limit}")
    
    return {'symbol': symbol, 'timeframe': timeframe, 'limit': limit}


def handle_head_and_shoulders_command(command_args: str) -> str:
    """
    Handle head_and_shoulders command
    
    Args:
        command_args: Command arguments string
        
    Returns:
        Formatted analysis output
    """
    try:
        # Parse command arguments
        params = parse_head_and_shoulders_command(command_args)
        
        # Perform analysis
        analysis = analyze_head_and_shoulders(
            symbol=params['symbol'],
            timeframe=params['timeframe'], 
            limit=params['limit']
        )
        
        # Format and return output
        return format_head_and_shoulders_output(analysis)
        
    except Exception as e:
        return f"❌ Head and Shoulders analysis failed: {str(e)}"


if __name__ == "__main__":
    # Test the handler
    test_commands = [
        "s=BTC/USDT t=4h l=150",
        "s=ETH/USDT t=1d l=100", 
        "s=XRP/USDT t=4h",
        ""  # Test defaults
    ]
    
    for cmd in test_commands:
        print(f"\n> head_and_shoulders {cmd}")
        print(handle_head_and_shoulders_command(cmd))
        print("-" * 80)