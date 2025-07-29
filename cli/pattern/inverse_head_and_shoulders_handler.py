"""
CLI Handler for Inverse Head and Shoulders Pattern Analysis

Handles inverse_head_and_shoulders command in the interactive CLI
Command format: inverse_head_and_shoulders s=SOL/USDT t=4h l=150
"""

import argparse
from patterns.inverse_head_and_shoulders import analyze_inverse_head_and_shoulders, format_inverse_head_and_shoulders_output


def parse_inverse_head_and_shoulders_command(command_args: str) -> dict:
    """
    Parse inverse_head_and_shoulders command arguments
    
    Args:
        command_args: Command arguments string (e.g., "s=SOL/USDT t=4h l=150")
        
    Returns:
        Dictionary with parsed arguments
    """
    # Default values
    symbol = "SOL/USDT"
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


def handle_inverse_head_and_shoulders_command(command_args: str) -> str:
    """
    Handle inverse_head_and_shoulders command
    
    Args:
        command_args: Command arguments string
        
    Returns:
        Formatted analysis output
    """
    try:
        # Parse command arguments
        params = parse_inverse_head_and_shoulders_command(command_args)
        
        # Perform analysis
        analysis = analyze_inverse_head_and_shoulders(
            symbol=params['symbol'],
            timeframe=params['timeframe'], 
            limit=params['limit']
        )
        
        # Format and return output
        return format_inverse_head_and_shoulders_output(analysis)
        
    except Exception as e:
        return f"❌ Inverse Head and Shoulders analysis failed: {str(e)}"


if __name__ == "__main__":
    # Test the handler
    test_commands = [
        "s=SOL/USDT t=4h l=150",
        "s=ETH/USDT t=1d l=100", 
        "s=XRP/USDT t=4h",
        ""  # Test defaults
    ]
    
    for cmd in test_commands:
        print(f"\n> inverse_head_and_shoulders {cmd}")
        print(handle_inverse_head_and_shoulders_command(cmd))
        print("-" * 80)