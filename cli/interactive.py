#!/usr/bin/env python3

import argparse
import shlex
import asyncio
from cli.commands import handle_analyze_command, handle_collect_command

# Enable readline for arrow key support in interactive mode
try:
    import readline
except ImportError:
    # On some systems, readline might not be available
    pass

def interactive_mode():
    """Run interactive CLI mode."""
    print("🚀 CryptoPat Interactive Mode")
    print("Type 'help' for available commands, 'exit' or 'quit' to leave")
    print("-" * 60)
    
    while True:
        try:
            # Get user input
            user_input = input("cryptopat> ").strip()
            
            # Skip empty input
            if not user_input:
                continue
            
            # Handle exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            # Handle help command
            if user_input.lower() in ['help', 'h', '?']:
                print_interactive_help()
                continue
            
            # Parse command using shlex to handle quoted arguments
            try:
                args_list = shlex.split(user_input)
            except ValueError as e:
                print(f"❌ Error parsing command: {e}")
                continue
            
            # Handle commands
            if not args_list:
                continue
                
            command = args_list[0].lower()
            
            if command == 'analyze':
                handle_analyze_command(args_list[1:])
            elif command == 'collect':
                handle_collect_command(args_list[1:])
            else:
                print(f"❌ Unknown command: {command}")
                print("Type 'help' for available commands")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break

def print_interactive_help():
    """Print help for interactive mode."""
    print("""
📚 Available Commands:

  analyze [options]           Run trend analysis
    -p, --predict-days N      Days to predict ahead (default: 2)
    -a, --analysis-days N     Days of historical data (default: 7)
    -s, --symbols SYM [...]   Symbols to analyze (default: BTC/USDT ETH/USDT ...)
    -e, --exchange NAME       Exchange to use (default: binance)
    -t, --timeframe TF        Timeframe: 1d, 4h, 1h (default: 1d)
    -m, --method METHOD       Analysis method: sma, ema, ema_cross, or combinations (default: sma,ema)

  collect [options]           Test data collection
    --symbol SYM              Symbol to test (default: BTC/USDT)
    --exchange NAME           Exchange to use (default: binance)
    --timeframe TF            Timeframe: 1d, 4h, 1h (default: 1d)

  help, h, ?                  Show this help
  exit, quit, q               Exit the program

📝 Examples:
  analyze
  analyze -p 5 -a 14
  analyze --symbols BTC/USDT ETH/USDT
  analyze -m ema_cross -p 3 -a 220
  analyze -m sma,ema,ema_cross -p 5 -a 250
  collect --symbol SOL/USDT
""")