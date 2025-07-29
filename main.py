#!/usr/bin/env python3

import logging
import sys
from trend.ema_9_21 import EMA9_21Strategy, parse_command as parse_ema_command
from trend.rsi_14 import RSI14Strategy, parse_command as parse_rsi_command
from trend.macd import MACDStrategy, parse_command as parse_macd_command
from cli.interactive_cli import InteractiveCLI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main application entry point."""
    if len(sys.argv) == 1:
        # No arguments - launch interactive CLI
        cli = InteractiveCLI()
        cli.run()
        return
    
    command = sys.argv[1]
    
    if command == 'ema_9_21':
        try:
            # Parse command arguments
            full_command = ' '.join(sys.argv[1:])
            symbol, timeframe, limit = parse_ema_command(full_command)
            
            # Run EMA 9/21 analysis
            strategy = EMA9_21Strategy()
            strategy.analyze(symbol, timeframe, limit)
            
        except Exception as e:
            print(f"Error: {e}")
    elif command == 'rsi_14':
        try:
            # Parse command arguments
            full_command = ' '.join(sys.argv[1:])
            symbol, timeframe, limit = parse_rsi_command(full_command)
            
            # Run RSI 14 analysis
            strategy = RSI14Strategy()
            strategy.analyze(symbol, timeframe, limit)
            
        except Exception as e:
            print(f"Error: {e}")
    elif command == 'macd':
        try:
            # Parse command arguments
            full_command = ' '.join(sys.argv[1:])
            symbol, timeframe, limit = parse_macd_command(full_command)
            
            # Run MACD analysis
            strategy = MACDStrategy()
            strategy.analyze(symbol, timeframe, limit)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    main()