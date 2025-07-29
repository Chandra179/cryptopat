"""
Interactive CLI for CryptoPat pattern recognition system.
Provides a terminal interface for running various analysis commands.
"""

import sys
from trend.ema_9_21 import EMA9_21Strategy, parse_command as parse_ema_command
from trend.rsi_14 import RSI14Strategy, parse_command as parse_rsi_command


class InteractiveCLI:
    """Interactive command-line interface for CryptoPat."""
    
    def __init__(self):
        self.running = True
        self.ema_strategy = EMA9_21Strategy()
        self.rsi_strategy = RSI14Strategy()
    
    def print_welcome(self):
        """Print welcome message and available commands."""
        print("=" * 60)
        print("CryptoPat - Interactive Cryptocurrency Pattern Analysis")
        print("=" * 60)
        print("\nAvailable commands:")
        print("  ema_9_21 s=SYMBOL t=TIMEFRAME l=LIMIT")
        print("    - Analyze EMA 9/21 crossover strategy")
        print("    - Example: ema_9_21 s=XRP/USDT t=1d l=30")
        print("  rsi_14 s=SYMBOL t=TIMEFRAME l=LIMIT")
        print("    - Analyze RSI(14) momentum and reversal signals")
        print("    - Example: rsi_14 s=XRP/USDT t=1d l=30")
        print("    - Timeframes: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 3d, 1w, 1M")
        print("\n  help - Show this help message")
        print("  exit - Exit the application")
        print("\n" + "=" * 60)
    
    def print_help(self):
        """Print help information."""
        print("\nCryptoPat Interactive CLI Help")
        print("=" * 40)
        print("\nCommands:")
        print("  ema_9_21 s=SYMBOL t=TIMEFRAME l=LIMIT")
        print("    Perform EMA 9/21 crossover analysis")
        print("    Parameters:")
        print("      s= : Trading symbol (required) - e.g., XRP/USDT, BTC/USDT")
        print("      t= : Timeframe (optional, default: 1d) - 1m, 5m, 1h, 4h, 1d, etc.")
        print("      l= : Limit of candles (optional, default: 30) - minimum 50 recommended")
        print("\n  rsi_14 s=SYMBOL t=TIMEFRAME l=LIMIT")
        print("    Perform RSI(14) momentum and reversal analysis")
        print("    Parameters:")
        print("      s= : Trading symbol (required) - e.g., XRP/USDT, BTC/USDT")
        print("      t= : Timeframe (optional, default: 1d) - 1m, 5m, 1h, 4h, 1d, etc.")
        print("      l= : Limit of candles (optional, default: 30) - minimum 20 recommended")
        print("\n  Examples:")
        print("    ema_9_21 s=XRP/USDT t=1d l=30")
        print("    ema_9_21 s=BTC/USDT t=4h l=50")
        print("    ema_9_21 s=ETH/USDT")  # Uses defaults: t=1d, l=30")
        print("    rsi_14 s=XRP/USDT t=1d l=30")
        print("    rsi_14 s=BTC/USDT t=4h l=50")
        print("\n  help - Show this help message")
        print("  exit - Exit the application")
    
    def handle_ema_9_21(self, command: str) -> bool:
        """
        Handle EMA 9/21 analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        try:
            symbol, timeframe, limit = parse_ema_command(command)
            print(f"\nExecuting: {command}")
            self.ema_strategy.analyze(symbol, timeframe, limit)
            return True
        except Exception as e:
            print(f"Error: {e}")
            print("Usage: ema_9_21 s=SYMBOL t=TIMEFRAME l=LIMIT")
            print("Example: ema_9_21 s=XRP/USDT t=1d l=30")
            return False
    
    def handle_rsi_14(self, command: str) -> bool:
        """
        Handle RSI 14 analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        try:
            symbol, timeframe, limit = parse_rsi_command(command)
            print(f"\nExecuting: {command}")
            self.rsi_strategy.analyze(symbol, timeframe, limit)
            return True
        except Exception as e:
            print(f"Error: {e}")
            print("Usage: rsi_14 s=SYMBOL t=TIMEFRAME l=LIMIT")
            print("Example: rsi_14 s=XRP/USDT t=1d l=30")
            return False
    
    def process_command(self, command: str) -> bool:
        """
        Process a user command.
        
        Args:
            command: The command string
            
        Returns:
            True to continue running, False to exit
        """
        command = command.strip()
        
        if not command:
            return True
        
        # Handle exit commands
        if command.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            return False
        
        # Handle help command
        if command.lower() in ['help', 'h', '?']:
            self.print_help()
            return True
        
        # Handle EMA 9/21 command
        if command.startswith('ema_9_21'):
            self.handle_ema_9_21(command)
            return True
        
        # Handle RSI 14 command
        if command.startswith('rsi_14'):
            self.handle_rsi_14(command)
            return True
        
        # Unknown command
        print(f"Unknown command: {command}")
        print("Type 'help' for available commands or 'exit' to quit.")
        return True
    
    def run(self):
        """Run the interactive CLI loop."""
        self.print_welcome()
        
        try:
            while self.running:
                try:
                    # Get user input
                    command = input("\n> ").strip()
                    
                    # Process command
                    self.running = self.process_command(command)
                    
                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    break
                except EOFError:
                    print("\n\nGoodbye!")
                    break
                    
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)


def main():
    """Main entry point for interactive CLI."""
    cli = InteractiveCLI()
    cli.run()


if __name__ == '__main__':
    main()