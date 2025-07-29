"""
EMA 9/21 handler for CryptoPat interactive CLI.
Handles EMA 9/21 crossover strategy analysis commands.
"""

from trend.ema_9_21 import EMA9_21Strategy, parse_command as parse_ema_command


class EMA921Handler:
    """Handler for EMA 9/21 analysis commands."""
    
    def __init__(self):
        self.ema_strategy = EMA9_21Strategy()
    
    def print_help(self):
        """Print EMA 9/21 specific help information."""
        print("  ema_9_21 s=SYMBOL t=TIMEFRAME l=LIMIT")
        print("    Perform EMA 9/21 crossover analysis")
        print("    Parameters:")
        print("      s= : Trading symbol (required) - e.g., XRP/USDT, BTC/USDT")
        print("      t= : Timeframe (optional, default: 1d) - 1m, 5m, 1h, 4h, 1d, etc.")
        print("      l= : Limit of candles (optional, default: 30) - minimum 50 recommended")
        print("  Examples:")
        print("    ema_9_21 s=XRP/USDT t=1d l=30")
        print("    ema_9_21 s=BTC/USDT t=4h l=50")
        print("    ema_9_21 s=ETH/USDT")
    
    def handle(self, command: str) -> bool:
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