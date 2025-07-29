"""
RSI 14 handler for CryptoPat interactive CLI.
Handles RSI(14) momentum and reversal analysis commands.
"""

from trend.rsi_14 import RSI14Strategy, parse_command as parse_rsi_command


class RSI14Handler:
    """Handler for RSI 14 analysis commands."""
    
    def __init__(self):
        self.rsi_strategy = RSI14Strategy()
    
    def print_help(self):
        """Print RSI 14 specific help information."""
        print("    Perform RSI(14) momentum and reversal analysis")
        print("    rsi_14 s=XRP/USDT t=1d l=30")
        print("    rsi_14 s=BTC/USDT t=4h l=50")
    
    def handle(self, command: str) -> bool:
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