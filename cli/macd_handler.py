"""
MACD handler for CryptoPat interactive CLI.
Handles MACD (Moving Average Convergence Divergence) analysis commands.
"""

from trend.macd import MACDStrategy, parse_command as parse_macd_command


class MACDHandler:
    """Handler for MACD analysis commands."""
    
    def __init__(self):
        self.macd_strategy = MACDStrategy()
    
    def print_help(self):
        """Print MACD specific help information."""
        print("    Perform MACD trend and momentum analysis")
        print("    macd s=XRP/USDT t=4h l=100")
        print("    macd s=BTC/USDT t=1d l=150")
        print("    macd s=ETH/USDT")
    
    def handle(self, command: str) -> bool:
        """
        Handle MACD analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        try:
            symbol, timeframe, limit = parse_macd_command(command)
            print(f"\nExecuting: {command}")
            self.macd_strategy.analyze(symbol, timeframe, limit)
            return True
        except Exception as e:
            print(f"Error: {e}")
            print("Usage: macd s=SYMBOL t=TIMEFRAME l=LIMIT")
            print("Example: macd s=XRP/USDT t=4h l=100")
            return False