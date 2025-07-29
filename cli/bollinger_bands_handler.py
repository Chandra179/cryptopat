"""
Bollinger Bands handler for CryptoPat interactive CLI.
Handles Bollinger Bands volatility and reversal analysis commands.
"""

from trend.bollinger_bands import BollingerBandsStrategy, parse_command as parse_bb_command


class BollingerBandsHandler:
    """Handler for Bollinger Bands analysis commands."""
    
    def __init__(self):
        self.bb_strategy = BollingerBandsStrategy()
    
    def print_help(self):
        """Print Bollinger Bands specific help information."""
        print("    Perform Bollinger Bands volatility analysis")
        print("    bb s=ETH/USDT t=4h l=100")
        print("    bb s=BTC/USDT t=1d l=50")
        print("    bb s=XRP/USDT")
    
    def handle(self, command: str) -> bool:
        """
        Handle Bollinger Bands analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        try:
            symbol, timeframe, limit = parse_bb_command(command)
            print(f"\nExecuting: {command}")
            self.bb_strategy.analyze(symbol, timeframe, limit)
            return True
        except Exception as e:
            print(f"Error: {e}")
            print("Usage: bb s=SYMBOL t=TIMEFRAME l=LIMIT")
            print("Example: bb s=ETH/USDT t=4h l=100")
            return False