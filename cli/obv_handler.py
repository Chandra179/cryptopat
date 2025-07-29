"""
OBV handler for CryptoPat interactive CLI.
Handles OBV (On-Balance Volume) analysis commands.
"""

from trend.obv import OBVStrategy, parse_command as parse_obv_command


class OBVHandler:
    """Handler for OBV analysis commands."""
    
    def __init__(self):
        self.obv_strategy = OBVStrategy()
    
    def print_help(self):
        """Print OBV specific help information."""
        print("    Perform OBV (On-Balance Volume) analysis")
        print("    obv s=BTC/USDT t=1d l=100")
        print("    obv s=ETH/USDT t=4h l=50")
        print("    obv s=XRP/USDT")
    
    def handle(self, command: str) -> bool:
        """
        Handle OBV analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        try:
            symbol, timeframe, limit = parse_obv_command(command)
            print(f"\nExecuting: {command}")
            self.obv_strategy.analyze(symbol, timeframe, limit)
            return True
        except Exception as e:
            print(f"Error: {e}")
            print("Usage: obv s=SYMBOL t=TIMEFRAME l=LIMIT")
            print("Example: obv s=BTC/USDT t=1d l=100")
            return False