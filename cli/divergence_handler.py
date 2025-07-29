"""
Divergence handler for CryptoPat interactive CLI.
Handles divergence detection analysis commands.
"""

from trend.divergence import DivergenceDetector, parse_command as parse_divergence_command


class DivergenceHandler:
    """Handler for divergence analysis commands."""
    
    def __init__(self):
        self.divergence_detector = DivergenceDetector()
    
    def print_help(self):
        """Print divergence specific help information."""
        print("    Perform divergence detection analysis using RSI, MACD, and OBV")
        print("    divergence s=SOL/USDT t=4h l=100")
        print("    divergence s=BTC/USDT t=1d l=200")
        print("    divergence s=ETH/USDT")
    
    def handle(self, command: str) -> bool:
        """
        Handle divergence analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        try:
            symbol, timeframe, limit = parse_divergence_command(command)
            print(f"\nExecuting: {command}")
            self.divergence_detector.analyze(symbol, timeframe, limit)
            return True
        except Exception as e:
            print(f"Error: {e}")
            print("Usage: divergence s=SYMBOL t=TIMEFRAME l=LIMIT")
            print("Example: divergence s=SOL/USDT t=4h l=100")
            return False