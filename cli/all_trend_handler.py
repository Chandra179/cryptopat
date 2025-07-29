"""
All trend handler for CryptoPat interactive CLI.
Handles all trend analysis commands combining EMA, MACD, and RSI strategies.
"""

from trend.all_trend import AllTrendStrategy, parse_command as parse_all_trend_command


class AllTrendHandler:
    """Handler for all trend analysis commands."""
    
    def __init__(self):
        self.all_trend_strategy = AllTrendStrategy()
    
    def print_help(self):
        """Print all trend specific help information."""
        print("    Perform comprehensive trend analysis (EMA + MACD + RSI + OBV + ATR+ADX)")
        print("    all_trend s=XRP/USDT t=4h l=100")
        print("    all_trend s=BTC/USDT t=1d l=50")
        print("    all_trend s=ETH/USDT")
    
    def handle(self, command: str) -> bool:
        """
        Handle all trend analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        try:
            symbol, timeframe, limit = parse_all_trend_command(command)
            print(f"\nExecuting: {command}")
            self.all_trend_strategy.analyze(symbol, timeframe, limit)
            return True
        except Exception as e:
            print(f"Error: {e}")
            print("Usage: all_trend s=SYMBOL t=TIMEFRAME l=LIMIT")
            print("Example: all_trend s=XRP/USDT t=4h l=100")
            return False