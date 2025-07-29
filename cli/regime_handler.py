"""
Market Regime handler for CryptoPat interactive CLI.
Handles market regime detection (trend vs range) analysis commands.
"""

from trend.regime import MarketRegimeStrategy, parse_command as parse_regime_command


class RegimeHandler:
    """Handler for market regime analysis commands."""
    
    def __init__(self):
        self.regime_strategy = MarketRegimeStrategy()
    
    def print_help(self):
        """Print regime analysis specific help information."""
        print("    Perform market regime detection (trend vs range) analysis")
        print("    regime s=BTC/USDT t=4h l=100")
        print("    regime s=ETH/USDT t=1d l=150")
        print("    regime s=SOL/USDT")
    
    def handle(self, command: str) -> bool:
        """
        Handle market regime analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        try:
            symbol, timeframe, limit = parse_regime_command(command)
            print(f"\nExecuting: {command}")
            self.regime_strategy.analyze(symbol, timeframe, limit)
            return True
        except Exception as e:
            print(f"Error: {e}")
            print("Usage: regime s=SYMBOL t=TIMEFRAME l=LIMIT")
            print("Example: regime s=BTC/USDT t=4h l=100")
            return False