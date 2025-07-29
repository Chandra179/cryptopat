"""
Smart Money Concepts (SMC) handler for CryptoPat interactive CLI.
Handles SMC analysis commands for market structure analysis.
"""

from trend.smc import SMCStrategy, parse_command as parse_smc_command


class SMCHandler:
    """Handler for Smart Money Concepts analysis commands."""
    
    def __init__(self):
        self.smc_strategy = SMCStrategy()
    
    def print_help(self):
        """Print SMC specific help information."""
        print("    Perform Smart Money Concepts analysis")
        print("    smc s=BTC/USDT t=1h l=300")
        print("    smc s=ETH/USDT t=4h l=500 zones=true")
        print("    smc s=XRP/USDT t=1d l=200 choch=true")
        print("    smc s=SOL/USDT t=15m l=400 zones=true choch=true")
    
    def handle(self, command: str) -> bool:
        """
        Handle SMC analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        try:
            symbol, timeframe, limit, zones, choch = parse_smc_command(command)
            print(f"\nExecuting: {command}")
            self.smc_strategy.analyze(symbol, timeframe, limit, zones, choch)
            return True
        except Exception as e:
            print(f"Error: {e}")
            print("Usage: smc s=SYMBOL t=TIMEFRAME l=LIMIT [zones=true] [choch=true]")
            print("Example: smc s=BTC/USDT t=1h l=300")
            print("Example: smc s=ETH/USDT t=4h l=500 zones=true choch=true")
            return False