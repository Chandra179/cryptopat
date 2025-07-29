"""
ATR+ADX handler for CryptoPat interactive CLI.
Handles ATR (Average True Range) + ADX (Average Directional Index) analysis commands.
"""

from trend.atr_adx import ATR_ADXStrategy, parse_command as parse_atr_adx_command


class ATRADXHandler:
    """Handler for ATR+ADX analysis commands."""
    
    def __init__(self):
        self.atr_adx_strategy = ATR_ADXStrategy()
    
    def print_help(self):
        """Print ATR+ADX specific help information."""
        print("    Perform ATR+ADX volatility and trend strength analysis")
        print("    atr_adx s=ETH/USDT t=4h l=14")
        print("    atr_adx s=BTC/USDT t=1d l=30")
        print("    atr_adx s=SOL/USDT")
    
    def handle(self, command: str) -> bool:
        """
        Handle ATR+ADX analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        try:
            symbol, timeframe, limit = parse_atr_adx_command(command)
            print(f"\nExecuting: {command}")
            self.atr_adx_strategy.analyze(symbol, timeframe, limit)
            return True
        except Exception as e:
            print(f"Error: {e}")
            print("Usage: atr_adx s=SYMBOL t=TIMEFRAME l=LIMIT")
            print("Example: atr_adx s=ETH/USDT t=4h l=14")
            return False