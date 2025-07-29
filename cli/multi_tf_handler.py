"""
Multi-Timeframe Confluence handler for CryptoPat interactive CLI.
Handles multi-timeframe confluence analysis commands.
"""

from trend.multi_tf_confluence import MultiTimeframeConfluence, parse_command as parse_multi_tf_command


class MultiTFHandler:
    """Handler for multi-timeframe confluence analysis commands."""
    
    def __init__(self):
        self.multi_tf_analyzer = MultiTimeframeConfluence()
    
    def print_help(self):
        """Print multi-timeframe confluence specific help information."""
        print("    Perform multi-timeframe confluence analysis")
        print("    multi_tf s=BTC/USDT t1=1d t2=4h t3=1h indicators=ema9/21,macd,rsi14 l=200")
        print("    multi_tf s=ETH/USDT t1=4h t2=1h t3=15m l=150")
        print("    multi_tf s=XRP/USDT t1=1d t2=4h")
        print("    ")
        print("    Parameters:")
        print("      s=SYMBOL     - Trading pair (required)")
        print("      t1/t2/t3=TF  - Timeframes to analyze (default: 1d,4h,1h)")
        print("      indicators=  - Comma-separated list (default: ema9/21,macd,rsi14)")
        print("      l=LIMIT      - Number of candles (default: 200)")
    
    def handle(self, command: str) -> bool:
        """
        Handle multi-timeframe confluence analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        try:
            symbol, timeframes, indicators, limit = parse_multi_tf_command(command)
            print(f"\nExecuting: {command}")
            self.multi_tf_analyzer.analyze(symbol, timeframes, indicators, limit)
            return True
        except Exception as e:
            print(f"Error: {e}")
            print("Usage: multi_tf s=SYMBOL t1=TF1 t2=TF2 t3=TF3 indicators=LIST l=LIMIT")
            print("Example: multi_tf s=BTC/USDT t1=1d t2=4h t3=1h indicators=ema9/21,macd,rsi14 l=200")
            return False