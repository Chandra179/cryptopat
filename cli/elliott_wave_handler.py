"""
Elliott Wave Analysis handler for CryptoPat interactive CLI.
Handles Elliott Wave pattern detection commands.
"""

import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trend.elliott_wave import ElliottWaveAnalyzer, parse_command as parse_elliott_command


class ElliottWaveHandler:
    """Handler for Elliott Wave Analysis commands."""
    
    def __init__(self):
        self.elliott_analyzer = ElliottWaveAnalyzer()
    
    def print_help(self):
        """Print Elliott Wave specific help information."""
        print("    Perform Elliott Wave pattern analysis")
        print("    elliott s=BTC/USDT t=4h l=150")
        print("    elliott s=ETH/USDT t=1d l=200 zz=3.5")
        print("    elliott s=XRP/USDT t=1h l=100 zz=7.0")
        print("    elliott s=SOL/USDT t=4h l=180")
    
    def handle(self, command: str) -> bool:
        """
        Handle Elliott Wave analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        try:
            symbol, timeframe, limit, zigzag_threshold = parse_elliott_command(command)
            print(f"\nExecuting: {command}")
            self.elliott_analyzer.analyze(symbol, timeframe, limit, zigzag_threshold)
            return True
        except Exception as e:
            print(f"Error: {e}")
            print("Usage: elliott s=SYMBOL t=TIMEFRAME l=LIMIT [zz=THRESHOLD]")
            print("Example: elliott s=BTC/USDT t=4h l=150")
            print("Example: elliott s=ETH/USDT t=1d l=200 zz=3.5")
            print("ZigZag threshold (zz=) controls swing sensitivity (default: 5.0%)")
            return False