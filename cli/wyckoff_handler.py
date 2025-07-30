"""
Wyckoff Structure Analysis handler for CryptoPat interactive CLI.
Handles Wyckoff analysis commands for market cycle detection.
"""

import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from structure.wyckoff import WyckoffAnalyzer, parse_command as parse_wyckoff_command


class WyckoffHandler:
    """Handler for Wyckoff Structure Analysis commands."""
    
    def __init__(self):
        self.wyckoff_analyzer = WyckoffAnalyzer()
    
    def print_help(self):
        """Print Wyckoff specific help information."""
        print("    Perform Wyckoff Structure Analysis")
        print("    wyckoff s=BTC/USDT t=4h l=600")
        print("    wyckoff s=ETH/USDT t=1d l=720 detect=events+phases")
        print("    wyckoff s=XRP/USDT t=1h l=500 detect=phases")
        print("    wyckoff s=SOL/USDT t=4h l=800 detect=events")
    
    def handle(self, command: str) -> bool:
        """
        Handle Wyckoff analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        try:
            symbol, timeframe, limit, detect = parse_wyckoff_command(command)
            print(f"\nExecuting: {command}")
            self.wyckoff_analyzer.analyze(symbol, timeframe, limit, detect)
            return True
        except Exception as e:
            print(f"Error: {e}")
            print("Usage: wyckoff s=SYMBOL t=TIMEFRAME l=LIMIT [detect=TYPE]")
            print("Example: wyckoff s=BTC/USDT t=4h l=600")
            print("Example: wyckoff s=ETH/USDT t=1d l=720 detect=events+phases")
            print("Detect options: 'phases', 'events', 'phases+events'")
            return False