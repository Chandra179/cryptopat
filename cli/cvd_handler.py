"""
CVD handler for CryptoPat interactive CLI.
Handles CVD (Cumulative Volume Delta) analysis commands.
"""

import logging
from orderflow.cvd import CVDAnalyzer, display_buyer_seller_pressure

logger = logging.getLogger(__name__)


class CVDHandler:
    """Handler for CVD analysis commands."""
    
    def __init__(self):
        self.cvd_analyzer = CVDAnalyzer()
    
    def print_help(self):
        """Print CVD specific help information."""
        print("    Perform CVD (Cumulative Volume Delta) analysis")
        print("    cvd s=XRP/USDT t=4h l=300")
        print("    cvd s=BTC/USDT t=1d l=500")
        print("    cvd s=ETH/USDT")
    
    def handle(self, command: str) -> bool:
        """
        Handle CVD analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        try:
            symbol, timeframe, limit = self.parse_command(command)
            print(f"\nExecuting: {command}")
            
            # Display visual buyer/seller pressure analysis (includes CVD calculation)
            display_buyer_seller_pressure(symbol, limit)
            return True
            
        except Exception as e:
            print(f"Error: {e}")
            print("Usage: cvd s=SYMBOL t=TIMEFRAME l=LIMIT")
            print("Example: cvd s=XRP/USDT t=4h l=300")
            return False
    
    def parse_command(self, command: str) -> tuple:
        """
        Parse CVD command arguments from string format.
        Expected format: cvd s=XRP/USDT t=4h l=300
        
        Args:
            command: Command string
            
        Returns:
            Tuple of (symbol, timeframe, limit)
        """
        defaults = {
            'symbol': 'XRP/USDT',
            'timeframe': 'Live', 
            'limit': 300
        }
        
        # Remove command name if present
        if command.startswith('cvd'):
            args_string = command[3:].strip()
        else:
            args_string = command.strip()
        
        if not args_string:
            return defaults['symbol'], defaults['timeframe'], defaults['limit']
        
        try:
            # Split arguments
            args = args_string.split()
            
            for arg in args:
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = key.lower()
                    
                    if key == 's':
                        defaults['symbol'] = value.upper()
                    elif key == 't':
                        defaults['timeframe'] = value.upper()
                    elif key == 'l':
                        defaults['limit'] = int(value)
            
            return defaults['symbol'], defaults['timeframe'], defaults['limit']
            
        except Exception as e:
            logger.warning(f"Error parsing CVD arguments: {e}")
            return defaults['symbol'], defaults['timeframe'], defaults['limit']

def main():
    """Main function for testing CVD handler directly."""
    import sys
    
    handler = CVDHandler()
    
    if len(sys.argv) > 1:
        command = ' '.join(sys.argv[1:])
    else:
        command = ""
    
    handler.handle(command)

if __name__ == "__main__":
    main()