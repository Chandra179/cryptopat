#!/usr/bin/env python3
"""
Interactive CLI for CryptoPat pattern analysis
Accepts parameters: s=symbol t=timeframe l=candles
Example: s=BTC/USDT t=1h l=100
"""

import sys
import os
import readline
import re
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from comprehensive_analyzer import ComprehensiveAnalyzer

class CryptoPatCLI:
    def __init__(self):
        self.history_file = os.path.expanduser("~/.cryptopat_history")
        self.setup_history()
        self.valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '3d', '1w', '1M']
        self.analyzer = ComprehensiveAnalyzer()
        
    def setup_history(self):
        """Setup readline history for command persistence"""
        try:
            readline.read_history_file(self.history_file)
        except FileNotFoundError:
            pass
        
        # Set maximum history length
        readline.set_history_length(1000)
        
    def save_history(self):
        """Save command history to file"""
        try:
            readline.write_history_file(self.history_file)
        except Exception as e:
            print(f"Warning: Could not save history: {e}")
    
    def parse_command(self, command):
        """Parse command string for s, t, l parameters"""
        params = {}
        
        # Parse s=symbol
        s_match = re.search(r's=([A-Za-z0-9/]+)', command)
        if s_match:
            params['symbol'] = s_match.group(1).upper()
        
        # Parse t=timeframe
        t_match = re.search(r't=([0-9]+[mhdwM])', command)
        if t_match:
            params['timeframe'] = t_match.group(1)
        
        # Parse l=length
        l_match = re.search(r'l=([0-9]+)', command)
        if l_match:
            params['length'] = int(l_match.group(1))
        
        return params
    
    def validate_params(self, params):
        """Validate parsed parameters"""
        errors = []
        
        if 'symbol' not in params:
            errors.append("Missing symbol (s=). Example: s=BTC/USDT")
        
        if 'timeframe' not in params:
            errors.append("Missing timeframe (t=). Example: t=1h")
        elif params['timeframe'] not in self.valid_timeframes:
            errors.append(f"Invalid timeframe. Valid options: {', '.join(self.valid_timeframes)}")
        
        if 'length' not in params:
            errors.append("Missing length (l=). Example: l=100")
        elif params['length'] <= 0:
            errors.append("Length must be a positive number")
        
        return errors
    
    def show_help(self):
        """Display help information"""
        print("""
CryptoPat Interactive CLI
========================

Parameters:
  s = Symbol (e.g., BTC/USDT, ETH/USDT, SOL/USDT)
  t = Timeframe (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 3d, 1w, 1M)
  l = Number of candles (positive integer)

Examples:
  s=BTC/USDT t=1h l=100
  s=ETH/USDT t=4h l=50

Commands:
  help - Show this help
  exit, quit - Exit the CLI
  clear - Clear screen
  
History:
  Use arrow keys to navigate command history
  History is saved between sessions
        """)
    
    def run(self):
        """Main CLI loop"""
        print("CryptoPat Interactive CLI")
        print("Type 'help' for usage information")
        print("Use Ctrl+C or 'exit' to quit\n")
        
        try:
            while True:
                try:
                    command = input("cryptopat> ").strip()
                    
                    if not command:
                        continue
                    
                    if command.lower() in ['exit', 'quit']:
                        break
                    
                    if command.lower() == 'help':
                        self.show_help()
                        continue
                    
                    if command.lower() == 'clear':
                        os.system('clear' if os.name == 'posix' else 'cls')
                        continue
                    
                    # Parse and validate command
                    params = self.parse_command(command)
                    errors = self.validate_params(params)
                    
                    if errors:
                        print("Errors:")
                        for error in errors:
                            print(f"  - {error}")
                        print("Type 'help' for usage information")
                        continue
                    
                    # Execute comprehensive analysis
                    print(f"\n🔍 Running comprehensive analysis for {params['symbol']} {params['timeframe']} ({params['length']} candles)...")
                    
                    # Run the comprehensive market analysis and get formatted output
                    formatted_output = self.analyzer.analyze_comprehensive(params['symbol'], params['timeframe'], params['length'])
                    
                    # Display the beautiful formatted report
                    print("\n" + formatted_output)
                    
                except KeyboardInterrupt:
                    print("\nUse 'exit' or 'quit' to exit")
                    continue
                except EOFError:
                    break
        
        finally:
            self.save_history()
            print("\nGoodbye!")

if __name__ == "__main__":
    cli = CryptoPatCLI()
    cli.run()