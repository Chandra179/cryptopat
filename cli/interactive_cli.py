"""
Interactive CLI for CryptoPat pattern recognition system.
Provides a terminal interface for running various analysis commands.
"""

import sys
import readline
import atexit
import os
from cli.ema_9_21_handler import EMA921Handler
from cli.rsi_14_handler import RSI14Handler
from cli.macd_handler import MACDHandler
from cli.all_trend_handler import AllTrendHandler
from cli.obv_handler import OBVHandler


class InteractiveCLI:
    """Interactive command-line interface for CryptoPat."""
    
    def __init__(self):
        self.running = True
        self.ema_handler = EMA921Handler()
        self.rsi_handler = RSI14Handler()
        self.macd_handler = MACDHandler()
        self.all_trend_handler = AllTrendHandler()
        self.obv_handler = OBVHandler()
        self._setup_readline()
    
    def _setup_readline(self):
        """Setup readline for command history and line editing."""
        # Enable tab completion
        readline.set_completer_delims(' \t\n')
        readline.parse_and_bind('tab: complete')
        
        # Enable arrow key navigation
        readline.parse_and_bind(r'"\e[A": previous-history')  # Up arrow
        readline.parse_and_bind(r'"\e[B": next-history')      # Down arrow
        readline.parse_and_bind(r'"\e[C": forward-char')      # Right arrow
        readline.parse_and_bind(r'"\e[D": backward-char')     # Left arrow
        
        # Enable other useful key bindings
        readline.parse_and_bind(r'"\C-a": beginning-of-line')  # Ctrl+A
        readline.parse_and_bind(r'"\C-e": end-of-line')        # Ctrl+E
        readline.parse_and_bind(r'"\C-k": kill-line')          # Ctrl+K
        readline.parse_and_bind(r'"\C-u": unix-line-discard')  # Ctrl+U
        
        # Setup history file
        history_file = os.path.expanduser('~/.cryptopat_history')
        try:
            readline.read_history_file(history_file)
        except FileNotFoundError:
            pass  # History file doesn't exist yet
        
        # Limit history size
        readline.set_history_length(1000)
        
        # Save history on exit
        atexit.register(readline.write_history_file, history_file)
    
    def print_welcome(self):
        """Print welcome message and available commands."""
        print("=" * 60)
        print("CryptoPat - Interactive Cryptocurrency Pattern Analysis")
        print("=" * 60)
        print("Navigation:")
        print("  ↑/↓ arrows: Command history")
        print("  ←/→ arrows: Move cursor within line")
        print("  Ctrl+A/E: Beginning/End of line")
        print("  Type 'help' for commands or 'exit' to quit")
        print("=" * 60)
    
    def print_help(self):
        """Print help information."""
        print("=" * 40)
        print("CryptoPat Interactive CLI Help")
        print("=" * 40)
        print("Parameters:")
        print("   s= : Trading symbol (required) - e.g., XRP/USDT, BTC/USDT")
        print("   t= : Timeframe (optional, default: 1d) - 1m, 5m, 1h, 4h, 1d, etc.")
        print("   l= : Limit of candles (optional, default: 30) - minimum 20 recommended")
        print("\nCommands:")
        self.ema_handler.print_help()
        print()
        self.rsi_handler.print_help()
        print()
        self.macd_handler.print_help()
        print()
        self.obv_handler.print_help()
        print()
        self.all_trend_handler.print_help()
        print("\n  help - Show this help message")
        print("  exit - Exit the application")
    
    def handle_ema_9_21(self, command: str) -> bool:
        """
        Handle EMA 9/21 analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        return self.ema_handler.handle(command)
    
    def handle_rsi_14(self, command: str) -> bool:
        """
        Handle RSI 14 analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        return self.rsi_handler.handle(command)
    
    def handle_macd(self, command: str) -> bool:
        """
        Handle MACD analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        return self.macd_handler.handle(command)
    
    def handle_all_trend(self, command: str) -> bool:
        """
        Handle all trend analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        return self.all_trend_handler.handle(command)
    
    def handle_obv(self, command: str) -> bool:
        """
        Handle OBV analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        return self.obv_handler.handle(command)
    
    def process_command(self, command: str) -> bool:
        """
        Process a user command.
        
        Args:
            command: The command string
            
        Returns:
            True to continue running, False to exit
        """
        command = command.strip()
        
        if not command:
            return True
        
        # Handle exit commands
        if command.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            return False
        
        # Handle help command
        if command.lower() in ['help', 'h', '?']:
            self.print_help()
            return True
        
        # Handle EMA 9/21 command
        if command.startswith('ema_9_21'):
            self.handle_ema_9_21(command)
            return True
        
        # Handle RSI 14 command
        if command.startswith('rsi_14'):
            self.handle_rsi_14(command)
            return True
        
        # Handle MACD command
        if command.startswith('macd'):
            self.handle_macd(command)
            return True
        
        # Handle OBV command
        if command.startswith('obv'):
            self.handle_obv(command)
            return True
        
        # Handle all trend command
        if command.startswith('all_trend'):
            self.handle_all_trend(command)
            return True
        
        # Unknown command
        print(f"Unknown command: {command}")
        print("Type 'help' for available commands or 'exit' to quit.")
        return True
    
    def run(self):
        """Run the interactive CLI loop."""
        self.print_welcome()
        
        try:
            while self.running:
                try:
                    # Get user input
                    command = input("\n> ").strip()
                    
                    # Process command
                    self.running = self.process_command(command)
                    
                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    break
                except EOFError:
                    print("\n\nGoodbye!")
                    break
                    
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)


def main():
    """Main entry point for interactive CLI."""
    cli = InteractiveCLI()
    cli.run()


if __name__ == '__main__':
    main()