"""
Interactive CLI for CryptoPat pattern recognition system.
Provides a terminal interface for running various analysis commands.
"""

import sys
from cli.ema_9_21_handler import EMA921Handler
from cli.rsi_14_handler import RSI14Handler


class InteractiveCLI:
    """Interactive command-line interface for CryptoPat."""
    
    def __init__(self):
        self.running = True
        self.ema_handler = EMA921Handler()
        self.rsi_handler = RSI14Handler()
    
    def print_welcome(self):
        """Print welcome message and available commands."""
        print("=" * 60)
        print("CryptoPat - Interactive Cryptocurrency Pattern Analysis")
        print("=" * 60)
    
    def print_help(self):
        """Print help information."""
        print("\nCryptoPat Interactive CLI Help")
        print("=" * 40)
        print("\nCommands:")
        self.ema_handler.print_help()
        print()
        self.rsi_handler.print_help()
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