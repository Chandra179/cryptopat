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
from cli.atr_adx_handler import ATRADXHandler
from cli.bollinger_bands_handler import BollingerBandsHandler
from cli.divergence_handler import DivergenceHandler
from cli.regime_handler import RegimeHandler
from cli.vwap_handler import handle_vwap_command, get_vwap_help
from cli.multi_tf_handler import MultiTFHandler
from cli.pattern.double_bottom_handler import handle_double_bottom_command, parse_double_bottom_args, get_double_bottom_help
from cli.pattern.all_patterns_handler import handle_all_patterns_command, parse_all_patterns_args, get_all_patterns_help
from cli.pattern.head_and_shoulders_handler import handle_head_and_shoulders_command
from cli.pattern.inverse_head_and_shoulders_handler import handle_inverse_head_and_shoulders_command
from cli.pattern.triangle_handler import handle_triangle_command, parse_triangle_args, get_triangle_help
from cli.pattern.flag_handler import handle_flag_command, parse_flag_args, get_flag_help
from cli.pattern.wedge_handler import handle_wedge_command, parse_wedge_args, get_wedge_help
from cli.smc_handler import SMCHandler
from cli.wyckoff_handler import WyckoffHandler
from cli.elliott_wave_handler import ElliottWaveHandler
from cli.shark_pattern_handler import handle_shark_pattern_command, parse_shark_pattern_args, get_shark_pattern_help
from cli.butterfly_pattern_handler import handle_butterfly_pattern_command, parse_butterfly_pattern_args, get_butterfly_pattern_help
from cli.cvd_handler import CVDHandler
from cli.orderbook_heatmap_handler import OrderBookHeatmapHandler, get_orderbook_heatmap_help
from cli.imbalance_handler import handle_imbalance_analysis
from cli.absorption_handler import handle_absorption_command, get_absorption_help
from cli.footprint_handler import FootprintHandler, get_footprint_help

class InteractiveCLI:
    """Interactive command-line interface for CryptoPat."""
    
    def __init__(self):
        self.running = True
        self.ema_handler = EMA921Handler()
        self.rsi_handler = RSI14Handler()
        self.macd_handler = MACDHandler()
        self.all_trend_handler = AllTrendHandler()
        self.obv_handler = OBVHandler()
        self.atr_adx_handler = ATRADXHandler()
        self.bb_handler = BollingerBandsHandler()
        self.divergence_handler = DivergenceHandler()
        self.regime_handler = RegimeHandler()
        self.multi_tf_handler = MultiTFHandler()
        self.smc_handler = SMCHandler()
        self.wyckoff_handler = WyckoffHandler()
        self.elliott_wave_handler = ElliottWaveHandler()
        self.cvd_handler = CVDHandler()
        self.orderbook_heatmap_handler = OrderBookHeatmapHandler()
        self.footprint_handler = FootprintHandler()
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
        self.atr_adx_handler.print_help()
        print()
        self.bb_handler.print_help()
        print()
        self.divergence_handler.print_help()
        print()
        self.regime_handler.print_help()
        print()
        print(get_vwap_help())
        print()
        self.multi_tf_handler.print_help()
        print()
        self.smc_handler.print_help()
        print()
        self.wyckoff_handler.print_help()
        print()
        self.elliott_wave_handler.print_help()
        print()
        self.all_trend_handler.print_help()
        print()
        print(get_double_bottom_help())
        print()
        print("  head_and_shoulders s=BTC/USDT t=4h l=150")
        print("  head_and_shoulders s=ETH/USDT t=1d l=100")
        print()
        print("  inverse_head_and_shoulders s=SOL/USDT t=4h l=150")
        print("  inverse_head_and_shoulders s=ETH/USDT t=1d l=100")
        print()
        print(get_triangle_help())
        print()
        print(get_flag_help())
        print()
        print(get_wedge_help())
        print()
        print(get_all_patterns_help())
        print()
        print(get_shark_pattern_help())
        print()
        print(get_butterfly_pattern_help())
        print()
        self.cvd_handler.print_help()
        print()
        print(get_orderbook_heatmap_help())
        print()
        print("  imbalance s=XRP/USDT d=30")
        print("  imbalance s=BTC/USDT d=60")
        print()
        print(get_absorption_help())
        print()
        print(get_footprint_help())
    
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
    
    def handle_atr_adx(self, command: str) -> bool:
        """
        Handle ATR+ADX analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        return self.atr_adx_handler.handle(command)
    
    def handle_bollinger_bands(self, command: str) -> bool:
        """
        Handle Bollinger Bands analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        return self.bb_handler.handle(command)
    
    def handle_divergence(self, command: str) -> bool:
        """
        Handle divergence analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        return self.divergence_handler.handle(command)
    
    def handle_regime(self, command: str) -> bool:
        """
        Handle market regime analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        return self.regime_handler.handle(command)
    
    def handle_multi_tf(self, command: str) -> bool:
        """
        Handle multi-timeframe confluence analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        return self.multi_tf_handler.handle(command)
    
    def handle_smc(self, command: str) -> bool:
        """
        Handle Smart Money Concepts analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        return self.smc_handler.handle(command)
    
    def handle_elliott_wave(self, command: str) -> bool:
        """
        Handle Elliott Wave analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        return self.elliott_wave_handler.handle(command)
    
    def handle_imbalance(self, command: str) -> bool:
        """
        Handle order flow imbalance analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        import asyncio
        
        # Parse command arguments
        parts = command.split()
        symbol = "XRP/USDT"  # default
        duration = 30  # default
        
        for part in parts[1:]:  # Skip 'imbalance'
            if part.startswith('s='):
                symbol = part[2:]
            elif part.startswith('d='):
                try:
                    duration = int(part[2:])
                except ValueError:
                    print(f"Invalid duration: {part[2:]}")
                    return True
        
        # Run the async handler
        try:
            asyncio.run(handle_imbalance_analysis(symbol, duration))
        except Exception as e:
            print(f"Error running imbalance analysis: {e}")
        
        return True
    
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
        
        # Handle ATR+ADX command
        if command.startswith('atr_adx'):
            self.handle_atr_adx(command)
            return True
        
        # Handle Bollinger Bands command
        if command.startswith('bb'):
            self.handle_bollinger_bands(command)
            return True
        
        # Handle divergence command
        if command.startswith('divergence'):
            self.handle_divergence(command)
            return True
        
        # Handle regime command
        if command.startswith('regime'):
            self.handle_regime(command)
            return True
        
        # Handle VWAP command
        if command.startswith('vwap'):
            result = handle_vwap_command(command)
            print(result)
            return True
        
        # Handle multi-timeframe confluence command
        if command.startswith('multi_tf'):
            self.handle_multi_tf(command)
            return True
        
        # Handle SMC command
        if command.startswith('smc'):
            self.handle_smc(command)
            return True
        
        # Handle all trend command
        if command.startswith('all_trend'):
            self.handle_all_trend(command)
            return True
        
        # Handle double bottom pattern command
        if command.startswith('double_bottom'):
            command_parts = command.split()
            args = parse_double_bottom_args(command_parts)
            result = handle_double_bottom_command(args)
            print(result)
            return True
        
        # Handle head and shoulders pattern command
        if command.startswith('head_and_shoulders'):
            # Extract command arguments after 'head_and_shoulders'
            command_args = command[len('head_and_shoulders'):].strip()
            result = handle_head_and_shoulders_command(command_args)
            print(result)
            return True
        
        # Handle inverse head and shoulders pattern command
        if command.startswith('inverse_head_and_shoulders'):
            # Extract command arguments after 'inverse_head_and_shoulders'
            command_args = command[len('inverse_head_and_shoulders'):].strip()
            result = handle_inverse_head_and_shoulders_command(command_args)
            print(result)
            return True
        
        # Handle triangle pattern command
        if command.startswith('triangle'):
            command_parts = command.split()
            args = parse_triangle_args(command_parts)
            result = handle_triangle_command(args)
            print(result)
            return True
        
        # Handle flag pattern command
        if command.startswith('flag'):
            command_parts = command.split()
            args = parse_flag_args(command_parts)
            result = handle_flag_command(args)
            print(result)
            return True
        
        # Handle wedge pattern command
        if command.startswith('wedge'):
            command_parts = command.split()
            args = parse_wedge_args(command_parts)
            result = handle_wedge_command(args)
            print(result)
            return True
        
        # Handle all patterns command
        if command.startswith('all_patterns'):
            command_parts = command.split()
            args = parse_all_patterns_args(command_parts)
            result = handle_all_patterns_command(args)
            print(result)
            return True
        
        # Handle wyckoff command
        if command.startswith('wyckoff'):
            self.wyckoff_handler.handle(command)
            return True
        
        # Handle elliott wave command
        if command.startswith('elliott'):
            self.handle_elliott_wave(command)
            return True
        
        # Handle shark pattern command
        if command.startswith('shark_pattern'):
            command_parts = command.split()
            args = parse_shark_pattern_args(command_parts)
            result = handle_shark_pattern_command(args)
            print(result)
            return True
        
        # Handle butterfly pattern command
        if command.startswith('butterfly'):
            command_parts = command.split()
            args = parse_butterfly_pattern_args(command_parts)
            result = handle_butterfly_pattern_command(args)
            print(result)
            return True
        
        # Handle CVD command
        if command.startswith('cvd'):
            self.cvd_handler.handle(command)
            return True
        
        # Handle Order Book Heatmap command
        if command.startswith('orderbook_heatmap'):
            self.orderbook_heatmap_handler.handle(command)
            return True
        
        # Handle Order Flow Imbalance command
        if command.startswith('imbalance'):
            self.handle_imbalance(command)
            return True
        
        # Handle Absorption Detection command
        if command.startswith('absorption'):
            result = handle_absorption_command(command)
            return True
        
        # Handle Volume Footprint Chart command
        if command.startswith('footprint'):
            self.footprint_handler.handle(command)
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