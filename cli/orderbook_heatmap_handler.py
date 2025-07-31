"""
CLI handler for Order Book Heatmap analysis.
"""

import asyncio
import logging
from typing import Dict, Any

from orderflow.orderbook_heatmap import OrderBookHeatmap, run_analysis

logger = logging.getLogger(__name__)

class OrderBookHeatmapHandler:
    """Handler for order book heatmap CLI commands."""
    
    def __init__(self):
        self.name = "orderbook_heatmap"
        self.description = "Real-time order book depth analysis and microstructure analytics"
    
    def print_help(self):
        """Print help information for order book heatmap commands."""
        print("  orderbook_heatmap s=XRP/USDT d=5 i=250 l=50")
        print("  orderbook_heatmap s=BTC/USDT d=10 i=500 l=25")
        print("    Parameters:")
        print("      s= : Symbol (required) - trading pair to analyze")
        print("      d= : Duration in minutes (optional, default: 5)")
        print("      i= : Snapshot interval in ms (optional, default: 250)")
        print("      l= : Depth levels to capture (optional, default: 50)")
        print("      h= : History minutes to maintain (optional, default: 60)")
    
    def parse_args(self, command_parts: list) -> Dict[str, Any]:
        """Parse command arguments."""
        args = {
            'symbol': 'XRP/USDT',
            'duration_minutes': 5,
            'snapshot_interval_ms': 250,
            'depth_levels': 50,
            'history_minutes': 60
        }
        
        for part in command_parts[1:]:  # Skip the command name
            if '=' in part:
                key, value = part.split('=', 1)
                if key == 's':
                    args['symbol'] = value.upper()
                elif key == 'd':
                    try:
                        args['duration_minutes'] = max(1, int(value))
                    except ValueError:
                        print(f"Invalid duration: {value}. Using default: 5")
                elif key == 'i':
                    try:
                        interval = int(value)
                        # Validate interval range (100-1000ms)
                        if 100 <= interval <= 1000:
                            args['snapshot_interval_ms'] = interval
                        else:
                            print(f"Interval must be between 100-1000ms. Using default: 250")
                    except ValueError:
                        print(f"Invalid interval: {value}. Using default: 250")
                elif key == 'l':
                    try:
                        levels = int(value)
                        # Validate levels range (10-100)
                        if 10 <= levels <= 100:
                            args['depth_levels'] = levels
                        else:
                            print(f"Depth levels must be between 10-100. Using default: 50")
                    except ValueError:
                        print(f"Invalid depth levels: {value}. Using default: 50")
                elif key == 'h':
                    try:
                        history = int(value)
                        # Validate history range (10-240 minutes)
                        if 10 <= history <= 240:
                            args['history_minutes'] = history
                        else:
                            print(f"History must be between 10-240 minutes. Using default: 60")
                    except ValueError:
                        print(f"Invalid history: {value}. Using default: 60")
        
        return args
    
    def handle(self, command: str) -> bool:
        """
        Handle order book heatmap analysis command.
        
        Args:
            command: The command string
            
        Returns:
            True if command was handled successfully, False otherwise
        """
        try:
            command_parts = command.split()
            args = self.parse_args(command_parts)
            
            print(f"Starting Order Book Heatmap Analysis...")
            print(f"Symbol: {args['symbol']}")
            print(f"Duration: {args['duration_minutes']} minutes")
            print(f"Snapshot Interval: {args['snapshot_interval_ms']}ms")
            print(f"Depth Levels: {args['depth_levels']}")
            print(f"History Buffer: {args['history_minutes']} minutes")
            print("=" * 60)
            
            # Run the async analysis
            result = asyncio.run(self._run_analysis(args))
            
            if result:
                self._display_results(result)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in order book heatmap handler: {e}")
            print(f"Error: {e}")
            return False
    
    async def _run_analysis(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run the order book analysis asynchronously."""
        try:
            result = await run_analysis(
                symbol=args['symbol'],
                duration_minutes=args['duration_minutes'],
                snapshot_interval_ms=args['snapshot_interval_ms'],
                depth_levels=args['depth_levels'],
                history_minutes=args['history_minutes']
            )
            return result
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            print(f"Analysis failed: {e}")
            return None
    
    def _display_results(self, result: Dict[str, Any]):
        """Display analysis results."""
        print("\nOrder Book Heatmap Analysis Results:")
        print("=" * 60)
        
        # Basic stats
        print(f"Market Regime: {result.get('final_regime', 'unknown')}")
        print(f"Total Signals Detected: {result.get('total_signals', 0)}")
        print(f"Total Snapshots Captured: {result.get('total_snapshots', 0)}")
        
        # Export info
        if 'export_path' in result:
            print(f"Data Exported To: {result['export_path']}_*.csv")
        
        # Display recent signals if available
        heatmap_data = result.get('heatmap_data', {})
        recent_signals = heatmap_data.get('recent_signals', [])
        
        if recent_signals:
            print("\nRecent Institutional Signals:")
            print("-" * 40)
            for signal in recent_signals[-5:]:  # Show last 5 signals
                print(f"  {signal.get('type', 'unknown').upper()}: {signal.get('description', 'N/A')}")
                print(f"    Confidence: {signal.get('confidence', 0):.2%}")
                print(f"    Price Level: ${signal.get('price', 0):.4f}")
                print()
        
        # Market microstructure summary
        metadata = heatmap_data.get('metadata', {})
        if metadata:
            print("Analysis Configuration:")
            print(f"  Symbol: {metadata.get('symbol', 'N/A')}")
            print(f"  Snapshot Interval: {metadata.get('snapshot_interval_ms', 'N/A')}ms")
            print(f"  Depth Levels: {metadata.get('depth_levels', 'N/A')}")
            print(f"  Total Snapshots: {metadata.get('total_snapshots', 'N/A')}")
        
        print("\nAnalysis complete! Check CSV files for detailed data.")


def handle_orderbook_heatmap_command(command: str) -> str:
    """
    Handle orderbook heatmap command from CLI.
    
    Args:
        command: Command string containing parameters
        
    Returns:
        Result string
    """
    handler = OrderBookHeatmapHandler()
    success = handler.handle(command)
    
    if success:
        return "Order book heatmap analysis completed successfully."
    else:
        return "Order book heatmap analysis failed."


def parse_orderbook_heatmap_args(command_parts: list) -> Dict[str, Any]:
    """Parse orderbook heatmap command arguments."""
    handler = OrderBookHeatmapHandler()
    return handler.parse_args(command_parts)


def get_orderbook_heatmap_help() -> str:
    """Get help text for orderbook heatmap command."""
    return """  orderbook_heatmap s=XRP/USDT d=5 i=250 l=50
  orderbook_heatmap s=BTC/USDT d=10 i=500 l=25
    Real-time order book depth analysis with institutional pattern detection
    Parameters:
      s= : Symbol (required) - trading pair to analyze
      d= : Duration in minutes (optional, default: 5)
      i= : Snapshot interval in ms (optional, default: 250)
      l= : Depth levels to capture (optional, default: 50)
      h= : History minutes to maintain (optional, default: 60)"""