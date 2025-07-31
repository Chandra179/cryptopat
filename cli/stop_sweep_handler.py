"""
CLI handler for stop run and liquidity sweep detection commands.
Processes terminal commands for stop_sweep analysis.
"""

import logging
from typing import Tuple, Optional
from orderflow.stop_sweep import analyze_stop_sweep

logger = logging.getLogger(__name__)

def parse_stop_sweep_command(command: str) -> Tuple[str, str, float, float, float, int, int]:
    """
    Parse stop_sweep command from terminal input.
    
    Command format: stop_sweep s=ETH/USDT t=1m zz=3 vol_th=5000 wall_vol_th=10000 w=5 s=3
    
    Args:
        command: Full command string
        
    Returns:
        Tuple of (symbol, timeframe, zigzag_threshold, volume_threshold, 
                 wall_volume_threshold, spike_window, sustain_seconds)
                 
    Raises:
        ValueError: If command format is invalid
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'stop_sweep':
        raise ValueError("Invalid command format. Use: stop_sweep s=SYMBOL t=TIMEFRAME zz=3 vol_th=5000 wall_vol_th=10000 w=5 s=3")
    
    # Default values
    symbol = None
    timeframe = '1m'
    zigzag_threshold = 3.0
    volume_threshold = 5000.0
    wall_volume_threshold = 10000.0
    spike_window = 5
    sustain_seconds = 3
    
    # Parse parameters in order to handle duplicate 's=' parameters
    for part in parts[1:]:
        try:
            if part.startswith('s='):
                if symbol is None:
                    # First s= is symbol
                    symbol = part[2:]
                else:
                    # Second s= is sustain_seconds
                    sustain_seconds = int(part[2:])
            elif part.startswith('t='):
                timeframe = part[2:]
            elif part.startswith('zz='):
                zigzag_threshold = float(part[3:])
            elif part.startswith('vol_th='):
                volume_threshold = float(part[7:])
            elif part.startswith('wall_vol_th='):
                wall_volume_threshold = float(part[12:])
            elif part.startswith('w='):
                spike_window = int(part[2:])
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid parameter format: {part}. Error: {e}")
    
    if symbol is None:
        raise ValueError("Symbol (s=) parameter is required")
    
    # Validate parameters
    if zigzag_threshold <= 0 or zigzag_threshold > 50:
        raise ValueError("ZigZag threshold must be between 0 and 50 percent")
    
    if volume_threshold <= 0:
        raise ValueError("Volume threshold must be positive")
        
    if wall_volume_threshold <= 0:
        raise ValueError("Wall volume threshold must be positive")
        
    if spike_window <= 0 or spike_window > 300:
        raise ValueError("Spike window must be between 1 and 300 seconds")
        
    if sustain_seconds <= 0 or sustain_seconds > 60:
        raise ValueError("Sustain seconds must be between 1 and 60 seconds")
    
    return symbol, timeframe, zigzag_threshold, volume_threshold, wall_volume_threshold, spike_window, sustain_seconds

def handle_stop_sweep_command(command: str) -> str:
    """
    Handle stop_sweep command execution.
    
    Args:
        command: Complete command string from CLI
        
    Returns:
        Formatted analysis results or error message
    """
    try:
        # Parse command parameters
        symbol, timeframe, zigzag_threshold, volume_threshold, wall_volume_threshold, spike_window, sustain_seconds = parse_stop_sweep_command(command)
        
        logger.info(f"Executing stop_sweep analysis for {symbol} with parameters: "
                   f"tf={timeframe}, zz={zigzag_threshold}%, vol_th={volume_threshold}, "
                   f"wall_vol_th={wall_volume_threshold}, w={spike_window}s, s={sustain_seconds}s")
        
        # Execute analysis
        result = analyze_stop_sweep(
            symbol=symbol,
            timeframe=timeframe,
            zigzag_threshold=zigzag_threshold,
            volume_threshold=volume_threshold,
            wall_volume_threshold=wall_volume_threshold,
            spike_window=spike_window,
            sustain_seconds=sustain_seconds
        )
        
        return result
        
    except ValueError as e:
        error_msg = f"Command Error: {str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Analysis Error: {str(e)}"
        logger.error(f"Unexpected error in stop_sweep analysis: {e}")
        return error_msg

def get_stop_sweep_help() -> str:
    """
    Get help text for enhanced stop_sweep command.
    
    Returns:
        Enhanced help text string
    """
    help_text = """
🔍 ENHANCED STOP RUN & LIQUIDITY SWEEP DETECTION

Command Format:
  stop_sweep s=SYMBOL t=TIMEFRAME zz=ZZ_THRESHOLD vol_th=VOL_THRESHOLD wall_vol_th=WALL_THRESHOLD w=WINDOW s=SUSTAIN

Parameters:
  s=SYMBOL           Trading pair symbol (required, e.g., ETH/USDT, BTC/USDT)
  t=TIMEFRAME        Timeframe for swing detection (default: 1m)
                     Options: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
  zz=ZZ_THRESHOLD    ZigZag threshold percentage (default: 3.0)
                     Range: 0.1 to 50.0
  vol_th=VOL_THRESHOLD    Minimum trade volume for spike detection (default: 5000)  
  wall_vol_th=WALL_THRESHOLD  Minimum resting volume for wall consumption (default: 10000)
  w=WINDOW           Window in seconds to detect volume spike (default: 5)
                     Range: 1 to 300 seconds
  s=SUSTAIN          Time in seconds to confirm reversal (default: 3)
                     Range: 1 to 60 seconds

Examples:
  stop_sweep s=ETH/USDT
  stop_sweep s=BTC/USDT t=5m zz=2.5
  stop_sweep s=XRP/USDT t=1m zz=3 vol_th=8000 wall_vol_th=15000 w=7 s=5

🚀 ENHANCED FEATURES:
  ✅ Delta Analysis: Order flow divergence confirmation using buy/sell pressure
  ✅ ATR-based Stops: Dynamic stop losses using Average True Range (2x ATR default)
  ✅ Confluence Scoring: Multiple S/R level alignment scoring (0.0-1.0)
  ✅ Market Regime Filter: Avoids signals during unfavorable conditions
  ✅ Session Filtering: London/NY/Overlap sessions preferred over Asian
  ✅ Historical Success Rate: Track and display past performance per confidence level
  ✅ Enhanced Risk/Reward: Dynamic R:R ratios based on confluence (2:1 to 2.5:1)

Detection Logic:
  1. Identifies swing highs/lows using ZigZag analysis
  2. Monitors for price probes beyond support/resistance levels  
  3. Detects volume spikes and liquidity wall consumption
  4. Analyzes order flow delta for directional confirmation
  5. Calculates ATR for dynamic stop placement
  6. Assesses market regime (volatility, volume, session)
  7. Confirms reversal back inside the level
  8. Generates enhanced signals with multi-factor confidence scoring

📊 Enhanced Output Includes:
  • Order Flow Delta & Buy/Sell Pressure Percentages
  • ATR-based Dynamic Stop Losses
  • Confluence Score (nearby level alignment)
  • Market Regime Assessment (volatility, volume, session)
  • Historical Success Rate for Similar Setups
  • Enhanced Risk/Reward Ratios

Signal Types:
  • Bullish Stop Run: Price sweeps below support, then reverses up
  • Bearish Stop Run: Price sweeps above resistance, then reverses down
  • Confidence: HIGH/MEDIUM/LOW with enhanced multi-factor scoring
"""
    return help_text.strip()

# Integration function for main CLI dispatcher
def register_stop_sweep_handler():
    """Register stop_sweep handler with main CLI system."""
    return {
        'command': 'stop_sweep',
        'handler': handle_stop_sweep_command,
        'help': get_stop_sweep_help,
        'description': 'Detect stop runs and liquidity sweeps using ZigZag levels'
    }