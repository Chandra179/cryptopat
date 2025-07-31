"""
CLI handler for absorption detection commands.
Provides parsing and execution of absorption analysis commands.
"""

import logging
from typing import Tuple, Optional
from orderflow.absorption import run_absorption_analysis

logger = logging.getLogger(__name__)

def parse_absorption_command(command: str) -> Tuple[str, float, int, float, float, int, float]:
    """
    Parse absorption command arguments.
    
    Expected format: absorption s=BTC/USDT t=1s w=10 vol_th=5000 price_th=2
    
    Args:
        command: Full command string
        
    Returns:
        Tuple of (symbol, snapshot_interval, window_seconds, volume_threshold, 
                 price_threshold, duration, snapshot_interval)
    """
    # Default values
    symbol = "BTC/USDT"
    snapshot_interval = 1.0  # seconds between snapshots
    window_seconds = 10
    volume_threshold = 5000.0
    price_threshold = 2.0
    duration = 60  # analysis duration in seconds
    
    # Parse command parts
    parts = command.split()
    
    for part in parts[1:]:  # Skip 'absorption'
        if part.startswith('s='):
            symbol = part[2:]
        elif part.startswith('t='):
            # Parse snapshot interval (e.g., t=1s -> 1.0 seconds)
            interval_str = part[2:]
            if interval_str.endswith('s'):
                interval_str = interval_str[:-1]
            try:
                snapshot_interval = float(interval_str)
            except ValueError:
                logger.warning(f"Invalid snapshot interval: {part[2:]}, using default 1.0s")
        elif part.startswith('w='):
            try:
                window_seconds = int(part[2:])
            except ValueError:
                logger.warning(f"Invalid window seconds: {part[2:]}, using default 10s")
        elif part.startswith('vol_th='):
            try:
                volume_threshold = float(part[7:])
            except ValueError:
                logger.warning(f"Invalid volume threshold: {part[7:]}, using default 5000")
        elif part.startswith('price_th='):
            try:
                price_threshold = float(part[9:])
            except ValueError:
                logger.warning(f"Invalid price threshold: {part[9:]}, using default 2.0")
        elif part.startswith('d='):
            try:
                duration = int(part[2:])
            except ValueError:
                logger.warning(f"Invalid duration: {part[2:]}, using default 60s")
    
    return symbol, snapshot_interval, window_seconds, volume_threshold, price_threshold, duration, snapshot_interval

def handle_absorption_command(command: str) -> bool:
    """
    Handle absorption detection command.
    
    Args:
        command: The command string
        
    Returns:
        True if command was handled successfully, False otherwise
    """
    try:
        # Parse command arguments
        symbol, snapshot_interval, window_seconds, volume_threshold, price_threshold, duration, _ = parse_absorption_command(command)
        
        logger.info(f"Running absorption analysis: {symbol}, window={window_seconds}s, "
                   f"vol_th={volume_threshold}, price_th={price_threshold}")
        
        # Run absorption analysis
        events = run_absorption_analysis(
            symbol=symbol,
            window_seconds=window_seconds,
            volume_threshold=volume_threshold,
            price_threshold=price_threshold,
            duration=duration,
            snapshot_interval=snapshot_interval
        )
        
        # Analysis results are printed by the absorption module
        # Just return success
        return True
        
    except Exception as e:
        logger.error(f"Error in absorption analysis: {e}")
        print(f"❌ Absorption analysis failed: {e}")
        return False

def get_absorption_help() -> str:
    """
    Get help text for absorption command.
    
    Returns:
        Help text string
    """
    return """  absorption s=BTC/USDT t=1s w=10 vol_th=5000 price_th=2 d=60
  - s=SYMBOL: Trading pair (required) - e.g., BTC/USDT, ETH/USDT
  - t=INTERVAL: Snapshot interval - e.g., 1s, 0.5s (default: 1s)
  - w=WINDOW: Analysis window in seconds (default: 10)
  - vol_th=VOLUME: Minimum volume threshold (default: 5000)
  - price_th=TICKS: Maximum price movement in ticks (default: 2)
  - d=DURATION: Analysis duration in seconds (default: 60)"""