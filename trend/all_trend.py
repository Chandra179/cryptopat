"""
All trend analysis module that orchestrates EMA 9/21, MACD, RSI 14, OBV, ATR+ADX, Bollinger Bands, Divergence, and Supertrend strategies.
Provides comprehensive market analysis by running all trend indicators together.
"""

import sys
from typing import Tuple
from trend.ema_9_21 import EMA9_21Strategy
from trend.macd import MACDStrategy
from trend.rsi_14 import RSI14Strategy
from trend.obv import OBVStrategy
from trend.atr_adx import ATR_ADXStrategy
from trend.bollinger_bands import BollingerBandsStrategy
from trend.divergence import DivergenceDetector
from trend.supertrend import SupertrendStrategy
from trend.vwap import run_vwap_analysis


class AllTrendStrategy:
    """All trend analysis strategy combining EMA, MACD, RSI, OBV, ATR+ADX, Bollinger Bands, Divergence, Supertrend, and VWAP indicators."""
    
    def __init__(self):
        self.ema_strategy = EMA9_21Strategy()
        self.macd_strategy = MACDStrategy()
        self.rsi_strategy = RSI14Strategy()
        self.obv_strategy = OBVStrategy()
        self.atr_adx_strategy = ATR_ADXStrategy()
        self.bb_strategy = BollingerBandsStrategy()
        self.divergence_detector = DivergenceDetector()
        self.supertrend_strategy = SupertrendStrategy()
    
    def analyze(self, symbol: str, timeframe: str, limit: int) -> None:
        """
        Perform comprehensive trend analysis using all strategies.
        
        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            timeframe: Timeframe (e.g., '4h', '1d', '1h')
            limit: Number of candles to analyze
        """
        print("=" * 80)
        print(f"All Trend Analysis for {symbol} ({timeframe})")
        print("=" * 80)
    
        self.ema_strategy.analyze(symbol, timeframe, limit)
        self.macd_strategy.analyze(symbol, timeframe, limit)
        self.rsi_strategy.analyze(symbol, timeframe, limit)
        self.obv_strategy.analyze(symbol, timeframe, limit)
        self.atr_adx_strategy.analyze(symbol, timeframe, limit)
        self.bb_strategy.analyze(symbol, timeframe, limit)
        self.divergence_detector.analyze(symbol, timeframe, limit)
        self.supertrend_strategy.analyze(symbol, timeframe, limit)
        
        # VWAP Analysis
        vwap_result = run_vwap_analysis(symbol, timeframe, limit)
        # Extract just the signal lines (skip header)
        vwap_lines = vwap_result.split('\n')[2:]  # Skip header and separator
        for line in vwap_lines[-5:]:  # Show last 5 signals
            if line.strip():
                print(line)


def parse_command(command: str) -> Tuple[str, str, int]:
    """
    Parse terminal command: all_trend s=XRP/USDT t=4h l=100
    
    Args:
        command: Command string
        
    Returns:
        Tuple of (symbol, timeframe, limit)
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'all_trend':
        raise ValueError("Invalid command format. Use: all_trend s=SYMBOL t=TIMEFRAME l=LIMIT")
    
    symbol = None
    timeframe = '4h'  # default
    limit = 100  # default
    
    for part in parts[1:]:
        if part.startswith('s='):
            symbol = part[2:]
        elif part.startswith('t='):
            timeframe = part[2:]
        elif part.startswith('l='):
            try:
                limit = int(part[2:])
            except ValueError:
                raise ValueError(f"Invalid limit value: {part[2:]}")
    
    if symbol is None:
        raise ValueError("Symbol (s=) is required")
    
    return symbol, timeframe, limit


def main():
    """Main entry point for all trend analysis."""
    if len(sys.argv) < 2:
        print("Usage: python all_trend.py s=SYMBOL t=TIMEFRAME l=LIMIT")
        print("Example: python all_trend.py s=XRP/USDT t=4h l=100")
        return
    
    try:
        # Parse command line arguments
        command = ' '.join(['all_trend'] + sys.argv[1:])
        symbol, timeframe, limit = parse_command(command)
        
        # Run analysis
        strategy = AllTrendStrategy()
        strategy.analyze(symbol, timeframe, limit)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()