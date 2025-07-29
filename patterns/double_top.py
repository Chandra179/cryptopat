"""
Double Top Pattern Detection Module.

Detects double top reversal patterns using High, Close, and Volume data.
A double top is a bearish reversal pattern consisting of two swing highs
at approximately the same level, separated by an intervening valley.
"""

import logging
from typing import List, Dict
from datetime import datetime
import numpy as np
from data import get_data_collector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DoubleTopDetector:
    """Double Top pattern detection and analysis."""
    
    def __init__(self, tolerance_pct: float = 2.0, min_valley_depth_pct: float = 3.0):
        """
        Initialize Double Top detector.
        
        Args:
            tolerance_pct: Tolerance for considering two highs as equal (in %)
            min_valley_depth_pct: Minimum depth of intervening valley below highs (in %)
        """
        self.tolerance_pct = tolerance_pct
        self.min_valley_depth_pct = min_valley_depth_pct
        self.data_collector = get_data_collector()
    
    def detect_pattern(self, symbol: str, timeframe: str = '4h', limit: int = 200) -> Dict:
        """
        Detect double top pattern for given symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol (e.g., 'ETH/USDT')
            timeframe: Timeframe for analysis
            limit: Number of candles to analyze
        
        Returns:
            Dictionary with pattern analysis results
        """
        try:
            # Fetch OHLCV data
            ohlcv_data = self.data_collector.fetch_ohlcv_data(symbol, timeframe, limit)
            
            if len(ohlcv_data) < 50:
                return {"error": "Insufficient data for pattern detection"}
            
            # Extract price and volume arrays
            timestamps = [candle[0] for candle in ohlcv_data]
            highs = np.array([candle[2] for candle in ohlcv_data])
            lows = np.array([candle[3] for candle in ohlcv_data])
            closes = np.array([candle[4] for candle in ohlcv_data])
            volumes = np.array([candle[5] for candle in ohlcv_data])
            
            # Find swing highs and lows
            swing_highs = self._find_swing_highs(highs, window=5)
            swing_lows = self._find_swing_lows(lows, window=5)
            
            # Detect double top pattern
            pattern_data = self._detect_double_top(
                timestamps, highs, lows, closes, volumes, 
                swing_highs, swing_lows
            )
            
            return pattern_data
            
        except Exception as e:
            logger.error(f"Error detecting double top pattern for {symbol}: {e}")
            return {"error": str(e)}
    
    def _find_swing_highs(self, highs: np.ndarray, window: int = 5) -> List[int]:
        """Find swing high indices where price is highest in the window."""
        swing_highs = []
        
        for i in range(window, len(highs) - window):
            is_swing_high = True
            current_high = highs[i]
            
            # Check if current point is highest in the window
            for j in range(i - window, i + window + 1):
                if j != i and highs[j] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append(i)
        
        return swing_highs
    
    def _find_swing_lows(self, lows: np.ndarray, window: int = 5) -> List[int]:
        """Find swing low indices where price is lowest in the window."""
        swing_lows = []
        
        for i in range(window, len(lows) - window):
            is_swing_low = True
            current_low = lows[i]
            
            # Check if current point is lowest in the window
            for j in range(i - window, i + window + 1):
                if j != i and lows[j] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append(i)
        
        return swing_lows
    
    def _detect_double_top(self, timestamps: List, highs: np.ndarray, 
                          lows: np.ndarray, closes: np.ndarray, 
                          volumes: np.ndarray, swing_highs: List[int], 
                          swing_lows: List[int]) -> Dict:
        """
        Detect double top pattern from swing points.
        
        Returns:
            Dictionary with pattern details and signals
        """
        pattern_result = {
            "pattern_detected": False,
            "high1_price": None,
            "high1_index": None,
            "high1_timestamp": None,
            "valley_price": None,
            "valley_index": None,
            "valley_timestamp": None,
            "high2_price": None,
            "high2_index": None,
            "high2_timestamp": None,
            "neckline": None,
            "current_price": closes[-1],
            "signal": "NONE",
            "pattern_status": "No Pattern",
            "volume_confirmation": False,
            "breakdown_confirmed": False
        }
        
        if len(swing_highs) < 2:
            return pattern_result
        
        # Look for potential double top patterns
        for i in range(len(swing_highs) - 1):
            high1_idx = swing_highs[i]
            high1_price = highs[high1_idx]
            
            # Find intervening valley between high1 and potential high2
            potential_valleys = [l for l in swing_lows if l > high1_idx]
            if not potential_valleys:
                continue
            
            # Look for second high after the valley
            for j in range(i + 1, len(swing_highs)):
                high2_idx = swing_highs[j]
                high2_price = highs[high2_idx]
                
                # Find the valley between high1 and high2
                intervening_valleys = [l for l in swing_lows if high1_idx < l < high2_idx]
                if not intervening_valleys:
                    continue
                
                valley_idx = min(intervening_valleys, key=lambda x: lows[x])
                valley_price = lows[valley_idx]
                
                # Check if the two highs are within tolerance
                price_diff_pct = abs(high1_price - high2_price) / max(high1_price, high2_price) * 100
                
                if price_diff_pct <= self.tolerance_pct:
                    # Check if valley is deep enough below the highs
                    max_high = max(high1_price, high2_price)
                    valley_depth_pct = (max_high - valley_price) / max_high * 100
                    
                    if valley_depth_pct >= self.min_valley_depth_pct:
                        # Valid double top pattern found
                        pattern_result.update({
                            "pattern_detected": True,
                            "high1_price": high1_price,
                            "high1_index": high1_idx,
                            "high1_timestamp": datetime.fromtimestamp(timestamps[high1_idx] / 1000),
                            "valley_price": valley_price,
                            "valley_index": valley_idx,
                            "valley_timestamp": datetime.fromtimestamp(timestamps[valley_idx] / 1000),
                            "high2_price": high2_price,
                            "high2_index": high2_idx,
                            "high2_timestamp": datetime.fromtimestamp(timestamps[high2_idx] / 1000),
                            "neckline": valley_price
                        })
                        
                        # Check volume confirmation (lower volume on second high)
                        vol1 = volumes[high1_idx]
                        vol2 = volumes[high2_idx]
                        pattern_result["volume_confirmation"] = vol2 <= vol1
                        
                        # Check current price relative to neckline
                        current_price = closes[-1]
                        
                        if current_price < valley_price:
                            # Breakdown confirmed
                            pattern_result.update({
                                "signal": "SELL",
                                "pattern_status": "Breakdown Confirmed",
                                "breakdown_confirmed": True
                            })
                        elif high2_idx >= len(closes) - 10:  # Recent second high
                            pattern_result.update({
                                "signal": "NONE",
                                "pattern_status": "Pattern Forming"
                            })
                        else:
                            pattern_result.update({
                                "signal": "NONE",
                                "pattern_status": "Pattern Complete - Awaiting Breakdown"
                            })
                        
                        return pattern_result
        
        return pattern_result
    
    def format_analysis(self, symbol: str, timeframe: str, pattern_data: Dict) -> str:
        """
        Format pattern analysis for terminal output.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe used
            pattern_data: Pattern detection results
        
        Returns:
            Formatted analysis string
        """
        if "error" in pattern_data:
            return f"❌ Error analyzing {symbol}: {pattern_data['error']}"
        
        if not pattern_data["pattern_detected"]:
            return f"{symbol} ({timeframe}): No Double Top pattern detected"
        
        result_lines = [f"\nDouble Top Analysis - {symbol} ({timeframe})"]
        
        # Pattern details
        high1_time = pattern_data["high1_timestamp"].strftime("%Y-%m-%d %H:%M")
        valley_time = pattern_data["valley_timestamp"].strftime("%Y-%m-%d %H:%M")
        high2_time = pattern_data["high2_timestamp"].strftime("%Y-%m-%d %H:%M")
        
        result_lines.append(
            f"[{high1_time}] High1: {pattern_data['high1_price']:.4f} | "
            f"Valley: {pattern_data['valley_price']:.4f} | "
            f"High2: {pattern_data['high2_price']:.4f} | "
            f"Neckline: {pattern_data['neckline']:.4f}"
        )
        
        # Current status
        signal_emoji = {"BUY": "🚀", "SELL": "📉", "NONE": "⏳"}
        status_emoji = {"Breakdown Confirmed": "✅", "Pattern Forming": "⏳", 
                       "Pattern Complete - Awaiting Breakdown": "⏸️"}
        
        result_lines.append(
            f"Price: {pattern_data['current_price']:.4f} | "
            f"Signal: {pattern_data['signal']} {signal_emoji.get(pattern_data['signal'], '')} | "
            f"{status_emoji.get(pattern_data['pattern_status'], '')} {pattern_data['pattern_status']}"
        )
        
        # Volume confirmation
        vol_status = "✅ Volume Confirmed" if pattern_data["volume_confirmation"] else "⚠️ Volume Warning"
        result_lines.append(f"Volume Analysis: {vol_status}")
        
        return "\n".join(result_lines)


def analyze_double_top(symbol: str, timeframe: str = '4h', limit: int = 200) -> str:
    """
    Analyze double top pattern for a symbol.
    
    Args:
        symbol: Trading pair symbol (e.g., 'ETH/USDT')
        timeframe: Timeframe for analysis
        limit: Number of candles to analyze
    
    Returns:
        Formatted analysis string
    """
    detector = DoubleTopDetector()
    pattern_data = detector.detect_pattern(symbol, timeframe, limit)
    return detector.format_analysis(symbol, timeframe, pattern_data)


def parse_command(command: str):
    """Parse command line arguments for double top analysis."""
    parts = command.split()
    
    symbol = "ETH/USDT"  # default
    timeframe = "4h"     # default
    limit = 100          # default
    
    for part in parts[1:]:  # Skip 'double_top'
        if part.startswith('s='):
            symbol = part[2:]
        elif part.startswith('t='):
            timeframe = part[2:]
        elif part.startswith('l='):
            limit = int(part[2:])
    
    return symbol, timeframe, limit


if __name__ == "__main__":
    # Example usage
    result = analyze_double_top("ETH/USDT", "4h", 100)
    print(result)