"""
Butterfly Pattern Detection Module.

Detects Butterfly harmonic pattern using OHLCV data with Fibonacci retracement validation.
A Butterfly pattern is a reversal pattern consisting of five points (X-A-B-C-D) with specific
Fibonacci relationships: AB=0.786 of XA, BC=0.382-0.886 of AB, CD=1.618-2.618 of BC, AD=1.27 of XA.
"""

import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np
from data import get_data_collector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ButterflyPatternDetector:
    """Butterfly harmonic pattern detection and analysis."""
    
    def __init__(self, zigzag_threshold: float = 5.0):
        """
        Initialize Butterfly pattern detector.
        
        Args:
            zigzag_threshold: ZigZag threshold percentage for swing point detection
        """
        self.zigzag_threshold = zigzag_threshold
        self.data_collector = get_data_collector()
        
        # Fibonacci validation ranges for Butterfly pattern
        self.ab_retracement = 0.786      # AB must be 0.786 retracement of XA
        self.ab_tolerance = 0.02         # ±2% tolerance for AB
        self.bc_retracement_min = 0.382  # BC minimum retracement of AB
        self.bc_retracement_max = 0.886  # BC maximum retracement of AB
        self.cd_extension_min = 1.618    # CD minimum extension of BC
        self.cd_extension_max = 2.618    # CD maximum extension of BC
        self.ad_extension = 1.27         # AD must be 1.27 extension of XA
        self.ad_tolerance = 0.05         # ±5% tolerance for AD
    
    def detect_pattern(self, symbol: str, timeframe: str = '4h', limit: int = 150) -> Dict:
        """
        Detect Butterfly pattern for given symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol (e.g., 'XRP/USDT')
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
            
            # Create ZigZag swing points
            swing_points = self._create_zigzag_points(highs, lows, self.zigzag_threshold)
            
            # Detect Butterfly pattern
            pattern_data = self._detect_butterfly_pattern(
                timestamps, highs, lows, closes, volumes, swing_points
            )
            
            return pattern_data
            
        except Exception as e:
            logger.error(f"Error detecting Butterfly pattern for {symbol}: {e}")
            return {"error": str(e)}
    
    def _create_zigzag_points(self, highs: np.ndarray, lows: np.ndarray, 
                             threshold_pct: float) -> List[Tuple[int, float, str]]:
        """
        Create ZigZag swing points with specified threshold percentage.
        
        Args:
            highs: Array of high prices
            lows: Array of low prices
            threshold_pct: Minimum percentage change for swing point
        
        Returns:
            List of (index, price, type) tuples where type is 'high' or 'low'
        """
        swing_points = []
        
        # Start with first point
        if len(highs) == 0:
            return swing_points
            
        current_high = highs[0]
        current_low = lows[0]
        current_high_idx = 0
        current_low_idx = 0
        last_swing_type = None
        
        for i in range(1, len(highs)):
            # Update current extremes
            if highs[i] > current_high:
                current_high = highs[i]
                current_high_idx = i
            if lows[i] < current_low:
                current_low = lows[i]
                current_low_idx = i
            
            # Check for swing high
            if last_swing_type != 'high':
                change_pct = (current_high - current_low) / current_low * 100
                if change_pct >= threshold_pct:
                    swing_points.append((current_high_idx, current_high, 'high'))
                    current_low = lows[i]
                    current_low_idx = i
                    last_swing_type = 'high'
            
            # Check for swing low
            if last_swing_type != 'low':
                change_pct = (current_high - current_low) / current_high * 100
                if change_pct >= threshold_pct:
                    swing_points.append((current_low_idx, current_low, 'low'))
                    current_high = highs[i]
                    current_high_idx = i
                    last_swing_type = 'low'
        
        return swing_points
    
    def _detect_butterfly_pattern(self, timestamps: List, highs: np.ndarray, 
                                 lows: np.ndarray, closes: np.ndarray, 
                                 volumes: np.ndarray, swing_points: List[Tuple]) -> Dict:
        """
        Detect Butterfly pattern from swing points using Fibonacci validation.
        
        Returns:
            Dictionary with pattern details and signals
        """
        pattern_result = {
            "pattern_detected": False,
            "pattern_type": "BUTTERFLY",
            "bias": "NONE",
            "x_price": None, "x_index": None, "x_timestamp": None,
            "a_price": None, "a_index": None, "a_timestamp": None,
            "b_price": None, "b_index": None, "b_timestamp": None,
            "c_price": None, "c_index": None, "c_timestamp": None,
            "d_price": None, "d_index": None, "d_timestamp": None,
            "ab_retracement": None, "bc_retracement": None, "cd_extension": None, "ad_extension": None,
            "fibonacci_valid": False, "volume_spike": False, "rejection_candle": False,
            "tp1_price": None, "tp2_price": None, "stop_loss": None, "current_price": closes[-1],
            "signal": "NONE", "confidence": "NONE", "pattern_status": "No Pattern"
        }
        
        if len(swing_points) < 5:
            return pattern_result
        
        # Look for valid 5-point Butterfly patterns (X-A-B-C-D)
        for i in range(len(swing_points) - 4):
            # Extract 5 consecutive swing points: X-A-B-C-D
            x_idx, x_price, x_type = swing_points[i]
            a_idx, a_price, a_type = swing_points[i + 1]
            b_idx, b_price, b_type = swing_points[i + 2]
            c_idx, c_price, c_type = swing_points[i + 3]
            d_idx, d_price, d_type = swing_points[i + 4]
            
            # Validate alternating pattern (high-low-high-low-high or low-high-low-high-low)
            if not self._is_valid_alternating_pattern([x_type, a_type, b_type, c_type, d_type]):
                continue
            
            # Determine pattern bias
            bias = "Bullish" if x_type == 'low' else "Bearish"
            
            # Validate Fibonacci relationships
            fib_validation = self._validate_fibonacci_ratios(
                x_price, a_price, b_price, c_price, d_price, bias
            )
            
            if fib_validation["valid"]:
                # Calculate targets and stops
                targets = self._calculate_targets_and_stops(
                    x_price, a_price, b_price, c_price, d_price, bias
                )
                
                # Check for volume spike at D
                volume_spike = self._check_volume_spike(volumes, d_idx)
                
                # Check for rejection candle at D
                rejection_candle = self._check_rejection_candle(
                    highs, lows, closes, d_idx, bias
                )
                
                # Calculate confidence
                confidence = self._calculate_confidence(
                    fib_validation, volume_spike, rejection_candle
                )
                
                # Update pattern result
                pattern_result.update({
                    "pattern_detected": True,
                    "bias": bias,
                    "x_price": x_price, "x_index": x_idx,
                    "x_timestamp": datetime.fromtimestamp(timestamps[x_idx] / 1000),
                    "a_price": a_price, "a_index": a_idx,
                    "a_timestamp": datetime.fromtimestamp(timestamps[a_idx] / 1000),
                    "b_price": b_price, "b_index": b_idx,
                    "b_timestamp": datetime.fromtimestamp(timestamps[b_idx] / 1000),
                    "c_price": c_price, "c_index": c_idx,
                    "c_timestamp": datetime.fromtimestamp(timestamps[c_idx] / 1000),
                    "d_price": d_price, "d_index": d_idx,
                    "d_timestamp": datetime.fromtimestamp(timestamps[d_idx] / 1000),
                    "ab_retracement": fib_validation["ab_ratio"],
                    "bc_retracement": fib_validation["bc_ratio"],
                    "cd_extension": fib_validation["cd_ratio"],
                    "ad_extension": fib_validation["ad_ratio"],
                    "fibonacci_valid": True,
                    "volume_spike": volume_spike,
                    "rejection_candle": rejection_candle,
                    "tp1_price": targets["tp1"],
                    "tp2_price": targets["tp2"],
                    "stop_loss": targets["stop_loss"],
                    "signal": "BUY" if bias == "Bullish" else "SELL",
                    "confidence": confidence,
                    "pattern_status": "✅ VALID"
                })
                
                return pattern_result
        
        return pattern_result
    
    def _is_valid_alternating_pattern(self, types: List[str]) -> bool:
        """Check if swing points alternate between high and low."""
        for i in range(1, len(types)):
            if types[i] == types[i-1]:
                return False
        return True
    
    def _validate_fibonacci_ratios(self, x_price: float, a_price: float, b_price: float,
                                  c_price: float, d_price: float, bias: str) -> Dict:
        """
        Validate Fibonacci relationships for Butterfly pattern.
        
        Args:
            x_price, a_price, b_price, c_price, d_price: The five key prices
            bias: "Bullish" or "Bearish"
        
        Returns:
            Dictionary with validation results and ratios
        """
        # Calculate distances
        xa_distance = abs(a_price - x_price)
        ab_distance = abs(b_price - a_price)
        bc_distance = abs(c_price - b_price)
        cd_distance = abs(d_price - c_price)
        ad_distance = abs(d_price - a_price)
        
        # Calculate ratios
        ab_ratio = ab_distance / xa_distance if xa_distance > 0 else 0
        bc_ratio = bc_distance / ab_distance if ab_distance > 0 else 0
        cd_ratio = cd_distance / bc_distance if bc_distance > 0 else 0
        ad_ratio = ad_distance / xa_distance if xa_distance > 0 else 0
        
        # Validate AB retracement (must be 0.786 ±2%)
        ab_valid = abs(ab_ratio - self.ab_retracement) <= self.ab_tolerance
        
        # Validate BC retracement (0.382 to 0.886 of AB)
        bc_valid = self.bc_retracement_min <= bc_ratio <= self.bc_retracement_max
        
        # Validate CD extension (1.618 to 2.618 of BC)
        cd_valid = self.cd_extension_min <= cd_ratio <= self.cd_extension_max
        
        # Validate AD extension (1.27 of XA ±5%)
        ad_valid = abs(ad_ratio - self.ad_extension) <= self.ad_tolerance
        
        return {
            "valid": ab_valid and bc_valid and cd_valid and ad_valid,
            "ab_ratio": ab_ratio,
            "bc_ratio": bc_ratio,
            "cd_ratio": cd_ratio,
            "ad_ratio": ad_ratio,
            "ab_valid": ab_valid,
            "bc_valid": bc_valid,
            "cd_valid": cd_valid,
            "ad_valid": ad_valid
        }
    
    def _calculate_targets_and_stops(self, x_price: float, a_price: float, b_price: float,
                                   c_price: float, d_price: float, bias: str) -> Dict:
        """Calculate target and stop loss levels according to Butterfly pattern rules."""
        if bias == "Bullish":
            # Calculate CD leg distance
            cd_distance = d_price - c_price
            # TP1: 38.2% retracement of CD from D
            tp1 = d_price + (cd_distance * 0.382)
            # TP2: 61.8% retracement of CD from D
            tp2 = d_price + (cd_distance * 0.618)
            # Stop Loss: slightly beyond point X
            stop_loss = x_price - (abs(a_price - x_price) * 0.05)  # 5% beyond X
        else:
            # Bearish pattern
            cd_distance = c_price - d_price
            # TP1: 38.2% retracement of CD from D
            tp1 = d_price - (cd_distance * 0.382)
            # TP2: 61.8% retracement of CD from D
            tp2 = d_price - (cd_distance * 0.618)
            # Stop Loss: slightly beyond point X
            stop_loss = x_price + (abs(x_price - a_price) * 0.05)  # 5% beyond X
        
        return {"tp1": tp1, "tp2": tp2, "stop_loss": stop_loss}
    
    def _check_volume_spike(self, volumes: np.ndarray, d_idx: int) -> bool:
        """Check for volume spike at point D."""
        if d_idx < 10 or d_idx >= len(volumes):
            return False
        
        # Compare D volume with average of previous 10 candles
        avg_volume = np.mean(volumes[max(0, d_idx-10):d_idx])
        d_volume = volumes[d_idx]
        
        return d_volume > avg_volume * 1.5  # 50% above average
    
    def _check_rejection_candle(self, highs: np.ndarray, lows: np.ndarray, 
                               closes: np.ndarray, d_idx: int, bias: str) -> bool:
        """Check for rejection candle or engulfing pattern at point D."""
        if d_idx >= len(closes) - 1:
            return False
        
        # Check next candle after D for rejection
        next_idx = d_idx + 1
        d_close = closes[d_idx]
        next_close = closes[next_idx]
        
        if bias == "Bullish":
            # Look for bullish engulfing or hammer-like rejection
            return next_close > d_close * 1.02  # 2% bullish move
        else:
            # Look for bearish engulfing or shooting star-like rejection
            return next_close < d_close * 0.98  # 2% bearish move
    
    def _calculate_confidence(self, fib_validation: Dict, volume_spike: bool, 
                            rejection_candle: bool) -> str:
        """Calculate confidence level based on validation criteria."""
        score = 0
        
        # Fibonacci validation (max 80 points)
        if fib_validation["ab_valid"]:
            score += 20
        if fib_validation["bc_valid"]:
            score += 20
        if fib_validation["cd_valid"]:
            score += 20
        if fib_validation["ad_valid"]:
            score += 20
        
        # Volume confirmation (10 points)
        if volume_spike:
            score += 10
        
        # Rejection candle (10 points)
        if rejection_candle:
            score += 10
        
        # Determine confidence level
        if score >= 85:
            return "HIGH"
        elif score >= 70:
            return "MEDIUM"
        else:
            return "LOW"
    
    def format_analysis(self, symbol: str, timeframe: str, pattern_data: Dict) -> str:
        """
        Format Butterfly pattern analysis for terminal output.
        
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
            return f"{symbol} ({timeframe}) - No Butterfly Pattern Detected"
        
        # Format the Butterfly pattern output
        bias_emoji = "📈" if pattern_data["bias"] == "Bullish" else "📉"
        
        ab_ratio = pattern_data["ab_retracement"]
        bc_ratio = pattern_data["bc_retracement"]
        cd_ratio = pattern_data["cd_extension"]
        ad_ratio = pattern_data["ad_extension"]
        
        volume_check = "✅" if pattern_data["volume_spike"] else "❌"
        rejection_check = "✅" if pattern_data["rejection_candle"] else "❌"
        
        return (
            f"[HARMONIC STRUCTURE: BUTTERFLY]\n"
            f"Symbol: {symbol} | Timeframe: {timeframe.upper()}\n"
            f"Pattern Status: {pattern_data['pattern_status']} | Bias: {bias_emoji} {pattern_data['bias']}\n"
            f"• X: {pattern_data['x_price']:.3f}\n"
            f"• A: {pattern_data['a_price']:.3f}\n"
            f"• B: {pattern_data['b_price']:.3f} (AB retrace: {ab_ratio:.3f}) ✅\n"
            f"• C: {pattern_data['c_price']:.3f} (BC retrace: {bc_ratio:.3f}) ✅\n"
            f"• D: {pattern_data['d_price']:.3f} (CD ext: {cd_ratio:.3f}) ✅ → 📍 Entry\n"
            f"Fibonacci Confluence ✅ | Volume Spike {volume_check} | Rejection Candle {rejection_check}\n"
            f"🎯 Target 1: {pattern_data['tp1_price']:.3f} (TP1)\n"
            f"🎯 Target 2: {pattern_data['tp2_price']:.3f} (TP2)\n"
            f"🛑 Stop Loss: {pattern_data['stop_loss']:.3f}\n"
            f"🚦 Signal: {pattern_data['signal']} | Confidence: {pattern_data['confidence']}"
        )


def analyze_butterfly_pattern(symbol: str, timeframe: str = '4h', limit: int = 150, 
                             zigzag_threshold: float = 5.0) -> str:
    """
    Analyze Butterfly pattern for a symbol.
    
    Args:
        symbol: Trading pair symbol (e.g., 'XRP/USDT')
        timeframe: Timeframe for analysis
        limit: Number of candles to analyze
        zigzag_threshold: ZigZag threshold percentage
    
    Returns:
        Formatted analysis string
    """
    detector = ButterflyPatternDetector(zigzag_threshold)
    pattern_data = detector.detect_pattern(symbol, timeframe, limit)
    return detector.format_analysis(symbol, timeframe, pattern_data)


if __name__ == "__main__":
    # Example usage
    result = analyze_butterfly_pattern("XRP/USDT", "4h", 150, 5.0)
    print(result)