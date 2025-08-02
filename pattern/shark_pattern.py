"""
Shark Pattern Detection Module.

Detects Shark harmonic pattern using OHLCV data with Fibonacci retracement validation.
A Shark pattern is a reversal pattern consisting of five points (O-X-A-B-C) with specific
Fibonacci relationships: XA=0.886, AB=1.13-1.618 of XA, BC=1.13 of OX.
"""

import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np
from data import get_data_collector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SharkStrategy:
    """Shark harmonic pattern detection and analysis."""
    
    def __init__(self, zigzag_threshold: float = 5.0):
        """
        Initialize Shark pattern detector.
        
        Args:
            zigzag_threshold: ZigZag threshold percentage for swing point detection
        """
        self.zigzag_threshold = zigzag_threshold
        self.data_collector = get_data_collector()
        
        # Fibonacci validation ranges
        self.xa_retracement = 0.886  # XA must be 0.886 retracement
        self.xa_tolerance = 0.02     # ±2% tolerance for XA
        self.ab_extension_min = 1.13  # AB minimum extension of XA
        self.ab_extension_max = 1.618 # AB maximum extension of XA
        self.bc_extension = 1.13      # BC must be 1.13 extension of OX
        self.bc_tolerance = 0.05      # ±5% tolerance for BC
    
    def analyze(self, symbol: str, timeframe: str, limit: int) -> Dict:
        """
        Analyze Shark pattern for given symbol and timeframe.
        
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
            
            if not ohlcv_data or len(ohlcv_data) < 50:
                return {
                    'error': f'Insufficient data: need at least 50 candles, got {len(ohlcv_data) if ohlcv_data else 0}',
                    'success': False,
                    'symbol': symbol,
                    'timeframe': timeframe
                }
            
            # Extract price and volume arrays
            timestamps = [candle[0] for candle in ohlcv_data]
            highs = np.array([candle[2] for candle in ohlcv_data])
            lows = np.array([candle[3] for candle in ohlcv_data])
            closes = np.array([candle[4] for candle in ohlcv_data])
            volumes = np.array([candle[5] for candle in ohlcv_data])
            
            # Create ZigZag swing points
            swing_points = self._create_zigzag_points(highs, lows, self.zigzag_threshold)
            
            # Detect Shark pattern
            pattern_data = self._detect_shark_pattern(
                timestamps, highs, lows, closes, volumes, swing_points
            )
            
            # Convert to expected format
            current_price = closes[-1]
            current_timestamp = timestamps[-1]
            dt = datetime.fromtimestamp(current_timestamp / 1000)
            
            result = {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'analysis_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': current_timestamp,
                'total_candles': len(ohlcv_data),
                'current_price': round(current_price, 4),
                'pattern_detected': pattern_data["pattern_detected"]
            }
            
            if pattern_data["pattern_detected"]:
                result.update({
                    'pattern_type': 'SHARK',
                    'bias': pattern_data["bias"],
                    'signal': pattern_data["signal"],
                    'confidence_score': pattern_data["confidence"],
                    'target_price': pattern_data["target_price"],
                    'stop_loss': pattern_data["stop_loss"],
                    'entry_window': "At point C" if pattern_data["signal"] != "NONE" else "Wait for pattern completion",
                    'exit_trigger': "Stop loss hit or target reached",
                    'fibonacci_valid': pattern_data["fibonacci_valid"],
                    'volume_spike': pattern_data["volume_spike"],
                    'rejection_candle': pattern_data["rejection_candle"],
                    'pattern_points': {
                        'O': {'price': pattern_data["o_price"], 'index': pattern_data["o_index"]},
                        'X': {'price': pattern_data["x_price"], 'index': pattern_data["x_index"]},
                        'A': {'price': pattern_data["a_price"], 'index': pattern_data["a_index"]},
                        'B': {'price': pattern_data["b_price"], 'index': pattern_data["b_index"]},
                        'C': {'price': pattern_data["c_price"], 'index': pattern_data["c_index"]}
                    },
                    'fibonacci_ratios': {
                        'xa_retracement': pattern_data["xa_retracement"],
                        'ab_extension': pattern_data["ab_extension"],
                        'bc_extension': pattern_data["bc_extension"]
                    },
                    'raw_data': {
                        'ohlcv_data': ohlcv_data,
                        'pattern_data': pattern_data
                    }
                })
            else:
                result.update({
                    'pattern_type': None,
                    'bias': 'NEUTRAL',
                    'signal': 'HOLD',
                    'confidence_score': 0,
                    'entry_window': "No pattern detected",
                    'exit_trigger': "Wait for pattern formation"
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing Shark pattern for {symbol}: {e}")
            return {
                'error': f'Analysis failed: {str(e)}',
                'success': False,
                'symbol': symbol,
                'timeframe': timeframe
            }
    
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
    
    def _detect_shark_pattern(self, timestamps: List, highs: np.ndarray, 
                             lows: np.ndarray, closes: np.ndarray, 
                             volumes: np.ndarray, swing_points: List[Tuple]) -> Dict:
        """
        Detect Shark pattern from swing points using Fibonacci validation.
        
        Returns:
            Dictionary with pattern details and signals
        """
        pattern_result = {
            "pattern_detected": False,
            "pattern_type": "SHARK",
            "bias": "NONE",
            "o_price": None, "o_index": None, "o_timestamp": None,
            "x_price": None, "x_index": None, "x_timestamp": None,
            "a_price": None, "a_index": None, "a_timestamp": None,
            "b_price": None, "b_index": None, "b_timestamp": None,
            "c_price": None, "c_index": None, "c_timestamp": None,
            "xa_retracement": None, "ab_extension": None, "bc_extension": None,
            "fibonacci_valid": False, "volume_spike": False, "rejection_candle": False,
            "target_price": None, "stop_loss": None, "current_price": closes[-1],
            "signal": "NONE", "confidence": "NONE", "pattern_status": "No Pattern"
        }
        
        if len(swing_points) < 5:
            return pattern_result
        
        # Look for valid 5-point Shark patterns
        for i in range(len(swing_points) - 4):
            # Extract 5 consecutive swing points: O-X-A-B-C
            o_idx, o_price, o_type = swing_points[i]
            x_idx, x_price, x_type = swing_points[i + 1]
            a_idx, a_price, a_type = swing_points[i + 2]
            b_idx, b_price, b_type = swing_points[i + 3]
            c_idx, c_price, c_type = swing_points[i + 4]
            
            # Validate alternating pattern (high-low-high-low-high or low-high-low-high-low)
            if not self._is_valid_alternating_pattern([o_type, x_type, a_type, b_type, c_type]):
                continue
            
            # Determine pattern bias
            bias = "Bullish" if o_type == 'low' else "Bearish"
            
            # Validate Fibonacci relationships
            fib_validation = self._validate_fibonacci_ratios(
                o_price, x_price, a_price, b_price, c_price, bias
            )
            
            if fib_validation["valid"]:
                # Calculate targets and stops
                targets = self._calculate_targets_and_stops(
                    o_price, x_price, a_price, b_price, c_price, bias
                )
                
                # Check for volume spike at C
                volume_spike = self._check_volume_spike(volumes, c_idx)
                
                # Check for rejection candle at C
                rejection_candle = self._check_rejection_candle(
                    highs, lows, closes, c_idx, bias
                )
                
                # Calculate confidence
                confidence = self._calculate_confidence(
                    fib_validation, volume_spike, rejection_candle
                )
                
                # Update pattern result
                pattern_result.update({
                    "pattern_detected": True,
                    "bias": bias,
                    "o_price": o_price, "o_index": o_idx,
                    "o_timestamp": datetime.fromtimestamp(timestamps[o_idx] / 1000),
                    "x_price": x_price, "x_index": x_idx,
                    "x_timestamp": datetime.fromtimestamp(timestamps[x_idx] / 1000),
                    "a_price": a_price, "a_index": a_idx,
                    "a_timestamp": datetime.fromtimestamp(timestamps[a_idx] / 1000),
                    "b_price": b_price, "b_index": b_idx,
                    "b_timestamp": datetime.fromtimestamp(timestamps[b_idx] / 1000),
                    "c_price": c_price, "c_index": c_idx,
                    "c_timestamp": datetime.fromtimestamp(timestamps[c_idx] / 1000),
                    "xa_retracement": fib_validation["xa_ratio"],
                    "ab_extension": fib_validation["ab_ratio"],
                    "bc_extension": fib_validation["bc_ratio"],
                    "fibonacci_valid": True,
                    "volume_spike": volume_spike,
                    "rejection_candle": rejection_candle,
                    "target_price": targets["target"],
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
    
    def _validate_fibonacci_ratios(self, o_price: float, x_price: float, a_price: float,
                                  b_price: float, c_price: float, bias: str) -> Dict:
        """
        Validate Fibonacci relationships for Shark pattern.
        
        Args:
            o_price, x_price, a_price, b_price, c_price: The five key prices
            bias: "Bullish" or "Bearish"
        
        Returns:
            Dictionary with validation results and ratios
        """
        # Calculate distances
        ox_distance = abs(x_price - o_price)
        xa_distance = abs(a_price - x_price)
        ab_distance = abs(b_price - a_price)
        bc_distance = abs(c_price - b_price)
        
        # Calculate ratios
        xa_ratio = xa_distance / ox_distance if ox_distance > 0 else 0
        ab_ratio = ab_distance / xa_distance if xa_distance > 0 else 0
        bc_ratio = bc_distance / ox_distance if ox_distance > 0 else 0
        
        # Validate XA retracement (must be 0.886 ±2%)
        xa_valid = abs(xa_ratio - self.xa_retracement) <= self.xa_tolerance
        
        # Validate AB extension (1.13 to 1.618 of XA)
        ab_valid = self.ab_extension_min <= ab_ratio <= self.ab_extension_max
        
        # Validate BC extension (1.13 of OX ±5%)
        bc_valid = abs(bc_ratio - self.bc_extension) <= self.bc_tolerance
        
        return {
            "valid": xa_valid and ab_valid and bc_valid,
            "xa_ratio": xa_ratio,
            "ab_ratio": ab_ratio,
            "bc_ratio": bc_ratio,
            "xa_valid": xa_valid,
            "ab_valid": ab_valid,
            "bc_valid": bc_valid
        }
    
    def _calculate_targets_and_stops(self, o_price: float, x_price: float, a_price: float,
                                   b_price: float, c_price: float, bias: str) -> Dict:
        """Calculate target and stop loss levels."""
        if bias == "Bullish":
            # Target: 0.618 retracement of BC leg from C
            bc_distance = c_price - b_price
            target = c_price + (bc_distance * 0.618)
            stop_loss = c_price - (abs(c_price - b_price) * 0.1)  # 10% below C
        else:
            # Bearish pattern
            bc_distance = b_price - c_price
            target = c_price - (bc_distance * 0.618)
            stop_loss = c_price + (abs(b_price - c_price) * 0.1)  # 10% above C
        
        return {"target": target, "stop_loss": stop_loss}
    
    def _check_volume_spike(self, volumes: np.ndarray, c_idx: int) -> bool:
        """Check for volume spike at point C."""
        if c_idx < 10 or c_idx >= len(volumes):
            return False
        
        # Compare C volume with average of previous 10 candles
        avg_volume = np.mean(volumes[max(0, c_idx-10):c_idx])
        c_volume = volumes[c_idx]
        
        return c_volume > avg_volume * 1.5  # 50% above average
    
    def _check_rejection_candle(self, highs: np.ndarray, lows: np.ndarray, 
                               closes: np.ndarray, c_idx: int, bias: str) -> bool:
        """Check for rejection candle or engulfing pattern at point C."""
        if c_idx >= len(closes) - 1:
            return False
        
        # Check next candle after C for rejection
        next_idx = c_idx + 1
        c_close = closes[c_idx]
        next_open = closes[c_idx]  # Assume next open = previous close
        next_close = closes[next_idx]
        
        if bias == "Bullish":
            # Look for bullish engulfing or hammer-like rejection
            return next_close > c_close * 1.02  # 2% bullish move
        else:
            # Look for bearish engulfing or shooting star-like rejection
            return next_close < c_close * 0.98  # 2% bearish move
    
    def _calculate_confidence(self, fib_validation: Dict, volume_spike: bool, 
                            rejection_candle: bool) -> str:
        """Calculate confidence level based on validation criteria."""
        score = 0
        
        # Fibonacci validation (max 60 points)
        if fib_validation["xa_valid"]:
            score += 20
        if fib_validation["ab_valid"]:
            score += 20
        if fib_validation["bc_valid"]:
            score += 20
        
        # Volume confirmation (20 points)
        if volume_spike:
            score += 20
        
        # Rejection candle (20 points)
        if rejection_candle:
            score += 20
        
        # Determine confidence level
        if score >= 80:
            return "HIGH"
        elif score >= 60:
            return "MEDIUM"
        else:
            return "LOW"
    
    def format_analysis(self, symbol: str, timeframe: str, pattern_data: Dict) -> str:
        """
        Format Shark pattern analysis for terminal output.
        
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
            return f"{symbol} ({timeframe}) - No Shark Pattern Detected"
        
        # Format the Shark pattern output
        bias_emoji = "📈" if pattern_data["bias"] == "Bullish" else "📉"
        signal_emoji = "🚀" if pattern_data["signal"] == "BUY" else "🔻"
        
        xa_ratio = pattern_data["xa_retracement"]
        ab_ratio = pattern_data["ab_extension"]
        bc_ratio = pattern_data["bc_extension"]
        
        volume_check = "✅" if pattern_data["volume_spike"] else "❌"
        rejection_check = "✅" if pattern_data["rejection_candle"] else "❌"
        
        return (
            f"🦈 SHARK PATTERN DETECTED\n"
            f"Symbol: {symbol} | Timeframe: {timeframe.upper()}\n"
            f"Pattern Status: {pattern_data['pattern_status']} | Bias: {bias_emoji} {pattern_data['bias']}\n"
            f"• O: {pattern_data['o_price']:.3f}\n"
            f"• X: {pattern_data['x_price']:.3f}\n"
            f"• A: {pattern_data['a_price']:.3f} (XA retrace: {xa_ratio:.3f}) ✅\n"
            f"• B: {pattern_data['b_price']:.3f} (AB extension: {ab_ratio:.2f} of XA) ✅\n"
            f"• C: {pattern_data['c_price']:.3f} (BC extension: {bc_ratio:.2f} of OX) ← 📍 Entry Zone\n"
            f"🎯 Target: {pattern_data['target_price']:.3f} | Stop Loss: {pattern_data['stop_loss']:.3f}\n"
            f"Fibonacci Ratios ✅ Confirmed\n"
            f"Volume Spike {volume_check} at C\n"
            f"Rejection Candle {rejection_check}\n"
            f"🚦 Signal: {pattern_data['signal']} | Confidence: {pattern_data['confidence']}"
        )



