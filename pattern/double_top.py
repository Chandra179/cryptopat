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
    
    def __init__(self, tolerance_pct: float = 3.0, min_valley_depth_pct: float = 5.0, min_time_separation: int = 10):
        """
        Initialize Double Top detector.
        
        Args:
            tolerance_pct: Tolerance for considering two highs as equal (in %)
            min_valley_depth_pct: Minimum depth of intervening valley below highs (in %)
            min_time_separation: Minimum candles between peaks for valid pattern
        """
        self.tolerance_pct = tolerance_pct
        self.min_valley_depth_pct = min_valley_depth_pct
        self.min_time_separation = min_time_separation
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
                    # Check chronological order (high2 must come after high1)
                    if high2_idx <= high1_idx:
                        continue
                    
                    # Check minimum time separation between peaks
                    if high2_idx - high1_idx < self.min_time_separation:
                        continue
                    
                    # Check if valley is deep enough below the highs
                    max_high = max(high1_price, high2_price)
                    valley_depth_pct = (max_high - valley_price) / max_high * 100
                    
                    if valley_depth_pct >= self.min_valley_depth_pct:
                        # Check for uptrend context (prior trend should be up)
                        trend_valid = self._validate_uptrend_context(closes, high1_idx)
                        
                        if not trend_valid:
                            continue
                        # Calculate price target using standard formula
                        pattern_height = max_high - valley_price
                        target_price = valley_price - pattern_height
                        
                        # Get volume data for confidence calculation
                        vol1 = volumes[high1_idx]
                        vol2 = volumes[high2_idx]
                        
                        # Calculate confidence score
                        confidence = self._calculate_confidence(
                            price_diff_pct, valley_depth_pct, vol1, vol2, 
                            high2_idx - high1_idx, len(closes)
                        )
                        
                        # Calculate additional metrics for enhanced output
                        stop_loss_price = max(high1_price, high2_price) * 1.02  # 2% above highest peak
                        risk_amount = stop_loss_price - valley_price
                        reward_amount = valley_price - target_price
                        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
                        
                        # Calculate pattern age and timing
                        current_time = datetime.now()
                        high1_time = datetime.fromtimestamp(timestamps[high1_idx] / 1000)
                        high2_time = datetime.fromtimestamp(timestamps[high2_idx] / 1000)
                        valley_time = datetime.fromtimestamp(timestamps[valley_idx] / 1000)
                        
                        pattern_age_days = (current_time - high2_time).days
                        time_between_peaks_days = (high2_time - high1_time).days
                        
                        # Calculate staged profit targets
                        target_25 = valley_price - (pattern_height * 0.25)
                        target_50 = valley_price - (pattern_height * 0.50)
                        target_75 = valley_price - (pattern_height * 0.75)
                        
                        # Valid double top pattern found
                        pattern_result.update({
                            "pattern_detected": True,
                            "high1_price": high1_price,
                            "high1_index": high1_idx,
                            "high1_timestamp": high1_time,
                            "valley_price": valley_price,
                            "valley_index": valley_idx,
                            "valley_timestamp": valley_time,
                            "high2_price": high2_price,
                            "high2_index": high2_idx,
                            "high2_timestamp": high2_time,
                            "neckline": valley_price,
                            "target_price": target_price,
                            "target_25": target_25,
                            "target_50": target_50,
                            "target_75": target_75,
                            "pattern_height": pattern_height,
                            "confidence": confidence,
                            "stop_loss_price": stop_loss_price,
                            "risk_reward_ratio": risk_reward_ratio,
                            "pattern_age_days": pattern_age_days,
                            "time_between_peaks_days": time_between_peaks_days,
                            "price_diff_pct": price_diff_pct,
                            "valley_depth_pct": valley_depth_pct,
                            "vol1": vol1,
                            "vol2": vol2,
                            "volume_change_pct": ((vol2 - vol1) / vol1 * 100) if vol1 > 0 else 0
                        })
                        
                        # Check volume confirmation (lower volume on second high)
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
    
    def _validate_uptrend_context(self, closes: np.ndarray, high1_idx: int) -> bool:
        """
        Validate that the pattern occurs in an uptrend context.
        
        Args:
            closes: Array of closing prices
            high1_idx: Index of first high
        
        Returns:
            True if prior trend is upward
        """
        # Look at trend before first high (minimum 20 candles)
        lookback = min(20, high1_idx)
        if lookback < 10:
            return True  # Not enough data, assume valid
        
        start_idx = high1_idx - lookback
        trend_start = closes[start_idx]
        trend_end = closes[high1_idx]
        
        # Calculate trend slope - should be positive for uptrend
        trend_change_pct = (trend_end - trend_start) / trend_start * 100
        return trend_change_pct > 5.0  # At least 5% upward movement
    
    def _calculate_confidence(self, price_diff_pct: float, valley_depth_pct: float, 
                            vol1: float, vol2: float, time_separation: int, 
                            total_candles: int) -> int:
        """
        Calculate confidence score for the double top pattern.
        
        Args:
            price_diff_pct: Percentage difference between peaks
            valley_depth_pct: Depth of valley as percentage
            vol1: Volume at first peak
            vol2: Volume at second peak
            time_separation: Candles between peaks
            total_candles: Total candles in dataset
        
        Returns:
            Confidence score (0-100)
        """
        confidence = 50  # Base confidence
        
        # Peak similarity (closer peaks = higher confidence)
        if price_diff_pct <= 1.0:
            confidence += 20
        elif price_diff_pct <= 2.0:
            confidence += 10
        
        # Valley depth (deeper valley = higher confidence)
        if valley_depth_pct >= 10.0:
            confidence += 15
        elif valley_depth_pct >= 7.0:
            confidence += 10
        elif valley_depth_pct >= 5.0:
            confidence += 5
        
        # Volume confirmation (decreasing volume = higher confidence)
        volume_ratio = vol2 / vol1 if vol1 > 0 else 1.0
        if volume_ratio <= 0.7:
            confidence += 15
        elif volume_ratio <= 0.9:
            confidence += 10
        elif volume_ratio <= 1.0:
            confidence += 5
        
        # Time separation (optimal range gives higher confidence)
        optimal_separation = total_candles * 0.2  # 20% of total timeframe
        separation_ratio = time_separation / optimal_separation if optimal_separation > 0 else 1.0
        if 0.5 <= separation_ratio <= 2.0:
            confidence += 10
        elif 0.3 <= separation_ratio <= 3.0:
            confidence += 5
        
        return min(100, max(0, confidence))
    
    def format_analysis(self, symbol: str, timeframe: str, pattern_data: Dict) -> str:
        """
        Format pattern analysis for terminal output using Phase 3 format.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe used
            pattern_data: Pattern detection results
        
        Returns:
            Formatted analysis string
        """
        if "error" in pattern_data:
            return f"""
===============================================================
DOUBLE TOP PATTERN ANALYSIS
===============================================================
ERROR: {pattern_data['error']}

SUMMARY: Insufficient data for pattern analysis
CONFIDENCE_SCORE: 0% | Based on data availability
TREND_DIRECTION: Unknown | MOMENTUM_STATE: Unknown
ENTRY_WINDOW: N/A
EXIT_TRIGGER: N/A

SUPPORT: N/A | RESISTANCE: N/A
STOP_ZONE: N/A | TP_ZONE: N/A
RR_RATIO: N/A | MAX_DRAWDOWN: N/A

ACTION: INSUFFICIENT_DATA"""
        
        if not pattern_data["pattern_detected"]:
            current_price = pattern_data.get('current_price', 0)
            return f"""
===============================================================
DOUBLE TOP PATTERN ANALYSIS
===============================================================
CURRENT_PRICE: ${current_price:.4f} | PATTERN_STATUS: NO_PATTERN | TIMEFRAME: {timeframe}

SUMMARY: No double top pattern detected in current data
CONFIDENCE_SCORE: 0% | Based on swing point analysis
TREND_DIRECTION: Neutral | MOMENTUM_STATE: Analyzing
ENTRY_WINDOW: Pattern not formed
EXIT_TRIGGER: N/A

SUPPORT: N/A | RESISTANCE: N/A
STOP_ZONE: N/A | TP_ZONE: N/A
RR_RATIO: N/A | MAX_DRAWDOWN: N/A

ACTION: NEUTRAL"""
        
        # Pattern detected - extract key metrics
        confidence = pattern_data.get('confidence', 0)
        risk_reward = pattern_data.get('risk_reward_ratio', 0)
        current_price = pattern_data.get('current_price', 0)
        valley_depth = pattern_data.get('valley_depth_pct', 0)
        volume_change = pattern_data.get('volume_change_pct', 0)
        pattern_age = pattern_data.get('pattern_age_days', 0)
        high1_time = pattern_data.get('high1_timestamp')
        high2_time = pattern_data.get('high2_timestamp')
        neckline = pattern_data.get('neckline', 0)
        target_price = pattern_data.get('target_price', 0)
        stop_loss_price = pattern_data.get('stop_loss_price', 0)
        breakdown_confirmed = pattern_data.get('breakdown_confirmed', False)
        volume_confirmation = pattern_data.get('volume_confirmation', False)
        pattern_height = pattern_data.get('pattern_height', 0)
        
        # Format timestamps
        high1_str = high1_time.strftime('%Y-%m-%d %H:%M:%S') if high1_time else "UNKNOWN"
        high2_str = high2_time.strftime('%Y-%m-%d %H:%M:%S') if high2_time else "UNKNOWN"
        
        # Volume confirmation status
        vol_conf_status = "✓" if volume_confirmation else "✗"
        
        # Determine action and momentum
        if breakdown_confirmed:
            action = "SELL"
            momentum = "Declining"
            trend_direction = "Bearish"
            entry_window = "Active breakdown"
        elif pattern_data.get('pattern_status') == "Pattern Forming":
            action = "WAITING FOR PATTERN"
            momentum = "Consolidating"
            trend_direction = "Neutral"
            entry_window = "Pattern incomplete"
        else:
            action = "WAITING FOR BREAKOUT"
            momentum = "Stalling"
            trend_direction = "Bearish Pending"
            entry_window = "Awaiting neckline break"
        
        # Calculate expected drawdown
        max_drawdown = (pattern_height / current_price * 100) if current_price > 0 else 0
        
        # Generate summary
        summary_parts = []
        if volume_confirmation:
            summary_parts.append("Volume confirmation present")
        if breakdown_confirmed:
            summary_parts.append("Neckline breakdown confirmed")
        else:
            summary_parts.append("Pattern complete, awaiting breakdown")
        if valley_depth >= 10:
            summary_parts.append("Strong valley depth")
        
        summary = " + ".join(summary_parts) if summary_parts else "Double top pattern identified"
        
        output = f"""
===============================================================
DOUBLE TOP PATTERN ANALYSIS
===============================================================
RISK_REWARD: {risk_reward:.1f}:1 | VOLUME_CONF: {vol_conf_status} ({confidence/100:.2f}) | VALLEY_DEPTH: {valley_depth:.1f}%
BREAKDOWN: {"CONFIRMED" if breakdown_confirmed else "PENDING"} | CURRENT_PRICE: ${current_price:.4f} | VOLUME_CHANGE: {volume_change:+.1f}%
PEAK1_TIME: {high1_str} | PEAK2_TIME: {high2_str}

SUMMARY: {summary}
CONFIDENCE_SCORE: {confidence:.0f}% | Based on pattern + volume + volatility match
TREND_DIRECTION: {trend_direction} | MOMENTUM_STATE: {momentum}
ENTRY_WINDOW: {entry_window}
EXIT_TRIGGER: Break above ${stop_loss_price:.4f} OR volume surge above peaks

SUPPORT: ${neckline:.4f} | RESISTANCE: ${max(pattern_data.get('high1_price', 0), pattern_data.get('high2_price', 0)):.4f}
STOP_ZONE: Above ${stop_loss_price:.4f} | TP_ZONE: ${target_price:.4f}–${pattern_data.get('target_75', target_price):.4f}
RR_RATIO: {risk_reward:.1f}:1 | MAX_DRAWDOWN: -{max_drawdown:.1f}% expected

ACTION: {action}"""
        
        return output


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



