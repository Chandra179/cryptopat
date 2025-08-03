"""
Double Top Pattern Detection Module.

Detects double top reversal patterns using High, Close, and Volume data.
A double top is a bearish reversal pattern consisting of two swing highs
at approximately the same level, separated by an intervening valley.
"""

from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
from data import get_data_collector


class DoubleTopStrategy:
    """Double Top pattern detection strategy for cryptocurrency trend analysis."""
    
    def __init__(self, tolerance_pct: float = 3.0, min_valley_depth_pct: float = 5.0, min_time_separation: int = 10):
        self.tolerance_pct = tolerance_pct
        self.min_valley_depth_pct = min_valley_depth_pct
        self.min_time_separation = min_time_separation
        self.collector = get_data_collector()
    
    def analyze(self, symbol: str, timeframe: str, limit: int, ohlcv_data: Optional[List] = None) -> Dict:
        """
        Analyze Double Top patterns for given symbol and timeframe
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for analysis
            limit: Number of candles to analyze
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Fetch OHLCV data if not provided
            if ohlcv_data is None:
                ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
            
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
            
            # Find swing highs and lows
            swing_highs = self._find_swing_highs(highs, window=5)
            swing_lows = self._find_swing_lows(lows, window=5)
            
            # Detect double top pattern
            pattern_data = self._detect_double_top(
                timestamps, highs, lows, closes, volumes, 
                swing_highs, swing_lows
            )
            
            # Get current price and timestamp info
            current_price = closes[-1]
            current_timestamp = timestamps[-1]
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(current_timestamp / 1000, tz=timezone.utc)
            
            result = {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'analysis_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': int(current_timestamp),
                'total_candles': len(ohlcv_data),
                'current_price': round(current_price, 4),
                'pattern_detected': pattern_data["pattern_detected"]
            }
            
            if pattern_data["pattern_detected"]:
                # Pattern detected - extract metrics for wedge-style output
                confidence = pattern_data.get('confidence', 50)
                breakdown_confirmed = pattern_data.get('breakdown_confirmed', False)
                signal = pattern_data.get('signal', 'HOLD')
                
                # Map pattern data to wedge-style structure
                if breakdown_confirmed:
                    bias = 'BEARISH'
                elif signal == 'SELL':
                    bias = 'BEARISH'  
                else:
                    bias = 'NEUTRAL'
                
                # Calculate support/resistance levels
                neckline = pattern_data.get('neckline', current_price * 0.98)
                high1_price = pattern_data.get('high1_price', current_price * 1.02)
                high2_price = pattern_data.get('high2_price', current_price * 1.02)
                resistance_level = max(high1_price, high2_price)
                support_level = neckline
                
                # Calculate stop loss and take profit zones
                target_price = pattern_data.get('target_price', support_level * 0.9)
                stop_loss_price = pattern_data.get('stop_loss_price', resistance_level * 1.02)
                
                if signal == 'SELL':
                    stop_zone = stop_loss_price
                    tp_low = target_price
                    tp_high = pattern_data.get('target_75', target_price * 0.95)
                else:
                    stop_zone = support_level * 0.995
                    tp_low = resistance_level * 1.005
                    tp_high = resistance_level * 1.02
                
                # Calculate Risk/Reward ratio
                if signal == 'SELL':
                    risk = abs(current_price - stop_zone)
                    reward = abs(current_price - tp_low) if tp_low != current_price else abs(current_price - tp_high)
                    rr_ratio = reward / risk if risk > 0 else 0
                else:
                    rr_ratio = pattern_data.get('risk_reward_ratio', 0)
                
                # Determine entry window
                if breakdown_confirmed and confidence > 70:
                    entry_window = "Optimal now"
                elif signal == 'SELL' and confidence > 50:
                    entry_window = "Good in next 2-3 bars"
                else:
                    entry_window = "Wait for better setup"
                
                # Exit trigger
                if signal == 'SELL':
                    exit_trigger = f"Price breaks above ${stop_loss_price:.4f}"
                else:
                    exit_trigger = "Wait for neckline breakdown"
                
                # Update result with pattern analysis
                result.update({
                    # Pattern specific data
                    'pattern_type': 'Double Top',
                    'bias': bias,
                    'breakout': 'Downward' if breakdown_confirmed else None,
                    'expected_breakout': 'Downward',
                    
                    # Price levels
                    'support_level': round(support_level, 4),
                    'resistance_level': round(resistance_level, 4),
                    'stop_zone': round(stop_zone, 4),
                    'tp_low': round(tp_low, 4),
                    'tp_high': round(tp_high, 4),
                    
                    # Trading analysis
                    'signal': signal,
                    'confidence_score': confidence,
                    'entry_window': entry_window,
                    'exit_trigger': exit_trigger,
                    'rr_ratio': round(rr_ratio, 1),
                    
                    # Pattern details
                    'pattern_height': pattern_data.get('pattern_height'),
                    'neckline': round(neckline, 4),
                    'high1_price': round(high1_price, 4),
                    'high2_price': round(high2_price, 4),
                    'breakdown_confirmed': breakdown_confirmed,
                    'volume_confirmation': pattern_data.get('volume_confirmation', False),
                    
                    # Raw data
                    'raw_data': {
                        'ohlcv_data': ohlcv_data,
                        'pattern_data': pattern_data
                    }
                })
            else:
                # No pattern detected
                result.update({
                    'pattern_type': None,
                    'bias': 'NEUTRAL',
                    'signal': 'HOLD',
                    'confidence_score': 0,
                    'entry_window': "No pattern detected",
                    'exit_trigger': "Wait for pattern formation",
                    'support_level': round(current_price * 0.98, 4),
                    'resistance_level': round(current_price * 1.02, 4),
                    'rr_ratio': 0
                })
                
            return result
            
        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'success': False,
                'symbol': symbol,
                'timeframe': timeframe
            }
    
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
    


