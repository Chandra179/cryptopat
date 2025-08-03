"""
Double Bottom Pattern Detection Module.

Detects double bottom reversal patterns using Low, Close, and Volume data.
A double bottom is a bullish reversal pattern consisting of two swing lows
at approximately the same level, separated by an intervening peak.
"""

import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Optional
import numpy as np
from data import get_data_collector


class DoubleBottomStrategy:
    """Double Bottom pattern detection and analysis."""
    
    def __init__(self):
        """Initialize Double Bottom strategy detector."""
        self.tolerance_pct = 2.0
        self.min_peak_height_pct = 5.0
        self.min_time_between_lows = 10
        self.max_time_between_lows = 100
        self.collector = get_data_collector()
    
    def analyze(self, symbol: str, timeframe: str, limit: int, ohlcv_data: Optional[List] = None) -> Dict:
        """
        Analyze Double Bottom patterns for given symbol and timeframe.
        
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
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Extract price and volume arrays
            timestamps = [candle[0] for candle in ohlcv_data]
            highs = np.array([candle[2] for candle in ohlcv_data])
            lows = np.array([candle[3] for candle in ohlcv_data])
            closes = np.array([candle[4] for candle in ohlcv_data])
            volumes = np.array([candle[5] for candle in ohlcv_data])
            
            # Find swing lows and highs
            swing_lows = self._find_swing_lows(lows, window=5)
            swing_highs = self._find_swing_highs(highs, window=5)
            
            # Detect double bottom pattern
            pattern_data = self._detect_double_bottom(
                timestamps, lows, highs, closes, volumes, 
                swing_lows, swing_highs
            )
            
            # Get current price and timestamp info
            current_price = df['close'].iloc[-1]
            current_timestamp = df['timestamp'].iloc[-1]
            dt = datetime.fromtimestamp(current_timestamp.timestamp(), tz=timezone.utc)
            
            # Build result similar to wedge.py structure
            result = {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'analysis_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': int(current_timestamp.timestamp() * 1000),
                'total_candles': len(df),
                'current_price': round(current_price, 4),
                'pattern_detected': pattern_data.get('pattern_detected', False)
            }
            
            if pattern_data.get('pattern_detected', False):
                # Pattern detected - extract trading data
                signal = pattern_data.get('signal', 'HOLD')
                confidence = pattern_data.get('confidence_score', 0)
                neckline = pattern_data.get('neckline', current_price)
                target = pattern_data.get('price_target', current_price)
                stop_loss = pattern_data.get('stop_loss', current_price * 0.98)
                
                # Calculate support/resistance levels
                support_level = min(pattern_data.get('low1_price', current_price), pattern_data.get('low2_price', current_price))
                resistance_level = neckline
                
                # Calculate stop loss and take profit zones
                if signal == 'BUY':
                    stop_zone = stop_loss
                    tp_low = target
                    tp_high = target * 1.02
                elif signal == 'WATCH':
                    stop_zone = support_level * 0.995
                    tp_low = resistance_level * 1.005
                    tp_high = target
                else:
                    stop_zone = support_level * 0.995
                    tp_low = resistance_level * 1.005
                    tp_high = target
                
                # Calculate Risk/Reward ratio
                risk = abs(current_price - stop_zone)
                reward = abs(tp_low - current_price) if tp_low != current_price else abs(tp_high - current_price)
                rr_ratio = reward / risk if risk > 0 else 0
                
                # Determine entry window
                if signal == 'BUY' and confidence > 70:
                    entry_window = "Optimal now"
                elif signal in ['BUY', 'WATCH'] and confidence > 50:
                    entry_window = "Good in next 2-3 bars"
                else:
                    entry_window = "Wait for better setup"
                
                # Exit trigger
                if signal == 'BUY':
                    exit_trigger = "Close below support or target reached"
                elif signal == 'WATCH':
                    exit_trigger = "Breakout above neckline or breakdown below support"
                else:
                    exit_trigger = "Wait for breakout signal"
                
                # Update result with pattern analysis
                result.update({
                    'pattern_type': 'Double Bottom',
                    'bias': 'BULLISH',
                    'signal': signal,
                    'confidence_score': confidence,
                    'entry_window': entry_window,
                    'exit_trigger': exit_trigger,
                    'support_level': round(support_level, 4),
                    'resistance_level': round(resistance_level, 4),
                    'stop_zone': round(stop_zone, 4),
                    'tp_low': round(tp_low, 4),
                    'tp_high': round(tp_high, 4),
                    'rr_ratio': round(rr_ratio, 1),
                    'pattern_height': pattern_data.get('pattern_height'),
                    'low1_price': pattern_data.get('low1_price'),
                    'low2_price': pattern_data.get('low2_price'),
                    'neckline': neckline,
                    'price_target': target,
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
    
    def _detect_double_bottom(self, timestamps: List, lows: np.ndarray, 
                            highs: np.ndarray, closes: np.ndarray, 
                            volumes: np.ndarray, swing_lows: List[int], 
                            swing_highs: List[int]) -> Dict:
        """
        Detect double bottom pattern from swing points.
        
        Returns:
            Dictionary with pattern details and signals
        """
        pattern_result = {
            "pattern_detected": False,
            "low1_price": None,
            "low1_index": None,
            "low1_timestamp": None,
            "peak_price": None,
            "peak_index": None,
            "peak_timestamp": None,
            "low2_price": None,
            "low2_index": None,
            "low2_timestamp": None,
            "neckline": None,
            "price_target": None,
            "pattern_height": None,
            "current_price": closes[-1],
            "signal": "NONE",
            "pattern_status": "No Pattern",
            "volume_confirmation": False,
            "breakout_confirmed": False,
            "confidence_score": 0,
            "time_symmetry": 0,
            "volume_ratio": 0,
            "relative_volume_strength": 0,
            "stop_loss": None,
            "entry_price": None,
            "risk_reward_ratio": 0,
            "trend_context": {"trend": "UNKNOWN", "strength": 0, "change_pct": 0}
        }
        
        if len(swing_lows) < 2:
            return pattern_result
        
        # Look for potential double bottom patterns
        for i in range(len(swing_lows) - 1):
            low1_idx = swing_lows[i]
            low1_price = lows[low1_idx]
            
            # Find intervening peak between low1 and potential low2
            potential_peaks = [h for h in swing_highs if h > low1_idx]
            if not potential_peaks:
                continue
            
            # Look for second low after the peak
            for j in range(i + 1, len(swing_lows)):
                low2_idx = swing_lows[j]
                low2_price = lows[low2_idx]
                
                # Find the peak between low1 and low2
                intervening_peaks = [h for h in swing_highs if low1_idx < h < low2_idx]
                if not intervening_peaks:
                    continue
                
                peak_idx = max(intervening_peaks, key=lambda x: highs[x])
                peak_price = highs[peak_idx]
                
                # Check time relationship between lows
                time_between_lows = low2_idx - low1_idx
                if not (self.min_time_between_lows <= time_between_lows <= self.max_time_between_lows):
                    continue
                
                # Check if the two lows are within tolerance
                price_diff_pct = abs(low1_price - low2_price) / min(low1_price, low2_price) * 100
                
                if price_diff_pct <= self.tolerance_pct:
                    # Check if peak is high enough above the lows
                    min_low = min(low1_price, low2_price)
                    pattern_height = peak_price - min_low
                    peak_height_pct = pattern_height / min_low * 100
                    
                    if peak_height_pct >= self.min_peak_height_pct:
                        # Calculate improved neckline using resistance points near peak
                        neckline = self._calculate_neckline(highs, peak_idx, peak_price)
                        
                        # Calculate price target (pattern height projected from neckline)
                        price_target = neckline + pattern_height
                        
                        # Valid double bottom pattern found
                        pattern_result.update({
                            "pattern_detected": True,
                            "low1_price": low1_price,
                            "low1_index": low1_idx,
                            "low1_timestamp": datetime.fromtimestamp(timestamps[low1_idx] / 1000),
                            "peak_price": peak_price,
                            "peak_index": peak_idx,
                            "peak_timestamp": datetime.fromtimestamp(timestamps[peak_idx] / 1000),
                            "low2_price": low2_price,
                            "low2_index": low2_idx,
                            "low2_timestamp": datetime.fromtimestamp(timestamps[low2_idx] / 1000),
                            "neckline": neckline,
                            "price_target": price_target,
                            "pattern_height": pattern_height
                        })
                        
                        # Enhanced volume analysis
                        vol1 = volumes[low1_idx]
                        vol2 = volumes[low2_idx]
                        volume_ratio = vol2 / vol1 if vol1 > 0 else 1.0
                        pattern_result["volume_confirmation"] = vol2 <= vol1
                        pattern_result["volume_ratio"] = volume_ratio
                        
                        # Calculate relative volume strength
                        avg_volume = np.mean(volumes[max(0, low1_idx-20):low2_idx+10])
                        relative_vol_strength = (vol1 + vol2) / (2 * avg_volume) if avg_volume > 0 else 1.0
                        pattern_result["relative_volume_strength"] = relative_vol_strength
                        
                        # Calculate time symmetry (how balanced the pattern timing is)
                        time_to_peak = peak_idx - low1_idx
                        time_from_peak = low2_idx - peak_idx
                        time_symmetry = 1 - abs(time_to_peak - time_from_peak) / max(time_to_peak, time_from_peak, 1)
                        pattern_result["time_symmetry"] = time_symmetry
                        
                        # Calculate comprehensive confidence score
                        confidence_score = self._calculate_confidence_score(
                            price_diff_pct, peak_height_pct, volume_ratio, 
                            time_symmetry, relative_vol_strength
                        )
                        pattern_result["confidence_score"] = confidence_score
                        
                        # Calculate risk management levels
                        stop_loss = min(low1_price, low2_price) * 0.98  # 2% below pattern lows
                        entry_price = neckline * 1.001  # Slightly above neckline for confirmation
                        risk_distance = entry_price - stop_loss
                        reward_distance = price_target - entry_price
                        risk_reward_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
                        
                        # Calculate trend context using recent price action
                        trend_context = self._analyze_trend_context(closes, low1_idx)
                        
                        # Update pattern result with enhanced metrics
                        pattern_result.update({
                            "stop_loss": stop_loss,
                            "entry_price": entry_price,
                            "risk_reward_ratio": risk_reward_ratio,
                            "trend_context": trend_context
                        })
                        
                        # Check current price relative to neckline
                        current_price = closes[-1]
                        
                        if current_price > neckline:
                            # Breakout confirmed
                            pattern_result.update({
                                "signal": "BUY",
                                "pattern_status": "Breakout Confirmed",
                                "breakout_confirmed": True
                            })
                        elif low2_idx >= len(closes) - 10:  # Recent second low
                            pattern_result.update({
                                "signal": "WATCH",
                                "pattern_status": "Pattern Forming"
                            })
                        else:
                            pattern_result.update({
                                "signal": "WATCH",
                                "pattern_status": "Pattern Complete - Awaiting Breakout"
                            })
                        
                        return pattern_result
        
        return pattern_result
    
    def _calculate_neckline(self, highs: np.ndarray, peak_idx: int, peak_price: float) -> float:
        """
        Calculate neckline using multiple resistance points around the peak.
        
        Args:
            highs: Array of high prices
            peak_idx: Index of the main peak
            peak_price: Price of the main peak
        
        Returns:
            Calculated neckline price
        """
        # Look for resistance points within 5% of peak price in surrounding area
        search_range = 10
        start_idx = max(0, peak_idx - search_range)
        end_idx = min(len(highs), peak_idx + search_range + 1)
        
        resistance_prices = []
        tolerance = 0.05  # 5% tolerance for resistance level
        
        for i in range(start_idx, end_idx):
            if abs(highs[i] - peak_price) / peak_price <= tolerance:
                resistance_prices.append(highs[i])
        
        # Return average of resistance points, fallback to peak price
        return np.mean(resistance_prices) if resistance_prices else peak_price
    
    def _calculate_confidence_score(self, price_diff_pct: float, peak_height_pct: float, 
                                  volume_ratio: float, time_symmetry: float, 
                                  relative_vol_strength: float) -> float:
        """
        Calculate comprehensive confidence score for the double bottom pattern.
        
        Args:
            price_diff_pct: Percentage difference between the two lows
            peak_height_pct: Height of peak above lows in percentage
            volume_ratio: Ratio of second low volume to first low volume
            time_symmetry: Symmetry of timing in the pattern (0-1)
            relative_vol_strength: Volume strength relative to average
        
        Returns:
            Confidence score between 0-100
        """
        confidence = 0
        
        # Price level similarity (0-30 points)
        if price_diff_pct <= 0.5:
            confidence += 30
        elif price_diff_pct <= 1.0:
            confidence += 25
        elif price_diff_pct <= 1.5:
            confidence += 20
        elif price_diff_pct <= 2.0:
            confidence += 15
        else:
            confidence += 10
        
        # Peak height significance (0-25 points)
        if peak_height_pct >= 10:
            confidence += 25
        elif peak_height_pct >= 7:
            confidence += 20
        elif peak_height_pct >= 5:
            confidence += 15
        else:
            confidence += 10
        
        # Volume confirmation (0-20 points)
        if volume_ratio <= 0.7:  # Strong volume confirmation
            confidence += 20
        elif volume_ratio <= 0.9:
            confidence += 15
        elif volume_ratio <= 1.1:
            confidence += 10
        else:
            confidence += 5
        
        # Time symmetry (0-15 points)
        confidence += int(time_symmetry * 15)
        
        # Relative volume strength (0-10 points)
        if relative_vol_strength >= 1.5:  # High volume relative to average
            confidence += 10
        elif relative_vol_strength >= 1.2:
            confidence += 7
        elif relative_vol_strength >= 1.0:
            confidence += 5
        else:
            confidence += 2
        
        return min(confidence, 100)  # Cap at 100%
    
    def _analyze_trend_context(self, closes: np.ndarray, pattern_start_idx: int) -> Dict:
        """
        Analyze trend context before the pattern formation.
        
        Args:
            closes: Array of closing prices
            pattern_start_idx: Index where pattern starts
        
        Returns:
            Dictionary with trend analysis
        """
        if pattern_start_idx < 20:
            return {"trend": "INSUFFICIENT_DATA", "strength": 0}
        
        # Analyze trend before pattern (20 periods before first low)
        pre_pattern_closes = closes[max(0, pattern_start_idx-20):pattern_start_idx]
        
        if len(pre_pattern_closes) < 10:
            return {"trend": "INSUFFICIENT_DATA", "strength": 0}
        
        # Calculate trend direction and strength
        start_price = pre_pattern_closes[0]
        end_price = pre_pattern_closes[-1]
        trend_change_pct = ((end_price - start_price) / start_price) * 100
        
        # Calculate trend strength using linear regression slope
        x = np.arange(len(pre_pattern_closes))
        slope, _ = np.polyfit(x, pre_pattern_closes, 1)
        trend_strength = abs(slope / np.mean(pre_pattern_closes)) * 100
        
        if trend_change_pct < -5:
            trend_direction = "DOWNTREND"
        elif trend_change_pct > 5:
            trend_direction = "UPTREND"
        else:
            trend_direction = "SIDEWAYS"
        
        return {
            "trend": trend_direction,
            "strength": trend_strength,
            "change_pct": trend_change_pct
        }
    

