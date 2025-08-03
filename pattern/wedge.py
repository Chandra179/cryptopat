"""
Wedge Pattern Detection

Detects Wedge patterns which are reversal patterns characterized by converging trendlines that slope in the same direction.

Wedge Types:
- Rising Wedge: upward sloping resistance and support lines (bearish reversal)
- Falling Wedge: downward sloping resistance and support lines (bullish reversal)

Pattern Components:
- Upper trendline (resistance) - connects swing highs
- Lower trendline (support) - connects swing lows  
- Both lines slope in same direction and converge
- Volume typically decreases during formation
- Breakout occurs against the wedge slope direction

Signal: Reversal when price breaks out against wedge direction
Best timeframes: 4h, 1d
"""

import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from data import get_data_collector


class WedgeStrategy:
    """Wedge pattern detection strategy for cryptocurrency trend analysis."""
    
    def __init__(self):
        self.collector = get_data_collector()


    def calculate_line_slope(self, points: List[Tuple[int, float]]) -> Optional[float]:
        """
        Calculate slope of line through given points
        
        Args:
            points: List of (index, price) tuples
            
        Returns:
            Slope value or None if insufficient points
        """
        if len(points) < 2:
            return None
            
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        # Linear regression
        n = len(points)
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
        sum_x2 = sum(x * x for x in x_vals)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return None
            
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope


    def find_wedge_points(self, highs: pd.Series, lows: pd.Series, window: int = 4) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        Find swing points for wedge pattern construction
        
        Args:
            highs: High prices series
            lows: Low prices series
            window: Lookback window for swing detection
            
        Returns:
            Tuple of (swing_highs, swing_lows)
        """
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(highs) - window):
            # Check for swing high
            is_high = all(highs.iloc[i] >= highs.iloc[j] for j in range(i - window, i + window + 1) if j != i)
            if is_high:
                swing_highs.append((i, highs.iloc[i]))
                
            # Check for swing low
            is_low = all(lows.iloc[i] <= lows.iloc[j] for j in range(i - window, i + window + 1) if j != i)
            if is_low:
                swing_lows.append((i, lows.iloc[i]))
        
        return swing_highs, swing_lows


    def detect_wedge_pattern(self, df: pd.DataFrame, min_touches: int = 3, check_volume: bool = True) -> Optional[Dict]:
        """
        Detect Wedge patterns in OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
            min_touches: Minimum touches required for valid trendline
            check_volume: Whether to validate decreasing volume pattern
            
        Returns:
            Dictionary with pattern details or None if not found
        """
        if len(df) < 50:
            return None
            
        swing_highs, swing_lows = self.find_wedge_points(df['high'], df['low'])
        
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return None
        
        # Look for wedge patterns in recent data
        for i in range(len(swing_highs) - 2):
            for j in range(len(swing_lows) - 2):
                # Test with different combinations of swing points
                resistance_points = swing_highs[i:i+3]
                support_points = swing_lows[j:j+3]
                
                # Both trendlines need at least 3 points
                if len(resistance_points) < 3 or len(support_points) < 3:
                    continue
                    
                # Calculate slopes
                resistance_slope = self.calculate_line_slope(resistance_points)
                support_slope = self.calculate_line_slope(support_points)
                
                if resistance_slope is None or support_slope is None:
                    continue
                
                # For wedge pattern, both lines must slope in same direction
                # and they must be converging (different slope magnitudes)
                same_direction = (resistance_slope > 0 and support_slope > 0) or (resistance_slope < 0 and support_slope < 0)
                
                if not same_direction:
                    continue
                    
                # Check convergence - slopes should be different enough
                slope_diff = abs(resistance_slope - support_slope)
                if slope_diff < 0.0001:  # Lines too parallel
                    continue
                
                # For proper wedge, lines must converge
                # Rising wedge: support steeper than resistance (both positive slopes)
                # Falling wedge: support less steep than resistance (both negative slopes)
                if resistance_slope > 0 and support_slope > 0:  # Rising wedge
                    if support_slope <= resistance_slope:  # Support not steep enough for convergence
                        continue
                    wedge_type = "Rising Wedge"
                    bias = "BEARISH"
                    expected_breakout = "Downward"
                elif resistance_slope < 0 and support_slope < 0:  # Falling wedge
                    if support_slope >= resistance_slope:  # Support too steep for convergence
                        continue
                    wedge_type = "Falling Wedge"
                    bias = "BULLISH" 
                    expected_breakout = "Upward"
                else:
                    continue  # Mixed directions not a valid wedge
                
                # Calculate current trendline levels
                current_idx = len(df) - 1
                
                # Linear equation: y = slope * x + intercept
                # Find intercept using last resistance point
                last_resistance_point = resistance_points[-1]
                resistance_intercept = last_resistance_point[1] - resistance_slope * last_resistance_point[0]
                current_resistance = resistance_slope * current_idx + resistance_intercept
                
                # Find intercept using last support point
                last_support_point = support_points[-1]
                support_intercept = last_support_point[1] - support_slope * last_support_point[0]
                current_support = support_slope * current_idx + support_intercept
                
                # Check current price and breakout status
                current_price = df['close'].iloc[-1]
                
                breakout = None
                signal = "PENDING"
                
                if current_price > current_resistance:
                    breakout = "Upward"
                    signal = "BUY" if expected_breakout == "Upward" else "SELL"
                elif current_price < current_support:
                    breakout = "Downward"
                    signal = "SELL" if expected_breakout == "Downward" else "BUY"
                
                # Find convergence point
                if slope_diff != 0:
                    convergence_x = (support_intercept - resistance_intercept) / (resistance_slope - support_slope)
                    convergence_y = resistance_slope * convergence_x + resistance_intercept
                else:
                    convergence_x = float('inf')
                    convergence_y = 0
                
                # Calculate pattern quality metrics
                pattern_height = abs(current_resistance - current_support)
                pattern_width = current_idx - min([p[0] for p in resistance_points + support_points])
                
                # Calculate confidence
                confidence = 50  # Base confidence
                
                # Add confidence for clear convergence
                if convergence_x > current_idx and convergence_x < current_idx + 50:
                    confidence += 20  # Good convergence timing
                
                # Add confidence for proper slope relationship
                if wedge_type == "Rising Wedge" and resistance_slope > 0 and support_slope > 0:
                    confidence += 15
                elif wedge_type == "Falling Wedge" and resistance_slope < 0 and support_slope < 0:
                    confidence += 15
                
                # Volume analysis for additional confidence
                if check_volume and len(df) >= 20:
                    recent_volume = df['volume'].tail(10).mean()
                    earlier_volume = df['volume'].tail(20).head(10).mean()
                    if recent_volume < earlier_volume:  # Decreasing volume pattern
                        confidence += 10
                
                # Add confidence for breakout
                if breakout == expected_breakout:
                    confidence += 20
                elif breakout and breakout != expected_breakout:
                    confidence -= 15
                    
                # Pattern is more reliable if it's not too narrow
                if pattern_height > df['close'].iloc[-1] * 0.02:  # At least 2% height
                    confidence += 10
                
                # Better convergence angle gives higher confidence
                convergence_angle = abs(resistance_slope - support_slope)
                if 0.0005 < convergence_angle < 0.01:  # Sweet spot for convergence
                    confidence += 5
                    
                confidence = max(20, min(95, confidence))
                
                return {
                    'pattern': wedge_type,
                    'bias': bias,
                    'resistance_line': {
                        'slope': round(resistance_slope, 6),
                        'intercept': round(resistance_intercept, 4),
                        'current_level': round(current_resistance, 4),
                        'points': [(p[0], round(p[1], 4)) for p in resistance_points]
                    },
                    'support_line': {
                        'slope': round(support_slope, 6),
                        'intercept': round(support_intercept, 4), 
                        'current_level': round(current_support, 4),
                        'points': [(p[0], round(p[1], 4)) for p in support_points]
                    },
                    'convergence_point': {
                        'x': round(convergence_x, 1) if convergence_x != float('inf') else None,
                        'y': round(convergence_y, 4) if convergence_x != float('inf') else None
                    },
                    'current_price': round(current_price, 4),
                    'pattern_height': round(pattern_height, 4),
                    'pattern_width': pattern_width,
                    'breakout': breakout,
                    'expected_breakout': expected_breakout,
                    'signal': signal,
                    'confidence': confidence
                }
        
        return None


    def analyze(self, symbol: str, timeframe: str, limit: int, ohlcv_data: Optional[List] = None) -> Dict:
        """
        Analyze Wedge patterns for given symbol and timeframe
        
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
            
            # Detect pattern
            pattern = self.detect_wedge_pattern(df)
            
            # Get current price and timestamp info
            current_price = df['close'].iloc[-1]
            current_timestamp = df['timestamp'].iloc[-1]
            dt = datetime.fromtimestamp(current_timestamp.timestamp(), tz=timezone.utc)
            
            result = {
                'analysis_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': int(current_timestamp.timestamp() * 1000),
                'total_candles': len(df),
                'current_price': round(current_price, 4),
                'pattern_detected': pattern is not None
            }
            
            if pattern:
                # Calculate additional metrics based on pattern
                confidence = pattern.get('confidence', 50)
                signal = pattern.get('signal', 'HOLD')
                bias = pattern.get('bias', 'NEUTRAL')
                
                # Calculate support/resistance levels
                support_level = pattern.get('support_line', {}).get('current_level', current_price * 0.98)
                resistance_level = pattern.get('resistance_line', {}).get('current_level', current_price * 1.02)
                
                # Calculate stop loss and take profit zones
                if signal == 'BUY':
                    stop_zone = support_level * 0.995  # Below support
                    tp_low = resistance_level * 1.005  # Above resistance
                    tp_high = resistance_level * 1.02   # Extended target
                elif signal == 'SELL':
                    stop_zone = resistance_level * 1.005  # Above resistance
                    tp_low = support_level * 0.995       # Below support
                    tp_high = support_level * 0.98       # Extended target
                else:
                    stop_zone = support_level * 0.995
                    tp_low = resistance_level * 1.005
                    tp_high = resistance_level * 1.02
                
                # Calculate Risk/Reward ratio
                if signal in ['BUY', 'SELL']:
                    risk = abs(current_price - stop_zone)
                    reward = abs(tp_low - current_price) if tp_low != current_price else abs(tp_high - current_price)
                    rr_ratio = reward / risk if risk > 0 else 0
                else:
                    rr_ratio = 0
                
                # Determine entry window
                if signal in ['BUY', 'SELL'] and confidence > 70:
                    entry_window = "Optimal now"
                elif signal in ['BUY', 'SELL'] and confidence > 50:
                    entry_window = "Good in next 2-3 bars"
                else:
                    entry_window = "Wait for better setup"
                
                # Exit trigger
                if signal == 'BUY':
                    exit_trigger = "Price breaks below support line"
                elif signal == 'SELL':
                    exit_trigger = "Price breaks above resistance line"
                else:
                    exit_trigger = "Wait for breakout signal"
                
                # Update result with pattern analysis
                result.update({
                    # Pattern specific data
                    'pattern_type': pattern.get('pattern', 'Unknown'),
                    'bias': bias,
                    'breakout': pattern.get('breakout'),
                    'expected_breakout': pattern.get('expected_breakout'),
                    
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
                    'pattern_height': pattern.get('pattern_height'),
                    'pattern_width': pattern.get('pattern_width'),
                    'convergence_point': pattern.get('convergence_point'),
                    'resistance_line': pattern.get('resistance_line'),
                    'support_line': pattern.get('support_line'),
                    
                    # Raw data
                    'raw_data': {
                        'ohlcv_data': ohlcv_data,
                        'pattern_data': pattern
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