"""
Triangle Pattern Detection

Detects Triangle patterns which are continuation patterns that form when price action is squeezed between converging trendlines.

Triangle Types:
- Ascending Triangle: horizontal resistance + rising support (bullish)
- Descending Triangle: falling resistance + horizontal support (bearish)  
- Symmetrical Triangle: converging resistance and support (neutral, breaks in trend direction)

Pattern Components:
- Upper trendline (resistance)
- Lower trendline (support)
- Convergence point where lines meet
- Breakout confirmation with volume

Signal: BUY/SELL when price breaks trendline with volume confirmation
Best timeframes: 4h, 1d
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from data import get_data_collector


def calculate_trendline(points: List[Tuple[int, float]]) -> Optional[Tuple[float, float]]:
    """
    Calculate trendline slope and intercept from price points
    
    Args:
        points: List of (index, price) tuples
        
    Returns:
        Tuple of (slope, intercept) or None if insufficient points
    """
    if len(points) < 2:
        return None
        
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    
    # Linear regression for trendline
    n = len(points)
    sum_x = sum(x_vals)
    sum_y = sum(y_vals)
    sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
    sum_x2 = sum(x * x for x in x_vals)
    
    # Prevent division by zero
    denominator = n * sum_x2 - sum_x * sum_x
    if abs(denominator) < 1e-10:
        return None
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    
    return slope, intercept


def validate_trendline_touches(df: pd.DataFrame, line_params: Tuple[float, float], 
                             is_resistance: bool, tolerance_pct: float = 0.02) -> Tuple[int, List[int]]:
    """
    Validate how many times price actually touched a trendline
    
    Args:
        df: OHLCV DataFrame
        line_params: (slope, intercept) of trendline
        is_resistance: True for resistance line, False for support
        tolerance_pct: Percentage tolerance for touch validation
        
    Returns:
        Tuple of (touch_count, touch_indices)
    """
    slope, intercept = line_params
    touches = []
    
    for i in range(len(df)):
        line_value = slope * i + intercept
        tolerance = line_value * tolerance_pct
        
        if is_resistance:
            # Check if high touched resistance line
            if abs(df['high'].iloc[i] - line_value) <= tolerance:
                touches.append(i)
        else:
            # Check if low touched support line
            if abs(df['low'].iloc[i] - line_value) <= tolerance:
                touches.append(i)
    
    return len(touches), touches


def calculate_pattern_strength(df: pd.DataFrame, resistance_line: Tuple[float, float], 
                             support_line: Tuple[float, float], start_idx: int, end_idx: int) -> float:
    """
    Calculate how well price respected the triangle pattern boundaries
    
    Args:
        df: OHLCV DataFrame
        resistance_line: (slope, intercept) of resistance
        support_line: (slope, intercept) of support
        start_idx: Pattern start index
        end_idx: Pattern end index
        
    Returns:
        Pattern strength score (0-100)
    """
    violations = 0
    total_candles = end_idx - start_idx
    
    if total_candles <= 0:
        return 0
    
    res_slope, res_intercept = resistance_line
    sup_slope, sup_intercept = support_line
    
    for i in range(start_idx, end_idx):
        res_level = res_slope * i + res_intercept
        sup_level = sup_slope * i + sup_intercept
        
        # Check for violations (price breaking through lines significantly)
        if df['high'].iloc[i] > res_level * 1.005:  # 0.5% buffer
            violations += 1
        elif df['low'].iloc[i] < sup_level * 0.995:  # 0.5% buffer
            violations += 1
    
    strength = max(0, 100 - (violations * 100 / total_candles))
    return strength


def find_swing_points(highs: pd.Series, lows: pd.Series, window: int = 5) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Find swing highs and lows for trendline construction
    
    Args:
        highs: High prices series
        lows: Low prices series
        window: Lookback window for swing detection
        
    Returns:
        Tuple of (swing_highs, swing_lows) as (index, price) lists
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


def detect_triangle_pattern(df: pd.DataFrame, min_touches: int = 3, min_duration: int = 20) -> Optional[Dict]:
    """
    Detect Triangle patterns in OHLCV data
    
    Args:
        df: DataFrame with OHLCV data
        min_touches: Minimum touches required for valid trendline
        
    Returns:
        Dictionary with pattern details or None if not found
    """
    if len(df) < 50:
        return None
        
    swing_highs, swing_lows = find_swing_points(df['high'], df['low'])
    
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return None
    
    # Try to find converging trendlines using sliding window approach
    best_pattern = None
    best_strength = 0
    
    for res_start in range(len(swing_highs) - 2):
        for sup_start in range(len(swing_lows) - 2):
            # Try different combinations of swing points for trendlines
            for res_end in range(res_start + 2, min(res_start + 6, len(swing_highs))):
                for sup_end in range(sup_start + 2, min(sup_start + 6, len(swing_lows))):
                    
                    # Test resistance line (upper trendline)
                    resistance_points = swing_highs[res_start:res_end]
                    resistance_line = calculate_trendline(resistance_points)
                    
                    if not resistance_line:
                        continue
                        
                    # Test support line (lower trendline)  
                    support_points = swing_lows[sup_start:sup_end]
                    support_line = calculate_trendline(support_points)
                    
                    if not support_line:
                        continue
                        
                    resistance_slope, resistance_intercept = resistance_line
                    support_slope, support_intercept = support_line
                    
                    # Check if lines are converging (different slopes)
                    if abs(resistance_slope - support_slope) < 0.0001:
                        continue
                        
                    # Find convergence point
                    convergence_x = (support_intercept - resistance_intercept) / (resistance_slope - support_slope)
                    convergence_y = resistance_slope * convergence_x + resistance_intercept
                    
                    # Convergence should be in future (ahead of current data)
                    if convergence_x <= len(df):
                        continue
                        
                    # Pattern duration check
                    pattern_start = min(resistance_points[0][0], support_points[0][0])
                    pattern_end = max(resistance_points[-1][0], support_points[-1][0])
                    if pattern_end - pattern_start < min_duration:
                        continue
                        
                    # Validate actual touches on trendlines
                    res_touches, _ = validate_trendline_touches(df, resistance_line, True)
                    sup_touches, _ = validate_trendline_touches(df, support_line, False)
                    
                    if res_touches < min_touches or sup_touches < min_touches:
                        continue
                        
                    # Calculate pattern strength
                    strength = calculate_pattern_strength(df, resistance_line, support_line, pattern_start, pattern_end)
                    if strength < 60:  # Minimum strength threshold
                        continue
                        
                    # Determine triangle type using percentage-based thresholds
                    price_range = df['high'].max() - df['low'].min()
                    slope_threshold = price_range / len(df) * 0.1  # Dynamic threshold
                    
                    triangle_type = None
                    if abs(resistance_slope) < slope_threshold:  # Nearly horizontal resistance
                        triangle_type = "Ascending Triangle" if support_slope > slope_threshold else "Rectangle"
                    elif abs(support_slope) < slope_threshold:  # Nearly horizontal support
                        triangle_type = "Descending Triangle" if resistance_slope < -slope_threshold else "Rectangle"
                    elif resistance_slope < -slope_threshold and support_slope > slope_threshold:
                        triangle_type = "Symmetrical Triangle"
                    else:
                        continue  # Not a valid triangle pattern
                        
                    if triangle_type in ["Rectangle"]:
                        continue  # Skip rectangle patterns
                        
                    # Keep the best pattern found
                    if strength > best_strength:
                        best_strength = strength
                        best_pattern = {
                            'resistance_line': resistance_line,
                            'support_line': support_line,
                            'triangle_type': triangle_type,
                            'convergence_x': convergence_x,
                            'convergence_y': convergence_y,
                            'pattern_start': pattern_start,
                            'pattern_end': pattern_end,
                            'strength': strength,
                            'res_touches': res_touches,
                            'sup_touches': sup_touches
                        }
    
    if not best_pattern:
        return None
        
    # Use the best pattern found
    resistance_line = best_pattern['resistance_line']
    support_line = best_pattern['support_line']
    triangle_type = best_pattern['triangle_type']
    convergence_x = best_pattern['convergence_x']
    convergence_y = best_pattern['convergence_y']
    resistance_slope, resistance_intercept = resistance_line
    support_slope, support_intercept = support_line
                
    # Check current price position and breakout
    current_price = df['close'].iloc[-1]
    current_idx = len(df) - 1
    current_volume = df['volume'].iloc[-1]
    
    # Calculate trendline values at current position
    current_resistance = resistance_slope * current_idx + resistance_intercept
    current_support = support_slope * current_idx + support_intercept
    
    # Volume analysis for breakout confirmation
    avg_volume = df['volume'].tail(20).mean()
    volume_surge = current_volume > avg_volume * 1.5
    
    # Determine breakout status with volume confirmation
    breakout = None
    signal = "PENDING"
    breakout_strength = 0
    
    if current_price > current_resistance * 1.002:  # 0.2% buffer for noise
        breakout = "Upward"
        if volume_surge:
            signal = "BUY"
            breakout_strength = 100
        else:
            signal = "WEAK_BUY"
            breakout_strength = 60
    elif current_price < current_support * 0.998:  # 0.2% buffer for noise
        breakout = "Downward"
        if volume_surge:
            signal = "SELL"
            breakout_strength = 100
        else:
            signal = "WEAK_SELL"
            breakout_strength = 60
    
    # Calculate pattern confidence based on multiple factors
    pattern_width = current_resistance - current_support
    
    # Distance to convergence (closer = higher urgency)
    convergence_distance = convergence_x - current_idx
    urgency = max(0, min(100, 100 - (convergence_distance * 2)))
    
    # Multi-factor confidence calculation
    base_confidence = best_pattern['strength']  # Pattern strength (60-100)
    touch_bonus = min(20, (best_pattern['res_touches'] + best_pattern['sup_touches'] - 6) * 2)
    urgency_factor = urgency * 0.2
    volume_factor = 10 if volume_surge else 0
    
    confidence = int(min(100, base_confidence + touch_bonus + urgency_factor + volume_factor))
    
    return {
        'pattern': triangle_type,
        'resistance_line': {
            'slope': round(resistance_slope, 6),
            'intercept': round(resistance_intercept, 4),
            'current_level': round(current_resistance, 4),
            'touches': best_pattern['res_touches']
        },
        'support_line': {
            'slope': round(support_slope, 6), 
            'intercept': round(support_intercept, 4),
            'current_level': round(current_support, 4),
            'touches': best_pattern['sup_touches']
        },
        'convergence_point': {
            'x': round(convergence_x, 1),
            'y': round(convergence_y, 4),
            'distance': round(convergence_distance, 1)
        },
        'current_price': round(current_price, 4),
        'breakout': breakout,
        'signal': signal,
        'confidence': confidence,
        'pattern_width': round(pattern_width, 4),
        'urgency': round(urgency, 1),
        'pattern_strength': round(best_pattern['strength'], 1),
        'volume_surge': volume_surge,
        'breakout_strength': breakout_strength,
        'pattern_duration': best_pattern['pattern_end'] - best_pattern['pattern_start']
    }


def analyze_triangle(symbol: str = "BTC/USDT", timeframe: str = "4h", limit: int = 100) -> Dict:
    """
    Analyze Triangle patterns for given symbol and timeframe
    
    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe for analysis
        limit: Number of candles to analyze
        
    Returns:
        Analysis results dictionary
    """
    try:
        collector = get_data_collector()
        ohlcv_data = collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if not ohlcv_data or len(ohlcv_data) < 50:
            return {
                'error': f'Insufficient data: need at least 50 candles, got {len(ohlcv_data) if ohlcv_data else 0}',
                'symbol': symbol,
                'timeframe': timeframe
            }
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Detect pattern
        pattern = detect_triangle_pattern(df)
        
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_candles': len(df),
            'current_price': round(df['close'].iloc[-1], 4),
            'current_time': df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S'),
            'pattern_detected': pattern is not None
        }
        
        if pattern:
            result.update(pattern)
            
        return result
        
    except Exception as e:
        return {
            'error': f'Analysis failed: {str(e)}',
            'symbol': symbol,
            'timeframe': timeframe
        }


def format_triangle_output(analysis: Dict) -> str:
    """
    Format Triangle analysis output for terminal display
    
    Args:
        analysis: Analysis results dictionary
        
    Returns:
        Formatted output string
    """
    if 'error' in analysis:
        return f"❌ Error: {analysis['error']}"
    
    if not analysis['pattern_detected']:
        symbol_clean = analysis['symbol'].replace('/', '').upper()
        return f"{symbol_clean} ({analysis['timeframe']}) - Triangle\nPrice: {analysis['current_price']} | Signal: NONE ⏳ | Neckline: —\nTarget: — | Confidence: —"
    
    # Pattern detected
    symbol_clean = analysis['symbol'].replace('/', '').upper()
    pattern_type = analysis['pattern']
    resistance = analysis['resistance_line']
    
    # Determine signal and emoji
    if analysis.get('breakout'):
        if analysis['breakout'] == "Upward":
            signal = "BUY"
            signal_emoji = "🚀"
        else:
            signal = "SELL"
            signal_emoji = "📉"
    else:
        signal = "NONE"
        signal_emoji = "⏳"
    
    # Use resistance level as neckline reference
    neckline = resistance['current_level']
    target = "—"  # Triangle patterns don't have specific targets like H&S
    
    output = f"{symbol_clean} ({analysis['timeframe']}) - {pattern_type}\n"
    output += f"Price: {analysis['current_price']} | Signal: {signal} {signal_emoji} | Neckline: {neckline}\n"
    output += f"Target: {target} | Confidence: {analysis['confidence']}%"
    
    return output


if __name__ == "__main__":
    # Test the Triangle analysis
    result = analyze_triangle("BTC/USDT", "4h", 100)
    print(format_triangle_output(result))