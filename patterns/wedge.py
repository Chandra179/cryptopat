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
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from data import get_data_collector


def calculate_line_slope(points: List[Tuple[int, float]]) -> Optional[float]:
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


def find_wedge_points(highs: pd.Series, lows: pd.Series, window: int = 4) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
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


def detect_wedge_pattern(df: pd.DataFrame, min_touches: int = 3) -> Optional[Dict]:
    """
    Detect Wedge patterns in OHLCV data
    
    Args:
        df: DataFrame with OHLCV data
        min_touches: Minimum touches required for valid trendline
        
    Returns:
        Dictionary with pattern details or None if not found
    """
    if len(df) < 50:
        return None
        
    swing_highs, swing_lows = find_wedge_points(df['high'], df['low'])
    
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
            resistance_slope = calculate_line_slope(resistance_points)
            support_slope = calculate_line_slope(support_points)
            
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
            
            # For proper wedge, support line should be less steep than resistance
            # (i.e., they should converge, not diverge)
            if resistance_slope > 0:  # Rising wedge
                if support_slope >= resistance_slope:  # Support steeper than resistance
                    continue
                wedge_type = "Rising Wedge"
                bias = "BEARISH"
                expected_breakout = "Downward"
            else:  # Falling wedge
                if support_slope <= resistance_slope:  # Support less steep than resistance
                    continue
                wedge_type = "Falling Wedge"
                bias = "BULLISH" 
                expected_breakout = "Upward"
            
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
            
            # Add confidence for breakout
            if breakout == expected_breakout:
                confidence += 20
            elif breakout and breakout != expected_breakout:
                confidence -= 15
                
            # Pattern is more reliable if it's not too narrow
            if pattern_height > df['close'].iloc[-1] * 0.02:  # At least 2% height
                confidence += 10
                
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


def analyze_wedge(symbol: str = "BTC/USDT", timeframe: str = "4h", limit: int = 100) -> Dict:
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
        pattern = detect_wedge_pattern(df)
        
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


def format_wedge_output(analysis: Dict) -> str:
    """
    Format Wedge analysis output for terminal display
    
    Args:
        analysis: Analysis results dictionary
        
    Returns:
        Formatted output string
    """
    if 'error' in analysis:
        return f"❌ Error: {analysis['error']}"

    if not analysis['pattern_detected']:
        symbol_clean = analysis['symbol'].replace('/', '').upper()
        return f"{symbol_clean} ({analysis['timeframe']}) - Wedge\nPrice: {analysis['current_price']} | Signal: NONE ⏳ | Neckline: —\nTarget: — | Confidence: —"
    
    # Pattern detected
    symbol_clean = analysis['symbol'].replace('/', '').upper()
    pattern_type = analysis['pattern']
    resistance = analysis['resistance_line']
    support = analysis['support_line']
    
    # Determine signal and emoji
    if analysis.get('breakout'):
        if analysis['breakout'] == "Upward":
            signal = "BUY"
            signal_emoji = "🚀"
        elif analysis['breakout'] == "Downward":
            signal = "SELL"
            signal_emoji = "📉"
        else:
            signal = "NONE"
            signal_emoji = "⏳"
    else:
        signal = "NONE"
        signal_emoji = "⏳"
    
    # Use the appropriate trendline as neckline
    neckline = resistance['current_level'] if pattern_type == "Rising Wedge" else support['current_level']
    target = "—"  # Wedge patterns don't have specific targets like H&S
    
    output = f"{symbol_clean} ({analysis['timeframe']}) - {pattern_type}\n"
    output += f"Price: {analysis['current_price']} | Signal: {signal} {signal_emoji} | Neckline: {neckline}\n"
    output += f"Target: {target} | Confidence: {analysis['confidence']}%"
    
    return output


if __name__ == "__main__":
    # Test the Wedge analysis
    result = analyze_wedge("BTC/USDT", "4h", 100)
    print(format_wedge_output(result))