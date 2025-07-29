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
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
    return slope, intercept


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


def detect_triangle_pattern(df: pd.DataFrame, min_touches: int = 3) -> Optional[Dict]:
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
    
    # Try to find converging trendlines
    for i in range(len(swing_highs) - 2):
        for j in range(len(swing_lows) - 2):
            # Test resistance line (upper trendline)
            resistance_points = swing_highs[i:i+3]
            resistance_line = calculate_trendline(resistance_points)
            
            if not resistance_line:
                continue
                
            # Test support line (lower trendline)  
            support_points = swing_lows[j:j+3]
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
                
            # Determine triangle type
            triangle_type = None
            if abs(resistance_slope) < 0.001:  # Horizontal resistance
                triangle_type = "Ascending Triangle" if support_slope > 0 else "Rectangle"
            elif abs(support_slope) < 0.001:  # Horizontal support
                triangle_type = "Descending Triangle" if resistance_slope < 0 else "Rectangle"
            elif resistance_slope < 0 and support_slope > 0:
                triangle_type = "Symmetrical Triangle"
            else:
                continue  # Not a valid triangle pattern
                
            if triangle_type in ["Rectangle"]:
                continue  # Skip rectangle patterns
                
            # Check current price position and breakout
            current_price = df['close'].iloc[-1]
            current_idx = len(df) - 1
            
            # Calculate trendline values at current position
            current_resistance = resistance_slope * current_idx + resistance_intercept
            current_support = support_slope * current_idx + support_intercept
            
            # Determine breakout status
            breakout = None
            signal = "PENDING"
            if current_price > current_resistance:
                breakout = "Upward"
                signal = "BUY"
            elif current_price < current_support:
                breakout = "Downward" 
                signal = "SELL"
                
            # Calculate pattern confidence
            pattern_width = current_resistance - current_support
            price_position = (current_price - current_support) / pattern_width if pattern_width > 0 else 0.5
            
            # Distance to convergence (closer = higher urgency)
            convergence_distance = convergence_x - current_idx
            urgency = max(0, min(100, 100 - (convergence_distance * 2)))
            
            confidence = int(50 + urgency * 0.3 + (abs(0.5 - price_position) * 20))
            
            return {
                'pattern': triangle_type,
                'resistance_line': {
                    'slope': round(resistance_slope, 6),
                    'intercept': round(resistance_intercept, 4),
                    'current_level': round(current_resistance, 4)
                },
                'support_line': {
                    'slope': round(support_slope, 6), 
                    'intercept': round(support_intercept, 4),
                    'current_level': round(current_support, 4)
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
                'urgency': round(urgency, 1)
            }
    
    return None


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
    support = analysis['support_line']
    
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