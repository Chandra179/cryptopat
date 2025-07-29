"""
Inverse Head and Shoulders Pattern Detection

Detects Inverse Head and Shoulders (iH&S) pattern which is a classic bullish reversal pattern.

Pattern Components:
- Left Shoulder (LS): first swing low
- Head: lower swing low 
- Right Shoulder (RS): higher low near LS level
- Neckline: resistance line connecting LS-RS highs
- Confirmation: Close above neckline after RS

Signal: BUY when price breaks above neckline after RS formation
Best timeframes: 4h, 1d
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from data import get_data_collector


def find_swing_highs_lows(prices: pd.Series, window: int = 5) -> Tuple[List[int], List[int]]:
    """
    Find swing highs and lows in price data
    
    Args:
        prices: Price series (typically Close prices)
        window: Lookback/lookahead window for swing detection
        
    Returns:
        Tuple of (swing_high_indices, swing_low_indices)
    """
    swing_highs = []
    swing_lows = []
    
    for i in range(window, len(prices) - window):
        # Check for swing high
        is_high = all(prices.iloc[i] >= prices.iloc[j] for j in range(i - window, i + window + 1) if j != i)
        if is_high:
            swing_highs.append(i)
            
        # Check for swing low
        is_low = all(prices.iloc[i] <= prices.iloc[j] for j in range(i - window, i + window + 1) if j != i)
        if is_low:
            swing_lows.append(i)
    
    return swing_highs, swing_lows


def detect_inverse_head_and_shoulders(prices: pd.Series, timestamps: pd.Series, tolerance: float = 0.03) -> Optional[Dict]:
    """
    Detect Inverse Head and Shoulders pattern
    
    Args:
        prices: Close price series
        timestamps: Timestamp series
        tolerance: Tolerance for shoulder height similarity (3% default)
        
    Returns:
        Dictionary with pattern details or None if not found
    """
    if len(prices) < 80:
        return None
        
    swing_highs, swing_lows = find_swing_highs_lows(prices)
    
    if len(swing_lows) < 3 or len(swing_highs) < 2:
        return None
    
    # Look for potential iH&S patterns in recent swing lows
    for i in range(len(swing_lows) - 2):
        left_shoulder_idx = swing_lows[i]
        head_idx = swing_lows[i + 1] 
        right_shoulder_idx = swing_lows[i + 2]
        
        left_shoulder_price = prices.iloc[left_shoulder_idx]
        head_price = prices.iloc[head_idx]
        right_shoulder_price = prices.iloc[right_shoulder_idx]
        
        # Head must be lower than both shoulders
        if head_price >= left_shoulder_price or head_price >= right_shoulder_price:
            continue
            
        # Shoulders should be near-equal height (within tolerance)
        shoulder_diff = abs(left_shoulder_price - right_shoulder_price) / left_shoulder_price
        if shoulder_diff > tolerance:
            continue
            
        # Find peaks between shoulders and head for neckline
        left_peak_candidates = [idx for idx in swing_highs if left_shoulder_idx < idx < head_idx]
        right_peak_candidates = [idx for idx in swing_highs if head_idx < idx < right_shoulder_idx]
        
        if not left_peak_candidates or not right_peak_candidates:
            continue
            
        left_peak_idx = left_peak_candidates[-1]  # Closest to head
        right_peak_idx = right_peak_candidates[0]  # Closest to head
        
        left_peak_price = prices.iloc[left_peak_idx]
        right_peak_price = prices.iloc[right_peak_idx]
        neckline_level = (left_peak_price + right_peak_price) / 2
        
        # Check if pattern is confirmed (price broke above neckline)
        current_price = prices.iloc[-1]
        confirmed = current_price > neckline_level
        
        # Calculate pattern confidence based on various factors
        head_depth = (min(left_shoulder_price, right_shoulder_price) - head_price) / head_price
        confidence = min(100, int(head_depth * 100 + (1 - shoulder_diff) * 50))
        
        return {
            'pattern': 'Inverse Head and Shoulders',
            'left_shoulder': {
                'price': round(left_shoulder_price, 4),
                'timestamp': timestamps.iloc[left_shoulder_idx],
                'index': left_shoulder_idx
            },
            'head': {
                'price': round(head_price, 4), 
                'timestamp': timestamps.iloc[head_idx],
                'index': head_idx
            },
            'right_shoulder': {
                'price': round(right_shoulder_price, 4),
                'timestamp': timestamps.iloc[right_shoulder_idx], 
                'index': right_shoulder_idx
            },
            'neckline': round(neckline_level, 4),
            'current_price': round(current_price, 4),
            'confirmed': confirmed,
            'signal': 'BUY' if confirmed else 'PENDING',
            'confidence': confidence,
            'target_price': round(neckline_level + (neckline_level - head_price), 4)  # Pattern target
        }
    
    return None


def analyze_inverse_head_and_shoulders(symbol: str = "SOL/USDT", timeframe: str = "4h", limit: int = 150) -> Dict:
    """
    Analyze Inverse Head and Shoulders pattern for given symbol and timeframe
    
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
        
        if not ohlcv_data or len(ohlcv_data) < 80:
            return {
                'error': f'Insufficient data: need at least 80 candles, got {len(ohlcv_data) if ohlcv_data else 0}',
                'symbol': symbol,
                'timeframe': timeframe
            }
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Detect pattern
        pattern = detect_inverse_head_and_shoulders(df['close'], df['timestamp'])
        
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


def format_inverse_head_and_shoulders_output(analysis: Dict) -> str:
    """
    Format Inverse Head and Shoulders analysis output for terminal display
    
    Args:
        analysis: Analysis results dictionary
        
    Returns:
        Formatted output string
    """
    if 'error' in analysis:
        return f"❌ Error: {analysis['error']}"
    
    output = []
    output.append(f"Inverse Head and Shoulders Analysis - {analysis['symbol']} ({analysis['timeframe']})")
    output.append(f"Current Price: {analysis['current_price']} | Time: {analysis['current_time']}")
    output.append("")
    
    if not analysis['pattern_detected']:
        output.append("🔍 No Inverse Head and Shoulders pattern detected in current data")
        output.append(f"Analyzed {analysis['total_candles']} candles - need clear L.Shoulder → Head → R.Shoulder formation")
        return '\n'.join(output)
    
    # Pattern detected
    ls = analysis['left_shoulder']
    head = analysis['head'] 
    rs = analysis['right_shoulder']
    
    output.append(f"🚀 Inverse Head and Shoulders Pattern Detected!")
    output.append(f"Left Shoulder:  {ls['price']} at {ls['timestamp'].strftime('%Y-%m-%d %H:%M')}")
    output.append(f"Head:          {head['price']} at {head['timestamp'].strftime('%Y-%m-%d %H:%M')}")
    output.append(f"Right Shoulder: {rs['price']} at {rs['timestamp'].strftime('%Y-%m-%d %H:%M')}")
    output.append(f"Neckline:      {analysis['neckline']}")
    output.append("")
    
    if analysis['confirmed']:
        output.append(f"✅ CONFIRMED: Price broke neckline → {analysis['signal']}")
        output.append(f"🎯 Target Price: {analysis['target_price']}")
        output.append("🚀 Breakout Reversal - Bull Trend Forming")
    else:
        output.append(f"⏳ PENDING: Waiting for neckline break at {analysis['neckline']}")
        output.append(f"Current: {analysis['current_price']} | Need close above {analysis['neckline']}")
    
    output.append(f"🎯 Confidence: {analysis['confidence']}%")
    
    return '\n'.join(output)


if __name__ == "__main__":
    # Test the Inverse Head and Shoulders analysis
    result = analyze_inverse_head_and_shoulders("SOL/USDT", "4h", 150)
    print(format_inverse_head_and_shoulders_output(result))