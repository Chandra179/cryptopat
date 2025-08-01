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


def detect_inverse_head_and_shoulders(prices: pd.Series, timestamps: pd.Series, volumes: pd.Series, tolerance: float = 0.03) -> Optional[Dict]:
    """
    Detect Inverse Head and Shoulders pattern
    
    Args:
        prices: Close price series
        timestamps: Timestamp series
        volumes: Volume series
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
            
        # Select most significant peaks (highest prices)
        left_peak_idx = max(left_peak_candidates, key=lambda x: prices.iloc[x])
        right_peak_idx = max(right_peak_candidates, key=lambda x: prices.iloc[x])
        
        left_peak_price = prices.iloc[left_peak_idx]
        right_peak_price = prices.iloc[right_peak_idx]
        
        # Calculate neckline using proper trendline (slope between peaks)
        if left_peak_idx != right_peak_idx:
            slope = (right_peak_price - left_peak_price) / (right_peak_idx - left_peak_idx)
            # Calculate neckline level at current timestamp
            current_idx = len(prices) - 1
            neckline_level = left_peak_price + slope * (current_idx - left_peak_idx)
        else:
            neckline_level = left_peak_price
        
        # Check if pattern is confirmed (price broke above neckline with volume)
        current_price = prices.iloc[-1]
        price_confirmed = current_price > neckline_level
        
        # Volume confirmation: recent volume > average volume of pattern period
        pattern_start_idx = left_shoulder_idx
        pattern_volumes = volumes.iloc[pattern_start_idx:]
        avg_pattern_volume = pattern_volumes.mean()
        recent_volume = volumes.iloc[-5:].mean()  # Last 5 candles average
        volume_confirmed = recent_volume > avg_pattern_volume * 1.2  # 20% above average
        
        confirmed = price_confirmed and volume_confirmed
        
        # Validate time proportions (each component should be reasonable length)
        left_section_length = head_idx - left_shoulder_idx
        right_section_length = right_shoulder_idx - head_idx
        time_ratio = min(left_section_length, right_section_length) / max(left_section_length, right_section_length)
        time_proportion_valid = time_ratio > 0.3  # Sections shouldn't be too unbalanced
        
        if not time_proportion_valid:
            continue
            
        # Calculate pattern confidence based on various factors
        head_depth = (min(left_shoulder_price, right_shoulder_price) - head_price) / head_price
        volume_factor = 1.2 if volume_confirmed else 0.8
        time_factor = time_ratio
        confidence = min(100, int(head_depth * 100 * volume_factor * time_factor + (1 - shoulder_diff) * 50))
        
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
            'price_confirmed': price_confirmed,
            'volume_confirmed': volume_confirmed,
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
        pattern = detect_inverse_head_and_shoulders(df['close'], df['timestamp'], df['volume'])
        
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
    
    output = "\n"
    output += "===============================================================\n"
    output += "INVERSE HEAD AND SHOULDERS PATTERN\n"
    output += "===============================================================\n"
    
    if not analysis['pattern_detected']:
        output += "PATTERN_STATUS: NOT_DETECTED | NECKLINE: — | TARGET: —\n"
        output += "CURRENT_PRICE: ${:.4f} | CONFIDENCE: 0% | VOLUME_CONFIRMED: ✗\n".format(analysis['current_price'])
        output += "LEFT_SHOULDER: — | HEAD: — | RIGHT_SHOULDER: —\n"
        output += "ANALYSIS_TIME: {} | DATA_POINTS: {}\n".format(analysis['current_time'], analysis['total_candles'])
        output += "ACTION: WAITING FOR PATTERN\n"
        return output
    
    # Pattern detected - format with phase 3 specifications
    volume_status = "✓" if analysis['volume_confirmed'] else "✗"
    price_status = "✓" if analysis['price_confirmed'] else "✗"
    
    # Calculate risk-reward ratio
    if analysis['confirmed']:
        risk_reward = (analysis['target_price'] - analysis['current_price']) / (analysis['current_price'] - analysis['head']['price'])
        risk_reward_str = f"{risk_reward:.2f}:1"
    else:
        risk_reward_str = "INSUFFICIENT_DATA"
    
    # Format timestamps
    left_shoulder_time = analysis['left_shoulder']['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    head_time = analysis['head']['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    right_shoulder_time = analysis['right_shoulder']['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    
    output += "RISK_REWARD: {} | VOLUME_CONFIRMED: {} ({}) | CONFIDENCE: {}%\n".format(
        risk_reward_str, volume_status, analysis['confidence']/100, analysis['confidence'])
    output += "PRICE_CONFIRMED: {} | NECKLINE: ${:.4f} | TARGET: ${:.4f}\n".format(
        price_status, analysis['neckline'], analysis['target_price'] if analysis['confirmed'] else 0)
    output += "LEFT_SHOULDER: ${:.4f} | HEAD: ${:.4f} | RIGHT_SHOULDER: ${:.4f}\n".format(
        analysis['left_shoulder']['price'], analysis['head']['price'], analysis['right_shoulder']['price'])
    output += "LS_TIME: {} | HEAD_TIME: {}\n".format(left_shoulder_time, head_time)
    output += "RS_TIME: {} | CURRENT_TIME: {}\n".format(right_shoulder_time, analysis['current_time'])
    
    # Determine action based on confirmation status
    if analysis['confirmed']:
        action = "BUY"
    elif analysis['pattern_detected'] and not analysis['confirmed']:
        action = "WAITING FOR BREAKOUT"
    else:
        action = "NEUTRAL"
    
    output += "ACTION: {}\n".format(action)
    
    return output


if __name__ == "__main__":
    # Test the Inverse Head and Shoulders analysis
    result = analyze_inverse_head_and_shoulders("SOL/USDT", "4h", 150)
    print(format_inverse_head_and_shoulders_output(result))