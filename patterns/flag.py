"""
Flag Pattern Detection

Detects Flag patterns which are short-term continuation patterns that form after strong price movements.

Flag Types:
- Bull Flag: strong uptrend + small downward consolidation (bullish continuation)
- Bear Flag: strong downtrend + small upward consolidation (bearish continuation)

Pattern Components:
- Flagpole: Strong directional move (trend)
- Flag: Small counter-trend consolidation (rectangular channel)
- Breakout: Continuation in original trend direction
- Volume: High on flagpole, low during flag, high on breakout

Signal: BUY/SELL when price breaks flag in trend direction
Best timeframes: 1h, 4h, 1d
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from data import get_data_collector


def find_flagpole(df: pd.DataFrame, min_move_percent: float = 5.0) -> Optional[Dict]:
    """
    Find the flagpole (strong directional move) in recent price action
    
    Args:
        df: OHLCV DataFrame
        min_move_percent: Minimum percentage move to qualify as flagpole
        
    Returns:
        Flagpole details or None if not found
    """
    if len(df) < 20:
        return None
    
    # Look for strong moves in recent data (last 50% of data)
    start_idx = len(df) // 2
    
    for i in range(start_idx, len(df) - 10):  # Need room for flag after pole
        for lookback in [5, 8, 12, 15]:  # Different flagpole lengths
            if i - lookback < 0:
                continue
                
            start_price = df['close'].iloc[i - lookback]
            end_price = df['close'].iloc[i]
            
            # Calculate percentage move
            move_percent = abs(end_price - start_price) / start_price * 100
            
            if move_percent >= min_move_percent:
                # Check if move is relatively straight (not too much retracement)
                prices_in_move = df['close'].iloc[i - lookback:i + 1]
                
                if end_price > start_price:  # Upward move
                    # For bull flag, check price didn't retrace significantly
                    max_price = prices_in_move.max()
                    min_retracement = (max_price - prices_in_move.min()) / (max_price - start_price)
                    
                    if min_retracement < 0.3:  # Less than 30% retracement
                        return {
                            'direction': 'up',
                            'start_idx': i - lookback,
                            'end_idx': i,
                            'start_price': round(start_price, 4),
                            'end_price': round(end_price, 4),
                            'move_percent': round(move_percent, 2),
                            'length': lookback
                        }
                else:  # Downward move
                    # For bear flag, check price didn't retrace significantly upward
                    min_price = prices_in_move.min()
                    max_retracement = (prices_in_move.max() - min_price) / (start_price - min_price)
                    
                    if max_retracement < 0.3:  # Less than 30% retracement
                        return {
                            'direction': 'down',
                            'start_idx': i - lookback,
                            'end_idx': i,
                            'start_price': round(start_price, 4),
                            'end_price': round(end_price, 4),
                            'move_percent': round(move_percent, 2),
                            'length': lookback
                        }
    
    return None


def detect_flag_consolidation(df: pd.DataFrame, flagpole: Dict) -> Optional[Dict]:
    """
    Detect flag consolidation pattern after flagpole
    
    Args:
        df: OHLCV DataFrame
        flagpole: Flagpole details
        
    Returns:
        Flag consolidation details or None if not found
    """
    flag_start = flagpole['end_idx']
    flag_data = df.iloc[flag_start:]
    
    if len(flag_data) < 5:  # Need minimum data for flag
        return None
    
    # Flag should be much smaller move than flagpole
    max_flag_length = min(flagpole['length'], 15)  # Flag shouldn't be longer than pole
    
    for flag_length in range(5, min(len(flag_data), max_flag_length + 1)):
        flag_segment = flag_data.iloc[:flag_length]
        
        flag_high = flag_segment['high'].max()
        flag_low = flag_segment['low'].min()
        flag_range = flag_high - flag_low
        
        # Flag range should be small compared to flagpole move
        flagpole_range = abs(flagpole['end_price'] - flagpole['start_price'])
        flag_to_pole_ratio = flag_range / flagpole_range
        
        if flag_to_pole_ratio > 0.6:  # Flag too big compared to pole
            continue
            
        # Check if consolidation is in expected direction
        flag_start_price = flag_segment['close'].iloc[0]
        flag_end_price = flag_segment['close'].iloc[-1]
        
        if flagpole['direction'] == 'up':
            # Bull flag should consolidate slightly down or sideways
            flag_direction = (flag_end_price - flag_start_price) / flag_start_price * 100
            if flag_direction > 2:  # Flag moving up too much for bull flag
                continue
        else:
            # Bear flag should consolidate slightly up or sideways  
            flag_direction = (flag_end_price - flag_start_price) / flag_start_price * 100
            if flag_direction < -2:  # Flag moving down too much for bear flag
                continue
        
        # Check for parallel channel (rectangular flag)
        highs = flag_segment['high'].values
        lows = flag_segment['low'].values
        
        # Simple parallel check - highs and lows should be relatively flat
        high_slope = (highs[-1] - highs[0]) / len(highs)
        low_slope = (lows[-1] - lows[0]) / len(lows)
        
        # Slopes should be small (consolidation, not trending)
        if abs(high_slope) < flag_range * 0.1 and abs(low_slope) < flag_range * 0.1:
            return {
                'start_idx': flag_start,
                'end_idx': flag_start + flag_length - 1,
                'length': flag_length,
                'high': round(flag_high, 4),
                'low': round(flag_low, 4),
                'range': round(flag_range, 4),
                'flag_to_pole_ratio': round(flag_to_pole_ratio, 3),
                'start_price': round(flag_start_price, 4),
                'end_price': round(flag_end_price, 4)
            }
    
    return None


def detect_flag_pattern(df: pd.DataFrame) -> Optional[Dict]:
    """
    Detect complete Flag pattern (flagpole + flag consolidation)
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        Complete flag pattern details or None if not found
    """
    # First find the flagpole
    flagpole = find_flagpole(df)
    if not flagpole:
        return None
    
    # Then find flag consolidation after the pole
    flag = detect_flag_consolidation(df, flagpole)
    if not flag:
        return None
    
    # Determine pattern type
    pattern_type = "Bull Flag" if flagpole['direction'] == 'up' else "Bear Flag"
    
    # Check for breakout
    current_price = df['close'].iloc[-1]
    current_idx = len(df) - 1
    
    breakout = None
    signal = "PENDING"
    
    if current_idx > flag['end_idx']:  # We have data after flag completion
        if flagpole['direction'] == 'up':
            # Bull flag - look for upward breakout above flag high
            if current_price > flag['high']:
                breakout = "Upward"
                signal = "BUY"
            elif current_price < flag['low']:
                breakout = "Failed"
                signal = "SELL"
        else:
            # Bear flag - look for downward breakout below flag low
            if current_price < flag['low']:
                breakout = "Downward" 
                signal = "SELL"
            elif current_price > flag['high']:
                breakout = "Failed"
                signal = "BUY"
    
    # Calculate target price (flagpole length projected from breakout)
    target_price = None
    if breakout and breakout != "Failed":
        flagpole_length = abs(flagpole['end_price'] - flagpole['start_price'])
        if flagpole['direction'] == 'up':
            target_price = round(flag['high'] + flagpole_length, 4)
        else:
            target_price = round(flag['low'] - flagpole_length, 4)
    
    # Calculate confidence based on pattern quality
    confidence = 60  # Base confidence
    
    # Add confidence for strong flagpole
    if flagpole['move_percent'] > 10:
        confidence += 20
    elif flagpole['move_percent'] > 7:
        confidence += 10
    
    # Add confidence for good flag proportion
    if flag['flag_to_pole_ratio'] < 0.3:
        confidence += 15
    elif flag['flag_to_pole_ratio'] < 0.5:
        confidence += 10
    
    # Add confidence for breakout
    if breakout == "Upward" or breakout == "Downward":
        confidence += 15
    elif breakout == "Failed":
        confidence -= 20
    
    confidence = max(10, min(95, confidence))
    
    return {
        'pattern': pattern_type,
        'flagpole': flagpole,
        'flag': flag,
        'current_price': round(current_price, 4),
        'breakout': breakout,
        'signal': signal,
        'target_price': target_price,
        'confidence': confidence
    }


def analyze_flag(symbol: str = "ETH/USDT", timeframe: str = "4h", limit: int = 100) -> Dict:
    """
    Analyze Flag patterns for given symbol and timeframe
    
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
        
        if not ohlcv_data or len(ohlcv_data) < 30:
            return {
                'error': f'Insufficient data: need at least 30 candles, got {len(ohlcv_data) if ohlcv_data else 0}',
                'symbol': symbol,
                'timeframe': timeframe
            }
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Detect pattern
        pattern = detect_flag_pattern(df)
        
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


def format_flag_output(analysis: Dict) -> str:
    """
    Format Flag analysis output for terminal display
    
    Args:
        analysis: Analysis results dictionary
        
    Returns:
        Formatted output string
    """
    if 'error' in analysis:
        return f"❌ Error: {analysis['error']}"
    
    if not analysis['pattern_detected']:
        symbol_clean = analysis['symbol'].replace('/', '').upper()
        return f"{symbol_clean} ({analysis['timeframe']}) - Flag\nPrice: {analysis['current_price']} | Signal: NONE ⏳ | Neckline: —\nTarget: — | Confidence: —"
    
    # Pattern detected
    symbol_clean = analysis['symbol'].replace('/', '').upper()
    pattern_type = analysis['pattern']
    flag = analysis['flag']
    
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
    
    # Use flag high as neckline reference
    neckline = flag['high'] if pattern_type == "Bull Flag" else flag['low']
    target = analysis.get('target_price', '—')
    
    output = f"{symbol_clean} ({analysis['timeframe']}) - {pattern_type}\n"
    output += f"Price: {analysis['current_price']} | Signal: {signal} {signal_emoji} | Neckline: {neckline}\n"
    output += f"Target: {target} | Confidence: {analysis['confidence']}%"
    
    return output


if __name__ == "__main__":
    # Test the Flag analysis
    result = analyze_flag("ETH/USDT", "4h", 100)
    print(format_flag_output(result))