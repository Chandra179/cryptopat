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
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from data import get_data_collector


class FlagStrategy:
    """Flag pattern detection strategy for cryptocurrency trend analysis."""
    
    def __init__(self):
        self.collector = get_data_collector()


    def find_flagpole(self, df: pd.DataFrame, min_move_percent: float = 5.0) -> Optional[Dict]:
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


    def detect_flag_consolidation(self, df: pd.DataFrame, flagpole: Dict) -> Optional[Dict]:
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


    def detect_flag_pattern(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect complete Flag pattern (flagpole + flag consolidation)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Complete flag pattern details or None if not found
        """
        # First find the flagpole
        flagpole = self.find_flagpole(df)
        if not flagpole:
            return None
        
        # Then find flag consolidation after the pole
        flag = self.detect_flag_consolidation(df, flagpole)
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


    def analyze(self, symbol: str, timeframe: str, limit: int) -> Dict:
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
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
            
            if not ohlcv_data or len(ohlcv_data) < 30:
                return {
                    'error': f'Insufficient data: need at least 30 candles, got {len(ohlcv_data) if ohlcv_data else 0}',
                    'success': False,
                    'symbol': symbol,
                    'timeframe': timeframe
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Detect pattern
            pattern = self.detect_flag_pattern(df)
            
            # Get current price and timestamp info
            current_price = df['close'].iloc[-1]
            current_timestamp = df['timestamp'].iloc[-1]
            dt = datetime.fromtimestamp(current_timestamp.timestamp(), tz=timezone.utc)
            
            result = {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
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
                pattern_type = pattern.get('pattern', 'Unknown')
                
                # Extract flag details for level calculations
                flag_details = pattern.get('flag', {})
                flagpole_details = pattern.get('flagpole', {})
                
                # Calculate support/resistance levels based on flag pattern
                if flagpole_details.get('direction') == 'up':
                    support_level = flag_details.get('low', current_price * 0.98)
                    resistance_level = flag_details.get('high', current_price * 1.02)
                    bias = 'BULLISH'
                else:
                    support_level = flag_details.get('low', current_price * 0.98)
                    resistance_level = flag_details.get('high', current_price * 1.02)
                    bias = 'BEARISH'
                
                # Calculate stop loss and take profit zones
                if signal == 'BUY':
                    stop_zone = support_level * 0.995
                    tp_low = pattern.get('target_price', resistance_level * 1.05)
                    tp_high = tp_low * 1.02
                elif signal == 'SELL':
                    stop_zone = resistance_level * 1.005
                    tp_low = pattern.get('target_price', support_level * 0.95)
                    tp_high = tp_low * 0.98
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
                    exit_trigger = "Price breaks below flag support"
                elif signal == 'SELL':
                    exit_trigger = "Price breaks above flag resistance"
                else:
                    exit_trigger = "Wait for breakout signal"
                
                # Update result with pattern analysis
                result.update({
                    # Pattern specific data
                    'pattern_type': pattern_type,
                    'bias': bias,
                    'breakout': pattern.get('breakout'),
                    
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
                    'flagpole': pattern.get('flagpole'),
                    'flag': pattern.get('flag'),
                    'target_price': pattern.get('target_price'),
                    
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

