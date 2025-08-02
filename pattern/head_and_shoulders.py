"""
Head and Shoulders Pattern Detection

Detects Head and Shoulders (H&S) pattern which is a classic bearish reversal pattern.

Pattern Components:
- Left Shoulder (LS): first swing high
- Head: higher swing high 
- Right Shoulder (RS): lower high near LS level
- Neckline: support line connecting LS-RS valleys
- Confirmation: Close below neckline after RS

Signal: SELL when price breaks neckline after RS formation
Best timeframes: 4h, 1d
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from data import get_data_collector


class HeadAndShouldersStrategy:
    """Head and Shoulders pattern detection strategy for cryptocurrency trend analysis."""
    
    def __init__(self):
        self.collector = get_data_collector()

    def find_swing_highs_lows(self, prices: pd.Series, volumes: pd.Series, window: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """
        Find swing highs and lows in price data with volume confirmation
        
        Args:
            prices: Price series (typically Close prices)
            volumes: Volume series for confirmation
            window: Lookback/lookahead window for swing detection
            
        Returns:
            Tuple of (swing_high_data, swing_low_data) with price, volume, and index info
        """
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(prices) - window):
            # Check for swing high
            is_high = all(prices.iloc[i] >= prices.iloc[j] for j in range(i - window, i + window + 1) if j != i)
            if is_high:
                swing_highs.append({
                    'index': i,
                    'price': prices.iloc[i],
                    'volume': volumes.iloc[i],
                    'significance': prices.iloc[i] / prices.iloc[max(0, i-20):i+21].mean()  # Relative prominence
                })
                
            # Check for swing low
            is_low = all(prices.iloc[i] <= prices.iloc[j] for j in range(i - window, i + window + 1) if j != i)
            if is_low:
                swing_lows.append({
                    'index': i,
                    'price': prices.iloc[i], 
                    'volume': volumes.iloc[i],
                    'significance': prices.iloc[max(0, i-20):i+21].mean() / prices.iloc[i]  # Relative depth
                })
        
        return swing_highs, swing_lows

    def calculate_neckline_slope(self, left_valley: Dict, right_valley: Dict) -> Tuple[float, float]:
        """
        Calculate neckline slope and level at any given point
        
        Args:
            left_valley: Left valley data with index and price
            right_valley: Right valley data with index and price
            
        Returns:
            Tuple of (slope, intercept) for neckline equation
        """
        x1, y1 = left_valley['index'], left_valley['price']
        x2, y2 = right_valley['index'], right_valley['price']
        
        if x2 == x1:
            return 0, y1  # Horizontal line
        
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        return slope, intercept

    def get_neckline_level(self, slope: float, intercept: float, index: int) -> float:
        """
        Get neckline level at specific index using trend line equation
        
        Args:
            slope: Neckline slope
            intercept: Neckline intercept
            index: Index to calculate level for
            
        Returns:
            Neckline level at given index
        """
        return slope * index + intercept

    def validate_volume_pattern(self, left_shoulder_vol: float, head_vol: float, right_shoulder_vol: float, 
                              left_valley_vol: float, right_valley_vol: float) -> float:
        """
        Validate volume pattern for Head and Shoulders (decreasing volume during formation)
        
        Args:
            left_shoulder_vol: Volume at left shoulder
            head_vol: Volume at head
            right_shoulder_vol: Volume at right shoulder
            left_valley_vol: Volume at left valley
            right_valley_vol: Volume at right valley
            
        Returns:
            Volume score (0-1, higher is better)
        """
        # Classic H&S shows decreasing volume from left shoulder to right shoulder
        volume_decline_score = 0
        if head_vol < left_shoulder_vol and right_shoulder_vol < head_vol:
            volume_decline_score += 0.5
        if right_shoulder_vol < left_shoulder_vol:
            volume_decline_score += 0.3
            
        # Valley volumes should be lower (selling exhaustion)
        valley_score = 0
        avg_shoulder_vol = (left_shoulder_vol + head_vol + right_shoulder_vol) / 3
        avg_valley_vol = (left_valley_vol + right_valley_vol) / 2
        if avg_valley_vol < avg_shoulder_vol:
            valley_score = 0.2
            
        return min(1.0, volume_decline_score + valley_score)

    def detect_head_and_shoulders(self, prices: pd.Series, volumes: pd.Series, timestamps: pd.Series, tolerance: float = 0.03) -> Optional[Dict]:
        """
        Detect Head and Shoulders pattern
        
        Args:
            prices: Close price series
            volumes: Volume series for confirmation
            timestamps: Timestamp series
            tolerance: Tolerance for shoulder height similarity (3% default)
            
        Returns:
            Dictionary with pattern details or None if not found
        """
        if len(prices) < 80:
            raise ValueError(f"Insufficient data for Head and Shoulders detection: need at least 80 prices, got {len(prices)}")
            
        swing_highs, swing_lows = self.find_swing_highs_lows(prices, volumes)
        
        if len(swing_highs) < 3 or len(swing_lows) < 2:
            raise ValueError(f"Insufficient swing points: need at least 3 highs and 2 lows, got {len(swing_highs)} highs and {len(swing_lows)} lows")
        
        # Look for potential H&S patterns in recent swing highs
        for i in range(len(swing_highs) - 2):
            left_shoulder = swing_highs[i]
            head = swing_highs[i + 1] 
            right_shoulder = swing_highs[i + 2]
            
            # Head must be higher than both shoulders
            if head['price'] <= left_shoulder['price'] or head['price'] <= right_shoulder['price']:
                continue
                
            # Shoulders should be near-equal height (within tolerance)
            shoulder_diff = abs(left_shoulder['price'] - right_shoulder['price']) / left_shoulder['price']
            if shoulder_diff > tolerance:
                continue
                
            # Time symmetry check - shoulders should be roughly equidistant from head
            left_time_dist = head['index'] - left_shoulder['index']
            right_time_dist = right_shoulder['index'] - head['index']
            time_symmetry = min(left_time_dist, right_time_dist) / max(left_time_dist, right_time_dist)
            if time_symmetry < 0.5:  # Allow up to 2:1 time ratio
                continue
                
            # Find most significant valleys between shoulders and head for neckline
            left_valley_candidates = [v for v in swing_lows if left_shoulder['index'] < v['index'] < head['index']]
            right_valley_candidates = [v for v in swing_lows if head['index'] < v['index'] < right_shoulder['index']]
            
            if not left_valley_candidates or not right_valley_candidates:
                continue
                
            # Select most significant valleys (deepest relative to surrounding prices)
            left_valley = max(left_valley_candidates, key=lambda x: x['significance'])
            right_valley = max(right_valley_candidates, key=lambda x: x['significance'])
            
            # Calculate proper neckline using trend line
            neckline_slope, neckline_intercept = self.calculate_neckline_slope(left_valley, right_valley)
            current_neckline_level = self.get_neckline_level(neckline_slope, neckline_intercept, len(prices) - 1)
            
            # Check if pattern is confirmed (price broke below neckline)
            current_price = prices.iloc[-1]
            confirmed = current_price < current_neckline_level
            
            # Volume pattern validation
            volume_score = self.validate_volume_pattern(
                left_shoulder['volume'], head['volume'], right_shoulder['volume'],
                left_valley['volume'], right_valley['volume']
            )
            
            # Calculate comprehensive pattern confidence
            head_prominence = (head['price'] - max(left_shoulder['price'], right_shoulder['price'])) / head['price']
            shoulder_symmetry_score = 1 - shoulder_diff
            
            confidence_factors = {
                'head_prominence': head_prominence * 30,
                'shoulder_symmetry': shoulder_symmetry_score * 25,
                'time_symmetry': time_symmetry * 20,
                'volume_pattern': volume_score * 25
            }
            
            confidence = min(100, int(sum(confidence_factors.values())))
            
            # Calculate target using proper neckline level at head position
            head_neckline_level = self.get_neckline_level(neckline_slope, neckline_intercept, head['index'])
            target_distance = head['price'] - head_neckline_level
            target_price = current_neckline_level - target_distance
            
            return {
                'pattern': 'Head and Shoulders',
                'left_shoulder': {
                    'price': round(left_shoulder['price'], 4),
                    'timestamp': timestamps.iloc[left_shoulder['index']],
                    'index': left_shoulder['index'],
                    'volume': round(left_shoulder['volume'], 2)
                },
                'head': {
                    'price': round(head['price'], 4), 
                    'timestamp': timestamps.iloc[head['index']],
                    'index': head['index'],
                    'volume': round(head['volume'], 2)
                },
                'right_shoulder': {
                    'price': round(right_shoulder['price'], 4),
                    'timestamp': timestamps.iloc[right_shoulder['index']], 
                    'index': right_shoulder['index'],
                    'volume': round(right_shoulder['volume'], 2)
                },
                'left_valley': {
                    'price': round(left_valley['price'], 4),
                    'timestamp': timestamps.iloc[left_valley['index']],
                    'index': left_valley['index']
                },
                'right_valley': {
                    'price': round(right_valley['price'], 4),
                    'timestamp': timestamps.iloc[right_valley['index']],
                    'index': right_valley['index']
                },
                'neckline': round(current_neckline_level, 4),
                'neckline_slope': round(neckline_slope, 6),
                'current_price': round(current_price, 4),
                'confirmed': confirmed,
                'signal': 'SELL' if confirmed else 'PENDING',
                'confidence': confidence,
                'confidence_breakdown': confidence_factors,
                'volume_score': round(volume_score, 2),
                'time_symmetry': round(time_symmetry, 2),
                'target_price': round(target_price, 4)
            }
        
        return None

    def analyze(self, symbol: str, timeframe: str, limit: int) -> Dict:
        """
        Analyze Head and Shoulders pattern for given symbol and timeframe
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for analysis
            limit: Number of candles to analyze
            
        Returns:
            Analysis results dictionary
        """
        try:
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
            
            if not ohlcv_data or len(ohlcv_data) < 80:
                return {
                    'error': f'Insufficient data: need at least 80 candles, got {len(ohlcv_data) if ohlcv_data else 0}',
                    'success': False,
                    'symbol': symbol,
                    'timeframe': timeframe
                }
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Detect pattern with volume analysis
            pattern = self.detect_head_and_shoulders(df['close'], df['volume'], df['timestamp'])
            
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
                'current_volume': round(df['volume'].iloc[-1], 2),
                'pattern_detected': pattern is not None
            }
            
            if pattern:
                # Calculate additional metrics based on pattern
                confidence = pattern.get('confidence', 50)
                signal = pattern.get('signal', 'HOLD')
                
                # Calculate support/resistance levels
                support_level = pattern.get('neckline', current_price * 0.98)
                resistance_level = pattern.get('head', {}).get('price', current_price * 1.02)
                
                # Calculate stop loss and take profit zones
                if signal == 'SELL':
                    stop_zone = resistance_level * 1.02  # Above head as invalidation
                    tp_low = pattern.get('target_price', support_level * 0.95)
                    tp_high = tp_low * 0.98  # Extended target
                else:
                    stop_zone = support_level * 0.98
                    tp_low = resistance_level * 1.02
                    tp_high = resistance_level * 1.05
                
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
                if signal == 'SELL':
                    exit_trigger = "Price breaks above head level"
                else:
                    exit_trigger = "Wait for neckline break signal"
                
                # Update result with pattern analysis
                result.update({
                    # Pattern specific data
                    'pattern_type': pattern.get('pattern', 'Head and Shoulders'),
                    'bias': 'BEARISH' if pattern.get('confirmed') else 'NEUTRAL',
                    
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
                    
                    # Pattern details from original analysis
                    'neckline_level': pattern.get('neckline'),
                    'target_price': pattern.get('target_price'),
                    'confirmed': pattern.get('confirmed', False),
                    'left_shoulder': pattern.get('left_shoulder'),
                    'head': pattern.get('head'),
                    'right_shoulder': pattern.get('right_shoulder'),
                    
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