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
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from data import get_data_collector


class InverseHeadAndShouldersStrategy:
    """Inverse Head and Shoulders pattern detection strategy for cryptocurrency trend analysis."""
    
    def __init__(self):
        self.collector = get_data_collector()

    def find_swing_highs_lows(self, prices: pd.Series, window: int = 5) -> Tuple[List[int], List[int]]:
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

    def detect_inverse_head_and_shoulders(self, prices: pd.Series, timestamps: pd.Series, volumes: pd.Series, tolerance: float = 0.03) -> Optional[Dict]:
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
            
        swing_highs, swing_lows = self.find_swing_highs_lows(prices)
        
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

    def analyze(self, symbol: str, timeframe: str, limit: int, ohlcv_data: Optional[List] = None) -> Dict:
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
            # Fetch OHLCV data if not provided
            if ohlcv_data is None:
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
            
            # Detect pattern
            pattern = self.detect_inverse_head_and_shoulders(df['close'], df['timestamp'], df['volume'])
            
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
                
                # Calculate support/resistance levels
                neckline_level = pattern.get('neckline', current_price)
                head_price = pattern.get('head', {}).get('price', current_price)
                
                # Calculate stop loss and take profit zones
                if signal == 'BUY':
                    stop_zone = head_price * 0.995  # Below head level
                    tp_low = neckline_level * 1.005  # Above neckline
                    tp_high = pattern.get('target_price', neckline_level * 1.02)  # Pattern target
                else:
                    stop_zone = current_price * 0.98
                    tp_low = current_price * 1.02
                    tp_high = current_price * 1.05
                
                # Calculate Risk/Reward ratio
                if signal == 'BUY':
                    risk = abs(current_price - stop_zone)
                    reward = abs(tp_high - current_price)
                    rr_ratio = reward / risk if risk > 0 else 0
                else:
                    rr_ratio = 0
                
                # Determine entry window
                if signal == 'BUY' and confidence > 70:
                    entry_window = "Optimal now"
                elif pattern.get('confirmed', False):
                    entry_window = "Good on breakout confirmation"
                else:
                    entry_window = "Wait for neckline breakout"
                
                # Exit trigger
                exit_trigger = "Price breaks below head level" if signal == 'BUY' else "Pattern invalidated"
                
                # Update result with pattern analysis
                result.update({
                    # Pattern specific data
                    'pattern_type': pattern.get('pattern', 'Inverse Head and Shoulders'),
                    'bias': 'BULLISH',
                    'confirmed': pattern.get('confirmed', False),
                    'price_confirmed': pattern.get('price_confirmed', False),
                    'volume_confirmed': pattern.get('volume_confirmed', False),
                    
                    # Price levels
                    'neckline_level': round(neckline_level, 4),
                    'head_price': round(head_price, 4),
                    'left_shoulder_price': round(pattern.get('left_shoulder', {}).get('price', 0), 4),
                    'right_shoulder_price': round(pattern.get('right_shoulder', {}).get('price', 0), 4),
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
                    'target_price': pattern.get('target_price'),
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
                    'neckline_level': round(current_price * 1.02, 4),
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