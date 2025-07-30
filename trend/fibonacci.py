import numpy as np
from typing import Dict, List, Tuple, Optional


class FibonacciCalculator:
    """Standard Fibonacci retracement and extension calculations"""
    
    RETRACEMENT_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
    EXTENSION_LEVELS = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618]
    
    @staticmethod
    def calculate_retracements(high: float, low: float) -> Dict[float, float]:
        """Calculate Fibonacci retracement levels"""
        diff = high - low
        return {
            level: high - (diff * level) 
            for level in FibonacciCalculator.RETRACEMENT_LEVELS
        }
    
    @staticmethod
    def calculate_extensions(start: float, end: float, reference: float) -> Dict[float, float]:
        """Calculate Fibonacci extension levels"""
        diff = abs(end - start)
        direction = 1 if end > start else -1
        
        return {
            level: reference + (diff * level * direction)
            for level in FibonacciCalculator.EXTENSION_LEVELS
        }
    
    @staticmethod
    def find_nearest_fib_level(price: float, fib_levels: Dict[float, float], 
                              tolerance: float = 0.01) -> Optional[Tuple[float, float]]:
        """Find nearest Fibonacci level within tolerance"""
        min_distance = float('inf')
        nearest_level = None
        
        for level, level_price in fib_levels.items():
            distance = abs(price - level_price) / level_price
            if distance <= tolerance and distance < min_distance:
                min_distance = distance
                nearest_level = (level, level_price)
        
        return nearest_level
    
    @staticmethod
    def is_at_fib_level(price: float, fib_levels: Dict[float, float], 
                       tolerance: float = 0.015) -> bool:
        """Check if price is at a Fibonacci level"""
        return FibonacciCalculator.find_nearest_fib_level(price, fib_levels, tolerance) is not None


class ZigZagDetector:
    """ZigZag pattern detection for identifying swing points"""
    
    @staticmethod
    def find_zigzag_points(highs: np.ndarray, lows: np.ndarray, 
                          threshold_percent: float = 4.0) -> List[Tuple[int, float, str]]:
        """Find ZigZag pivot points with minimum threshold"""
        points = []
        last_direction = None
        last_point = None
        
        for i in range(len(highs)):
            high = highs[i]
            low = lows[i]
            
            if last_point is None:
                last_point = (i, high, 'high')
                points.append(last_point)
                continue
            
            if last_point[2] == 'high':
                if high > last_point[1]:
                    last_point = (i, high, 'high')
                    points[-1] = last_point
                elif (last_point[1] - low) / last_point[1] * 100 >= threshold_percent:
                    last_point = (i, low, 'low')
                    points.append(last_point)
            else:
                if low < last_point[1]:
                    last_point = (i, low, 'low')
                    points[-1] = last_point
                elif (high - last_point[1]) / last_point[1] * 100 >= threshold_percent:
                    last_point = (i, high, 'high')
                    points.append(last_point)
        
        return points
    
    @staticmethod
    def get_swing_points(data: np.ndarray, window: int = 5) -> List[Tuple[int, float, str]]:
        """Get swing high and low points using window-based detection"""
        swing_points = []
        
        for i in range(window, len(data) - window):
            is_swing_high = all(data[i] >= data[j] for j in range(i - window, i + window + 1) if j != i)
            is_swing_low = all(data[i] <= data[j] for j in range(i - window, i + window + 1) if j != i)
            
            if is_swing_high:
                swing_points.append((i, data[i], 'high'))
            elif is_swing_low:
                swing_points.append((i, data[i], 'low'))
        
        return swing_points