"""
Pivot Point Analysis Module
Calculates standard, Fibonacci, Woodie's and Camarilla pivot points for support/resistance levels
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from data import get_data_collector


class PivotPointAnalyzer:
    def __init__(self):
        self.data_collector = get_data_collector()
    
    def calculate_standard_pivots(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate standard pivot points"""
        pp = (high + low + close) / 3
        
        s1 = (2 * pp) - high
        s2 = pp - (high - low)
        s3 = low - 2 * (high - pp)
        
        r1 = (2 * pp) - low
        r2 = pp + (high - low)
        r3 = high + 2 * (pp - low)
        
        return {
            'pivot': pp,
            's1': s1, 's2': s2, 's3': s3,
            'r1': r1, 'r2': r2, 'r3': r3
        }
    
    def calculate_fibonacci_pivots(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate Fibonacci pivot points"""
        pp = (high + low + close) / 3
        range_hl = high - low
        
        s1 = pp - 0.382 * range_hl
        s2 = pp - 0.618 * range_hl
        s3 = pp - 1.000 * range_hl
        
        r1 = pp + 0.382 * range_hl
        r2 = pp + 0.618 * range_hl
        r3 = pp + 1.000 * range_hl
        
        return {
            'pivot': pp,
            's1': s1, 's2': s2, 's3': s3,
            'r1': r1, 'r2': r2, 'r3': r3
        }
    
    def calculate_woodies_pivots(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate Woodie's pivot points"""
        pp = (high + low + 2 * close) / 4
        
        s1 = (2 * pp) - high
        s2 = pp - (high - low)
        s3 = low - 2 * (high - pp)
        
        r1 = (2 * pp) - low
        r2 = pp + (high - low)
        r3 = high + 2 * (pp - low)
        
        return {
            'pivot': pp,
            's1': s1, 's2': s2, 's3': s3,
            'r1': r1, 'r2': r2, 'r3': r3
        }
    
    def calculate_camarilla_pivots(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate Camarilla pivot points"""
        pp = (high + low + close) / 3
        range_hl = high - low
        
        s1 = close - 1.1/12 * range_hl
        s2 = close - 1.1/6 * range_hl
        s3 = close - 1.1/4 * range_hl
        s4 = close - 1.1/2 * range_hl
        
        r1 = close + 1.1/12 * range_hl
        r2 = close + 1.1/6 * range_hl
        r3 = close + 1.1/4 * range_hl
        r4 = close + 1.1/2 * range_hl
        
        return {
            'pivot': pp,
            's1': s1, 's2': s2, 's3': s3, 's4': s4,
            'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4
        }
    
    def get_pivot_analysis(self, current_price: float, pivots: Dict[str, float]) -> Dict[str, Any]:
        """Analyze current price position relative to pivot levels"""
        pp = pivots['pivot']
        
        # Determine bias
        bias = "bullish" if current_price > pp else "bearish"
        
        # Find nearest support and resistance
        supports = [v for k, v in pivots.items() if k.startswith('s') and v < current_price]
        resistances = [v for k, v in pivots.items() if k.startswith('r') and v > current_price]
        
        nearest_support = max(supports) if supports else None
        nearest_resistance = min(resistances) if resistances else None
        
        # Calculate distances
        support_distance = abs(current_price - nearest_support) if nearest_support else None
        resistance_distance = abs(nearest_resistance - current_price) if nearest_resistance else None
        
        return {
            'bias': bias,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_distance': support_distance,
            'resistance_distance': resistance_distance,
            'support_distance_pct': (support_distance / current_price * 100) if support_distance else None,
            'resistance_distance_pct': (resistance_distance / current_price * 100) if resistance_distance else None
        }
    
    def analyze(self, symbol: str, timeframe: str, limit: int = 100, ohlcv_data=None) -> Dict[str, Any]:
        """
        Perform comprehensive pivot point analysis
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe for analysis (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch
            
        Returns:
            Dictionary containing pivot point analysis results
        """
        try:
            # Use provided OHLCV data or fetch new data
            if ohlcv_data is not None:
                raw_ohlcv = ohlcv_data
            else:
                raw_ohlcv = self.data_collector.fetch_ohlcv_data(symbol, timeframe, limit)
            
            if not raw_ohlcv or len(raw_ohlcv) == 0:
                return {'success': False, 'error': 'No data available'}
            
            # Convert to pandas DataFrame for easier access
            df = pd.DataFrame(raw_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Get current price and previous period's high, low, close
            current_price = df['close'].iloc[-1]
            prev_high = df['high'].iloc[-2] if len(df) > 1 else df['high'].iloc[-1]
            prev_low = df['low'].iloc[-2] if len(df) > 1 else df['low'].iloc[-1]
            prev_close = df['close'].iloc[-2] if len(df) > 1 else df['close'].iloc[-1]
            
            # Calculate all pivot types
            standard_pivots = self.calculate_standard_pivots(prev_high, prev_low, prev_close)
            fibonacci_pivots = self.calculate_fibonacci_pivots(prev_high, prev_low, prev_close)
            woodies_pivots = self.calculate_woodies_pivots(prev_high, prev_low, prev_close)
            camarilla_pivots = self.calculate_camarilla_pivots(prev_high, prev_low, prev_close)
            
            # Get analysis for standard pivots (most commonly used)
            pivot_analysis = self.get_pivot_analysis(current_price, standard_pivots)
            
            # Calculate pivot strength (how close to pivot levels)
            all_levels = list(standard_pivots.values())
            distances = [abs(current_price - level) for level in all_levels]
            min_distance = min(distances)
            pivot_strength = max(0, 100 - (min_distance / current_price * 100))
            
            return {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': current_price,
                'standard_pivots': standard_pivots,
                'fibonacci_pivots': fibonacci_pivots,
                'woodies_pivots': woodies_pivots,
                'camarilla_pivots': camarilla_pivots,
                'analysis': pivot_analysis,
                'pivot_strength': pivot_strength,
                'signal': self._generate_signal(pivot_analysis, pivot_strength)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_signal(self, analysis: Dict[str, Any], strength: float) -> str:
        """Generate trading signal based on pivot analysis"""
        bias = analysis['bias']
        support_dist = analysis.get('support_distance_pct', 100)
        resistance_dist = analysis.get('resistance_distance_pct', 100)
        
        if strength > 80:  # Very close to pivot level
            return "NEUTRAL - At Key Level"
        elif bias == "bullish":
            if support_dist and support_dist < 2:  # Within 2% of support
                return "BUY - Near Support"
            elif resistance_dist and resistance_dist < 2:  # Near resistance
                return "SELL - Near Resistance"
            else:
                return "HOLD - Bullish Bias"
        else:  # bearish bias
            if resistance_dist and resistance_dist < 2:  # Near resistance
                return "SELL - Near Resistance"
            elif support_dist and support_dist < 2:  # Near support
                return "BUY - Near Support"
            else:
                return "HOLD - Bearish Bias"


def analyze_pivot_points(symbol: str, timeframe: str, limit: int = 100) -> Dict[str, Any]:
    """Convenience function for pivot point analysis"""
    analyzer = PivotPointAnalyzer()
    return analyzer.analyze(symbol, timeframe, limit)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python techin/pivotpoint.py <symbol> [timeframe] [limit]")
        print("Example: python techin/pivotpoint.py BTC/USDT 1h 100")
        sys.exit(1)
    
    symbol = sys.argv[1]
    timeframe = sys.argv[2] if len(sys.argv) > 2 else '1h'
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    result = analyze_pivot_points(symbol, timeframe, limit)
    
    if result['success']:
        print(f"\n=== PIVOT POINT ANALYSIS: {symbol} ({timeframe}) ===")
        print(f"Current Price: ${result['current_price']:.4f}")
        print(f"Bias: {result['analysis']['bias'].upper()}")
        print(f"Signal: {result['signal']}")
        print(f"Pivot Strength: {result['pivot_strength']:.1f}%")
        
        print("\n--- Standard Pivot Points ---")
        pivots = result['standard_pivots']
        for level in ['r3', 'r2', 'r1', 'pivot', 's1', 's2', 's3']:
            print(f"{level.upper()}: ${pivots[level]:.4f}")
        
        if result['analysis']['nearest_support']:
            print(f"\nNearest Support: ${result['analysis']['nearest_support']:.4f} ({result['analysis']['support_distance_pct']:.2f}% away)")
        if result['analysis']['nearest_resistance']:
            print(f"Nearest Resistance: ${result['analysis']['nearest_resistance']:.4f} ({result['analysis']['resistance_distance_pct']:.2f}% away)")
    else:
        print(f"Error: {result['error']}")