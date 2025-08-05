"""
Renko Chart implementation for cryptocurrency pattern analysis.
Creates price-based bricks that filter out time and noise, focusing on significant price movements.
"""

from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from data import get_data_collector
import math


class RenkoStrategy:
    """Renko chart strategy for trend analysis using price-based bricks."""
    
    def __init__(self):
        self.collector = get_data_collector()
    
    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Calculate Average True Range for dynamic brick sizing."""
        if len(highs) < period + 1:
            return 0.0
            
        true_ranges = []
        for i in range(1, len(highs)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            true_ranges.append(max(hl, hc, lc))
        
        if len(true_ranges) < period:
            return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
            
        return sum(true_ranges[-period:]) / period
    
    def calculate_brick_size(self, closes: List[float], highs: List[float], lows: List[float], 
                           method: str = 'atr', atr_multiplier: float = 2.0, fixed_size: Optional[float] = None) -> float:
        """
        Calculate Renko brick size using different methods.
        
        Args:
            closes: List of closing prices
            highs: List of high prices  
            lows: List of low prices
            method: 'atr', 'fixed', or 'percentage'
            atr_multiplier: Multiplier for ATR-based brick size
            fixed_size: Fixed brick size (for 'fixed' method)
            
        Returns:
            Calculated brick size
        """
        if method == 'fixed' and fixed_size:
            return fixed_size
        elif method == 'percentage':
            avg_price = sum(closes[-50:]) / min(50, len(closes))
            return avg_price * 0.02  # 2% of average price
        else:  # ATR method (default)
            atr = self.calculate_atr(highs, lows, closes)
            return atr * atr_multiplier if atr > 0 else closes[-1] * 0.01
    
    def generate_renko_bricks(self, closes: List[float], highs: List[float], lows: List[float], 
                             timestamps: List[datetime], brick_size: float) -> List[Dict]:
        """
        Generate Renko bricks from OHLCV data.
        
        Args:
            closes: List of closing prices
            highs: List of high prices
            lows: List of low prices  
            timestamps: List of timestamps
            brick_size: Size of each Renko brick
            
        Returns:
            List of Renko brick dictionaries
        """
        if not closes or brick_size <= 0:
            return []
            
        bricks = []
        current_brick_high = None
        current_brick_low = None
        
        # Initialize first brick based on first price
        first_price = closes[0]
        current_brick_low = first_price
        current_brick_high = first_price + brick_size
        
        bricks.append({
            'timestamp': timestamps[0],
            'open': first_price,
            'close': current_brick_high,
            'high': current_brick_high,
            'low': current_brick_low,
            'direction': 'bullish',
            'brick_number': 1
        })
        
        brick_count = 1
        
        for i in range(1, len(closes)):
            price = closes[i]
            
            # Check for bullish brick formation
            while price >= current_brick_high + brick_size:
                brick_count += 1
                new_low = current_brick_high
                new_high = current_brick_high + brick_size
                
                bricks.append({
                    'timestamp': timestamps[i],
                    'open': new_low,
                    'close': new_high,
                    'high': new_high,
                    'low': new_low,
                    'direction': 'bullish',
                    'brick_number': brick_count
                })
                
                current_brick_low = new_low
                current_brick_high = new_high
            
            # Check for bearish brick formation
            while price <= current_brick_low - brick_size:
                brick_count += 1
                new_high = current_brick_low
                new_low = current_brick_low - brick_size
                
                bricks.append({
                    'timestamp': timestamps[i],
                    'open': new_high,
                    'close': new_low,
                    'high': new_high,
                    'low': new_low,
                    'direction': 'bearish',
                    'brick_number': brick_count
                })
                
                current_brick_high = new_high
                current_brick_low = new_low
        
        return bricks
    
    def analyze_renko_pattern(self, bricks: List[Dict]) -> Dict:
        """
        Analyze Renko brick patterns for trend signals.
        
        Args:
            bricks: List of Renko brick dictionaries
            
        Returns:
            Dictionary with pattern analysis results
        """
        if len(bricks) < 3:
            return {
                'signal': 'INSUFFICIENT_DATA',
                'trend': 'unknown',
                'consecutive_bricks': 0,
                'trend_strength': 'unknown',
                'reversal_signal': False
            }
        
        # Count consecutive bricks in same direction
        consecutive_count = 1
        current_direction = bricks[-1]['direction']
        
        for i in range(len(bricks) - 2, -1, -1):
            if bricks[i]['direction'] == current_direction:
                consecutive_count += 1
            else:
                break
        
        # Determine trend strength based on consecutive bricks
        if consecutive_count >= 5:
            trend_strength = 'strong'
        elif consecutive_count >= 3:
            trend_strength = 'moderate'
        else:
            trend_strength = 'weak'
        
        # Check for reversal signals
        reversal_signal = False
        if len(bricks) >= 2:
            if bricks[-1]['direction'] != bricks[-2]['direction']:
                reversal_signal = True
        
        # Generate trading signal
        signal = 'HOLD'
        if current_direction == 'bullish' and consecutive_count >= 3:
            signal = 'BUY'
        elif current_direction == 'bearish' and consecutive_count >= 3:
            signal = 'SELL'
        elif reversal_signal and consecutive_count == 1:
            signal = 'REVERSAL_ALERT'
        
        return {
            'signal': signal,
            'trend': current_direction,
            'consecutive_bricks': consecutive_count,
            'trend_strength': trend_strength,
            'reversal_signal': reversal_signal,
            'total_bricks': len(bricks),
            'bullish_bricks': len([b for b in bricks if b['direction'] == 'bullish']),
            'bearish_bricks': len([b for b in bricks if b['direction'] == 'bearish'])
        }
    
    def analyze(self, symbol: str, timeframe: str, limit: int = 100, 
                brick_method: str = 'atr', atr_multiplier: float = 2.0, ohlcv_data: Optional[List[List]] = None) -> Dict:
        """
        Perform complete Renko analysis on a trading pair.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for analysis (e.g., '1h', '4h', '1d')
            limit: Number of candles to analyze
            brick_method: Method for brick sizing ('atr', 'fixed', 'percentage')
            atr_multiplier: Multiplier for ATR-based brick size
            ohlcv_data: Pre-fetched OHLCV data (optional, will fetch if not provided)
            
        Returns:
            Dictionary containing complete Renko analysis
        """
        try:
            # Use provided OHLCV data or fetch if not provided
            if ohlcv_data is None:
                ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
            
            if not ohlcv_data:
                return {
                    'success': False,
                    'error': f'No data available for {symbol} on {timeframe}',
                    'symbol': symbol,
                    'timeframe': timeframe
                }
            
            # Extract price data
            timestamps = [datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc) for candle in ohlcv_data]
            opens = [float(candle[1]) for candle in ohlcv_data]
            highs = [float(candle[2]) for candle in ohlcv_data]
            lows = [float(candle[3]) for candle in ohlcv_data]
            closes = [float(candle[4]) for candle in ohlcv_data]
            
            # Calculate brick size
            brick_size = self.calculate_brick_size(closes, highs, lows, brick_method, atr_multiplier)
            
            # Generate Renko bricks
            bricks = self.generate_renko_bricks(closes, highs, lows, timestamps, brick_size)
            
            # Analyze patterns
            pattern_analysis = self.analyze_renko_pattern(bricks)
            
            # Calculate additional metrics
            price_change = ((closes[-1] - closes[0]) / closes[0]) * 100
            
            return {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now(timezone.utc),
                'current_price': closes[-1],
                'price_change_pct': round(price_change, 2),
                'brick_size': round(brick_size, 6),
                'brick_method': brick_method,
                'total_bricks': len(bricks),
                'pattern_analysis': pattern_analysis,
                'recent_bricks': bricks[-5:] if len(bricks) >= 5 else bricks,
                'signal': pattern_analysis['signal'],
                'trend': pattern_analysis['trend'],
                'trend_strength': pattern_analysis['trend_strength']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now(timezone.utc)
            }


def analyze_renko(symbol: str, timeframe: str = '1h', limit: int = 100, 
                  brick_method: str = 'atr', atr_multiplier: float = 2.0) -> Dict:
    """
    Convenience function for Renko analysis.
    
    Args:
        symbol: Trading pair symbol
        timeframe: Analysis timeframe
        limit: Number of candles
        brick_method: Brick sizing method
        atr_multiplier: ATR multiplier
        
    Returns:
        Renko analysis results
    """
    strategy = RenkoStrategy()
    return strategy.analyze(symbol, timeframe, limit, brick_method, atr_multiplier)


if __name__ == "__main__":
    import sys
    
    # Default parameters
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTC/USDT'
    timeframe = sys.argv[2] if len(sys.argv) > 2 else '1h'
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    print(f"\n🧱 Renko Chart Analysis for {symbol} ({timeframe})")
    print("=" * 50)
    
    result = analyze_renko(symbol, timeframe, limit)
    
    if result['success']:
        print(f"Current Price: ${result['current_price']:,.2f}")
        print(f"Price Change: {result['price_change_pct']:+.2f}%")
        print(f"Brick Size: ${result['brick_size']:.6f}")
        print(f"Total Bricks: {result['total_bricks']}")
        print(f"\nSignal: {result['signal']}")
        print(f"Trend: {result['trend'].upper()}")
        print(f"Trend Strength: {result['trend_strength'].upper()}")
        
        pattern = result['pattern_analysis']
        print(f"\nPattern Analysis:")
        print(f"  Consecutive Bricks: {pattern['consecutive_bricks']}")
        print(f"  Bullish Bricks: {pattern['bullish_bricks']}")
        print(f"  Bearish Bricks: {pattern['bearish_bricks']}")
        print(f"  Reversal Signal: {pattern['reversal_signal']}")
        
        if result['recent_bricks']:
            print(f"\nRecent Bricks:")
            for i, brick in enumerate(result['recent_bricks']):
                direction_symbol = "🟢" if brick['direction'] == 'bullish' else "🔴"
                print(f"  {direction_symbol} Brick #{brick['brick_number']}: ${brick['low']:.2f} → ${brick['high']:.2f}")
    else:
        print(f"❌ Error: {result['error']}")