"""
Elliott Wave Pattern Detection Module

Implements Elliott Wave theory analysis with core rules validation:
- 5-wave impulse patterns (1-2-3-4-5)
- 3-wave corrective patterns (A-B-C)
- ZigZag swing detection
- Fibonacci ratio validation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import get_data_collector

class WaveType(Enum):
    IMPULSE = "Impulse Wave"
    CORRECTIVE = "Corrective Wave"
    UNKNOWN = "Unknown"

class TrendDirection(Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"

@dataclass
class SwingPoint:
    """Represents a swing high or low point"""
    index: int
    price: float
    is_high: bool
    timestamp: str

@dataclass
class Wave:
    """Represents an Elliott Wave"""
    number: int
    start_point: SwingPoint
    end_point: SwingPoint
    price_change: float
    percentage_change: float
    
    @property
    def length(self) -> float:
        return abs(self.price_change)

@dataclass
class ElliottWavePattern:
    """Complete Elliott Wave pattern analysis"""
    symbol: str
    timeframe: str
    pattern_type: WaveType
    trend_direction: TrendDirection
    waves: List[Wave]
    is_valid: bool
    validation_results: Dict[str, bool]
    fibonacci_ratios: Dict[str, float]

class ElliottWaveAnalyzer:
    """Elliott Wave pattern detection and validation"""
    
    def __init__(self):
        self.data_collector = get_data_collector()
        
    def detect_zigzag_swings(self, df: pd.DataFrame, threshold_percent: float = 5.0) -> List[SwingPoint]:
        """
        Detect significant swing points using ZigZag algorithm
        
        Args:
            df: OHLCV DataFrame with High, Low, Close columns
            threshold_percent: Minimum percentage change to qualify as swing
            
        Returns:
            List of SwingPoint objects
        """
        if len(df) < 3:
            return []
            
        swings = []
        highs = df['high'].values
        lows = df['low'].values
        timestamps = df['datetime'].astype(str).tolist() if 'datetime' in df.columns else df.index.astype(str).tolist()
        
        # Find initial direction
        current_high = highs[0]
        current_low = lows[0]
        current_high_idx = 0
        current_low_idx = 0
        
        looking_for_high = True
        threshold = threshold_percent / 100.0
        
        for i in range(1, len(df)):
            if looking_for_high:
                if highs[i] > current_high:
                    current_high = highs[i]
                    current_high_idx = i
                elif (current_high - lows[i]) / current_high >= threshold:
                    # Confirmed swing high
                    swings.append(SwingPoint(
                        index=current_high_idx,
                        price=current_high,
                        is_high=True,
                        timestamp=timestamps[current_high_idx]
                    ))
                    current_low = lows[i]
                    current_low_idx = i
                    looking_for_high = False
            else:
                if lows[i] < current_low:
                    current_low = lows[i]
                    current_low_idx = i
                elif (highs[i] - current_low) / current_low >= threshold:
                    # Confirmed swing low
                    swings.append(SwingPoint(
                        index=current_low_idx,
                        price=current_low,
                        is_high=False,
                        timestamp=timestamps[current_low_idx]
                    ))
                    current_high = highs[i]
                    current_high_idx = i
                    looking_for_high = True
        
        return swings
    
    def create_waves_from_swings(self, swings: List[SwingPoint]) -> List[Wave]:
        """Convert swing points into wave structures"""
        if len(swings) < 2:
            return []
            
        waves = []
        for i in range(len(swings) - 1):
            start = swings[i]
            end = swings[i + 1]
            
            price_change = end.price - start.price
            percentage_change = ((end.price - start.price) / start.price) * 100
            
            wave = Wave(
                number=i + 1,
                start_point=start,
                end_point=end,
                price_change=price_change,
                percentage_change=percentage_change
            )
            waves.append(wave)
            
        return waves
    
    def validate_impulse_wave_rules(self, waves: List[Wave]) -> Dict[str, bool]:
        """
        Validate core Elliott Wave rules for impulse patterns (5 waves)
        
        Returns:
            Dictionary of rule validation results
        """
        if len(waves) < 5:
            return {"insufficient_waves": False}
            
        results = {}
        
        # Extract waves 1, 3, 5 for comparison
        wave1 = waves[0]
        wave2 = waves[1]
        wave3 = waves[2]
        wave4 = waves[3]
        wave5 = waves[4]
        
        # Rule 1: Wave 3 is never the shortest of waves 1, 3, 5
        wave_lengths = [wave1.length, wave3.length, wave5.length]
        results["wave3_not_shortest"] = wave3.length != min(wave_lengths)
        
        # Rule 2: Wave 2 cannot retrace more than 100% of Wave 1
        wave2_retrace = abs(wave2.price_change) / wave1.length
        results["wave2_retrace_valid"] = wave2_retrace <= 1.0
        
        # Rule 3: Wave 4 cannot overlap into Wave 1 territory
        if wave1.price_change > 0:  # Bullish wave 1
            wave1_end = wave1.end_point.price
            wave4_low = min(wave4.start_point.price, wave4.end_point.price)
            results["no_wave4_overlap"] = wave4_low > wave1_end
        else:  # Bearish wave 1
            wave1_end = wave1.end_point.price
            wave4_high = max(wave4.start_point.price, wave4.end_point.price)
            results["no_wave4_overlap"] = wave4_high < wave1_end
            
        return results
    
    def validate_corrective_wave_rules(self, waves: List[Wave]) -> Dict[str, bool]:
        """
        Validate core Elliott Wave rules for corrective patterns (3 waves: A-B-C)
        
        Returns:
            Dictionary of rule validation results
        """
        if len(waves) < 3:
            return {"insufficient_waves": False}
            
        results = {}
        
        waveA = waves[0]
        waveB = waves[1]
        waveC = waves[2]
        
        # Rule 1: Wave B typically retraces 38.2-78.6% of Wave A (rarely exceeds 100%)
        waveB_retrace = abs(waveB.price_change) / waveA.length
        results["waveB_retrace_valid"] = waveB_retrace <= 1.0  # Should not exceed 100%
        
        # Rule 2: Wave C length should be related to Wave A by Fibonacci ratios
        # Common ratios: 1.0x, 1.618x, or 2.618x Wave A
        waveC_ratio = waveC.length / waveA.length
        results["waveC_ratio_valid"] = 0.5 <= waveC_ratio <= 3.0  # Reasonable range
        
        return results
    
    def calculate_fibonacci_ratios(self, waves: List[Wave]) -> Dict[str, float]:
        """Calculate Fibonacci ratios between waves"""
        ratios = {}
        
        if len(waves) == 5:  # Impulse wave ratios
            wave1 = waves[0]
            wave2 = waves[1]
            wave3 = waves[2]
            wave4 = waves[3]
            wave5 = waves[4]
            
            # Wave relationships
            if wave1.length > 0:
                ratios["wave3_to_wave1"] = wave3.length / wave1.length
                ratios["wave5_to_wave1"] = wave5.length / wave1.length
                
            if wave3.length > 0:
                ratios["wave4_to_wave3"] = abs(wave4.price_change) / wave3.length
                
            # Retracement ratios
            ratios["wave2_retrace"] = abs(wave2.price_change) / wave1.length
            ratios["wave4_retrace"] = abs(wave4.price_change) / wave3.length
            
        elif len(waves) == 3:  # Corrective wave ratios
            waveA = waves[0]
            waveB = waves[1] 
            waveC = waves[2]
            
            # Corrective wave relationships
            if waveA.length > 0:
                ratios["waveC_to_waveA"] = waveC.length / waveA.length
                ratios["waveB_retrace"] = abs(waveB.price_change) / waveA.length
        
        return ratios
    
    def determine_pattern_type_and_direction(self, waves: List[Wave]) -> Tuple[WaveType, TrendDirection]:
        """Determine if pattern is impulse/corrective and bullish/bearish"""
        if len(waves) == 5:
            pattern_type = WaveType.IMPULSE
        elif len(waves) == 3:
            pattern_type = WaveType.CORRECTIVE
        else:
            pattern_type = WaveType.UNKNOWN
            
        # Determine overall direction
        if len(waves) >= 2:
            total_change = waves[-1].end_point.price - waves[0].start_point.price
            direction = TrendDirection.BULLISH if total_change > 0 else TrendDirection.BEARISH
        else:
            direction = TrendDirection.BULLISH
            
        return pattern_type, direction
    
    def analyze_elliott_wave(self, symbol: str, timeframe: str, limit: int = 150, 
                           zigzag_threshold: float = 5.0) -> Optional[ElliottWavePattern]:
        """
        Complete Elliott Wave analysis for given symbol and timeframe
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Time interval (e.g., '1h', '4h', '1d')
            limit: Number of candles to analyze
            zigzag_threshold: ZigZag threshold percentage for swing detection
            
        Returns:
            ElliottWavePattern object or None if analysis fails
        """
        try:
            # Fetch OHLCV data
            ohlcv_data = self.data_collector.fetch_ohlcv_data(symbol, timeframe, limit)
            if ohlcv_data is None or len(ohlcv_data) < 10:
                print("Insufficient data")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Detect swing points
            swings = self.detect_zigzag_swings(df, zigzag_threshold)
            if len(swings) < 5:
                return None
                
            # Try to identify corrective pattern (3 waves) or impulse pattern (5 waves)
            if len(swings) >= 6:  # Need 6 swings for 5 waves
                waves = self.create_waves_from_swings(swings[-6:])  # 6 swings = 5 waves
                if len(waves) >= 5:
                    waves = waves[-5:]  # Take last 5 for impulse analysis
                else:
                    return None
            elif len(swings) >= 4:  # Try 3-wave corrective pattern
                waves = self.create_waves_from_swings(swings[-4:])  # 4 swings = 3 waves
                if len(waves) >= 3:
                    waves = waves[-3:]  # Take last 3 for corrective analysis
                else:
                    return None
            else:
                return None
                
            
            # Validate Elliott Wave rules based on pattern type
            if len(waves) == 5:
                validation_results = self.validate_impulse_wave_rules(waves)
            else:
                validation_results = self.validate_corrective_wave_rules(waves)
            is_valid = all(validation_results.values())
            
            # Calculate Fibonacci ratios
            fibonacci_ratios = self.calculate_fibonacci_ratios(waves)
            
            # Determine pattern type and direction
            pattern_type, trend_direction = self.determine_pattern_type_and_direction(waves)
            
            return ElliottWavePattern(
                symbol=symbol,
                timeframe=timeframe,
                pattern_type=pattern_type,
                trend_direction=trend_direction,
                waves=waves,
                is_valid=is_valid,
                validation_results=validation_results,
                fibonacci_ratios=fibonacci_ratios
            )
            
        except Exception as e:
            import traceback
            print(f"Error in Elliott Wave analysis: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def format_analysis_output(self, pattern: ElliottWavePattern) -> str:
        """Format Elliott Wave analysis results for terminal output"""
        if not pattern:
            return "No Elliott Wave pattern detected"
            
        output = []
        output.append("[ELLIOTT WAVE STRUCTURE]")
        output.append(f"Symbol: {pattern.symbol} | TF: {pattern.timeframe} | Pattern: {pattern.pattern_type.value} ({pattern.trend_direction.value})")
        
        # Core rules validation
        status = "✅ VALID" if pattern.is_valid else "❌ INVALID"
        rule_details = []
        if "wave3_not_shortest" in pattern.validation_results:
            rule_details.append("Wave 3 longest" if pattern.validation_results["wave3_not_shortest"] else "Wave 3 shortest")
        if "no_wave4_overlap" in pattern.validation_results:
            rule_details.append("no overlaps" if pattern.validation_results["no_wave4_overlap"] else "overlap detected")
            
        output.append(f"Core Rules: {status} ({', '.join(rule_details)})")
        
        # Wave details
        output.append("Waves:")
        for i, wave in enumerate(pattern.waves):
            if pattern.pattern_type == WaveType.IMPULSE:
                wave_label = str(i + 1)
            else:  # Corrective
                wave_labels = ['A', 'B', 'C']
                wave_label = wave_labels[i] if i < len(wave_labels) else str(i + 1)
                
            start_price = wave.start_point.price
            end_price = wave.end_point.price
            points = abs(wave.price_change)
            
            # Add retracement info
            retrace_info = ""
            validation_icon = "✅"
            
            if pattern.pattern_type == WaveType.IMPULSE:
                if (i + 1) == 2 and "wave2_retrace" in pattern.fibonacci_ratios:
                    retrace_pct = pattern.fibonacci_ratios["wave2_retrace"] * 100
                    retrace_info = f" ({retrace_pct:.1f}% retrace)"
                    validation_icon = "✅" if retrace_pct <= 100 else "❌"
                elif (i + 1) == 4 and "wave4_retrace" in pattern.fibonacci_ratios:
                    retrace_pct = pattern.fibonacci_ratios["wave4_retrace"] * 100
                    retrace_info = f" ({retrace_pct:.1f}% retrace)"
                    validation_icon = "✅" if retrace_pct <= 100 else "❌"
            else:  # Corrective
                if wave_label == 'B' and "waveB_retrace" in pattern.fibonacci_ratios:
                    retrace_pct = pattern.fibonacci_ratios["waveB_retrace"] * 100
                    retrace_info = f" ({retrace_pct:.1f}% retrace)"
                    validation_icon = "✅" if retrace_pct <= 100 else "❌"
            
            output.append(f"{wave_label}: {start_price:.2f} → {end_price:.2f} ({points:.0f}pts){retrace_info} {validation_icon}")
        
        # Fibonacci ratios
        if pattern.fibonacci_ratios:
            output.append("Fibonacci Ratios:")
            for ratio_name, ratio_value in pattern.fibonacci_ratios.items():
                if "retrace" not in ratio_name.lower():
                    formatted_name = ratio_name.replace('_', ' ').replace('wave', 'Wave').replace('to', 'to')
                    output.append(f"  {formatted_name}: {ratio_value:.3f}")
        
        return "\n".join(output)
    
    def analyze(self, symbol: str, timeframe: str, limit: int = 150, zigzag_threshold: float = 5.0):
        """
        Perform Elliott Wave analysis and print results
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Time interval (e.g., '1h', '4h', '1d')
            limit: Number of candles to analyze
            zigzag_threshold: ZigZag threshold percentage for swing detection
        """
        pattern = self.analyze_elliott_wave(symbol, timeframe, limit, zigzag_threshold)
        if pattern:
            print(self.format_analysis_output(pattern))
        else:
            print(f"No valid Elliott Wave pattern found for {symbol} on {timeframe}")


def parse_command(command: str) -> Tuple[str, str, int, float]:
    """
    Parse terminal command: elliott s=BTC/USDT t=4h l=150 zz=5.0
    
    Args:
        command: Command string
        
    Returns:
        Tuple of (symbol, timeframe, limit, zigzag_threshold)
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'elliott':
        raise ValueError("Invalid command format. Use: elliott s=SYMBOL t=TIMEFRAME l=LIMIT zz=THRESHOLD")
    
    symbol = None
    timeframe = '4h'  # default
    limit = 150  # default
    zigzag_threshold = 5.0  # default
    
    for part in parts[1:]:
        if part.startswith('s='):
            symbol = part[2:]
        elif part.startswith('t='):
            timeframe = part[2:]
        elif part.startswith('l='):
            try:
                limit = int(part[2:])
            except ValueError:
                raise ValueError(f"Invalid limit value: {part[2:]}")
        elif part.startswith('zz='):
            try:
                zigzag_threshold = float(part[2:])
            except ValueError:
                raise ValueError(f"Invalid zigzag threshold value: {part[2:]}")
    
    if symbol is None:
        raise ValueError("Symbol (s=) is required")
    
    return symbol, timeframe, limit, zigzag_threshold


def main():
    """Command line interface for Elliott Wave analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Elliott Wave Pattern Analysis')
    parser.add_argument('--symbol', '-s', default='BTC/USDT', help='Trading symbol (default: BTC/USDT)')
    parser.add_argument('--timeframe', '-t', default='4h', help='Timeframe (default: 4h)')
    parser.add_argument('--limit', '-l', type=int, default=150, help='Candle limit (default: 150)')
    parser.add_argument('--zigzag', '-zz', type=float, default=5.0, help='ZigZag threshold % (default: 5.0)')
    
    args = parser.parse_args()
    
    analyzer = ElliottWaveAnalyzer()
    pattern = analyzer.analyze_elliott_wave(args.symbol, args.timeframe, args.limit, args.zigzag)
    
    if pattern:
        print(analyzer.format_analysis_output(pattern))
    else:
        print(f"No valid Elliott Wave pattern found for {args.symbol} on {args.timeframe}")

if __name__ == "__main__":
    main()