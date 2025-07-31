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
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import get_data_collector
from trend.output_formatter import OutputFormatter

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
    
    def calculate_pattern_confidence(self, pattern: ElliottWavePattern) -> int:
        """Calculate confidence score (0-100) based on pattern quality"""
        if not pattern:
            return 0
            
        confidence = 0
        max_score = 100
        
        # Base score for having a pattern
        confidence += 20
        
        # Validation rules compliance (40 points max)
        if pattern.validation_results:
            valid_rules = sum(1 for valid in pattern.validation_results.values() if valid)
            total_rules = len(pattern.validation_results)
            if total_rules > 0:
                confidence += int((valid_rules / total_rules) * 40)
        
        # Fibonacci ratio quality (25 points max)
        if pattern.fibonacci_ratios:
            fib_score = 0
            ideal_ratios = [0.618, 1.0, 1.618, 2.618]
            
            for ratio_key, ratio_val in pattern.fibonacci_ratios.items():
                if 'retrace' not in ratio_key.lower():  # Skip retracement ratios
                    # Check how close to ideal Fibonacci ratios
                    min_diff = min(abs(ratio_val - ideal) for ideal in ideal_ratios)
                    if min_diff < 0.1:
                        fib_score += 10
                    elif min_diff < 0.3:
                        fib_score += 5
            
            confidence += min(fib_score, 25)
        
        # Wave structure quality (15 points max)
        if len(pattern.waves) >= 3:
            # Check for alternation in corrective waves
            if pattern.pattern_type == WaveType.CORRECTIVE and len(pattern.waves) >= 3:
                confidence += 8
            elif pattern.pattern_type == WaveType.IMPULSE and len(pattern.waves) >= 5:
                confidence += 15
        
        return min(confidence, max_score)
    
    def get_validation_details(self, pattern: ElliottWavePattern) -> str:
        """Get detailed validation failure explanations"""
        if not pattern or not pattern.validation_results:
            return "No validation data available"
            
        details = []
        
        for rule, is_valid in pattern.validation_results.items():
            if not is_valid:
                if rule == "wave3_not_shortest":
                    details.append("Wave 3 is shortest (violates core rule)")
                elif rule == "wave2_retrace_valid":
                    details.append("Wave 2 retraces >100% of Wave 1")
                elif rule == "no_wave4_overlap":
                    details.append("Wave 4 overlaps with Wave 1 territory")
                elif rule == "waveB_retrace_valid":
                    details.append("Wave B retraces >100% of Wave A")
                elif rule == "waveC_ratio_valid":
                    details.append("Wave C/A ratio outside reasonable range")
                elif rule == "insufficient_waves":
                    details.append("Insufficient waves for pattern type")
        
        return "; ".join(details) if details else "All rules validated"
    
    def get_wave_structure_summary(self, pattern: ElliottWavePattern) -> Dict[str, str]:
        """Get detailed wave structure information"""
        if not pattern or not pattern.waves:
            return {}
            
        structure = {}
        
        # Overall pattern move
        start_price = pattern.waves[0].start_point.price
        end_price = pattern.waves[-1].end_point.price
        total_move = ((end_price - start_price) / start_price) * 100
        structure["TOTAL_MOVE"] = f"{total_move:+.1f}%"
        
        # Largest wave
        largest_wave = max(pattern.waves, key=lambda w: abs(w.percentage_change))
        structure["LARGEST_WAVE"] = f"W{largest_wave.number} ({largest_wave.percentage_change:+.1f}%)"
        
        # Price levels
        highest_price = max(max(w.start_point.price, w.end_point.price) for w in pattern.waves)
        lowest_price = min(min(w.start_point.price, w.end_point.price) for w in pattern.waves)
        structure["RANGE"] = f"{lowest_price:.2f}-{highest_price:.2f}"
        
        return structure
    
    def get_prediction_targets(self, pattern: ElliottWavePattern) -> Dict[str, str]:
        """Calculate potential target levels based on Elliott Wave theory"""
        if not pattern or not pattern.waves:
            return {}
            
        targets = {}
        current_price = pattern.waves[-1].end_point.price
        
        if pattern.pattern_type == WaveType.IMPULSE and len(pattern.waves) >= 5:
            # Pattern completed, look for corrective phase
            wave1_length = abs(pattern.waves[0].price_change)
            
            # Common retracement levels for correction
            if pattern.trend_direction == TrendDirection.BULLISH:
                # Expect bearish correction
                correction_38 = current_price - (wave1_length * 0.382)
                correction_62 = current_price - (wave1_length * 0.618)
                targets["CORRECTION_38"] = f"{correction_38:.2f}"
                targets["CORRECTION_62"] = f"{correction_62:.2f}"
            else:
                # Expect bullish correction
                correction_38 = current_price + (wave1_length * 0.382)
                correction_62 = current_price + (wave1_length * 0.618)
                targets["CORRECTION_38"] = f"{correction_38:.2f}"
                targets["CORRECTION_62"] = f"{correction_62:.2f}"
                
        elif pattern.pattern_type == WaveType.CORRECTIVE and len(pattern.waves) >= 3:
            # Corrective pattern completing, look for next impulse
            waveA_length = abs(pattern.waves[0].price_change)
            
            if pattern.trend_direction == TrendDirection.BEARISH:
                # Expect bullish impulse after bearish correction
                target_100 = current_price + waveA_length
                target_162 = current_price + (waveA_length * 1.618)
                targets["IMPULSE_100"] = f"{target_100:.2f}"
                targets["IMPULSE_162"] = f"{target_162:.2f}"
            else:
                # Expect bearish impulse after bullish correction
                target_100 = current_price - waveA_length
                target_162 = current_price - (waveA_length * 1.618)
                targets["IMPULSE_100"] = f"{target_100:.2f}"
                targets["IMPULSE_162"] = f"{target_162:.2f}"
        
        return targets
    
    def format_analysis_output(self, pattern: ElliottWavePattern, timestamp: int) -> str:
        """Format Elliott Wave analysis results with the new template format"""
        if not pattern:
            return "No Elliott Wave pattern detected"
        
        # Calculate confidence score
        confidence = self.calculate_pattern_confidence(pattern)
        
        # Get current price from latest wave
        current_price = pattern.waves[-1].end_point.price if pattern.waves else None
        
        # Header line
        output = f"Symbol: {pattern.symbol} | Timeframe: {pattern.timeframe.upper()}\n"
        
        # Pattern type with trend direction
        pattern_display = f"{pattern.pattern_type.value} ({pattern.trend_direction.value})"
        output += f"Pattern Type: {pattern_display}\n"
        
        # Wave count display
        if pattern.pattern_type == WaveType.IMPULSE and len(pattern.waves) >= 5:
            output += "Wave Count: 1 → 2 → 3 → 4 → 5\n"
        elif pattern.pattern_type == WaveType.CORRECTIVE and len(pattern.waves) >= 3:
            output += "Wave Count: A → B → C\n"
        else:
            wave_nums = " → ".join(str(i+1) for i in range(len(pattern.waves)))
            output += f"Wave Count: {wave_nums}\n"
        
        # Status validation
        status = "✅ VALID" if pattern.is_valid else "❌ INVALID"
        output += f"Status: {status}\n"
        
        # Rule checks
        output += "Rule Checks:\n"
        if pattern.validation_results:
            for rule, is_valid in pattern.validation_results.items():
                check_mark = "✅" if is_valid else "❌"
                if rule == "wave3_not_shortest":
                    output += f"- Wave 3 not shortest: {check_mark}\n"
                elif rule == "no_wave4_overlap":
                    output += f"- Wave 4 does not overlap Wave 1: {check_mark}\n"
                elif rule == "wave2_retrace_valid":
                    retrace_pct = pattern.fibonacci_ratios.get('wave2_retrace', 0) * 100
                    output += f"- Wave 2 retracement: {retrace_pct:.1f}% {check_mark}\n"
                elif rule == "waveB_retrace_valid":
                    retrace_pct = pattern.fibonacci_ratios.get('waveB_retrace', 0) * 100
                    output += f"- Wave B retracement: {retrace_pct:.1f}% {check_mark}\n"
        
        # Add wave extension info if available
        if pattern.fibonacci_ratios:
            if "wave3_to_wave1" in pattern.fibonacci_ratios:
                w3_ratio = pattern.fibonacci_ratios["wave3_to_wave1"]
                check_mark = "✅" if w3_ratio > 1.0 else "❌"
                output += f"- Wave 3 extension: {w3_ratio:.2f}x of Wave 1 {check_mark}\n"
            if "waveC_to_waveA" in pattern.fibonacci_ratios:
                wc_ratio = pattern.fibonacci_ratios["waveC_to_waveA"]
                check_mark = "✅" if 0.8 <= wc_ratio <= 2.0 else "❌"
                output += f"- Wave C extension: {wc_ratio:.2f}x of Wave A {check_mark}\n"
        
        output += "\n"
        
        # Current status and projections
        if pattern.pattern_type == WaveType.IMPULSE:
            if len(pattern.waves) >= 5:
                output += "🎯 Current: Wave 5 completed\n"
                # Calculate wave 5 target based on fibonacci
                if len(pattern.waves) >= 5:
                    wave5_end = pattern.waves[4].end_point.price
                    output += f"🔮 Projection: Wave 5 Target = {wave5_end:.3f}\n"
            else:
                current_wave = len(pattern.waves)
                output += f"🎯 Current: Wave {current_wave} in progress\n"
                # Calculate next wave target
                if current_wave < 5 and len(pattern.waves) >= 1:
                    # Simple projection based on previous wave
                    next_target = current_price * 1.05 if pattern.trend_direction == TrendDirection.BULLISH else current_price * 0.95
                    output += f"🔮 Projection: Wave {current_wave + 1} Target = {next_target:.3f}\n"
        else:  # Corrective
            if len(pattern.waves) >= 3:
                output += "🎯 Current: Wave C completed\n"
                wave_c_end = pattern.waves[2].end_point.price
                output += f"🔮 Projection: Wave C Target = {wave_c_end:.3f}\n"
            else:
                current_wave_letter = chr(ord('A') + len(pattern.waves) - 1)
                output += f"🎯 Current: Wave {current_wave_letter} in progress\n"
        
        # Entry zone (use last significant low/high)
        if pattern.waves and len(pattern.waves) >= 2:
            if pattern.trend_direction == TrendDirection.BULLISH:
                entry_level = min(w.start_point.price for w in pattern.waves[-2:])
                output += f"📍 Entry Zone: After Wave {len(pattern.waves)-1} low confirmed ({entry_level:.3f})\n"
            else:
                entry_level = max(w.start_point.price for w in pattern.waves[-2:])
                output += f"📍 Entry Zone: After Wave {len(pattern.waves)-1} high confirmed ({entry_level:.3f})\n"
        
        # Signal and confidence
        if confidence >= 70:
            if pattern.pattern_type == WaveType.IMPULSE:
                signal = "BUY" if pattern.trend_direction == TrendDirection.BULLISH else "SELL"
            else:  # Corrective - expect reversal
                signal = "SELL" if pattern.trend_direction == TrendDirection.BULLISH else "BUY"
            conf_level = "HIGH"
        elif confidence >= 50:
            signal = "BUY" if pattern.trend_direction == TrendDirection.BULLISH else "SELL"
            conf_level = "MEDIUM"
        else:
            signal = "HOLD"
            conf_level = "LOW"
        
        output += f"🚦 Signal: {signal} | Confidence: {conf_level}"
        
        return output
    
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
            # Get current timestamp from the last data point
            try:
                ohlcv_data = self.data_collector.fetch_ohlcv_data(symbol, timeframe, 1)
                current_timestamp = ohlcv_data[0][0] if ohlcv_data else int(datetime.now().timestamp() * 1000)
            except:
                current_timestamp = int(datetime.now().timestamp() * 1000)
                
            print(self.format_analysis_output(pattern, current_timestamp))
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
        # Get current timestamp for output
        try:
            ohlcv_data = analyzer.data_collector.fetch_ohlcv_data(args.symbol, args.timeframe, 1)
            current_timestamp = ohlcv_data[0][0] if ohlcv_data else int(datetime.now().timestamp() * 1000)
        except:
            current_timestamp = int(datetime.now().timestamp() * 1000)
            
        print(analyzer.format_analysis_output(pattern, current_timestamp))
    else:
        print(f"No valid Elliott Wave pattern found for {args.symbol} on {args.timeframe}")

if __name__ == "__main__":
    main()