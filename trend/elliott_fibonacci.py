import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from data import get_data_collector
from .fibonacci import FibonacciCalculator, ZigZagDetector


class Wave(NamedTuple):
    """Elliott Wave structure"""
    start_idx: int
    end_idx: int
    start_price: float
    end_price: float
    wave_type: str  # '1', '2', '3', '4', '5', 'A', 'B', 'C'
    confidence: float


class ElliottWavePattern(NamedTuple):
    """Complete Elliott Wave pattern"""
    pattern_type: str  # 'impulse' or 'corrective'
    waves: List[Wave]
    current_wave: Optional[str]
    next_targets: Dict[str, float]
    confluence_score: float
    corrective_rules: Optional[Dict[str, bool]] = None


class ElliottFibonacciAnalyzer:
    """Elliott Wave analysis with Fibonacci confluence"""
    
    def __init__(self):
        self.data_collector = get_data_collector()
        self.fib_calc = FibonacciCalculator()
        self.zigzag = ZigZagDetector()
    
    def analyze(self, symbol: str, timeframe: str, limit: int = 150, 
               threshold: float = 4.0) -> Dict:
        """Main analysis function"""
        try:
            ohlcv_data = self.data_collector.fetch_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            if not ohlcv_data:
                return {"error": "No data available"}
            
            highs = np.array([candle[2] for candle in ohlcv_data])
            lows = np.array([candle[3] for candle in ohlcv_data])
            closes = np.array([candle[4] for candle in ohlcv_data])
            
            swing_points = self.zigzag.find_zigzag_points(highs, lows, threshold)
            
            if len(swing_points) < 5:
                return {"error": "Insufficient swing points for Elliott Wave analysis"}
            
            pattern = self._identify_elliott_pattern(swing_points, closes)
            
            if pattern:
                return self._format_output(pattern, symbol, timeframe)
            else:
                return {"error": "No clear Elliott Wave pattern identified"}
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _identify_elliott_pattern(self, swing_points: List[Tuple[int, float, str]], 
                                 closes: np.ndarray) -> Optional[ElliottWavePattern]:
        """Identify Elliott Wave patterns from swing points"""
        if len(swing_points) < 5:
            return None
        
        impulse_pattern = self._detect_impulse_pattern(swing_points)
        if impulse_pattern:
            return impulse_pattern
        
        corrective_pattern = self._detect_corrective_pattern(swing_points)
        return corrective_pattern
    
    def _detect_impulse_pattern(self, swing_points: List[Tuple[int, float, str]]) -> Optional[ElliottWavePattern]:
        """Detect 5-wave impulse pattern with Elliott Wave rules"""
        if len(swing_points) < 6:
            return None
        
        recent_points = swing_points[-6:]
        
        if not self._is_valid_impulse_sequence(recent_points):
            return None
        
        waves = []
        confidence_scores = []
        
        for i in range(5):
            start_point = recent_points[i]
            end_point = recent_points[i + 1]
            
            wave = Wave(
                start_idx=start_point[0],
                end_idx=end_point[0],
                start_price=start_point[1],
                end_price=end_point[1],
                wave_type=str(i + 1),
                confidence=0.0
            )
            
            confidence = self._calculate_wave_confidence(wave, waves, i + 1)
            wave = wave._replace(confidence=confidence)
            waves.append(wave)
            confidence_scores.append(confidence)
        
        if not self._validate_elliott_rules(waves):
            return None
        
        current_wave, next_targets = self._project_next_wave(waves)
        confluence_score = np.mean(confidence_scores)
        
        return ElliottWavePattern(
            pattern_type='impulse',
            waves=waves,
            current_wave=current_wave,
            next_targets=next_targets,
            confluence_score=confluence_score,
            corrective_rules=None
        )
    
    def _detect_corrective_pattern(self, swing_points: List[Tuple[int, float, str]]) -> Optional[ElliottWavePattern]:
        """Detect A-B-C corrective pattern"""
        if len(swing_points) < 4:
            return None
        
        recent_points = swing_points[-4:]
        
        if not self._is_valid_corrective_sequence(recent_points):
            return None
        
        waves = []
        wave_labels = ['A', 'B', 'C']
        
        for i in range(3):
            start_point = recent_points[i]
            end_point = recent_points[i + 1]
            
            wave = Wave(
                start_idx=start_point[0],
                end_idx=end_point[0],
                start_price=start_point[1],
                end_price=end_point[1],
                wave_type=wave_labels[i],
                confidence=self._calculate_corrective_confidence(start_point, end_point, i)
            )
            waves.append(wave)
        
        current_wave, next_targets = self._project_corrective_targets(waves)
        confluence_score = np.mean([w.confidence for w in waves])
        
        # Validate corrective rules and store results
        corrective_rules = self._validate_corrective_rules(waves)
        
        pattern = ElliottWavePattern(
            pattern_type='corrective',
            waves=waves,
            current_wave=current_wave,
            next_targets=next_targets,
            confluence_score=confluence_score,
            corrective_rules=corrective_rules
        )
        
        return pattern
    
    def _is_valid_impulse_sequence(self, points: List[Tuple[int, float, str]]) -> bool:
        """Check if swing points form a valid impulse sequence"""
        if len(points) < 6:
            return False
        
        expected_sequence = ['low', 'high', 'low', 'high', 'low', 'high'] if points[0][2] == 'low' else ['high', 'low', 'high', 'low', 'high', 'low']
        actual_sequence = [p[2] for p in points]
        
        return actual_sequence == expected_sequence
    
    def _is_valid_corrective_sequence(self, points: List[Tuple[int, float, str]]) -> bool:
        """Check if swing points form a valid corrective sequence"""
        if len(points) < 4:
            return False
        
        expected_sequence = ['high', 'low', 'high', 'low'] if points[0][2] == 'high' else ['low', 'high', 'low', 'high']
        actual_sequence = [p[2] for p in points]
        
        return actual_sequence == expected_sequence
    
    def _validate_elliott_rules(self, waves: List[Wave]) -> bool:
        """Validate Elliott Wave cardinal rules and guidelines"""
        if len(waves) < 5:
            return False
        
        wave1_length = abs(waves[0].end_price - waves[0].start_price)
        wave2_length = abs(waves[1].end_price - waves[1].start_price)
        wave3_length = abs(waves[2].end_price - waves[2].start_price)
        wave4_length = abs(waves[3].end_price - waves[3].start_price)
        wave5_length = abs(waves[4].end_price - waves[4].start_price)
        
        # Rule 1: Wave 3 cannot be the shortest
        if wave3_length < wave1_length and wave3_length < wave5_length:
            return False
        
        # Rule 2: Wave 2 cannot retrace more than 100% of Wave 1
        wave2_retrace = abs(waves[1].end_price - waves[0].start_price) / wave1_length
        if wave2_retrace > 1.0:
            return False
        
        # Rule 3: Wave 4 cannot overlap Wave 1 (in normal impulse)
        if waves[0].start_price < waves[0].end_price:  # Bullish
            if waves[3].end_price <= waves[0].end_price:
                return False
        else:  # Bearish
            if waves[3].end_price >= waves[0].end_price:
                return False
        
        # Guideline: Alternation between Wave 2 and Wave 4
        # This is a soft rule - waves should alternate in character
        wave2_retrace_pct = wave2_retrace
        wave4_retrace_pct = abs(waves[3].end_price - waves[2].end_price) / wave3_length
        
        # Check for alternation pattern (one sharp, one sideways)
        # Sharp correction: > 50% retracement, Sideways: < 50% retracement
        wave2_sharp = wave2_retrace_pct > 0.5
        wave4_sharp = wave4_retrace_pct > 0.5
        
        # Alternation guideline satisfied if waves have different characteristics
        alternation_satisfied = wave2_sharp != wave4_sharp
        
        return True  # Main rules satisfied, alternation is guideline
    
    def _validate_corrective_rules(self, waves: List[Wave]) -> Dict[str, bool]:
        """Validate A-B-C corrective wave rules"""
        rules = {}
        
        if len(waves) < 3:
            return {"insufficient_waves": False}
        
        wave_a_length = abs(waves[0].end_price - waves[0].start_price)
        wave_b_length = abs(waves[1].end_price - waves[1].start_price)
        wave_c_length = abs(waves[2].end_price - waves[2].start_price)
        
        # Rule 1: Wave B should retrace 38.2% to 78.6% of Wave A
        wave_b_retrace = wave_b_length / wave_a_length
        rules["wave_b_valid_retrace"] = 0.382 <= wave_b_retrace <= 0.786
        
        # Rule 2: Wave C should be at least equal to Wave A (typically 1.0 or 1.618x)
        wave_c_ratio = wave_c_length / wave_a_length
        rules["wave_c_adequate_length"] = wave_c_ratio >= 0.618
        rules["wave_c_fibonacci_ratio"] = (0.9 <= wave_c_ratio <= 1.1) or (1.5 <= wave_c_ratio <= 1.7)
        
        # Rule 3: Wave C should move in same direction as Wave A
        wave_a_direction = 1 if waves[0].end_price > waves[0].start_price else -1
        wave_c_direction = 1 if waves[2].end_price > waves[2].start_price else -1
        rules["wave_c_correct_direction"] = wave_a_direction == wave_c_direction
        
        return rules
    
    def _calculate_wave_confidence(self, wave: Wave, previous_waves: List[Wave], wave_num: int) -> float:
        """Calculate confidence score for each wave using Fibonacci ratios"""
        confidence = 0.5
        
        if wave_num == 2 and len(previous_waves) >= 1:
            # Wave 2 should retrace 50-78.6% of Wave 1
            wave1_length = abs(previous_waves[0].end_price - previous_waves[0].start_price)
            wave2_retrace = abs(wave.end_price - previous_waves[0].start_price) / wave1_length
            
            if 0.5 <= wave2_retrace <= 0.786:
                confidence += 0.3
            if 0.61 <= wave2_retrace <= 0.63:  # Golden ratio range
                confidence += 0.2
        
        elif wave_num == 3 and len(previous_waves) >= 2:
            # Wave 3 should be 1.618x Wave 1 (most common extension)
            wave1_length = abs(previous_waves[0].end_price - previous_waves[0].start_price)
            wave3_length = abs(wave.end_price - wave.start_price)
            wave3_ratio = wave3_length / wave1_length
            
            # Standard guideline: Wave 3 is often the longest and extends to 161.8%
            if 1.4 <= wave3_ratio <= 1.8:
                confidence += 0.3
            if 1.6 <= wave3_ratio <= 1.65:  # Close to 1.618 (golden ratio)
                confidence += 0.2
            # Additional confidence if Wave 3 is clearly the longest
            if wave3_length > wave1_length and (len(previous_waves) < 5 or wave3_length > abs(previous_waves[4].end_price - previous_waves[4].start_price) if len(previous_waves) >= 5 else True):
                confidence += 0.1
        
        elif wave_num == 4 and len(previous_waves) >= 3:
            # Wave 4 should retrace 23.6-38.2% of Wave 3
            wave3_length = abs(previous_waves[2].end_price - previous_waves[2].start_price)
            wave4_retrace = abs(wave.end_price - previous_waves[2].end_price) / wave3_length
            
            if 0.236 <= wave4_retrace <= 0.382:
                confidence += 0.3
        
        elif wave_num == 5 and len(previous_waves) >= 4:
            # Wave 5 should be 0.618x Wave 1 or equal to Wave 1
            wave1_length = abs(previous_waves[0].end_price - previous_waves[0].start_price)
            wave5_length = abs(wave.end_price - wave.start_price)
            wave5_ratio = wave5_length / wave1_length
            
            if 0.6 <= wave5_ratio <= 1.1:
                confidence += 0.2
            if 0.9 <= wave5_ratio <= 1.1:  # Equal waves
                confidence += 0.1
            if 0.6 <= wave5_ratio <= 0.65:  # 0.618 ratio
                confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _calculate_corrective_confidence(self, start_point: Tuple, end_point: Tuple, wave_num: int) -> float:
        """Calculate confidence for corrective waves"""
        return 0.7  # Simplified for now
    
    def _project_next_wave(self, waves: List[Wave]) -> Tuple[Optional[str], Dict[str, float]]:
        """Project next wave targets with enhanced logic"""
        if len(waves) < 5:
            return None, {}
        
        current_wave = "5"
        
        # Project potential Wave 5 targets if in Wave 4
        if len(waves) >= 4:
            wave1_length = abs(waves[0].end_price - waves[0].start_price)
            wave3_length = abs(waves[2].end_price - waves[2].start_price)
            wave1_direction = 1 if waves[0].end_price > waves[0].start_price else -1
            wave3_direction = 1 if waves[2].end_price > waves[2].start_price else -1
            
            targets = {}
            
            # Standard Fibonacci targets for Wave 5
            # 0.618 of Wave 1
            targets["Wave_5_0.618"] = waves[3].end_price + (wave1_length * 0.618 * wave1_direction)
            
            # Equal to Wave 1 (common when Wave 3 is extended)
            targets["Wave_5_1.0"] = waves[3].end_price + (wave1_length * wave1_direction)
            
            # 0.382 of Wave 3 (when Wave 3 is very extended)
            targets["Wave_5_0.382_W3"] = waves[3].end_price + (wave3_length * 0.382 * wave3_direction)
            
            # Enhanced logic: If Wave 3 is extended (>1.618x Wave 1), Wave 5 often equals Wave 1
            wave3_to_wave1_ratio = wave3_length / wave1_length
            if wave3_to_wave1_ratio > 1.6:
                # When Wave 3 is extended, Wave 5 typically equals Wave 1
                targets["Wave_5_Equal_W1"] = waves[3].end_price + (wave1_length * wave1_direction)
            else:
                # When Wave 3 is not extended, Wave 5 can extend
                targets["Wave_5_1.618"] = waves[3].end_price + (wave1_length * 1.618 * wave1_direction)
            
            return current_wave, targets
        
        return current_wave, {}
    
    def _project_corrective_targets(self, waves: List[Wave]) -> Tuple[Optional[str], Dict[str, float]]:
        """Project corrective wave targets"""
        if len(waves) < 2:
            return None, {}
        
        wave_a_length = abs(waves[0].end_price - waves[0].start_price)
        wave_b_end = waves[1].end_price
        
        # Determine Wave A direction to project Wave C in same direction
        wave_a_direction = 1 if waves[0].end_price > waves[0].start_price else -1
        
        targets = {}
        # Standard Fibonacci targets for Wave C
        targets["C_1.0"] = wave_b_end + (wave_a_length * 1.0 * wave_a_direction)
        targets["C_1.618"] = wave_b_end + (wave_a_length * 1.618 * wave_a_direction)
        
        return "C", targets
    
    def _format_output(self, pattern: ElliottWavePattern, symbol: str, timeframe: str) -> Dict:
        """Format analysis output"""
        output = {
            "symbol": symbol,
            "timeframe": timeframe,
            "pattern_type": pattern.pattern_type.title(),
            "waves": [],
            "current_wave": pattern.current_wave,
            "next_targets": pattern.next_targets,
            "confluence_score": round(pattern.confluence_score, 2),
            "confluence_strength": self._get_confluence_strength(pattern.confluence_score)
        }
        
        # Add corrective rules validation if available
        if pattern.corrective_rules:
            output["corrective_rules"] = pattern.corrective_rules
        
        for wave in pattern.waves:
            output["waves"].append({
                "wave": wave.wave_type,
                "start_price": round(wave.start_price, 4),
                "end_price": round(wave.end_price, 4),
                "length": round(abs(wave.end_price - wave.start_price), 4),
                "confidence": round(wave.confidence, 2)
            })
        
        return output
    
    def _get_confluence_strength(self, score: float) -> str:
        """Get confluence strength description"""
        if score >= 0.8:
            return "Very Strong"
        elif score >= 0.7:
            return "Strong"
        elif score >= 0.6:
            return "Moderate"
        else:
            return "Weak"


def parse_command(command: str) -> Tuple[str, str, int, float]:
    """
    Parse terminal command: elliott_fibonacci s=XRP/USDT t=4h l=150 zz=4
    
    Args:
        command: Command string
        
    Returns:
        Tuple of (symbol, timeframe, limit, zigzag_threshold)
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'elliott_fibonacci':
        raise ValueError("Invalid command format. Use: elliott_fibonacci s=SYMBOL t=TIMEFRAME l=LIMIT zz=THRESHOLD")
    
    symbol = None
    timeframe = None
    limit = 150  # default
    zigzag_threshold = 4.0  # default
    
    for part in parts[1:]:
        if '=' in part:
            key, value = part.split('=', 1)
            if key == 's':
                symbol = value
            elif key == 't':
                timeframe = value
            elif key == 'l':
                try:
                    limit = int(value)
                except ValueError:
                    raise ValueError(f"Invalid limit value: {value}")
            elif key == 'zz':
                try:
                    zigzag_threshold = float(value)
                except ValueError:
                    raise ValueError(f"Invalid zigzag threshold value: {value}")
    
    if not symbol:
        raise ValueError("Symbol (s=) is required")
    if not timeframe:
        raise ValueError("Timeframe (t=) is required")
    
    return symbol, timeframe, limit, zigzag_threshold