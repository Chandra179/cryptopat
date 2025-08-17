"""
Signal Aggregation for Cryptocurrency Technical Analysis
Uses Majority Voting + Weighted Ensemble approach based on research
"""

from typing import Dict, Optional
from datetime import datetime


class SignalAggregator:
    def __init__(self):
        self.indicator_results = {}
        self.signal_mappings = {}
        
    def add_indicator_result(self, indicator_name: str, result: Dict, signal_mapping: Dict):
        """Store individual indicator result with its signal mapping"""
        self.indicator_results[indicator_name] = result
        self.signal_mappings[indicator_name] = signal_mapping
        
    def get_numerical_signal(self, indicator_name: str) -> Optional[float]:
        """Extract numerical signal value from indicator result"""
        if indicator_name not in self.indicator_results:
            return None
            
        result = self.indicator_results[indicator_name]
        signal_enum = result.get('signal', 'neutral')
        mapping = self.signal_mappings.get(indicator_name, {})
        
        return mapping.get(signal_enum, 0.0)
    
    def majority_vote(self) -> str:
        """Determine directional consensus via majority voting"""
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for indicator_name in self.indicator_results:
            signal_value = self.get_numerical_signal(indicator_name)
            if signal_value is None:
                continue
                
            if signal_value > 0.1:
                bullish_count += 1
            elif signal_value < -0.1:
                bearish_count += 1
            else:
                neutral_count += 1
        
        if bullish_count > bearish_count and bullish_count > neutral_count:
            return "bullish"
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            return "bearish"
        else:
            return "neutral"
    
    def weighted_average_signal(self) -> float:
        """Calculate weighted average of all numerical signals"""
        total_signal = 0.0
        total_weight = 0.0
        
        for indicator_name in self.indicator_results:
            signal_value = self.get_numerical_signal(indicator_name)
            if signal_value is None:
                continue
                
            weight = 1.0  # Equal weight for now, can be enhanced later
            total_signal += signal_value * weight
            total_weight += weight
        
        return total_signal / total_weight if total_weight > 0 else 0.0
    
    def get_consensus_strength(self) -> float:
        """Calculate how many indicators agree on direction (0-1)"""
        if not self.indicator_results:
            return 0.0
            
        majority_direction = self.majority_vote()
        agreeing_count = 0
        total_count = 0
        
        for indicator_name in self.indicator_results:
            signal_value = self.get_numerical_signal(indicator_name)
            if signal_value is None:
                continue
                
            total_count += 1
            
            if majority_direction == "bullish" and signal_value > 0.1:
                agreeing_count += 1
            elif majority_direction == "bearish" and signal_value < -0.1:
                agreeing_count += 1
            elif majority_direction == "neutral" and -0.1 <= signal_value <= 0.1:
                agreeing_count += 1
        
        return agreeing_count / total_count if total_count > 0 else 0.0
    
    def generate_summary(self) -> Dict:
        """Generate complete signal aggregation summary"""
        majority_signal = self.majority_vote()
        weighted_signal = self.weighted_average_signal()
        consensus_strength = self.get_consensus_strength()
        
        # Signal strength based on weighted average
        if abs(weighted_signal) >= 0.7:
            signal_strength = "strong"
        elif abs(weighted_signal) >= 0.3:
            signal_strength = "moderate"
        else:
            signal_strength = "weak"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "majority_vote": majority_signal,
            "weighted_signal": round(weighted_signal, 3),
            "signal_strength": signal_strength,
            "consensus_strength": round(consensus_strength, 3),
            "total_indicators": len(self.indicator_results),
            "individual_signals": {
                name: self.get_numerical_signal(name) 
                for name in self.indicator_results
            }
        }
    
    def clear_results(self):
        """Clear stored results for new analysis"""
        self.indicator_results.clear()
        self.signal_mappings.clear()