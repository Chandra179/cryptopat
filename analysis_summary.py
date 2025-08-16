"""
Analysis Summary Generator

Collects and summarizes results from all technical indicators in memory.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class IndicatorResult:
    name: str
    signal: str
    value: Optional[float] = None
    strength: str = "medium"
    support: Optional[float] = None
    resistance: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class AnalysisSummarizer:
    def __init__(self):
        self.results: List[IndicatorResult] = []
        
        self.indicator_weights = {
            'Ichimoku Cloud': 1.0,
            'SuperTrend': 0.9,
            'VWAP': 0.8,
            'EMA 20/50': 0.8,
            'MACD': 0.7,
            'Bollinger Bands': 0.7,
            'Keltner Channel': 0.6,
            'RSI': 0.6,
            'Parabolic SAR': 0.5,
            'Donchian Channel': 0.5,
            'Pivot Point': 0.4,
            'Chaikin Money Flow': 0.4,
            'OBV': 0.3,
            'Renko Chart': 0.2
        }
        
        self.signal_mapping = {
            'strong_bullish': 2,
            'bullish_breakout': 1.5,
            'bullish': 1,
            'neutral': 0,
            'hold': 0,
            'weakening_bullish': 0.5,
            'bearish': -1,
            'distribution': -1,
            'strong_bearish': -2
        }
        
        self.sentiment_phrases = {
            'strong_bullish': 'a strongly bullish outlook',
            'bullish': 'a cautiously bullish stance',
            'neutral': 'mixed signals with no clear direction',
            'bearish': 'a bearish bias',
            'strong_bearish': 'a strongly bearish outlook'
        }

    def add_result(self, result: IndicatorResult):
        """Add an indicator result to the collection."""
        self.results.append(result)
        
    def clear_results(self):
        """Clear all stored results."""
        self.results.clear()

    def extract_support_resistance(self) -> Tuple[List[float], List[float]]:
        """Extract support and resistance levels from indicator results."""
        support_levels = []
        resistance_levels = []
        
        for result in self.results:
            if result.support:
                support_levels.append(result.support)
            if result.resistance:
                resistance_levels.append(result.resistance)
                
            if result.metadata:
                if 'support_levels' in result.metadata:
                    support_levels.extend(result.metadata['support_levels'])
                if 'resistance_levels' in result.metadata:
                    resistance_levels.extend(result.metadata['resistance_levels'])
        
        return support_levels, resistance_levels

    def aggregate_signals(self) -> Dict:
        """Aggregate and classify all indicator signals."""
        weighted_score = 0
        total_weight = 0
        signal_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        
        for result in self.results:
            weight = self.indicator_weights.get(result.name, 0.3)
            signal_value = self.signal_mapping.get(result.signal, 0)
            
            weighted_score += signal_value * weight
            total_weight += weight
            
            if signal_value > 0:
                signal_counts['bullish'] += 1
            elif signal_value < 0:
                signal_counts['bearish'] += 1
            else:
                signal_counts['neutral'] += 1
        
        avg_score = weighted_score / total_weight if total_weight > 0 else 0
        
        if avg_score >= 1.0:
            overall_sentiment = 'strong_bullish'
        elif avg_score >= 0.3:
            overall_sentiment = 'bullish'
        elif avg_score <= -1.0:
            overall_sentiment = 'strong_bearish'
        elif avg_score <= -0.3:
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'sentiment': overall_sentiment,
            'score': avg_score,
            'counts': signal_counts
        }

    def find_key_levels(self, levels: List[float]) -> str:
        """Find clustered support/resistance ranges."""
        if not levels:
            return "undefined"
        
        levels = sorted(set(levels))
        
        if len(levels) == 1:
            return f"${levels[0]:,.0f}"
        elif len(levels) <= 3:
            return f"${min(levels):,.0f}-${max(levels):,.0f}"
        else:
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if level - current_cluster[-1] <= (max(levels) - min(levels)) * 0.05:
                    current_cluster.append(level)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [level]
            
            clusters.append(current_cluster)
            
            key_cluster = max(clusters, key=len)
            return f"${min(key_cluster):,.0f}-${max(key_cluster):,.0f}"

    def analyze_momentum(self) -> str:
        """Analyze momentum characteristics."""
        momentum_indicators = ['MACD', 'RSI', 'Chaikin Money Flow', 'OBV']
        momentum_signals = []
        
        for result in self.results:
            if result.name in momentum_indicators:
                momentum_signals.append(result.signal)
        
        bullish_momentum = sum(1 for s in momentum_signals if 'bullish' in s)
        bearish_momentum = sum(1 for s in momentum_signals if any(x in s for x in ['bearish', 'distribution']))
        
        if bullish_momentum > bearish_momentum:
            return "momentum appears constructive"
        elif bearish_momentum > bullish_momentum:
            return "momentum shows signs of weakness"
        else:
            return "momentum remains mixed"

    def get_key_observations(self) -> List[str]:
        """Extract notable market conditions."""
        observations = []
        
        for result in self.results:
            if result.name == 'VWAP' and result.metadata:
                deviation = result.metadata.get('deviation_percent')
                if deviation and deviation > 15:
                    condition = "overbought" if result.metadata.get('position') == 'above' else "oversold"
                    observations.append(f"price trading {condition} relative to VWAP with {deviation:.1f}% deviation")
            
            if result.name == 'Ichimoku Cloud' and result.metadata:
                if result.metadata.get('cloud_position') in ['above_cloud', 'in_green_cloud']:
                    observations.append("price positioned above the Ichimoku cloud")
            
            if result.signal in ['bullish_breakout', 'bearish_breakout']:
                observations.append(f"{result.name.lower()} suggesting potential breakout")
            
            if result.name == 'RSI' and result.value:
                if result.value > 70:
                    observations.append("RSI indicating overbought conditions")
                elif result.value < 30:
                    observations.append("RSI indicating oversold conditions")
        
        return observations

    def generate_summary(self, symbol: str = "BTC/USDT", timeframe: str = "1d") -> str:
        """Generate a coherent narrative summary from collected indicator results."""
        if not self.results:
            return "Insufficient data for analysis summary."
        
        signal_analysis = self.aggregate_signals()
        support_levels, resistance_levels = self.extract_support_resistance()
        
        support_range = self.find_key_levels(support_levels)
        resistance_range = self.find_key_levels(resistance_levels)
        
        momentum_desc = self.analyze_momentum()
        observations = self.get_key_observations()
        
        sentiment_phrase = self.sentiment_phrases.get(signal_analysis['sentiment'], 'mixed signals')
        
        trend_direction = "an upward trend" if signal_analysis['score'] > 0 else "a downward trend" if signal_analysis['score'] < 0 else "sideways movement"
        
        strength_desc = "strong" if abs(signal_analysis['score']) > 0.7 else "moderate" if abs(signal_analysis['score']) > 0.3 else "weak"
        
        summary = f"The current market analysis for {symbol} on {timeframe} timeframe shows {sentiment_phrase}. "
        summary += f"Price action indicates {trend_direction} with {strength_desc} conviction. "
        
        if support_range != "undefined" and resistance_range != "undefined":
            summary += f"Key support levels are identified around {support_range} while resistance zones appear near {resistance_range}. "
        
        counts = signal_analysis['counts']
        summary += f"Technical indicators show {counts['bullish']} bullish signals, {counts['bearish']} bearish signals, and {counts['neutral']} neutral readings. "
        
        summary += f"Overall, {momentum_desc}. "
        
        if observations:
            if len(observations) == 1:
                summary += f"Notable observation: {observations[0]}. "
            else:
                summary += f"Notable observations include {', '.join(observations[:-1])}, and {observations[-1]}. "
        
        pivot_result = next((r for r in self.results if r.name == 'Pivot Point'), None)
        if pivot_result and pivot_result.value:
            summary += f"Traders should monitor the ${pivot_result.value:,.0f} pivot level for potential directional cues."
        
        return summary

# Global summarizer instance
_summarizer = AnalysisSummarizer()

def add_indicator_result(result: IndicatorResult):
    """Add an indicator result to the global summarizer."""
    _summarizer.add_result(result)

def clear_all_results():
    """Clear all stored indicator results."""
    _summarizer.clear_results()

def generate_analysis_summary(symbol: str = "BTC/USDT", timeframe: str = "1d") -> str:
    """Generate the final analysis summary from all collected results."""
    return _summarizer.generate_summary(symbol, timeframe)