"""
Analysis Summary Generator

Collects and summarizes results from all technical indicators in memory.
"""

import logging
import time
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
        
        # Remove duplicates and sort
        unique_levels = sorted(set(levels))
        
        if len(unique_levels) == 1:
            return f"${unique_levels[0]:,.0f}"
        elif len(unique_levels) == 2:
            return f"${min(unique_levels):,.0f}–${max(unique_levels):,.0f}"
        elif len(unique_levels) <= 4:
            # For small sets, show min-max range
            return f"${min(unique_levels):,.0f}–${max(unique_levels):,.0f}"
        else:
            # For larger sets, find the most significant cluster
            clusters = []
            current_cluster = [unique_levels[0]]
            threshold = (max(unique_levels) - min(unique_levels)) * 0.08  # 8% clustering threshold
            
            for level in unique_levels[1:]:
                if level - current_cluster[-1] <= threshold:
                    current_cluster.append(level)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [level]
            
            clusters.append(current_cluster)
            
            # Find the cluster with the most levels
            key_cluster = max(clusters, key=len)
            if len(key_cluster) == 1:
                return f"${key_cluster[0]:,.0f}"
            else:
                return f"${min(key_cluster):,.0f}–${max(key_cluster):,.0f}"

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

    def get_structured_analysis(self, symbol: str = "BTC/USDT", timeframe: str = "1d", current_price: float = None) -> Dict[str, Any]:
        """Generate structured analysis data matching output_schema.json."""
        if not self.results:
            return self._empty_analysis_structure(symbol, timeframe)
        
        signal_analysis = self.aggregate_signals()
        support_levels, resistance_levels = self.extract_support_resistance()
        momentum_desc = self.analyze_momentum()
        observations = self.get_key_observations()
        
        # Get current price from ticker or first result with value
        if current_price is None:
            current_price = next((r.value for r in self.results if r.value), 0)
        
        # Build structured analysis
        return {
            "metadata": self._build_metadata(symbol, timeframe, current_price),
            "analysis_summary": self._build_analysis_summary(signal_analysis, support_levels, resistance_levels, momentum_desc, observations),
            "indicators": self._build_indicators_structure(),
            "detailed_breakdown": self._build_detailed_breakdown(symbol, timeframe, signal_analysis, support_levels, resistance_levels, momentum_desc, observations)
        }
    
    def _empty_analysis_structure(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Return empty structure when no data available."""
        return {
            "metadata": {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": int(time.time() * 1000),
                "data_points": 0,
                "current_price": 0
            },
            "analysis_summary": {
                "sentiment": "neutral",
                "sentiment_score": 0,
                "trend_direction": "sideways",
                "trend_strength": "weak",
                "signal_distribution": {"bullish": 0, "bearish": 0, "neutral": 0},
                "momentum": "insufficient data",
                "key_observations": ["Insufficient data for analysis"]
            },
            "indicators": {"trend": {}, "momentum": {}, "volatility": {}, "support_resistance": {}},
            "detailed_breakdown": {"full_markdown": "## Insufficient Data\n\nNot enough indicator data available for comprehensive analysis."}
        }
    
    def _build_metadata(self, symbol: str, timeframe: str, current_price: float) -> Dict[str, Any]:
        """Build metadata section."""
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": int(time.time() * 1000),
            "data_points": len(self.results),
            "current_price": current_price
        }
    
    def _build_analysis_summary(self, signal_analysis: Dict, support_levels: List[float], resistance_levels: List[float], momentum_desc: str, observations: List[str]) -> Dict[str, Any]:
        """Build analysis summary section."""
        support_range = self.find_key_levels(support_levels)
        resistance_range = self.find_key_levels(resistance_levels)
        
        trend_direction = "upward" if signal_analysis['score'] > 0 else "downward" if signal_analysis['score'] < 0 else "sideways"
        trend_strength = "strong" if abs(signal_analysis['score']) > 0.7 else "moderate" if abs(signal_analysis['score']) > 0.3 else "weak"
        
        pivot_result = next((r for r in self.results if r.name == 'Pivot Point'), None)
        key_pivot_level = pivot_result.value if pivot_result and pivot_result.value else None
        
        return {
            "sentiment": signal_analysis['sentiment'],
            "sentiment_score": round(signal_analysis['score'], 2),
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "support_levels": support_levels[:5],  # Top 5 levels
            "resistance_levels": resistance_levels[:5],  # Top 5 levels
            "support_range": support_range,
            "resistance_range": resistance_range,
            "signal_distribution": signal_analysis['counts'],
            "momentum": momentum_desc,
            "key_observations": observations[:5],  # Top 5 observations
            "key_pivot_level": key_pivot_level
        }

    def _build_indicators_structure(self) -> Dict[str, Any]:
        """Build indicators section organized by category."""
        indicators = {
            "trend": {},
            "momentum": {},
            "volatility": {},
            "support_resistance": {}
        }
        
        # Categorize indicators
        trend_indicators = ['Ichimoku Cloud', 'SuperTrend', 'EMA 20/50', 'VWAP']
        momentum_indicators = ['RSI', 'MACD', 'Chaikin Money Flow', 'OBV']
        volatility_indicators = ['Bollinger Bands', 'Keltner Channel', 'Donchian Channel']
        support_resistance_indicators = ['Pivot Point', 'Parabolic SAR', 'Renko Chart']
        
        for result in self.results:
            indicator_data = self._format_indicator_result(result)
            
            if result.name in trend_indicators:
                key = result.name.lower().replace(' ', '_').replace('/', '_')
                indicators["trend"][key] = indicator_data
            elif result.name in momentum_indicators:
                key = result.name.lower().replace(' ', '_').replace('/', '_')
                indicators["momentum"][key] = indicator_data
            elif result.name in volatility_indicators:
                key = result.name.lower().replace(' ', '_').replace('/', '_')
                indicators["volatility"][key] = indicator_data
            elif result.name in support_resistance_indicators:
                key = result.name.lower().replace(' ', '_').replace('/', '_')
                indicators["support_resistance"][key] = indicator_data
        
        return indicators
    
    def _format_indicator_result(self, result: IndicatorResult) -> Dict[str, Any]:
        """Format an individual indicator result according to schema."""
        base_data = {
            "name": result.name,
            "signal": result.signal,
            "strength": result.strength,
            "parameters": result.metadata.get('parameters', {}) if result.metadata else {}
        }
        
        if result.value is not None:
            base_data["value"] = result.value
        if result.support is not None:
            base_data["support"] = result.support
        if result.resistance is not None:
            base_data["resistance"] = result.resistance
        
        # Add indicator-specific fields from metadata
        if result.metadata:
            base_data.update(result.metadata)
        
        return base_data
    
    def _build_detailed_breakdown(self, symbol: str, timeframe: str, signal_analysis: Dict, support_levels: List[float], resistance_levels: List[float], momentum_desc: str, observations: List[str]) -> Dict[str, Any]:
        """Build detailed markdown breakdown sections."""
        
        # Generate formatted sections
        core_summary = self._format_core_summary(symbol, timeframe, signal_analysis, support_levels, resistance_levels, momentum_desc, observations)
        trend_analysis = self._format_trend_analysis()
        momentum_analysis = self._format_momentum_analysis()
        volatility_analysis = self._format_volatility_analysis()
        support_resistance_analysis = self._format_support_resistance_analysis()
        
        full_markdown = f"{core_summary}\n\n{trend_analysis}\n\n{momentum_analysis}\n\n{volatility_analysis}\n\n{support_resistance_analysis}"
        
        return {
            "core_summary": core_summary,
            "trend_analysis": trend_analysis,
            "momentum_analysis": momentum_analysis,
            "volatility_analysis": volatility_analysis,
            "support_resistance_analysis": support_resistance_analysis,
            "full_markdown": full_markdown
        }
    
    def _format_core_summary(self, symbol: str, timeframe: str, signal_analysis: Dict, support_levels: List[float], resistance_levels: List[float], momentum_desc: str, observations: List[str]) -> str:
        """Format the core summary section."""
        support_range = self.find_key_levels(support_levels)
        resistance_range = self.find_key_levels(resistance_levels)
        sentiment_phrase = self.sentiment_phrases.get(signal_analysis['sentiment'], 'mixed signals')
        
        trend_direction = "Upward trend" if signal_analysis['score'] > 0 else "Downward trend" if signal_analysis['score'] < 0 else "Sideways movement"
        strength_desc = "strong" if abs(signal_analysis['score']) > 0.7 else "moderate" if abs(signal_analysis['score']) > 0.3 else "weak"
        
        pivot_result = next((r for r in self.results if r.name == 'Pivot Point'), None)
        key_pivot = f"${pivot_result.value:,.0f}" if pivot_result and pivot_result.value else "undefined"
        
        counts = signal_analysis['counts']
        
        summary = f"- Market Analysis for {symbol} ({timeframe} timeframe)\n\n"
        summary += "- Core Summary\n"
        summary += f"  - Overall Sentiment: {sentiment_phrase.title()}\n"
        summary += f"  - Trend Direction: {trend_direction} with {strength_desc} conviction\n"
        
        if support_range != "undefined":
            summary += f"  - Support Levels: {support_range}\n"
        if resistance_range != "undefined":
            summary += f"  - Resistance Levels: {resistance_range}\n"
        
        summary += f"  - Momentum: {momentum_desc.title()}\n"
        summary += f"  - Signal Distribution: {counts['bullish']} bullish, {counts['bearish']} bearish, {counts['neutral']} neutral\n"
        
        if observations:
            summary += "  - Notable Observations:\n"
            for obs in observations[:3]:  # Top 3 observations
                summary += f"    - {obs.title()}\n"
        
        summary += f"  - Key Pivot Level: {key_pivot}"
        
        return summary
    
    def _format_trend_analysis(self) -> str:
        """Format trend indicators section."""
        trend_indicators = ['Ichimoku Cloud', 'SuperTrend', 'EMA 20/50', 'VWAP']
        section = "- Indicator Breakdown\n  - Trend"
        
        for result in self.results:
            if result.name in trend_indicators:
                description = self._get_indicator_description(result)
                section += f"\n    - {result.name}: {description}"
        
        return section
    
    def _format_momentum_analysis(self) -> str:
        """Format momentum indicators section."""
        momentum_indicators = ['RSI', 'MACD', 'Chaikin Money Flow', 'OBV']
        section = "  - Momentum"
        
        for result in self.results:
            if result.name in momentum_indicators:
                description = self._get_indicator_description(result)
                section += f"\n    - {result.name}: {description}"
        
        return section
    
    def _format_volatility_analysis(self) -> str:
        """Format volatility indicators section."""
        volatility_indicators = ['Bollinger Bands', 'Keltner Channel', 'Donchian Channel']
        section = "  - Volatility & Channels"
        
        for result in self.results:
            if result.name in volatility_indicators:
                description = self._get_indicator_description(result)
                section += f"\n    - {result.name}: {description}"
        
        return section
    
    def _format_support_resistance_analysis(self) -> str:
        """Format support/resistance indicators section."""
        sr_indicators = ['Pivot Point', 'Parabolic SAR', 'Renko Chart']
        section = "  - Support & Resistance"
        
        for result in self.results:
            if result.name in sr_indicators:
                description = self._get_indicator_description(result)
                section += f"\n    - {result.name}: {description}"
        
        return section
    
    def _get_indicator_description(self, result: IndicatorResult) -> str:
        """Generate a descriptive text for an indicator result."""
        signal_desc = result.signal.replace('_', ' ').title()
        
        if result.name == 'RSI' and result.value:
            return f"{result.value:.0f} → {signal_desc.lower()}"
        elif result.name == 'MACD':
            return f"{signal_desc}, histogram {'rising' if result.metadata and result.metadata.get('histogram_increasing') else 'declining'}"
        elif result.name == 'VWAP' and result.metadata:
            position = result.metadata.get('position', 'unknown')
            deviation = result.metadata.get('deviation_percent', 0)
            return f"Trading {position} VWAP with {deviation:+.0f}% deviation"
        elif result.name == 'Ichimoku Cloud' and result.metadata:
            position = result.metadata.get('cloud_position', 'unknown')
            return f"Price {position.replace('_', ' ')}"
        elif result.name == 'SuperTrend' and result.metadata:
            trend = result.metadata.get('trend_direction', result.signal)
            support_level = result.support
            if support_level:
                return f"{trend.title()}, support at ${support_level:,.0f}"
            return f"{trend.title()}"
        elif result.name == 'Bollinger Bands' and result.metadata:
            position = result.metadata.get('position', 'unknown')
            squeeze = result.metadata.get('squeeze', False)
            expansion = result.metadata.get('expansion', False)
            if squeeze:
                return f"Price {position}, volatility squeeze"
            elif expansion:
                return f"Price {position}, volatility expansion"
            return f"Price {position}"
        else:
            return signal_desc.lower()
    
    def generate_summary(self, symbol: str = "BTC/USDT", timeframe: str = "1d") -> str:
        """Generate a coherent narrative summary from collected indicator results (legacy method)."""
        structured_data = self.get_structured_analysis(symbol, timeframe)
        return structured_data["detailed_breakdown"]["full_markdown"]

# Global summarizer instance
_summarizer = AnalysisSummarizer()

def add_indicator_result(result: IndicatorResult):
    """Add an indicator result to the global summarizer."""
    _summarizer.add_result(result)

def clear_all_results():
    """Clear all stored indicator results."""
    _summarizer.clear_results()

def generate_analysis_summary(symbol: str = "BTC/USDT", timeframe: str = "1d") -> str:
    """Generate the final analysis summary from all collected results (legacy method)."""
    return _summarizer.generate_summary(symbol, timeframe)

def get_structured_analysis(symbol: str = "BTC/USDT", timeframe: str = "1d", current_price: float = None) -> Dict[str, Any]:
    """Generate structured analysis data matching output_schema.json."""
    return _summarizer.get_structured_analysis(symbol, timeframe, current_price)