"""
Analysis Builder - Handles complex data structure creation for analysis results.

This module provides a builder pattern implementation for creating structured
analysis data, separating the construction logic from the main analyzer.
"""

import time
from typing import Dict, List, Any, Optional

from .formatters import IndicatorResult
from .analysis_data import (
    StructuredAnalysis, AnalysisMetadata, AnalysisSummary, MarketLevels,
    SignalDistribution
)
from .analysis_core import get_trend_direction, get_trend_strength
from .config import IndicatorConfig


class AnalysisBuilder:
    """Builder class for creating structured analysis data."""
    
    def __init__(self, config: IndicatorConfig):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset builder state."""
        self._metadata = None
        self._market_levels = None
        self._signal_distribution = None
        self._analysis_summary = None
        return self
    
    def build_metadata(self, symbol: str, timeframe: str, current_price: float, 
                      data_points: int) -> 'AnalysisBuilder':
        """Build metadata section."""
        self._metadata = AnalysisMetadata(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=int(time.time() * 1000),
            data_points=data_points,
            current_price=current_price
        )
        return self
    
    def build_market_levels(self, support_levels: List[float], 
                           resistance_levels: List[float],
                           level_analyzer, pivot_level: Optional[float] = None) -> 'AnalysisBuilder':
        """Build market levels section."""
        self._market_levels = MarketLevels(
            support_levels=support_levels[:5],
            resistance_levels=resistance_levels[:5],
            support_range=level_analyzer.find_key_levels(support_levels),
            resistance_range=level_analyzer.find_key_levels(resistance_levels),
            key_pivot_level=pivot_level
        )
        return self
    
    def build_signal_distribution(self, signal_analysis: Dict) -> 'AnalysisBuilder':
        """Build signal distribution section."""
        counts = signal_analysis['counts']
        self._signal_distribution = SignalDistribution(
            bullish=counts['bullish'],
            bearish=counts['bearish'],
            neutral=counts['neutral']
        )
        return self
    
    def build_analysis_summary(self, signal_analysis: Dict, momentum_desc: str,
                              observations: List[str], confluence_data: Dict,
                              probabilities: Dict, scenarios: Any) -> 'AnalysisBuilder':
        """Build analysis summary section."""
        self._analysis_summary = AnalysisSummary(
            sentiment=signal_analysis['sentiment'],
            sentiment_score=round(signal_analysis['score'], 2),
            trend_direction=get_trend_direction(signal_analysis['score']),
            trend_strength=get_trend_strength(signal_analysis['score']),
            signal_distribution=self._signal_distribution,
            momentum_description=momentum_desc,
            key_observations=observations[:5],
            market_levels=self._market_levels,
            confluence_data=confluence_data,
            probabilistic_assessment=probabilities,
            scenario_analysis=scenarios
        )
        return self
    
    def build_indicators_structure(self, results: List[IndicatorResult]) -> Dict[str, Any]:
        """Build indicators section organized by category."""
        indicators = {
            "trend": {},
            "momentum": {},
            "volatility": {},
            "support_resistance": {}
        }
        
        for result in results:
            category = self.config.get_indicator_category(result.name)
            if category in indicators:
                key = result.name.lower().replace(' ', '_').replace('/', '_')
                indicators[category][key] = self._format_indicator_result(result)
        
        return indicators
    
    def _format_indicator_result(self, result: IndicatorResult) -> Dict[str, Any]:
        """Format an individual indicator result."""
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
        
        if result.metadata:
            base_data.update(result.metadata)
        
        return base_data
    
    def build(self, detailed_breakdown: Dict[str, Any]) -> StructuredAnalysis:
        """Build the final structured analysis."""
        if not all([self._metadata, self._analysis_summary]):
            raise ValueError("Missing required components for StructuredAnalysis")
        
        return StructuredAnalysis(
            metadata=self._metadata,
            analysis_summary=self._analysis_summary,
            indicators={},  # Will be built separately
            detailed_breakdown=detailed_breakdown
        )