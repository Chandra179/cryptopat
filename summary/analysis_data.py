"""
Data structures for analysis results and configuration.

This module contains data classes and type definitions used throughout
the analysis system for better type safety and maintainability.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class SentimentType(Enum):
    """Sentiment classification types."""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    WEAK_BULLISH = "weak_bullish"
    NEUTRAL = "neutral"
    WEAK_BEARISH = "weak_bearish"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


class TrendDirection(Enum):
    """Trend direction types."""
    UPWARD = "upward"
    DOWNWARD = "downward"
    SIDEWAYS = "sideways"


class TrendStrength(Enum):
    """Trend strength types."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


@dataclass
class ConfluenceData:
    """Signal confluence analysis results."""
    bullish_confluence: float
    bearish_confluence: float
    confluence_ratio: float
    
    @property
    def has_strong_confluence(self) -> bool:
        """Check if confluence is considered strong."""
        return self.confluence_ratio > 2.0
    
    @property
    def has_moderate_confluence(self) -> bool:
        """Check if confluence is considered moderate."""
        return 1.5 < self.confluence_ratio <= 2.0


@dataclass
class ProbabilisticAssessment:
    """Probabilistic risk assessment results."""
    bullish_probability: float
    bearish_probability: float
    neutral_probability: float
    confidence_level: float
    
    @property
    def dominant_scenario(self) -> str:
        """Get the scenario with highest probability."""
        probs = {
            'bullish': self.bullish_probability,
            'bearish': self.bearish_probability,
            'neutral': self.neutral_probability
        }
        return max(probs, key=probs.get)


@dataclass
class ScenarioData:
    """Individual scenario analysis data."""
    probability: float
    target_price: float
    scenario_description: str
    key_drivers: List[str]
    risk_reward_ratio: float


@dataclass
class ScenarioAnalysis:
    """Complete scenario analysis results."""
    bull_case: ScenarioData
    base_case: ScenarioData
    bear_case: ScenarioData
    
    @property
    def primary_scenario(self) -> tuple[str, ScenarioData]:
        """Get the scenario with highest probability."""
        scenarios = {
            'bull_case': self.bull_case,
            'base_case': self.base_case,
            'bear_case': self.bear_case
        }
        return max(scenarios.items(), key=lambda x: x[1].probability)


@dataclass
class SignalDistribution:
    """Distribution of signal types."""
    bullish: int
    bearish: int
    neutral: int
    
    @property
    def total_signals(self) -> int:
        """Total number of signals."""
        return self.bullish + self.bearish + self.neutral
    
    @property
    def bullish_ratio(self) -> float:
        """Ratio of bullish signals."""
        total = self.total_signals
        return self.bullish / total if total > 0 else 0.0


@dataclass
class MarketLevels:
    """Support and resistance level data."""
    support_levels: List[float]
    resistance_levels: List[float]
    support_range: str
    resistance_range: str
    key_pivot_level: Optional[float] = None
    
    @property
    def has_support_levels(self) -> bool:
        """Check if support levels are available."""
        return bool(self.support_levels)
    
    @property
    def has_resistance_levels(self) -> bool:
        """Check if resistance levels are available."""
        return bool(self.resistance_levels)


@dataclass
class AnalysisMetadata:
    """Metadata for analysis results."""
    symbol: str
    timeframe: str
    timestamp: int
    data_points: int
    current_price: Optional[float] = None


@dataclass
class AnalysisSummary:
    """Core analysis summary data."""
    sentiment: SentimentType
    sentiment_score: float
    trend_direction: TrendDirection
    trend_strength: TrendStrength
    signal_distribution: SignalDistribution
    momentum_description: str
    key_observations: List[str]
    market_levels: MarketLevels
    confluence_data: ConfluenceData
    probabilistic_assessment: ProbabilisticAssessment
    scenario_analysis: ScenarioAnalysis
    
    @property
    def primary_scenario_name(self) -> str:
        """Get primary scenario name."""
        return self.scenario_analysis.primary_scenario[0]
    
    @property
    def primary_scenario_probability(self) -> float:
        """Get primary scenario probability."""
        return self.scenario_analysis.primary_scenario[1].probability


@dataclass
class StructuredAnalysis:
    """Complete structured analysis results."""
    metadata: AnalysisMetadata
    analysis_summary: AnalysisSummary
    indicators: Dict[str, Any]
    detailed_breakdown: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backward compatibility."""
        return {
            "metadata": {
                "symbol": self.metadata.symbol,
                "timeframe": self.metadata.timeframe,
                "timestamp": self.metadata.timestamp,
                "data_points": self.metadata.data_points,
                "current_price": self.metadata.current_price
            },
            "analysis_summary": {
                "sentiment": self.analysis_summary.sentiment.value,
                "sentiment_score": self.analysis_summary.sentiment_score,
                "trend_direction": self.analysis_summary.trend_direction.value,
                "trend_strength": self.analysis_summary.trend_strength.value,
                "support_levels": self.analysis_summary.market_levels.support_levels[:5],
                "resistance_levels": self.analysis_summary.market_levels.resistance_levels[:5],
                "support_range": self.analysis_summary.market_levels.support_range,
                "resistance_range": self.analysis_summary.market_levels.resistance_range,
                "signal_distribution": {
                    "bullish": self.analysis_summary.signal_distribution.bullish,
                    "bearish": self.analysis_summary.signal_distribution.bearish,
                    "neutral": self.analysis_summary.signal_distribution.neutral
                },
                "momentum": self.analysis_summary.momentum_description,
                "key_observations": self.analysis_summary.key_observations[:5],
                "key_pivot_level": self.analysis_summary.market_levels.key_pivot_level,
                "bullish_probability": self.analysis_summary.probabilistic_assessment.bullish_probability,
                "bearish_probability": self.analysis_summary.probabilistic_assessment.bearish_probability,
                "neutral_probability": self.analysis_summary.probabilistic_assessment.neutral_probability,
                "confidence_level": self.analysis_summary.probabilistic_assessment.confidence_level,
                "confluence_score": self.analysis_summary.confluence_data.confluence_ratio,
                "bullish_confluence": self.analysis_summary.confluence_data.bullish_confluence,
                "bearish_confluence": self.analysis_summary.confluence_data.bearish_confluence,
                "primary_scenario": self.analysis_summary.primary_scenario_name,
                "primary_scenario_probability": self.analysis_summary.primary_scenario_probability,
                "primary_target": self.analysis_summary.scenario_analysis.primary_scenario[1].target_price
            },
            "indicators": self.indicators,
            "probabilistic_assessment": {
                "bullish_probability": self.analysis_summary.probabilistic_assessment.bullish_probability,
                "bearish_probability": self.analysis_summary.probabilistic_assessment.bearish_probability,
                "neutral_probability": self.analysis_summary.probabilistic_assessment.neutral_probability,
                "confidence_level": self.analysis_summary.probabilistic_assessment.confidence_level
            },
            "confluence_analysis": {
                "bullish_confluence": self.analysis_summary.confluence_data.bullish_confluence,
                "bearish_confluence": self.analysis_summary.confluence_data.bearish_confluence,
                "confluence_ratio": self.analysis_summary.confluence_data.confluence_ratio
            },
            "scenario_analysis": {
                "bull_case": {
                    "probability": self.analysis_summary.scenario_analysis.bull_case.probability,
                    "target_price": self.analysis_summary.scenario_analysis.bull_case.target_price,
                    "scenario_description": self.analysis_summary.scenario_analysis.bull_case.scenario_description,
                    "key_drivers": self.analysis_summary.scenario_analysis.bull_case.key_drivers,
                    "risk_reward_ratio": self.analysis_summary.scenario_analysis.bull_case.risk_reward_ratio
                },
                "base_case": {
                    "probability": self.analysis_summary.scenario_analysis.base_case.probability,
                    "target_price": self.analysis_summary.scenario_analysis.base_case.target_price,
                    "scenario_description": self.analysis_summary.scenario_analysis.base_case.scenario_description,
                    "key_drivers": self.analysis_summary.scenario_analysis.base_case.key_drivers,
                    "risk_reward_ratio": self.analysis_summary.scenario_analysis.base_case.risk_reward_ratio
                },
                "bear_case": {
                    "probability": self.analysis_summary.scenario_analysis.bear_case.probability,
                    "target_price": self.analysis_summary.scenario_analysis.bear_case.target_price,
                    "scenario_description": self.analysis_summary.scenario_analysis.bear_case.scenario_description,
                    "key_drivers": self.analysis_summary.scenario_analysis.bear_case.key_drivers,
                    "risk_reward_ratio": self.analysis_summary.scenario_analysis.bear_case.risk_reward_ratio
                }
            },
            "detailed_breakdown": self.detailed_breakdown
        }