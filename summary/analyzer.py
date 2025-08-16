"""
Analysis Summarizer - Refactored for Better Maintainability

Enhanced analysis summarizer with signal confluence weighting, probabilistic risk assessment,
and scenario analysis based on academic research and industry standards.

This module has been refactored into a modular architecture for better maintainability:
- analysis_data.py: Data structures and type definitions
- analysis_core.py: Core analysis components  
- risk_assessment.py: Probabilistic analysis and confluence calculations
- scenario_analysis.py: Bull/Bear/Base case scenario planning

Academic References:
- Amat, C., Michalot, T., & Miqueu, E. (2021). Evidence and Behaviour of Support and Resistance Levels 
  in Financial Time Series. arXiv:2101.07410. https://arxiv.org/abs/2101.07410
- Akyildirim, E., et al. (2022). Support Resistance Levels towards Profitability in Intelligent 
  Algorithmic Trading Models. Mathematics, 10(20), 3888. https://www.mdpi.com/2227-7390/10/20/3888
- Han, Y., et al. (2024). The predictive ability of technical trading rules: an empirical analysis of 
  developed and emerging equity markets. Financial Markets and Portfolio Management.
- Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. Springer-Verlag.
- CFA Institute. (2024). Financial Modeling Practical Skills Module.
"""

import logging
from typing import Dict, List, Any

from .config import IndicatorConfig
from .formatters import IndicatorResult
from .analysis_core import (
    LevelAnalyzer, SignalAggregator, MomentumAnalyzer, ObservationExtractor
)
from .risk_assessment import ProbabilisticRiskAssessment, ConfluenceAnalyzer
from .scenario_analysis import ScenarioAnalyzer
from .analysis_builder import AnalysisBuilder
from .analysis_formatter import AnalysisFormatter
from .analysis_helpers import (
    DataExtractor, AnalysisProcessor, AnalysisValidator, AnalysisOrchestrator
)

logger = logging.getLogger(__name__)




class AnalysisSummarizer:
    """Enhanced analysis summarizer with advanced signal processing and scenario analysis.
    
    Refactored for better maintainability with modular architecture.
    Implements comprehensive technical analysis methodology combining:
    - Enhanced weighted indicator consensus with confluence analysis
    - Monte Carlo probabilistic risk assessment
    - Bull/Bear/Base case scenario planning
    - Support/resistance clustering algorithms
    - Multi-dimensional market analysis framework
    
    Follows academic standards and industry best practices for systematic technical analysis.
    """
    
    def __init__(self):
        self.results: List[IndicatorResult] = []
        self.config = IndicatorConfig()
        
        # Initialize analysis components
        self.signal_aggregator = SignalAggregator(self.config)
        self.level_analyzer = LevelAnalyzer()
        self.momentum_analyzer = MomentumAnalyzer()
        self.observation_extractor = ObservationExtractor()
        self.confluence_analyzer = ConfluenceAnalyzer()
        self.risk_assessment = ProbabilisticRiskAssessment()
        self.scenario_analyzer = ScenarioAnalyzer()
        
        # Initialize helper components
        self.builder = AnalysisBuilder(self.config)
        self.formatter = AnalysisFormatter(self.config)
        self.data_extractor = DataExtractor()
        self.validator = AnalysisValidator()
        
        # Initialize processor and orchestrator
        self.processor = AnalysisProcessor(
            self.signal_aggregator, self.confluence_analyzer, self.risk_assessment,
            self.level_analyzer, self.momentum_analyzer, self.observation_extractor,
            self.scenario_analyzer
        )
        self.orchestrator = AnalysisOrchestrator(
            self.processor, self.data_extractor, self.validator
        )

    def add_result(self, result: IndicatorResult):
        """Add an indicator result to the collection."""
        self.results.append(result)
        
    def clear_results(self):
        """Clear all stored results."""
        self.results.clear()

    def extract_support_resistance(self):
        """Extract support and resistance levels from indicator results (legacy method)."""
        return self.data_extractor.extract_support_resistance(self.results)

    def get_structured_analysis(self, symbol: str = "BTC/USDT", timeframe: str = "1d", 
                              current_price: float = None) -> Dict[str, Any]:
        """Generate structured analysis using the new modular architecture."""
        try:
            # Orchestrate the analysis process
            analysis_data = self.orchestrator.orchestrate_analysis(
                self.results, symbol, timeframe, current_price
            )
            
            # Build structured analysis using builder pattern
            structured_analysis = self._build_structured_analysis(analysis_data)
            
            # Return as dictionary for backward compatibility
            return structured_analysis.to_dict()
            
        except Exception as e:
            logger.error(f"Error in get_structured_analysis: {str(e)}")
            # Return minimal structure on error
            return self._get_fallback_analysis(symbol, timeframe, current_price)
    
    def _build_structured_analysis(self, analysis_data: Dict[str, Any]):
        """Build structured analysis from processed data."""
        # Reset and configure builder
        self.builder.reset()
        
        # Build metadata
        self.builder.build_metadata(
            analysis_data['symbol'], 
            analysis_data['timeframe'],
            analysis_data['current_price'],
            analysis_data['data_points']
        )
        
        # Build market levels
        self.builder.build_market_levels(
            analysis_data['support_levels'],
            analysis_data['resistance_levels'],
            self.level_analyzer,
            analysis_data['pivot_level']
        )
        
        # Build signal distribution
        self.builder.build_signal_distribution(analysis_data['signal_analysis'])
        
        # Build analysis summary
        self.builder.build_analysis_summary(
            analysis_data['signal_analysis'],
            analysis_data['momentum_desc'],
            analysis_data['observations'],
            analysis_data['confluence_data'],
            analysis_data['probabilities'],
            analysis_data['scenarios']
        )
        
        # Build detailed breakdown
        detailed_breakdown = self.formatter.format_enhanced_detailed_breakdown(
            analysis_data['symbol'],
            analysis_data['timeframe'],
            analysis_data['signal_analysis'],
            analysis_data['support_levels'],
            analysis_data['resistance_levels'],
            analysis_data['momentum_desc'],
            analysis_data['observations'],
            analysis_data['scenarios'],
            analysis_data['confluence_data'],
            analysis_data['probabilities'],
            self.results
        )
        
        # Build final structure
        structured_analysis = self.builder.build(detailed_breakdown)
        
        # Add indicators structure
        structured_analysis.indicators = self.builder.build_indicators_structure(self.results)
        
        return structured_analysis
    
    def _get_fallback_analysis(self, symbol: str, timeframe: str, current_price: float) -> Dict[str, Any]:
        """Provide fallback analysis structure in case of errors."""
        import time
        
        current_price = self.validator.validate_price(current_price)
        
        return {
            "metadata": {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": int(time.time() * 1000),
                "data_points": len(self.results),
                "current_price": current_price
            },
            "analysis_summary": {
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "trend_direction": "sideways",
                "trend_strength": "weak",
                "signal_distribution": {"bullish": 0, "bearish": 0, "neutral": 0},
                "momentum_description": "insufficient data",
                "key_observations": [],
                "market_levels": {
                    "support_levels": [],
                    "resistance_levels": [],
                    "support_range": "undefined",
                    "resistance_range": "undefined",
                    "key_pivot_level": None
                }
            },
            "indicators": {},
            "detailed_breakdown": {
                "full_markdown": "Analysis unavailable due to processing error."
            }
        }
    
    

    
    
    
    
    
    
    def generate_summary(self, symbol: str = "BTC/USDT", timeframe: str = "1d") -> str:
        """Generate a coherent narrative summary (legacy method)."""
        structured_data = self.get_structured_analysis(symbol, timeframe)
        return structured_data["detailed_breakdown"]["full_markdown"]