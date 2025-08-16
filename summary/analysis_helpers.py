"""
Analysis Helpers - Utility functions for data extraction and processing.

This module provides helper functions that extract and process data
from indicator results, keeping the main analyzer focused on orchestration.
"""

from typing import Dict, List, Tuple, Any, Optional
from .formatters import IndicatorResult


class DataExtractor:
    """Handles data extraction from indicator results."""
    
    @staticmethod
    def extract_support_resistance(results: List[IndicatorResult]) -> Tuple[List[float], List[float]]:
        """Extract support and resistance levels from indicator results."""
        support_levels = []
        resistance_levels = []
        
        for result in results:
            if result.support:
                support_levels.append(result.support)
            if result.resistance:
                resistance_levels.append(result.resistance)
                
            if result.metadata:
                support_levels.extend(result.metadata.get('support_levels', []))
                resistance_levels.extend(result.metadata.get('resistance_levels', []))
        
        return support_levels, resistance_levels
    
    @staticmethod
    def extract_current_price(results: List[IndicatorResult], provided_price: Optional[float] = None) -> float:
        """Extract current price from results or use provided price."""
        if provided_price is not None:
            try:
                return float(provided_price)
            except (TypeError, ValueError):
                pass
        
        # Try to extract from results
        current_price = next((r.value for r in results if r.value), 0)
        if current_price:
            try:
                return float(current_price)
            except (TypeError, ValueError):
                pass
        
        return 0.0
    
    @staticmethod
    def extract_pivot_level(results: List[IndicatorResult]) -> Optional[float]:
        """Extract pivot level from results."""
        pivot_result = next((r for r in results if r.name == 'Pivot Point'), None)
        return pivot_result.value if pivot_result and pivot_result.value else None


class AnalysisProcessor:
    """Handles processing and calculation of analysis data."""
    
    def __init__(self, signal_aggregator, confluence_analyzer, risk_assessment, 
                 level_analyzer, momentum_analyzer, observation_extractor, scenario_analyzer):
        self.signal_aggregator = signal_aggregator
        self.confluence_analyzer = confluence_analyzer
        self.risk_assessment = risk_assessment
        self.level_analyzer = level_analyzer
        self.momentum_analyzer = momentum_analyzer
        self.observation_extractor = observation_extractor
        self.scenario_analyzer = scenario_analyzer
    
    def process_analysis_data(self, results: List[IndicatorResult], current_price: float) -> Dict[str, Any]:
        """Process all analysis data and return consolidated results."""
        # Calculate confluence scores
        confluence_data = self.confluence_analyzer.calculate_confluence_score(results, current_price)
        
        # Enhanced signal analysis with confluence
        signal_analysis = self.signal_aggregator.aggregate_signals(results, confluence_data)
        
        # Calculate probabilistic assessment
        probabilities = self.risk_assessment.calculate_probabilities(
            signal_analysis['score'], confluence_data)
        
        # Extract market data
        support_levels, resistance_levels = DataExtractor.extract_support_resistance(results)
        momentum_desc = self.momentum_analyzer.analyze_momentum(results)
        observations = self.observation_extractor.get_key_observations(results)
        
        # Generate scenario analysis
        scenarios = self.scenario_analyzer.generate_scenarios(
            support_levels, resistance_levels, current_price, probabilities)
        
        return {
            'signal_analysis': signal_analysis,
            'confluence_data': confluence_data,
            'probabilities': probabilities,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'momentum_desc': momentum_desc,
            'observations': observations,
            'scenarios': scenarios
        }


class AnalysisValidator:
    """Validates analysis data and ensures data integrity."""
    
    @staticmethod
    def validate_price(price: Any) -> float:
        """Validate and convert price to float."""
        if price is None:
            return 0.0
        
        try:
            return float(price)
        except (TypeError, ValueError):
            return 0.0
    
    @staticmethod
    def validate_results(results: List[IndicatorResult]) -> bool:
        """Validate that results list is not empty and contains valid data."""
        return bool(results) and all(isinstance(r, IndicatorResult) for r in results)
    
    @staticmethod
    def validate_analysis_components(metadata, analysis_summary) -> bool:
        """Validate that required analysis components are present."""
        return metadata is not None and analysis_summary is not None


class AnalysisOrchestrator:
    """Orchestrates the analysis process with clean separation of concerns."""
    
    def __init__(self, processor: AnalysisProcessor, data_extractor: DataExtractor,
                 validator: AnalysisValidator):
        self.processor = processor
        self.data_extractor = data_extractor
        self.validator = validator
    
    def orchestrate_analysis(self, results: List[IndicatorResult], 
                           symbol: str, timeframe: str, provided_price: Optional[float] = None) -> Dict[str, Any]:
        """Orchestrate the complete analysis process."""
        # Validate inputs
        if not self.validator.validate_results(results):
            raise ValueError("Invalid or empty results provided")
        
        # Extract current price
        current_price = self.data_extractor.extract_current_price(results, provided_price)
        current_price = self.validator.validate_price(current_price)
        
        # Process analysis data
        analysis_data = self.processor.process_analysis_data(results, current_price)
        
        # Add metadata
        analysis_data.update({
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': current_price,
            'data_points': len(results),
            'pivot_level': self.data_extractor.extract_pivot_level(results)
        })
        
        return analysis_data