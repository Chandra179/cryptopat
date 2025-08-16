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
import time
from typing import Dict, List, Tuple, Any

from .config import IndicatorConfig
from .formatters import IndicatorResult, FormatterFactory
from .analysis_data import (
    StructuredAnalysis, AnalysisMetadata, AnalysisSummary, MarketLevels,
    SignalDistribution, SentimentType, TrendDirection, TrendStrength
)
from .analysis_core import (
    LevelAnalyzer, SignalAggregator, MomentumAnalyzer, ObservationExtractor,
    get_trend_direction, get_trend_strength
)
from .risk_assessment import ProbabilisticRiskAssessment, ConfluenceAnalyzer
from .scenario_analysis import ScenarioAnalyzer

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
                support_levels.extend(result.metadata.get('support_levels', []))
                resistance_levels.extend(result.metadata.get('resistance_levels', []))
        
        return support_levels, resistance_levels

    def get_structured_analysis(self, symbol: str = "BTC/USDT", timeframe: str = "1d", 
                              current_price: float = None) -> Dict[str, Any]:
        """Generate enhanced structured analysis with confluence, probabilities, and scenarios."""
        if not self.results:
            return self._create_empty_analysis(symbol, timeframe)
        
        # Ensure current_price is properly converted to float
        if current_price is not None:
            try:
                current_price = float(current_price)
            except (TypeError, ValueError):
                current_price = None
                
        if current_price is None:
            current_price = next((r.value for r in self.results if r.value), 0)
            if current_price:
                try:
                    current_price = float(current_price)
                except (TypeError, ValueError):
                    current_price = 0
        
        # Calculate confluence scores
        confluence_data = self.confluence_analyzer.calculate_confluence_score(
            self.results, current_price)
        
        # Enhanced signal analysis with confluence
        signal_analysis = self.signal_aggregator.aggregate_signals(self.results, confluence_data)
        
        # Calculate probabilistic assessment
        probabilities = self.risk_assessment.calculate_probabilities(
            signal_analysis['score'], confluence_data)
        
        # Extract market levels and characteristics
        support_levels, resistance_levels = self.extract_support_resistance()
        momentum_desc = self.momentum_analyzer.analyze_momentum(self.results)
        observations = self.observation_extractor.get_key_observations(self.results)
        
        # Generate scenario analysis
        scenarios = self.scenario_analyzer.generate_scenarios(
            signal_analysis, support_levels, resistance_levels, 
            current_price, probabilities)
        
        # Build structured analysis using new data classes
        metadata = AnalysisMetadata(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=int(time.time() * 1000),
            data_points=len(self.results),
            current_price=current_price
        )
        
        # Create market levels
        market_levels = MarketLevels(
            support_levels=support_levels[:5],
            resistance_levels=resistance_levels[:5],
            support_range=self.level_analyzer.find_key_levels(support_levels),
            resistance_range=self.level_analyzer.find_key_levels(resistance_levels),
            key_pivot_level=self._get_pivot_level()
        )
        
        # Create signal distribution
        signal_distribution = SignalDistribution(
            bullish=signal_analysis['counts']['bullish'],
            bearish=signal_analysis['counts']['bearish'],
            neutral=signal_analysis['counts']['neutral']
        )
        
        # Create analysis summary
        analysis_summary = AnalysisSummary(
            sentiment=signal_analysis['sentiment'],
            sentiment_score=round(signal_analysis['score'], 2),
            trend_direction=get_trend_direction(signal_analysis['score']),
            trend_strength=get_trend_strength(signal_analysis['score']),
            signal_distribution=signal_distribution,
            momentum_description=momentum_desc,
            key_observations=observations[:5],
            market_levels=market_levels,
            confluence_data=confluence_data,
            probabilistic_assessment=probabilities,
            scenario_analysis=scenarios
        )
        
        # Create structured analysis
        structured_analysis = StructuredAnalysis(
            metadata=metadata,
            analysis_summary=analysis_summary,
            indicators=self._build_indicators_structure(),
            detailed_breakdown=self._build_enhanced_detailed_breakdown(
                symbol, timeframe, signal_analysis, support_levels,
                resistance_levels, momentum_desc, observations, scenarios,
                confluence_data, probabilities
            )
        )
        
        # Return as dictionary for backward compatibility
        return structured_analysis.to_dict()
    
    def _get_pivot_level(self) -> float:
        """Get key pivot level from results."""
        pivot_result = next((r for r in self.results if r.name == 'Pivot Point'), None)
        return pivot_result.value if pivot_result and pivot_result.value else None
    
    def _build_metadata(self, symbol: str, timeframe: str, current_price: float) -> Dict[str, Any]:
        """Build metadata section."""
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": int(time.time() * 1000),
            "data_points": len(self.results),
            "current_price": current_price
        }
    
    def _build_enhanced_analysis_summary(self, signal_analysis: Dict, support_levels: List[float], 
                                       resistance_levels: List[float], momentum_desc: str, 
                                       observations: List[str], scenarios: Dict[str, Dict]) -> Dict[str, Any]:
        """Build enhanced analysis summary with probabilistic and confluence data."""
        support_range = self.level_analyzer.find_key_levels(support_levels)
        resistance_range = self.level_analyzer.find_key_levels(resistance_levels)
        
        trend_direction = self._get_trend_direction(signal_analysis['score'])
        trend_strength = self._get_trend_strength(signal_analysis['score'])
        
        pivot_result = next((r for r in self.results if r.name == 'Pivot Point'), None)
        key_pivot_level = pivot_result.value if pivot_result and pivot_result.value else None
        
        # Extract probabilistic data
        probabilities = signal_analysis.get('probabilities', {})
        confluence_data = signal_analysis.get('confluence_data', {})
        
        # Get highest probability scenario
        best_scenario = max(scenarios.items(), key=lambda x: x[1]['probability']) if scenarios else ('base_case', {})
        
        return {
            "sentiment": signal_analysis['sentiment'],
            "sentiment_score": round(signal_analysis['score'], 2),
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "support_levels": support_levels[:5],
            "resistance_levels": resistance_levels[:5],
            "support_range": support_range,
            "resistance_range": resistance_range,
            "signal_distribution": signal_analysis['counts'],
            "momentum": momentum_desc,
            "key_observations": observations[:5],
            "key_pivot_level": key_pivot_level,
            # Enhanced probabilistic data
            "bullish_probability": probabilities.get('bullish_probability', 0.0),
            "bearish_probability": probabilities.get('bearish_probability', 0.0),
            "neutral_probability": probabilities.get('neutral_probability', 0.0),
            "confidence_level": probabilities.get('confidence_level', 0.0),
            # Confluence analysis
            "confluence_score": confluence_data.get('confluence_ratio', 0.0),
            "bullish_confluence": confluence_data.get('bullish_confluence', 0.0),
            "bearish_confluence": confluence_data.get('bearish_confluence', 0.0),
            # Primary scenario
            "primary_scenario": best_scenario[0],
            "primary_scenario_probability": best_scenario[1].get('probability', 0.0),
            "primary_target": best_scenario[1].get('target_price', 0.0)
        }
    
    def _get_trend_direction(self, score: float) -> str:
        """Get trend direction from score."""
        if score > 0:
            return "upward"
        elif score < 0:
            return "downward"
        else:
            return "sideways"
    
    def _get_trend_strength(self, score: float) -> str:
        """Get trend strength from score."""
        abs_score = abs(score)
        if abs_score > 0.7:
            return "strong"
        elif abs_score > 0.3:
            return "moderate"
        else:
            return "weak"

    def _build_indicators_structure(self) -> Dict[str, Any]:
        """Build indicators section organized by category."""
        indicators = {
            "trend": {},
            "momentum": {},
            "volatility": {},
            "support_resistance": {}
        }
        
        for result in self.results:
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
    
    def _build_enhanced_detailed_breakdown(self, symbol: str, timeframe: str, signal_analysis: Dict, 
                                         support_levels: List[float], resistance_levels: List[float], 
                                         momentum_desc: str, observations: List[str], 
                                         scenarios, confluence_data, probabilities) -> Dict[str, Any]:
        """Build enhanced detailed markdown breakdown with probabilistic and scenario analysis."""
        core_summary = self._format_enhanced_core_summary(
            symbol, timeframe, signal_analysis, support_levels, 
            resistance_levels, momentum_desc, observations, scenarios,
            confluence_data, probabilities
        )
        trend_analysis = self._format_category_analysis('trend')
        momentum_analysis = self._format_category_analysis('momentum')
        volatility_analysis = self._format_category_analysis('volatility')
        support_resistance_analysis = self._format_category_analysis('support_resistance')
        probabilistic_analysis = self._format_probabilistic_analysis(confluence_data, probabilities)
        scenario_breakdown = self._format_scenario_breakdown(scenarios)
        
        full_markdown = (f"{core_summary}\n\n{trend_analysis}\n\n"
                        f"{momentum_analysis}\n\n{volatility_analysis}\n\n"
                        f"{support_resistance_analysis}\n\n"
                        f"{probabilistic_analysis}\n\n{scenario_breakdown}")
        
        return {
            "core_summary": core_summary,
            "trend_analysis": trend_analysis,
            "momentum_analysis": momentum_analysis,
            "volatility_analysis": volatility_analysis,
            "support_resistance_analysis": support_resistance_analysis,
            "probabilistic_analysis": probabilistic_analysis,
            "scenario_analysis": scenario_breakdown,
            "full_markdown": full_markdown
        }
    
    def _format_enhanced_core_summary(self, symbol: str, timeframe: str, signal_analysis: Dict, 
                                     support_levels: List[float], resistance_levels: List[float], 
                                     momentum_desc: str, observations: List[str], 
                                     scenarios, confluence_data, probabilities) -> str:
        """Format enhanced core summary with probabilistic and scenario data."""
        support_range = self.level_analyzer.find_key_levels(support_levels)
        resistance_range = self.level_analyzer.find_key_levels(resistance_levels)
        sentiment_phrase = self.config.get_sentiment_phrase(signal_analysis['sentiment'])
        
        trend_direction = ("Upward trend" if signal_analysis['score'] > 0 
                         else "Downward trend" if signal_analysis['score'] < 0 
                         else "Sideways movement")
        strength_desc = self._get_trend_strength(signal_analysis['score'])
        
        pivot_result = next((r for r in self.results if r.name == 'Pivot Point'), None)
        key_pivot = f"${pivot_result.value:,.0f}" if pivot_result and pivot_result.value else "undefined"
        
        counts = signal_analysis['counts']
        
        # Get primary scenario
        primary_scenario = scenarios.primary_scenario if scenarios else ('base_case', type('obj', (object,), {'probability': 0.0, 'target_price': 0.0})())
        
        summary = f"- Market Analysis for {symbol} ({timeframe} timeframe)\n"
        summary += "- Enhanced Core Summary\n"
        summary += f"  - Overall Sentiment: {sentiment_phrase.title()}\n"
        summary += f"  - Trend Direction: {trend_direction} with {strength_desc} conviction\n"
        
        # Add probabilistic assessment
        if probabilities:
            bull_prob = int(probabilities.bullish_probability * 100)
            bear_prob = int(probabilities.bearish_probability * 100)
            neutral_prob = int(probabilities.neutral_probability * 100)
            summary += f"  - Probability Assessment: {bull_prob}% bullish, {bear_prob}% bearish, {neutral_prob}% neutral\n"
            summary += f"  - Confidence Level: {int(probabilities.confidence_level * 100)}%\n"
        
        # Add confluence analysis
        if confluence_data:
            confluence_ratio = confluence_data.confluence_ratio
            if confluence_ratio > 2.0:
                summary += f"  - Signal Confluence: Strong ({confluence_ratio:.1f}x)\n"
            elif confluence_ratio > 1.5:
                summary += f"  - Signal Confluence: Moderate ({confluence_ratio:.1f}x)\n"
            else:
                summary += f"  - Signal Confluence: Weak ({confluence_ratio:.1f}x)\n"
        
        if support_range != "undefined":
            summary += f"  - Support Levels: {support_range}\n"
        if resistance_range != "undefined":
            summary += f"  - Resistance Levels: {resistance_range}\n"
        
        summary += f"  - Momentum: {momentum_desc.title()}\n"
        summary += f"  - Signal Distribution: {counts['bullish']} bullish, {counts['bearish']} bearish, {counts['neutral']} neutral\n"
        
        # Add primary scenario
        if primary_scenario:
            scenario_name, scenario_data = primary_scenario
            prob_pct = int(scenario_data.probability * 100)
            target = f"${scenario_data.target_price:,.0f}" if scenario_data.target_price else "undefined"
            summary += f"  - Primary Scenario: {scenario_name.replace('_', ' ').title()} ({prob_pct}% probability, target {target})\n"
        
        if observations:
            summary += "  - Notable Observations:\n"
            for obs in observations[:3]:
                summary += f"    - {obs.title()}\n"
        
        summary += f"  - Key Pivot Level: {key_pivot}"
        
        return summary
    
    def _format_probabilistic_analysis(self, confluence_data, probabilities) -> str:
        """Format probabilistic risk assessment section."""
        
        if not probabilities and not confluence_data:
            return "- Probabilistic Analysis\n  - Insufficient data for probabilistic assessment"
        
        section = "- Probabilistic Risk Assessment\n"
        
        if probabilities:
            bull_prob = probabilities.bullish_probability * 100
            bear_prob = probabilities.bearish_probability * 100
            neutral_prob = probabilities.neutral_probability * 100
            confidence = probabilities.confidence_level * 100
            
            section += f"  - Monte Carlo Simulation Results (1,000 iterations):\n"
            section += f"    - Bullish Outcome Probability: {bull_prob:.1f}%\n"
            section += f"    - Bearish Outcome Probability: {bear_prob:.1f}%\n"
            section += f"    - Neutral Outcome Probability: {neutral_prob:.1f}%\n"
            section += f"    - Statistical Confidence Level: {confidence:.0f}%\n"
        
        if confluence_data:
            section += f"  - Signal Confluence Analysis:\n"
            bull_conf = confluence_data.bullish_confluence
            bear_conf = confluence_data.bearish_confluence
            ratio = confluence_data.confluence_ratio
            
            section += f"    - Bullish Signal Confluence Score: {bull_conf:.1f}\n"
            section += f"    - Bearish Signal Confluence Score: {bear_conf:.1f}\n"
            section += f"    - Confluence Ratio: {ratio:.2f}x\n"
            
            if ratio > 2.0:
                section += f"    - Assessment: Strong confluence supporting directional bias\n"
            elif ratio > 1.5:
                section += f"    - Assessment: Moderate confluence with mixed signals\n"
            else:
                section += f"    - Assessment: Weak confluence, conflicting signals present\n"
        
        return section
    
    def _format_scenario_breakdown(self, scenarios) -> str:
        """Format scenario analysis breakdown."""
        if not scenarios:
            return "- Scenario Analysis\n  - Insufficient data for scenario modeling"
        
        section = "- Bull/Bear/Base Case Scenario Analysis\n"
        
        scenario_dict = {
            'bull_case': scenarios.bull_case,
            'base_case': scenarios.base_case,
            'bear_case': scenarios.bear_case
        }
        
        for scenario_name, scenario_data in scenario_dict.items():
            prob_pct = scenario_data.probability * 100
            target = scenario_data.target_price
            description = scenario_data.scenario_description
            drivers = scenario_data.key_drivers
            risk_reward = scenario_data.risk_reward_ratio
            
            scenario_title = scenario_name.replace('_', ' ').title()
            section += f"  - {scenario_title} ({prob_pct:.0f}% probability):\n"
            section += f"    - Target Price: ${target:,.0f}\n"
            section += f"    - Description: {description}\n"
            section += f"    - Risk/Reward Ratio: {risk_reward:.2f}\n"
            
            if drivers:
                section += f"    - Key Drivers: {', '.join(drivers)}\n"
            
            section += "\n"
        
        return section.rstrip()
    
    def _format_category_analysis(self, category: str) -> str:
        """Format analysis section for a specific category."""
        category_indicators = self.config.get_category_indicators(category)
        category_titles = {
            'trend': 'Trend',
            'momentum': 'Momentum', 
            'volatility': 'Volatility & Channels',
            'support_resistance': 'Support & Resistance'
        }
        
        section = f"- Indicator Breakdown\n  - {category_titles.get(category, category.title())}"
        
        for result in self.results:
            if result.name in category_indicators:
                description = FormatterFactory.format_indicator(result)
                section += f"\n    - {result.name}: {description}"
        
        return section
    
    def generate_summary(self, symbol: str = "BTC/USDT", timeframe: str = "1d") -> str:
        """Generate a coherent narrative summary (legacy method)."""
        structured_data = self.get_structured_analysis(symbol, timeframe)
        return structured_data["detailed_breakdown"]["full_markdown"]
    
    def _create_empty_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Create empty analysis structure when no results are available."""
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
                "sentiment_score": 0.0,
                "trend_direction": "sideways",
                "trend_strength": "weak",
                "support_levels": [],
                "resistance_levels": [],
                "support_range": "undefined",
                "resistance_range": "undefined",
                "signal_distribution": {"bullish": 0, "bearish": 0, "neutral": 0},
                "momentum": "insufficient data",
                "key_observations": [],
                "key_pivot_level": None,
                "bullish_probability": 0.0,
                "bearish_probability": 0.0,
                "neutral_probability": 1.0,
                "confidence_level": 0.0,
                "confluence_score": 0.0,
                "bullish_confluence": 0.0,
                "bearish_confluence": 0.0,
                "primary_scenario": "base_case",
                "primary_scenario_probability": 1.0,
                "primary_target": 0.0
            },
            "indicators": {"trend": {}, "momentum": {}, "volatility": {}, "support_resistance": {}},
            "probabilistic_assessment": {"bullish_probability": 0.0, "bearish_probability": 0.0, 
                                       "neutral_probability": 1.0, "confidence_level": 0.0},
            "confluence_analysis": {"bullish_confluence": 0.0, "bearish_confluence": 0.0, 
                                  "confluence_ratio": 0.0},
            "scenario_analysis": {
                "bull_case": {"probability": 0.0, "target_price": 0.0, "scenario_description": "Insufficient data", 
                             "key_drivers": [], "risk_reward_ratio": 0.0},
                "base_case": {"probability": 1.0, "target_price": 0.0, "scenario_description": "No data available", 
                             "key_drivers": [], "risk_reward_ratio": 0.0},
                "bear_case": {"probability": 0.0, "target_price": 0.0, "scenario_description": "Insufficient data", 
                             "key_drivers": [], "risk_reward_ratio": 0.0}
            },
            "detailed_breakdown": {
                "core_summary": f"- Market Analysis for {symbol} ({timeframe} timeframe)\n- No analysis data available",
                "trend_analysis": "- Trend Analysis\n  - No trend indicators available",
                "momentum_analysis": "- Momentum Analysis\n  - No momentum indicators available", 
                "volatility_analysis": "- Volatility Analysis\n  - No volatility indicators available",
                "support_resistance_analysis": "- Support & Resistance Analysis\n  - No support/resistance data available",
                "probabilistic_analysis": "- Probabilistic Analysis\n  - Insufficient data for probabilistic assessment",
                "scenario_analysis": "- Scenario Analysis\n  - Insufficient data for scenario modeling",
                "full_markdown": f"- Market Analysis for {symbol} ({timeframe} timeframe)\n- No analysis data available"
            }
        }