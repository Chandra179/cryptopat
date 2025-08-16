"""
Analysis Formatter - Handles all formatting logic for analysis results.

This module separates the formatting concerns from the main analyzer,
providing clean, focused methods for generating formatted output.
"""

from typing import Dict, List, Any
from .formatters import IndicatorResult, FormatterFactory
from .config import IndicatorConfig


class AnalysisFormatter:
    """Handles formatting of analysis results into human-readable text."""
    
    def __init__(self, config: IndicatorConfig):
        self.config = config
    
    def format_enhanced_detailed_breakdown(self, symbol: str, timeframe: str, 
                                         signal_analysis: Dict, support_levels: List[float], 
                                         resistance_levels: List[float], momentum_desc: str, 
                                         observations: List[str], scenarios: Any,
                                         confluence_data: Dict, probabilities: Dict,
                                         results: List[IndicatorResult]) -> Dict[str, Any]:
        """Build enhanced detailed markdown breakdown."""
        core_summary = self.format_core_summary(
            symbol, timeframe, signal_analysis, support_levels, 
            resistance_levels, momentum_desc, observations, scenarios,
            confluence_data, probabilities, results
        )
        
        trend_analysis = self.format_category_analysis('trend', results)
        momentum_analysis = self.format_category_analysis('momentum', results)
        volatility_analysis = self.format_category_analysis('volatility', results)
        support_resistance_analysis = self.format_category_analysis('support_resistance', results)
        probabilistic_analysis = self.format_probabilistic_analysis(confluence_data, probabilities)
        scenario_breakdown = self.format_scenario_breakdown(scenarios)
        
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
    
    def format_core_summary(self, symbol: str, timeframe: str, signal_analysis: Dict, 
                           support_levels: List[float], resistance_levels: List[float], 
                           momentum_desc: str, observations: List[str], 
                           scenarios: Any, confluence_data: Dict, probabilities: Dict,
                           results: List[IndicatorResult]) -> str:
        """Format enhanced core summary with probabilistic and scenario data."""
        from .analysis_core import LevelAnalyzer
        level_analyzer = LevelAnalyzer()
        
        support_range = level_analyzer.find_key_levels(support_levels)
        resistance_range = level_analyzer.find_key_levels(resistance_levels)
        sentiment_phrase = self.config.get_sentiment_phrase(signal_analysis['sentiment'])
        
        trend_direction = self._get_trend_direction_text(signal_analysis['score'])
        strength_desc = self._get_trend_strength_text(signal_analysis['score'])
        
        pivot_result = next((r for r in results if r.name == 'Pivot Point'), None)
        key_pivot = f"${pivot_result.value:,.0f}" if pivot_result and pivot_result.value else "undefined"
        
        summary = self._build_basic_summary(symbol, timeframe, sentiment_phrase, trend_direction, strength_desc)
        summary += self._add_probabilistic_info(probabilities)
        summary += self._add_confluence_info(confluence_data)
        summary += self._add_levels_info(support_range, resistance_range)
        summary += self._add_momentum_and_signals(momentum_desc, signal_analysis['counts'])
        summary += self._add_scenario_info(scenarios)
        summary += self._add_observations(observations)
        summary += f"  - Key Pivot Level: {key_pivot}"
        
        return summary
    
    def _build_basic_summary(self, symbol: str, timeframe: str, sentiment_phrase: str,
                           trend_direction: str, strength_desc: str) -> str:
        """Build basic summary information."""
        summary = f"- Market Analysis for {symbol} ({timeframe} timeframe)\n"
        summary += "- Enhanced Core Summary\n"
        summary += f"  - Overall Sentiment: {sentiment_phrase.title()}\n"
        summary += f"  - Trend Direction: {trend_direction} with {strength_desc} conviction\n"
        return summary
    
    def _add_probabilistic_info(self, probabilities: Dict) -> str:
        """Add probabilistic assessment information."""
        if not probabilities:
            return ""
        
        bull_prob = int(probabilities.bullish_probability * 100)
        bear_prob = int(probabilities.bearish_probability * 100)
        neutral_prob = int(probabilities.neutral_probability * 100)
        confidence = int(probabilities.confidence_level * 100)
        
        info = f"  - Probability Assessment: {bull_prob}% bullish, {bear_prob}% bearish, {neutral_prob}% neutral\n"
        info += f"  - Confidence Level: {confidence}%\n"
        return info
    
    def _add_confluence_info(self, confluence_data: Dict) -> str:
        """Add confluence analysis information."""
        if not confluence_data:
            return ""
        
        confluence_ratio = confluence_data.confluence_ratio
        if confluence_ratio > 2.0:
            return f"  - Signal Confluence: Strong ({confluence_ratio:.1f}x)\n"
        elif confluence_ratio > 1.5:
            return f"  - Signal Confluence: Moderate ({confluence_ratio:.1f}x)\n"
        else:
            return f"  - Signal Confluence: Weak ({confluence_ratio:.1f}x)\n"
    
    def _add_levels_info(self, support_range: str, resistance_range: str) -> str:
        """Add support and resistance levels information."""
        info = ""
        if support_range != "undefined":
            info += f"  - Support Levels: {support_range}\n"
        if resistance_range != "undefined":
            info += f"  - Resistance Levels: {resistance_range}\n"
        return info
    
    def _add_momentum_and_signals(self, momentum_desc: str, counts: Dict) -> str:
        """Add momentum and signal distribution information."""
        info = f"  - Momentum: {momentum_desc.title()}\n"
        info += f"  - Signal Distribution: {counts['bullish']} bullish, {counts['bearish']} bearish, {counts['neutral']} neutral\n"
        return info
    
    def _add_scenario_info(self, scenarios: Any) -> str:
        """Add primary scenario information."""
        if not scenarios:
            return ""
        
        primary_scenario = scenarios.primary_scenario if hasattr(scenarios, 'primary_scenario') else None
        if not primary_scenario:
            return ""
        
        scenario_name, scenario_data = primary_scenario
        prob_pct = int(scenario_data.probability * 100)
        target = f"${scenario_data.target_price:,.0f}" if scenario_data.target_price else "undefined"
        return f"  - Primary Scenario: {scenario_name.replace('_', ' ').title()} ({prob_pct}% probability, target {target})\n"
    
    def _add_observations(self, observations: List[str]) -> str:
        """Add notable observations information."""
        if not observations:
            return ""
        
        info = "  - Notable Observations:\n"
        for obs in observations[:3]:
            info += f"    - {obs.title()}\n"
        return info
    
    def _get_trend_direction_text(self, score: float) -> str:
        """Get trend direction description."""
        if score > 0:
            return "Upward trend"
        elif score < 0:
            return "Downward trend"
        else:
            return "Sideways movement"
    
    def _get_trend_strength_text(self, score: float) -> str:
        """Get trend strength description."""
        abs_score = abs(score)
        if abs_score > 0.7:
            return "strong"
        elif abs_score > 0.3:
            return "moderate"
        else:
            return "weak"
    
    def format_probabilistic_analysis(self, confluence_data: Dict, probabilities: Dict) -> str:
        """Format probabilistic risk assessment section."""
        if not probabilities and not confluence_data:
            return "- Probabilistic Analysis\n  - Insufficient data for probabilistic assessment"
        
        section = "- Probabilistic Risk Assessment\n"
        section += self._format_monte_carlo_results(probabilities)
        section += self._format_confluence_analysis(confluence_data)
        
        return section
    
    def _format_monte_carlo_results(self, probabilities: Dict) -> str:
        """Format Monte Carlo simulation results."""
        if not probabilities:
            return ""
        
        bull_prob = probabilities.bullish_probability * 100
        bear_prob = probabilities.bearish_probability * 100
        neutral_prob = probabilities.neutral_probability * 100
        confidence = probabilities.confidence_level * 100
        
        section = f"  - Monte Carlo Simulation Results (1,000 iterations):\n"
        section += f"    - Bullish Outcome Probability: {bull_prob:.1f}%\n"
        section += f"    - Bearish Outcome Probability: {bear_prob:.1f}%\n"
        section += f"    - Neutral Outcome Probability: {neutral_prob:.1f}%\n"
        section += f"    - Statistical Confidence Level: {confidence:.0f}%\n"
        
        return section
    
    def _format_confluence_analysis(self, confluence_data: Dict) -> str:
        """Format signal confluence analysis."""
        if not confluence_data:
            return ""
        
        section = f"  - Signal Confluence Analysis:\n"
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
    
    def format_scenario_breakdown(self, scenarios: Any) -> str:
        """Format scenario analysis breakdown."""
        if not scenarios:
            return "- Scenario Analysis\n  - Insufficient data for scenario modeling"
        
        section = "- Bull/Bear/Base Case Scenario Analysis\n"
        
        scenario_dict = {
            'bull_case': getattr(scenarios, 'bull_case', None),
            'base_case': getattr(scenarios, 'base_case', None),
            'bear_case': getattr(scenarios, 'bear_case', None)
        }
        
        for scenario_name, scenario_data in scenario_dict.items():
            if scenario_data:
                section += self._format_single_scenario(scenario_name, scenario_data)
        
        return section.rstrip()
    
    def _format_single_scenario(self, scenario_name: str, scenario_data: Any) -> str:
        """Format a single scenario."""
        prob_pct = scenario_data.probability * 100
        target = scenario_data.target_price
        description = scenario_data.scenario_description
        drivers = getattr(scenario_data, 'key_drivers', [])
        risk_reward = scenario_data.risk_reward_ratio
        
        scenario_title = scenario_name.replace('_', ' ').title()
        section = f"  - {scenario_title} ({prob_pct:.0f}% probability):\n"
        section += f"    - Target Price: ${target:,.0f}\n"
        section += f"    - Description: {description}\n"
        section += f"    - Risk/Reward Ratio: {risk_reward:.2f}\n"
        
        if drivers:
            section += f"    - Key Drivers: {', '.join(drivers)}\n"
        
        section += "\n"
        return section
    
    def format_category_analysis(self, category: str, results: List[IndicatorResult]) -> str:
        """Format analysis section for a specific category."""
        category_indicators = self.config.get_category_indicators(category)
        category_titles = {
            'trend': 'Trend',
            'momentum': 'Momentum', 
            'volatility': 'Volatility & Channels',
            'support_resistance': 'Support & Resistance'
        }
        
        section = f"- Indicator Breakdown\n  - {category_titles.get(category, category.title())}"
        
        for result in results:
            if result.name in category_indicators:
                description = FormatterFactory.format_indicator(result)
                section += f"\n    - {result.name}: {description}"
        
        return section