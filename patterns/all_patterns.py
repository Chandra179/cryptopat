"""
All Patterns Analysis Module.

Consolidates all pattern recognition methods into a unified analysis.
Similar to /trend/all_trend.py but focused on chart patterns.
"""

import logging
from typing import Dict
from datetime import datetime
from patterns.double_bottom import analyze_double_bottom
from patterns.double_top import analyze_double_top
from patterns.head_and_shoulders import analyze_head_and_shoulders, format_head_and_shoulders_output
from patterns.inverse_head_and_shoulders import analyze_inverse_head_and_shoulders, format_inverse_head_and_shoulders_output

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AllPatternsAnalyzer:
    """Consolidated pattern analysis using all available pattern detection methods."""
    
    def __init__(self):
        """Initialize the all patterns analyzer."""
        self.pattern_methods = {
            'double_bottom': analyze_double_bottom,
            'double_top': analyze_double_top,
            'head_and_shoulders': analyze_head_and_shoulders,
            'inverse_head_and_shoulders': analyze_inverse_head_and_shoulders,
            # Future patterns will be added here:
            # 'triangle': analyze_triangle,
            # 'flag': analyze_flag,
            # 'wedge': analyze_wedge,
        }
    
    def analyze_all_patterns(self, symbol: str, timeframe: str = '4h', limit: int = 200) -> str:
        """
        Run all pattern analysis methods and return consolidated results.
        
        Args:
            symbol: Trading pair symbol (e.g., 'ADA/USDT')
            timeframe: Timeframe for analysis
            limit: Number of candles to analyze
        
        Returns:
            Consolidated analysis results as formatted string
        """
        try:
            logger.info(f"Running all pattern analysis for {symbol} on {timeframe}")
            
            results = []
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Header
            results.append("=" * 80)
            results.append(f"📊 ALL PATTERNS ANALYSIS - {symbol} ({timeframe})")
            results.append(f"🕒 Analysis Time: {timestamp}")
            results.append(f"📈 Analyzing {limit} candles")
            results.append("=" * 80)
            
            # Run each pattern analysis
            pattern_results = {}
            for pattern_name, analysis_func in self.pattern_methods.items():
                try:
                    logger.info(f"Running {pattern_name} analysis...")
                    result = analysis_func(symbol, timeframe, limit)
                    pattern_results[pattern_name] = result
                    
                    # Format result based on pattern type
                    if pattern_name == 'head_and_shoulders':
                        # Head and Shoulders returns a dict, format it
                        formatted_result = format_head_and_shoulders_output(result)
                        results.append(formatted_result)
                    elif pattern_name == 'inverse_head_and_shoulders':
                        # Inverse Head and Shoulders returns a dict, format it
                        formatted_result = format_inverse_head_and_shoulders_output(result)
                        results.append(formatted_result)
                    else:
                        # Other patterns return formatted strings
                        results.append(result)
                    
                except Exception as e:
                    error_msg = f"❌ Error in {pattern_name} analysis: {e}"
                    logger.error(error_msg)
                    results.append(f"\n🔍 {pattern_name.upper().replace('_', ' ')} PATTERN:")
                    results.append("-" * 60)
                    results.append(error_msg)
            
            return "\n".join(results)
            
        except Exception as e:
            logger.error(f"Error in all patterns analysis: {e}")
            return f"❌ Error running all patterns analysis: {e}"

def analyze_all_patterns(symbol: str, timeframe: str = '4h', limit: int = 200) -> str:
    """
    Convenience function to run all pattern analysis.
    
    Args:
        symbol: Trading pair symbol (e.g., 'ADA/USDT')
        timeframe: Timeframe for analysis
        limit: Number of candles to analyze
    
    Returns:
        Consolidated analysis results
    """
    analyzer = AllPatternsAnalyzer()
    return analyzer.analyze_all_patterns(symbol, timeframe, limit)


if __name__ == "__main__":
    # Example usage
    result = analyze_all_patterns("ADA/USDT", "4h", 200)
    print(result)