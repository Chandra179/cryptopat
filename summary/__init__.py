"""
Summary Module

Refactored analysis summary system with improved maintainability.
"""

from .analyzer import AnalysisSummarizer
from .formatters import IndicatorResult
from .config import IndicatorConfig

# Global summarizer instance for backward compatibility
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

def get_structured_analysis(symbol: str = "BTC/USDT", timeframe: str = "1d", current_price: float = None):
    """Generate structured analysis data matching output_schema.json."""
    return _summarizer.get_structured_analysis(symbol, timeframe, current_price)

__all__ = [
    'AnalysisSummarizer',
    'IndicatorResult', 
    'IndicatorConfig',
    'add_indicator_result',
    'clear_all_results',
    'generate_analysis_summary',
    'get_structured_analysis'
]