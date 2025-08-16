"""
Indicator Formatters

Specialized formatters for different types of indicators.

Academic References:
- Lo, A., Mamaysky, H., & Wang, J. (2000). Foundations of Technical Analysis: Computational 
  Algorithms, Statistical Inference, and Empirical Implementation. Journal of Finance, 55(4), 1705-1770.
- Murphy, J. J. (1999). Technical Analysis of the Financial Markets. New York Institute of Finance.
- Bollinger, J. (2001). Bollinger on Bollinger Bands. McGraw-Hill.
- Wilder, J. W. (1978). New Concepts in Technical Trading Systems. Trend Research.
- Appel, G. (2005). Technical Analysis: Power Tools for Active Investors. Financial Times Prentice Hall.
- Ichimoku, G. (1969). Ichimoku Kinko Hyo. (Original Japanese technical analysis methodology)
- Seban, O. (2008). SuperTrend Indicator. (Original development documentation)
- Pring, M. J. (2002). Technical Analysis Explained. McGraw-Hill.
- Elder, A. (1993). Trading for a Living. John Wiley & Sons.
- Academic validation studies from Financial Innovation, Computational Economics, and MDPI Mathematics.
"""

from typing import Any, Dict
from dataclasses import dataclass


@dataclass
class IndicatorResult:
    name: str
    signal: str
    value: float = None
    strength: str = "medium"
    support: float = None
    resistance: float = None
    metadata: Dict[str, Any] = None


class IndicatorFormatter:
    """Base class for indicator formatters."""
    
    def format_description(self, result: IndicatorResult) -> str:
        """Format indicator description."""
        signal_desc = result.signal.replace('_', ' ').title()
        return signal_desc.lower()


class RSIFormatter(IndicatorFormatter):
    """Formatter for RSI indicator."""
    
    def format_description(self, result: IndicatorResult) -> str:
        if result.value:
            return f"{result.value:.0f} → {result.signal.replace('_', ' ').lower()}"
        return super().format_description(result)


class MACDFormatter(IndicatorFormatter):
    """Formatter for MACD indicator."""
    
    def format_description(self, result: IndicatorResult) -> str:
        signal_desc = result.signal.replace('_', ' ').title()
        histogram_trend = 'rising' if (result.metadata and 
                                     result.metadata.get('histogram_increasing')) else 'declining'
        return f"{signal_desc}, histogram {histogram_trend}"


class VWAPFormatter(IndicatorFormatter):
    """Formatter for VWAP indicator."""
    
    def format_description(self, result: IndicatorResult) -> str:
        if not result.metadata:
            return super().format_description(result)
        
        position = result.metadata.get('position', 'unknown')
        deviation = result.metadata.get('deviation_percent', 0)
        return f"Trading {position} VWAP with {deviation:+.0f}% deviation"


class IchimokuFormatter(IndicatorFormatter):
    """Formatter for Ichimoku Cloud indicator."""
    
    def format_description(self, result: IndicatorResult) -> str:
        if not result.metadata:
            return super().format_description(result)
        
        position = result.metadata.get('cloud_position', 'unknown')
        return f"Price {position.replace('_', ' ')}"


class SuperTrendFormatter(IndicatorFormatter):
    """Formatter for SuperTrend indicator."""
    
    def format_description(self, result: IndicatorResult) -> str:
        if not result.metadata:
            return super().format_description(result)
        
        trend = result.metadata.get('trend_direction', result.signal)
        if result.support:
            return f"{trend.title()}, support at ${result.support:,.0f}"
        return f"{trend.title()}"


class BollingerBandsFormatter(IndicatorFormatter):
    """Formatter for Bollinger Bands indicator."""
    
    def format_description(self, result: IndicatorResult) -> str:
        if not result.metadata:
            return super().format_description(result)
        
        position = result.metadata.get('position', 'unknown')
        squeeze = result.metadata.get('squeeze', False)
        expansion = result.metadata.get('expansion', False)
        
        if squeeze:
            return f"Price {position}, volatility squeeze"
        elif expansion:
            return f"Price {position}, volatility expansion"
        return f"Price {position}"


class FormatterFactory:
    """Factory for creating appropriate formatters."""
    
    FORMATTERS = {
        'RSI': RSIFormatter,
        'MACD': MACDFormatter,
        'VWAP': VWAPFormatter,
        'Ichimoku Cloud': IchimokuFormatter,
        'SuperTrend': SuperTrendFormatter,
        'Bollinger Bands': BollingerBandsFormatter,
    }
    
    @classmethod
    def get_formatter(cls, indicator_name: str) -> IndicatorFormatter:
        """Get appropriate formatter for an indicator."""
        formatter_class = cls.FORMATTERS.get(indicator_name, IndicatorFormatter)
        return formatter_class()
    
    @classmethod
    def format_indicator(cls, result: IndicatorResult) -> str:
        """Format an indicator result using appropriate formatter."""
        formatter = cls.get_formatter(result.name)
        return formatter.format_description(result)