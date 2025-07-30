"""
Standardized output formatter for all trend analysis modules.
Provides consistent timestamp, metrics, signal, and trend indicator formatting.
"""

from datetime import datetime
from typing import Dict, Any, Optional


class OutputFormatter:
    """Standardized output formatter for trend analysis results."""
    
    @staticmethod
    def format_analysis_output(timestamp: int, metrics: Dict[str, Any], 
                              signal: str, trend: str, 
                              price: Optional[float] = None,
                              symbol: Optional[str] = None,
                              timeframe: Optional[str] = None,
                              multiline: bool = False) -> str:
        """
        Format analysis output with standardized structure.
        
        Args:
            timestamp: Unix timestamp in milliseconds
            metrics: Dictionary of metric names and values
            signal: Trading signal (BUY, SELL, HOLD, NONE, etc.)
            trend: Trend direction (BULLISH, BEARISH, NEUTRAL, etc.)
            price: Optional current price
            symbol: Optional trading symbol (e.g., BTC/USDT)
            timeframe: Optional chart timeframe (e.g., 1h, 4h, 1d)
            multiline: If True, format with better visual hierarchy
            
        Returns:
            Formatted output string (single or multi-line)
        """
        # Convert timestamp to readable format
        dt = datetime.fromtimestamp(timestamp / 1000)
        timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Build context string (symbol/timeframe)
        context_parts = []
        if symbol:
            context_parts.append(symbol)
        if timeframe:
            context_parts.append(timeframe)
        context_str = f"[{' | '.join(context_parts)}] " if context_parts else ""
        
        # Format metrics with angle brackets
        metric_parts = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if abs(value) >= 1000:
                    metric_parts.append(f"<{key}>: {value:.0f}")
                elif abs(value) >= 1:
                    metric_parts.append(f"<{key}>: {value:.2f}")
                else:
                    metric_parts.append(f"<{key}>: {value:.4f}")
            else:
                metric_parts.append(f"<{key}>: {value}")
        
        # Get trend emoji
        trend_emoji = OutputFormatter.get_trend_emoji(trend)
        
        if multiline:
            # Multi-line format with better visual hierarchy
            lines = []
            lines.append(f"┌─ {context_str}[{timestamp_str}]")
            
            # Group metrics by category for better readability
            primary_metrics = []
            secondary_metrics = []
            
            for part in metric_parts:
                key = part.split(":")[0].replace("<", "").replace(">", "")
                if key in ['PRICE', 'SIGNAL', 'CONFIDENCE', 'FIB_LEVEL', 'PATTERN', 'BOS', 'CHOCH']:
                    primary_metrics.append(part)
                else:
                    secondary_metrics.append(part)
            
            if primary_metrics:
                lines.append(f"├─ Primary: {' | '.join(primary_metrics)}")
            if secondary_metrics:
                lines.append(f"├─ Metrics: {' | '.join(secondary_metrics)}")
            
            if price is not None:
                lines.append(f"├─ Price: {price:.4f}")
            
            lines.append(f"└─ Signal: {signal} | {trend_emoji} {trend}")
            
            return "\n".join(lines)
        else:
            # Single-line format (compatible with existing code)
            metrics_str = " | ".join(metric_parts)
            
            # Add price if provided
            if price is not None:
                price_str = f" | <PRICE>: {price:.4f}"
            else:
                price_str = ""
            
            return f"{context_str}[{timestamp_str}] {metrics_str}{price_str} | Signal: {signal} | {trend_emoji} {trend}"
    
    @staticmethod
    def get_trend_emoji(trend: str) -> str:
        """
        Get emoji for trend direction.
        
        Args:
            trend: Trend direction string
            
        Returns:
            Corresponding emoji
        """
        trend_upper = trend.upper()
        
        if trend_upper in ['BULLISH', 'BUY', 'UP', 'LONG']:
            return "📈"
        elif trend_upper in ['BEARISH', 'SELL', 'DOWN', 'SHORT']:
            return "📉"
        elif trend_upper in ['NEUTRAL', 'SIDEWAYS', 'CONSOLIDATION']:
            return "➖"
        elif trend_upper in ['STRONG_BULLISH', 'VERY_BULLISH']:
            return "🚀"
        elif trend_upper in ['STRONG_BEARISH', 'VERY_BEARISH']:
            return "🔻"
        elif trend_upper in ['VOLATILE', 'CHOPPY']:
            return "🌊"
        elif trend_upper in ['UNCERTAIN', 'MIXED']:
            return "❓"
        else:
            return "➖"  # Default to neutral
    
    @staticmethod
    def format_smc_output(timestamp: int, bos: bool, choch: bool, ob_hit: bool,
                         signal: str, trend: str, confidence: int = 0,
                         price: Optional[float] = None, symbol: Optional[str] = None,
                         timeframe: Optional[str] = None, multiline: bool = False) -> str:
        """
        Format SMC (Smart Money Concepts) analysis output.
        
        Args:
            timestamp: Unix timestamp in milliseconds
            bos: Break of Structure detected
            choch: Change of Character detected
            ob_hit: Order Block hit
            signal: Trading signal
            trend: Trend direction
            confidence: Confidence percentage
            price: Optional current price
            symbol: Optional trading symbol
            timeframe: Optional chart timeframe
            multiline: If True, use multi-line format
            
        Returns:
            Formatted output string
        """
        metrics = {
            "BOS": "YES" if bos else "NO",
            "CHOCH": "YES" if choch else "NO",
            "OB_HIT": "YES" if ob_hit else "NO"
        }
        
        if confidence > 0:
            metrics["CONFIDENCE"] = f"{confidence}%"
        
        return OutputFormatter.format_analysis_output(
            timestamp, metrics, signal, trend, price, symbol, timeframe, multiline
        )