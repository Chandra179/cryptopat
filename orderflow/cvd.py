"""
Cumulative Volume Delta (CVD) calculation module.
Analyzes order flow to detect aggressive buying vs selling pressure.
"""

import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from data import get_data_collector

logger = logging.getLogger(__name__)

class CVDAnalyzer:
    """Cumulative Volume Delta analyzer for order flow analysis."""
    
    def __init__(self):
        self.collector = get_data_collector()
    
    def calculate_cvd(self, symbol: str, limit: int = 300) -> Dict:
        """
        Calculate CVD for a symbol using recent trades.
        
        Args:
            symbol: Trading pair symbol (e.g., 'XRP/USDT')
            limit: Number of recent trades to analyze
            
        Returns:
            Dict with CVD metrics and analysis
        """
        try:
            # Fetch recent trades
            trades = self.collector.fetch_recent_trades(symbol, limit)
            
            if not trades:
                return self._empty_result(symbol, "No trade data available")
            
            # Calculate CVD components
            buy_volume = 0.0
            sell_volume = 0.0
            total_volume = 0.0
            
            for trade in trades:
                amount = float(trade.get('amount', 0))
                side = trade.get('side', '').lower()
                
                total_volume += amount
                
                if side == 'buy':
                    buy_volume += amount
                elif side == 'sell':
                    sell_volume += amount
            
            # Calculate CVD and metrics
            cvd = buy_volume - sell_volume
            buy_percentage = (buy_volume / total_volume * 100) if total_volume > 0 else 0
            sell_percentage = (sell_volume / total_volume * 100) if total_volume > 0 else 0
            
            # Get current price
            ticker = self.collector.fetch_ticker(symbol)
            current_price = float(ticker.get('last', 0)) if ticker else 0
            
            # Determine dominant flow
            dominant_flow = "🔺 Aggressive Buyers" if buy_percentage > sell_percentage else "🔻 Aggressive Sellers"
            dominant_percentage = max(buy_percentage, sell_percentage)
            
            # Calculate CVD per minute (approximate)
            time_span_minutes = self._estimate_time_span(trades)
            cvd_per_minute = cvd / time_span_minutes if time_span_minutes > 0 else 0
            
            # Determine bias and confidence
            bias, confidence = self._determine_bias(cvd, buy_percentage, sell_percentage)
            
            # Check for divergence (simplified - would need price history for full analysis)
            divergence_status = self._check_divergence(cvd, current_price)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': current_price,
                'cvd': cvd,
                'cvd_per_minute': cvd_per_minute,
                'total_volume': total_volume,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'buy_percentage': buy_percentage,
                'sell_percentage': sell_percentage,
                'dominant_flow': dominant_flow,
                'dominant_percentage': dominant_percentage,
                'bias': bias,
                'confidence': confidence,
                'divergence': divergence_status,
                'trades_analyzed': len(trades)
            }
            
        except Exception as e:
            logger.error(f"Error calculating CVD for {symbol}: {e}")
            return self._empty_result(symbol, f"Error: {e}")
    
    def _estimate_time_span(self, trades: List[Dict]) -> float:
        """Estimate time span of trades in minutes."""
        if len(trades) < 2:
            return 1.0  # Default to 1 minute
        
        try:
            first_timestamp = trades[0].get('timestamp', 0)
            last_timestamp = trades[-1].get('timestamp', 0)
            
            # Convert milliseconds to minutes
            time_span_ms = abs(last_timestamp - first_timestamp)
            time_span_minutes = time_span_ms / (1000 * 60)
            
            return max(time_span_minutes, 1.0)  # Minimum 1 minute
            
        except Exception:
            return 1.0
    
    def _determine_bias(self, cvd: float, buy_pct: float, sell_pct: float) -> Tuple[str, str]:
        """Determine market bias and confidence level."""
        # Calculate dominance strength
        dominance = abs(buy_pct - sell_pct)
        
        if dominance >= 20:
            confidence = "HIGH"
        elif dominance >= 10:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        if buy_pct > sell_pct:
            if dominance >= 15:
                bias = "📈 Long Setup"
            else:
                bias = "📈 Bullish Lean"
        else:
            if dominance >= 15:
                bias = "📉 Short Setup"
            else:
                bias = "📉 Bearish Lean"
        
        return bias, confidence
    
    def _check_divergence(self, cvd: float, current_price: float) -> str:
        """
        Check for CVD divergence (simplified version).
        Full implementation would require price history comparison.
        """
        # Simplified divergence check - would need historical data for full analysis
        if abs(cvd) > 1000:  # Arbitrary threshold for demonstration
            if cvd > 0:
                return "⚠️ Potential Bullish Divergence"
            else:
                return "⚠️ Potential Bearish Divergence"
        
        return "✅ No Clear Divergence"
    
    def _empty_result(self, symbol: str, error_msg: str) -> Dict:
        """Return empty result structure for error cases."""
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'error': error_msg,
            'current_price': 0,
            'cvd': 0,
            'cvd_per_minute': 0,
            'total_volume': 0,
            'buy_volume': 0,
            'sell_volume': 0,
            'buy_percentage': 0,
            'sell_percentage': 0,
            'dominant_flow': "❌ No Data",
            'dominant_percentage': 0,
            'bias': "❌ Unknown",
            'confidence': "NONE",
            'divergence': "❌ No Data",
            'trades_analyzed': 0
        }

def format_cvd_output(cvd_data: Dict, timeframe: str = "Live") -> str:
    """
    Format CVD analysis results for terminal output.
    
    Args:
        cvd_data: CVD analysis results
        timeframe: Timeframe description for display
        
    Returns:
        Formatted string for terminal output
    """
    timestamp = cvd_data['timestamp'].strftime("%Y-%m-%d %H:%M")
    symbol = cvd_data['symbol']
    price = cvd_data['current_price']
    cvd = cvd_data['cvd']
    cvd_per_min = cvd_data['cvd_per_minute']
    volume = cvd_data['total_volume']
    dominant_flow = cvd_data['dominant_flow']
    dominant_pct = cvd_data['dominant_percentage']
    divergence = cvd_data['divergence']
    bias = cvd_data['bias']
    confidence = cvd_data['confidence']
    trades_count = cvd_data['trades_analyzed']
    
    # Format numbers
    cvd_formatted = f"{cvd:+.1f}K" if abs(cvd) >= 1000 else f"{cvd:+.2f}"
    cvd_per_min_formatted = f"{cvd_per_min:+.1f}K" if abs(cvd_per_min) >= 1000 else f"{cvd_per_min:+.2f}"
    volume_formatted = f"{volume:.0f}K" if volume >= 1000 else f"{volume:.1f}"
    
    output = f"""[{timestamp}] {symbol} | TF: {timeframe}
Price: {price:.4f} | ΔCVD: {cvd_formatted} | ΔCVD/min: {cvd_per_min_formatted} | Volume: {volume_formatted}
Dominant Flow: {dominant_flow} ({dominant_pct:.0f}%)
Divergence: {divergence}
Bias: {bias} | Confidence: {confidence}
Trades Analyzed: {trades_count}"""
    
    return output