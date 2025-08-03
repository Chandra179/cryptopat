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
    
    def analyze(self, symbol: str, timeframe: str, limit: int, ohlcv_data: Optional[List] = None) -> Dict:
        """
        Analyze CVD patterns for given symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for analysis (not used in CVD but kept for consistency)
            limit: Number of recent trades to analyze
            ohlcv_data: Optional pre-fetched OHLCV data (not used in CVD analysis)
            
        Returns:
            Analysis results dictionary
        """
        try:
            cvd_data = self.calculate_cvd(symbol, limit)
            
            if 'error' in cvd_data:
                return {
                    'error': cvd_data['error'],
                    'success': False,
                    'symbol': symbol,
                    'timeframe': timeframe
                }
            
            # Transform CVD data to match expected output format
            result = {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'analysis_time': cvd_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': int(cvd_data['timestamp'].timestamp() * 1000),
                'current_price': cvd_data['current_price'],
                'pattern_detected': abs(cvd_data['cvd']) > 100,  # Pattern detected if significant CVD
                
                # CVD specific metrics
                'cvd_value': cvd_data['cvd'],
                'cvd_side': cvd_data['cvd_side'],
                'cvd_price': cvd_data['cvd_price'],
                'cvd_per_minute': cvd_data['cvd_per_minute'],
                'total_volume': cvd_data['total_volume'],
                'buy_volume': cvd_data['buy_volume'],
                'sell_volume': cvd_data['sell_volume'],
                'buy_percentage': cvd_data['buy_percentage'],
                'sell_percentage': cvd_data['sell_percentage'],
                'dominant_flow': cvd_data['dominant_flow'],
                'dominant_percentage': cvd_data['dominant_percentage'],
                'bias': cvd_data['bias'],
                'confidence_score': self._map_confidence_to_score(cvd_data['confidence']),
                'divergence': cvd_data['divergence'],
                'trades_analyzed': cvd_data['trades_analyzed'],
                'classification_method': cvd_data['classification_method'],
                'classification_accuracy': cvd_data['classification_accuracy'],
                'bid_ask_available': cvd_data['bid_ask_available'],
                
                # Trading signals derived from CVD
                'signal': self._determine_signal(cvd_data),
                'entry_window': self._determine_entry_window(cvd_data),
                'exit_trigger': self._determine_exit_trigger(cvd_data),
                'rr_ratio': 0,  # Not applicable for CVD analysis
                
                # Raw CVD data
                'raw_data': cvd_data
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f'CVD analysis failed: {str(e)}',
                'success': False,
                'symbol': symbol,
                'timeframe': timeframe
            }
    
    def _map_confidence_to_score(self, confidence: str) -> int:
        """Map confidence string to numerical score."""
        confidence_map = {
            'HIGH': 85,
            'MEDIUM': 65,
            'LOW': 35,
            'NONE': 0
        }
        return confidence_map.get(confidence, 50)
    
    def _determine_signal(self, cvd_data: Dict) -> str:
        """Determine trading signal from CVD data."""
        buy_pct = cvd_data['buy_percentage']
        sell_pct = cvd_data['sell_percentage']
        dominance = abs(buy_pct - sell_pct)
        
        if dominance >= 15:
            return 'BUY' if buy_pct > sell_pct else 'SELL'
        else:
            return 'HOLD'
    
    def _determine_entry_window(self, cvd_data: Dict) -> str:
        """Determine optimal entry window based on CVD metrics."""
        confidence = cvd_data['confidence']
        dominance = abs(cvd_data['buy_percentage'] - cvd_data['sell_percentage'])
        
        if confidence == 'HIGH' and dominance >= 20:
            return "Optimal now"
        elif confidence in ['HIGH', 'MEDIUM'] and dominance >= 10:
            return "Good entry opportunity"
        else:
            return "Wait for clearer signal"
    
    def _determine_exit_trigger(self, cvd_data: Dict) -> str:
        """Determine exit trigger based on CVD analysis."""
        signal = self._determine_signal(cvd_data)
        
        if signal == 'BUY':
            return "Exit if selling pressure increases significantly"
        elif signal == 'SELL':
            return "Exit if buying pressure increases significantly"
        else:
            return "Monitor for clear directional flow"
    
    def calculate_cvd(self, symbol: str, limit: int = 300) -> Dict:
        """
        Calculate CVD for a symbol using recent trades with enhanced classification.
        
        Args:
            symbol: Trading pair symbol (e.g., 'XRP/USDT')
            limit: Number of recent trades to analyze
            
        Returns:
            Dict with CVD metrics and analysis
        """
        try:
            # Fetch recent trades and order book
            trades = self.collector.fetch_recent_trades(symbol, limit)
            order_book = self.collector.fetch_order_book(symbol, limit=5)
            
            if not trades:
                return self._empty_result(symbol, "No trade data available")
            
            # Get bid/ask for price-based classification
            bid_ask_midpoint = self._get_bid_ask_midpoint(order_book)
            
            # Calculate CVD components with enhanced classification
            buy_volume_side = 0.0  # Based on exchange side field
            sell_volume_side = 0.0
            buy_volume_price = 0.0  # Based on price classification
            sell_volume_price = 0.0
            total_volume = 0.0
            classification_matches = 0
            
            # Sort trades by timestamp for uptick/downtick analysis
            sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', 0))
            
            for i, trade in enumerate(sorted_trades):
                amount = float(trade.get('amount', 0))
                side = trade.get('side', '').lower()
                
                total_volume += amount
                
                # Side-based classification (original method)
                if side == 'buy':
                    buy_volume_side += amount
                elif side == 'sell':
                    sell_volume_side += amount
                
                # Price-based classification
                price_side = self._classify_trade_by_price(trade, bid_ask_midpoint, sorted_trades, i)
                if price_side == 'buy':
                    buy_volume_price += amount
                elif price_side == 'sell':
                    sell_volume_price += amount
                
                # Track classification agreement
                if side and price_side and side == price_side:
                    classification_matches += 1
            
            # Calculate CVD using both methods
            cvd_side = buy_volume_side - sell_volume_side
            cvd_price = buy_volume_price - sell_volume_price
            
            # Use price-based CVD as primary, side-based as validation
            cvd = cvd_price if bid_ask_midpoint else cvd_side
            buy_volume = buy_volume_price if bid_ask_midpoint else buy_volume_side
            sell_volume = sell_volume_price if bid_ask_midpoint else sell_volume_side
            
            buy_percentage = (buy_volume / total_volume * 100) if total_volume > 0 else 0
            sell_percentage = (sell_volume / total_volume * 100) if total_volume > 0 else 0
            
            # Calculate classification accuracy
            classification_accuracy = (classification_matches / len(trades) * 100) if trades else 0
            
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
                'cvd_side': cvd_side,
                'cvd_price': cvd_price,
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
                'trades_analyzed': len(trades),
                'classification_method': 'price-based' if bid_ask_midpoint else 'side-based',
                'classification_accuracy': classification_accuracy,
                'bid_ask_available': bool(bid_ask_midpoint)
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
    
    def _check_divergence(self, cvd_value: float, current_price: float) -> str:
        """
        Check for CVD divergence (simplified version).
        Full implementation would require price history comparison.
        """
        # Simplified divergence check - would need historical data for full analysis
        if abs(cvd_value) > 1000:  # Arbitrary threshold for demonstration
            if cvd_value > 0:
                return "⚠️ Potential Bullish Divergence"
            else:
                return "⚠️ Potential Bearish Divergence"
        
        return "✅ No Clear Divergence"
    
    def _get_bid_ask_midpoint(self, order_book: Dict) -> Optional[float]:
        """
        Calculate bid-ask midpoint from order book data.
        
        Args:
            order_book: Order book data with bids and asks
            
        Returns:
            Midpoint price or None if unavailable
        """
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                raise ValueError("Order book missing bids or asks data")
            
            best_bid = float(bids[0][0])  # Highest bid
            best_ask = float(asks[0][0])  # Lowest ask
            
            return (best_bid + best_ask) / 2
            
        except (IndexError, ValueError, TypeError) as e:
            logger.error(f"Error calculating midpoint price: {e}")
            raise RuntimeError(f"Failed to calculate midpoint price: {e}")
    
    def _classify_trade_by_price(self, trade: Dict, midpoint: Optional[float], 
                                sorted_trades: List[Dict], index: int) -> str:
        """
        Classify trade as buy/sell using price-based methods.
        
        Args:
            trade: Trade data
            midpoint: Bid-ask midpoint price
            sorted_trades: All trades sorted by timestamp
            index: Current trade index
            
        Returns:
            'buy', 'sell', or 'unknown'
        """
        try:
            price = float(trade.get('price', 0))
            
            # Method 1: Compare to bid-ask midpoint
            if midpoint:
                if price > midpoint:
                    return 'buy'  # Above midpoint = aggressive buy
                elif price < midpoint:
                    return 'sell'  # Below midpoint = aggressive sell
            
            # Method 2: Uptick/Downtick rule (if no midpoint available)
            if index > 0:
                prev_price = float(sorted_trades[index - 1].get('price', 0))
                if price > prev_price:
                    return 'buy'  # Uptick = buy pressure
                elif price < prev_price:
                    return 'sell'  # Downtick = sell pressure
            
            return 'unknown'
            
        except (ValueError, TypeError):
            return 'unknown'
    
    def _empty_result(self, symbol: str, error_msg: str) -> Dict:
        """Return empty result structure for error cases."""
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'error': error_msg,
            'current_price': 0,
            'cvd': 0,
            'cvd_side': 0,
            'cvd_price': 0,
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
            'trades_analyzed': 0,
            'classification_method': 'none',
            'classification_accuracy': 0,
            'bid_ask_available': False
        }

def format_cvd_output(cvd_data: Dict, timeframe: str = "Live") -> str:
    """
    Format CVD analysis results for terminal output with enhanced metrics.
    
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
    cvd_side = cvd_data.get('cvd_side', 0)
    cvd_price = cvd_data.get('cvd_price', 0)
    cvd_per_min = cvd_data['cvd_per_minute']
    volume = cvd_data['total_volume']
    dominant_flow = cvd_data['dominant_flow']
    dominant_pct = cvd_data['dominant_percentage']
    divergence = cvd_data['divergence']
    bias = cvd_data['bias']
    confidence = cvd_data['confidence']
    trades_count = cvd_data['trades_analyzed']
    classification_method = cvd_data.get('classification_method', 'side-based')
    classification_accuracy = cvd_data.get('classification_accuracy', 0)
    bid_ask_available = cvd_data.get('bid_ask_available', False)
    
    # Format numbers
    cvd_formatted = f"{cvd:+.1f}K" if abs(cvd) >= 1000 else f"{cvd:+.2f}"
    cvd_per_min_formatted = f"{cvd_per_min:+.1f}K" if abs(cvd_per_min) >= 1000 else f"{cvd_per_min:+.2f}"
    volume_formatted = f"{volume:.0f}K" if volume >= 1000 else f"{volume:.1f}"
    
    # Classification method indicator
    method_indicator = "📊 Price-Based" if classification_method == 'price-based' else "📈 Side-Based"
    accuracy_info = f" | Accuracy: {classification_accuracy:.0f}%" if classification_method == 'price-based' else ""
    
    # CVD comparison (if both methods available)
    cvd_comparison = ""
    if cvd_side != cvd_price and bid_ask_available:
        cvd_side_fmt = f"{cvd_side:+.1f}K" if abs(cvd_side) >= 1000 else f"{cvd_side:+.2f}"
        cvd_price_fmt = f"{cvd_price:+.1f}K" if abs(cvd_price) >= 1000 else f"{cvd_price:+.2f}"
        cvd_comparison = f"\nCVD Comparison: Side={cvd_side_fmt} | Price={cvd_price_fmt}"
    
    output = f"""[{timestamp}] {symbol} | TF: {timeframe}
Price: {price:.4f} | ΔCVD: {cvd_formatted} | ΔCVD/min: {cvd_per_min_formatted} | Volume: {volume_formatted}
Method: {method_indicator}{accuracy_info}{cvd_comparison}
Dominant Flow: {dominant_flow} ({dominant_pct:.0f}%)
Divergence: {divergence}
Bias: {bias} | Confidence: {confidence}
Trades Analyzed: {trades_count}"""
    
    return output

def display_buyer_seller_pressure(symbol: str, limit: int = 300):
    """
    Display buyer/seller pressure using CVD analysis in terminal.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        limit: Number of recent trades to analyze
        
    Returns:
        Formatted analysis string
    """
    analyzer = CVDAnalyzer()
    cvd_data = analyzer.calculate_cvd(symbol, limit)
    
    if 'error' in cvd_data:
        return f"\n❌ Error analyzing {symbol}: {cvd_data['error']}"
    
    # Extract data
    buy_pct = cvd_data['buy_percentage']
    sell_pct = cvd_data['sell_percentage'] 
    buy_vol = cvd_data['buy_volume']
    sell_vol = cvd_data['sell_volume']
    total_vol = cvd_data['total_volume']
    cvd = cvd_data['cvd']
    price = cvd_data['current_price']
    bias = cvd_data['bias']
    confidence = cvd_data['confidence']
    dominant_flow = cvd_data['dominant_flow']
    trades_count = cvd_data['trades_analyzed']
    method = cvd_data['classification_method']
    accuracy = cvd_data['classification_accuracy']
    
    # Build output string
    output = f"\n{'='*70}\n"
    output += f"CVD BUYER/SELLER PRESSURE ANALYSIS: {symbol}\n"
    output += f"{'='*70}\n"
    output += f"💰 Current Price: ${price:.4f}\n"
    output += f"📊 Method: {method.title()} | Trades: {trades_count}\n"
    if method == 'price-based':
        output += f"🎯 Classification Accuracy: {accuracy:.1f}%\n"
    
    # Buyer/Seller percentages with visual bars
    output += f"\n📈 BUYER/SELLER PRESSURE\n"
    output += f"{'─'*40}\n"
    
    # Visual bar representation (max 30 chars)
    buy_bar = '█' * int(buy_pct * 30 / 100) 
    sell_bar = '█' * int(sell_pct * 30 / 100)
    
    output += f"🟢 Buyers:  {buy_bar:<30} {buy_pct:.1f}%\n"
    output += f"🔴 Sellers: {sell_bar:<30} {sell_pct:.1f}%\n"
    
    # Volume details
    output += f"\n📊 VOLUME BREAKDOWN\n"
    output += f"{'─'*40}\n"
    output += f"🟢 Buy Volume:    {buy_vol:>10.2f} ({buy_pct:.1f}%)\n"
    output += f"🔴 Sell Volume:   {sell_vol:>10.2f} ({sell_pct:.1f}%)\n"
    output += f"📊 Total Volume:  {total_vol:>10.2f}\n"
    
    # CVD and bias
    output += f"\n⚖️  CVD ANALYSIS\n"
    output += f"{'─'*40}\n"
    cvd_formatted = f"{cvd:+.2f}K" if abs(cvd) >= 1000 else f"{cvd:+.2f}"
    output += f"📊 CVD Value: {cvd_formatted}\n"
    output += f"🎯 {dominant_flow}\n"
    output += f"📈 {bias} | Confidence: {confidence}\n"
    
    # Market sentiment
    output += f"\n🔍 MARKET SENTIMENT\n"
    output += f"{'─'*40}\n"
    if buy_pct > sell_pct:
        sentiment = "🟢 BULLISH" if buy_pct - sell_pct > 10 else "🟡 SLIGHTLY BULLISH"
        pressure = f"Buyers dominating by {buy_pct - sell_pct:.1f}%"
    else:
        sentiment = "🔴 BEARISH" if sell_pct - buy_pct > 10 else "🟡 SLIGHTLY BEARISH" 
        pressure = f"Sellers dominating by {sell_pct - buy_pct:.1f}%"
    
    output += f"📊 Sentiment: {sentiment}\n"
    output += f"⚡ Pressure: {pressure}"
    
    return output

