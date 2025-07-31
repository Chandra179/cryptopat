"""
Volume Footprint Chart implementation for order flow analysis.
Creates tick-by-tick volume footprint charts showing buy/sell volume distribution.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data import get_data_collector

logger = logging.getLogger(__name__)

class FootprintAnalyzer:
    """
    Volume Footprint Chart analyzer for detecting volume-based order flow signals.
    """
    
    def __init__(self):
        """Initialize the FootprintAnalyzer."""
        self.collector = get_data_collector()
        
    def fetch_footprint_data(self, symbol: str, timeframe: str = '5m', 
                           limit: int = 50) -> Tuple[List[List], List[Dict]]:
        """
        Fetch OHLCV and trade data for footprint analysis.
        
        Args:
            symbol: Trading pair symbol (e.g., 'XRP/USDT')
            timeframe: Candle timeframe (e.g., '1m', '5m', '15m')
            limit: Number of candles to fetch
            
        Returns:
            Tuple of (ohlcv_data, trades_data)
        """
        try:
            # Fetch OHLCV data for candle structure
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
            
            # Fetch recent trades data for volume distribution
            # Calculate how many trades we need based on timeframe
            trade_limit = self._calculate_trade_limit(timeframe, limit)
            trades_data = self.collector.fetch_trades_stream(symbol, limit=trade_limit)
            
            logger.info(f"Fetched {len(ohlcv_data)} candles and {len(trades_data)} trades for {symbol}")
            return ohlcv_data, trades_data
            
        except Exception as e:
            logger.error(f"Error fetching footprint data for {symbol}: {e}")
            raise
    
    def _calculate_trade_limit(self, timeframe: str, candle_limit: int) -> int:
        """
        Calculate appropriate trade limit based on timeframe and candle count.
        Aims for 100-500 ticks per candle for meaningful analysis.
        """
        timeframe_multipliers = {
            '1m': 200,   # ~200 trades per minute
            '3m': 500,   # ~500 trades per 3 minutes  
            '5m': 800,   # ~800 trades per 5 minutes
            '15m': 2000, # ~2000 trades per 15 minutes
            '30m': 3500, # ~3500 trades per 30 minutes
            '1h': 6000,  # ~6000 trades per hour
            '4h': 20000, # ~20000 trades per 4 hours
        }
        
        base_trades = timeframe_multipliers.get(timeframe, 1000)
        return min(base_trades * candle_limit, 10000)  # Cap at 10k trades for performance
    
    def classify_trade_direction(self, trade: Dict, orderbook_data: Optional[Dict] = None) -> str:
        """
        Classify trade as 'buy' (lifting ask) or 'sell' (hitting bid).
        
        Args:
            trade: Trade data with price, amount, timestamp
            orderbook_data: Optional L2 orderbook for bid/ask reference
            
        Returns:
            'buy' if lifting ask, 'sell' if hitting bid, 'neutral' if unclear
        """
        # Method 1: Use CCXT 'side' field if available
        if 'side' in trade and trade['side']:
            return trade['side']
        
        # Method 2: Use price movement classification from enhanced trades
        if 'price_movement' in trade:
            movement = trade['price_movement']
            if movement == 'uptick':
                return 'buy'  # Price moved up, likely aggressive buyer
            elif movement == 'downtick':
                return 'sell'  # Price moved down, likely aggressive seller
        
        # Method 3: Compare with orderbook if available
        if orderbook_data and 'bids' in orderbook_data and 'asks' in orderbook_data:
            trade_price = float(trade.get('price', 0))
            
            if orderbook_data['bids'] and orderbook_data['asks']:
                best_bid = float(orderbook_data['bids'][0][0])
                best_ask = float(orderbook_data['asks'][0][0])
                
                # If trade price closer to ask, likely a buy (lift ask)
                # If trade price closer to bid, likely a sell (hit bid)
                if abs(trade_price - best_ask) < abs(trade_price - best_bid):
                    return 'buy'
                else:
                    return 'sell'
        
        # Default: use tick rule - uptick = buy, downtick = sell
        return 'neutral'
    
    def create_price_bins(self, candle_high: float, candle_low: float, num_bins: int = 40) -> List[Tuple[float, float]]:
        """
        Create price bins for footprint chart within candle's high-low range.
        
        Args:
            candle_high: Candle high price
            candle_low: Candle low price  
            num_bins: Number of price subdivisions
            
        Returns:
            List of (bin_low, bin_high) tuples
        """
        if candle_high <= candle_low or num_bins <= 0:
            return []
        
        price_range = candle_high - candle_low
        bin_size = price_range / num_bins
        
        bins = []
        for i in range(num_bins):
            bin_low = candle_low + (i * bin_size)
            bin_high = candle_low + ((i + 1) * bin_size)
            bins.append((bin_low, bin_high))
        
        return bins
    
    def build_footprint_matrix(self, ohlcv_data: List[List], trades_data: List[Dict], 
                             num_bins: int = 40) -> List[Dict]:
        """
        Build 2D footprint matrix: rows=candles, columns=price bins, cells=[buy_vol, sell_vol].
        
        Args:
            ohlcv_data: OHLCV candle data
            trades_data: Enhanced trade data
            num_bins: Number of price bins per candle
            
        Returns:
            List of footprint data for each candle
        """
        footprint_candles = []
        
        for i, candle in enumerate(ohlcv_data):
            timestamp, open_price, high, low, close, volume = candle
            
            # Create price bins for this candle
            price_bins = self.create_price_bins(high, low, num_bins)
            
            # Initialize bin volumes
            bin_volumes = {}
            for j, (bin_low, bin_high) in enumerate(price_bins):
                bin_volumes[j] = {'buy_volume': 0.0, 'sell_volume': 0.0, 'price_range': (bin_low, bin_high)}
            
            # Filter trades that fall within this candle's timeframe
            candle_start = timestamp
            # Estimate candle end time (this is approximate)
            if i < len(ohlcv_data) - 1:
                candle_end = ohlcv_data[i + 1][0]
            else:
                # For last candle, estimate duration
                candle_end = timestamp + (60 * 1000 * 5)  # Assume 5min default
            
            candle_trades = [
                trade for trade in trades_data 
                if candle_start <= trade.get('timestamp', 0) < candle_end
            ]
            
            # Distribute trades into price bins
            for trade in candle_trades:
                trade_price = float(trade.get('price', 0))
                trade_amount = float(trade.get('amount', 0))
                
                # Find which bin this trade belongs to
                for bin_id, (bin_low, bin_high) in enumerate(price_bins):
                    if bin_low <= trade_price < bin_high:
                        trade_direction = self.classify_trade_direction(trade)
                        
                        if trade_direction == 'buy':
                            bin_volumes[bin_id]['buy_volume'] += trade_amount
                        elif trade_direction == 'sell':
                            bin_volumes[bin_id]['sell_volume'] += trade_amount
                        
                        break
            
            # Store footprint data for this candle
            footprint_data = {
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp / 1000),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'bins': bin_volumes,
                'num_trades': len(candle_trades),
                'price_range': high - low
            }
            
            footprint_candles.append(footprint_data)
        
        return footprint_candles
    
    def detect_footprint_signals(self, footprint_data: List[Dict]) -> Dict[str, Any]:
        """
        Detect volume footprint signals for trading insights.
        
        Args:
            footprint_data: Footprint matrix data
            
        Returns:
            Dictionary with detected signals and analysis
        """
        signals = {
            'volume_exhaustion': [],
            'delta_imbalances': [],
            'absorption_zones': [],
            'overall_signal': 'NEUTRAL',
            'confidence': 'LOW'
        }
        
        if not footprint_data:
            return signals
        
        latest_candle = footprint_data[-1]
        bins = latest_candle['bins']
        
        # 1. Volume Exhaustion Detection
        # Look for very thin volume bars at price extremes
        total_bins = len(bins)
        top_bins = range(int(total_bins * 0.8), total_bins)  # Top 20%
        bottom_bins = range(0, int(total_bins * 0.2))  # Bottom 20%
        
        # Calculate average volume in middle vs extremes
        middle_bins = range(int(total_bins * 0.3), int(total_bins * 0.7))
        
        top_volume = sum(bins[i]['buy_volume'] + bins[i]['sell_volume'] for i in top_bins if i in bins)
        bottom_volume = sum(bins[i]['buy_volume'] + bins[i]['sell_volume'] for i in bottom_bins if i in bins)
        middle_volume = sum(bins[i]['buy_volume'] + bins[i]['sell_volume'] for i in middle_bins if i in bins)
        
        avg_middle_volume = middle_volume / len(middle_bins) if middle_bins else 0
        avg_top_volume = top_volume / len(top_bins) if top_bins else 0
        avg_bottom_volume = bottom_volume / len(bottom_bins) if bottom_bins else 0
        
        # Volume exhaustion signals
        if avg_middle_volume > 0:
            if avg_top_volume < (avg_middle_volume * 0.3):  # Top 30% less volume than middle
                signals['volume_exhaustion'].append({
                    'type': 'bearish_exhaustion',
                    'location': 'top',
                    'strength': 'high' if avg_top_volume < (avg_middle_volume * 0.1) else 'medium'
                })
            
            if avg_bottom_volume < (avg_middle_volume * 0.3):  # Bottom 30% less volume than middle
                signals['volume_exhaustion'].append({
                    'type': 'bullish_exhaustion', 
                    'location': 'bottom',
                    'strength': 'high' if avg_bottom_volume < (avg_middle_volume * 0.1) else 'medium'
                })
        
        # 2. Delta Imbalance Detection  
        # Look for bins with significant buy/sell volume skew
        for bin_id, bin_data in bins.items():
            buy_vol = bin_data['buy_volume']
            sell_vol = bin_data['sell_volume']
            total_vol = buy_vol + sell_vol
            
            if total_vol > 0:
                buy_ratio = buy_vol / total_vol
                sell_ratio = sell_vol / total_vol
                
                # Significant imbalance threshold
                if buy_ratio > 0.75:  # 75%+ buy volume
                    signals['delta_imbalances'].append({
                        'bin_id': bin_id, 
                        'type': 'buy_imbalance',
                        'ratio': buy_ratio,
                        'volume': total_vol,
                        'price_range': bin_data['price_range']
                    })
                elif sell_ratio > 0.75:  # 75%+ sell volume
                    signals['delta_imbalances'].append({
                        'bin_id': bin_id,
                        'type': 'sell_imbalance', 
                        'ratio': sell_ratio,
                        'volume': total_vol,
                        'price_range': bin_data['price_range']
                    })
        
        # 3. Overall Signal Classification
        exhaustion_signals = len(signals['volume_exhaustion'])
        imbalance_signals = len(signals['delta_imbalances'])
        
        if exhaustion_signals > 0:
            # Check if we have bearish or bullish exhaustion
            bearish_exhaustion = any(s['type'] == 'bearish_exhaustion' for s in signals['volume_exhaustion'])
            bullish_exhaustion = any(s['type'] == 'bullish_exhaustion' for s in signals['volume_exhaustion'])
            
            if bearish_exhaustion and not bullish_exhaustion:
                signals['overall_signal'] = 'BEARISH_REVERSAL'
                signals['confidence'] = 'HIGH' if exhaustion_signals >= 2 else 'MEDIUM'
            elif bullish_exhaustion and not bearish_exhaustion:
                signals['overall_signal'] = 'BULLISH_REVERSAL'
                signals['confidence'] = 'HIGH' if exhaustion_signals >= 2 else 'MEDIUM'
        
        return signals
    
    def render_footprint_chart(self, footprint_data: List[Dict], symbol: str, 
                             timeframe: str, num_bins: int) -> str:
        """
        Render ASCII footprint chart for terminal display.
        
        Args:
            footprint_data: Footprint matrix data
            symbol: Trading symbol
            timeframe: Timeframe string
            num_bins: Number of price bins
            
        Returns:
            Formatted string output for terminal
        """
        if not footprint_data:
            return "No footprint data available"
        
        output = []
        latest_candle = footprint_data[-1]
        
        # Header
        output.append(f"[{latest_candle['datetime'].strftime('%Y-%m-%d %H:%M')}] {symbol} | TF: {timeframe} | Bins: {num_bins}")
        
        candle_start = latest_candle['datetime']
        # Estimate candle end (approximate)
        timeframe_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60}
        duration = timeframe_minutes.get(timeframe, 5)
        candle_end = candle_start + timedelta(minutes=duration)
        
        output.append(f"Candle {candle_start.strftime('%H:%M')}–{candle_end.strftime('%H:%M')}")
        output.append(f"Price Range: {latest_candle['low']:.4f}–{latest_candle['high']:.4f}")
        output.append("")
        
        # Table header
        output.append("Bin Price      | Sell Vol │ Buy Vol")
        output.append("─" * 40)
        
        # Sort bins by price (highest to lowest for display)
        bins = latest_candle['bins']
        sorted_bins = sorted(bins.items(), key=lambda x: x[1]['price_range'][1], reverse=True)
        
        # Find max volume for scaling bars
        max_volume = 0
        for _, bin_data in sorted_bins:
            max_volume = max(max_volume, bin_data['buy_volume'], bin_data['sell_volume'])
        
        if max_volume == 0:
            output.append("No trade data available for this candle")
            return "\n".join(output)
        
        # Render each bin
        for bin_id, bin_data in sorted_bins:
            price_low, price_high = bin_data['price_range']
            buy_vol = bin_data['buy_volume']
            sell_vol = bin_data['sell_volume']
            
            # Scale volumes to bar width (max 10 chars)
            max_bar_width = 10
            buy_bar_width = int((buy_vol / max_volume) * max_bar_width) if max_volume > 0 else 0
            sell_bar_width = int((sell_vol / max_volume) * max_bar_width) if max_volume > 0 else 0
            
            # Create volume bars
            sell_bar = "█" * sell_bar_width
            buy_bar = "█" * buy_bar_width
            
            # Format price range
            price_range = f"{price_high:.4f}–{price_low:.4f}"
            
            output.append(f"{price_range:14} | {sell_bar:10} │ {buy_bar:10}")
        
        return "\n".join(output)
    
    def analyze_footprint(self, symbol: str, timeframe: str = '5m', 
                        limit: int = 50, num_bins: int = 40) -> str:
        """
        Complete footprint analysis with signal detection and chart rendering.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            limit: Number of candles to analyze
            num_bins: Number of price bins per candle
            
        Returns:
            Complete analysis output string
        """
        try:
            # Fetch data
            ohlcv_data, trades_data = self.fetch_footprint_data(symbol, timeframe, limit)
            
            if not ohlcv_data or not trades_data:
                return f"Insufficient data for footprint analysis of {symbol}"
            
            # Build footprint matrix
            footprint_data = self.build_footprint_matrix(ohlcv_data, trades_data, num_bins)
            
            if not footprint_data:
                return f"Failed to build footprint matrix for {symbol}"
            
            # Detect signals
            signals = self.detect_footprint_signals(footprint_data)
            
            # Render chart
            chart_output = self.render_footprint_chart(footprint_data, symbol, timeframe, num_bins)
            
            # Add signal analysis
            output = [chart_output, ""]
            
            # Volume exhaustion signals
            if signals['volume_exhaustion']:
                output.append("Volume Exhaustion Signals:")
                for signal in signals['volume_exhaustion']:
                    signal_type = signal['type'].replace('_', ' ').title()
                    location = signal['location'].title()
                    strength = signal['strength'].upper()
                    output.append(f"  • {signal_type} at {location} ({strength})")
                output.append("")
            
            # Delta imbalance signals  
            if signals['delta_imbalances']:
                output.append("Delta Imbalance Signals:")
                for signal in signals['delta_imbalances'][:3]:  # Show top 3
                    imbalance_type = signal['type'].replace('_', ' ').title()
                    ratio = signal['ratio']
                    price_low, price_high = signal['price_range']
                    output.append(f"  • {imbalance_type}: {ratio:.1%} at {price_high:.4f}–{price_low:.4f}")
                output.append("")
            
            # Overall signal
            overall_signal = signals['overall_signal'].replace('_', ' ').title()
            confidence = signals['confidence'].title()
            
            if signals['overall_signal'] != 'NEUTRAL':
                output.append(f"Signal: {overall_signal}")
                output.append(f"Confidence: {confidence}")
            else:
                output.append("Signal: No clear directional bias")
                output.append("Confidence: LOW")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Error in footprint analysis for {symbol}: {e}")
            return f"Error analyzing footprint for {symbol}: {str(e)}"


def analyze_footprint(symbol: str, timeframe: str = '5m', limit: int = 50, bins: int = 40) -> str:
    """
    Convenience function for footprint analysis.
    
    Args:
        symbol: Trading pair symbol (e.g., 'XRP/USDT')
        timeframe: Candle timeframe (e.g., '1m', '5m', '15m')
        limit: Number of candles to analyze
        bins: Number of price bins per candle
        
    Returns:
        Complete footprint analysis output
    """
    analyzer = FootprintAnalyzer()
    return analyzer.analyze_footprint(symbol, timeframe, limit, bins)