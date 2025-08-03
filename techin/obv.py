"""
OBV (On-Balance Volume) strategy implementation for cryptocurrency trend analysis.
Uses volume and price relationship to detect bullish and bearish trend signals.
"""

from datetime import datetime
from typing import List, Optional
from data import get_data_collector


class OBVStrategy:
    """OBV (On-Balance Volume) strategy for trend detection."""
    
    def __init__(self):
        self.collector = get_data_collector()
    
    def calculate_obv(self, closes: List[float], volumes: List[float]) -> List[float]:
        """
        Calculate On-Balance Volume.
        
        Args:
            closes: List of closing prices
            volumes: List of volume values
            
        Returns:
            List of OBV values
        """
        if len(closes) != len(volumes) or len(closes) < 2:
            return []
        
        obv_values = [volumes[0]]  # Start with first volume as initial OBV
        
        for i in range(1, len(closes)):
            current_close = closes[i]
            previous_close = closes[i-1]
            current_volume = volumes[i]
            previous_obv = obv_values[-1]
            
            if current_close > previous_close:
                # Price up, add volume
                obv = previous_obv + current_volume
            elif current_close < previous_close:
                # Price down, subtract volume
                obv = previous_obv - current_volume
            else:
                # Price unchanged, OBV unchanged
                obv = previous_obv
            
            obv_values.append(obv)
        
        return obv_values
    
    def detect_obv_signals(self, obv: List[float], closes: List[float]) -> List[dict]:
        """
        Detect OBV trend signals.
        
        Args:
            obv: OBV values
            closes: Closing prices
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        for i in range(10, len(obv)):  # Need some history for trend detection
            signal = {
                'index': i,
                'close': closes[i],
                'obv': obv[i],
                'signal': 'NONE',
                'trend_confirmation': 'NEUTRAL',
                'divergence': False
            }
            
            # Look at OBV trend over last 5 periods
            obv_trend_periods = 5
            if i >= obv_trend_periods:
                recent_obv = obv[i-obv_trend_periods:i+1]
                recent_prices = closes[i-obv_trend_periods:i+1]
                
                # Calculate OBV trend (simple linear trend)
                obv_trend = recent_obv[-1] - recent_obv[0]
                price_trend = recent_prices[-1] - recent_prices[0]
                
                # Strong OBV threshold (adjust based on typical volumes)
                obv_threshold = abs(sum(recent_obv)) * 0.05  # 5% of average OBV magnitude
                
                # Detect signals based on OBV trend
                if obv_trend > obv_threshold:
                    signal['signal'] = 'BUY'
                    signal['trend_confirmation'] = 'UPTREND' if price_trend > 0 else 'CONFIRMED_UPTREND'
                    
                    # Check for divergence (price down, OBV up)
                    if price_trend < 0:
                        signal['divergence'] = True
                        signal['trend_confirmation'] = 'BULLISH_DIVERGENCE'
                        
                elif obv_trend < -obv_threshold:
                    signal['signal'] = 'SELL'
                    signal['trend_confirmation'] = 'DOWNTREND' if price_trend < 0 else 'CONFIRMED_DOWNTREND'
                    
                    # Check for divergence (price up, OBV down)
                    if price_trend > 0:
                        signal['divergence'] = True
                        signal['trend_confirmation'] = 'BEARISH_DIVERGENCE'
            
            signals.append(signal)
        
        return signals
    
    def analyze(self, symbol: str, timeframe: str, limit: int, ohlcv_data: Optional[List] = None) -> dict:
        """
        Perform OBV analysis and return results as structured data.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1d', '4h', '1h')
            limit: Number of candles to analyze
            ohlcv_data: Optional pre-fetched OHLCV data
            
        Returns:
            Dictionary containing analysis results
        """
        # Fetch OHLCV data if not provided
        if ohlcv_data is None:
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 50:
            return {
                'error': f"Need at least 50 candles for OBV calculation. Got {len(ohlcv_data)}",
                'success': False
            }
        
        # Extract closes and volumes
        timestamps = [candle[0] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        volumes = [candle[5] for candle in ohlcv_data]
        
        # Calculate OBV
        obv = self.calculate_obv(closes, volumes)
        
        if not obv:
            return {
                'error': "Unable to calculate OBV",
                'success': False
            }
        
        # Detect signals
        signals = self.detect_obv_signals(obv, closes)
        
        # Get latest signal for analysis
        if not signals:
            return {
                'error': "No signals generated",
                'success': False
            }
            
        latest_signal = signals[-1]
        timestamp_idx = latest_signal['index']
        
        if timestamp_idx >= len(timestamps):
            return {
                'error': "Invalid signal index",
                'success': False
            }
            
        dt = datetime.fromtimestamp(timestamps[timestamp_idx] / 1000)
        
        # Calculate additional metrics
        current_obv = latest_signal['obv']
        current_price = latest_signal['close']
        obv_change = ((current_obv - obv[0]) / abs(obv[0]) * 100) if obv[0] != 0 else 0
        
        # Volume analysis
        recent_volumes = volumes[-10:] if len(volumes) >= 10 else volumes
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        volume_spike = (volumes[-1] / avg_volume) if avg_volume > 0 else 1
        
        # Trend strength calculation
        obv_trend_strength = abs(obv_change)
        confidence_score = min(95, max(20, 50 + (obv_trend_strength * 2) + (volume_spike - 1) * 10))
        
        # Determine trend direction and momentum
        if latest_signal['signal'] == 'BUY':
            trend_direction = "Bullish"
            momentum_state = "Accelerating" if latest_signal['divergence'] else "Building"
            action = "BUY"
        elif latest_signal['signal'] == 'SELL':
            trend_direction = "Bearish"
            momentum_state = "Declining" if latest_signal['divergence'] else "Weakening"
            action = "SELL"
        else:
            trend_direction = "Neutral"
            momentum_state = "Consolidating"
            action = "NEUTRAL"
        
        # Calculate support and resistance (simple approximation)
        recent_prices = closes[-20:] if len(closes) >= 20 else closes
        support = min(recent_prices)
        resistance = max(recent_prices)
        
        # Risk/Reward calculation
        if action == "BUY":
            stop_zone = support * 0.98
            tp_zone_low = resistance * 1.02
            tp_zone_high = resistance * 1.05
            rr_ratio = (tp_zone_low - current_price) / (current_price - stop_zone) if current_price > stop_zone else 0
        elif action == "SELL":
            stop_zone = resistance * 1.02
            tp_zone_low = support * 0.95
            tp_zone_high = support * 0.98
            rr_ratio = (current_price - tp_zone_low) / (stop_zone - current_price) if stop_zone > current_price else 0
        else:
            stop_zone = support * 0.98
            tp_zone_low = resistance * 1.02
            tp_zone_high = resistance * 1.05
            rr_ratio = 1.0
        
        # Return structured analysis results
        return {
            'success': True,
            'analysis_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': timestamps[timestamp_idx],
            
            # Core indicators
            'obv_value': current_obv,
            'obv_change': round(obv_change, 1),
            'volume_spike': round(volume_spike, 2),
            'divergence': latest_signal['divergence'],
            'trend_confirmation': latest_signal['trend_confirmation'],
            
            # Price levels
            'current_price': round(current_price, 4),
            'support': round(support, 4),
            'resistance': round(resistance, 4),
            'stop_zone': round(stop_zone, 4),
            'tp_low': round(tp_zone_low, 4),
            'tp_high': round(tp_zone_high, 4),
            
            # Trading analysis
            'signal': latest_signal['signal'],
            'confidence_score': confidence_score,
            'trend_direction': trend_direction,
            'momentum_state': momentum_state,
            'entry_window': 'Immediate' if action != 'NEUTRAL' else 'Wait for breakout',
            'exit_trigger': 'OBV reversal' if action != 'NEUTRAL' else 'Clear trend signal',
            'action': action,
            'rr_ratio': round(rr_ratio, 1),
            'max_drawdown': round((1-stop_zone/current_price)*100, 1),
            
            # Additional data
            'all_signals': signals,
            'raw_data': {
                'obv_values': obv
            }
        }


