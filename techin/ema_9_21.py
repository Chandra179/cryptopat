"""
EMA 9/21 crossover strategy implementation for cryptocurrency trend analysis.
Uses exponential moving averages to detect bullish and bearish trend signals.
"""

from datetime import datetime
from typing import List, Tuple, Dict, Optional
import pandas as pd
import statistics
from data import get_data_collector


class EMA9_21Strategy:
    """Enhanced EMA 9/21 crossover strategy with statistical validation and confidence scoring."""
    
    def __init__(self):
        self.collector = get_data_collector()
        self.signal_history = []  # Track historical signals for validation
    
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate Exponential Moving Average using pandas for accuracy.
        
        Args:
            prices: List of closing prices
            period: EMA period (9 or 21)
            
        Returns:
            List of EMA values, same length as prices with None for insufficient data
        """
        if len(prices) < period:
            return [None] * len(prices)
        
        # Use pandas for accurate EMA calculation
        df = pd.DataFrame({'price': prices})
        ema_series = df['price'].ewm(span=period, adjust=False).mean()
        
        # Convert to list, keeping NaN as None for early values
        ema_values = []
        for i, value in enumerate(ema_series.tolist()):
            # First (period-1) values should be None as EMA needs time to stabilize
            if i < period - 1 or pd.isna(value):
                ema_values.append(None)
            else:
                ema_values.append(value)
        
        return ema_values
    
    def calculate_atr(self, ohlcv_data: List[List], period: int = 14) -> List[float]:
        """
        Calculate Average True Range for volatility measurement.
        
        Args:
            ohlcv_data: OHLCV candlestick data
            period: ATR calculation period
            
        Returns:
            List of ATR values
        """
        if len(ohlcv_data) < period + 1:
            return [None] * len(ohlcv_data)
        
        true_ranges = []
        for i in range(1, len(ohlcv_data)):
            high = ohlcv_data[i][2]
            low = ohlcv_data[i][3]
            prev_close = ohlcv_data[i-1][4]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        # Calculate ATR using simple moving average
        atr_values = [None]  # First candle has no ATR
        for i in range(period - 1, len(true_ranges)):
            atr = sum(true_ranges[i-period+1:i+1]) / period
            atr_values.append(atr)
        
        # Pad to match ohlcv_data length
        while len(atr_values) < len(ohlcv_data):
            atr_values.append(None)
            
        return atr_values
    
    def detect_market_regime(self, closes: List[float], ema9: List[float], 
                           ema21: List[float], atr_values: List[float]) -> str:
        """
        Detect market regime (trending, ranging, volatile).
        
        Args:
            closes: Closing prices
            ema9: EMA 9 values
            ema21: EMA 21 values
            atr_values: ATR values
            
        Returns:
            Market regime classification
        """
        if len(closes) < 30:
            return "insufficient_data"
        
        recent_closes = closes[-20:]  # Last 20 periods
        recent_atr = [atr for atr in atr_values[-10:] if atr is not None]
        
        if not recent_atr:
            return "insufficient_data"
        
        # Trend strength based on EMA separation
        recent_ema9 = [ema for ema in ema9[-20:] if ema is not None]
        recent_ema21 = [ema for ema in ema21[-20:] if ema is not None]
        
        if len(recent_ema9) < 10 or len(recent_ema21) < 10:
            return "insufficient_data"
        
        # Calculate EMA separation as percentage
        ema_separations = []
        for i in range(min(len(recent_ema9), len(recent_ema21))):
            separation = abs(recent_ema9[i] - recent_ema21[i]) / recent_ema21[i]
            ema_separations.append(separation)
        
        avg_separation = statistics.mean(ema_separations)
        avg_atr = statistics.mean(recent_atr)
        current_price = recent_closes[-1]
        atr_percentage = avg_atr / current_price
        
        # Regime classification
        if avg_separation > 0.01 and atr_percentage < 0.02:  # 1% EMA sep, 2% ATR
            return "trending_low_vol"
        elif avg_separation > 0.01 and atr_percentage >= 0.02:
            return "trending_high_vol"
        elif avg_separation <= 0.005 and atr_percentage < 0.015:  # 0.5% EMA sep, 1.5% ATR
            return "ranging_low_vol"
        elif avg_separation <= 0.005 and atr_percentage >= 0.015:
            return "ranging_high_vol"
        else:
            return "transitional"
    
    def calculate_signal_confidence(self, signal: Dict, market_regime: str, 
                                  momentum_score: float, volume_context: float) -> float:
        """
        Calculate confidence score for EMA crossover signal.
        
        Args:
            signal: Signal dictionary
            market_regime: Current market regime
            momentum_score: Price momentum score
            volume_context: Volume context score
            
        Returns:
            Confidence score from 0.0 to 1.0
        """
        confidence = 0.0
        
        # Base confidence from signal type
        if signal['signal'] in ['BUY', 'SELL']:
            confidence += 0.3
        
        # Volume confirmation bonus
        if signal.get('volume_spike', False):
            confidence += 0.25
        
        # Volume context bonus (relative to average)
        if volume_context > 1.5:  # 50% above average
            confidence += 0.15
        elif volume_context > 1.2:  # 20% above average
            confidence += 0.1
        
        # Market regime bonus
        regime_bonus = {
            "trending_low_vol": 0.2,
            "trending_high_vol": 0.15,
            "transitional": 0.1,
            "ranging_low_vol": 0.05,
            "ranging_high_vol": 0.0
        }
        confidence += regime_bonus.get(market_regime, 0.0)
        
        # Momentum bonus
        if momentum_score > 0.7:
            confidence += 0.1
        elif momentum_score > 0.5:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def calculate_momentum_score(self, closes: List[float], ema9: List[float], 
                               lookback: int = 5) -> float:
        """
        Calculate price momentum score based on recent price action.
        
        Args:
            closes: Closing prices
            ema9: EMA 9 values
            lookback: Periods to look back
            
        Returns:
            Momentum score from 0.0 to 1.0
        """
        if len(closes) < lookback + 1:
            return 0.5
        
        recent_closes = closes[-lookback-1:]
        recent_ema9 = [ema for ema in ema9[-lookback-1:] if ema is not None]
        
        if len(recent_ema9) < lookback:
            return 0.5
        
        # Price vs EMA momentum
        price_ema_scores = []
        for i in range(len(recent_closes) - 1):
            if recent_ema9[i] is not None:
                score = 1.0 if recent_closes[i] > recent_ema9[i] else 0.0
                price_ema_scores.append(score)
        
        # Price direction momentum
        price_changes = []
        for i in range(1, len(recent_closes)):
            change = 1.0 if recent_closes[i] > recent_closes[i-1] else 0.0
            price_changes.append(change)
        
        if not price_ema_scores or not price_changes:
            return 0.5
        
        momentum_score = (statistics.mean(price_ema_scores) + statistics.mean(price_changes)) / 2
        return momentum_score
    
    def calculate_statistical_significance(self, signal: Dict, 
                                         historical_signals: List[Dict]) -> float:
        """
        Calculate statistical significance using z-score of signal strength.
        
        Args:
            signal: Current signal
            historical_signals: Previous signals for comparison
            
        Returns:
            Z-score indicating statistical significance
        """
        if len(historical_signals) < 10:
            return 0.0  # Need historical data for significance
        
        # Calculate signal strength metric (EMA separation)
        current_strength = abs(signal['ema9'] - signal['ema21']) / signal['ema21']
        
        # Historical signal strengths
        historical_strengths = []
        for hist_signal in historical_signals:
            if hist_signal.get('ema9') and hist_signal.get('ema21'):
                strength = abs(hist_signal['ema9'] - hist_signal['ema21']) / hist_signal['ema21']
                historical_strengths.append(strength)
        
        if len(historical_strengths) < 5:
            return 0.0
        
        # Calculate z-score
        mean_strength = statistics.mean(historical_strengths)
        std_strength = statistics.stdev(historical_strengths) if len(historical_strengths) > 1 else 0.001
        
        z_score = (current_strength - mean_strength) / std_strength
        return z_score
    
    def detect_volume_spike(self, volumes: List[float], window: int = 10) -> Tuple[List[bool], List[float]]:
        """
        Enhanced volume spike detection with context scoring.
        
        Args:
            volumes: List of volume values
            window: Lookback window for average calculation
            
        Returns:
            Tuple of (spike_flags, volume_context_scores)
        """
        spikes = []
        context_scores = []
        
        for i in range(len(volumes)):
            if i < window:
                spikes.append(False)
                context_scores.append(1.0)
                continue
                
            # Average volume over lookback window
            avg_volume = sum(volumes[i-window:i]) / window
            current_volume = volumes[i]
            
            # Volume context score (ratio to average)
            context_score = current_volume / avg_volume if avg_volume > 0 else 1.0
            context_scores.append(context_score)
            
            # Volume spike if current > 2x average (less sensitive)
            spike = current_volume > (avg_volume * 2.0)
            spikes.append(spike)
        
        return spikes, context_scores
    
    def detect_crossovers(self, ema9: List[float], ema21: List[float], 
                         closes: List[float], volumes: List[float], 
                         ohlcv_data: List[List]) -> List[dict]:
        """
        Detect EMA crossovers and generate signals.
        
        Args:
            ema9: EMA 9 values (aligned with data indices)
            ema21: EMA 21 values (aligned with data indices)
            closes: Closing prices
            volumes: Volume values
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        volume_spikes, volume_contexts = self.detect_volume_spike(volumes)
        atr_values = self.calculate_atr(ohlcv_data)
        market_regime = self.detect_market_regime(closes, ema9, ema21, atr_values)
        
        # Start from index 21 where both EMAs are valid and we have previous values
        for i in range(21, len(closes)):
            # Skip if either current or previous EMA values are None
            if (ema9[i] is None or ema21[i] is None or 
                ema9[i-1] is None or ema21[i-1] is None):
                continue
                
            # Calculate momentum score
            momentum_score = self.calculate_momentum_score(closes[:i+1], ema9[:i+1])
            volume_context = volume_contexts[i] if i < len(volume_contexts) else 1.0
            
            signal = {
                'data_index': i,
                'close': closes[i],
                'ema9': ema9[i],
                'ema21': ema21[i],
                'volume_spike': volume_spikes[i] if i < len(volume_spikes) else False,
                'volume_context': volume_context,
                'momentum_score': momentum_score,
                'market_regime': market_regime,
                'atr': atr_values[i] if i < len(atr_values) and atr_values[i] is not None else 0.0,
                'signal': 'NONE',
                'trend': 'NEUTRAL',
                'confirmed': False,
                'confidence': 0.0,
                'z_score': 0.0
            }
            
            prev_ema9 = ema9[i-1]
            prev_ema21 = ema21[i-1]
            
            # Bullish crossover: EMA9 crosses above EMA21
            if prev_ema9 <= prev_ema21 and ema9[i] > ema21[i]:
                signal['signal'] = 'BUY'
                signal['trend'] = 'BULLISH'
                signal['confirmed'] = volume_spikes[i] if i < len(volume_spikes) else False
                signal['confidence'] = self.calculate_signal_confidence(signal, market_regime, 
                                                                       momentum_score, volume_context)
                signal['z_score'] = self.calculate_statistical_significance(signal, self.signal_history)
            
            # Bearish crossover: EMA9 crosses below EMA21  
            elif prev_ema9 >= prev_ema21 and ema9[i] < ema21[i]:
                signal['signal'] = 'SELL'
                signal['trend'] = 'BEARISH'
                signal['confirmed'] = volume_spikes[i] if i < len(volume_spikes) else False
                signal['confidence'] = self.calculate_signal_confidence(signal, market_regime, 
                                                                       momentum_score, volume_context)
                signal['z_score'] = self.calculate_statistical_significance(signal, self.signal_history)
            
            # Add trend information even for non-crossover points
            elif ema9[i] > ema21[i]:
                signal['trend'] = 'BULLISH'
            elif ema9[i] < ema21[i]:
                signal['trend'] = 'BEARISH'
            
            signals.append(signal)
            
            # Add to signal history for statistical analysis
            if signal['signal'] in ['BUY', 'SELL']:
                self.signal_history.append(signal)
                # Keep only last 100 signals
                if len(self.signal_history) > 100:
                    self.signal_history = self.signal_history[-100:]
        
        return signals
    
    def analyze(self, symbol: str, timeframe: str, limit: int, ohlcv_data: Optional[List] = None) -> dict:
        """
        Perform EMA 9/21 analysis and return structured results.
        
        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            timeframe: Timeframe (e.g., '1d', '4h', '1h')
            limit: Number of candles to analyze
            ohlcv_data: Optional pre-fetched OHLCV data
            
        Returns:
            dict: Structured analysis results
        """
        # Fetch OHLCV data if not provided
        if ohlcv_data is None:
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 30:
            return {
                'success': False,
                'error': f'Need at least 30 candles for EMA 9/21 calculation. Got {len(ohlcv_data)}',
                'symbol': symbol,
                'timeframe': timeframe
            }
        
        # Extract closes and volumes
        timestamps = [candle[0] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        volumes = [candle[5] for candle in ohlcv_data]
        highs = [candle[2] for candle in ohlcv_data]
        lows = [candle[3] for candle in ohlcv_data]
        
        # Calculate EMAs
        ema9 = self.calculate_ema(closes, 9)
        ema21 = self.calculate_ema(closes, 21)
        
        if not ema9 or not ema21:
            return {
                'success': False,
                'error': 'Unable to calculate EMAs',
                'symbol': symbol,
                'timeframe': timeframe
            }
        
        # Detect volume information and signals
        volume_spikes, volume_contexts = self.detect_volume_spike(volumes)
        signals = self.detect_crossovers(ema9, ema21, closes, volumes, ohlcv_data)
        
        # Get latest signal data
        latest_signal = signals[-1] if signals else None
        if not latest_signal:
            return {
                'success': False,
                'error': 'No signals generated',
                'symbol': symbol,
                'timeframe': timeframe
            }
            
        timestamp_idx = latest_signal['data_index']
        dt = datetime.fromtimestamp(timestamps[timestamp_idx] / 1000)
        
        # Calculate additional metrics for Phase 2 format
        current_price = latest_signal['close']
        ema9_val = latest_signal['ema9']
        ema21_val = latest_signal['ema21']
        
        # EMA separation ratio
        ema_ratio = abs(ema9_val - ema21_val) / ema21_val if ema21_val != 0 else 0
        
        # Volume spike confirmation
        volume_confirmed = "✓" if latest_signal['volume_spike'] else "✗"
        volume_context = latest_signal['volume_context']
        
        # Price change percentage
        if len(closes) >= 2:
            price_change = ((current_price - closes[-2]) / closes[-2]) * 100
        else:
            price_change = 0.0
            
        # Support and resistance levels (based on recent highs/lows)
        recent_highs = highs[-min(20, len(highs)):]
        recent_lows = lows[-min(20, len(lows)):]
        resistance = max(recent_highs) if recent_highs else current_price * 1.02
        support = min(recent_lows) if recent_lows else current_price * 0.98
        
        # Stop loss and take profit zones
        atr_val = latest_signal['atr']
        stop_zone = current_price - (atr_val * 2) if atr_val > 0 else support * 0.99
        tp_zone_low = current_price + (atr_val * 2.5) if atr_val > 0 else resistance * 1.01
        tp_zone_high = current_price + (atr_val * 3.5) if atr_val > 0 else resistance * 1.05
        
        # Risk/Reward ratio
        risk = abs(current_price - stop_zone)
        reward = abs(tp_zone_low - current_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Expected max drawdown
        max_drawdown = (atr_val / current_price) * 100 if atr_val > 0 else 2.0
        
        # Trend momentum state
        momentum_score = latest_signal['momentum_score']
        if momentum_score > 0.7:
            momentum_state = "Accelerating"
        elif momentum_score > 0.5:
            momentum_state = "Building"
        elif momentum_score > 0.3:
            momentum_state = "Weakening"
        else:
            momentum_state = "Decelerating"
            
        # Entry window timing
        if latest_signal['signal'] in ['BUY', 'SELL']:
            entry_window = "Optimal in next 1-2 bars"
        else:
            entry_window = "Waiting for crossover signal"
            
        # Exit trigger
        if latest_signal['trend'] == 'BULLISH':
            exit_trigger = f"Cross below EMA 21 (${ema21_val:.4f})"
        elif latest_signal['trend'] == 'BEARISH':
            exit_trigger = f"Cross above EMA 21 (${ema21_val:.4f})"
        else:
            exit_trigger = "Wait for trend confirmation"
            
        # Summary description
        trend_desc = latest_signal['trend'].lower()
        regime_desc = latest_signal['market_regime'].replace('_', ' ')
        if latest_signal['volume_spike']:
            summary = f"EMA 9/21 {trend_desc} crossover with volume confirmation in {regime_desc} market"
        else:
            summary = f"EMA 9/21 showing {trend_desc} bias in {regime_desc} market conditions"
        
        # Return structured analysis results
        return {
            'success': True,
            'analysis_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': timestamps[timestamp_idx],
            
            # Core indicators
            'ema9_value': round(ema9_val, 4),
            'ema21_value': round(ema21_val, 4),
            'ema_ratio': round(ema_ratio, 3),
            'volume_confirmed': latest_signal['volume_spike'],
            'volume_context': round(volume_context, 2),
            'price_change': round(price_change, 1),
            'z_score': round(latest_signal['z_score'], 2),
            'atr_percentage': round((atr_val/current_price)*100, 2) if atr_val > 0 else 0.0,
            'market_regime': latest_signal['market_regime'],
            
            # Price levels
            'current_price': round(current_price, 4),
            'support_level': round(support, 4),
            'resistance_level': round(resistance, 4),
            'stop_zone': round(stop_zone, 4),
            'tp_low': round(tp_zone_low, 4),
            'tp_high': round(tp_zone_high, 4),
            
            # Trading analysis
            'signal': latest_signal['signal'],
            'description': summary,
            'confidence_score': round(latest_signal['confidence'] * 100, 0),
            'trend_direction': latest_signal['trend'],
            'momentum_state': momentum_state,
            'momentum_score': round(momentum_score, 2),
            'entry_window': entry_window,
            'exit_trigger': exit_trigger,
            'rr_ratio': round(rr_ratio, 1),
            'max_drawdown': round(max_drawdown, 1),
            
            # Additional data
            'data_points': len(ohlcv_data),
            'all_signals': signals,
            'raw_data': {
                'ema9_values': ema9,
                'ema21_values': ema21,
                'ohlcv_data': ohlcv_data,
                'volume_spikes': volume_spikes,
                'volume_contexts': volume_contexts
            }
        }
