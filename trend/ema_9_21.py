"""
EMA 9/21 crossover strategy implementation for cryptocurrency trend analysis.
Uses exponential moving averages to detect bullish and bearish trend signals.
"""

import sys
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
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
    
    def analyze(self, symbol: str, timeframe: str, limit: int) -> None:
        """
        Perform EMA 9/21 analysis and display results.
        
        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            timeframe: Timeframe (e.g., '1d', '4h', '1h')
            limit: Number of candles to analyze
        """
        # Fetch OHLCV data
        ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 30:
            print(f"Error: Need at least 30 candles for EMA 9/21 calculation. Got {len(ohlcv_data)}")
            return
        
        # Extract closes and volumes
        timestamps = [candle[0] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        volumes = [candle[5] for candle in ohlcv_data]
        
        # Calculate EMAs
        ema9 = self.calculate_ema(closes, 9)
        ema21 = self.calculate_ema(closes, 21)
        
        if not ema9 or not ema21:
            print("Error: Unable to calculate EMAs")
            return
        
        # Detect signals
        signals = self.detect_crossovers(ema9, ema21, closes, volumes, ohlcv_data)
        
        today = datetime.now().date()
        today_signals = []
        
        for signal in signals:
            timestamp_idx = signal['data_index']
            if timestamp_idx < len(timestamps):
                dt = datetime.fromtimestamp(timestamps[timestamp_idx] / 1000)
                if dt.date() == today:
                    today_signals.append((signal, dt))
        
        if not today_signals:
            print(f"No signals found for today ({today})")
            # Show latest signal for reference
            if signals:
                latest_signal = signals[-1]
                timestamp_idx = latest_signal['data_index']
                if timestamp_idx < len(timestamps):
                    dt = datetime.fromtimestamp(timestamps[timestamp_idx] / 1000)
                    
                    print(f"\nLatest signal:")
                    trend_emoji = "📈" if latest_signal['trend'] == 'BULLISH' else "📉" if latest_signal['trend'] == 'BEARISH' else "➖"
                    confidence_str = f"CONF: {latest_signal['confidence']:.2f}" if latest_signal['confidence'] > 0 else "CONF: N/A"
                    regime_str = latest_signal['market_regime'].replace('_', ' ').title()
                    z_score_str = f"Z: {latest_signal['z_score']:.2f}" if latest_signal['z_score'] != 0 else "Z: N/A"
                    
                    print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                          f"CLOSE: {latest_signal['close']:.4f} | "
                          f"EMA9: {latest_signal['ema9']:.4f} | "
                          f"EMA21: {latest_signal['ema21']:.4f} | "
                          f"Signal: {latest_signal['signal']} | "
                          f"{trend_emoji} {latest_signal['trend']} | "
                          f"{confidence_str} | {z_score_str} | "
                          f"Regime: {regime_str}")
        else:
            for signal, dt in today_signals:
                trend_emoji = "📈" if signal['trend'] == 'BULLISH' else "📉" if signal['trend'] == 'BEARISH' else "➖"
                confidence_str = f"CONF: {signal['confidence']:.2f}" if signal['confidence'] > 0 else "CONF: N/A"
                regime_str = signal['market_regime'].replace('_', ' ').title()
                z_score_str = f"Z: {signal['z_score']:.2f}" if signal['z_score'] != 0 else "Z: N/A"
                
                print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"CLOSE: {signal['close']:.4f} | "
                      f"EMA9: {signal['ema9']:.4f} | "
                      f"EMA21: {signal['ema21']:.4f} | "
                      f"Signal: {signal['signal']} | "
                      f"{trend_emoji} {signal['trend']} | "
                      f"{confidence_str} | {z_score_str} | "
                      f"Regime: {regime_str}")


def parse_command(command: str) -> Tuple[str, str, int]:
    """
    Parse terminal command: ema_9_21 s=XRP/USDT t=1d l=30
    
    Args:
        command: Command string
        
    Returns:
        Tuple of (symbol, timeframe, limit)
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'ema_9_21':
        raise ValueError("Invalid command format. Use: ema_9_21 s=SYMBOL t=TIMEFRAME l=LIMIT")
    
    symbol = None
    timeframe = '1d'  # default
    limit = 30  # default
    
    for part in parts[1:]:
        if part.startswith('s='):
            symbol = part[2:]
        elif part.startswith('t='):
            timeframe = part[2:]
        elif part.startswith('l='):
            try:
                limit = int(part[2:])
            except ValueError:
                raise ValueError(f"Invalid limit value: {part[2:]}")
    
    if symbol is None:
        raise ValueError("Symbol (s=) is required")
    
    return symbol, timeframe, limit


def main():
    """Main entry point for EMA 9/21 strategy."""
    if len(sys.argv) < 2:
        print("Usage: python ema_9_21.py s=SYMBOL t=TIMEFRAME l=LIMIT")
        print("Example: python ema_9_21.py s=XRP/USDT t=1d l=30")
        return
    
    try:
        # Parse command line arguments
        command = ' '.join(['ema_9_21'] + sys.argv[1:])
        symbol, timeframe, limit = parse_command(command)
        
        # Run analysis
        strategy = EMA9_21Strategy()
        strategy.analyze(symbol, timeframe, limit)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()