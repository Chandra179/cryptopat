"""
MACD (Moving Average Convergence Divergence) implementation for cryptocurrency trend analysis.
Uses MACD line, signal line, and histogram to detect trend changes and momentum shifts.
"""

from datetime import datetime
from typing import List, Tuple
from data import get_data_collector


class MACDStrategy:
    """MACD strategy for trend detection and momentum analysis."""
    
    def __init__(self):
        self.collector = get_data_collector()
    
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: List of closing prices
            period: EMA period
            
        Returns:
            List of EMA values
        """
        if len(prices) < period:
            return []
        
        # EMA multiplier
        multiplier = 2 / (period + 1)
        
        # Initialize with SMA for first value
        sma = sum(prices[:period]) / period
        ema_values = [sma]
        
        # Calculate EMA for remaining values
        for i in range(period, len(prices)):
            ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
        
        return ema_values
    
    def calculate_macd(self, prices: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate MACD line, signal line, and histogram using standard formula.
        
        Args:
            prices: List of closing prices
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        if len(prices) < 34:  # Need 26 for EMA26 + 9 for signal = 35 minimum
            return [], [], []
        
        # Calculate EMA(12) and EMA(26)
        ema12 = self.calculate_ema(prices, 12)
        ema26 = self.calculate_ema(prices, 26)
        
        if not ema12 or not ema26:
            return [], [], []
        
        # MACD Line = EMA(12) - EMA(26)
        # Standard alignment: both EMAs calculated from same price data
        # EMA26 determines the start since it needs more data (26 vs 12)
        ema26_start_idx = 25  # EMA26 starts at index 25 (26th price)
        ema12_offset = ema26_start_idx - 11  # EMA12 starts at index 11, so offset is 14
        
        macd_line = []
        for i in range(len(ema26)):
            macd_value = ema12[i + ema12_offset] - ema26[i]
            macd_line.append(macd_value)
        
        # Signal Line = EMA(9) of MACD Line
        signal_line = self.calculate_ema(macd_line, 9)
        
        # Histogram = MACD - Signal (properly aligned)
        histogram = []
        if signal_line:
            # Signal line starts 8 positions after MACD line starts
            signal_start_offset = 8
            for i in range(len(signal_line)):
                hist_value = macd_line[i + signal_start_offset] - signal_line[i]
                histogram.append(hist_value)
        
        return macd_line, signal_line, histogram
    
    def detect_crossovers(self, macd_line: List[float], signal_line: List[float], 
                         histogram: List[float], closes: List[float]) -> List[dict]:
        """
        Detect MACD crossovers and generate signals.
        
        Args:
            macd_line: MACD line values
            signal_line: Signal line values
            histogram: Histogram values
            closes: Closing prices
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        if len(signal_line) < 2 or len(histogram) < 2:
            return signals
        
        # Calculate base offset: EMA26 starts at index 25, signal starts 8 after MACD
        base_offset = 25 + 8  # 33 total offset from original prices
        
        for i in range(1, len(signal_line)):
            # Align with original price data
            close_idx = i + base_offset
            macd_idx = i + 8  # Signal aligns to MACD with 8 offset
            
            # Safety checks
            if (close_idx >= len(closes) or 
                macd_idx >= len(macd_line) or 
                i >= len(histogram)):
                break
            
            signal = {
                'index': i,
                'close': closes[close_idx],
                'macd': macd_line[macd_idx],
                'signal_line': signal_line[i],
                'histogram': histogram[i],
                'signal': 'NONE',
                'trend': 'NEUTRAL',
                'momentum': 'WEAK'
            }
            
            # Previous values for crossover detection
            prev_macd = macd_line[macd_idx - 1] if macd_idx > 0 else macd_line[macd_idx]
            prev_signal = signal_line[i - 1]
            
            # Bullish crossover: MACD crosses above Signal
            if prev_macd <= prev_signal and signal['macd'] > signal['signal_line']:
                signal['signal'] = 'BUY'
                signal['trend'] = 'BULLISH'
                
                # Strong momentum if histogram is growing
                if i > 0 and histogram[i] > histogram[i-1]:
                    signal['momentum'] = 'STRONG'
                else:
                    signal['momentum'] = 'CONFIRMING'
            
            # Bearish crossover: MACD crosses below Signal
            elif prev_macd >= prev_signal and signal['macd'] < signal['signal_line']:
                signal['signal'] = 'SELL'
                signal['trend'] = 'BEARISH'
                
                # Strong momentum if histogram is growing (more negative)
                if i > 0 and histogram[i] < histogram[i-1]:
                    signal['momentum'] = 'STRONG'
                else:
                    signal['momentum'] = 'CONFIRMING'
            
            # No crossover but check momentum
            else:
                if signal['macd'] > signal['signal_line']:
                    signal['trend'] = 'BULLISH'
                    if histogram[i] > 0:
                        signal['momentum'] = 'UPTREND'
                elif signal['macd'] < signal['signal_line']:
                    signal['trend'] = 'BEARISH' 
                    if histogram[i] < 0:
                        signal['momentum'] = 'DOWNTREND'
            
            signals.append(signal)
        
        return signals
    
    def analyze(self, symbol: str, timeframe: str, limit: int):
        """
        Perform MACD analysis and return results as structured data.
        
        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            timeframe: Timeframe (e.g., '4h', '1d', '1h')
            limit: Number of candles to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        # Fetch OHLCV data
        ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 50:
            return {
                'error': f"Need at least 50 candles for MACD calculation. Got {len(ohlcv_data)}",
                'success': False
            }
        
        # Extract data
        timestamps = [candle[0] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        volumes = [candle[5] for candle in ohlcv_data]
        
        # Calculate MACD
        macd_line, signal_line, histogram = self.calculate_macd(closes)
        
        if not macd_line or not signal_line or not histogram:
            return {
                'error': "Unable to calculate MACD",
                'success': False
            }
        
        # Get latest values for analysis
        latest_macd = macd_line[-1]
        latest_signal = signal_line[-1]
        latest_histogram = histogram[-1]
        latest_close = closes[-1]
        
        # Calculate additional metrics
        macd_signal_ratio = abs(latest_macd / latest_signal) if latest_signal != 0 else 0
        histogram_strength = abs(latest_histogram)
        
        # Determine trend and momentum
        is_bullish = latest_macd > latest_signal
        is_growing = len(histogram) > 1 and histogram[-1] > histogram[-2]
        
        # Calculate confidence based on signal strength and alignment
        confidence = 0
        if abs(latest_histogram) > 0.001:  # Strong histogram
            confidence += 30
        if is_bullish and is_growing:  # Bullish momentum building
            confidence += 25
        elif not is_bullish and not is_growing:  # Bearish momentum building
            confidence += 25
        if macd_signal_ratio > 1.5:  # Strong divergence
            confidence += 20
        if len(volumes) > 10 and volumes[-1] > sum(volumes[-10:]) / 10:  # Volume confirmation
            confidence += 25
        
        confidence = min(confidence, 100)
        
        # Determine action
        action = "NEUTRAL"
        if latest_histogram > 0.002 and is_growing:
            action = "BUY"
        elif latest_histogram < -0.002 and not is_growing:
            action = "SELL"
        elif abs(latest_histogram) < 0.001:
            action = "WAITING FOR PATTERN"
        
        # Create summary
        summary = f"MACD {'above' if is_bullish else 'below'} signal line"
        if is_growing and is_bullish:
            summary += " + momentum accelerating upward"
        elif not is_growing and not is_bullish:
            summary += " + momentum accelerating downward"
        else:
            summary += " + momentum consolidating"
        
        # Calculate support/resistance (simplified)
        recent_closes = closes[-20:] if len(closes) >= 20 else closes
        support = min(recent_closes)
        resistance = max(recent_closes)
        
        # Risk/reward calculation
        risk_distance = latest_close - support
        reward_distance = resistance - latest_close
        rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
        
        # Get timestamps
        latest_timestamp = datetime.fromtimestamp(timestamps[-1] / 1000)
        first_timestamp = datetime.fromtimestamp(timestamps[0] / 1000)
        
        # Determine trend direction
        trend_direction = "Bullish" if is_bullish else "Bearish"
        
        # Determine momentum state
        momentum_state = "Accelerating" if is_growing else "Decelerating"
        
        # Entry window assessment
        if action in ['BUY', 'SELL']:
            entry_window = "Optimal now"
        else:
            entry_window = "Wait for crossover"
        
        # Exit trigger
        if is_bullish:
            exit_trigger = "MACD cross below signal"
        else:
            exit_trigger = "MACD cross above signal"
        
        # Expected drawdown
        max_drawdown = ((latest_close - support) / latest_close * 100)
        
        return {
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'analysis_time': latest_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': timestamps[-1],
            
            # Core indicators
            'macd_line': round(latest_macd, 6),
            'signal_line': round(latest_signal, 6),
            'histogram': round(latest_histogram, 6),
            'divergence_ratio': round(macd_signal_ratio, 2),
            'histogram_strength': round(histogram_strength, 6),
            
            # Price levels
            'current_price': round(latest_close, 4),
            'support_level': round(support, 4),
            'resistance_level': round(resistance, 4),
            'stop_zone': round(support * 0.98, 4),
            'tp_low': round(resistance * 1.02, 4),
            'tp_high': round(resistance * 1.05, 4),
            
            # Trading analysis
            'signal': action,
            'summary': summary,
            'confidence_score': confidence,
            'trend_direction': trend_direction,
            'momentum_state': momentum_state,
            'entry_window': entry_window,
            'exit_trigger': exit_trigger,
            'rr_ratio': round(rr_ratio, 1),
            'max_drawdown': round(max_drawdown, 1),
            
            # Additional data
            'is_bullish': is_bullish,
            'is_growing': is_growing,
            'volume_spike': volumes[-1],
            'raw_data': {
                'macd_values': macd_line,
                'signal_values': signal_line,
                'histogram_values': histogram,
                'ohlcv_data': ohlcv_data
            }
        }
    
    def _get_momentum_icon(self, momentum: str) -> str:
        """Get emoji icon for momentum type."""
        momentum_icons = {
            'STRONG': '🔥',
            'CONFIRMING': '🔄',
            'UPTREND': '📈',
            'DOWNTREND': '📉',
            'WEAK': '🧨'
        }
        return momentum_icons.get(momentum, '➖')