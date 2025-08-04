"""
Keltner Channel strategy implementation for cryptocurrency trend analysis.
Uses EMA with ATR-based bands to detect volatility and trend continuation/reversal signals.
"""

import math
from datetime import datetime
from typing import List, Tuple, Optional
from data import get_data_collector


class KeltnerChannelStrategy:
    """Keltner Channel strategy for trend detection and volatility analysis."""
    
    def __init__(self, ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0):
        self.collector = get_data_collector()
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier
    
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
            raise ValueError(f"Insufficient data: need at least {period} prices, got {len(prices)}")
        
        ema_values = [0.0] * len(prices)
        multiplier = 2.0 / (period + 1)
        
        # Start with SMA for first value
        ema_values[period - 1] = sum(prices[:period]) / period
        
        # Calculate EMA for remaining values
        for i in range(period, len(prices)):
            ema_values[i] = (prices[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
            
        return ema_values
    
    def calculate_true_range(self, highs: List[float], lows: List[float], closes: List[float]) -> List[float]:
        """
        Calculate True Range values.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            
        Returns:
            List of True Range values
        """
        if len(highs) != len(lows) or len(lows) != len(closes):
            raise ValueError("Highs, lows, and closes must have the same length")
        
        if len(highs) < 2:
            return [0.0] * len(highs)
        
        true_ranges = [highs[0] - lows[0]]  # First TR is just high - low
        
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        return true_ranges
    
    def calculate_atr(self, true_ranges: List[float]) -> List[float]:
        """
        Calculate Average True Range using EMA smoothing.
        
        Args:
            true_ranges: List of True Range values
            
        Returns:
            List of ATR values
        """
        if len(true_ranges) < self.atr_period:
            return [0.0] * len(true_ranges)
        
        atr_values = [0.0] * len(true_ranges)
        multiplier = 2.0 / (self.atr_period + 1)
        
        # Start with SMA for first ATR value
        atr_values[self.atr_period - 1] = sum(true_ranges[:self.atr_period]) / self.atr_period
        
        # Calculate ATR using EMA smoothing
        for i in range(self.atr_period, len(true_ranges)):
            atr_values[i] = (true_ranges[i] - atr_values[i-1]) * multiplier + atr_values[i-1]
            
        return atr_values
    
    def calculate_keltner_channels(self, highs: List[float], lows: List[float], closes: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate Keltner Channels (Upper, Middle, Lower).
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            
        Returns:
            Tuple of (upper_channel, middle_channel, lower_channel)
        """
        # Calculate middle channel (EMA)
        middle_channel = self.calculate_ema(closes, self.ema_period)
        
        if not middle_channel:
            raise RuntimeError("Failed to calculate middle channel (EMA)")
        
        # Calculate True Range and ATR
        true_ranges = self.calculate_true_range(highs, lows, closes)
        atr_values = self.calculate_atr(true_ranges)
        
        if not atr_values:
            raise RuntimeError("Failed to calculate ATR")
        
        # Calculate upper and lower channels
        # Align ATR with EMA (both start being valid at their respective periods)
        start_index = max(self.ema_period - 1, self.atr_period - 1)
        
        upper_channel = []
        lower_channel = []
        aligned_middle = []
        
        for i in range(start_index, len(closes)):
            ema_val = middle_channel[i]
            atr_val = atr_values[i]
            
            upper_channel.append(ema_val + (self.multiplier * atr_val))
            lower_channel.append(ema_val - (self.multiplier * atr_val))
            aligned_middle.append(ema_val)
        
        return upper_channel, aligned_middle, lower_channel
    
    def calculate_volume_confirmation(self, volumes: List[float], window: int = 20) -> List[float]:
        """
        Calculate volume ratio for breakout confirmation.
        
        Args:
            volumes: List of volume values
            window: Lookback period for volume average
            
        Returns:
            List of volume ratios (current volume / average volume)
        """
        if len(volumes) < window:
            return [1.0] * len(volumes)
        
        volume_ratios = [1.0] * (window - 1)
        window_sum = sum(volumes[:window])
        
        for i in range(window - 1, len(volumes)):
            if i >= window:
                window_sum = window_sum - volumes[i - window] + volumes[i]
            
            avg_volume = window_sum / window
            ratio = volumes[i] / avg_volume if avg_volume > 0 else 1.0
            volume_ratios.append(ratio)
        
        return volume_ratios
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """
        Calculate RSI for directional bias confirmation.
        
        Args:
            prices: Closing prices
            period: RSI period (default 14)
            
        Returns:
            List of RSI values
        """
        if len(prices) < period + 1:
            return [50.0] * len(prices)
        
        rsi_values = [50.0] * len(prices)
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            delta = prices[i] - prices[i-1]
            gains.append(delta if delta > 0 else 0)
            losses.append(-delta if delta < 0 else 0)
        
        if len(gains) < period:
            return rsi_values
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        alpha = 1.0 / period
        
        for i in range(period, len(gains)):
            avg_gain = avg_gain * (1 - alpha) + gains[i] * alpha
            avg_loss = avg_loss * (1 - alpha) + losses[i] * alpha
            
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values[i + 1] = rsi
        
        return rsi_values
    
    def detect_squeeze(self, upper_channel: List[float], lower_channel: List[float], window: int = 20) -> List[bool]:
        """
        Detect Keltner Channel squeeze (low volatility periods).
        
        Args:
            upper_channel: Upper Keltner Channel values
            lower_channel: Lower Keltner Channel values
            window: Lookback period for squeeze detection
            
        Returns:
            List of boolean values indicating squeeze periods
        """
        if len(upper_channel) != len(lower_channel) or len(upper_channel) < window:
            return [False] * len(upper_channel)
        
        squeezes = []
        
        for i in range(len(upper_channel)):
            if i < window:
                squeezes.append(False)
                continue
            
            current_width = upper_channel[i] - lower_channel[i]
            avg_width = sum(upper_channel[j] - lower_channel[j] for j in range(i - window, i)) / window
            
            squeeze = current_width < (avg_width * 0.75)
            squeezes.append(squeeze)
        
        return squeezes
    
    def generate_signals(self, highs: List[float], lows: List[float], closes: List[float], 
                        upper_channel: List[float], middle_channel: List[float], 
                        lower_channel: List[float], volumes: Optional[List[float]] = None) -> List[dict]:
        """
        Generate Keltner Channel trading signals with volume confirmation and RSI bias.
        
        Args:
            highs: High prices
            lows: Low prices
            closes: Closing prices
            upper_channel: Upper Keltner Channel values
            middle_channel: Middle Keltner Channel values (EMA)
            lower_channel: Lower Keltner Channel values
            volumes: Volume data for confirmation (optional)
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        squeezes = self.detect_squeeze(upper_channel, lower_channel)
        
        # Align data - channels start later due to EMA and ATR periods
        start_index = max(self.ema_period - 1, self.atr_period - 1)
        aligned_highs = highs[start_index:]
        aligned_lows = lows[start_index:]
        aligned_closes = closes[start_index:]
        
        # Calculate indicators
        volume_ratios = self.calculate_volume_confirmation(volumes) if volumes else [1.0] * len(closes)
        rsi_values = self.calculate_rsi(aligned_closes)
        
        # Align volume ratios
        aligned_volume_ratios = volume_ratios[start_index:] if len(volume_ratios) > start_index else [1.0] * len(aligned_closes)
        
        for i in range(1, len(middle_channel)):
            if i >= len(aligned_closes):
                break
                
            signal = {
                'index': i,
                'high': aligned_highs[i],
                'low': aligned_lows[i],
                'close': aligned_closes[i],
                'upper_channel': upper_channel[i],
                'middle_channel': middle_channel[i],
                'lower_channel': lower_channel[i],
                'signal': 'HOLD',
                'squeeze': squeezes[i] if i < len(squeezes) else False,
                'description': 'Neutral',
                'volume_ratio': aligned_volume_ratios[i] if i < len(aligned_volume_ratios) else 1.0,
                'rsi': rsi_values[i] if i < len(rsi_values) else 50.0,
                'volume_confirmed': False,
                'directional_bias': 'NEUTRAL'
            }
            
            current_close = aligned_closes[i]
            prev_close = aligned_closes[i - 1]
            current_volume_ratio = aligned_volume_ratios[i] if i < len(aligned_volume_ratios) else 1.0
            current_rsi = rsi_values[i] if i < len(rsi_values) else 50.0
            
            # Calculate position within channels
            channel_width = upper_channel[i] - lower_channel[i]
            if channel_width > 0:
                position = (current_close - lower_channel[i]) / channel_width
            else:
                position = 0.5
            
            # Volume confirmation
            volume_confirmed = current_volume_ratio >= 1.5
            signal['volume_confirmed'] = volume_confirmed
            
            # Directional bias from RSI
            if current_rsi > 70:
                signal['directional_bias'] = 'BEARISH'
            elif current_rsi < 30:
                signal['directional_bias'] = 'BULLISH'
            elif current_rsi > 60:
                signal['directional_bias'] = 'WEAK_BEARISH'
            elif current_rsi < 40:
                signal['directional_bias'] = 'WEAK_BULLISH'
            else:
                signal['directional_bias'] = 'NEUTRAL'
            
            # Signal generation logic
            # BUY signals: Breakout above upper channel or bounce from lower channel
            if current_close > upper_channel[i] and prev_close <= upper_channel[i - 1]:
                if volume_confirmed and signal['directional_bias'] != 'BEARISH':
                    signal['signal'] = 'BUY'
                    signal['description'] = 'Upper Channel Breakout (Volume Confirmed)'
                    if squeezes[i]:
                        signal['description'] = 'Squeeze Breakout - Buy (Volume Confirmed)'
                elif signal['directional_bias'] in ['BULLISH', 'WEAK_BULLISH']:
                    signal['signal'] = 'BUY'
                    signal['description'] = 'Upper Channel Breakout (RSI Bullish)'
                else:
                    signal['signal'] = 'HOLD'
                    signal['description'] = 'Weak Upper Breakout'
            
            # Bounce from lower channel
            elif prev_close <= lower_channel[i - 1] and current_close > lower_channel[i]:
                if volume_confirmed and signal['directional_bias'] != 'BEARISH':
                    signal['signal'] = 'BUY'
                    signal['description'] = 'Lower Channel Bounce (Volume Confirmed)'
                elif signal['directional_bias'] in ['BULLISH', 'WEAK_BULLISH']:
                    signal['signal'] = 'BUY'
                    signal['description'] = 'Lower Channel Bounce (RSI Bullish)'
                else:
                    signal['signal'] = 'HOLD'
                    signal['description'] = 'Potential Bounce (Low Confidence)'
            
            # Middle channel cross up
            elif prev_close < middle_channel[i - 1] and current_close >= middle_channel[i]:
                if volume_confirmed:
                    signal['signal'] = 'BUY'
                    signal['description'] = 'Middle Channel Break Up (Volume Confirmed)'
                elif signal['directional_bias'] in ['BULLISH', 'WEAK_BULLISH']:
                    signal['signal'] = 'BUY'
                    signal['description'] = 'Middle Channel Break Up (RSI Bullish)'
                else:
                    signal['signal'] = 'HOLD'
                    signal['description'] = 'Weak Middle Break Up'
            
            # SELL signals: Breakdown below lower channel or rejection from upper channel
            elif current_close < lower_channel[i] and prev_close >= lower_channel[i - 1]:
                if volume_confirmed and signal['directional_bias'] != 'BULLISH':
                    signal['signal'] = 'SELL'
                    signal['description'] = 'Lower Channel Breakdown (Volume Confirmed)'
                elif signal['directional_bias'] in ['BEARISH', 'WEAK_BEARISH']:
                    signal['signal'] = 'SELL'
                    signal['description'] = 'Lower Channel Breakdown (RSI Bearish)'
                else:
                    signal['signal'] = 'HOLD'
                    signal['description'] = 'Weak Lower Breakdown'
            
            # Rejection from upper channel
            elif prev_close >= upper_channel[i - 1] and current_close < upper_channel[i]:
                if volume_confirmed and signal['directional_bias'] != 'BULLISH':
                    signal['signal'] = 'SELL'
                    signal['description'] = 'Upper Channel Rejection (Volume Confirmed)'
                elif signal['directional_bias'] in ['BEARISH', 'WEAK_BEARISH']:
                    signal['signal'] = 'SELL'
                    signal['description'] = 'Upper Channel Rejection (RSI Bearish)'
                else:
                    signal['signal'] = 'HOLD'
                    signal['description'] = 'Potential Rejection (Low Confidence)'
            
            # Middle channel cross down
            elif prev_close > middle_channel[i - 1] and current_close <= middle_channel[i]:
                if volume_confirmed:
                    signal['signal'] = 'SELL'
                    signal['description'] = 'Middle Channel Break Down (Volume Confirmed)'
                elif signal['directional_bias'] in ['BEARISH', 'WEAK_BEARISH']:
                    signal['signal'] = 'SELL'
                    signal['description'] = 'Middle Channel Break Down (RSI Bearish)'
                else:
                    signal['signal'] = 'HOLD'
                    signal['description'] = 'Weak Middle Break Down'
            
            # Position-based descriptions for HOLD signals
            elif position >= 0.85:
                signal['signal'] = 'HOLD'
                signal['description'] = f'Near Upper Channel ({position:.1%})'
            elif position <= 0.15:
                signal['signal'] = 'HOLD'
                signal['description'] = f'Near Lower Channel ({position:.1%})'
            elif squeezes[i]:
                signal['signal'] = 'HOLD'
                signal['description'] = f'Volatility Squeeze ({position:.1%})'
            else:
                signal['signal'] = 'HOLD'
                signal['description'] = f'Normal Range ({position:.1%})'
            
            signals.append(signal)
        
        return signals
    
    def analyze(self, symbol: str, timeframe: str, limit: int, ohlcv_data: Optional[List] = None) -> dict:
        """
        Perform Keltner Channel analysis and return structured results.
        
        Args:
            symbol: Trading pair (e.g., 'ETH/USDT')
            timeframe: Timeframe (e.g., '1d', '4h', '1h')
            limit: Number of candles to analyze
            ohlcv_data: Optional pre-fetched OHLCV data
            
        Returns:
            Dictionary containing analysis results
        """
        # Fetch OHLCV data if not provided
        if ohlcv_data is None:
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        min_required = max(self.ema_period, self.atr_period) + 20
        if len(ohlcv_data) < min_required:
            return {
                'error': f"Need at least {min_required} candles for Keltner Channel calculation. Got {len(ohlcv_data)}",
                'success': False
            }
        
        # Extract data
        timestamps = [candle[0] for candle in ohlcv_data]
        opens = [candle[1] for candle in ohlcv_data]
        highs = [candle[2] for candle in ohlcv_data]
        lows = [candle[3] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        volumes = [candle[5] for candle in ohlcv_data]
        
        # Calculate Keltner Channels
        upper_channel, middle_channel, lower_channel = self.calculate_keltner_channels(highs, lows, closes)
        
        if not upper_channel or not middle_channel or not lower_channel:
            return {
                'error': "Unable to calculate Keltner Channels",
                'success': False
            }
        
        # Generate signals
        signals = self.generate_signals(highs, lows, closes, upper_channel, middle_channel, lower_channel, volumes)
        
        if not signals:
            return {
                'error': "No signals generated",
                'success': False
            }
            
        latest_signal = signals[-1]
        
        # Calculate confidence score
        channel_position = ((latest_signal['close'] - latest_signal['lower_channel']) / 
                           (latest_signal['upper_channel'] - latest_signal['lower_channel']) * 100) \
                           if latest_signal['upper_channel'] != latest_signal['lower_channel'] else 50
                           
        # Calculate volatility
        recent_closes = closes[-20:] if len(closes) >= 20 else closes
        volatility = (max(recent_closes) - min(recent_closes)) / min(recent_closes) * 100
        
        # Volume ratio
        recent_volumes = volumes[-10:] if len(volumes) >= 10 else volumes
        avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1
        volume_ratio = latest_signal.get('volume_ratio', 1.0)
        
        confidence_score = self._calculate_confidence(latest_signal, volatility, volume_ratio, channel_position)
        
        # Return structured analysis results
        return {
            'success': True,
            'analysis_time': datetime.fromtimestamp(timestamps[-1] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': timestamps[-1],
            
            # Core indicators
            'current_price': latest_signal['close'],
            'upper_channel': latest_signal['upper_channel'],
            'middle_channel': latest_signal['middle_channel'],
            'lower_channel': latest_signal['lower_channel'],
            'channel_position': channel_position,
            
            # Trading signals
            'signal': latest_signal['signal'],
            'description': latest_signal['description'],
            'squeeze': latest_signal['squeeze'],
            'confidence_score': confidence_score,
            
            # Additional data
            'all_signals': signals,
            'raw_data': {
                'upper_channel': upper_channel,
                'middle_channel': middle_channel,
                'lower_channel': lower_channel,
                'highs': highs,
                'lows': lows,
                'closes': closes,
                'timestamps': timestamps,
                'volumes': volumes
            }
        }
    
    def _calculate_confidence(self, signal: dict, volatility: float, volume_ratio: float, position_pct: float) -> int:
        """Calculate confidence score based on multiple factors."""
        confidence = 40  # Base confidence
        
        # Signal strength
        if signal['signal'] == 'BUY' or signal['signal'] == 'SELL':
            confidence += 20
        
        # Channel position strength
        if position_pct > 85 or position_pct < 15:
            confidence += 15  # Strong position
        elif position_pct > 70 or position_pct < 30:
            confidence += 10  # Moderate position
        
        # Volume confirmation
        if signal.get('volume_confirmed', False):
            confidence += 20
        elif volume_ratio > 1.2:
            confidence += 10
        elif volume_ratio < 0.8:
            confidence -= 10
        
        # Directional bias confirmation
        directional_bias = signal.get('directional_bias', 'NEUTRAL')
        if directional_bias != 'NEUTRAL':
            if ((signal['signal'] == 'BUY' and 'BULLISH' in directional_bias) or 
                (signal['signal'] == 'SELL' and 'BEARISH' in directional_bias)):
                confidence += 15
            elif ((signal['signal'] == 'BUY' and 'BEARISH' in directional_bias) or 
                  (signal['signal'] == 'SELL' and 'BULLISH' in directional_bias)):
                confidence -= 15
        
        # Squeeze condition
        if signal.get('squeeze', False):
            confidence += 12
        
        # Volatility consideration
        if 2 <= volatility <= 8:
            confidence += 8
        elif volatility > 15:
            confidence -= 15
        elif volatility < 1:
            confidence -= 8
        
        # RSI extreme levels
        rsi = signal.get('rsi', 50)
        if rsi > 70 and signal['signal'] == 'SELL':
            confidence += 10
        elif rsi < 30 and signal['signal'] == 'BUY':
            confidence += 10
        elif rsi > 80 or rsi < 20:
            confidence += 5
        
        return min(95, max(15, confidence))