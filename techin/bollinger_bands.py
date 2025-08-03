"""
Bollinger Bands strategy implementation for cryptocurrency trend analysis.
Uses 20-period SMA with 2 standard deviation bands to detect volatility and reversal signals.
"""

import math
from datetime import datetime
from typing import List, Tuple, Optional
from data import get_data_collector


class BollingerBandsStrategy:
    """Bollinger Bands strategy for trend detection and volatility analysis."""
    
    def __init__(self, period: int = 20, multiplier: float = 2.0):
        self.collector = get_data_collector()
        self.period = period
        self.multiplier = multiplier
    
    def calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: List of closing prices
            period: SMA period (default 20)
            
        Returns:
            List of SMA values
        """
        if len(prices) < period:
            raise ValueError(f"Insufficient data: need at least {period} prices, got {len(prices)}")
        
        sma_values = []
        for i in range(period - 1, len(prices)):
            sma = sum(prices[i - period + 1:i + 1]) / period
            sma_values.append(sma)
        
        return sma_values
    
    def calculate_standard_deviation(self, prices: List[float], period: int, sma_values: List[float]) -> List[float]:
        """
        Calculate standard deviation for Bollinger Bands.
        
        Args:
            prices: List of closing prices
            period: Period for calculation
            sma_values: List of SMA values
            
        Returns:
            List of standard deviation values
        """
        if len(prices) < period or len(sma_values) == 0:
            raise ValueError(f"Insufficient data for standard deviation calculation: prices={len(prices)}, period={period}")
        
        std_values = []
        for i in range(len(sma_values)):
            price_idx = i + period - 1
            mean = sma_values[i]
            
            # Calculate variance
            variance = sum((prices[price_idx - j] - mean) ** 2 for j in range(period)) / period
            std_dev = math.sqrt(variance)
            std_values.append(std_dev)
        
        return std_values
    
    def calculate_bollinger_bands(self, prices: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate Bollinger Bands (Upper, Middle, Lower).
        
        Args:
            prices: List of closing prices
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        # Calculate middle band (SMA)
        middle_band = self.calculate_sma(prices, self.period)
        
        if not middle_band:
            raise RuntimeError("Failed to calculate middle band (SMA)")
        
        # Calculate standard deviation
        std_dev = self.calculate_standard_deviation(prices, self.period, middle_band)
        
        if not std_dev:
            raise RuntimeError("Failed to calculate standard deviation")
        
        # Calculate upper and lower bands
        upper_band = [mb + (self.multiplier * sd) for mb, sd in zip(middle_band, std_dev)]
        lower_band = [mb - (self.multiplier * sd) for mb, sd in zip(middle_band, std_dev)]
        
        return upper_band, middle_band, lower_band
    
    def calculate_volume_confirmation(self, volumes: List[float], window: int = 20) -> List[float]:
        """
        Calculate volume ratio for breakout confirmation (optimized).
        Industry standard: Volume should be 1.5x+ average for valid breakouts.
        
        Args:
            volumes: List of volume values
            window: Lookback period for volume average
            
        Returns:
            List of volume ratios (current volume / average volume)
        """
        if len(volumes) < window:
            return [1.0] * len(volumes)
        
        volume_ratios = [1.0] * (window - 1)  # Pre-fill initial values
        
        # Calculate initial window sum
        window_sum = sum(volumes[:window])
        
        for i in range(window - 1, len(volumes)):
            # Use rolling average for O(1) performance per iteration
            if i >= window:
                window_sum = window_sum - volumes[i - window] + volumes[i]
            
            avg_volume = window_sum / window
            ratio = volumes[i] / avg_volume if avg_volume > 0 else 1.0
            volume_ratios.append(ratio)
        
        return volume_ratios
    
    def detect_head_fake(self, prices: List[float], upper_band: List[float], 
                        lower_band: List[float], lookback: int = 3) -> List[bool]:
        """
        Detect potential head fake (false breakout) patterns.
        John Bollinger warns about brief breaks that quickly reverse.
        
        Args:
            prices: Closing prices
            upper_band: Upper Bollinger Band values
            lower_band: Lower Bollinger Band values
            lookback: Periods to check for reversal
            
        Returns:
            List indicating potential head fake patterns
        """
        if len(prices) < lookback + 2:
            return [False] * len(prices)
        
        head_fakes = []
        
        for i in range(len(prices)):
            if i < lookback + 1:
                head_fakes.append(False)
                continue
            
            head_fake = False
            
            # Check for upper band head fake (brief break above, then reversal)
            if i < len(upper_band):
                broke_upper = prices[i - lookback] > upper_band[i - lookback]
                back_inside = all(prices[j] < upper_band[j] for j in range(i - lookback + 1, i + 1) 
                                if j < len(upper_band))
                
                if broke_upper and back_inside:
                    head_fake = True
            
            # Check for lower band head fake (brief break below, then reversal)
            if i < len(lower_band) and not head_fake:
                broke_lower = prices[i - lookback] < lower_band[i - lookback]
                back_inside = all(prices[j] > lower_band[j] for j in range(i - lookback + 1, i + 1)
                                if j < len(lower_band))
                
                if broke_lower and back_inside:
                    head_fake = True
            
            head_fakes.append(head_fake)
        
        return head_fakes
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """
        Calculate RSI for directional bias confirmation (optimized).
        Industry standard 14-period RSI using Wilder's smoothing.
        
        Args:
            prices: Closing prices
            period: RSI period (default 14)
            
        Returns:
            List of RSI values
        """
        if len(prices) < period + 1:
            return [50.0] * len(prices)
        
        # Pre-allocate arrays for better performance
        rsi_values = [50.0] * len(prices)
        
        # Calculate initial gains and losses
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            delta = prices[i] - prices[i-1]
            gains.append(delta if delta > 0 else 0)
            losses.append(-delta if delta < 0 else 0)
        
        if len(gains) < period:
            return rsi_values
        
        # Initial average gain/loss (simple average)
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        # Calculate RSI values using Wilder's smoothing
        alpha = 1.0 / period  # Smoothing factor
        
        for i in range(period, len(gains)):
            # Wilder's exponential smoothing (more efficient than recalculating)
            avg_gain = avg_gain * (1 - alpha) + gains[i] * alpha
            avg_loss = avg_loss * (1 - alpha) + losses[i] * alpha
            
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values[i + 1] = rsi  # +1 to account for deltas starting at index 1
        
        return rsi_values
    
    def calculate_macd_line(self, prices: List[float], fast: int = 12, slow: int = 26) -> List[float]:
        """
        Calculate MACD line for additional trend confirmation.
        Industry standard: 12-period EMA - 26-period EMA.
        
        Args:
            prices: Closing prices
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            
        Returns:
            List of MACD line values
        """
        if len(prices) < slow:
            return [0.0] * len(prices)
        
        # Calculate EMAs
        def calculate_ema(data: List[float], period: int) -> List[float]:
            ema_values = [0.0] * len(data)
            multiplier = 2.0 / (period + 1)
            
            # Start with SMA for first value
            ema_values[period - 1] = sum(data[:period]) / period
            
            # Calculate EMA for remaining values
            for i in range(period, len(data)):
                ema_values[i] = (data[i] - ema_values[i-1]) * multiplier + ema_values[i-1]
                
            return ema_values
        
        fast_ema = calculate_ema(prices, fast)
        slow_ema = calculate_ema(prices, slow)
        
        # MACD Line = Fast EMA - Slow EMA
        macd_line = []
        for i in range(len(prices)):
            if i >= slow - 1:
                macd_line.append(fast_ema[i] - slow_ema[i])
            else:
                macd_line.append(0.0)
                
        return macd_line
    
    def detect_squeeze(self, upper_band: List[float], lower_band: List[float], window: int = 20) -> List[bool]:
        """
        Detect Bollinger Band squeeze (low volatility periods).
        
        Args:
            upper_band: Upper Bollinger Band values
            lower_band: Lower Bollinger Band values
            window: Lookback period for squeeze detection
            
        Returns:
            List of boolean values indicating squeeze periods
        """
        if len(upper_band) != len(lower_band) or len(upper_band) < window:
            return [False] * len(upper_band)
        
        squeezes = []
        
        for i in range(len(upper_band)):
            if i < window:
                squeezes.append(False)
                continue
            
            # Current band width
            current_width = upper_band[i] - lower_band[i]
            
            # Average band width over lookback window
            avg_width = sum(upper_band[j] - lower_band[j] for j in range(i - window, i)) / window
            
            # Squeeze if current width is significantly smaller than average
            squeeze = current_width < (avg_width * 0.7)
            squeezes.append(squeeze)
        
        return squeezes
    
    def generate_signals(self, prices: List[float], upper_band: List[float], 
                        middle_band: List[float], lower_band: List[float], 
                        volumes: Optional[List[float]] = None) -> List[dict]:
        """
        Generate enhanced Bollinger Bands trading signals with volume confirmation,
        head fake protection, and RSI directional bias.
        
        Args:
            prices: Closing prices (aligned with bands)
            upper_band: Upper Bollinger Band values
            middle_band: Middle Bollinger Band values (SMA)
            lower_band: Lower Bollinger Band values
            volumes: Volume data for confirmation (optional)
            
        Returns:
            List of enhanced signal dictionaries
        """
        signals = []
        squeezes = self.detect_squeeze(upper_band, lower_band)
        
        # Align prices with bands (bands start at index period-1)
        aligned_prices = prices[self.period - 1:]
        
        # Calculate enhanced indicators
        volume_ratios = self.calculate_volume_confirmation(volumes) if volumes else [1.0] * len(prices)
        head_fakes = self.detect_head_fake(aligned_prices, upper_band, lower_band)
        rsi_values = self.calculate_rsi(aligned_prices)
        macd_values = self.calculate_macd_line(aligned_prices)
        
        # Align volume ratios with prices
        aligned_volume_ratios = volume_ratios[self.period - 1:] if len(volume_ratios) >= self.period else [1.0] * len(aligned_prices)
        
        for i in range(1, len(middle_band)):
            if i >= len(aligned_prices):
                break
                
            signal = {
                'index': i,
                'price': aligned_prices[i],
                'upper_band': upper_band[i],
                'middle_band': middle_band[i],
                'lower_band': lower_band[i],
                'signal': 'HOLD',
                'squeeze': squeezes[i] if i < len(squeezes) else False,
                'description': 'Neutral',
                'volume_ratio': aligned_volume_ratios[i] if i < len(aligned_volume_ratios) else 1.0,
                'head_fake_risk': head_fakes[i] if i < len(head_fakes) else False,
                'rsi': rsi_values[i] if i < len(rsi_values) else 50.0,
                'macd': macd_values[i] if i < len(macd_values) else 0.0,
                'volume_confirmed': False,
                'directional_bias': 'NEUTRAL'
            }
            
            current_price = aligned_prices[i]
            prev_price = aligned_prices[i - 1]
            current_volume_ratio = aligned_volume_ratios[i] if i < len(aligned_volume_ratios) else 1.0
            current_rsi = rsi_values[i] if i < len(rsi_values) else 50.0
            current_macd = macd_values[i] if i < len(macd_values) else 0.0
            is_head_fake = head_fakes[i] if i < len(head_fakes) else False
            
            # Calculate position within bands (0 = lower band, 0.5 = middle, 1 = upper band)
            band_width = upper_band[i] - lower_band[i]
            if band_width > 0:
                position = (current_price - lower_band[i]) / band_width
            else:
                position = 0.5
            
            # Enhanced signal logic with volume confirmation and directional bias
            volume_confirmed = current_volume_ratio >= 1.5  # Industry standard
            signal['volume_confirmed'] = volume_confirmed
            
            # Determine directional bias from RSI and MACD confluence
            rsi_bullish = current_rsi > 60
            rsi_bearish = current_rsi < 40
            macd_bullish = current_macd > 0 and i > 0 and current_macd > macd_values[i-1] if i < len(macd_values) else False
            macd_bearish = current_macd < 0 and i > 0 and current_macd < macd_values[i-1] if i < len(macd_values) else False
            
            # Enhanced directional bias with confluence
            if (rsi_bullish and macd_bullish) or (rsi_bullish and current_macd > 0):
                signal['directional_bias'] = 'STRONG_BULLISH'
            elif (rsi_bearish and macd_bearish) or (rsi_bearish and current_macd < 0):
                signal['directional_bias'] = 'STRONG_BEARISH'
            elif rsi_bullish or macd_bullish:
                signal['directional_bias'] = 'BULLISH'
            elif rsi_bearish or macd_bearish:
                signal['directional_bias'] = 'BEARISH'
            else:
                signal['directional_bias'] = 'NEUTRAL'
            
            # Enhanced BUY signals with volume confirmation and head fake protection
            if prev_price <= lower_band[i - 1] and current_price > lower_band[i] and not is_head_fake:
                if volume_confirmed and signal['directional_bias'] != 'BEARISH':
                    signal['signal'] = 'BUY'
                    signal['description'] = 'Oversold Bounce (Volume Confirmed)'
                    if squeezes[i]:
                        signal['description'] = 'Squeeze Breakout - Buy (Volume Confirmed)'
                elif signal['directional_bias'] == 'BULLISH':
                    signal['signal'] = 'BUY'
                    signal['description'] = 'Oversold Bounce (RSI Bullish)'
                else:
                    signal['signal'] = 'HOLD'
                    signal['description'] = 'Potential Bounce (Low Volume/Bearish RSI)'
            
            # Strong BUY: Price breaks above middle band from below
            elif prev_price < middle_band[i - 1] and current_price >= middle_band[i] and not is_head_fake:
                if volume_confirmed:
                    signal['signal'] = 'BUY'
                    signal['description'] = 'Middle Band Breakout (Volume Confirmed)'
                elif signal['directional_bias'] == 'BULLISH':
                    signal['signal'] = 'BUY'
                    signal['description'] = 'Middle Band Breakout (RSI Bullish)'
                else:
                    signal['signal'] = 'HOLD'
                    signal['description'] = 'Weak Middle Band Break'
            
            # Enhanced SELL signals with volume confirmation and head fake protection
            elif prev_price >= upper_band[i - 1] and current_price < upper_band[i] and not is_head_fake:
                if volume_confirmed and signal['directional_bias'] != 'BULLISH':
                    signal['signal'] = 'SELL'
                    signal['description'] = 'Overbought Reversal (Volume Confirmed)'
                elif signal['directional_bias'] == 'BEARISH':
                    signal['signal'] = 'SELL'
                    signal['description'] = 'Overbought Reversal (RSI Bearish)'
                else:
                    signal['signal'] = 'HOLD'
                    signal['description'] = 'Potential Reversal (Low Volume/Bullish RSI)'
            
            # Strong SELL: Price breaks below middle band from above
            elif prev_price > middle_band[i - 1] and current_price <= middle_band[i] and not is_head_fake:
                if volume_confirmed:
                    signal['signal'] = 'SELL'
                    signal['description'] = 'Middle Band Breakdown (Volume Confirmed)'
                elif signal['directional_bias'] == 'BEARISH':
                    signal['signal'] = 'SELL'
                    signal['description'] = 'Middle Band Breakdown (RSI Bearish)'
                else:
                    signal['signal'] = 'HOLD'
                    signal['description'] = 'Weak Middle Band Break'
            
            # Head fake detected - reduce signal strength
            elif is_head_fake:
                signal['signal'] = 'HOLD'
                signal['description'] = 'Head Fake Detected - Avoid Trade'
            
            # Position-based descriptions for HOLD signals
            elif position >= 0.8:  # Near upper band
                signal['signal'] = 'HOLD'
                signal['description'] = f'Near Upper Band ({position:.1%})'
                
            elif position <= 0.2:  # Near lower band
                signal['signal'] = 'HOLD'
                signal['description'] = f'Near Lower Band ({position:.1%})'
                
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
        Perform Bollinger Bands analysis and return structured results.
        
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
        
        if len(ohlcv_data) < 50:
            return {
                'error': f"Need at least 50 candles for Bollinger Bands calculation. Got {len(ohlcv_data)}",
                'success': False
            }
        
        # Extract data
        timestamps = [candle[0] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        volumes = [candle[5] for candle in ohlcv_data]
        
        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(closes)
        
        if not upper_band or not middle_band or not lower_band:
            return {
                'error': "Unable to calculate Bollinger Bands",
                'success': False
            }
        
        # Generate enhanced signals with volume data
        signals = self.generate_signals(closes, upper_band, middle_band, lower_band, volumes)
        
        # Get latest signal for analysis
        if not signals:
            return {
                'error': "No signals generated",
                'success': False
            }
            
        latest_signal = signals[-1]
        
        # Calculate confidence score
        band_position = ((latest_signal['price'] - latest_signal['lower_band']) / 
                        (latest_signal['upper_band'] - latest_signal['lower_band']) * 100) \
                        if latest_signal['upper_band'] != latest_signal['lower_band'] else 50
                        
        # Calculate volatility and volume ratio for confidence
        recent_closes = closes[-20:] if len(closes) >= 20 else closes
        volatility = (max(recent_closes) - min(recent_closes)) / min(recent_closes) * 100
        
        recent_volumes = volumes[-10:] if len(volumes) >= 10 else volumes
        avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1
        volume_ratio = latest_signal.get('volume', avg_volume) / avg_volume if avg_volume > 0 else 1
        
        confidence_score = self._calculate_confidence(latest_signal, volatility, volume_ratio, band_position)
        
        # Return structured analysis results
        return {
            'success': True,
            'analysis_time': datetime.fromtimestamp(timestamps[-1] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': timestamps[-1],
            
            # Core indicators
            'current_price': latest_signal['price'],
            'upper_band': latest_signal['upper_band'],
            'middle_band': latest_signal['middle_band'],
            'lower_band': latest_signal['lower_band'],
            'band_position': band_position,
            
            # Trading signals
            'signal': latest_signal['signal'],
            'description': latest_signal['description'],
            'squeeze': latest_signal['squeeze'],
            'confidence_score': confidence_score,
            
            # Additional data
            'all_signals': signals,
            'raw_data': {
                'upper_band': upper_band,
                'middle_band': middle_band,
                'lower_band': lower_band,
                'closes': closes,
                'timestamps': timestamps,
                'volumes': volumes
            }
        }
    
    
    def _calculate_confidence(self, signal: dict, volatility: float, volume_ratio: float, position_pct: float) -> int:
        """Calculate enhanced confidence score based on multiple factors including new enhancements."""
        confidence = 40  # Base confidence (reduced to account for new factors)
        
        # Signal strength
        if signal['signal'] == 'BUY' or signal['signal'] == 'SELL':
            confidence += 20
        
        # Band position strength
        if position_pct > 80 or position_pct < 20:
            confidence += 15  # Strong position
        elif position_pct > 60 or position_pct < 40:
            confidence += 10  # Moderate position
        
        # Enhanced volume confirmation (industry standard)
        if signal.get('volume_confirmed', False):
            confidence += 20  # Strong volume confirmation
        elif volume_ratio > 1.2:
            confidence += 10  # Moderate volume
        elif volume_ratio < 0.8:
            confidence -= 10  # Weak volume
        
        # Enhanced directional bias confirmation from RSI + MACD confluence
        directional_bias = signal.get('directional_bias', 'NEUTRAL')
        if directional_bias != 'NEUTRAL':
            if ((signal['signal'] == 'BUY' and directional_bias in ['BULLISH', 'STRONG_BULLISH']) or 
                (signal['signal'] == 'SELL' and directional_bias in ['BEARISH', 'STRONG_BEARISH'])):
                if 'STRONG_' in directional_bias:
                    confidence += 25  # Strong confluence (RSI + MACD) confirms signal
                else:
                    confidence += 15  # Single indicator confirms signal direction
            elif ((signal['signal'] == 'BUY' and directional_bias in ['BEARISH', 'STRONG_BEARISH']) or 
                  (signal['signal'] == 'SELL' and directional_bias in ['BULLISH', 'STRONG_BULLISH'])):
                if 'STRONG_' in directional_bias:
                    confidence -= 25  # Strong confluence contradicts signal
                else:
                    confidence -= 15  # Single indicator contradicts signal
        
        # Head fake protection
        if signal.get('head_fake_risk', False):
            confidence -= 25  # Significant penalty for head fake risk
        
        # Squeeze condition (volatility compression)
        if signal.get('squeeze', False):
            confidence += 12  # Squeeze suggests pending breakout
        
        # Volatility consideration (industry standard ranges)
        if 2 <= volatility <= 8:
            confidence += 8  # Optimal volatility
        elif volatility > 15:
            confidence -= 15  # Too volatile for reliable signals
        elif volatility < 1:
            confidence -= 8  # Too low volatility
        
        # RSI extreme levels provide additional confidence
        rsi = signal.get('rsi', 50)
        if rsi > 70 and signal['signal'] == 'SELL':
            confidence += 10  # Overbought confirmation
        elif rsi < 30 and signal['signal'] == 'BUY':
            confidence += 10  # Oversold confirmation
        elif rsi > 80 or rsi < 20:
            confidence += 5  # Extreme RSI levels
        
        return min(95, max(15, confidence))
    
    def _analyze_trend_momentum(self, signal: dict, recent_closes: List[float]) -> tuple:
        """Analyze trend direction and momentum state."""
        if len(recent_closes) < 3:
            return "Neutral", "Unclear"
        
        # Trend direction
        if signal['signal'] == 'BUY':
            trend_direction = "Bullish"
        elif signal['signal'] == 'SELL':
            trend_direction = "Bearish"
        else:
            # Determine from recent price action
            if recent_closes[-1] > recent_closes[0]:
                trend_direction = "Bullish"
            elif recent_closes[-1] < recent_closes[0]:
                trend_direction = "Bearish"
            else:
                trend_direction = "Neutral"
        
        # Momentum state
        price_change = (recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 100
        if abs(price_change) > 2:
            momentum_state = "Accelerating"
        elif abs(price_change) > 0.5:
            momentum_state = "Building"
        else:
            momentum_state = "Consolidating"
        
        return trend_direction, momentum_state


