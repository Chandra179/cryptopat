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
                        middle_band: List[float], lower_band: List[float]) -> List[dict]:
        """
        Generate Bollinger Bands trading signals.
        
        Args:
            prices: Closing prices (aligned with bands)
            upper_band: Upper Bollinger Band values
            middle_band: Middle Bollinger Band values (SMA)
            lower_band: Lower Bollinger Band values
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        squeezes = self.detect_squeeze(upper_band, lower_band)
        
        # Align prices with bands (bands start at index period-1)
        aligned_prices = prices[self.period - 1:]
        
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
                'description': 'Neutral'
            }
            
            current_price = aligned_prices[i]
            prev_price = aligned_prices[i - 1]
            
            # Calculate position within bands (0 = lower band, 0.5 = middle, 1 = upper band)
            band_width = upper_band[i] - lower_band[i]
            if band_width > 0:
                position = (current_price - lower_band[i]) / band_width
            else:
                position = 0.5
            
            # BUY signals
            if prev_price <= lower_band[i - 1] and current_price > lower_band[i]:
                signal['signal'] = 'BUY'
                signal['description'] = 'Oversold Bounce'
                if squeezes[i]:
                    signal['description'] = 'Squeeze Breakout (Buy)'
            
            # Strong BUY: Price breaks above middle band from below
            elif prev_price < middle_band[i - 1] and current_price >= middle_band[i]:
                signal['signal'] = 'BUY'
                signal['description'] = 'Middle Band Breakout'
            
            # SELL signals
            elif prev_price >= upper_band[i - 1] and current_price < upper_band[i]:
                signal['signal'] = 'SELL'
                signal['description'] = 'Overbought Reversal'
            
            # Strong SELL: Price breaks below middle band from above
            elif prev_price > middle_band[i - 1] and current_price <= middle_band[i]:
                signal['signal'] = 'SELL'
                signal['description'] = 'Middle Band Breakdown'
            
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
        
        # Generate signals
        signals = self.generate_signals(closes, upper_band, middle_band, lower_band)
        
        # Get latest signal for analysis
        if not signals:
            return {
                'error': "No signals generated",
                'success': False
            }
            
        latest_signal = signals[-1]
        
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
            'band_position': ((latest_signal['price'] - latest_signal['lower_band']) / 
                            (latest_signal['upper_band'] - latest_signal['lower_band']) * 100) 
                            if latest_signal['upper_band'] != latest_signal['lower_band'] else 50,
            
            # Trading signals
            'signal': latest_signal['signal'],
            'description': latest_signal['description'],
            'squeeze': latest_signal['squeeze'],
            
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
        """Calculate confidence score based on multiple factors."""
        confidence = 50  # Base confidence
        
        # Signal strength
        if signal['signal'] == 'BUY' or signal['signal'] == 'SELL':
            confidence += 20
        
        # Band position
        if position_pct > 80 or position_pct < 20:
            confidence += 15  # Strong position
        elif position_pct > 60 or position_pct < 40:
            confidence += 10  # Moderate position
        
        # Volume confirmation
        if volume_ratio > 1.5:
            confidence += 15
        elif volume_ratio > 1.2:
            confidence += 10
        
        # Squeeze condition
        if signal['squeeze']:
            confidence += 10
        
        # Volatility consideration
        if 2 <= volatility <= 8:
            confidence += 5  # Optimal volatility
        elif volatility > 12:
            confidence -= 10  # Too volatile
        
        return min(95, max(20, confidence))
    
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


