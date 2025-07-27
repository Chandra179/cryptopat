#!/usr/bin/env python3

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class RSIAnalyzer:
    """RSI (Relative Strength Index) trend analysis."""
    
    def __init__(self, period: int = 14):
        """
        Initialize RSI analyzer.
        
        Args:
            period: RSI calculation period (default: 14)
        """
        self.period = period
        self.overbought_level = 70
        self.oversold_level = 30
    
    def calculate_rsi(self, prices: List[float]) -> np.ndarray:
        """
        Calculate RSI (Relative Strength Index).
        
        Args:
            prices: List of closing prices
            
        Returns:
            Array of RSI values
        """
        if len(prices) < self.period + 1:
            logger.warning(f"Insufficient data for RSI calculation. Need at least {self.period + 1} points")
            return np.array([])
        
        prices_array = np.array(prices)
        deltas = np.diff(prices_array)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial average gain and loss
        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])
        
        # Initialize RSI array
        rsi = np.zeros(len(prices))
        
        # Calculate RSI for each period
        for i in range(self.period, len(prices)):
            if i == self.period:
                # First RSI calculation
                if avg_loss == 0:
                    rsi[i] = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi[i] = 100 - (100 / (1 + rs))
            else:
                # Smoothed RSI calculation (Wilder's smoothing)
                current_gain = gains[i-1] if i-1 < len(gains) else 0
                current_loss = losses[i-1] if i-1 < len(losses) else 0
                
                avg_gain = ((avg_gain * (self.period - 1)) + current_gain) / self.period
                avg_loss = ((avg_loss * (self.period - 1)) + current_loss) / self.period
                
                if avg_loss == 0:
                    rsi[i] = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi[self.period:]
    
    def detect_signals(self, rsi_values: np.ndarray) -> Dict[str, List[int]]:
        """
        Detect RSI buy/sell signals based on overbought/oversold levels.
        
        Args:
            rsi_values: Array of RSI values
            
        Returns:
            Dictionary with 'buy_signals' and 'sell_signals' indices
        """
        buy_signals = []
        sell_signals = []
        
        for i in range(1, len(rsi_values)):
            # RSI crosses above oversold level (potential buy)
            if rsi_values[i-1] <= self.oversold_level and rsi_values[i] > self.oversold_level:
                buy_signals.append(i)
            
            # RSI crosses below overbought level (potential sell)
            elif rsi_values[i-1] >= self.overbought_level and rsi_values[i] < self.overbought_level:
                sell_signals.append(i)
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        }
    
    def detect_divergence(self, prices: List[float], rsi_values: np.ndarray, lookback: int = 5) -> Dict[str, bool]:
        """
        Detect bullish and bearish divergence between price and RSI.
        
        Args:
            prices: List of closing prices
            rsi_values: Array of RSI values
            lookback: Number of periods to look back for divergence
            
        Returns:
            Dictionary indicating presence of bullish or bearish divergence
        """
        if len(prices) < lookback * 2 or len(rsi_values) < lookback * 2:
            return {'bullish_divergence': False, 'bearish_divergence': False}
        
        # Get recent price and RSI data
        recent_prices = prices[-lookback*2:]
        recent_rsi = rsi_values[-lookback*2:]
        
        # Find local highs and lows
        price_high_1 = max(recent_prices[:lookback])
        price_high_2 = max(recent_prices[lookback:])
        price_low_1 = min(recent_prices[:lookback])
        price_low_2 = min(recent_prices[lookback:])
        
        rsi_high_1 = max(recent_rsi[:lookback])
        rsi_high_2 = max(recent_rsi[lookback:])
        rsi_low_1 = min(recent_rsi[:lookback])
        rsi_low_2 = min(recent_rsi[lookback:])
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        bullish_divergence = (price_low_2 < price_low_1) and (rsi_low_2 > rsi_low_1)
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        bearish_divergence = (price_high_2 > price_high_1) and (rsi_high_2 < rsi_high_1)
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }
    
    def get_trend_strength(self, rsi_values: np.ndarray) -> Dict[str, float]:
        """
        Calculate trend strength based on RSI.
        
        Args:
            rsi_values: Array of RSI values
            
        Returns:
            Dictionary with bullish and bearish confidence percentages
        """
        if len(rsi_values) == 0:
            return {'bullish_confidence': 0.0, 'bearish_confidence': 0.0}
        
        current_rsi = rsi_values[-1]
        
        # Recent trend (last 5 periods or available data)
        recent_periods = min(5, len(rsi_values))
        recent_rsi = rsi_values[-recent_periods:]
        
        # RSI trend direction
        rsi_trend_up = len(recent_rsi) > 1 and recent_rsi[-1] > recent_rsi[-2]
        
        # Calculate confidence based on RSI level and trend
        if current_rsi >= self.overbought_level:
            # Overbought - bearish bias
            bearish_confidence = 70 + min(30, (current_rsi - self.overbought_level) * 2)
            bullish_confidence = 100 - bearish_confidence
        elif current_rsi <= self.oversold_level:
            # Oversold - bullish bias
            bullish_confidence = 70 + min(30, (self.oversold_level - current_rsi) * 2)
            bearish_confidence = 100 - bullish_confidence
        else:
            # Neutral zone - use RSI value and trend
            if current_rsi > 50:
                base_bullish = 50 + ((current_rsi - 50) / 20) * 30  # Scale 50-70 to 50-80
            else:
                base_bullish = 50 - ((50 - current_rsi) / 20) * 30  # Scale 30-50 to 20-50
            
            # Adjust for trend
            if rsi_trend_up:
                bullish_confidence = min(85, base_bullish + 10)
            else:
                bullish_confidence = max(15, base_bullish - 10)
            
            bearish_confidence = 100 - bullish_confidence
        
        return {
            'bullish_confidence': round(bullish_confidence, 2),
            'bearish_confidence': round(bearish_confidence, 2)
        }
    
    def analyze_ohlcv_data(self, ohlcv_data: List[List]) -> Dict:
        """
        Analyze OHLCV data using RSI.
        
        Args:
            ohlcv_data: List of [timestamp, open, high, low, close, volume]
            
        Returns:
            Analysis results with RSI data and trend assessment
        """
        if not ohlcv_data:
            logger.error("No OHLCV data provided for RSI analysis")
            return {}
        
        try:
            # Extract closing prices
            close_prices = [float(candle[4]) for candle in ohlcv_data]
            
            # Calculate RSI
            rsi_values = self.calculate_rsi(close_prices)
            
            if len(rsi_values) == 0:
                return {
                    'method': 'RSI',
                    'error': 'Insufficient data for analysis',
                    'bullish_confidence': 0.0,
                    'bearish_confidence': 0.0
                }
            
            # Detect signals
            signals = self.detect_signals(rsi_values)
            
            # Detect divergence
            divergence = self.detect_divergence(close_prices, rsi_values)
            
            # Get trend strength
            trend_strength = self.get_trend_strength(rsi_values)
            
            # Current RSI status
            current_rsi = float(rsi_values[-1])
            
            # Determine market condition
            if current_rsi >= self.overbought_level:
                market_condition = 'overbought'
            elif current_rsi <= self.oversold_level:
                market_condition = 'oversold'
            else:
                market_condition = 'neutral'
            
            return {
                'method': 'RSI',
                'current_rsi': current_rsi,
                'market_condition': market_condition,
                'overbought_level': self.overbought_level,
                'oversold_level': self.oversold_level,
                'bullish_divergence': divergence['bullish_divergence'],
                'bearish_divergence': divergence['bearish_divergence'],
                'recent_buy_signals': len([s for s in signals['buy_signals'] if s >= len(rsi_values) - 10]),
                'recent_sell_signals': len([s for s in signals['sell_signals'] if s >= len(rsi_values) - 10]),
                'bullish_confidence': trend_strength['bullish_confidence'],
                'bearish_confidence': trend_strength['bearish_confidence'],
                'data_points': len(close_prices)
            }
            
        except Exception as e:
            logger.error(f"Error in RSI analysis: {e}")
            return {
                'method': 'RSI',
                'error': str(e),
                'bullish_confidence': 0.0,
                'bearish_confidence': 0.0
            }