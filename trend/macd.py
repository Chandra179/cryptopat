#!/usr/bin/env python3

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MACDAnalyzer:
    """MACD (Moving Average Convergence Divergence) trend analysis."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize MACD analyzer with customizable periods.
        
        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate_ema(self, prices: List[float], period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        prices_array = np.array(prices)
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices_array)
        ema[0] = prices_array[0]
        
        for i in range(1, len(prices_array)):
            ema[i] = alpha * prices_array[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def calculate_macd(self, prices: List[float]) -> Dict[str, np.ndarray]:
        """
        Calculate MACD line, signal line, and histogram.
        
        Args:
            prices: List of closing prices
            
        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' arrays
        """
        if len(prices) < max(self.slow_period, self.signal_period):
            logger.warning(f"Insufficient data for MACD calculation. Need at least {max(self.slow_period, self.signal_period)} points")
            return {'macd': np.array([]), 'signal': np.array([]), 'histogram': np.array([])}
        
        fast_ema = self.calculate_ema(prices, self.fast_period)
        slow_ema = self.calculate_ema(prices, self.slow_period)
        
        macd_line = fast_ema - slow_ema
        signal_line = self.calculate_ema(macd_line.tolist(), self.signal_period)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def detect_signals(self, macd_data: Dict[str, np.ndarray]) -> Dict[str, List[int]]:
        """
        Detect MACD buy/sell signals.
        
        Args:
            macd_data: MACD calculation results
            
        Returns:
            Dictionary with 'buy_signals' and 'sell_signals' indices
        """
        macd = macd_data['macd']
        signal = macd_data['signal']
        histogram = macd_data['histogram']
        
        buy_signals = []
        sell_signals = []
        
        for i in range(1, len(macd)):
            # MACD line crosses above signal line (bullish)
            if macd[i-1] <= signal[i-1] and macd[i] > signal[i]:
                buy_signals.append(i)
            
            # MACD line crosses below signal line (bearish)
            elif macd[i-1] >= signal[i-1] and macd[i] < signal[i]:
                sell_signals.append(i)
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals
        }
    
    def get_trend_strength(self, macd_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate trend strength based on MACD indicators.
        
        Args:
            macd_data: MACD calculation results
            
        Returns:
            Dictionary with bullish and bearish confidence percentages
        """
        if len(macd_data['macd']) == 0:
            return {'bullish_confidence': 0.0, 'bearish_confidence': 0.0}
        
        macd = macd_data['macd']
        signal = macd_data['signal']
        histogram = macd_data['histogram']
        
        # Recent values (last 5 periods or available data)
        recent_periods = min(5, len(macd))
        recent_macd = macd[-recent_periods:]
        recent_signal = signal[-recent_periods:]
        recent_histogram = histogram[-recent_periods:]
        
        # Calculate trend indicators
        macd_above_signal = np.mean(recent_macd > recent_signal)
        histogram_positive = np.mean(recent_histogram > 0)
        macd_trend = 1 if len(recent_macd) > 1 and recent_macd[-1] > recent_macd[-2] else 0
        
        # Calculate confidence scores
        bullish_score = (macd_above_signal * 0.4 + histogram_positive * 0.4 + macd_trend * 0.2) * 100
        bearish_score = 100 - bullish_score
        
        return {
            'bullish_confidence': round(bullish_score, 2),
            'bearish_confidence': round(bearish_score, 2)
        }
    
    def analyze_ohlcv_data(self, ohlcv_data: List[List]) -> Dict:
        """
        Analyze OHLCV data using MACD.
        
        Args:
            ohlcv_data: List of [timestamp, open, high, low, close, volume]
            
        Returns:
            Analysis results with MACD data and trend assessment
        """
        if not ohlcv_data:
            logger.error("No OHLCV data provided for MACD analysis")
            return {}
        
        try:
            # Extract closing prices
            close_prices = [float(candle[4]) for candle in ohlcv_data]
            
            # Calculate MACD
            macd_data = self.calculate_macd(close_prices)
            
            if len(macd_data['macd']) == 0:
                return {
                    'method': 'MACD',
                    'error': 'Insufficient data for analysis',
                    'bullish_confidence': 0.0,
                    'bearish_confidence': 0.0
                }
            
            # Detect signals
            signals = self.detect_signals(macd_data)
            
            # Get trend strength
            trend_strength = self.get_trend_strength(macd_data)
            
            # Current MACD status
            current_macd = float(macd_data['macd'][-1])
            current_signal = float(macd_data['signal'][-1])
            current_histogram = float(macd_data['histogram'][-1])
            
            return {
                'method': 'MACD',
                'current_macd': current_macd,
                'current_signal': current_signal,
                'current_histogram': current_histogram,
                'trend_direction': 'bullish' if current_macd > current_signal else 'bearish',
                'recent_buy_signals': len([s for s in signals['buy_signals'] if s >= len(macd_data['macd']) - 10]),
                'recent_sell_signals': len([s for s in signals['sell_signals'] if s >= len(macd_data['macd']) - 10]),
                'bullish_confidence': trend_strength['bullish_confidence'],
                'bearish_confidence': trend_strength['bearish_confidence'],
                'data_points': len(close_prices)
            }
            
        except Exception as e:
            logger.error(f"Error in MACD analysis: {e}")
            return {
                'method': 'MACD',
                'error': str(e),
                'bullish_confidence': 0.0,
                'bearish_confidence': 0.0
            }