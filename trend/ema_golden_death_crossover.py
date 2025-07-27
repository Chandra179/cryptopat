#!/usr/bin/env python3

import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

from data.collector import DataCollector
from config.settings import calculate_data_limit

logger = logging.getLogger(__name__)

class EMACrossoverAnalyzer:
    """EMA Golden Cross and Death Cross pattern analyzer for cryptocurrency prediction."""
    
    def __init__(self, collector: DataCollector, short_period: int = 50, long_period: int = 200):
        """
        Initialize the analyzer with configurable EMA periods.
        
        Args:
            collector: DataCollector instance for fetching market data
            short_period: Short-term EMA period (default: 50)
            long_period: Long-term EMA period (default: 200)
        """
        self.collector = collector
        self.short_period = short_period
        self.long_period = long_period
    
    def predict_ema_golden_death_crossover_trend(self, symbol: str, predict_days: int, analysis_days: int, timeframe: str = '1d') -> Dict:
        """
        Predict price trend based on configurable EMA golden cross and death cross patterns.
        
        Args:
            symbol: Trading pair symbol
            predict_days: Number of days to predict ahead
            analysis_days: Number of historical days to analyze
            timeframe: Data timeframe for analysis
            
        Returns:
            Dictionary with prediction results including crossover patterns
        """
        logger.info(f"Analyzing {symbol} with {self.short_period}/{self.long_period} EMA crossovers - predicting {predict_days} days using {analysis_days} days of data")
        
        # Ensure we have enough data for long EMA calculation
        min_days = max(analysis_days, self.long_period + 20)  # Extra buffer for accurate long EMA
        
        # Fetch historical data
        since_timestamp = int((datetime.now() - timedelta(days=min_days)).timestamp() * 1000)
        data_limit = calculate_data_limit(min_days, timeframe)
        ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit=data_limit, since=since_timestamp)
        
        if not ohlcv_data:
            logger.error(f"No data retrieved for {symbol}")
            return {'error': 'No data available'}
        
        if len(ohlcv_data) < self.long_period:
            logger.error(f"Insufficient data for {self.long_period} EMA calculation. Got {len(ohlcv_data)} data points, need at least {self.long_period}")
            return {'error': f'Insufficient data: need at least {self.long_period} data points, got {len(ohlcv_data)}'}
        
        # Validate data
        is_valid, message = self.collector.validate_data(ohlcv_data)
        if not is_valid:
            logger.error(f"Data validation failed for {symbol}: {message}")
            return {'error': f'Data validation failed: {message}'}
        
        # Get current ticker for reference
        ticker = self.collector.fetch_ticker(symbol)
        current_price = ticker.get('last', 0) if ticker else 0
        
        # Calculate crossover analysis
        crossover_analysis = self._calculate_crossover_analysis(ohlcv_data)
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'prediction_days': predict_days,
            'analysis_days': analysis_days,
            'data_points': len(ohlcv_data),
            'pattern': crossover_analysis['pattern'],
            'crossover_strength': crossover_analysis['strength'],
            'days_since_crossover': crossover_analysis['days_since_crossover'],
            f'ema_{self.short_period}': crossover_analysis['ema_short_current'],
            f'ema_{self.long_period}': crossover_analysis['ema_long_current'],
            'bullish_confidence': round(crossover_analysis['bullish_confidence'], 2),
            'bearish_confidence': round(crossover_analysis['bearish_confidence'], 2),
            'trend': 'bullish' if crossover_analysis['bullish_confidence'] > crossover_analysis['bearish_confidence'] else 'bearish'
        }
    
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate Exponential Moving Average for given prices.
        
        Args:
            prices: List of price values
            period: Period for EMA calculation
            
        Returns:
            List of EMA values
        """
        if len(prices) < period:
            return []
        
        # Calculate smoothing factor (alpha)
        alpha = 2 / (period + 1)
        
        ema_values = []
        
        # Initialize with simple moving average for first value
        sma = sum(prices[:period]) / period
        ema_values.append(sma)
        
        # Calculate EMA for remaining values
        for i in range(period, len(prices)):
            ema = alpha * prices[i] + (1 - alpha) * ema_values[-1]
            ema_values.append(ema)
        
        return ema_values
    
    def detect_golden_cross(self, ema_50: List[float], ema_200: List[float]) -> Tuple[bool, Optional[int]]:
        """
        Detect golden cross pattern (50 EMA crossing above 200 EMA).
        
        Args:
            ema_50: 50-period EMA values
            ema_200: 200-period EMA values
            
        Returns:
            Tuple of (is_golden_cross, days_ago) where days_ago is None if no recent cross
        """
        if len(ema_50) < 2 or len(ema_200) < 2:
            return False, None
        
        # Check for golden cross in recent periods (last 10 days)
        min_len = min(len(ema_50), len(ema_200))
        lookback = min(10, min_len - 1)
        
        for i in range(1, lookback + 1):
            idx = -i
            prev_idx = idx - 1
            
            # Golden cross: 50 EMA crosses above 200 EMA
            if (ema_50[prev_idx] <= ema_200[prev_idx] and 
                ema_50[idx] > ema_200[idx]):
                return True, i - 1
        
        return False, None
    
    def detect_death_cross(self, ema_50: List[float], ema_200: List[float]) -> Tuple[bool, Optional[int]]:
        """
        Detect death cross pattern (50 EMA crossing below 200 EMA).
        
        Args:
            ema_50: 50-period EMA values
            ema_200: 200-period EMA values
            
        Returns:
            Tuple of (is_death_cross, days_ago) where days_ago is None if no recent cross
        """
        if len(ema_50) < 2 or len(ema_200) < 2:
            return False, None
        
        # Check for death cross in recent periods (last 10 days)
        min_len = min(len(ema_50), len(ema_200))
        lookback = min(10, min_len - 1)
        
        for i in range(1, lookback + 1):
            idx = -i
            prev_idx = idx - 1
            
            # Death cross: 50 EMA crosses below 200 EMA
            if (ema_50[prev_idx] >= ema_200[prev_idx] and 
                ema_50[idx] < ema_200[idx]):
                return True, i - 1
        
        return False, None
    
    def _calculate_crossover_analysis(self, ohlcv_data: List[List]) -> Dict:
        """
        Calculate comprehensive crossover analysis including pattern detection and confidence scoring.
        
        Args:
            ohlcv_data: List of OHLCV candlestick data
            
        Returns:
            Dictionary with crossover analysis results
        """
        # Extract closing prices
        closes = [candle[4] for candle in ohlcv_data]
        
        # Calculate EMAs
        ema_short = self.calculate_ema(closes, self.short_period)
        ema_long = self.calculate_ema(closes, self.long_period)
        
        if len(ema_short) == 0 or len(ema_long) == 0:
            return {
                'pattern': 'insufficient_data',
                'strength': 0,
                'days_since_crossover': None,
                'ema_short_current': 0,
                'ema_long_current': 0,
                'bullish_confidence': 50,
                'bearish_confidence': 50
            }
        
        # Detect patterns
        is_golden_cross, golden_days_ago = self.detect_golden_cross(ema_short, ema_long)
        is_death_cross, death_days_ago = self.detect_death_cross(ema_short, ema_long)
        
        # Determine current pattern
        pattern = 'neutral'
        days_since_crossover = None
        crossover_strength = 0
        
        if is_golden_cross:
            pattern = 'golden_cross'
            days_since_crossover = golden_days_ago
        elif is_death_cross:
            pattern = 'death_cross'
            days_since_crossover = death_days_ago
        else:
            # Check current EMA relationship
            if ema_short[-1] > ema_long[-1]:
                pattern = 'bullish_alignment'
            elif ema_short[-1] < ema_long[-1]:
                pattern = 'bearish_alignment'
        
        # Calculate crossover strength based on EMA separation
        ema_separation = abs(ema_short[-1] - ema_long[-1]) / ema_long[-1] * 100
        crossover_strength = min(ema_separation * 10, 100)  # Scale to 0-100
        
        # Calculate confidence scores
        bullish_confidence, bearish_confidence = self._calculate_confidence_scores(
            ema_short, ema_long, pattern, days_since_crossover, crossover_strength
        )
        
        return {
            'pattern': pattern,
            'strength': round(crossover_strength, 2),
            'days_since_crossover': days_since_crossover,
            'ema_short_current': round(ema_short[-1], 2),
            'ema_long_current': round(ema_long[-1], 2),
            'bullish_confidence': bullish_confidence,
            'bearish_confidence': bearish_confidence
        }
    
    def _calculate_confidence_scores(self, ema_short: List[float], ema_long: List[float], 
                                   pattern: str, days_since_crossover: Optional[int], 
                                   strength: float) -> Tuple[float, float]:
        """
        Calculate bullish and bearish confidence scores based on crossover patterns.
        
        Args:
            ema_short: Short-period EMA values
            ema_long: Long-period EMA values
            pattern: Detected pattern type
            days_since_crossover: Days since last crossover (None if no recent crossover)
            strength: Crossover strength percentage
            
        Returns:
            Tuple of (bullish_confidence, bearish_confidence)
        """
        bullish_score = 0
        bearish_score = 0
        
        # Base scoring based on pattern (60% weight)
        if pattern == 'golden_cross':
            bullish_score += 85
            bearish_score += 15
        elif pattern == 'death_cross':
            bullish_score += 15
            bearish_score += 85
        elif pattern == 'bullish_alignment':
            bullish_score += 65
            bearish_score += 35
        elif pattern == 'bearish_alignment':
            bullish_score += 35
            bearish_score += 65
        else:  # neutral
            bullish_score += 50
            bearish_score += 50
        
        # Adjust based on crossover recency (20% weight)
        if days_since_crossover is not None:
            recency_factor = max(0, 1 - (days_since_crossover / 10))  # Decay over 10 days
            if pattern == 'golden_cross':
                bullish_score += 20 * recency_factor
                bearish_score -= 20 * recency_factor
            elif pattern == 'death_cross':
                bearish_score += 20 * recency_factor
                bullish_score -= 20 * recency_factor
        
        # Adjust based on EMA trends (20% weight)
        if len(ema_short) >= 5 and len(ema_long) >= 5:
            ema_short_trend = (ema_short[-1] - ema_short[-5]) / ema_short[-5]
            ema_long_trend = (ema_long[-1] - ema_long[-5]) / ema_long[-5]
            
            # Both EMAs trending up is bullish
            if ema_short_trend > 0 and ema_long_trend > 0:
                bullish_score += 20
            # Both EMAs trending down is bearish
            elif ema_short_trend < 0 and ema_long_trend < 0:
                bearish_score += 20
            # Mixed trends - moderate adjustment
            elif ema_short_trend > 0:
                bullish_score += 10
            elif ema_short_trend < 0:
                bearish_score += 10
        
        # Normalize to ensure scores are between 0-100 and sum to 100
        total_score = bullish_score + bearish_score
        if total_score > 0:
            bullish_confidence = max(0, min(100, (bullish_score / total_score) * 100))
            bearish_confidence = max(0, min(100, (bearish_score / total_score) * 100))
        else:
            bullish_confidence = bearish_confidence = 50
        
        return bullish_confidence, bearish_confidence