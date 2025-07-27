#!/usr/bin/env python3

import logging
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

from data.collector import DataCollector
from config.settings import calculate_data_limit

logger = logging.getLogger(__name__)

class ExponentialMovingAverageAnalyzer:
    """Exponential moving average based trend analysis for cryptocurrency prediction."""
    
    def __init__(self, collector: DataCollector):
        """
        Initialize the analyzer.
        
        Args:
            collector: DataCollector instance for fetching market data
        """
        self.collector = collector
    
    def predict_ema_trend(self, symbol: str, predict_days: int, analysis_days: int, timeframe: str = '1d') -> Dict:
        """
        Predict price trend for a cryptocurrency using exponential moving average analysis.
        
        Args:
            symbol: Trading pair symbol
            predict_days: Number of days to predict ahead
            analysis_days: Number of historical days to analyze
            timeframe: Data timeframe for analysis
            
        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Analyzing {symbol} with EMA - predicting {predict_days} days using {analysis_days} days of data")
        
        # Fetch historical data
        since_timestamp = int((datetime.now() - timedelta(days=analysis_days)).timestamp() * 1000)
        data_limit = calculate_data_limit(analysis_days, timeframe)
        ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit=data_limit, since=since_timestamp)
        
        if not ohlcv_data:
            logger.error(f"No data retrieved for {symbol}")
            return {'error': 'No data available'}
        
        # Validate data
        is_valid, message = self.collector.validate_data(ohlcv_data)
        if not is_valid:
            logger.error(f"Data validation failed for {symbol}: {message}")
            return {'error': f'Data validation failed: {message}'}
        
        # Get current ticker for reference
        ticker = self.collector.fetch_ticker(symbol)
        current_price = ticker.get('last', 0) if ticker else 0
        
        # Calculate EMA trend analysis
        bullish_confidence, bearish_confidence = self._calculate_ema_trend_confidence(ohlcv_data)
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'prediction_days': predict_days,
            'analysis_days': analysis_days,
            'data_points': len(ohlcv_data),
            'bullish_confidence': round(bullish_confidence, 2),
            'bearish_confidence': round(bearish_confidence, 2),
            'trend': 'bullish' if bullish_confidence > bearish_confidence else 'bearish'
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
    
    def _calculate_ema_trend_confidence(self, ohlcv_data: List[List]) -> Tuple[float, float]:
        """
        Calculate bullish and bearish confidence based on EMA crossovers and trends.
        
        Args:
            ohlcv_data: List of OHLCV candlestick data
            
        Returns:
            Tuple of (bullish_confidence, bearish_confidence) percentages
        """
        # Extract closing prices
        closes = [candle[4] for candle in ohlcv_data]
        
        if len(closes) < 26:  # Need enough data for both EMAs
            return 50.0, 50.0
        
        # Calculate short and long EMAs
        ema_12 = self.calculate_ema(closes, 12)
        ema_26 = self.calculate_ema(closes, 26)
        
        if len(ema_12) < 5 or len(ema_26) < 5:
            return 50.0, 50.0
        
        # Analyze EMA signals
        signals = self._analyze_ema_signals(ema_12, ema_26)
        
        # Calculate confidence based on multiple factors
        bullish_score = 0
        bearish_score = 0
        
        # Factor 1: Current EMA position (40% weight)
        if ema_12[-1] > ema_26[-1]:
            bullish_score += 40
        else:
            bearish_score += 40
        
        # Factor 2: EMA trend direction (30% weight)
        if len(ema_12) >= 3:
            ema_12_trend = (ema_12[-1] - ema_12[-3]) / ema_12[-3] * 100
            if ema_12_trend > 0:
                bullish_score += 30
            else:
                bearish_score += 30
        
        # Factor 3: Recent crossovers (30% weight)
        recent_crossovers = signals['crossovers'][-3:] if len(signals['crossovers']) >= 3 else signals['crossovers']
        if recent_crossovers:
            bullish_crossovers = sum(1 for signal in recent_crossovers if signal == 'bullish')
            bearish_crossovers = sum(1 for signal in recent_crossovers if signal == 'bearish')
            
            if bullish_crossovers > bearish_crossovers:
                bullish_score += 30
            elif bearish_crossovers > bullish_crossovers:
                bearish_score += 30
            else:
                bullish_score += 15
                bearish_score += 15
        
        # Normalize scores
        total_score = bullish_score + bearish_score
        if total_score > 0:
            bullish_confidence = (bullish_score / total_score) * 100
            bearish_confidence = (bearish_score / total_score) * 100
        else:
            bullish_confidence = bearish_confidence = 50
        
        return bullish_confidence, bearish_confidence
    
    def _analyze_ema_signals(self, ema_12: List[float], ema_26: List[float]) -> Dict:
        """
        Analyze EMA signals for crossovers and trends.
        
        Args:
            ema_12: 12-period EMA values
            ema_26: 26-period EMA values
            
        Returns:
            Dictionary with signal analysis
        """
        crossovers = []
        
        # Find crossovers
        for i in range(1, min(len(ema_12), len(ema_26))):
            prev_diff = ema_12[i-1] - ema_26[i-1]
            curr_diff = ema_12[i] - ema_26[i]
            
            # Bullish crossover: EMA 12 crosses above EMA 26
            if prev_diff <= 0 and curr_diff > 0:
                crossovers.append('bullish')
            # Bearish crossover: EMA 12 crosses below EMA 26
            elif prev_diff >= 0 and curr_diff < 0:
                crossovers.append('bearish')
        
        return {
            'crossovers': crossovers,
            'current_position': 'above' if ema_12[-1] > ema_26[-1] else 'below',
            'ema_12_trend': 'up' if ema_12[-1] > ema_12[0] else 'down',
            'ema_26_trend': 'up' if ema_26[-1] > ema_26[0] else 'down'
        }