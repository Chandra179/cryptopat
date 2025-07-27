#!/usr/bin/env python3

import logging
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

from data.collector import DataCollector
from config.settings import calculate_data_limit

logger = logging.getLogger(__name__)

class SimpleMovingAverageAnalyzer:
    """Simple moving average based trend analysis for cryptocurrency prediction."""
    
    def __init__(self, collector: DataCollector):
        """
        Initialize the analyzer.
        
        Args:
            collector: DataCollector instance for fetching market data
        """
        self.collector = collector
    
    def predict_simple_moving_avg_trend(self, symbol: str, predict_days: int, analysis_days: int, timeframe: str = '1d') -> Dict:
        """
        Predict price trend for a cryptocurrency using simple moving average analysis.
        
        Args:
            symbol: Trading pair symbol
            predict_days: Number of days to predict ahead
            analysis_days: Number of historical days to analyze
            timeframe: Data timeframe for analysis
            
        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Analyzing {symbol} - predicting {predict_days} days using {analysis_days} days of data")
        
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
        
        # Calculate trend analysis
        bullish_confidence, bearish_confidence = self._calculate_simple_moving_avg_trend_confidence(ohlcv_data)
        
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
    
    def _calculate_simple_moving_avg_trend_confidence(self, ohlcv_data: List[List]) -> Tuple[float, float]:
        """
        Calculate bullish and bearish confidence based on recent price changes.
        
        Args:
            ohlcv_data: List of OHLCV candlestick data
            
        Returns:
            Tuple of (bullish_confidence, bearish_confidence) percentages
        """
        # Basic trend analysis (Phase 1 - simple implementation)
        # Calculate recent price changes
        recent_closes = [candle[4] for candle in ohlcv_data[-7:]]  # Last 7 closes
        price_changes = [recent_closes[i] - recent_closes[i-1] for i in range(1, len(recent_closes))]
        
        # Simple bullish/bearish confidence based on recent trend
        positive_changes = sum(1 for change in price_changes if change > 0)
        total_changes = len(price_changes)
        
        if total_changes > 0:
            bullish_confidence = (positive_changes / total_changes) * 100
            bearish_confidence = 100 - bullish_confidence
        else:
            bullish_confidence = bearish_confidence = 50
            
        return bullish_confidence, bearish_confidence