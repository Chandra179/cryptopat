#!/usr/bin/env python3

import asyncio
import logging
from typing import List, Dict
from data.collector import DataCollector
from trend.simple_moving_average import SimpleMovingAverageAnalyzer
from trend.exponential_moving_average import ExponentialMovingAverageAnalyzer
from trend.ema_golden_death_crossover import EMACrossoverAnalyzer
from trend.macd import MACDAnalyzer
from trend.rsi import RSIAnalyzer
from ui.display import display_results
from config.settings import DEFAULT_SYMBOLS, calculate_data_limit

logger = logging.getLogger(__name__)

class CryptoTrendApp:
    """Main application class for cryptocurrency trend prediction."""
    
    def __init__(self, exchange: str = 'binance'):
        """
        Initialize the application.
        
        Args:
            exchange: Exchange name to use for data collection
        """
        self.collector = DataCollector(exchange)
        self.sma_analyzer = SimpleMovingAverageAnalyzer(self.collector)
        self.ema_analyzer = ExponentialMovingAverageAnalyzer(self.collector)
        self.ema_crossover_analyzer = EMACrossoverAnalyzer(self.collector)
        self.macd_analyzer = MACDAnalyzer()
        self.rsi_analyzer = RSIAnalyzer()
        self.symbols = DEFAULT_SYMBOLS
    
    def predict_trend(self, symbol: str, predict_days: int, analysis_days: int, timeframe: str = '1d', methods: List[str] = ['sma', 'ema']):
        """
        Predict price trend for a cryptocurrency.
        
        Args:
            symbol: Trading pair symbol
            predict_days: Number of days to predict ahead
            analysis_days: Number of historical days to analyze
            timeframe: Data timeframe for analysis
            methods: List of analysis methods (['sma'], ['ema'], ['ema_cross'], ['macd'], ['rsi'], or combinations)
            
        Returns:
            Dictionary with prediction results
        """
        results = {}
        
        # Execute requested analysis methods
        if 'sma' in methods:
            results['sma'] = self.sma_analyzer.predict_simple_moving_avg_trend(symbol, predict_days, analysis_days, timeframe)
        if 'ema' in methods:
            results['ema'] = self.ema_analyzer.predict_ema_trend(symbol, predict_days, analysis_days, timeframe)
        if 'ema_cross' in methods:
            results['ema_cross'] = self.ema_crossover_analyzer.predict_ema_golden_death_crossover_trend(symbol, predict_days, analysis_days, timeframe)
        
        # For MACD and RSI, fetch data once and reuse
        if 'macd' in methods or 'rsi' in methods:
            from datetime import datetime, timedelta
            since_timestamp = int((datetime.now() - timedelta(days=analysis_days)).timestamp() * 1000)
            data_limit = calculate_data_limit(analysis_days, timeframe)
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit=data_limit, since=since_timestamp)
            
            if ohlcv_data:
                if 'macd' in methods:
                    macd_result = self.macd_analyzer.analyze_ohlcv_data(ohlcv_data)
                    if 'error' not in macd_result:
                        # Add standard fields for consistency
                        macd_result.update({
                            'symbol': symbol,
                            'prediction_days': predict_days,
                            'analysis_days': analysis_days,
                            'current_price': float(ohlcv_data[-1][4]) if ohlcv_data else 0,
                            'trend': 'bullish' if macd_result.get('bullish_confidence', 0) > macd_result.get('bearish_confidence', 0) else 'bearish'
                        })
                    results['macd'] = macd_result
                
                if 'rsi' in methods:
                    rsi_result = self.rsi_analyzer.analyze_ohlcv_data(ohlcv_data)
                    if 'error' not in rsi_result:
                        # Add standard fields for consistency
                        rsi_result.update({
                            'symbol': symbol,
                            'prediction_days': predict_days,
                            'analysis_days': analysis_days,
                            'current_price': float(ohlcv_data[-1][4]) if ohlcv_data else 0,
                            'trend': 'bullish' if rsi_result.get('bullish_confidence', 0) > rsi_result.get('bearish_confidence', 0) else 'bearish'
                        })
                    results['rsi'] = rsi_result
            else:
                if 'macd' in methods:
                    results['macd'] = {'error': 'No data available for MACD analysis'}
                if 'rsi' in methods:
                    results['rsi'] = {'error': 'No data available for RSI analysis'}
        
        if not results:
            raise ValueError(f"No valid analysis methods provided: {methods}")
        
        # If only one method, return its result directly with method info
        if len(results) == 1:
            method_name = list(results.keys())[0]
            result = list(results.values())[0]
            result['methods'] = [method_name]
            return result
        
        # Combine multiple method results
        first_result = list(results.values())[0]
        combined_result = {
            'symbol': symbol,
            'current_price': first_result.get('current_price', 0),
            'prediction_days': predict_days,
            'analysis_days': analysis_days,
            'data_points': first_result.get('data_points', 0),
            'methods': methods
        }
        
        # Add individual method results
        for method_name, result in results.items():
            combined_result[f'{method_name}_analysis'] = {
                'bullish_confidence': result.get('bullish_confidence', 0),
                'bearish_confidence': result.get('bearish_confidence', 0),
                'trend': result.get('trend', 'neutral')
            }
        
        # Calculate combined confidence (average of all methods)
        total_bullish = sum(result.get('bullish_confidence', 0) for result in results.values())
        total_bearish = sum(result.get('bearish_confidence', 0) for result in results.values())
        combined_bullish = total_bullish / len(results)
        combined_bearish = total_bearish / len(results)
        
        combined_result.update({
            'bullish_confidence': round(combined_bullish, 2),
            'bearish_confidence': round(combined_bearish, 2),
            'trend': 'bullish' if combined_bullish > combined_bearish else 'bearish'
        })
        
        return combined_result
    
    async def run_analysis(self, predict_days: int, analysis_days: int, symbols: List[str] = None, methods: List[str] = ['sma', 'ema'], timeframe: str = '1d'):
        """
        Run trend analysis for specified symbols.
        
        Args:
            predict_days: Days to predict ahead
            analysis_days: Days of historical data to analyze
            symbols: List of symbols to analyze (uses default if None)
            methods: List of analysis methods to use (['sma'], ['ema'], ['ema_cross'], ['macd'], ['rsi'], or combinations)
            timeframe: Data timeframe for analysis (default: '1d')
        """
        if symbols is None:
            symbols = self.symbols
        
        logger.info(f"Starting analysis for {len(symbols)} symbols")
        logger.info(f"Prediction window: {predict_days} days")
        logger.info(f"Analysis window: {analysis_days} days")
        logger.info(f"Analysis methods: {', '.join(m.upper() for m in methods)}")
        
        async def analyze_symbol(symbol):
            try:
                result = self.predict_trend(symbol, predict_days, analysis_days, timeframe, methods=methods)
                # Add delay to respect rate limits
                await asyncio.sleep(1.2)
                return result
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                return {'symbol': symbol, 'error': str(e)}
        
        # Run all analyses concurrently
        tasks = [analyze_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        # Display results
        display_results(results, methods)
        return results