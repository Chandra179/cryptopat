"""
Absorption Detection Module for CryptoPat.

This module implements absorption detection using historical data
to identify when large volumes are absorbed with minimal price movement,
indicating potential support/resistance levels.
"""

import pandas as pd
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from data import get_data_collector

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AbsorptionEvent:
    """Data class representing an absorption event."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    price_level: float
    aggressive_volume: float
    price_delta: float
    confidence: int  # 0-100 scale
    signal_type: str  # 'bullish_absorption', 'bearish_absorption', 'hidden_absorption'

class AbsorptionStrategy:
    """Absorption pattern detection strategy for cryptocurrency order flow analysis."""
    
    def __init__(self):
        self.collector = get_data_collector()

    def detect_absorption_patterns(self, df: pd.DataFrame, volume_threshold: float = 5000, 
                                 price_threshold: float = 0.002) -> Optional[Dict]:
        """
        Detect absorption patterns in OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            volume_threshold: Minimum volume threshold for absorption detection
            price_threshold: Maximum price movement threshold (as percentage)
            
        Returns:
            Dictionary with absorption pattern details or None if not found
        """
        if len(df) < 20:
            return None
        
        # Look for absorption patterns in recent candles
        recent_df = df.tail(20)
        
        for i in range(5, len(recent_df)):
            current_candle = recent_df.iloc[i]
            
            # Check volume spike
            avg_volume = recent_df['volume'].iloc[:i].mean()
            if current_candle['volume'] < volume_threshold:
                continue
            if current_candle['volume'] < avg_volume * 1.5:  # Volume must be 1.5x average
                continue
            
            # Check price movement is minimal
            price_change = abs(current_candle['close'] - current_candle['open']) / current_candle['open']
            if price_change > price_threshold:
                continue
            
            # Determine absorption type
            body_size = abs(current_candle['close'] - current_candle['open'])
            upper_wick = current_candle['high'] - max(current_candle['open'], current_candle['close'])
            lower_wick = min(current_candle['open'], current_candle['close']) - current_candle['low']
            
            # Absorption characteristics
            if upper_wick > body_size * 2 and lower_wick < body_size:
                absorption_type = "Selling Absorption"
                signal_bias = "BULLISH"
                signal = "BUY"
            elif lower_wick > body_size * 2 and upper_wick < body_size:
                absorption_type = "Buying Absorption"
                signal_bias = "BEARISH"
                signal = "SELL"
            elif body_size < (current_candle['high'] - current_candle['low']) * 0.3:
                absorption_type = "Volume Absorption"
                signal_bias = "NEUTRAL"
                signal = "HOLD"
            else:
                continue
            
            # Calculate confidence based on multiple factors
            confidence = 50  # Base confidence
            
            # Volume factor
            volume_ratio = current_candle['volume'] / avg_volume
            if volume_ratio > 3:
                confidence += 20
            elif volume_ratio > 2:
                confidence += 10
            
            # Price action factor
            if price_change < price_threshold * 0.5:
                confidence += 15
            
            # Wick analysis factor
            total_range = current_candle['high'] - current_candle['low']
            if absorption_type == "Selling Absorption" and upper_wick > total_range * 0.6:
                confidence += 15
            elif absorption_type == "Buying Absorption" and lower_wick > total_range * 0.6:
                confidence += 15
            
            confidence = max(20, min(95, confidence))
            
            return {
                'pattern': absorption_type,
                'bias': signal_bias,
                'signal': signal,
                'confidence': confidence,
                'candle_index': i,
                'volume_ratio': round(volume_ratio, 2),
                'price_change': round(price_change * 100, 3),
                'body_size': round(body_size, 4),
                'upper_wick': round(upper_wick, 4),
                'lower_wick': round(lower_wick, 4),
                'absorption_level': round(current_candle['close'], 4)
            }
        
        return None

    def analyze(self, symbol: str, timeframe: str, limit: int) -> Dict:
        """
        Analyze absorption patterns for given symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for analysis
            limit: Number of candles to analyze
            
        Returns:
            Analysis results dictionary
        """
        try:
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
            
            if not ohlcv_data or len(ohlcv_data) < 20:
                return {
                    'error': f'Insufficient data: need at least 20 candles, got {len(ohlcv_data) if ohlcv_data else 0}',
                    'success': False,
                    'symbol': symbol,
                    'timeframe': timeframe
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Detect pattern
            pattern = self.detect_absorption_patterns(df)
            
            # Get current price and timestamp info
            current_price = df['close'].iloc[-1]
            current_timestamp = df['timestamp'].iloc[-1]
            dt = datetime.fromtimestamp(current_timestamp.timestamp(), tz=timezone.utc)
            
            result = {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'analysis_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': int(current_timestamp.timestamp() * 1000),
                'total_candles': len(df),
                'current_price': round(current_price, 4),
                'pattern_detected': pattern is not None
            }
            
            if pattern:
                # Calculate additional metrics based on pattern
                confidence = pattern.get('confidence', 50)
                signal = pattern.get('signal', 'HOLD')
                bias = pattern.get('bias', 'NEUTRAL')
                absorption_level = pattern.get('absorption_level', current_price)
                
                # Calculate support/resistance levels based on absorption
                if signal == 'BUY':
                    support_level = absorption_level * 0.99
                    resistance_level = current_price * 1.02
                    stop_zone = absorption_level * 0.985
                    tp_low = resistance_level
                    tp_high = resistance_level * 1.015
                elif signal == 'SELL':
                    support_level = current_price * 0.98
                    resistance_level = absorption_level * 1.01
                    stop_zone = absorption_level * 1.015
                    tp_low = support_level
                    tp_high = support_level * 0.985
                else:
                    support_level = current_price * 0.99
                    resistance_level = current_price * 1.01
                    stop_zone = current_price * 0.99
                    tp_low = current_price * 1.01
                    tp_high = current_price * 1.02
                
                # Calculate Risk/Reward ratio
                if signal in ['BUY', 'SELL']:
                    risk = abs(current_price - stop_zone)
                    reward = abs(tp_low - current_price) if tp_low != current_price else abs(tp_high - current_price)
                    rr_ratio = reward / risk if risk > 0 else 0
                else:
                    rr_ratio = 0
                
                # Determine entry window
                if signal in ['BUY', 'SELL'] and confidence > 70:
                    entry_window = "Optimal now"
                elif signal in ['BUY', 'SELL'] and confidence > 50:
                    entry_window = "Good in next 2-3 bars"
                else:
                    entry_window = "Wait for better setup"
                
                # Exit trigger
                if signal == 'BUY':
                    exit_trigger = "Price breaks below absorption level"
                elif signal == 'SELL':
                    exit_trigger = "Price breaks above absorption level"
                else:
                    exit_trigger = "Wait for clear directional move"
                
                # Update result with pattern analysis
                result.update({
                    # Pattern specific data
                    'pattern_type': pattern.get('pattern', 'Unknown'),
                    'bias': bias,
                    'absorption_level': round(absorption_level, 4),
                    
                    # Price levels
                    'support_level': round(support_level, 4),
                    'resistance_level': round(resistance_level, 4),
                    'stop_zone': round(stop_zone, 4),
                    'tp_low': round(tp_low, 4),
                    'tp_high': round(tp_high, 4),
                    
                    # Trading analysis
                    'signal': signal,
                    'confidence_score': confidence,
                    'entry_window': entry_window,
                    'exit_trigger': exit_trigger,
                    'rr_ratio': round(rr_ratio, 1),
                    
                    # Pattern details
                    'volume_ratio': pattern.get('volume_ratio'),
                    'price_change_pct': pattern.get('price_change'),
                    'body_size': pattern.get('body_size'),
                    'upper_wick': pattern.get('upper_wick'),
                    'lower_wick': pattern.get('lower_wick'),
                    
                    # Raw data
                    'raw_data': {
                        'ohlcv_data': ohlcv_data,
                        'pattern_data': pattern
                    }
                })
            else:
                # No pattern detected
                result.update({
                    'pattern_type': None,
                    'bias': 'NEUTRAL',
                    'signal': 'HOLD',
                    'confidence_score': 0,
                    'entry_window': "No absorption pattern detected",
                    'exit_trigger': "Wait for volume absorption",
                    'support_level': round(current_price * 0.99, 4),
                    'resistance_level': round(current_price * 1.01, 4),
                    'rr_ratio': 0
                })
                
            return result
            
        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'success': False,
                'symbol': symbol,
                'timeframe': timeframe
            }