"""
RSI(14) implementation for cryptocurrency trend analysis.
Uses Relative Strength Index to detect overbought/oversold conditions and trend momentum.
"""

from datetime import datetime, timezone
from typing import List, Dict
from data import get_data_collector


class RSI14Strategy:
    """RSI(14) strategy for momentum and reversal detection."""
    
    def __init__(self):
        self.collector = get_data_collector()
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: List of closing prices
            period: RSI period (default 14)
            
        Returns:
            List of RSI values
        """
        if len(prices) < period + 1:
            raise ValueError(f"Insufficient data for RSI calculation: need at least {period + 1} prices, got {len(prices)}")
        
        # Calculate price changes
        price_changes = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            price_changes.append(change)
        
        if len(price_changes) < period:
            raise ValueError(f"Insufficient price changes for RSI calculation: need at least {period}, got {len(price_changes)}")
        
        rsi_values = []
        
        # Calculate initial average gains and losses
        gains = [max(0, change) for change in price_changes[:period]]
        losses = [max(0, -change) for change in price_changes[:period]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        # Calculate first RSI value
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
        
        # Calculate subsequent RSI values using smoothed averages
        for i in range(period, len(price_changes)):
            gain = max(0, price_changes[i])
            loss = max(0, -price_changes[i])
            
            # Smoothed averages (Wilder's smoothing)
            avg_gain = ((avg_gain * (period - 1)) + gain) / period
            avg_loss = ((avg_loss * (period - 1)) + loss) / period
            
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)
        
        return rsi_values
    
    def detect_rsi_signals(self, rsi_values: List[float], closes: List[float]) -> List[dict]:
        """
        Detect RSI-based trading signals.
        
        Args:
            rsi_values: List of RSI values
            closes: Corresponding closing prices
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        for i in range(len(rsi_values)):
            close_idx = i + 15  # RSI starts after period+1 price changes
            signal = {
                'index': i,
                'close': closes[close_idx] if close_idx < len(closes) else closes[-1],
                'rsi': rsi_values[i],
                'signal': 'NEUTRAL',
                'condition': 'NORMAL',
                'confirmed': False
            }
            
            # Determine RSI conditions and signals
            if rsi_values[i] > 70:
                signal['condition'] = 'OVERBOUGHT'
                signal['signal'] = 'SELL'
                
                # Check for confirmation (RSI dropping from overbought)
                if i > 0 and rsi_values[i] < rsi_values[i-1]:
                    signal['confirmed'] = True
                    
            elif rsi_values[i] < 30:
                signal['condition'] = 'OVERSOLD'
                signal['signal'] = 'BUY'
                
                # Check for confirmation (RSI rising from oversold)
                if i > 0 and rsi_values[i] > rsi_values[i-1]:
                    signal['confirmed'] = True
                    
            elif rsi_values[i] >= 40 and rsi_values[i] <= 60:
                signal['condition'] = 'SIDEWAYS'
                signal['signal'] = 'NEUTRAL'
                
            else:
                # RSI between 30-40 or 60-70 - transition zones
                if rsi_values[i] > 50:
                    signal['condition'] = 'BULLISH_MOMENTUM'
                    signal['signal'] = 'NEUTRAL'
                else:
                    signal['condition'] = 'BEARISH_MOMENTUM'
                    signal['signal'] = 'NEUTRAL'
            
            signals.append(signal)
        
        return signals
    
    def analyze(self, symbol: str, timeframe: str, limit: int) -> Dict:
        """
        Perform RSI(14) analysis and return results as structured data.
        
        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            timeframe: Timeframe (e.g., '1d', '4h', '1h')
            limit: Number of candles to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        # Fetch OHLCV data
        ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 20:
            return {
                'error': f"Need at least 20 candles for RSI calculation. Got {len(ohlcv_data)}",
                'success': False
            }
        
        # Extract timestamps and closes
        timestamps = [candle[0] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        
        # Calculate RSI
        rsi_values = self.calculate_rsi(closes, 14)
        
        if not rsi_values:
            return {
                'error': "Unable to calculate RSI values",
                'success': False
            }
        
        # Detect signals
        signals = self.detect_rsi_signals(rsi_values, closes)
        
        if not signals:
            return {
                'error': "No signals generated",
                'success': False
            }
        
        # Get latest signal for analysis
        latest_signal = signals[-1]
        timestamp_idx = latest_signal['index'] + 15  # RSI starts after 15 periods
        if timestamp_idx >= len(timestamps):
            timestamp_idx = len(timestamps) - 1
        
        dt = datetime.fromtimestamp(timestamps[timestamp_idx] / 1000, tz=timezone.utc)
        
        # Calculate additional metrics
        current_price = closes[-1]
        current_rsi = latest_signal['rsi']
        
        # Calculate confidence score based on RSI conditions
        confidence = 0
        if latest_signal['signal'] in ['BUY', 'SELL']:
            if latest_signal['confirmed']:
                confidence += 40
            else:
                confidence += 20
                
            # Add RSI extremes confidence
            if current_rsi > 80 or current_rsi < 20:
                confidence += 30
            elif current_rsi > 70 or current_rsi < 30:
                confidence += 20
            else:
                confidence += 10
                
            # Add trend consistency
            if latest_signal['condition'] in ['OVERBOUGHT', 'OVERSOLD']:
                confidence += 25
            else:
                confidence += 15
        else:
            confidence = 30  # Neutral signals have lower confidence
        
        confidence = min(confidence, 95)  # Cap at 95%
        
        # Determine momentum state
        if current_rsi > 70:
            momentum_state = "Overbought"
        elif current_rsi < 30:
            momentum_state = "Oversold"
        elif current_rsi > 60:
            momentum_state = "Bullish"
        elif current_rsi < 40:
            momentum_state = "Bearish"
        else:
            momentum_state = "Neutral"
        
        # Determine trend direction based on RSI level
        if current_rsi > 50:
            trend_direction = "Bullish"
        elif current_rsi < 50:
            trend_direction = "Bearish"
        else:
            trend_direction = "Neutral"
        
        # Entry window assessment
        if latest_signal['signal'] in ['BUY', 'SELL'] and latest_signal['confirmed'] and confidence > 60:
            entry_window = "Optimal now"
        elif latest_signal['signal'] in ['BUY', 'SELL'] and confidence > 40:
            entry_window = "Good in next 2-3 bars"
        else:
            entry_window = "Wait for better setup"
        
        # Exit trigger
        if latest_signal['signal'] == 'BUY':
            exit_trigger = "RSI > 70 OR price reaches resistance"
        elif latest_signal['signal'] == 'SELL':
            exit_trigger = "RSI < 30 OR price reaches support"
        else:
            exit_trigger = "Wait for directional signal"
        
        # Calculate signal consistency
        signal_consistency = None
        if len(signals) > 1:
            recent_signals = signals[-5:]  # Last 5 signals
            signal_consistency_count = sum(1 for s in recent_signals if s['signal'] == latest_signal['signal'])
            signal_consistency = (signal_consistency_count / len(recent_signals)) * 100
        
        # Return structured analysis results
        return {
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'analysis_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': timestamps[timestamp_idx],
            
            # Core indicators
            'rsi_value': round(current_rsi, 1),
            'rsi_condition': latest_signal['condition'],
            'rsi_confirmed': latest_signal['confirmed'],
            
            # Price levels
            'current_price': round(current_price, 4),
            
            # Trading analysis
            'signal': latest_signal['signal'],
            'confidence_score': confidence,
            'trend_direction': trend_direction,
            'momentum_state': momentum_state,
            'entry_window': entry_window,
            'exit_trigger': exit_trigger,
            
            # Additional data
            'signal_consistency': round(signal_consistency, 0) if signal_consistency is not None else None,
            'all_signals': signals,
            'raw_data': {
                'rsi_values': rsi_values,
                'ohlcv_data': ohlcv_data
            }
        }