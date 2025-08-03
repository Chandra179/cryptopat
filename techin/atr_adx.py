"""
ATR (Average True Range) + ADX (Average Directional Index) strategy implementation.
Measures volatility and trend strength for cryptocurrency trend analysis.
"""

from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional
from data import get_data_collector


class ATR_ADXStrategy:
    """ATR + ADX strategy for volatility and trend strength analysis."""
    
    def __init__(self):
        self.collector = get_data_collector()
    
    def calculate_true_range(self, highs: List[float], lows: List[float], closes: List[float]) -> List[float]:
        """
        Calculate True Range values.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            
        Returns:
            List of True Range values
        """
        true_ranges = []
        
        for i in range(len(highs)):
            if i == 0:
                # First candle: TR = High - Low
                true_ranges.append(highs[i] - lows[i])
            else:
                # TR = max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
                hl = highs[i] - lows[i]
                hc = abs(highs[i] - closes[i-1])
                lc = abs(lows[i] - closes[i-1])
                true_ranges.append(max(hl, hc, lc))
        
        return true_ranges
    
    def calculate_atr(self, true_ranges: List[float], period: int = 14) -> List[float]:
        """
        Calculate Average True Range using smoothed moving average.
        
        Args:
            true_ranges: List of True Range values
            period: ATR period (default 14)
            
        Returns:
            List of ATR values
        """
        if len(true_ranges) < period:
            return []
        
        atr_values = []
        
        # First ATR is simple average of first 'period' TRs
        first_atr = sum(true_ranges[:period]) / period
        atr_values.append(first_atr)
        
        # Subsequent ATRs use Wilder's smoothing: ATR = ((prior ATR * (period-1)) + current TR) / period
        for i in range(period, len(true_ranges)):
            atr = ((atr_values[-1] * (period - 1)) + true_ranges[i]) / period
            atr_values.append(atr)
        
        return atr_values
    
    def calculate_directional_movement(self, highs: List[float], lows: List[float]) -> Tuple[List[float], List[float]]:
        """
        Calculate directional movement (+DM and -DM).
        
        Args:
            highs: List of high prices
            lows: List of low prices
            
        Returns:
            Tuple of (+DM, -DM) lists
        """
        plus_dm = []
        minus_dm = []
        
        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
                minus_dm.append(0)
            elif down_move > up_move and down_move > 0:
                plus_dm.append(0)
                minus_dm.append(down_move)
            else:
                plus_dm.append(0)
                minus_dm.append(0)
        
        return plus_dm, minus_dm
    
    def smooth_values(self, values: List[float], period: int = 14) -> List[float]:
        """
        Apply Wilder's smoothing to a list of values.
        
        Args:
            values: List of values to smooth
            period: Smoothing period (default 14)
            
        Returns:
            List of smoothed values
        """
        if len(values) < period:
            return []
        
        smoothed = []
        
        # First smoothed value is simple average
        first_smooth = sum(values[:period]) / period
        smoothed.append(first_smooth)
        
        # Apply Wilder's smoothing
        for i in range(period, len(values)):
            smooth = ((smoothed[-1] * (period - 1)) + values[i]) / period
            smoothed.append(smooth)
        
        return smoothed
    
    def calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], 
                     period: int = 14) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate ADX, +DI, and -DI.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            period: ADX period (default 14)
            
        Returns:
            Tuple of (ADX, +DI, -DI) lists
        """
        # Calculate True Range and ATR
        true_ranges = self.calculate_true_range(highs, lows, closes)
        atr_values = self.calculate_atr(true_ranges, period)
        
        # Calculate Directional Movement
        plus_dm, minus_dm = self.calculate_directional_movement(highs, lows)
        
        # Smooth DM values
        plus_dm_smooth = self.smooth_values(plus_dm, period)
        minus_dm_smooth = self.smooth_values(minus_dm, period)
        
        # Calculate DI values
        plus_di = []
        minus_di = []
        
        # Ensure arrays are same length
        min_length = min(len(atr_values), len(plus_dm_smooth), len(minus_dm_smooth))
        
        for i in range(min_length):
            if atr_values[i] != 0:
                plus_di.append((plus_dm_smooth[i] / atr_values[i]) * 100)
                minus_di.append((minus_dm_smooth[i] / atr_values[i]) * 100)
            else:
                plus_di.append(0)
                minus_di.append(0)
        
        # Calculate DX
        dx_values = []
        for i in range(len(plus_di)):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum != 0:
                dx = abs(plus_di[i] - minus_di[i]) / di_sum * 100
                dx_values.append(dx)
            else:
                dx_values.append(0)
        
        # Calculate ADX (smoothed DX)
        adx_values = self.smooth_values(dx_values, period)
        
        return adx_values, plus_di, minus_di
    
    def generate_signals(self, atr_values: List[float], adx_values: List[float], 
                        plus_di: List[float], minus_di: List[float],
                        closes: List[float], timestamps: List[int], period: int = 14) -> List[Dict]:
        """
        Generate ATR+ADX trading signals.
        
        Args:
            atr_values: ATR values
            adx_values: ADX values
            plus_di: +DI values
            minus_di: -DI values
            closes: Closing prices
            timestamps: Timestamp values
            period: Period used for calculations (for proper alignment)
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        # Calculate the offset needed to align all indicators
        # ATR starts at index 'period' (14 by default)
        # ADX starts at index 'period' + smoothing period (28 by default)
        adx_offset = period * 2  # 28 for default period of 14
        
        # Ensure we have enough data
        if len(closes) < adx_offset or len(adx_values) == 0:
            return signals
        
        # Generate signals starting from where we have all indicators
        for i in range(1, len(adx_values)):
            # Calculate the corresponding index in the original data
            data_idx = adx_offset + i
            
            # Ensure we don't go out of bounds
            if data_idx >= len(closes) or data_idx >= len(timestamps):
                break
                
            # Get ATR value (ATR array is longer than ADX array)
            atr_idx = period + i  # ATR starts at period offset
            if atr_idx >= len(atr_values):
                atr_idx = len(atr_values) - 1
                
            # Calculate volatility percentile for ATR threshold
            current_atr = atr_values[atr_idx]
            recent_atr_values = atr_values[max(0, atr_idx-20):atr_idx+1]
            avg_atr = sum(recent_atr_values) / len(recent_atr_values) if recent_atr_values else current_atr
            high_volatility = current_atr > avg_atr * 1.2  # 20% above average
                
            signal = {
                'timestamp': timestamps[data_idx],
                'close': closes[data_idx],
                'atr': current_atr,
                'adx': adx_values[i],
                'plus_di': plus_di[i] if i < len(plus_di) else 0,
                'minus_di': minus_di[i] if i < len(minus_di) else 0,
                'signal': 'HOLD',
                'trend_strength': 'WEAK',
                'volatility': 'HIGH' if high_volatility else 'NORMAL',
                'description': ''
            }
            
            # Trend strength based on ADX
            if signal['adx'] > 25:
                signal['trend_strength'] = 'STRONG'
            elif signal['adx'] > 20:
                signal['trend_strength'] = 'MODERATE'
            else:
                signal['trend_strength'] = 'WEAK'
            
            # Generate signals based on DI crossovers, ADX strength, and ATR
            if i > 0:  # Need previous values for crossover detection
                prev_plus_di = plus_di[i-1] if i-1 < len(plus_di) else 0
                prev_minus_di = minus_di[i-1] if i-1 < len(minus_di) else 0
                
                # BUY signal: Strong trend + bullish crossover + sufficient volatility
                if (signal['adx'] > 25 and 
                    prev_plus_di <= prev_minus_di and 
                    signal['plus_di'] > signal['minus_di'] and
                    high_volatility):
                    signal['signal'] = 'BUY'
                    signal['description'] = 'Bullish Trend Confirmed - High Volatility'
                
                # Strong BUY without high volatility requirement
                elif (signal['adx'] > 30 and 
                      prev_plus_di <= prev_minus_di and 
                      signal['plus_di'] > signal['minus_di']):
                    signal['signal'] = 'BUY'
                    signal['description'] = 'Strong Bullish Trend Confirmed'
                
                # SELL signal: Strong trend + bearish crossover + sufficient volatility
                elif (signal['adx'] > 25 and 
                      prev_plus_di >= prev_minus_di and 
                      signal['plus_di'] < signal['minus_di'] and
                      high_volatility):
                    signal['signal'] = 'SELL'
                    signal['description'] = 'Bearish Trend Confirmed - High Volatility'
                
                # Strong SELL without high volatility requirement
                elif (signal['adx'] > 30 and 
                      prev_plus_di >= prev_minus_di and 
                      signal['plus_di'] < signal['minus_di']):
                    signal['signal'] = 'SELL'
                    signal['description'] = 'Strong Bearish Trend Confirmed'
                
                # Weak trend warning
                elif signal['adx'] < 20:
                    signal['description'] = 'Weak Trend - Avoid New Positions'
                
                # Moderate trend
                elif signal['adx'] < 25:
                    if signal['plus_di'] > signal['minus_di']:
                        signal['description'] = 'Moderate Bullish Trend'
                    else:
                        signal['description'] = 'Moderate Bearish Trend'
                
                else:
                    signal['description'] = 'Strong Trend - No Clear Direction'
            
            signals.append(signal)
        
        return signals
    
    def analyze(self, symbol: str, timeframe: str, limit: int, ohlcv_data: Optional[List] = None) -> Dict:
        """
        Perform ATR+ADX analysis and return results as structured data.
        
        Args:
            symbol: Trading pair (e.g., 'ETH/USDT')
            timeframe: Timeframe (e.g., '4h', '1d')
            limit: Number of candles to analyze
            ohlcv_data: Optional pre-fetched OHLCV data
            
        Returns:
            Dictionary containing analysis results
        """
        # Fetch OHLCV data if not provided
        if ohlcv_data is None:
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 30:
            return {
                'error': f"Need at least 30 candles for ATR+ADX calculation. Got {len(ohlcv_data)}",
                'success': False
            }
        
        # Extract OHLCV components
        timestamps = [candle[0] for candle in ohlcv_data]
        opens = [candle[1] for candle in ohlcv_data]
        highs = [candle[2] for candle in ohlcv_data]
        lows = [candle[3] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        volumes = [candle[5] for candle in ohlcv_data]
        
        # Calculate ATR
        true_ranges = self.calculate_true_range(highs, lows, closes)
        atr_values = self.calculate_atr(true_ranges, 14)
        
        # Calculate ADX, +DI, -DI
        adx_values, plus_di, minus_di = self.calculate_adx(highs, lows, closes, 14)
        
        if not atr_values or not adx_values:
            return {
                'error': "Unable to calculate ATR+ADX indicators",
                'success': False
            }
        
        # Generate signals
        signals = self.generate_signals(atr_values, adx_values, plus_di, minus_di, closes, timestamps, 14)
        
        # Get latest signal for analysis
        if not signals:
            return {
                'error': "No signals generated",
                'success': False
            }
            
        latest_signal = signals[-1]
        dt = datetime.fromtimestamp(latest_signal['timestamp'] / 1000, tz=timezone.utc)
        
        # Calculate additional metrics for analysis
        current_price = closes[-1]
        atr_current = latest_signal['atr']
        adx_current = latest_signal['adx']
        
        # Calculate support/resistance levels based on ATR
        support_level = current_price - (atr_current * 2)
        resistance_level = current_price + (atr_current * 2)
        
        # Calculate stop loss and take profit zones
        if latest_signal['signal'] == 'BUY':
            stop_zone = current_price - (atr_current * 1.5)
            tp_low = current_price + (atr_current * 2.5)
            tp_high = current_price + (atr_current * 3.5)
        elif latest_signal['signal'] == 'SELL':
            stop_zone = current_price + (atr_current * 1.5)
            tp_low = current_price - (atr_current * 2.5)
            tp_high = current_price - (atr_current * 3.5)
        else:
            stop_zone = current_price - (atr_current * 1.5)
            tp_low = current_price + (atr_current * 2)
            tp_high = current_price + (atr_current * 3)
        
        # Calculate Risk/Reward ratio
        if latest_signal['signal'] in ['BUY', 'SELL']:
            risk = abs(current_price - stop_zone)
            reward = abs(tp_low - current_price) if tp_low != current_price else abs(tp_high - current_price)
            rr_ratio = reward / risk if risk > 0 else 0
        else:
            rr_ratio = 0
        
        # Calculate confidence score based on ATR+ADX strength
        confidence = 0
        if adx_current > 30:
            confidence += 40
        elif adx_current > 25:
            confidence += 30
        elif adx_current > 20:
            confidence += 20
        
        # Add volatility component
        if latest_signal.get('volatility') == 'HIGH':
            confidence += 20
        else:
            confidence += 10
        
        # Add directional strength
        di_diff = abs(latest_signal['plus_di'] - latest_signal['minus_di'])
        if di_diff > 20:
            confidence += 20
        elif di_diff > 10:
            confidence += 15
        else:
            confidence += 5
        
        # Add trend consistency (simplified)
        if latest_signal['trend_strength'] == 'STRONG':
            confidence += 15
        elif latest_signal['trend_strength'] == 'MODERATE':
            confidence += 10
        else:
            confidence += 5
        
        confidence = min(confidence, 95)  # Cap at 95%
        
        # Determine momentum state
        if adx_current > 30 and di_diff > 15:
            momentum_state = "Accelerating"
        elif adx_current > 20 and di_diff > 10:
            momentum_state = "Building"
        elif adx_current < 20:
            momentum_state = "Stalling"
        else:
            momentum_state = "Consolidating"
        
        # Determine trend direction
        if latest_signal['plus_di'] > latest_signal['minus_di']:
            trend_direction = "Bullish"
        elif latest_signal['plus_di'] < latest_signal['minus_di']:
            trend_direction = "Bearish"
        else:
            trend_direction = "Neutral"
        
        # Entry window assessment
        if latest_signal['signal'] in ['BUY', 'SELL'] and confidence > 70:
            entry_window = "Optimal now"
        elif latest_signal['signal'] in ['BUY', 'SELL'] and confidence > 50:
            entry_window = "Good in next 2-3 bars"
        else:
            entry_window = "Wait for better setup"
        
        # Exit trigger
        if latest_signal['signal'] == 'BUY':
            exit_trigger = f"ADX < 20 OR -DI crosses above +DI"
        elif latest_signal['signal'] == 'SELL':
            exit_trigger = f"ADX < 20 OR +DI crosses above -DI"
        else:
            exit_trigger = f"Wait for directional signal"
        
        # Expected drawdown
        max_drawdown = (atr_current / current_price) * 100 * 1.5
        
        # Calculate signal consistency
        signal_consistency = None
        if len(signals) > 1:
            recent_signals = signals[-5:]  # Last 5 signals
            signal_consistency_count = sum(1 for s in recent_signals if s['signal'] == latest_signal['signal'])
            signal_consistency = (signal_consistency_count / len(recent_signals)) * 100
        
        # Return structured analysis results
        return {
            'success': True,
            'analysis_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': latest_signal['timestamp'],
            
            # Core indicators
            'atr_value': round(atr_current, 4),
            'adx_strength': round(adx_current, 1),
            'plus_di': round(latest_signal['plus_di'], 1),
            'minus_di': round(latest_signal['minus_di'], 1),
            'di_divergence': round(di_diff, 1),
            'volatility': latest_signal.get('volatility', 'NORMAL'),
            'trend_strength': latest_signal['trend_strength'],
            
            # Price levels
            'current_price': round(current_price, 4),
            'support_level': round(support_level, 4),
            'resistance_level': round(resistance_level, 4),
            'stop_zone': round(stop_zone, 4),
            'tp_low': round(tp_low, 4),
            'tp_high': round(tp_high, 4),
            
            # Trading analysis
            'signal': latest_signal['signal'],
            'description': latest_signal['description'],
            'confidence_score': confidence,
            'trend_direction': trend_direction,
            'momentum_state': momentum_state,
            'entry_window': entry_window,
            'exit_trigger': exit_trigger,
            'rr_ratio': round(rr_ratio, 1),
            'max_drawdown': round(max_drawdown, 1),
            
            # Additional data
            'signal_consistency': round(signal_consistency, 0) if signal_consistency is not None else None,
            'all_signals': signals,
            'raw_data': {
                'atr_values': atr_values,
                'adx_values': adx_values,
                'plus_di_values': plus_di,
                'minus_di_values': minus_di,
                'ohlcv_data': ohlcv_data
            }
        }
    




