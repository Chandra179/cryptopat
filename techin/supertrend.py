"""
Supertrend indicator implementation for cryptocurrency trend analysis.
Uses ATR-based dynamic support/resistance levels for trend identification.
"""

from datetime import datetime
from typing import List, Tuple, Optional
from data import get_data_collector


class SupertrendStrategy:
    """Supertrend indicator for dynamic trend identification."""
    
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
            period: Period for ATR calculation (default 14)
            
        Returns:
            List of ATR values
        """
        atr_values = []
        
        for i in range(len(true_ranges)):
            if i < period - 1:
                atr_values.append(None)
            elif i == period - 1:
                # First ATR = simple average of first 'period' TRs
                atr_values.append(sum(true_ranges[:period]) / period)
            else:
                # Smoothed ATR = (Previous ATR * (period-1) + Current TR) / period
                prev_atr = atr_values[i-1]
                current_tr = true_ranges[i]
                smoothed_atr = (prev_atr * (period - 1) + current_tr) / period
                atr_values.append(smoothed_atr)
        
        return atr_values
    
    def calculate_supertrend(self, highs: List[float], lows: List[float], closes: List[float], 
                           atr_period: int = 10, multiplier: float = 3.0) -> Tuple[List[float], List[str]]:
        """
        Calculate Supertrend indicator.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            atr_period: Period for ATR calculation (default 10)
            multiplier: ATR multiplier (default 3.0)
            
        Returns:
            Tuple of (supertrend_values, trend_directions)
        """
        # Calculate ATR
        true_ranges = self.calculate_true_range(highs, lows, closes)
        atr_values = self.calculate_atr(true_ranges, atr_period)
        
        # Calculate basic upper and lower bands
        basic_upper_bands = []
        basic_lower_bands = []
        
        for i in range(len(closes)):
            if atr_values[i] is None:
                basic_upper_bands.append(None)
                basic_lower_bands.append(None)
            else:
                hl2 = (highs[i] + lows[i]) / 2  # Typical price
                basic_upper_bands.append(hl2 + (multiplier * atr_values[i]))
                basic_lower_bands.append(hl2 - (multiplier * atr_values[i]))
        
        # Calculate final upper and lower bands
        final_upper_bands = []
        final_lower_bands = []
        
        for i in range(len(closes)):
            if basic_upper_bands[i] is None:
                final_upper_bands.append(None)
                final_lower_bands.append(None)
            else:
                # Final Upper Band
                if i == 0 or basic_upper_bands[i-1] is None:
                    final_upper_bands.append(basic_upper_bands[i])
                else:
                    if basic_upper_bands[i] < final_upper_bands[i-1] or closes[i-1] > final_upper_bands[i-1]:
                        final_upper_bands.append(basic_upper_bands[i])
                    else:
                        final_upper_bands.append(final_upper_bands[i-1])
                
                # Final Lower Band
                if i == 0 or basic_lower_bands[i-1] is None:
                    final_lower_bands.append(basic_lower_bands[i])
                else:
                    if basic_lower_bands[i] > final_lower_bands[i-1] or closes[i-1] < final_lower_bands[i-1]:
                        final_lower_bands.append(basic_lower_bands[i])
                    else:
                        final_lower_bands.append(final_lower_bands[i-1])
        
        # Calculate Supertrend and trend direction
        supertrend_values = []
        trend_directions = []
        
        for i in range(len(closes)):
            if final_upper_bands[i] is None or final_lower_bands[i] is None:
                supertrend_values.append(None)
                trend_directions.append("UNKNOWN")
            else:
                if i == 0:
                    # Initial trend determination
                    if closes[i] <= final_lower_bands[i]:
                        supertrend_values.append(final_upper_bands[i])
                        trend_directions.append("BEARISH")
                    else:
                        supertrend_values.append(final_lower_bands[i])
                        trend_directions.append("BULLISH")
                else:
                    prev_trend = trend_directions[i-1]
                    
                    if prev_trend == "BULLISH":
                        if closes[i] < final_lower_bands[i]:
                            supertrend_values.append(final_upper_bands[i])
                            trend_directions.append("BEARISH")
                        else:
                            supertrend_values.append(final_lower_bands[i])
                            trend_directions.append("BULLISH")
                    else:  # prev_trend == "BEARISH"
                        if closes[i] > final_upper_bands[i]:
                            supertrend_values.append(final_lower_bands[i])
                            trend_directions.append("BULLISH")
                        else:
                            supertrend_values.append(final_upper_bands[i])
                            trend_directions.append("BEARISH")
        
        return supertrend_values, trend_directions
    
    def get_signal(self, current_trend: str, previous_trend: str, close: float, supertrend: float) -> Tuple[str, str]:
        """
        Generate trading signal based on Supertrend.
        
        Args:
            current_trend: Current trend direction
            previous_trend: Previous trend direction
            close: Current close price
            supertrend: Current supertrend value
            
        Returns:
            Tuple of (signal, status)
        """
        if current_trend == "UNKNOWN" or previous_trend == "UNKNOWN":
            return "HOLD", "⏳ Insufficient Data"
        
        # Trend reversal signals
        if previous_trend == "BEARISH" and current_trend == "BULLISH":
            return "BUY", "✅ Trend Reversal Confirmed"
        elif previous_trend == "BULLISH" and current_trend == "BEARISH":
            return "SELL", "🔻 Bearish Trend Shift"
        
        # Trend continuation
        if current_trend == "BULLISH":
            if close > supertrend:
                return "HOLD", "📈 Bullish Trend Continues"
            else:
                return "HOLD", "⚠️ Price Near Support"
        else:  # BEARISH
            if close < supertrend:
                return "HOLD", "📉 Bearish Trend Continues"
            else:
                return "HOLD", "⚠️ Price Near Resistance"
    
    def analyze(self, symbol: str, timeframe: str, limit: int, ohlcv_data: Optional[List] = None) -> dict:
        """
        Perform Supertrend analysis and return results as structured data.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h')
            limit: Number of candles to analyze
            ohlcv_data: Optional pre-fetched OHLCV data
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Fetch OHLCV data if not provided
            if ohlcv_data is None:
                ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
            
            if not ohlcv_data or len(ohlcv_data) < 30:
                return {
                    'error': f"Need at least 30 candles for Supertrend calculation. Got {len(ohlcv_data)}",
                    'success': False
                }
            
            # Extract price data
            timestamps = [candle[0] for candle in ohlcv_data]
            opens = [float(candle[1]) for candle in ohlcv_data]
            highs = [float(candle[2]) for candle in ohlcv_data]
            lows = [float(candle[3]) for candle in ohlcv_data]
            closes = [float(candle[4]) for candle in ohlcv_data]
            volumes = [float(candle[5]) for candle in ohlcv_data]
            
            # Calculate Supertrend with default parameters
            atr_period = 10
            multiplier = 3.0
            supertrend_values, trend_directions = self.calculate_supertrend(
                highs, lows, closes, atr_period, multiplier
            )
            
            # Find latest valid signal
            latest_idx = len(ohlcv_data) - 1
            while latest_idx >= 0 and supertrend_values[latest_idx] is None:
                latest_idx -= 1
                
            if latest_idx < 0:
                return {
                    'error': "No valid Supertrend signals found",
                    'success': False
                }
            
            # Get current values
            current_price = closes[latest_idx]
            current_supertrend = supertrend_values[latest_idx]
            current_trend = trend_directions[latest_idx]
            previous_trend = trend_directions[latest_idx-1] if latest_idx > 0 else "UNKNOWN"
            
            # Get signal
            signal, status = self.get_signal(current_trend, previous_trend, current_price, current_supertrend)
            
            # Calculate ATR for additional metrics
            true_ranges = self.calculate_true_range(highs, lows, closes)
            atr_values = self.calculate_atr(true_ranges, atr_period)
            current_atr = atr_values[-1] if atr_values else 0
            
            # Calculate distance from Supertrend
            distance_from_st = abs(current_price - current_supertrend)
            distance_percent = (distance_from_st / current_price) * 100
            
            # Calculate support/resistance levels
            if current_trend == "BULLISH":
                support_level = current_supertrend
                resistance_level = current_price + (current_atr * 1.5) if current_atr else current_price * 1.02
            else:
                support_level = current_price - (current_atr * 1.5) if current_atr else current_price * 0.98
                resistance_level = current_supertrend
            
            # Calculate confidence score
            confidence = 50  # Base confidence
            
            # Add trend strength
            if distance_percent < 1:
                confidence += 20  # Close to Supertrend line
            elif distance_percent < 2:
                confidence += 15
            elif distance_percent < 3:
                confidence += 10
            
            # Add signal strength
            if signal in ["BUY", "SELL"]:
                confidence += 20
            
            # Add trend consistency (check last few candles)
            trend_consistency = 0
            lookback = min(5, len(trend_directions))
            for i in range(len(trend_directions) - lookback, len(trend_directions)):
                if i >= 0 and trend_directions[i] == current_trend:
                    trend_consistency += 1
            
            consistency_score = (trend_consistency / lookback) * 20
            confidence += consistency_score
            
            confidence = min(confidence, 95)  # Cap at 95%
            
            # Determine momentum state
            if signal in ["BUY", "SELL"] and confidence > 70:
                momentum_state = "Strong"
            elif current_trend != "UNKNOWN" and confidence > 50:
                momentum_state = "Moderate"
            else:
                momentum_state = "Weak"
            
            # Entry timing
            if signal in ["BUY", "SELL"] and confidence > 70:
                entry_window = "Immediate"
            elif signal in ["BUY", "SELL"]:
                entry_window = "Wait for confirmation"
            else:
                entry_window = "No entry signal"
            
            # Stop loss and take profit
            if signal == "BUY":
                stop_loss = support_level
                take_profit = current_price + (distance_from_st * 2)
            elif signal == "SELL":
                stop_loss = resistance_level
                take_profit = current_price - (distance_from_st * 2)
            else:
                stop_loss = support_level if current_trend == "BULLISH" else resistance_level
                take_profit = current_price + (current_atr * 2) if current_atr else current_price * 1.02
            
            # Risk/Reward ratio
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit - current_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Analysis timestamp
            dt = datetime.fromtimestamp(timestamps[latest_idx] / 1000)
            
            return {
                'success': True,
                'analysis_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': timestamps[latest_idx],
                
                # Core indicators
                'current_price': round(current_price, 4),
                'supertrend_value': round(current_supertrend, 4),
                'trend_direction': current_trend,
                'distance_from_st': round(distance_from_st, 4),
                'distance_percent': round(distance_percent, 2),
                'atr_value': round(current_atr, 4) if current_atr else 0,
                
                # Price levels
                'support_level': round(support_level, 4),
                'resistance_level': round(resistance_level, 4),
                'stop_loss': round(stop_loss, 4),
                'take_profit': round(take_profit, 4),
                
                # Trading analysis
                'signal': signal,
                'status': status,
                'confidence_score': round(confidence, 0),
                'momentum_state': momentum_state,
                'entry_window': entry_window,
                'rr_ratio': round(rr_ratio, 1),
                'trend_consistency': round(consistency_score, 0),
                
                # Additional data
                'multiplier': multiplier,
                'atr_period': atr_period,
                'raw_data': {
                    'supertrend_values': supertrend_values,
                    'trend_directions': trend_directions,
                    'atr_values': atr_values,
                    'ohlcv_data': ohlcv_data
                }
            }
            
        except Exception as e:
            return {
                'error': f"Analysis failed: {str(e)}",
                'success': False
            }

