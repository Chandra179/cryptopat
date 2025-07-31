"""
Market Regime Detection System

Advanced market regime classification using multiple technical indicators to provide
context-aware analysis for trading strategies. Identifies trending vs ranging markets,
volatility regimes, and session-based patterns.
"""

import logging
from typing import Dict, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import statistics
from data import get_data_collector

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Professional market regime detection system for cryptocurrency markets.
    
    Detects:
    - Trend strength and direction (trending vs ranging)
    - Volatility regimes (low, normal, high)
    - Session types (Asian, European, US)
    - Market phase transitions
    """
    
    def __init__(self):
        """Initialize the market regime detector."""
        self.collector = get_data_collector()
        
    def calculate_adx(self, ohlcv_data: List[List], period: int = 14) -> Tuple[List[float], List[float]]:
        """
        Calculate Average Directional Index (ADX) for trend strength measurement.
        
        Args:
            ohlcv_data: OHLCV candlestick data
            period: ADX calculation period
            
        Returns:
            Tuple of (adx_values, trend_direction_values)
        """
        if len(ohlcv_data) < period + 20:
            return [None] * len(ohlcv_data), [None] * len(ohlcv_data)
        
        highs = [candle[2] for candle in ohlcv_data]
        lows = [candle[3] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        
        # Calculate True Range (TR)
        tr_values = []
        for i in range(1, len(ohlcv_data)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i-1]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)
        
        # Calculate Directional Movement (DM+, DM-)
        dm_plus = []
        dm_minus = []
        
        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                dm_plus.append(up_move)
            else:
                dm_plus.append(0)
                
            if down_move > up_move and down_move > 0:
                dm_minus.append(down_move)
            else:
                dm_minus.append(0)
        
        # Smooth TR, DM+, DM- using Wilder's smoothing
        def wilders_smoothing(values: List[float], period: int) -> List[float]:
            if len(values) < period:
                return [None] * len(values)
            
            smoothed = []
            # First smoothed value is simple average
            first_avg = sum(values[:period]) / period
            smoothed.extend([None] * (period - 1))
            smoothed.append(first_avg)
            
            # Subsequent values use Wilder's formula
            for i in range(period, len(values)):
                prev_smoothed = smoothed[-1]
                new_smoothed = (prev_smoothed * (period - 1) + values[i]) / period
                smoothed.append(new_smoothed)
            
            return smoothed
        
        smoothed_tr = wilders_smoothing(tr_values, period)
        smoothed_dm_plus = wilders_smoothing(dm_plus, period)
        smoothed_dm_minus = wilders_smoothing(dm_minus, period)
        
        # Calculate DI+, DI-, DX, ADX
        adx_values = [None]  # First value has no previous data
        trend_direction = [None]
        
        for i in range(period, len(smoothed_tr)):
            if (smoothed_tr[i] is None or smoothed_dm_plus[i] is None or 
                smoothed_dm_minus[i] is None or smoothed_tr[i] == 0):
                adx_values.append(None)
                trend_direction.append(None)
                continue
            
            di_plus = (smoothed_dm_plus[i] / smoothed_tr[i]) * 100
            di_minus = (smoothed_dm_minus[i] / smoothed_tr[i]) * 100
            
            dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100
            adx_values.append(dx)  # We'll smooth this next
            
            # Trend direction
            if di_plus > di_minus:
                trend_direction.append(1)  # Uptrend
            else:
                trend_direction.append(-1)  # Downtrend
        
        # Smooth DX to get ADX
        dx_values = [v for v in adx_values if v is not None]
        if len(dx_values) >= period:
            adx_smoothed = wilders_smoothing(dx_values, period)
            
            # Align with original data
            final_adx = [None] * (len(adx_values) - len(dx_values))
            final_adx.extend(adx_smoothed)
        else:
            final_adx = adx_values
        
        # Pad to match input length
        while len(final_adx) < len(ohlcv_data):
            final_adx.append(None)
        
        while len(trend_direction) < len(ohlcv_data):
            trend_direction.append(None)
            
        return final_adx, trend_direction
    
    def calculate_volatility_regime(self, ohlcv_data: List[List], lookback: int = 20) -> str:
        """
        Calculate volatility regime classification.
        
        Args:
            ohlcv_data: OHLCV candlestick data
            lookback: Periods to look back for volatility calculation
            
        Returns:
            Volatility regime: "low", "normal", "high"
        """
        if len(ohlcv_data) < lookback + 10:
            return "insufficient_data"
        
        # Calculate price changes (returns)
        closes = [candle[4] for candle in ohlcv_data]
        returns = []
        for i in range(1, len(closes)):
            ret = (closes[i] - closes[i-1]) / closes[i-1]
            returns.append(ret)
        
        if len(returns) < lookback:
            return "insufficient_data"
        
        # Calculate rolling volatility (standard deviation of returns)
        recent_returns = returns[-lookback:]
        volatility = statistics.stdev(recent_returns) if len(recent_returns) > 1 else 0
        
        # Historical volatility for comparison
        if len(returns) >= lookback * 3:
            historical_returns = returns[-lookback*3:-lookback]
            historical_vol = statistics.stdev(historical_returns) if len(historical_returns) > 1 else volatility
        else:
            historical_vol = volatility
        
        # Classify volatility regime
        vol_ratio = volatility / historical_vol if historical_vol > 0 else 1.0
        
        if vol_ratio < 0.7:
            return "low"
        elif vol_ratio > 1.5:
            return "high"
        else:
            return "normal"
    
    def detect_session_type(self, timestamp: float) -> str:
        """
        Detect trading session based on UTC timestamp.
        
        Args:
            timestamp: UTC timestamp in milliseconds
            
        Returns:
            Session type: "asian", "european", "us", "overlap"
        """
        dt = datetime.fromtimestamp(timestamp / 1000)
        hour_utc = dt.hour
        
        # Define session hours (UTC)
        # Asian: 00:00-09:00 UTC (Tokyo/Sydney)
        # European: 07:00-16:00 UTC (London)
        # US: 13:00-22:00 UTC (New York)
        
        asian_hours = range(0, 9)
        european_hours = range(7, 16)
        us_hours = range(13, 22)
        
        in_asian = hour_utc in asian_hours
        in_european = hour_utc in european_hours
        in_us = hour_utc in us_hours
        
        # Check for overlaps
        if in_european and in_us:
            return "european_us_overlap"
        elif in_asian and in_european:
            return "asian_european_overlap"
        elif in_asian:
            return "asian"
        elif in_european:
            return "european"
        elif in_us:
            return "us"
        else:
            return "off_hours"
    
    def detect_price_action_regime(self, ohlcv_data: List[List], 
                                 ema_fast: List[float], ema_slow: List[float]) -> str:
        """
        Detect price action regime based on EMA behavior and price structure.
        
        Args:
            ohlcv_data: OHLCV candlestick data
            ema_fast: Fast EMA values
            ema_slow: Slow EMA values
            
        Returns:
            Price action regime classification
        """
        if len(ohlcv_data) < 20 or len(ema_fast) < 20 or len(ema_slow) < 20:
            return "insufficient_data"
        
        closes = [candle[4] for candle in ohlcv_data]
        
        # Analyze recent EMA relationship
        recent_periods = 15
        recent_fast = [ema for ema in ema_fast[-recent_periods:] if ema is not None]
        recent_slow = [ema for ema in ema_slow[-recent_periods:] if ema is not None]
        recent_closes = closes[-recent_periods:]
        
        if len(recent_fast) < 10 or len(recent_slow) < 10:
            return "insufficient_data"
        
        # Calculate EMA separation consistency
        separations = []
        for i in range(min(len(recent_fast), len(recent_slow))):
            if recent_slow[i] != 0:
                separation = (recent_fast[i] - recent_slow[i]) / recent_slow[i]
                separations.append(separation)
        
        if not separations:
            return "insufficient_data"
        
        # Analyze separation characteristics
        avg_separation = statistics.mean(separations)
        separation_consistency = 1 - (statistics.stdev(separations) / abs(avg_separation)) if avg_separation != 0 else 0
        
        # Analyze price vs EMA relationship
        price_above_fast = sum(1 for i, close in enumerate(recent_closes) 
                              if i < len(recent_fast) and close > recent_fast[i])
        price_above_slow = sum(1 for i, close in enumerate(recent_closes)
                              if i < len(recent_slow) and close > recent_slow[i])
        
        fast_above_slow = sum(1 for i in range(min(len(recent_fast), len(recent_slow)))
                             if recent_fast[i] > recent_slow[i])
        
        # Classification logic
        strong_trend_threshold = 0.8
        trend_consistency = fast_above_slow / min(len(recent_fast), len(recent_slow))
        
        if abs(avg_separation) > 0.01 and separation_consistency > 0.6:
            if trend_consistency > strong_trend_threshold:
                return "strong_uptrend" if avg_separation > 0 else "strong_downtrend"
            else:
                return "weak_uptrend" if avg_separation > 0 else "weak_downtrend"
        elif abs(avg_separation) < 0.005 and separation_consistency < 0.4:
            return "ranging_choppy"
        else:
            return "transitional"
    
    def analyze_regime(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Dict:
        """
        Comprehensive market regime analysis.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Analysis timeframe
            limit: Number of candles to analyze
            
        Returns:
            Complete regime analysis dictionary
        """
        try:
            # Fetch data
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
            if len(ohlcv_data) < 50:
                return {"error": f"Insufficient data for {symbol}"}
            
            # Calculate technical indicators
            closes = [candle[4] for candle in ohlcv_data]
            
            # EMAs for trend analysis
            df = pd.DataFrame({'close': closes})
            ema_9 = df['close'].ewm(span=9, adjust=False).mean().tolist()
            ema_21 = df['close'].ewm(span=21, adjust=False).mean().tolist()
            
            # ADX for trend strength
            adx_values, trend_direction = self.calculate_adx(ohlcv_data)
            
            # Current market state
            current_timestamp = ohlcv_data[-1][0]
            current_close = closes[-1]
            current_adx = adx_values[-1] if adx_values[-1] is not None else 0
            
            # Regime classifications
            volatility_regime = self.calculate_volatility_regime(ohlcv_data)
            session_type = self.detect_session_type(current_timestamp)
            price_action_regime = self.detect_price_action_regime(ohlcv_data, ema_9, ema_21)
            
            # Trend strength classification
            if current_adx >= 30:
                trend_strength = "strong"
            elif current_adx >= 20:
                trend_strength = "moderate"
            elif current_adx >= 15:
                trend_strength = "weak"
            else:
                trend_strength = "no_trend"
            
            # Overall regime synthesis
            if "trend" in price_action_regime and trend_strength in ["strong", "moderate"]:
                overall_regime = f"{trend_strength}_trend"
            elif price_action_regime == "ranging_choppy" or trend_strength == "no_trend":
                overall_regime = "ranging"
            else:
                overall_regime = "transitional"
            
            # Trading recommendations
            if overall_regime == "strong_trend" and volatility_regime != "high":
                trading_advice = "favorable_for_trend_following"
            elif overall_regime == "ranging" and volatility_regime == "low":
                trading_advice = "favorable_for_mean_reversion"
            elif volatility_regime == "high":
                trading_advice = "high_risk_reduce_position_size"
            else:
                trading_advice = "neutral_wait_for_clearer_signals"
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": current_timestamp,
                "current_price": current_close,
                "overall_regime": overall_regime,
                "trend_strength": trend_strength,
                "trend_direction": "bullish" if trend_direction[-1] == 1 else "bearish" if trend_direction[-1] == -1 else "neutral",
                "adx_value": current_adx,
                "volatility_regime": volatility_regime,
                "session_type": session_type,
                "price_action_regime": price_action_regime,
                "trading_advice": trading_advice,
                "confidence": self._calculate_regime_confidence(trend_strength, volatility_regime, current_adx)
            }
            
        except Exception as e:
            logger.error(f"Error in regime analysis for {symbol}: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _calculate_regime_confidence(self, trend_strength: str, volatility_regime: str, adx_value: float) -> str:
        """Calculate confidence level for regime classification."""
        confidence_score = 0.0
        
        # ADX-based confidence
        if adx_value >= 35:
            confidence_score += 0.4
        elif adx_value >= 25:
            confidence_score += 0.3
        elif adx_value >= 15:
            confidence_score += 0.2
        
        # Trend strength confidence
        if trend_strength == "strong":
            confidence_score += 0.3
        elif trend_strength == "moderate":
            confidence_score += 0.2
        
        # Volatility stability confidence
        if volatility_regime in ["low", "normal"]:
            confidence_score += 0.3
        
        if confidence_score >= 0.8:
            return "HIGH"
        elif confidence_score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def format_regime_output(self, regime_data: Dict) -> str:
        """Format regime analysis for terminal display."""
        if "error" in regime_data:
            return f"Market Regime Analysis Error: {regime_data['error']}"
        
        dt = datetime.fromtimestamp(regime_data['timestamp'] / 1000)
        
        output = []
        output.append(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] {regime_data['symbol']} | TF: {regime_data['timeframe']}")
        output.append(f"Current Price: {regime_data['current_price']:.4f}")
        output.append("")
        
        # Main regime classification
        regime = regime_data['overall_regime'].replace('_', ' ').title()
        trend_dir = regime_data['trend_direction'].title()
        trend_str = regime_data['trend_strength'].replace('_', ' ').title()
        
        output.append(f"📊 Overall Regime: {regime}")
        output.append(f"📈 Trend: {trend_dir} ({trend_str})")
        output.append(f"🎯 ADX: {regime_data['adx_value']:.1f}")
        
        # Market context
        vol_regime = regime_data['volatility_regime'].title()
        session = regime_data['session_type'].replace('_', ' ').title()
        price_action = regime_data['price_action_regime'].replace('_', ' ').title()
        
        output.append(f"📊 Volatility: {vol_regime}")
        output.append(f"🌍 Session: {session}")
        output.append(f"💹 Price Action: {price_action}")
        
        # Trading advice
        advice = regime_data['trading_advice'].replace('_', ' ').title()
        confidence = regime_data['confidence']
        
        output.append("")
        output.append(f"💡 Trading Advice: {advice}")
        output.append(f"🎯 Confidence: {confidence}")
        
        return "\n".join(output)

def analyze_market_regime(symbol: str, timeframe: str = '1h', limit: int = 100) -> str:
    """
    Convenience function for market regime analysis.
    
    Args:
        symbol: Trading pair symbol
        timeframe: Analysis timeframe
        limit: Number of candles to analyze
        
    Returns:
        Formatted regime analysis output
    """
    detector = MarketRegimeDetector()
    regime_data = detector.analyze_regime(symbol, timeframe, limit)
    return detector.format_regime_output(regime_data)