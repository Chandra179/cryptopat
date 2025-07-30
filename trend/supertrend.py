"""
Supertrend indicator implementation for cryptocurrency trend analysis.
Uses ATR-based dynamic support/resistance levels for trend identification.
"""

import sys
from datetime import datetime
from typing import List, Tuple, Dict
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
    
    def analyze(self, symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 100,
                atr_period: int = 10, multiplier: float = 3.0):
        """
        Run Supertrend analysis on cryptocurrency data.
        
        Args:
            symbol: Trading pair symbol (default "BTC/USDT")
            timeframe: Timeframe for analysis (default "1h")
            limit: Number of candles to analyze (default 100)
            atr_period: Period for ATR calculation (default 10)
            multiplier: ATR multiplier (default 3.0)
        """
        try:
            # Fetch OHLCV data
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
            
            if not ohlcv_data or len(ohlcv_data) < max(20, atr_period + 10):
                print(f"❌ Insufficient data for Supertrend analysis. Need at least {max(20, atr_period + 10)} candles.")
                return
            
            # Extract price data
            timestamps = [datetime.fromtimestamp(candle[0] / 1000) for candle in ohlcv_data]
            opens = [float(candle[1]) for candle in ohlcv_data]
            highs = [float(candle[2]) for candle in ohlcv_data]
            lows = [float(candle[3]) for candle in ohlcv_data]
            closes = [float(candle[4]) for candle in ohlcv_data]
            
            # Calculate Supertrend
            supertrend_values, trend_directions = self.calculate_supertrend(
                highs, lows, closes, atr_period, multiplier
            )
            
            # Display results for last 10 candles
            display_count = min(10, len(ohlcv_data))
            start_idx = len(ohlcv_data) - display_count
            
            for i in range(start_idx, len(ohlcv_data)):
                timestamp = timestamps[i]
                close = closes[i]
                supertrend = supertrend_values[i]
                current_trend = trend_directions[i]
                
                if supertrend is None:
                    continue
                
                # Get signal
                previous_trend = trend_directions[i-1] if i > 0 else "UNKNOWN"
                signal, status = self.get_signal(current_trend, previous_trend, close, supertrend)
                
                # Format output
                trend_icon = "📈" if current_trend == "BULLISH" else "📉" if current_trend == "BEARISH" else "❓"
                signal_icon = "🟢 BUY" if signal == "BUY" else "🔴 SELL" if signal == "SELL" else "🟡 HOLD"
                
                # Format trend emoji
                trend_emoji = "📈" if current_trend == "BULLISH" else "📉" if current_trend == "BEARISH" else "➖"
                
                print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"Price: {close:,.2f} | Supertrend: {supertrend:,.2f} | "
                      f"Signal: {signal} | {trend_emoji} {current_trend}")
            
        except Exception as e:
            print(f"❌ Error in Supertrend analysis: {e}")
            import traceback
            traceback.print_exc()


def parse_command(command: str) -> Tuple[str, str, int, int, float]:
    """
    Parse command line arguments for Supertrend analysis.
    
    Args:
        command: Command string (e.g., "supertrend s=BTC/USDT t=1h l=100 p=10 m=3.0")
        
    Returns:
        Tuple of (symbol, timeframe, limit, atr_period, multiplier)
    """
    # Default values
    symbol = "BTC/USDT"
    timeframe = "1h"
    limit = 100
    atr_period = 10
    multiplier = 3.0
    
    # Parse arguments
    parts = command.split()
    for part in parts[1:]:  # Skip the command name
        if '=' in part:
            key, value = part.split('=', 1)
            if key == 's':
                symbol = value
            elif key == 't':
                timeframe = value
            elif key == 'l':
                limit = int(value)
            elif key == 'p':
                atr_period = int(value)
            elif key == 'm':
                multiplier = float(value)
    
    return symbol, timeframe, limit, atr_period, multiplier


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Parse command line arguments
        full_command = ' '.join(sys.argv)
        symbol, timeframe, limit, atr_period, multiplier = parse_command(full_command)
        
        # Run analysis
        strategy = SupertrendStrategy()
        strategy.analyze(symbol, timeframe, limit, atr_period, multiplier)
    else:
        # Default analysis
        strategy = SupertrendStrategy()
        strategy.analyze()