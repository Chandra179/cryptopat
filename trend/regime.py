"""
Market Regime Detection (Trend vs Range) strategy implementation.
Classifies market conditions as trending or ranging using ATR, ADX, EMA slope, and Bollinger Band width.
"""

import sys
import math
from datetime import datetime
from typing import List, Tuple, Dict
from data import get_data_collector


class MarketRegimeStrategy:
    """Market Regime Detection for trend vs range classification."""
    
    def __init__(self):
        self.collector = get_data_collector()
    
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: List of closing prices
            period: EMA period
            
        Returns:
            List of EMA values
        """
        if len(prices) < period:
            return []
        
        multiplier = 2 / (period + 1)
        sma = sum(prices[:period]) / period
        ema_values = [sma]
        
        for i in range(period, len(prices)):
            ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
        
        return ema_values
    
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
                true_ranges.append(highs[i] - lows[i])
            else:
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
        first_atr = sum(true_ranges[:period]) / period
        atr_values.append(first_atr)
        
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
        first_smooth = sum(values[:period]) / period
        smoothed.append(first_smooth)
        
        for i in range(period, len(values)):
            smooth = ((smoothed[-1] * (period - 1)) + values[i]) / period
            smoothed.append(smooth)
        
        return smoothed
    
    def calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], 
                     period: int = 14) -> List[float]:
        """
        Calculate ADX (Average Directional Index).
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            period: ADX period (default 14)
            
        Returns:
            List of ADX values
        """
        true_ranges = self.calculate_true_range(highs, lows, closes)
        atr_values = self.calculate_atr(true_ranges, period)
        
        plus_dm, minus_dm = self.calculate_directional_movement(highs, lows)
        plus_dm_smooth = self.smooth_values(plus_dm, period)
        minus_dm_smooth = self.smooth_values(minus_dm, period)
        
        plus_di = []
        minus_di = []
        
        min_length = min(len(atr_values), len(plus_dm_smooth), len(minus_dm_smooth))
        
        for i in range(min_length):
            if atr_values[i] != 0:
                plus_di.append((plus_dm_smooth[i] / atr_values[i]) * 100)
                minus_di.append((minus_dm_smooth[i] / atr_values[i]) * 100)
            else:
                plus_di.append(0)
                minus_di.append(0)
        
        dx_values = []
        for i in range(len(plus_di)):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum != 0:
                dx = abs(plus_di[i] - minus_di[i]) / di_sum * 100
                dx_values.append(dx)
            else:
                dx_values.append(0)
        
        adx_values = self.smooth_values(dx_values, period)
        return adx_values
    
    def calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: List of closing prices
            period: SMA period
            
        Returns:
            List of SMA values
        """
        if len(prices) < period:
            return []
        
        sma_values = []
        for i in range(period - 1, len(prices)):
            sma = sum(prices[i - period + 1:i + 1]) / period
            sma_values.append(sma)
        
        return sma_values
    
    def calculate_standard_deviation(self, prices: List[float], period: int, sma_values: List[float]) -> List[float]:
        """
        Calculate standard deviation for Bollinger Bands.
        
        Args:
            prices: List of closing prices
            period: Period for calculation
            sma_values: List of SMA values
            
        Returns:
            List of standard deviation values
        """
        if len(prices) < period or len(sma_values) == 0:
            return []
        
        std_values = []
        for i in range(len(sma_values)):
            price_idx = i + period - 1
            mean = sma_values[i]
            
            variance = sum((prices[price_idx - j] - mean) ** 2 for j in range(period)) / period
            std_dev = math.sqrt(variance)
            std_values.append(std_dev)
        
        return std_values
    
    def calculate_bb_width(self, prices: List[float], period: int = 20, multiplier: float = 2.0) -> List[float]:
        """
        Calculate standard Bollinger Band width.
        Formula: (Upper Band - Lower Band) / Middle Band
        
        Args:
            prices: List of closing prices
            period: BB period (default 20)
            multiplier: BB multiplier (default 2.0)
            
        Returns:
            List of BB width values (normalized by middle band)
        """
        sma_values = self.calculate_sma(prices, period)
        if not sma_values:
            return []
        
        std_values = self.calculate_standard_deviation(prices, period, sma_values)
        if not std_values:
            return []
        
        # Standard formula: (Upper - Lower) / Middle = (2 * multiplier * std) / sma
        bb_width = [(2 * multiplier * std) / sma for std, sma in zip(std_values, sma_values)]
        return bb_width
    
    def calculate_ema_slope(self, ema_values: List[float], lookback: int = 5) -> List[float]:
        """
        Calculate EMA slope to determine trend direction strength.
        
        Args:
            ema_values: List of EMA values
            lookback: Periods to look back for slope calculation
            
        Returns:
            List of slope values (positive = upward, negative = downward)
        """
        if len(ema_values) < lookback:
            return []
        
        slopes = []
        for i in range(lookback, len(ema_values)):
            # Simple slope calculation over lookback period
            slope = (ema_values[i] - ema_values[i - lookback]) / lookback
            slopes.append(slope)
        
        return slopes
    
    def classify_regime(self, adx: float, ema9: float, ema21: float, atr_pct: float, 
                       bb_width: float, bb_avg: float) -> Tuple[str, str, int]:
        """
        Classify market regime using standard Wilder's ADX methodology with volatility filters.
        Based on Wilder's original ADX interpretation and modern volatility analysis.
        
        Args:
            adx: ADX value (0-100)
            ema9: EMA 9 value
            ema21: EMA 21 value
            atr_pct: ATR as percentage of close
            bb_width: Bollinger Band width (normalized)
            bb_avg: Average BB width over period
            
        Returns:
            Tuple of (regime, direction, adx_value)
        """
        # Standard Wilder's ADX thresholds
        # ADX > 40: Very strong trend
        # ADX 25-40: Strong trend
        # ADX 20-25: Weak trend
        # ADX < 20: No trend (ranging)
        
        # Determine trend direction
        direction = "↑" if ema9 > ema21 else "↓"
        
        # Volatility filter using BB width
        low_volatility = bb_width < bb_avg * 0.75
        high_volatility = bb_width > bb_avg * 1.25
        
        # Standard ADX-based regime classification
        if adx >= 40:
            regime = "VERY_STRONG_TREND"
        elif adx >= 25:
            # Strong trend, but check for low volatility (potential false signals)
            if low_volatility and atr_pct < 1.0:
                regime = "WEAK_TREND"  # ADX high but low volatility = weakening trend
            else:
                regime = "STRONG_TREND"
        elif adx >= 20:
            regime = "WEAK_TREND"
        else:
            # ADX < 20: Ranging market
            if high_volatility:
                regime = "VOLATILE_RANGE"  # High volatility but no trend
            else:
                regime = "RANGING"
            direction = "⏸️"  # No clear direction in ranging market
        
        return regime, direction, int(adx)
    
    def analyze(self, symbol: str, timeframe: str, limit: int) -> None:
        """
        Perform market regime analysis and display results.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '4h', '1d')
            limit: Number of candles to analyze
        """
        # Fetch OHLCV data
        ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 100:
            print(f"Error: Need at least 100 candles for regime analysis. Got {len(ohlcv_data)}")
            return
        
        # Extract OHLCV components
        timestamps = [candle[0] for candle in ohlcv_data]
        highs = [candle[2] for candle in ohlcv_data]
        lows = [candle[3] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        
        # Calculate indicators
        ema9 = self.calculate_ema(closes, 9)
        ema21 = self.calculate_ema(closes, 21)
        true_ranges = self.calculate_true_range(highs, lows, closes)
        atr_values = self.calculate_atr(true_ranges, 14)
        adx_values = self.calculate_adx(highs, lows, closes, 14)
        bb_width = self.calculate_bb_width(closes, 20, 2.0)
        
        if not all([ema9, ema21, atr_values, adx_values, bb_width]):
            print("Error: Unable to calculate required indicators")
            return
        
        # Calculate average BB width for comparison
        bb_avg = sum(bb_width[-20:]) / min(20, len(bb_width)) if bb_width else 0
        
        # Generate regime classifications
        signals = []
        today = datetime.now().date()
        
        # Align all indicators to the same length (use the shortest)
        # All indicators should end at the same time point
        min_len = min(len(adx_values), len(atr_values), len(bb_width), len(ema9), len(ema21))
        
        # Start from the latest data points working backwards
        for i in range(min_len):
            # Use indices from the end of each array
            adx_idx = len(adx_values) - min_len + i
            atr_idx = len(atr_values) - min_len + i  
            bb_idx = len(bb_width) - min_len + i
            ema9_idx = len(ema9) - min_len + i
            ema21_idx = len(ema21) - min_len + i
            close_idx = len(closes) - min_len + i
            
            # Calculate ATR percentage
            atr_pct = (atr_values[atr_idx] / closes[close_idx]) * 100
            
            # Classify regime
            regime, direction, adx_strength = self.classify_regime(
                adx_values[adx_idx], ema9[ema9_idx], ema21[ema21_idx], 
                atr_pct, bb_width[bb_idx], bb_avg
            )
            
            signal = {
                'timestamp': timestamps[close_idx],
                'close': closes[close_idx],
                'adx': adx_values[adx_idx],
                'atr_pct': atr_pct,
                'ema9': ema9[ema9_idx],
                'ema21': ema21[ema21_idx],
                'bb_width': bb_width[bb_idx],
                'regime': regime,
                'direction': direction,
                'strength': adx_strength
            }
            
            # Check if signal is from today
            dt = datetime.fromtimestamp(signal['timestamp'] / 1000)
            if dt.date() == today:
                signals.append((signal, dt))
        
        # Display results
        if not signals:
            print(f"No signals found for today ({today})")
            # Show latest available signal
            if min_len > 0:
                # Get the last available data point (all aligned)
                last_idx = -1
                close_idx = len(closes) + last_idx
                
                atr_pct = (atr_values[last_idx] / closes[close_idx]) * 100
                regime, direction, adx_strength = self.classify_regime(
                    adx_values[last_idx], ema9[last_idx], ema21[last_idx], 
                    atr_pct, bb_width[last_idx], bb_avg
                )
                
                dt = datetime.fromtimestamp(timestamps[close_idx] / 1000)
                print(f"\nLatest regime analysis:")
                self._print_regime_signal(
                    adx_values[last_idx], atr_pct, ema9[last_idx] > ema21[last_idx],
                    bb_width[last_idx] > bb_avg, regime, direction, adx_strength, dt
                )
        else:
            for signal, dt in signals:
                bb_expanding = signal['bb_width'] > bb_avg
                
                self._print_regime_signal(
                    signal['adx'], signal['atr_pct'], signal['ema9'] > signal['ema21'],
                    bb_expanding, signal['regime'], signal['direction'], signal['strength'], dt
                )
    
    def _print_regime_signal(self, adx: float, atr_pct: float, ema_positive: bool, 
                           bb_expanding: bool, regime: str, direction: str, adx_strength: int, dt: datetime) -> None:
        """Print formatted regime signal output using standard ADX interpretation."""
        ema_trend = "Bullish" if ema_positive else "Bearish"
        bb_status = "Expanding" if bb_expanding else "Contracting"
        
        # Standard regime descriptions based on Wilder's ADX
        regime_map = {
            "VERY_STRONG_TREND": ("🔥", "Very Strong Trend"),
            "STRONG_TREND": ("📈" if ema_positive else "📉", "Strong Trend"),
            "WEAK_TREND": ("📋", "Weak Trend"),
            "RANGING": ("⏸️", "Sideways/Range"),
            "VOLATILE_RANGE": ("🌊", "Volatile Range")
        }
        
        trend_emoji, _ = regime_map.get(regime, ("❓", "Unknown"))
        
        print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
              f"ADX: {adx:.1f} | "
              f"ATR: {atr_pct:.1f}% | "
              f"EMA: {ema_trend} | "
              f"Volatility: {bb_status} | "
              f"{trend_emoji} {regime.replace('_', ' ')} {direction} | "
              f"ADX Score: {adx_strength}")


def parse_command(command: str) -> Tuple[str, str, int]:
    """
    Parse terminal command: regime s=BTC/USDT t=4h l=100
    
    Args:
        command: Command string
        
    Returns:
        Tuple of (symbol, timeframe, limit)
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'regime':
        raise ValueError("Invalid command format. Use: regime s=SYMBOL t=TIMEFRAME l=LIMIT")
    
    symbol = None
    timeframe = '4h'  # default
    limit = 100  # default
    
    for part in parts[1:]:
        if part.startswith('s='):
            symbol = part[2:]
        elif part.startswith('t='):
            timeframe = part[2:]
        elif part.startswith('l='):
            try:
                limit = int(part[2:])
            except ValueError:
                raise ValueError(f"Invalid limit value: {part[2:]}")
    
    if symbol is None:
        raise ValueError("Symbol (s=) is required")
    
    return symbol, timeframe, limit


def main():
    """Main entry point for Market Regime strategy."""
    if len(sys.argv) < 2:
        print("Usage: python regime.py s=SYMBOL t=TIMEFRAME l=LIMIT")
        print("Example: python regime.py s=BTC/USDT t=4h l=100")
        return
    
    try:
        # Parse command line arguments
        command = ' '.join(['regime'] + sys.argv[1:])
        symbol, timeframe, limit = parse_command(command)
        
        # Run analysis
        strategy = MarketRegimeStrategy()
        strategy.analyze(symbol, timeframe, limit)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()