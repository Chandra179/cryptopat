"""
Bollinger Bands strategy implementation for cryptocurrency trend analysis.
Uses 20-period SMA with 2 standard deviation bands to detect volatility and reversal signals.
"""

import sys
import math
from datetime import datetime
from typing import List, Tuple
from data import get_data_collector


class BollingerBandsStrategy:
    """Bollinger Bands strategy for trend detection and volatility analysis."""
    
    def __init__(self, period: int = 20, multiplier: float = 2.0):
        self.collector = get_data_collector()
        self.period = period
        self.multiplier = multiplier
    
    def calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: List of closing prices
            period: SMA period (default 20)
            
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
            
            # Calculate variance
            variance = sum((prices[price_idx - j] - mean) ** 2 for j in range(period)) / period
            std_dev = math.sqrt(variance)
            std_values.append(std_dev)
        
        return std_values
    
    def calculate_bollinger_bands(self, prices: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate Bollinger Bands (Upper, Middle, Lower).
        
        Args:
            prices: List of closing prices
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        # Calculate middle band (SMA)
        middle_band = self.calculate_sma(prices, self.period)
        
        if not middle_band:
            return [], [], []
        
        # Calculate standard deviation
        std_dev = self.calculate_standard_deviation(prices, self.period, middle_band)
        
        if not std_dev:
            return [], [], []
        
        # Calculate upper and lower bands
        upper_band = [mb + (self.multiplier * sd) for mb, sd in zip(middle_band, std_dev)]
        lower_band = [mb - (self.multiplier * sd) for mb, sd in zip(middle_band, std_dev)]
        
        return upper_band, middle_band, lower_band
    
    def detect_squeeze(self, upper_band: List[float], lower_band: List[float], window: int = 20) -> List[bool]:
        """
        Detect Bollinger Band squeeze (low volatility periods).
        
        Args:
            upper_band: Upper Bollinger Band values
            lower_band: Lower Bollinger Band values
            window: Lookback period for squeeze detection
            
        Returns:
            List of boolean values indicating squeeze periods
        """
        if len(upper_band) != len(lower_band) or len(upper_band) < window:
            return [False] * len(upper_band)
        
        squeezes = []
        
        for i in range(len(upper_band)):
            if i < window:
                squeezes.append(False)
                continue
            
            # Current band width
            current_width = upper_band[i] - lower_band[i]
            
            # Average band width over lookback window
            avg_width = sum(upper_band[j] - lower_band[j] for j in range(i - window, i)) / window
            
            # Squeeze if current width is significantly smaller than average
            squeeze = current_width < (avg_width * 0.7)
            squeezes.append(squeeze)
        
        return squeezes
    
    def generate_signals(self, prices: List[float], upper_band: List[float], 
                        middle_band: List[float], lower_band: List[float]) -> List[dict]:
        """
        Generate Bollinger Bands trading signals.
        
        Args:
            prices: Closing prices (aligned with bands)
            upper_band: Upper Bollinger Band values
            middle_band: Middle Bollinger Band values (SMA)
            lower_band: Lower Bollinger Band values
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        squeezes = self.detect_squeeze(upper_band, lower_band)
        
        # Align prices with bands (bands start at index period-1)
        aligned_prices = prices[self.period - 1:]
        
        for i in range(1, len(middle_band)):
            if i >= len(aligned_prices):
                break
                
            signal = {
                'index': i,
                'price': aligned_prices[i],
                'upper_band': upper_band[i],
                'middle_band': middle_band[i],
                'lower_band': lower_band[i],
                'signal': 'HOLD',
                'squeeze': squeezes[i] if i < len(squeezes) else False,
                'description': 'Neutral'
            }
            
            current_price = aligned_prices[i]
            prev_price = aligned_prices[i - 1]
            
            # BUY signal: Price crosses above lower band (oversold bounce)
            if prev_price <= lower_band[i - 1] and current_price > lower_band[i]:
                signal['signal'] = 'BUY'
                signal['description'] = 'Oversold Bounce'
                if squeezes[i]:
                    signal['description'] = 'Squeeze Breakout'
            
            # SELL signal: Price crosses below upper band (overbought reversal)
            elif prev_price >= upper_band[i - 1] and current_price < upper_band[i]:
                signal['signal'] = 'SELL'
                signal['description'] = 'Overbought Reversal'
            
            # Strong uptrend: Price riding upper band
            elif current_price >= upper_band[i] * 0.98:  # Within 2% of upper band
                signal['signal'] = 'HOLD'
                signal['description'] = 'Strong Uptrend'
                
            # Strong downtrend: Price riding lower band
            elif current_price <= lower_band[i] * 1.02:  # Within 2% of lower band
                signal['signal'] = 'HOLD'
                signal['description'] = 'Strong Downtrend'
            
            # Squeeze condition
            elif squeezes[i]:
                signal['signal'] = 'HOLD'
                signal['description'] = 'Volatility Squeeze'
            
            signals.append(signal)
        
        return signals
    
    def analyze(self, symbol: str, timeframe: str, limit: int) -> None:
        """
        Perform Bollinger Bands analysis and display results.
        
        Args:
            symbol: Trading pair (e.g., 'ETH/USDT')
            timeframe: Timeframe (e.g., '1d', '4h', '1h')
            limit: Number of candles to analyze
        """
        # Fetch OHLCV data
        ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 50:
            print(f"Error: Need at least 50 candles for Bollinger Bands calculation. Got {len(ohlcv_data)}")
            return
        
        # Extract data
        timestamps = [candle[0] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        
        # Calculate Bollinger Bands
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(closes)
        
        if not upper_band or not middle_band or not lower_band:
            print("Error: Unable to calculate Bollinger Bands")
            return
        
        # Generate signals
        signals = self.generate_signals(closes, upper_band, middle_band, lower_band)
        
        # Filter for today's signals
        today = datetime.now().date()
        today_signals = []
        
        for signal in signals:
            timestamp_idx = signal['index'] + self.period - 1
            if timestamp_idx < len(timestamps):
                dt = datetime.fromtimestamp(timestamps[timestamp_idx] / 1000)
                if dt.date() == today:
                    today_signals.append((signal, dt))
        
        if not today_signals:
            print(f"No signals found for today ({today})")
            # Show latest signal for reference
            if signals:
                latest_signal = signals[-1]
                timestamp_idx = latest_signal['index'] + self.period - 1
                if timestamp_idx < len(timestamps):
                    dt = datetime.fromtimestamp(timestamps[timestamp_idx] / 1000)
                    self._print_signal(latest_signal, dt, is_latest=True)
        else:
            for signal, dt in today_signals:
                self._print_signal(signal, dt)
    
    def _print_signal(self, signal: dict, dt: datetime, is_latest: bool = False) -> None:
        """Print formatted signal output."""
        # Format trend emoji based on signal and description
        if signal['signal'] == 'BUY':
            trend_emoji = "📈"
        elif signal['signal'] == 'SELL':
            trend_emoji = "📉"
        elif "Uptrend" in signal['description']:
            trend_emoji = "📈"
        elif "Downtrend" in signal['description']:
            trend_emoji = "📉"
        else:
            trend_emoji = "➖"
        
        prefix = "Latest signal:\n" if is_latest else ""
        
        print(f"{prefix}[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Price: {signal['price']:.2f} | "
              f"Upper Band: {signal['upper_band']:.2f} | "
              f"Middle Band: {signal['middle_band']:.2f} | "
              f"Lower Band: {signal['lower_band']:.2f} | "
              f"Signal: {signal['signal']} | "
              f"{trend_emoji} {signal['description']}")


def parse_command(command: str) -> Tuple[str, str, int]:
    """
    Parse terminal command: bb s=ETH/USDT t=4h l=100
    
    Args:
        command: Command string
        
    Returns:
        Tuple of (symbol, timeframe, limit)
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'bb':
        raise ValueError("Invalid command format. Use: bb s=SYMBOL t=TIMEFRAME l=LIMIT")
    
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
    """Main entry point for Bollinger Bands strategy."""
    if len(sys.argv) < 2:
        print("Usage: python bollinger_bands.py s=SYMBOL t=TIMEFRAME l=LIMIT")
        print("Example: python bollinger_bands.py s=ETH/USDT t=4h l=100")
        return
    
    try:
        # Parse command line arguments
        command = ' '.join(['bb'] + sys.argv[1:])
        symbol, timeframe, limit = parse_command(command)
        
        # Run analysis
        strategy = BollingerBandsStrategy()
        strategy.analyze(symbol, timeframe, limit)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()