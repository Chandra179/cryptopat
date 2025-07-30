"""
MACD (Moving Average Convergence Divergence) implementation for cryptocurrency trend analysis.
Uses MACD line, signal line, and histogram to detect trend changes and momentum shifts.
"""

import sys
from datetime import datetime
from typing import List, Tuple
from data import get_data_collector


class MACDStrategy:
    """MACD strategy for trend detection and momentum analysis."""
    
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
        
        # EMA multiplier
        multiplier = 2 / (period + 1)
        
        # Initialize with SMA for first value
        sma = sum(prices[:period]) / period
        ema_values = [sma]
        
        # Calculate EMA for remaining values
        for i in range(period, len(prices)):
            ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
        
        return ema_values
    
    def calculate_macd(self, prices: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate MACD line, signal line, and histogram using standard formula.
        
        Args:
            prices: List of closing prices
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        if len(prices) < 34:  # Need 26 for EMA26 + 9 for signal = 35 minimum
            return [], [], []
        
        # Calculate EMA(12) and EMA(26)
        ema12 = self.calculate_ema(prices, 12)
        ema26 = self.calculate_ema(prices, 26)
        
        if not ema12 or not ema26:
            return [], [], []
        
        # MACD Line = EMA(12) - EMA(26)
        # Standard alignment: both EMAs calculated from same price data
        # EMA26 determines the start since it needs more data (26 vs 12)
        ema26_start_idx = 25  # EMA26 starts at index 25 (26th price)
        ema12_offset = ema26_start_idx - 11  # EMA12 starts at index 11, so offset is 14
        
        macd_line = []
        for i in range(len(ema26)):
            macd_value = ema12[i + ema12_offset] - ema26[i]
            macd_line.append(macd_value)
        
        # Signal Line = EMA(9) of MACD Line
        signal_line = self.calculate_ema(macd_line, 9)
        
        # Histogram = MACD - Signal (properly aligned)
        histogram = []
        if signal_line:
            # Signal line starts 8 positions after MACD line starts
            signal_start_offset = 8
            for i in range(len(signal_line)):
                hist_value = macd_line[i + signal_start_offset] - signal_line[i]
                histogram.append(hist_value)
        
        return macd_line, signal_line, histogram
    
    def detect_crossovers(self, macd_line: List[float], signal_line: List[float], 
                         histogram: List[float], closes: List[float]) -> List[dict]:
        """
        Detect MACD crossovers and generate signals.
        
        Args:
            macd_line: MACD line values
            signal_line: Signal line values
            histogram: Histogram values
            closes: Closing prices
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        if len(signal_line) < 2 or len(histogram) < 2:
            return signals
        
        # Calculate base offset: EMA26 starts at index 25, signal starts 8 after MACD
        base_offset = 25 + 8  # 33 total offset from original prices
        
        for i in range(1, len(signal_line)):
            # Align with original price data
            close_idx = i + base_offset
            macd_idx = i + 8  # Signal aligns to MACD with 8 offset
            
            # Safety checks
            if (close_idx >= len(closes) or 
                macd_idx >= len(macd_line) or 
                i >= len(histogram)):
                break
            
            signal = {
                'index': i,
                'close': closes[close_idx],
                'macd': macd_line[macd_idx],
                'signal_line': signal_line[i],
                'histogram': histogram[i],
                'signal': 'NONE',
                'trend': 'NEUTRAL',
                'momentum': 'WEAK'
            }
            
            # Previous values for crossover detection
            prev_macd = macd_line[macd_idx - 1] if macd_idx > 0 else macd_line[macd_idx]
            prev_signal = signal_line[i - 1]
            
            # Bullish crossover: MACD crosses above Signal
            if prev_macd <= prev_signal and signal['macd'] > signal['signal_line']:
                signal['signal'] = 'BUY'
                signal['trend'] = 'BULLISH'
                
                # Strong momentum if histogram is growing
                if i > 0 and histogram[i] > histogram[i-1]:
                    signal['momentum'] = 'STRONG'
                else:
                    signal['momentum'] = 'CONFIRMING'
            
            # Bearish crossover: MACD crosses below Signal
            elif prev_macd >= prev_signal and signal['macd'] < signal['signal_line']:
                signal['signal'] = 'SELL'
                signal['trend'] = 'BEARISH'
                
                # Strong momentum if histogram is growing (more negative)
                if i > 0 and histogram[i] < histogram[i-1]:
                    signal['momentum'] = 'STRONG'
                else:
                    signal['momentum'] = 'CONFIRMING'
            
            # No crossover but check momentum
            else:
                if signal['macd'] > signal['signal_line']:
                    signal['trend'] = 'BULLISH'
                    if histogram[i] > 0:
                        signal['momentum'] = 'UPTREND'
                elif signal['macd'] < signal['signal_line']:
                    signal['trend'] = 'BEARISH' 
                    if histogram[i] < 0:
                        signal['momentum'] = 'DOWNTREND'
            
            signals.append(signal)
        
        return signals
    
    def analyze(self, symbol: str, timeframe: str, limit: int) -> None:
        """
        Perform MACD analysis and display results.
        
        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            timeframe: Timeframe (e.g., '4h', '1d', '1h')
            limit: Number of candles to analyze
        """
        # Fetch OHLCV data
        ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 50:
            print(f"Error: Need at least 50 candles for MACD calculation. Got {len(ohlcv_data)}")
            return
        
        # Extract closes and timestamps
        timestamps = [candle[0] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        
        # Calculate MACD
        macd_line, signal_line, histogram = self.calculate_macd(closes)
        
        if not macd_line or not signal_line or not histogram:
            print("Error: Unable to calculate MACD")
            return
        
        
        # Detect signals
        signals = self.detect_crossovers(macd_line, signal_line, histogram, closes)
        
        today = datetime.now().date()
        today_signals = []
        
        for signal in signals:
            close_idx = signal['index'] + 33  # Updated total offset
            if close_idx < len(timestamps):
                dt = datetime.fromtimestamp(timestamps[close_idx] / 1000)
                if dt.date() == today:
                    today_signals.append((signal, dt))
        
        if not today_signals:
            print(f"No signals found for today ({today})")
            # Show latest signal for reference
            if signals:
                latest_signal = signals[-1]
                close_idx = latest_signal['index'] + 33
                if close_idx < len(timestamps):
                    dt = datetime.fromtimestamp(timestamps[close_idx] / 1000)
                    
                    print(f"\nLatest signal:")
                    trend_emoji = "📈" if latest_signal['trend'] == 'BULLISH' else "📉" if latest_signal['trend'] == 'BEARISH' else "➖"
                    print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                          f"MACD: {latest_signal['macd']:.6f} | "
                          f"SIGNAL: {latest_signal['signal_line']:.6f} | "
                          f"HIST: {latest_signal['histogram']:.6f} | "
                          f"Signal: {latest_signal['signal']} | "
                          f"{trend_emoji} {latest_signal['trend']}")
        else:
            for signal, dt in today_signals:
                
                trend_emoji = "📈" if signal['trend'] == 'BULLISH' else "📉" if signal['trend'] == 'BEARISH' else "➖"
                print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"MACD: {signal['macd']:.6f} | "
                      f"SIGNAL: {signal['signal_line']:.6f} | "
                      f"HIST: {signal['histogram']:.6f} | "
                      f"Signal: {signal['signal']} | "
                      f"{trend_emoji} {signal['trend']}")
    
    def _get_momentum_icon(self, momentum: str) -> str:
        """Get emoji icon for momentum type."""
        momentum_icons = {
            'STRONG': '🔥',
            'CONFIRMING': '🔄',
            'UPTREND': '📈',
            'DOWNTREND': '📉',
            'WEAK': '🧨'
        }
        return momentum_icons.get(momentum, '➖')


def parse_command(command: str) -> Tuple[str, str, int]:
    """
    Parse terminal command: macd s=XRP/USDT t=4h l=100
    
    Args:
        command: Command string
        
    Returns:
        Tuple of (symbol, timeframe, limit)
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'macd':
        raise ValueError("Invalid command format. Use: macd s=SYMBOL t=TIMEFRAME l=LIMIT")
    
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
    """Main entry point for MACD strategy."""
    if len(sys.argv) < 2:
        print("Usage: python macd.py s=SYMBOL t=TIMEFRAME l=LIMIT")
        print("Example: python macd.py s=XRP/USDT t=4h l=100")
        return
    
    try:
        # Parse command line arguments
        command = ' '.join(['macd'] + sys.argv[1:])
        symbol, timeframe, limit = parse_command(command)
        
        # Run analysis
        strategy = MACDStrategy()
        strategy.analyze(symbol, timeframe, limit)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()