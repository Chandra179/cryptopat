"""
ATR (Average True Range) + ADX (Average Directional Index) strategy implementation.
Measures volatility and trend strength for cryptocurrency trend analysis.
"""

import sys
from datetime import datetime
from typing import List, Tuple, Dict
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
                        closes: List[float], timestamps: List[int]) -> List[Dict]:
        """
        Generate ATR+ADX trading signals.
        
        Args:
            atr_values: ATR values
            adx_values: ADX values
            plus_di: +DI values
            minus_di: -DI values
            closes: Closing prices
            timestamps: Timestamp values
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        # Ensure we have valid data arrays
        min_length = min(len(adx_values), len(atr_values), len(plus_di), len(minus_di))
        
        for i in range(1, min_length):
            # Calculate indices for original data arrays, accounting for ATR/ADX offset
            data_idx = len(closes) - len(adx_values) + i
            
            # Ensure data_idx is within bounds
            if data_idx >= len(closes) or data_idx >= len(timestamps):
                continue
                
            signal = {
                'timestamp': timestamps[data_idx],
                'close': closes[data_idx],
                'atr': atr_values[min(i, len(atr_values)-1)],
                'adx': adx_values[i],
                'plus_di': plus_di[min(i, len(plus_di)-1)],
                'minus_di': minus_di[min(i, len(minus_di)-1)],
                'signal': 'HOLD',
                'trend_strength': 'WEAK',
                'description': ''
            }
            
            # Trend strength based on ADX
            if signal['adx'] > 25:
                signal['trend_strength'] = 'STRONG'
            elif signal['adx'] > 20:
                signal['trend_strength'] = 'MODERATE'
            else:
                signal['trend_strength'] = 'WEAK'
            
            # Generate signals based on DI crossovers and ADX strength
            prev_plus_di = plus_di[max(0, i-1)]
            prev_minus_di = minus_di[max(0, i-1)]
            
            # BUY signal: ADX > 25 and +DI crosses above -DI
            if (signal['adx'] > 25 and 
                prev_plus_di <= prev_minus_di and 
                signal['plus_di'] > signal['minus_di']):
                signal['signal'] = 'BUY'
                signal['description'] = 'Strong Trend Confirmed'
            
            # SELL signal: ADX > 25 and -DI crosses above +DI
            elif (signal['adx'] > 25 and 
                  prev_plus_di >= prev_minus_di and 
                  signal['plus_di'] < signal['minus_di']):
                signal['signal'] = 'SELL'
                signal['description'] = 'Strong Trend Confirmed'
            
            # Weak trend warning
            elif signal['adx'] < 25:
                signal['description'] = 'Weak Trend - Avoid New Positions'
            
            else:
                signal['description'] = 'No Clear Signal'
            
            signals.append(signal)
        
        return signals
    
    def analyze(self, symbol: str, timeframe: str, limit: int) -> None:
        """
        Perform ATR+ADX analysis and display results.
        
        Args:
            symbol: Trading pair (e.g., 'ETH/USDT')
            timeframe: Timeframe (e.g., '4h', '1d')
            limit: Number of candles to analyze
        """
        # Fetch OHLCV data
        ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 30:
            print(f"Error: Need at least 30 candles for ATR+ADX calculation. Got {len(ohlcv_data)}")
            return
        
        # Extract OHLCV components
        timestamps = [candle[0] for candle in ohlcv_data]
        opens = [candle[1] for candle in ohlcv_data]
        highs = [candle[2] for candle in ohlcv_data]
        lows = [candle[3] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        
        # Calculate ATR
        true_ranges = self.calculate_true_range(highs, lows, closes)
        atr_values = self.calculate_atr(true_ranges, 14)
        
        # Calculate ADX, +DI, -DI
        adx_values, plus_di, minus_di = self.calculate_adx(highs, lows, closes, 14)
        
        if not atr_values or not adx_values:
            print("Error: Unable to calculate ATR+ADX indicators")
            return
        
        # Generate signals
        signals = self.generate_signals(atr_values, adx_values, plus_di, minus_di, closes, timestamps)
        
        # Filter for today's signals
        today = datetime.now().date()
        today_signals = []
        
        for signal in signals:
            dt = datetime.fromtimestamp(signal['timestamp'] / 1000)
            if dt.date() == today:
                today_signals.append((signal, dt))
        
        if not today_signals:
            print(f"No signals found for today ({today})")
            # Show latest signal for reference
            if signals:
                latest_signal = signals[-1]
                dt = datetime.fromtimestamp(latest_signal['timestamp'] / 1000)
                signal_icon = self._get_signal_icon(latest_signal['signal'])
                trend_icon = self._get_trend_icon(latest_signal['trend_strength'])
                
                print(f"\nLatest signal:")
                # Get appropriate trend emoji for latest signal
                if latest_signal['signal'] == 'BUY':
                    trend_emoji = '📈'
                elif latest_signal['signal'] == 'SELL':
                    trend_emoji = '📉'
                else:
                    trend_emoji = '➖'
                
                print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"ATR: {latest_signal['atr']:.0f} | "
                      f"ADX: {latest_signal['adx']:.0f} | "
                      f"+DI: {latest_signal['plus_di']:.0f} | "
                      f"-DI: {latest_signal['minus_di']:.0f} | "
                      f"Signal: {latest_signal['signal']} | "
                      f"{trend_emoji} {latest_signal['description']}")
        else:
            for signal, dt in today_signals:
                # Get appropriate trend emoji
                if signal['signal'] == 'BUY':
                    trend_emoji = '📈'
                elif signal['signal'] == 'SELL':
                    trend_emoji = '📉'
                else:
                    trend_emoji = '➖'
                
                print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"ATR: {signal['atr']:.0f} | "
                      f"ADX: {signal['adx']:.0f} | "
                      f"+DI: {signal['plus_di']:.0f} | "
                      f"-DI: {signal['minus_di']:.0f} | "
                      f"Signal: {signal['signal']} | "
                      f"{trend_emoji} {signal['description']}")
    
    def _get_signal_icon(self, signal: str) -> str:
        """Get icon for signal type."""
        icons = {
            'BUY': '⬆️',
            'SELL': '⬇️',
            'HOLD': '➖'
        }
        return icons.get(signal, '❓')
    
    def _get_trend_icon(self, trend_strength: str) -> str:
        """Get icon for trend strength."""
        icons = {
            'STRONG': '📊',
            'MODERATE': '📈',
            'WEAK': '⚠️'
        }
        return icons.get(trend_strength, '❓')


def parse_command(command: str) -> Tuple[str, str, int]:
    """
    Parse terminal command: atr_adx s=ETH/USDT t=4h l=14
    
    Args:
        command: Command string
        
    Returns:
        Tuple of (symbol, timeframe, limit)
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'atr_adx':
        raise ValueError("Invalid command format. Use: atr_adx s=SYMBOL t=TIMEFRAME l=LIMIT")
    
    symbol = None
    timeframe = '4h'  # default
    limit = 14  # default
    
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
    """Main entry point for ATR+ADX strategy."""
    if len(sys.argv) < 2:
        print("Usage: python atr_adx.py s=SYMBOL t=TIMEFRAME l=LIMIT")
        print("Example: python atr_adx.py s=ETH/USDT t=4h l=14")
        return
    
    try:
        # Parse command line arguments
        command = ' '.join(['atr_adx'] + sys.argv[1:])
        symbol, timeframe, limit = parse_command(command)
        
        # Run analysis
        strategy = ATR_ADXStrategy()
        strategy.analyze(symbol, timeframe, limit)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()