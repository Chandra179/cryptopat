"""
EMA 9/21 crossover strategy implementation for cryptocurrency trend analysis.
Uses exponential moving averages to detect bullish and bearish trend signals.
"""

import sys
from datetime import datetime
from typing import List, Tuple
import pandas as pd
from data import get_data_collector


class EMA9_21Strategy:
    """EMA 9/21 crossover strategy for trend detection."""
    
    def __init__(self):
        self.collector = get_data_collector()
    
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate Exponential Moving Average using pandas for accuracy.
        
        Args:
            prices: List of closing prices
            period: EMA period (9 or 21)
            
        Returns:
            List of EMA values, same length as prices with NaN for initial values
        """
        if len(prices) < period:
            return [None] * len(prices)
        
        # Use pandas for accurate EMA calculation
        df = pd.DataFrame({'price': prices})
        ema_series = df['price'].ewm(span=period, adjust=False).mean()
        
        # Convert to list and replace NaN with None for consistency
        ema_values = ema_series.tolist()
        return ema_values
    
    def detect_volume_spike(self, volumes: List[float], window: int = 10) -> List[bool]:
        """
        Detect volume spikes by comparing current volume to average.
        
        Args:
            volumes: List of volume values
            window: Lookback window for average calculation
            
        Returns:
            List of boolean values indicating volume spikes
        """
        spikes = []
        
        for i in range(len(volumes)):
            if i < window:
                spikes.append(False)
                continue
                
            # Average volume over lookback window
            avg_volume = sum(volumes[i-window:i]) / window
            current_volume = volumes[i]
            
            # Volume spike if current > 1.5x average
            spike = current_volume > (avg_volume * 1.5)
            spikes.append(spike)
        
        return spikes
    
    def detect_crossovers(self, ema9: List[float], ema21: List[float], 
                         closes: List[float], volumes: List[float]) -> List[dict]:
        """
        Detect EMA crossovers and generate signals.
        
        Args:
            ema9: EMA 9 values (aligned with data indices)
            ema21: EMA 21 values (aligned with data indices)
            closes: Closing prices
            volumes: Volume values
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        volume_spikes = self.detect_volume_spike(volumes)
        
        # Start from index 21 where both EMAs are valid (EMA21 needs 21 data points)
        for i in range(20, len(closes)):  # EMA21 starts at index 20 (21st element)
            # Skip if either EMA has NaN values (pandas may have NaN for early values)
            if pd.isna(ema9[i]) or pd.isna(ema21[i]):
                continue
                
            signal = {
                'data_index': i,
                'close': closes[i],
                'ema9': ema9[i],
                'ema21': ema21[i],
                'volume_spike': volume_spikes[i] if i < len(volume_spikes) else False,
                'signal': 'NONE',
                'trend': 'NEUTRAL',
                'confirmed': False
            }
            
            # Previous values (ensure they exist and are not NaN)
            if i > 20 and not pd.isna(ema9[i-1]) and not pd.isna(ema21[i-1]):
                prev_ema9 = ema9[i-1]
                prev_ema21 = ema21[i-1]
                
                # Bullish crossover: EMA9 crosses above EMA21
                if prev_ema9 <= prev_ema21 and ema9[i] > ema21[i]:
                    if closes[i] > ema9[i] and closes[i] > ema21[i]:
                        signal['signal'] = 'BUY'
                        signal['trend'] = 'BULLISH'
                        signal['confirmed'] = volume_spikes[i] if i < len(volume_spikes) else False
                
                # Bearish crossover: EMA9 crosses below EMA21  
                elif prev_ema9 >= prev_ema21 and ema9[i] < ema21[i]:
                    if closes[i] < ema9[i] and closes[i] < ema21[i]:
                        signal['signal'] = 'SELL'
                        signal['trend'] = 'BEARISH'
                        signal['confirmed'] = volume_spikes[i] if i < len(volume_spikes) else False
            
            signals.append(signal)
        
        return signals
    
    def analyze(self, symbol: str, timeframe: str, limit: int) -> None:
        """
        Perform EMA 9/21 analysis and display results.
        
        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            timeframe: Timeframe (e.g., '1d', '4h', '1h')
            limit: Number of candles to analyze
        """
        # Fetch OHLCV data
        ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 50:
            print(f"Error: Need at least 50 candles for EMA calculation. Got {len(ohlcv_data)}")
            return
        
        # Extract closes and volumes
        timestamps = [candle[0] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        volumes = [candle[5] for candle in ohlcv_data]
        
        # Calculate EMAs
        ema9 = self.calculate_ema(closes, 9)
        ema21 = self.calculate_ema(closes, 21)
        
        if not ema9 or not ema21:
            print("Error: Unable to calculate EMAs")
            return
        
        # Detect signals
        signals = self.detect_crossovers(ema9, ema21, closes, volumes)
        
        today = datetime.now().date()
        today_signals = []
        
        for signal in signals:
            timestamp_idx = signal['data_index']
            if timestamp_idx < len(timestamps):
                dt = datetime.fromtimestamp(timestamps[timestamp_idx] / 1000)
                if dt.date() == today:
                    today_signals.append((signal, dt))
        
        if not today_signals:
            print(f"No signals found for today ({today})")
            # Show latest signal for reference
            if signals:
                latest_signal = signals[-1]
                timestamp_idx = latest_signal['data_index']
                if timestamp_idx < len(timestamps):
                    dt = datetime.fromtimestamp(timestamps[timestamp_idx] / 1000)
                    
                    print(f"\nLatest signal:")
                    trend_emoji = "📈" if latest_signal['trend'] == 'BULLISH' else "📉" if latest_signal['trend'] == 'BEARISH' else "➖"
                    print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                          f"CLOSE: {latest_signal['close']:.4f} | "
                          f"EMA9: {latest_signal['ema9']:.4f} | "
                          f"EMA21: {latest_signal['ema21']:.4f} | "
                          f"Signal: {latest_signal['signal']} | "
                          f"{trend_emoji} {latest_signal['trend']}")
        else:
            for signal, dt in today_signals:
                trend_emoji = "📈" if signal['trend'] == 'BULLISH' else "📉" if signal['trend'] == 'BEARISH' else "➖"
                
                print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"CLOSE: {signal['close']:.4f} | "
                      f"EMA9: {signal['ema9']:.4f} | "
                      f"EMA21: {signal['ema21']:.4f} | "
                      f"Signal: {signal['signal']} | "
                      f"{trend_emoji} {signal['trend']}")


def parse_command(command: str) -> Tuple[str, str, int]:
    """
    Parse terminal command: ema_9_21 s=XRP/USDT t=1d l=30
    
    Args:
        command: Command string
        
    Returns:
        Tuple of (symbol, timeframe, limit)
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'ema_9_21':
        raise ValueError("Invalid command format. Use: ema_9_21 s=SYMBOL t=TIMEFRAME l=LIMIT")
    
    symbol = None
    timeframe = '1d'  # default
    limit = 30  # default
    
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
    """Main entry point for EMA 9/21 strategy."""
    if len(sys.argv) < 2:
        print("Usage: python ema_9_21.py s=SYMBOL t=TIMEFRAME l=LIMIT")
        print("Example: python ema_9_21.py s=XRP/USDT t=1d l=30")
        return
    
    try:
        # Parse command line arguments
        command = ' '.join(['ema_9_21'] + sys.argv[1:])
        symbol, timeframe, limit = parse_command(command)
        
        # Run analysis
        strategy = EMA9_21Strategy()
        strategy.analyze(symbol, timeframe, limit)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()