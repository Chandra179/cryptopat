"""
EMA 9/21 crossover strategy implementation for cryptocurrency trend analysis.
Uses exponential moving averages to detect bullish and bearish trend signals.
"""

import sys
from datetime import datetime
from typing import List, Tuple
from data.collector import DataCollector


class EMA9_21Strategy:
    """EMA 9/21 crossover strategy for trend detection."""
    
    def __init__(self):
        self.collector = DataCollector()
    
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: List of closing prices
            period: EMA period (9 or 21)
            
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
            ema9: EMA 9 values
            ema21: EMA 21 values  
            closes: Closing prices
            volumes: Volume values
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        volume_spikes = self.detect_volume_spike(volumes)
        
        for i in range(1, len(ema21)):  # Use ema21 length since it's shorter
            close_idx = i + 20  # EMA21 starts at index 20 (21st element)
            signal = {
                'index': i,
                'close': closes[close_idx],
                'ema9': ema9[i + 12],  # EMA9 starts at index 8, offset by 12 more
                'ema21': ema21[i],
                'volume_spike': volume_spikes[close_idx] if close_idx < len(volume_spikes) else False,
                'signal': 'NONE',
                'trend': 'NEUTRAL',
                'confirmed': False
            }
            
            # Previous values
            prev_ema9 = ema9[i + 11]  # Previous EMA9 value
            prev_ema21 = ema21[i-1]
            
            # Bullish crossover: EMA9 crosses above EMA21
            if prev_ema9 <= prev_ema21 and signal['ema9'] > ema21[i]:
                if closes[close_idx] > signal['ema9'] and closes[close_idx] > ema21[i]:
                    signal['signal'] = 'BUY'
                    signal['trend'] = 'BULLISH'
                    signal['confirmed'] = volume_spikes[close_idx] if close_idx < len(volume_spikes) else False
            
            # Bearish crossover: EMA9 crosses below EMA21  
            elif prev_ema9 >= prev_ema21 and signal['ema9'] < ema21[i]:
                if closes[close_idx] < signal['ema9'] and closes[close_idx] < ema21[i]:
                    signal['signal'] = 'SELL'
                    signal['trend'] = 'BEARISH'
                    signal['confirmed'] = volume_spikes[close_idx] if close_idx < len(volume_spikes) else False
            
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
        
        # Display results - TODAY ONLY
        print(f"\nEMA 9/21 Analysis for {symbol} ({timeframe}) - TODAY ONLY")
        print("=" * 80)
        
        today = datetime.now().date()
        today_signals = []
        
        for signal in signals:
            timestamp_idx = signal['index'] + 20
            if timestamp_idx < len(timestamps):
                dt = datetime.fromtimestamp(timestamps[timestamp_idx] / 1000)
                if dt.date() == today:
                    today_signals.append((signal, dt))
        
        if not today_signals:
            print(f"No signals found for today ({today})")
            # Show latest signal for reference
            if signals:
                latest_signal = signals[-1]
                timestamp_idx = latest_signal['index'] + 20
                if timestamp_idx < len(timestamps):
                    dt = datetime.fromtimestamp(timestamps[timestamp_idx] / 1000)
                    signal_icon = "⬆️" if latest_signal['signal'] == 'BUY' else "⬇️" if latest_signal['signal'] == 'SELL' else "➖"
                    confirmed_icon = "✔️" if latest_signal['confirmed'] else "⏳"
                    
                    print(f"\nLatest signal:")
                    print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                          f"CLOSE: {latest_signal['close']:.4f} | "
                          f"EMA9: {latest_signal['ema9']:.4f} | "
                          f"EMA21: {latest_signal['ema21']:.4f} | "
                          f"{signal_icon} {latest_signal['signal']} | "
                          f"Trend: {latest_signal['trend']} | "
                          f"{confirmed_icon} {'Confirmed' if latest_signal['confirmed'] else 'Waiting'}")
        else:
            for signal, dt in today_signals:
                # Format signal indicators
                signal_icon = "⬆️" if signal['signal'] == 'BUY' else "⬇️" if signal['signal'] == 'SELL' else "➖"
                confirmed_icon = "✔️" if signal['confirmed'] else "⏳"
                
                print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"CLOSE: {signal['close']:.4f} | "
                      f"EMA9: {signal['ema9']:.4f} | "
                      f"EMA21: {signal['ema21']:.4f} | "
                      f"{signal_icon} {signal['signal']} | "
                      f"Trend: {signal['trend']} | "
                      f"{confirmed_icon} {'Confirmed' if signal['confirmed'] else 'Waiting'}")


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