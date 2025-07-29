"""
RSI(14) implementation for cryptocurrency trend analysis.
Uses Relative Strength Index to detect overbought/oversold conditions and trend momentum.
"""

import sys
from datetime import datetime
from typing import List, Tuple
from data import get_data_collector


class RSI14Strategy:
    """RSI(14) strategy for momentum and reversal detection."""
    
    def __init__(self):
        self.collector = get_data_collector()
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: List of closing prices
            period: RSI period (default 14)
            
        Returns:
            List of RSI values
        """
        if len(prices) < period + 1:
            return []
        
        # Calculate price changes
        price_changes = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            price_changes.append(change)
        
        if len(price_changes) < period:
            return []
        
        rsi_values = []
        
        # Calculate initial average gains and losses
        gains = [max(0, change) for change in price_changes[:period]]
        losses = [max(0, -change) for change in price_changes[:period]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        # Calculate first RSI value
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
        
        # Calculate subsequent RSI values using smoothed averages
        for i in range(period, len(price_changes)):
            gain = max(0, price_changes[i])
            loss = max(0, -price_changes[i])
            
            # Smoothed averages (Wilder's smoothing)
            avg_gain = ((avg_gain * (period - 1)) + gain) / period
            avg_loss = ((avg_loss * (period - 1)) + loss) / period
            
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)
        
        return rsi_values
    
    def detect_rsi_signals(self, rsi_values: List[float], closes: List[float]) -> List[dict]:
        """
        Detect RSI-based trading signals.
        
        Args:
            rsi_values: List of RSI values
            closes: Corresponding closing prices
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        for i in range(len(rsi_values)):
            close_idx = i + 15  # RSI starts after period+1 price changes
            signal = {
                'index': i,
                'close': closes[close_idx] if close_idx < len(closes) else closes[-1],
                'rsi': rsi_values[i],
                'signal': 'NEUTRAL',
                'condition': 'NORMAL',
                'confirmed': False
            }
            
            # Determine RSI conditions and signals
            if rsi_values[i] > 70:
                signal['condition'] = 'OVERBOUGHT'
                signal['signal'] = 'SELL'
                
                # Check for confirmation (RSI dropping from overbought)
                if i > 0 and rsi_values[i] < rsi_values[i-1]:
                    signal['confirmed'] = True
                    
            elif rsi_values[i] < 30:
                signal['condition'] = 'OVERSOLD'
                signal['signal'] = 'BUY'
                
                # Check for confirmation (RSI rising from oversold)
                if i > 0 and rsi_values[i] > rsi_values[i-1]:
                    signal['confirmed'] = True
                    
            elif rsi_values[i] >= 40 and rsi_values[i] <= 60:
                signal['condition'] = 'SIDEWAYS'
                signal['signal'] = 'NEUTRAL'
                
            else:
                # RSI between 30-40 or 60-70 - transition zones
                if rsi_values[i] > 50:
                    signal['condition'] = 'BULLISH_MOMENTUM'
                    signal['signal'] = 'NEUTRAL'
                else:
                    signal['condition'] = 'BEARISH_MOMENTUM'
                    signal['signal'] = 'NEUTRAL'
            
            signals.append(signal)
        
        return signals
    
    def analyze(self, symbol: str, timeframe: str, limit: int) -> None:
        """
        Perform RSI(14) analysis and display results.
        
        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            timeframe: Timeframe (e.g., '1d', '4h', '1h')
            limit: Number of candles to analyze
        """
        # Fetch OHLCV data
        ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 20:
            print(f"Error: Need at least 20 candles for RSI calculation. Got {len(ohlcv_data)}")
            return
        
        # Extract timestamps and closes
        timestamps = [candle[0] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        
        # Calculate RSI
        rsi_values = self.calculate_rsi(closes, 14)
        
        if not rsi_values:
            print("Error: Unable to calculate RSI")
            return
        
        # Detect signals
        signals = self.detect_rsi_signals(rsi_values, closes)
        
        # Display results - TODAY ONLY
        print(f"\nRSI(14) Analysis for {symbol} ({timeframe}) - TODAY ONLY")
        print("=" * 80)
        
        today = datetime.now().date()
        today_signals = []
        
        for signal in signals:
            timestamp_idx = signal['index'] + 15  # RSI starts after 15 periods
            if timestamp_idx < len(timestamps):
                dt = datetime.fromtimestamp(timestamps[timestamp_idx] / 1000)
                if dt.date() == today:
                    today_signals.append((signal, dt))
        
        if not today_signals:
            print(f"No signals found for today ({today})")
            # Show latest signal for reference
            if signals:
                latest_signal = signals[-1]
                timestamp_idx = latest_signal['index'] + 15
                if timestamp_idx < len(timestamps):
                    dt = datetime.fromtimestamp(timestamps[timestamp_idx] / 1000)
                    self._print_signal(latest_signal, dt, is_latest=True)
        else:
            for signal, dt in today_signals:
                self._print_signal(signal, dt)
    
    def _print_signal(self, signal: dict, dt: datetime, is_latest: bool = False) -> None:
        """
        Print formatted RSI signal.
        
        Args:
            signal: Signal dictionary
            dt: Datetime object
            is_latest: Whether this is the latest signal (for reference)
        """
        # Format condition indicators
        if signal['condition'] == 'OVERBOUGHT':
            condition_icon = "⚠️"
        elif signal['condition'] == 'OVERSOLD':
            condition_icon = "🔽"
        elif signal['condition'] == 'SIDEWAYS':
            condition_icon = "↔️"
        elif signal['condition'] == 'BULLISH_MOMENTUM':
            condition_icon = "📈"
        elif signal['condition'] == 'BEARISH_MOMENTUM':
            condition_icon = "📉"
        else:
            condition_icon = "➖"
        
        # Format signal indicators
        signal_icon = "⬆️" if signal['signal'] == 'BUY' else "⬇️" if signal['signal'] == 'SELL' else "➖"
        confirmed_icon = "✅" if signal['confirmed'] else "⏳"
        
        prefix = "\nLatest signal:" if is_latest else ""
        if prefix:
            print(prefix)
        
        print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
              f"CLOSE: {signal['close']:.4f} | "
              f"RSI(14): {signal['rsi']:.2f} | "
              f"{condition_icon} {signal['condition']} | "
              f"Signal: {signal['signal']} | "
              f"{confirmed_icon} {'Confirmed' if signal['confirmed'] else 'Waiting'}")


def parse_command(command: str) -> Tuple[str, str, int]:
    """
    Parse terminal command: rsi_14 s=XRP/USDT t=1d l=30
    
    Args:
        command: Command string
        
    Returns:
        Tuple of (symbol, timeframe, limit)
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'rsi_14':
        raise ValueError("Invalid command format. Use: rsi_14 s=SYMBOL t=TIMEFRAME l=LIMIT")
    
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
    """Main entry point for RSI(14) strategy."""
    if len(sys.argv) < 2:
        print("Usage: python rsi_14.py s=SYMBOL t=TIMEFRAME l=LIMIT")
        print("Example: python rsi_14.py s=XRP/USDT t=1d l=30")
        return
    
    try:
        # Parse command line arguments
        command = ' '.join(['rsi_14'] + sys.argv[1:])
        symbol, timeframe, limit = parse_command(command)
        
        # Run analysis
        strategy = RSI14Strategy()
        strategy.analyze(symbol, timeframe, limit)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()