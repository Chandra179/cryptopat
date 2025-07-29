"""
OBV (On-Balance Volume) strategy implementation for cryptocurrency trend analysis.
Uses volume and price relationship to detect bullish and bearish trend signals.
"""

import sys
from datetime import datetime
from typing import List, Tuple
from data import get_data_collector


class OBVStrategy:
    """OBV (On-Balance Volume) strategy for trend detection."""
    
    def __init__(self):
        self.collector = get_data_collector()
    
    def calculate_obv(self, closes: List[float], volumes: List[float]) -> List[float]:
        """
        Calculate On-Balance Volume.
        
        Args:
            closes: List of closing prices
            volumes: List of volume values
            
        Returns:
            List of OBV values
        """
        if len(closes) != len(volumes) or len(closes) < 2:
            return []
        
        obv_values = [volumes[0]]  # Start with first volume as initial OBV
        
        for i in range(1, len(closes)):
            current_close = closes[i]
            previous_close = closes[i-1]
            current_volume = volumes[i]
            previous_obv = obv_values[-1]
            
            if current_close > previous_close:
                # Price up, add volume
                obv = previous_obv + current_volume
            elif current_close < previous_close:
                # Price down, subtract volume
                obv = previous_obv - current_volume
            else:
                # Price unchanged, OBV unchanged
                obv = previous_obv
            
            obv_values.append(obv)
        
        return obv_values
    
    def detect_obv_signals(self, obv: List[float], closes: List[float]) -> List[dict]:
        """
        Detect OBV trend signals.
        
        Args:
            obv: OBV values
            closes: Closing prices
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        for i in range(10, len(obv)):  # Need some history for trend detection
            signal = {
                'index': i,
                'close': closes[i],
                'obv': obv[i],
                'signal': 'NONE',
                'trend_confirmation': 'NEUTRAL',
                'divergence': False
            }
            
            # Look at OBV trend over last 5 periods
            obv_trend_periods = 5
            if i >= obv_trend_periods:
                recent_obv = obv[i-obv_trend_periods:i+1]
                recent_prices = closes[i-obv_trend_periods:i+1]
                
                # Calculate OBV trend (simple linear trend)
                obv_trend = recent_obv[-1] - recent_obv[0]
                price_trend = recent_prices[-1] - recent_prices[0]
                
                # Strong OBV threshold (adjust based on typical volumes)
                obv_threshold = abs(sum(recent_obv)) * 0.05  # 5% of average OBV magnitude
                
                # Detect signals based on OBV trend
                if obv_trend > obv_threshold:
                    signal['signal'] = 'BUY'
                    signal['trend_confirmation'] = 'UPTREND' if price_trend > 0 else 'CONFIRMED_UPTREND'
                    
                    # Check for divergence (price down, OBV up)
                    if price_trend < 0:
                        signal['divergence'] = True
                        signal['trend_confirmation'] = 'BULLISH_DIVERGENCE'
                        
                elif obv_trend < -obv_threshold:
                    signal['signal'] = 'SELL'
                    signal['trend_confirmation'] = 'DOWNTREND' if price_trend < 0 else 'CONFIRMED_DOWNTREND'
                    
                    # Check for divergence (price up, OBV down)
                    if price_trend > 0:
                        signal['divergence'] = True
                        signal['trend_confirmation'] = 'BEARISH_DIVERGENCE'
            
            signals.append(signal)
        
        return signals
    
    def analyze(self, symbol: str, timeframe: str, limit: int) -> None:
        """
        Perform OBV analysis and display results.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1d', '4h', '1h')
            limit: Number of candles to analyze
        """
        # Fetch OHLCV data
        ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 50:
            print(f"Error: Need at least 50 candles for OBV calculation. Got {len(ohlcv_data)}")
            return
        
        # Extract closes and volumes
        timestamps = [candle[0] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        volumes = [candle[5] for candle in ohlcv_data]
        
        # Calculate OBV
        obv = self.calculate_obv(closes, volumes)
        
        if not obv:
            print("Error: Unable to calculate OBV")
            return
        
        # Detect signals
        signals = self.detect_obv_signals(obv, closes)
        
        today = datetime.now().date()
        today_signals = []
        
        for signal in signals:
            timestamp_idx = signal['index']
            if timestamp_idx < len(timestamps):
                dt = datetime.fromtimestamp(timestamps[timestamp_idx] / 1000)
                if dt.date() == today:
                    today_signals.append((signal, dt))
        
        if not today_signals:
            print(f"No signals found for today ({today})")
            # Show latest signal for reference
            if signals:
                latest_signal = signals[-1]
                timestamp_idx = latest_signal['index']
                if timestamp_idx < len(timestamps):
                    dt = datetime.fromtimestamp(timestamps[timestamp_idx] / 1000)
                    
                    # Format signal indicators
                    signal_icon = "📈" if latest_signal['signal'] == 'BUY' else "📉" if latest_signal['signal'] == 'SELL' else "➖"
                    divergence_icon = "⚠️" if latest_signal['divergence'] else ""
                    
                    print(f"\nLatest signal:")
                    print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                          f"OBV: {latest_signal['obv']:,.0f} | "
                          f"PRICE: {latest_signal['close']:.4f} | "
                          f"Signal: {latest_signal['signal']} | "
                          f"{signal_icon} {latest_signal['trend_confirmation']} {divergence_icon}")
        else:
            for signal, dt in today_signals:
                # Format signal indicators
                signal_icon = "📈" if signal['signal'] == 'BUY' else "📉" if signal['signal'] == 'SELL' else "➖"
                divergence_icon = "⚠️" if signal['divergence'] else ""
                
                print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"OBV: {signal['obv']:,.0f} | "
                      f"PRICE: {signal['close']:.4f} | "
                      f"Signal: {signal['signal']} | "
                      f"{signal_icon} {signal['trend_confirmation']} {divergence_icon}")


def parse_command(command: str) -> Tuple[str, str, int]:
    """
    Parse terminal command: obv s=BTC/USDT t=1d l=100
    
    Args:
        command: Command string
        
    Returns:
        Tuple of (symbol, timeframe, limit)
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'obv':
        raise ValueError("Invalid command format. Use: obv s=SYMBOL t=TIMEFRAME l=LIMIT")
    
    symbol = None
    timeframe = '1d'  # default
    limit = 100  # default for OBV
    
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
    """Main entry point for OBV strategy."""
    if len(sys.argv) < 2:
        print("Usage: python obv.py s=SYMBOL t=TIMEFRAME l=LIMIT")
        print("Example: python obv.py s=BTC/USDT t=1d l=100")
        return
    
    try:
        # Parse command line arguments
        command = ' '.join(['obv'] + sys.argv[1:])
        symbol, timeframe, limit = parse_command(command)
        
        # Run analysis
        strategy = OBVStrategy()
        strategy.analyze(symbol, timeframe, limit)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()