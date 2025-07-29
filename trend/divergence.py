"""
Divergence Detection Engine for cryptocurrency trend analysis.
Detects bullish and bearish divergences using RSI, MACD, and OBV indicators
to identify potential trend reversals and trading opportunities.
"""

import sys
from datetime import datetime
from typing import List, Tuple, Dict
from data import get_data_collector
from trend.rsi_14 import RSI14Strategy
from trend.macd import MACDStrategy
from trend.obv import OBVStrategy


class DivergenceDetector:
    """Divergence detection engine for trend reversal analysis."""
    
    def __init__(self):
        self.collector = get_data_collector()
        self.rsi_strategy = RSI14Strategy()
        self.macd_strategy = MACDStrategy()
        self.obv_strategy = OBVStrategy()
    
    def find_swing_highs_lows(self, values: List[float], window: int = 5) -> Tuple[List[int], List[int]]:
        """
        Find swing highs and lows in a series of values.
        
        Args:
            values: List of values to analyze
            window: Lookback/lookahead window for swing detection
            
        Returns:
            Tuple of (swing_high_indices, swing_low_indices)
        """
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(values) - window):
            # Check for swing high
            is_high = True
            for j in range(i - window, i + window + 1):
                if j != i and values[j] >= values[i]:
                    is_high = False
                    break
            if is_high:
                swing_highs.append(i)
            
            # Check for swing low
            is_low = True
            for j in range(i - window, i + window + 1):
                if j != i and values[j] <= values[i]:
                    is_low = False
                    break
            if is_low:
                swing_lows.append(i)
        
        return swing_highs, swing_lows
    
    def detect_rsi_divergence(self, closes: List[float], rsi_values: List[float]) -> List[Dict]:
        """
        Detect RSI divergences.
        
        Args:
            closes: Closing prices
            rsi_values: RSI values
            
        Returns:
            List of divergence dictionaries
        """
        divergences = []
        
        # Find swing highs and lows for both price and RSI
        price_highs, price_lows = self.find_swing_highs_lows(closes[15:])  # RSI starts at index 15
        rsi_highs, rsi_lows = self.find_swing_highs_lows(rsi_values)
        
        # Adjust price indices to match RSI indices
        price_highs = [i for i in price_highs]
        price_lows = [i for i in price_lows]
        
        # Detect bearish divergence (price higher high, RSI lower high)
        for i in range(1, len(price_highs)):
            for j in range(1, len(rsi_highs)):
                price_idx1, price_idx2 = price_highs[i-1], price_highs[i]
                rsi_idx1, rsi_idx2 = rsi_highs[j-1], rsi_highs[j]
                
                # Check if indices are close enough to be considered related
                if abs(price_idx2 - rsi_idx2) <= 3 and abs(price_idx1 - rsi_idx1) <= 3:
                    price1, price2 = closes[price_idx1 + 15], closes[price_idx2 + 15]
                    rsi1, rsi2 = rsi_values[rsi_idx1], rsi_values[rsi_idx2]
                    
                    if price2 > price1 and rsi2 < rsi1:  # Higher high in price, lower high in RSI
                        strength = self.calculate_divergence_strength(
                            (price2 - price1) / price1, (rsi1 - rsi2) / rsi1
                        )
                        
                        divergences.append({
                            'type': 'bearish',
                            'indicator': 'RSI',
                            'price_points': (price1, price2),
                            'indicator_points': (rsi1, rsi2),
                            'indices': (price_idx1 + 15, price_idx2 + 15),
                            'strength': strength,
                            'signal': 'SELL'
                        })
        
        # Detect bullish divergence (price lower low, RSI higher low)
        for i in range(1, len(price_lows)):
            for j in range(1, len(rsi_lows)):
                price_idx1, price_idx2 = price_lows[i-1], price_lows[i]
                rsi_idx1, rsi_idx2 = rsi_lows[j-1], rsi_lows[j]
                
                # Check if indices are close enough to be considered related
                if abs(price_idx2 - rsi_idx2) <= 3 and abs(price_idx1 - rsi_idx1) <= 3:
                    price1, price2 = closes[price_idx1 + 15], closes[price_idx2 + 15]
                    rsi1, rsi2 = rsi_values[rsi_idx1], rsi_values[rsi_idx2]
                    
                    if price2 < price1 and rsi2 > rsi1:  # Lower low in price, higher low in RSI
                        strength = self.calculate_divergence_strength(
                            (price1 - price2) / price1, (rsi2 - rsi1) / rsi1
                        )
                        
                        divergences.append({
                            'type': 'bullish',
                            'indicator': 'RSI',
                            'price_points': (price1, price2),
                            'indicator_points': (rsi1, rsi2),
                            'indices': (price_idx1 + 15, price_idx2 + 15),
                            'strength': strength,
                            'signal': 'BUY'
                        })
        
        return divergences
    
    def detect_macd_divergence(self, closes: List[float], macd_line: List[float]) -> List[Dict]:
        """
        Detect MACD divergences.
        
        Args:
            closes: Closing prices
            macd_line: MACD line values
            
        Returns:
            List of divergence dictionaries
        """
        divergences = []
        
        if not macd_line:
            return divergences
        
        # MACD starts much later due to EMA calculations, need to align properly
        macd_start_idx = 50  # Approximate start index for MACD in price array
        aligned_closes = closes[macd_start_idx:macd_start_idx + len(macd_line)]
        
        if len(aligned_closes) != len(macd_line):
            return divergences
        
        # Find swing highs and lows
        price_highs, price_lows = self.find_swing_highs_lows(aligned_closes)
        macd_highs, macd_lows = self.find_swing_highs_lows(macd_line)
        
        # Detect bearish divergence (price higher high, MACD lower high)
        for i in range(1, len(price_highs)):
            for j in range(1, len(macd_highs)):
                price_idx1, price_idx2 = price_highs[i-1], price_highs[i]
                macd_idx1, macd_idx2 = macd_highs[j-1], macd_highs[j]
                
                if abs(price_idx2 - macd_idx2) <= 3 and abs(price_idx1 - macd_idx1) <= 3:
                    price1, price2 = aligned_closes[price_idx1], aligned_closes[price_idx2]
                    macd1, macd2 = macd_line[macd_idx1], macd_line[macd_idx2]
                    
                    if price2 > price1 and macd2 < macd1:
                        strength = self.calculate_divergence_strength(
                            (price2 - price1) / price1, abs(macd1 - macd2) / abs(macd1) if macd1 != 0 else 0
                        )
                        
                        divergences.append({
                            'type': 'bearish',
                            'indicator': 'MACD',
                            'price_points': (price1, price2),
                            'indicator_points': (macd1, macd2),
                            'indices': (price_idx1 + macd_start_idx, price_idx2 + macd_start_idx),
                            'strength': strength,
                            'signal': 'SELL'
                        })
        
        # Detect bullish divergence (price lower low, MACD higher low)
        for i in range(1, len(price_lows)):
            for j in range(1, len(macd_lows)):
                price_idx1, price_idx2 = price_lows[i-1], price_lows[i]
                macd_idx1, macd_idx2 = macd_lows[j-1], macd_lows[j]
                
                if abs(price_idx2 - macd_idx2) <= 3 and abs(price_idx1 - macd_idx1) <= 3:
                    price1, price2 = aligned_closes[price_idx1], aligned_closes[price_idx2]
                    macd1, macd2 = macd_line[macd_idx1], macd_line[macd_idx2]
                    
                    if price2 < price1 and macd2 > macd1:
                        strength = self.calculate_divergence_strength(
                            (price1 - price2) / price1, abs(macd2 - macd1) / abs(macd1) if macd1 != 0 else 0
                        )
                        
                        divergences.append({
                            'type': 'bullish',
                            'indicator': 'MACD',
                            'price_points': (price1, price2),
                            'indicator_points': (macd1, macd2),
                            'indices': (price_idx1 + macd_start_idx, price_idx2 + macd_start_idx),
                            'strength': strength,
                            'signal': 'BUY'
                        })
        
        return divergences
    
    def detect_obv_divergence(self, closes: List[float], obv_values: List[float]) -> List[Dict]:
        """
        Detect OBV divergences.
        
        Args:
            closes: Closing prices
            obv_values: OBV values
            
        Returns:
            List of divergence dictionaries
        """
        divergences = []
        
        if len(closes) != len(obv_values):
            return divergences
        
        # Find swing highs and lows
        price_highs, price_lows = self.find_swing_highs_lows(closes)
        obv_highs, obv_lows = self.find_swing_highs_lows(obv_values)
        
        # Detect bearish divergence (price higher high, OBV lower high)
        for i in range(1, len(price_highs)):
            for j in range(1, len(obv_highs)):
                price_idx1, price_idx2 = price_highs[i-1], price_highs[i]
                obv_idx1, obv_idx2 = obv_highs[j-1], obv_highs[j]
                
                if abs(price_idx2 - obv_idx2) <= 3 and abs(price_idx1 - obv_idx1) <= 3:
                    price1, price2 = closes[price_idx1], closes[price_idx2]
                    obv1, obv2 = obv_values[obv_idx1], obv_values[obv_idx2]
                    
                    if price2 > price1 and obv2 < obv1:
                        strength = self.calculate_divergence_strength(
                            (price2 - price1) / price1, abs(obv1 - obv2) / abs(obv1) if obv1 != 0 else 0
                        )
                        
                        divergences.append({
                            'type': 'bearish',
                            'indicator': 'OBV',
                            'price_points': (price1, price2),
                            'indicator_points': (obv1, obv2),
                            'indices': (price_idx1, price_idx2),
                            'strength': strength,
                            'signal': 'SELL'
                        })
        
        # Detect bullish divergence (price lower low, OBV higher low)
        for i in range(1, len(price_lows)):
            for j in range(1, len(obv_lows)):
                price_idx1, price_idx2 = price_lows[i-1], price_lows[i]
                obv_idx1, obv_idx2 = obv_lows[j-1], obv_lows[j]
                
                if abs(price_idx2 - obv_idx2) <= 3 and abs(price_idx1 - obv_idx1) <= 3:
                    price1, price2 = closes[price_idx1], closes[price_idx2]
                    obv1, obv2 = obv_values[obv_idx1], obv_values[obv_idx2]
                    
                    if price2 < price1 and obv2 > obv1:
                        strength = self.calculate_divergence_strength(
                            (price1 - price2) / price1, abs(obv2 - obv1) / abs(obv1) if obv1 != 0 else 0
                        )
                        
                        divergences.append({
                            'type': 'bullish',
                            'indicator': 'OBV',
                            'price_points': (price1, price2),
                            'indicator_points': (obv1, obv2),
                            'indices': (price_idx1, price_idx2),
                            'strength': strength,
                            'signal': 'BUY'
                        })
        
        return divergences
    
    def calculate_divergence_strength(self, price_change_pct: float, indicator_change_pct: float) -> str:
        """
        Calculate divergence strength based on price and indicator changes.
        
        Args:
            price_change_pct: Percentage change in price
            indicator_change_pct: Percentage change in indicator
            
        Returns:
            Divergence strength ('Weak', 'Moderate', 'Strong')
        """
        # Simple strength calculation based on magnitude of changes
        combined_strength = price_change_pct + indicator_change_pct
        
        if combined_strength > 0.15:  # 15% combined change
            return 'Strong'
        elif combined_strength > 0.08:  # 8% combined change
            return 'Moderate'
        else:
            return 'Weak'
    
    def analyze(self, symbol: str, timeframe: str, limit: int) -> None:
        """
        Perform divergence analysis and display results.
        
        Args:
            symbol: Trading pair (e.g., 'SOL/USDT')
            timeframe: Timeframe (e.g., '4h', '1d')
            limit: Number of candles to analyze
        """
        # Fetch OHLCV data
        ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 50:
            print(f"Error: Need at least 50 candles for divergence analysis. Got {len(ohlcv_data)}")
            return
        
        # Extract data
        timestamps = [candle[0] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        volumes = [candle[5] for candle in ohlcv_data]
        
        # Calculate indicators
        rsi_values = self.rsi_strategy.calculate_rsi(closes)
        macd_line, _, _ = self.macd_strategy.calculate_macd(closes)
        obv_values = self.obv_strategy.calculate_obv(closes, volumes)
        
        # Detect divergences
        all_divergences = []
        
        if rsi_values:
            rsi_divergences = self.detect_rsi_divergence(closes, rsi_values)
            all_divergences.extend(rsi_divergences)
        
        if macd_line:
            macd_divergences = self.detect_macd_divergence(closes, macd_line)
            all_divergences.extend(macd_divergences)
        
        if obv_values:
            obv_divergences = self.detect_obv_divergence(closes, obv_values)
            all_divergences.extend(obv_divergences)
        
        # Sort by index (chronological order)
        all_divergences.sort(key=lambda x: x['indices'][1])
        
        # Filter for recent divergences
        recent_divergences = []
        
        for div in all_divergences:
            timestamp_idx = div['indices'][1]
            if timestamp_idx < len(timestamps):
                dt = datetime.fromtimestamp(timestamps[timestamp_idx] / 1000)
                # Show divergences from last 7 days
                if (datetime.now() - dt).days <= 7:
                    recent_divergences.append((div, dt))
        
        if not recent_divergences:
            print(f"No recent divergences found for {symbol} on {timeframe}")
            # Show latest divergence if any
            if all_divergences:
                latest_div = all_divergences[-1]
                timestamp_idx = latest_div['indices'][1]
                if timestamp_idx < len(timestamps):
                    dt = datetime.fromtimestamp(timestamps[timestamp_idx] / 1000)
                    self.print_divergence(latest_div, dt, "Latest")
        else:
            print("=" * 60)
            print(f"Divergence Analysis for {symbol} ({timeframe}):")
            print("=" * 60)
            
            for div, dt in recent_divergences:
                self.print_divergence(div, dt)
    
    def print_divergence(self, div: Dict, dt: datetime, prefix: str = "") -> None:
        """
        Print formatted divergence information.
        
        Args:
            div: Divergence dictionary
            dt: Datetime of divergence
            prefix: Optional prefix for output
        """
        indicator = div['indicator']
        div_type = div['type'].capitalize()
        strength = div['strength']
        signal = div['signal']
        
        price1, price2 = div['price_points']
        ind1, ind2 = div['indicator_points']
        
        # Format indicator values
        if indicator == 'RSI':
            ind_str = f"RSI {'HL' if div_type == 'Bullish' else 'LH'}: {ind1:.1f} → {ind2:.1f}"
            icon = "🧠" if div_type == 'Bullish' else "⚠️"
            context = "Early Reversal" if div_type == 'Bullish' else "Weak Momentum"
        elif indicator == 'MACD':
            ind_str = f"MACD {'HL' if div_type == 'Bullish' else 'LH'}: {ind1:.3f} → {ind2:.3f}"
            icon = "📈" if div_type == 'Bullish' else "📉"  
            context = "Momentum Shift" if div_type == 'Bullish' else "Trend Weakening"
        else:  # OBV
            ind_str = f"OBV {'Rising' if div_type == 'Bullish' else 'Falling'} vs Price {'Drop' if div_type == 'Bullish' else 'Rise'}"
            icon = "🔋" if div_type == 'Bullish' else "🔻"
            context = "Accumulation Phase" if div_type == 'Bullish' else "Distribution Phase"
        
        price_str = f"Price {'LL' if div_type == 'Bullish' else 'HH'}: {price1:.4f} → {price2:.4f}"
        
        prefix_str = f"{prefix}: " if prefix else ""
        
        print(f"{prefix_str}[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
              f"{indicator} Divergence: {div_type} | "
              f"{price_str} | {ind_str} | "
              f"Signal: {signal} | {icon} {context} ({strength})")


def parse_command(command: str) -> Tuple[str, str, int]:
    """
    Parse terminal command: divergence s=SOL/USDT t=4h l=100
    
    Args:
        command: Command string
        
    Returns:
        Tuple of (symbol, timeframe, limit)
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'divergence':
        raise ValueError("Invalid command format. Use: divergence s=SYMBOL t=TIMEFRAME l=LIMIT")
    
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
    """Main entry point for divergence detection."""
    if len(sys.argv) < 2:
        print("Usage: python divergence.py s=SYMBOL t=TIMEFRAME l=LIMIT")
        print("Example: python divergence.py s=SOL/USDT t=4h l=100")
        return
    
    try:
        # Parse command line arguments
        command = ' '.join(['divergence'] + sys.argv[1:])
        symbol, timeframe, limit = parse_command(command)
        
        # Run analysis
        detector = DivergenceDetector()
        detector.analyze(symbol, timeframe, limit)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()