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
    
    def find_swing_highs_lows(self, values: List[float], lookback: int = 2) -> Tuple[List[int], List[int]]:
        """
        Find swing highs and lows using standard pivot point method.
        
        Args:
            values: List of values to analyze
            lookback: Number of periods to look back/forward (default 2)
            
        Returns:
            Tuple of (swing_high_indices, swing_low_indices)
        """
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(values) - lookback):
            # Check for swing high - current value is highest in lookback window
            is_high = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and values[j] >= values[i]:
                    is_high = False
                    break
            if is_high:
                swing_highs.append(i)
            
            # Check for swing low - current value is lowest in lookback window  
            is_low = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and values[j] <= values[i]:
                    is_low = False
                    break
            if is_low:
                swing_lows.append(i)
        
        return swing_highs, swing_lows
    
    def detect_rsi_divergence(self, closes: List[float], rsi_values: List[float]) -> List[Dict]:
        """
        Detect RSI divergences with proper index alignment.
        
        Args:
            closes: Closing prices
            rsi_values: RSI values (already aligned with closes from index 14 onwards)
            
        Returns:
            List of divergence dictionaries
        """
        divergences = []
        
        # RSI calculation starts from index 13 (0-based, needs 14 values: 0-13)
        rsi_start_idx = 13
        
        # Validate data lengths
        if len(closes) < rsi_start_idx + 1 or len(rsi_values) == 0:
            return divergences
        
        # RSI values should align with closes starting from rsi_start_idx
        expected_rsi_length = len(closes) - rsi_start_idx
        if len(rsi_values) != expected_rsi_length:
            # Take the minimum length to avoid index errors
            min_length = min(len(rsi_values), expected_rsi_length)
            aligned_closes = closes[rsi_start_idx:rsi_start_idx + min_length]
            aligned_rsi = rsi_values[:min_length]
        else:
            aligned_closes = closes[rsi_start_idx:]
            aligned_rsi = rsi_values
        
        if len(aligned_closes) != len(aligned_rsi) or len(aligned_closes) < 10:
            return divergences
        
        # Find swing highs and lows for both aligned price and RSI
        price_highs, price_lows = self.find_swing_highs_lows(aligned_closes)
        rsi_highs, rsi_lows = self.find_swing_highs_lows(aligned_rsi)
        
        # Detect bearish divergence (price higher high, RSI lower high)
        bearish_pairs = self._find_matching_swing_pairs(price_highs, rsi_highs)
        for (price_idx1, price_idx2), (rsi_idx1, rsi_idx2) in bearish_pairs:
            price1, price2 = aligned_closes[price_idx1], aligned_closes[price_idx2]
            rsi1, rsi2 = aligned_rsi[rsi_idx1], aligned_rsi[rsi_idx2]
            
            # Standard bearish divergence: Higher High in price, Lower High in RSI
            if price2 > price1 and rsi2 < rsi1:
                divergences.append({
                    'type': 'bearish',
                    'indicator': 'RSI',
                    'price_points': (price1, price2),
                    'indicator_points': (rsi1, rsi2),
                    'indices': (price_idx1 + rsi_start_idx, price_idx2 + rsi_start_idx),
                    'strength': 'Valid',
                    'signal': 'SELL'
                })
        
        # Detect bullish divergence (price lower low, RSI higher low)
        bullish_pairs = self._find_matching_swing_pairs(price_lows, rsi_lows)
        for (price_idx1, price_idx2), (rsi_idx1, rsi_idx2) in bullish_pairs:
            price1, price2 = aligned_closes[price_idx1], aligned_closes[price_idx2]
            rsi1, rsi2 = aligned_rsi[rsi_idx1], aligned_rsi[rsi_idx2]
            
            # Standard bullish divergence: Lower Low in price, Higher Low in RSI
            if price2 < price1 and rsi2 > rsi1:
                divergences.append({
                    'type': 'bullish',
                    'indicator': 'RSI',
                    'price_points': (price1, price2),
                    'indicator_points': (rsi1, rsi2),
                    'indices': (price_idx1 + rsi_start_idx, price_idx2 + rsi_start_idx),
                    'strength': 'Valid',
                    'signal': 'BUY'
                })
        
        return divergences
    
    def detect_macd_divergence(self, closes: List[float], macd_line: List[float]) -> List[Dict]:
        """
        Detect MACD divergences with proper index alignment.
        
        Args:
            closes: Closing prices
            macd_line: MACD line values (already aligned with closes from index 25 onwards)
            
        Returns:
            List of divergence dictionaries
        """
        divergences = []
        
        if not macd_line:
            return divergences
        
        # MACD calculation requires 26 periods for EMA26, so starts at index 25 (0-based)
        macd_start_idx = 25
        
        # Validate data lengths
        if len(closes) < macd_start_idx + 1 or len(macd_line) == 0:
            return divergences
        
        # MACD values should align with closes starting from macd_start_idx
        expected_macd_length = len(closes) - macd_start_idx
        if len(macd_line) != expected_macd_length:
            # Take the minimum length to avoid index errors
            min_length = min(len(macd_line), expected_macd_length)
            aligned_closes = closes[macd_start_idx:macd_start_idx + min_length]
            aligned_macd = macd_line[:min_length]
        else:
            aligned_closes = closes[macd_start_idx:]
            aligned_macd = macd_line
        
        if len(aligned_closes) != len(aligned_macd) or len(aligned_closes) < 10:
            return divergences
        
        # Find swing highs and lows
        price_highs, price_lows = self.find_swing_highs_lows(aligned_closes)
        macd_highs, macd_lows = self.find_swing_highs_lows(aligned_macd)
        
        # Detect bearish divergence (price higher high, MACD lower high)
        bearish_pairs = self._find_matching_swing_pairs(price_highs, macd_highs)
        for (price_idx1, price_idx2), (macd_idx1, macd_idx2) in bearish_pairs:
            price1, price2 = aligned_closes[price_idx1], aligned_closes[price_idx2]
            macd1, macd2 = aligned_macd[macd_idx1], aligned_macd[macd_idx2]
            
            # Standard bearish divergence: Higher High in price, Lower High in MACD
            if price2 > price1 and macd2 < macd1:
                divergences.append({
                    'type': 'bearish',
                    'indicator': 'MACD',
                    'price_points': (price1, price2),
                    'indicator_points': (macd1, macd2),
                    'indices': (price_idx1 + macd_start_idx, price_idx2 + macd_start_idx),
                    'strength': 'Valid',
                    'signal': 'SELL'
                })
        
        # Detect bullish divergence (price lower low, MACD higher low)
        bullish_pairs = self._find_matching_swing_pairs(price_lows, macd_lows)
        for (price_idx1, price_idx2), (macd_idx1, macd_idx2) in bullish_pairs:
            price1, price2 = aligned_closes[price_idx1], aligned_closes[price_idx2]
            macd1, macd2 = aligned_macd[macd_idx1], aligned_macd[macd_idx2]
            
            # Standard bullish divergence: Lower Low in price, Higher Low in MACD
            if price2 < price1 and macd2 > macd1:
                divergences.append({
                    'type': 'bullish',
                    'indicator': 'MACD',
                    'price_points': (price1, price2),
                    'indicator_points': (macd1, macd2),
                    'indices': (price_idx1 + macd_start_idx, price_idx2 + macd_start_idx),
                    'strength': 'Valid',
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
        bearish_pairs = self._find_matching_swing_pairs(price_highs, obv_highs)
        for (price_idx1, price_idx2), (obv_idx1, obv_idx2) in bearish_pairs:
            price1, price2 = closes[price_idx1], closes[price_idx2]
            obv1, obv2 = obv_values[obv_idx1], obv_values[obv_idx2]
            
            # Standard bearish divergence: Higher High in price, Lower High in OBV
            if price2 > price1 and obv2 < obv1:
                divergences.append({
                    'type': 'bearish',
                    'indicator': 'OBV',
                    'price_points': (price1, price2),
                    'indicator_points': (obv1, obv2),
                    'indices': (price_idx1, price_idx2),
                    'strength': 'Valid',
                    'signal': 'SELL'
                })
        
        # Detect bullish divergence (price lower low, OBV higher low)
        bullish_pairs = self._find_matching_swing_pairs(price_lows, obv_lows)
        for (price_idx1, price_idx2), (obv_idx1, obv_idx2) in bullish_pairs:
            price1, price2 = closes[price_idx1], closes[price_idx2]
            obv1, obv2 = obv_values[obv_idx1], obv_values[obv_idx2]
            
            # Standard bullish divergence: Lower Low in price, Higher Low in OBV
            if price2 < price1 and obv2 > obv1:
                divergences.append({
                    'type': 'bullish',
                    'indicator': 'OBV',
                    'price_points': (price1, price2),
                    'indicator_points': (obv1, obv2),
                    'indices': (price_idx1, price_idx2),
                    'strength': 'Valid',
                    'signal': 'BUY'
                })
        
        return divergences
    
    def _find_matching_swing_pairs(self, price_swings: List[int], indicator_swings: List[int]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Find matching swing pairs between price and indicator using standard time alignment.
        
        Args:
            price_swings: List of price swing indices
            indicator_swings: List of indicator swing indices
            
        Returns:
            List of matching swing pairs: ((price_idx1, price_idx2), (ind_idx1, ind_idx2))
        """
        matching_pairs = []
        
        if len(price_swings) < 2 or len(indicator_swings) < 2:
            return matching_pairs
        
        # Standard tolerance: exact match or ±1-2 candles
        max_tolerance = 2
        
        # For each consecutive pair of price swings
        for i in range(len(price_swings) - 1):
            price_idx1, price_idx2 = price_swings[i], price_swings[i + 1]
            
            # Find exact or near-exact matching indicator swing pair
            for j in range(len(indicator_swings) - 1):
                ind_idx1, ind_idx2 = indicator_swings[j], indicator_swings[j + 1]
                
                # Check for close time alignment (standard approach)
                alignment1 = abs(price_idx1 - ind_idx1)
                alignment2 = abs(price_idx2 - ind_idx2)
                
                if alignment1 <= max_tolerance and alignment2 <= max_tolerance:
                    # Ensure proper chronological order
                    if price_idx1 < price_idx2 and ind_idx1 < ind_idx2:
                        matching_pairs.append(((price_idx1, price_idx2), (ind_idx1, ind_idx2)))
                        break  # Take first good match (most aligned)
        
        return matching_pairs
    
    
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
        
        # Calculate indicators with proper error handling
        rsi_values = []
        try:
            rsi_values = self.rsi_strategy.calculate_rsi(closes)
        except Exception as e:
            print(f"Warning: RSI calculation failed: {e}")
        
        macd_line = []
        try:
            macd_line, _, _ = self.macd_strategy.calculate_macd(closes)
        except Exception as e:
            print(f"Warning: MACD calculation failed: {e}")
        
        obv_values = []
        try:
            obv_values = self.obv_strategy.calculate_obv(closes, volumes)
        except Exception as e:
            print(f"Warning: OBV calculation failed: {e}")
        
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
                # Show divergences from last 14 days (more flexible)
                if (datetime.now() - dt).days <= 14:
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
        
        # Format indicator and price values with swing point labels
        if indicator == 'RSI':
            metric1 = f"RSI_1ST: {ind1:.1f}"
            metric2 = f"RSI_2ND: {ind2:.1f}"
            trend_emoji = "📈" if div_type == 'Bullish' else "📉"
            trend_label = "Bullish Divergence" if div_type == 'Bullish' else "Bearish Divergence"
        elif indicator == 'MACD':
            metric1 = f"MACD_1ST: {ind1:.3f}"
            metric2 = f"MACD_2ND: {ind2:.3f}"
            trend_emoji = "📈" if div_type == 'Bullish' else "📉"
            trend_label = "Bullish Divergence" if div_type == 'Bullish' else "Bearish Divergence"
        else:  # OBV
            metric1 = f"OBV_1ST: {ind1:.0f}"
            metric2 = f"OBV_2ND: {ind2:.0f}"
            trend_emoji = "📈" if div_type == 'Bullish' else "📉"
            trend_label = "Bullish Divergence" if div_type == 'Bullish' else "Bearish Divergence"
        
        price_prev = f"PRICE_1ST: {price1:.4f}"
        price_curr = f"PRICE_2ND: {price2:.4f}"
        strength_metric = f"STRENGTH: {strength}"
        
        prefix_str = f"{prefix}: " if prefix else ""
        
        print(f"{prefix_str}[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
              f"{price_prev} | {price_curr} | {metric1} | {metric2} | {strength_metric} | "
              f"Signal: {signal} | {trend_emoji} {trend_label}")


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