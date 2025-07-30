"""
Wyckoff Structure Analysis implementation for cryptocurrency markets.
Detects accumulation/distribution phases and key Wyckoff events using OHLCV data.
"""

import sys
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import statistics

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import get_data_collector


class WyckoffAnalyzer:
    """Wyckoff Structure Analysis for market cycle detection."""
    
    def __init__(self):
        self.collector = get_data_collector()
        self.min_range_periods = 20  # Minimum periods for range detection
        self.volume_ma_period = 20   # Volume moving average period
        
    def detect_ranges(self, highs: List[float], lows: List[float], 
                     lookback: int = 15) -> List[Dict]:
        """
        Detect trading ranges where price is consolidating.
        
        Args:
            highs: List of high prices
            lows: List of low prices  
            lookback: Period for range validation
            
        Returns:
            List of range dictionaries with start/end indices and levels
        """
        ranges = []
        tolerance = 0.03  # 3% tolerance for range boundaries
        
        for i in range(lookback, len(highs) - self.min_range_periods):
            # Get recent highs and lows
            recent_highs = highs[i-lookback:i+self.min_range_periods]
            recent_lows = lows[i-lookback:i+self.min_range_periods]
            
            range_high = max(recent_highs)
            range_low = min(recent_lows)
            range_size = range_high - range_low
            
            # Check if price is ranging (multiple touches of boundaries)
            high_touches = sum(1 for h in recent_highs if abs(h - range_high) / range_high <= tolerance)
            low_touches = sum(1 for l in recent_lows if abs(l - range_low) / range_low <= tolerance)
            
            # Valid range needs multiple boundary touches
            if high_touches >= 2 and low_touches >= 2:
                ranges.append({
                    'start_index': i,
                    'end_index': i + self.min_range_periods,
                    'high': range_high,
                    'low': range_low,
                    'range_size': range_size,
                    'high_touches': high_touches,
                    'low_touches': low_touches
                })
        
        return ranges
    
    def detect_phase(self, price_data: List[float], volume_data: List[float],
                    range_info: Dict, start_idx: int, end_idx: int) -> str:
        """
        Determine Wyckoff phase based on price action and volume characteristics.
        
        Args:
            price_data: Close prices
            volume_data: Volume data
            range_info: Range boundary information
            start_idx: Start index of phase
            end_idx: End index of phase
            
        Returns:
            Phase name: 'Accumulation', 'Markup', 'Distribution', 'Markdown'
        """
        phase_prices = price_data[start_idx:end_idx]
        phase_volumes = volume_data[start_idx:end_idx]
        
        if len(phase_prices) < 10:
            return 'Unknown'
            
        # Calculate trend and volatility metrics
        price_trend = (phase_prices[-1] - phase_prices[0]) / phase_prices[0]
        avg_volume = statistics.mean(phase_volumes)
        volume_trend = (statistics.mean(phase_volumes[-5:]) - 
                       statistics.mean(phase_volumes[:5])) / statistics.mean(phase_volumes[:5])
        
        # Range-bound phases (Accumulation/Distribution)
        if abs(price_trend) < 0.05:  # Less than 5% movement
            if volume_trend > 0.1:  # Increasing volume
                return 'Accumulation'
            elif volume_trend < -0.1:  # Decreasing volume  
                return 'Distribution'
            else:
                return 'Consolidation'
        
        # Trending phases (Markup/Markdown)
        elif price_trend > 0.05:  # Uptrend
            if volume_trend < -0.1:  # Decreasing volume on uptrend
                return 'Markup'
            else:
                return 'Early_Markup'
        
        elif price_trend < -0.05:  # Downtrend
            if volume_trend < -0.1:  # Decreasing volume on downtrend
                return 'Markdown'
            else:
                return 'Early_Markdown'
        
        return 'Transition'
    
    def detect_wyckoff_events(self, opens: List[float], highs: List[float],
                             lows: List[float], closes: List[float], 
                             volumes: List[float]) -> List[Dict]:
        """
        Detect key Wyckoff events: SC, AR, ST, Spring, UTAD, LPS/LPSY.
        
        Args:
            opens: Open prices
            highs: High prices
            lows: Low prices  
            closes: Close prices
            volumes: Volume data
            
        Returns:
            List of detected Wyckoff events
        """
        events = []
        vol_ma = self._calculate_volume_ma(volumes)
        
        for i in range(5, len(closes) - 1):
            candle_body = abs(closes[i] - opens[i])
            candle_range = highs[i] - lows[i]
            lower_wick = min(opens[i], closes[i]) - lows[i]
            upper_wick = highs[i] - max(opens[i], closes[i])
            
            # Volume surge detection
            volume_surge = volumes[i] > vol_ma[i] * 1.5
            extreme_volume = volumes[i] > vol_ma[i] * 2.0
            
            # Selling Climax (SC) - Sharp drop + high volume + long lower wick
            if (closes[i] < closes[i-1] * 0.95 and  # 5% drop
                volume_surge and
                lower_wick > candle_range * 0.3):  # Lower wick > 30% of range
                
                events.append({
                    'index': i,
                    'type': 'SC',
                    'name': 'Selling Climax',
                    'price': closes[i],
                    'volume': volumes[i],
                    'volume_ratio': volumes[i] / vol_ma[i],
                    'wick_ratio': lower_wick / candle_range
                })
            
            # Automatic Rally (AR) - Bounce after sharp drop
            elif (i > 0 and closes[i] > closes[i-1] * 1.03 and  # 3% bounce
                  closes[i-1] < closes[i-2] * 0.97 and  # Previous was decline
                  volume_surge):
                
                events.append({
                    'index': i,
                    'type': 'AR', 
                    'name': 'Automatic Rally',
                    'price': closes[i],
                    'volume': volumes[i],
                    'bounce_size': (closes[i] - closes[i-1]) / closes[i-1]
                })
            
            # Secondary Test (ST) - Retest with lower volume
            elif (self._is_retest_of_low(closes, lows, i, 10) and
                  volumes[i] < vol_ma[i] * 0.8):  # Lower volume
                
                events.append({
                    'index': i,
                    'type': 'ST',
                    'name': 'Secondary Test',
                    'price': closes[i],
                    'volume': volumes[i],
                    'volume_ratio': volumes[i] / vol_ma[i]
                })
            
            # Spring - False breakdown with reversal
            elif (self._is_false_breakdown(highs, lows, closes, i, 15) and
                  volumes[i] > vol_ma[i] * 1.2):
                
                events.append({
                    'index': i,
                    'type': 'Spring',
                    'name': 'Spring',
                    'price': closes[i],
                    'volume': volumes[i],
                    'reversal_strength': (closes[i] - lows[i]) / (highs[i] - lows[i])
                })
            
            # Upthrust After Distribution (UTAD) - False breakout above range
            elif (self._is_false_breakout(highs, lows, closes, i, 15) and
                  upper_wick > candle_range * 0.3):
                
                events.append({
                    'index': i,
                    'type': 'UTAD',
                    'name': 'Upthrust After Distribution', 
                    'price': closes[i],
                    'volume': volumes[i],
                    'rejection_ratio': upper_wick / candle_range
                })
            
            # Last Point of Support/Supply (LPS/LPSY)
            elif self._is_last_retest(closes, highs, lows, i, 20):
                event_type = 'LPS' if closes[i] > closes[i-5] else 'LPSY'
                events.append({
                    'index': i,
                    'type': event_type,
                    'name': 'Last Point of Support' if event_type == 'LPS' else 'Last Point of Supply',
                    'price': closes[i],
                    'volume': volumes[i]
                })
        
        return events
    
    def _calculate_volume_ma(self, volumes: List[float]) -> List[float]:
        """Calculate volume moving average."""
        vol_ma = []
        for i in range(len(volumes)):
            start_idx = max(0, i - self.volume_ma_period + 1)
            vol_ma.append(statistics.mean(volumes[start_idx:i+1]))
        return vol_ma
    
    def _is_retest_of_low(self, closes: List[float], lows: List[float], 
                         current_idx: int, lookback: int) -> bool:
        """Check if current price is retesting a previous low."""
        if current_idx < lookback:
            return False
            
        recent_lows = lows[current_idx-lookback:current_idx]
        min_low = min(recent_lows)
        tolerance = 0.02  # 2% tolerance
        
        return abs(closes[current_idx] - min_low) / min_low <= tolerance
    
    def _is_false_breakdown(self, highs: List[float], lows: List[float],
                           closes: List[float], current_idx: int, lookback: int) -> bool:
        """Detect false breakdown below support with quick reversal."""
        if current_idx < lookback + 2:
            return False
            
        # Find recent support level
        recent_lows = lows[current_idx-lookback:current_idx-2]
        support_level = min(recent_lows)
        
        # Check if broke below support but closed back above
        return (lows[current_idx] < support_level * 0.99 and  # Broke support
                closes[current_idx] > support_level * 1.01)    # Closed back above
    
    def _is_false_breakout(self, highs: List[float], lows: List[float],
                          closes: List[float], current_idx: int, lookback: int) -> bool:
        """Detect false breakout above resistance with rejection."""
        if current_idx < lookback + 2:
            return False
            
        # Find recent resistance level  
        recent_highs = highs[current_idx-lookback:current_idx-2]
        resistance_level = max(recent_highs)
        
        # Check if broke above resistance but closed back below
        return (highs[current_idx] > resistance_level * 1.01 and  # Broke resistance
                closes[current_idx] < resistance_level * 0.99)    # Closed back below
    
    def _is_last_retest(self, closes: List[float], highs: List[float],
                       lows: List[float], current_idx: int, lookback: int) -> bool:
        """Check if this is likely the last retest before trend resumption.""" 
        if current_idx < lookback:
            return False
            
        # Look for previous similar retests and check if this is weaker
        recent_range = lookback // 2
        recent_volumes_available = current_idx >= recent_range
        
        if not recent_volumes_available:
            return False
            
        # Simple check: lower volume and price holding key level
        prev_high = max(highs[current_idx-recent_range:current_idx])
        prev_low = min(lows[current_idx-recent_range:current_idx])
        
        # Price holding within recent range suggests last retest
        return (prev_low <= closes[current_idx] <= prev_high and
                abs(closes[current_idx] - prev_low) / prev_low <= 0.05)
    
    def effort_vs_result_analysis(self, prices: List[float], volumes: List[float],
                                 window: int = 10) -> List[Dict]:
        """
        Analyze effort (volume) vs result (price movement) for anomalies.
        
        Args:
            prices: Close prices
            volumes: Volume data
            window: Analysis window size
            
        Returns:
            List of EVR anomaly signals
        """
        evr_signals = []
        vol_ma = self._calculate_volume_ma(volumes)
        
        for i in range(window, len(prices) - window):
            # Calculate price movement and volume effort over window
            price_start = prices[i - window]
            price_end = prices[i + window]
            price_movement = abs(price_end - price_start) / price_start
            
            avg_volume = statistics.mean(volumes[i-window:i+window])
            volume_ratio = avg_volume / vol_ma[i]
            
            # Detect anomalies
            if volume_ratio > 1.3 and price_movement < 0.02:  # High volume, little movement
                evr_signals.append({
                    'index': i,
                    'type': 'absorption',
                    'volume_ratio': volume_ratio,
                    'price_movement': price_movement,
                    'signal': 'Professional absorption likely'
                })
            
            elif volume_ratio < 0.7 and price_movement > 0.05:  # Low volume, big movement
                evr_signals.append({
                    'index': i,
                    'type': 'lack_of_interest',
                    'volume_ratio': volume_ratio, 
                    'price_movement': price_movement,
                    'signal': 'Lack of participation'
                })
        
        return evr_signals
    
    def generate_wyckoff_signals(self, events: List[Dict], phases: List[Dict],
                               evr_signals: List[Dict], closes: List[float],
                               timestamps: List[int]) -> List[Dict]:
        """
        Generate BUY/SELL signals based on Wyckoff methodology.
        
        Args:
            events: Detected Wyckoff events
            phases: Detected market phases  
            evr_signals: Effort vs Result signals
            closes: Close prices
            timestamps: Timestamps
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Look for signal setups in recent data
        recent_window = min(50, len(closes))
        
        for i in range(len(closes) - recent_window, len(closes)):
            signal = {
                'index': i,
                'timestamp': timestamps[i],
                'close': closes[i],
                'signal': 'NONE',
                'confidence': 0,
                'setup_type': '',
                'phase': 'Unknown',
                'events': []
            }
            
            # Find events near this index
            nearby_events = [e for e in events if abs(e['index'] - i) <= 10]
            signal['events'] = [e['type'] for e in nearby_events]
            
            # Find current phase
            current_phases = [p for p in phases if p['start_index'] <= i <= p['end_index']]
            if current_phases:
                signal['phase'] = current_phases[0]['phase']
            
            # BUY Signal Logic: Spring + breakout + LPS
            spring_events = [e for e in nearby_events if e['type'] == 'Spring']
            lps_events = [e for e in nearby_events if e['type'] == 'LPS'] 
            
            if spring_events and signal['phase'] in ['Accumulation']:
                signal['signal'] = 'BUY'
                signal['confidence'] = 70
                signal['setup_type'] = 'Spring_Setup'
                
                if lps_events:  # Additional confirmation
                    signal['confidence'] = 85
                    signal['setup_type'] = 'Spring_LPS_Setup'
            
            # SELL Signal Logic: UTAD + breakdown + LPSY  
            utad_events = [e for e in nearby_events if e['type'] == 'UTAD']
            lpsy_events = [e for e in nearby_events if e['type'] == 'LPSY']
            
            if utad_events and signal['phase'] in ['Distribution']:
                signal['signal'] = 'SELL'
                signal['confidence'] = 70
                signal['setup_type'] = 'UTAD_Setup'
                
                if lpsy_events:  # Additional confirmation
                    signal['confidence'] = 85
                    signal['setup_type'] = 'UTAD_LPSY_Setup'
            
            # Volume confirmation from EVR
            nearby_evr = [e for e in evr_signals if abs(e['index'] - i) <= 5]
            if nearby_evr:
                absorption_signals = [e for e in nearby_evr if e['type'] == 'absorption']
                if absorption_signals and signal['signal'] == 'BUY':
                    signal['confidence'] = min(95, signal['confidence'] + 10)
                elif absorption_signals and signal['signal'] == 'SELL':
                    signal['confidence'] = min(95, signal['confidence'] + 10)
            
            # Only add meaningful signals
            if signal['signal'] != 'NONE' or signal['events']:
                signals.append(signal)
        
        return signals
    
    def analyze(self, symbol: str, timeframe: str, limit: int,
               detect: str = 'phases+events') -> None:
        """
        Perform Wyckoff structure analysis and display results.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')  
            limit: Number of candles (minimum 300 for full cycle mapping)
            detect: What to detect ('phases', 'events', 'phases+events')
        """
        if limit < 300:
            print(f"Warning: Wyckoff analysis requires ≥300 candles for full cycle context. Using minimum 300.")
            limit = 300
        
        # Fetch OHLCV data
        ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 300:
            print(f"Error: Need at least 300 candles for Wyckoff analysis. Got {len(ohlcv_data)}")
            return
        
        # Extract OHLCV data
        timestamps = [candle[0] for candle in ohlcv_data]
        opens = [candle[1] for candle in ohlcv_data]
        highs = [candle[2] for candle in ohlcv_data]
        lows = [candle[3] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        volumes = [candle[5] for candle in ohlcv_data]
        
        results = {'ranges': [], 'phases': [], 'events': [], 'evr_signals': [], 'signals': []}
        
        # Detect trading ranges first
        ranges = self.detect_ranges(highs, lows)
        results['ranges'] = ranges
        
        # Detect phases within ranges and trends
        phases = []
        for i in range(0, len(closes), 50):  # Analyze in 50-candle segments
            end_idx = min(i + 50, len(closes))
            if end_idx - i < 20:  # Skip too small segments
                continue
                
            range_info = {'high': max(highs[i:end_idx]), 'low': min(lows[i:end_idx])}
            phase = self.detect_phase(closes, volumes, range_info, i, end_idx)
            
            phases.append({
                'start_index': i,
                'end_index': end_idx,
                'phase': phase,
                'price_range': range_info
            })
        
        results['phases'] = phases
        
        # Detect Wyckoff events if requested
        if 'events' in detect:
            events = self.detect_wyckoff_events(opens, highs, lows, closes, volumes)
            results['events'] = events
        
        # Effort vs Result analysis
        evr_signals = self.effort_vs_result_analysis(closes, volumes)
        results['evr_signals'] = evr_signals
        
        # Generate trading signals
        signals = self.generate_wyckoff_signals(results['events'], results['phases'], 
                                               evr_signals, closes, timestamps)
        results['signals'] = signals
        
        # Display results
        self._display_results(results, timestamps, closes)
    
    def _display_results(self, results: Dict, timestamps: List[int], closes: List[float]):
        """Display analysis results in terminal format."""
        
        # Recent Wyckoff Signals
        recent_signals = results['signals'][-10:] if results['signals'] else []
        
        if recent_signals:
            for signal in recent_signals:
                dt = datetime.fromtimestamp(signal['timestamp'] / 1000)
                
                # Phase and events display
                phase_emoji = self._get_phase_emoji(signal['phase'])
                events_str = ", ".join(signal['events']) if signal['events'] else "None"
                
                if signal['signal'] != 'NONE':
                    signal_emoji = "📈" if signal['signal'] == 'BUY' else "📉"
                    print(f"[{dt.strftime('%Y-%m-%d %H:%M')}] {signal_emoji} {signal['signal']} "
                          f"({signal['confidence']}%) | {phase_emoji} {signal['phase']} | "
                          f"Setup: {signal['setup_type']} | Events: {events_str}")
                else:
                    print(f"[{dt.strftime('%Y-%m-%d %H:%M')}] {phase_emoji} {signal['phase']} | "
                          f"Events: {events_str}")
        
        # Key Events Summary
        if results['events']:
            print(f"\n🔍 KEY WYCKOFF EVENTS (Last 5):")
            recent_events = results['events'][-5:]
            for event in recent_events:
                dt = datetime.fromtimestamp(timestamps[event['index']] / 1000)
                event_emoji = self._get_event_emoji(event['type'])
                print(f"  {event_emoji} {event['type']}: {event['name']} "
                      f"({dt.strftime('%m-%d %H:%M')}) @ ${event['price']:.4f}")
        
        # Phase Distribution
        if results['phases']:
            phase_counts = {}
            for phase in results['phases']:
                phase_name = phase['phase']
                phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1
            
            print(f"\n📊 PHASE DISTRIBUTION:")
            for phase, count in phase_counts.items():
                phase_emoji = self._get_phase_emoji(phase)
                print(f"  {phase_emoji} {phase}: {count} segments")
        
        # EVR Anomalies
        if results['evr_signals']:
            print(f"\n⚖️  EFFORT vs RESULT ANOMALIES (Last 3):")
            recent_evr = results['evr_signals'][-3:]
            for evr in recent_evr:
                dt = datetime.fromtimestamp(timestamps[evr['index']] / 1000)
                evr_emoji = "🔴" if evr['type'] == 'absorption' else "🟡"
                print(f"  {evr_emoji} {evr['type'].upper()}: {evr['signal']} "
                      f"({dt.strftime('%m-%d %H:%M')})")
    
    def _get_phase_emoji(self, phase: str) -> str:
        """Get emoji for phase type."""
        phase_emojis = {
            'Accumulation': '📦',
            'Markup': '🚀', 
            'Distribution': '📤',
            'Markdown': '📉',
            'Consolidation': '➖',
            'Transition': '🔄'
        }
        return phase_emojis.get(phase, '❓')
    
    def _get_event_emoji(self, event_type: str) -> str:
        """Get emoji for event type."""
        event_emojis = {
            'SC': '💥',    # Selling Climax
            'AR': '⬆️',     # Automatic Rally
            'ST': '🔍',    # Secondary Test
            'Spring': '🌸', # Spring
            'UTAD': '⚡',   # Upthrust After Distribution
            'LPS': '✅',   # Last Point of Support
            'LPSY': '❌'   # Last Point of Supply
        }
        return event_emojis.get(event_type, '📍')


def parse_command(command: str) -> Tuple[str, str, int, str]:
    """
    Parse terminal command: wyckoff s=BTC/USDT t=4h l=600 detect=events+phases
    
    Args:
        command: Command string
        
    Returns:
        Tuple of (symbol, timeframe, limit, detect)
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'wyckoff':
        raise ValueError("Invalid command format. Use: wyckoff s=SYMBOL t=TIMEFRAME l=LIMIT [detect=TYPE]")
    
    symbol = None
    timeframe = '4h'  # default
    limit = 600  # default
    detect = 'phases+events'  # default
    
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
        elif part.startswith('detect='):
            detect = part[7:]
    
    if symbol is None:
        raise ValueError("Symbol (s=) is required")
    
    return symbol, timeframe, limit, detect


def main():
    """Main entry point for Wyckoff analysis."""
    if len(sys.argv) < 2:
        print("Usage: python wyckoff.py s=SYMBOL t=TIMEFRAME l=LIMIT [detect=TYPE]")
        print("Example: python wyckoff.py s=BTC/USDT t=4h l=600")
        print("Example: python wyckoff.py s=ETH/USDT t=1D l=720 detect=events+phases")
        print("Detect options: 'phases', 'events', 'phases+events'")
        return
    
    try:
        # Parse command line arguments
        command = ' '.join(['wyckoff'] + sys.argv[1:])
        symbol, timeframe, limit, detect = parse_command(command)
        
        # Run analysis
        analyzer = WyckoffAnalyzer()
        analyzer.analyze(symbol, timeframe, limit, detect)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()