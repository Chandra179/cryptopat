"""
Smart Money Concepts (SMC) analysis implementation for cryptocurrency markets.
Detects liquidity zones, order blocks, break of structure (BOS), and change of character (CHOCH).
"""

import sys
from datetime import datetime
from typing import List, Tuple
from data import get_data_collector
from trend.output_formatter import OutputFormatter


class SMCStrategy:
    """Smart Money Concepts strategy for market structure analysis."""
    
    def __init__(self):
        self.collector = get_data_collector()
    
    def find_swing_points(self, highs: List[float], lows: List[float], 
                         lookback: int = 5) -> Tuple[List[dict], List[dict]]:
        """
        Find swing highs and swing lows.
        
        Args:
            highs: List of high prices
            lows: List of low prices
            lookback: Number of candles to look back/forward for validation
            
        Returns:
            Tuple of (swing_highs, swing_lows) as list of dicts
        """
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(highs) - lookback):
            # Check for swing high
            is_swing_high = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and highs[j] >= highs[i]:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append({
                    'index': i,
                    'price': highs[i],
                    'type': 'swing_high'
                })
            
            # Check for swing low
            is_swing_low = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and lows[j] <= lows[i]:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append({
                    'index': i,
                    'price': lows[i],
                    'type': 'swing_low'
                })
        
        return swing_highs, swing_lows
    
    def detect_liquidity_zones(self, swing_highs: List[dict], swing_lows: List[dict],
                              _highs: List[float], _lows: List[float]) -> List[dict]:
        """
        Detect liquidity zones (equal highs/lows and recent swing points).
        
        Args:
            swing_highs: List of swing high points
            swing_lows: List of swing low points
            _highs: List of high prices (unused in current implementation)
            _lows: List of low prices (unused in current implementation)
            
        Returns:
            List of liquidity zone dictionaries
        """
        liquidity_zones = []
        tolerance = 0.002  # 0.2% tolerance for "equal" levels
        
        # Find equal highs (buy-side liquidity)
        for i in range(len(swing_highs)):
            equal_count = 1
            base_price = swing_highs[i]['price']
            
            for j in range(i + 1, len(swing_highs)):
                if abs(swing_highs[j]['price'] - base_price) / base_price <= tolerance:
                    equal_count += 1
            
            if equal_count >= 2:  # At least 2 equal highs
                liquidity_zones.append({
                    'index': swing_highs[i]['index'],
                    'price': base_price,
                    'type': 'buy_side_liquidity',
                    'zone': 'above_highs',
                    'equal_count': equal_count
                })
        
        # Find equal lows (sell-side liquidity)
        for i in range(len(swing_lows)):
            equal_count = 1
            base_price = swing_lows[i]['price']
            
            for j in range(i + 1, len(swing_lows)):
                if abs(swing_lows[j]['price'] - base_price) / base_price <= tolerance:
                    equal_count += 1
            
            if equal_count >= 2:  # At least 2 equal lows
                liquidity_zones.append({
                    'index': swing_lows[i]['index'],
                    'price': base_price,
                    'type': 'sell_side_liquidity',
                    'zone': 'below_lows',
                    'equal_count': equal_count
                })
        
        return liquidity_zones
    
    def detect_order_blocks(self, opens: List[float], highs: List[float], 
                           lows: List[float], closes: List[float], 
                           volumes: List[float]) -> List[dict]:
        """
        Detect order blocks (last bullish/bearish candle before strong move).
        
        Args:
            opens: List of open prices
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices
            volumes: List of volume values
            
        Returns:
            List of order block dictionaries
        """
        order_blocks = []
        
        for i in range(1, len(closes) - 1):
            # Calculate body size and volume metrics
            body_size = abs(closes[i] - opens[i])
            avg_body = sum(abs(closes[j] - opens[j]) for j in range(max(0, i-10), i)) / min(i, 10)
            
            # Check if this is a large-bodied candle
            if body_size < avg_body * 1.5:
                continue
            
            # Check for volume surge
            avg_volume = sum(volumes[max(0, i-10):i]) / min(i, 10) if i > 0 else volumes[i]
            volume_surge = volumes[i] > avg_volume * 1.3
            
            # Look for strong move after this candle
            next_candle_move = abs(closes[i+1] - opens[i+1])
            strong_move = next_candle_move > avg_body * 1.2
            
            if strong_move and volume_surge:
                # Determine if bullish or bearish order block
                if closes[i] > opens[i] and closes[i+1] > closes[i]:  # Bullish OB
                    order_blocks.append({
                        'index': i,
                        'type': 'bullish_ob',
                        'high': highs[i],
                        'low': lows[i],
                        'open': opens[i],
                        'close': closes[i],
                        'volume_surge': volume_surge,
                        'body_size': body_size
                    })
                elif closes[i] < opens[i] and closes[i+1] < closes[i]:  # Bearish OB
                    order_blocks.append({
                        'index': i,
                        'type': 'bearish_ob',
                        'high': highs[i],
                        'low': lows[i],
                        'open': opens[i],
                        'close': closes[i],
                        'volume_surge': volume_surge,
                        'body_size': body_size
                    })
        
        return order_blocks
    
    def detect_bos(self, swing_highs: List[dict], swing_lows: List[dict], 
                   closes: List[float]) -> List[dict]:
        """
        Detect Break of Structure (BOS) - when price closes above/below previous swing.
        
        Args:
            swing_highs: List of swing high points
            swing_lows: List of swing low points
            closes: List of close prices
            
        Returns:
            List of BOS events
        """
        bos_events = []
        
        # Check for bullish BOS (close above previous swing high)
        for high in swing_highs:
            high_idx = high['index']
            high_price = high['price']
            
            # Look for closes after this swing high
            for i in range(high_idx + 1, len(closes)):
                if closes[i] > high_price:
                    bos_events.append({
                        'index': i,
                        'type': 'bullish_bos',
                        'broken_level': high_price,
                        'break_price': closes[i],
                        'swing_index': high_idx
                    })
                    break  # Only record first break
        
        # Check for bearish BOS (close below previous swing low)
        for low in swing_lows:
            low_idx = low['index']
            low_price = low['price']
            
            # Look for closes after this swing low
            for i in range(low_idx + 1, len(closes)):
                if closes[i] < low_price:
                    bos_events.append({
                        'index': i,
                        'type': 'bearish_bos',
                        'broken_level': low_price,
                        'break_price': closes[i],
                        'swing_index': low_idx
                    })
                    break  # Only record first break
        
        return bos_events
    
    def detect_choch(self, swing_highs: List[dict], swing_lows: List[dict], 
                     closes: List[float]) -> List[dict]:
        """
        Detect Change of Character (CHOCH) - early reversal signals.
        
        Args:
            swing_highs: List of swing high points
            swing_lows: List of swing low points
            closes: List of close prices
            
        Returns:
            List of CHOCH events
        """
        choch_events = []
        
        # Combine and sort swing points by index
        all_swings = swing_highs + swing_lows
        all_swings.sort(key=lambda x: x['index'])
        
        for i in range(len(all_swings) - 1):
            current_swing = all_swings[i]
            
            # Look for minor swing breaks that indicate character change
            if current_swing['type'] == 'swing_high':
                # Look for break below recent minor low
                for j in range(current_swing['index'] + 1, min(current_swing['index'] + 20, len(closes))):
                    # Find recent minor lows
                    recent_lows = [s for s in all_swings if s['type'] == 'swing_low' 
                                  and s['index'] > current_swing['index'] - 10 
                                  and s['index'] < current_swing['index']]
                    
                    if recent_lows:
                        minor_low = min(recent_lows, key=lambda x: x['price'])['price']
                        if closes[j] < minor_low:
                            choch_events.append({
                                'index': j,
                                'type': 'bearish_choch',
                                'broken_level': minor_low,
                                'break_price': closes[j],
                                'swing_index': current_swing['index']
                            })
                            break
            
            elif current_swing['type'] == 'swing_low':
                # Look for break above recent minor high
                for j in range(current_swing['index'] + 1, min(current_swing['index'] + 20, len(closes))):
                    # Find recent minor highs
                    recent_highs = [s for s in all_swings if s['type'] == 'swing_high' 
                                   and s['index'] > current_swing['index'] - 10 
                                   and s['index'] < current_swing['index']]
                    
                    if recent_highs:
                        minor_high = max(recent_highs, key=lambda x: x['price'])['price']
                        if closes[j] > minor_high:
                            choch_events.append({
                                'index': j,
                                'type': 'bullish_choch',
                                'broken_level': minor_high,
                                'break_price': closes[j],
                                'swing_index': current_swing['index']
                            })
                            break
        
        return choch_events
    
    def generate_signals(self, liquidity_zones: List[dict], order_blocks: List[dict],
                        bos_events: List[dict], choch_events: List[dict],
                        closes: List[float], timestamps: List[int]) -> List[dict]:
        """
        Generate SMC trading signals based on confluence of factors.
        
        Args:
            liquidity_zones: Detected liquidity zones
            order_blocks: Detected order blocks
            bos_events: Break of structure events
            choch_events: Change of character events
            closes: Close prices
            timestamps: Timestamps
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Look for signal confluence in recent candles
        recent_window = min(50, len(closes))
        
        for i in range(len(closes) - recent_window, len(closes)):
            signal = {
                'index': i,
                'timestamp': timestamps[i],
                'close': closes[i],
                'bos': False,
                'choch': False,
                'ob_hit': False,
                'liquidity_sweep': False,
                'signal': 'NONE',
                'trend': 'NEUTRAL',
                'confidence': 0
            }
            
            # Check for BOS at this index
            bos_here = [b for b in bos_events if b['index'] == i]
            if bos_here:
                signal['bos'] = True
                signal['bos_type'] = bos_here[0]['type']
            
            # Check for CHOCH at this index
            choch_here = [c for c in choch_events if c['index'] == i]
            if choch_here:
                signal['choch'] = True
                signal['choch_type'] = choch_here[0]['type']
            
            # Check for order block retest (price near OB levels)
            for ob in order_blocks:
                if abs(i - ob['index']) <= 10:  # Within 10 candles of OB
                    ob_low, ob_high = ob['low'], ob['high']
                    if ob_low <= closes[i] <= ob_high:
                        signal['ob_hit'] = True
                        signal['ob_type'] = ob['type']
            
            # Check for liquidity sweep
            for lz in liquidity_zones:
                price_diff = abs(closes[i] - lz['price']) / lz['price']
                if price_diff <= 0.005:  # Within 0.5% of liquidity zone
                    signal['liquidity_sweep'] = True
                    signal['sweep_type'] = lz['type']
            
            # Generate trading signals based on confluence
            if signal['choch'] and signal['ob_hit']:
                if (signal.get('choch_type') == 'bullish_choch' and 
                    signal.get('ob_type') == 'bullish_ob'):
                    signal['signal'] = 'BUY'
                    signal['trend'] = 'BULLISH'
                    signal['confidence'] = 75
                    
                    if signal['liquidity_sweep']:
                        signal['confidence'] = 90
                
                elif (signal.get('choch_type') == 'bearish_choch' and 
                      signal.get('ob_type') == 'bearish_ob'):
                    signal['signal'] = 'SELL'
                    signal['trend'] = 'BEARISH' 
                    signal['confidence'] = 75
                    
                    if signal['liquidity_sweep']:
                        signal['confidence'] = 90
            
            # Add signal if it has trading value
            if signal['signal'] != 'NONE' or signal['bos'] or signal['choch']:
                signals.append(signal)
        
        return signals
    
    def analyze(self, symbol: str, timeframe: str, limit: int, 
                zones: bool = False, choch: bool = False) -> None:
        """
        Perform SMC analysis and display results.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            limit: Number of candles to analyze (minimum 200)
            zones: Whether to show liquidity zones details
            choch: Whether to show CHOCH details
        """
        if limit < 200:
            print(f"Warning: SMC requires ≥200 candles for structure context. Using minimum 200.")
            limit = 200
        
        # Fetch OHLCV data
        ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 200:
            print(f"Error: Need at least 200 candles for SMC analysis. Got {len(ohlcv_data)}")
            return
        
        # Extract OHLCV data
        timestamps = [candle[0] for candle in ohlcv_data]
        opens = [candle[1] for candle in ohlcv_data]
        highs = [candle[2] for candle in ohlcv_data]
        lows = [candle[3] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        volumes = [candle[5] for candle in ohlcv_data]
        
        # Detect SMC components
        swing_highs, swing_lows = self.find_swing_points(highs, lows)
        liquidity_zones = self.detect_liquidity_zones(swing_highs, swing_lows, highs, lows)
        order_blocks = self.detect_order_blocks(opens, highs, lows, closes, volumes)
        bos_events = self.detect_bos(swing_highs, swing_lows, closes)
        choch_events = self.detect_choch(swing_highs, swing_lows, closes)
        
        # Generate signals
        signals = self.generate_signals(liquidity_zones, order_blocks, bos_events, 
                                       choch_events, closes, timestamps)
        
        # Display recent signals
        recent_signals = []
        
        for signal in signals[-10:]:  # Last 10 signals
            dt = datetime.fromtimestamp(signal['timestamp'] / 1000)
            recent_signals.append((signal, dt))
        
        if not recent_signals:
            print("No SMC signals detected in recent data")
            return
        
        # Display signals using standardized formatter
        for signal, dt in recent_signals:
            formatted_output = OutputFormatter.format_smc_output(
                timestamp=signal['timestamp'],
                bos=signal['bos'],
                choch=signal['choch'],
                ob_hit=signal['ob_hit'],
                signal=signal['signal'],
                trend=signal['trend'],
                confidence=signal['confidence'] if signal['confidence'] > 0 else 0,
                price=signal['close']
            )
            print(formatted_output)
        
        # Additional details if requested
        if zones and liquidity_zones:
            print(f"\n📍 Liquidity Zones Detected: {len(liquidity_zones)}")
            for lz in liquidity_zones[-3:]:  # Show last 3
                print(f"  {lz['type'].upper()}: ${lz['price']:.4f} ({lz['equal_count']} equal levels)")
        
        if choch and choch_events:
            print(f"\n🔄 Recent CHOCH Events: {len(choch_events[-5:])}")
            for ch in choch_events[-3:]:  # Show last 3
                dt = datetime.fromtimestamp(timestamps[ch['index']] / 1000)
                print(f"  {ch['type'].upper()}: {dt.strftime('%m-%d %H:%M')} at ${ch['break_price']:.4f}")


def parse_command(command: str) -> Tuple[str, str, int, bool, bool]:
    """
    Parse terminal command: smc s=BTC/USDT t=1h l=300 zones=true choch=true
    
    Args:
        command: Command string
        
    Returns:
        Tuple of (symbol, timeframe, limit, zones, choch)
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'smc':
        raise ValueError("Invalid command format. Use: smc s=SYMBOL t=TIMEFRAME l=LIMIT [zones=true] [choch=true]")
    
    symbol = None
    timeframe = '1h'  # default
    limit = 300  # default
    zones = False
    choch = False
    
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
        elif part.startswith('zones='):
            zones = part[6:].lower() == 'true'
        elif part.startswith('choch='):
            choch = part[6:].lower() == 'true'
    
    if symbol is None:
        raise ValueError("Symbol (s=) is required")
    
    return symbol, timeframe, limit, zones, choch


def main():
    """Main entry point for SMC analysis."""
    if len(sys.argv) < 2:
        print("Usage: python smc.py s=SYMBOL t=TIMEFRAME l=LIMIT [zones=true] [choch=true]")
        print("Example: python smc.py s=BTC/USDT t=1h l=300")
        print("Example: python smc.py s=ETH/USDT t=4h l=500 zones=true choch=true")
        return
    
    try:
        # Parse command line arguments
        command = ' '.join(['smc'] + sys.argv[1:])
        symbol, timeframe, limit, zones, choch = parse_command(command)
        
        # Run analysis
        strategy = SMCStrategy()
        strategy.analyze(symbol, timeframe, limit, zones, choch)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()