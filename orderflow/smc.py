"""
Smart Money Concepts (SMC) analysis implementation for cryptocurrency markets.
Detects liquidity zones, order blocks, break of structure (BOS), and change of character (CHOCH).
"""

from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional
from data import get_data_collector


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
                              highs: List[float], lows: List[float]) -> List[dict]:
        """
        Detect liquidity zones (equal highs/lows and recent swing points).
        
        Args:
            swing_highs: List of swing high points
            swing_lows: List of swing low points
            highs: List of high prices (unused in current implementation)
            lows: List of low prices (unused in current implementation)
            
        Returns:
            List of liquidity zone dictionaries
        """
        liquidity_zones = []
        tolerance = 0.005  # 0.5% tolerance for "equal" levels (better for crypto volatility)
        
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
                           lows: List[float], closes: List[float]) -> List[dict]:
        """
        Detect order blocks using standard SMC methodology.
        Order block = last opposing candle before structure break.
        
        Args:
            opens: List of open prices
            highs: List of high prices
            lows: List of low prices
            closes: List of close prices
            
        Returns:
            List of order block dictionaries
        """
        order_blocks = []
        
        for i in range(1, len(closes) - 3):  # Need at least 3 candles ahead
            current_candle_bullish = closes[i] > opens[i]
            current_candle_bearish = closes[i] < opens[i]
            
            # Skip doji candles (very small body)
            body_size = abs(closes[i] - opens[i])
            if body_size < (highs[i] - lows[i]) * 0.3:  # Body < 30% of total range
                continue
            
            # Look ahead for structure break (impulse move)
            strong_bearish_move = False
            strong_bullish_move = False
            
            # Check next 3 candles for impulse
            for j in range(i + 1, min(i + 4, len(closes))):
                # Bearish impulse: closes below current low
                if closes[j] < lows[i]:
                    strong_bearish_move = True
                    break
                # Bullish impulse: closes above current high  
                if closes[j] > highs[i]:
                    strong_bullish_move = True
                    break
            
            # Bullish Order Block: Last bearish candle before bullish impulse
            if current_candle_bearish and strong_bullish_move:
                order_blocks.append({
                    'index': i,
                    'type': 'bullish_ob',
                    'high': highs[i],
                    'low': lows[i], 
                    'open': opens[i],
                    'close': closes[i],
                    'body_size': body_size
                })
            
            # Bearish Order Block: Last bullish candle before bearish impulse
            elif current_candle_bullish and strong_bearish_move:
                order_blocks.append({
                    'index': i,
                    'type': 'bearish_ob',
                    'high': highs[i],
                    'low': lows[i],
                    'open': opens[i], 
                    'close': closes[i],
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
        Detect Change of Character (CHOCH) using standard SMC methodology.
        CHOCH = break of most recent swing in opposite direction to trend.
        
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
        
        # Determine trend by comparing recent swings
        if len(all_swings) < 4:
            return choch_events
        
        for i in range(2, len(all_swings)):  # Need at least 2 previous swings
            current_swing = all_swings[i]
            prev_swing = all_swings[i-1]
            
            # Skip if swings are same type
            if current_swing['type'] == prev_swing['type']:
                continue
                
            # Determine if this creates a CHOCH
            if current_swing['type'] == 'swing_high' and prev_swing['type'] == 'swing_low':
                # Bearish CHOCH: New swing high is lower than previous swing high
                prev_high = None
                for j in range(i-1, -1, -1):
                    if all_swings[j]['type'] == 'swing_high':
                        prev_high = all_swings[j]
                        break
                
                if prev_high and current_swing['price'] < prev_high['price']:
                    # Look for close below the swing low that preceded this high
                    target_low = prev_swing['price']
                    for k in range(current_swing['index'] + 1, min(current_swing['index'] + 10, len(closes))):
                        if closes[k] < target_low:
                            choch_events.append({
                                'index': k,
                                'type': 'bearish_choch',
                                'broken_level': target_low,
                                'break_price': closes[k],
                                'swing_index': current_swing['index']
                            })
                            break
            
            elif current_swing['type'] == 'swing_low' and prev_swing['type'] == 'swing_high':
                # Bullish CHOCH: New swing low is higher than previous swing low  
                prev_low = None
                for j in range(i-1, -1, -1):
                    if all_swings[j]['type'] == 'swing_low':
                        prev_low = all_swings[j]
                        break
                
                if prev_low and current_swing['price'] > prev_low['price']:
                    # Look for close above the swing high that preceded this low
                    target_high = prev_swing['price']
                    for k in range(current_swing['index'] + 1, min(current_swing['index'] + 10, len(closes))):
                        if closes[k] > target_high:
                            choch_events.append({
                                'index': k,
                                'type': 'bullish_choch',
                                'broken_level': target_high,
                                'break_price': closes[k],
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
            
            # Generate individual component signals (standard SMC approach)
            confidence = 0
            
            # BOS signals (strong momentum)
            if signal['bos']:
                if signal.get('bos_type') == 'bullish_bos':
                    signal['signal'] = 'BUY'
                    signal['trend'] = 'BULLISH'
                    confidence = 60
                elif signal.get('bos_type') == 'bearish_bos':
                    signal['signal'] = 'SELL'
                    signal['trend'] = 'BEARISH'
                    confidence = 60
            
            # CHOCH signals (early reversal)
            elif signal['choch']:
                if signal.get('choch_type') == 'bullish_choch':
                    signal['signal'] = 'BUY'
                    signal['trend'] = 'BULLISH'
                    confidence = 50
                elif signal.get('choch_type') == 'bearish_choch':
                    signal['signal'] = 'SELL'
                    signal['trend'] = 'BEARISH'
                    confidence = 50
            
            # Order Block retest signals
            elif signal['ob_hit']:
                if signal.get('ob_type') == 'bullish_ob':
                    signal['signal'] = 'BUY'
                    signal['trend'] = 'BULLISH'
                    confidence = 45
                elif signal.get('ob_type') == 'bearish_ob':
                    signal['signal'] = 'SELL'
                    signal['trend'] = 'BEARISH'
                    confidence = 45
            
            # Liquidity sweep signals
            elif signal['liquidity_sweep']:
                if signal.get('sweep_type') == 'sell_side_liquidity':
                    signal['signal'] = 'BUY'  # Sweep below = bullish
                    signal['trend'] = 'BULLISH'
                    confidence = 40
                elif signal.get('sweep_type') == 'buy_side_liquidity':
                    signal['signal'] = 'SELL'  # Sweep above = bearish
                    signal['trend'] = 'BEARISH'
                    confidence = 40
            
            # Confluence bonuses
            confluence_count = sum([signal['bos'], signal['choch'], signal['ob_hit'], signal['liquidity_sweep']])
            if confluence_count >= 2:
                confidence += 20  # Bonus for multiple confirmations
            
            signal['confidence'] = confidence
            
            # Add signal if it has any SMC component
            if signal['signal'] != 'NONE' or signal['bos'] or signal['choch'] or signal['ob_hit'] or signal['liquidity_sweep']:
                signals.append(signal)
        
        return signals
    
    def analyze(self, symbol: str, timeframe: str, limit: int, ohlcv_data: Optional[List] = None) -> Dict:
        """
        Perform SMC analysis and return results.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            limit: Number of candles to analyze (minimum 200)
            
        Returns:
            Analysis results dictionary
        """
        try:
            if limit < 200:
                limit = 200
            
            # Fetch OHLCV data if not provided
            if ohlcv_data is None:
                ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
            
            if not ohlcv_data or len(ohlcv_data) < 200:
                return {
                    'error': f'Insufficient data: need at least 200 candles, got {len(ohlcv_data) if ohlcv_data else 0}',
                    'success': False,
                    'symbol': symbol,
                    'timeframe': timeframe
                }
            
            # Extract OHLCV data
            timestamps = [candle[0] for candle in ohlcv_data]
            opens = [candle[1] for candle in ohlcv_data]
            highs = [candle[2] for candle in ohlcv_data]
            lows = [candle[3] for candle in ohlcv_data]
            closes = [candle[4] for candle in ohlcv_data]
            
            # Get current price and timestamp info
            current_price = closes[-1]
            current_timestamp = timestamps[-1]
            dt = datetime.fromtimestamp(current_timestamp / 1000, tz=timezone.utc)
            
            # Detect SMC components
            swing_highs, swing_lows = self.find_swing_points(highs, lows)
            liquidity_zones = self.detect_liquidity_zones(swing_highs, swing_lows, highs, lows)
            order_blocks = self.detect_order_blocks(opens, highs, lows, closes)
            bos_events = self.detect_bos(swing_highs, swing_lows, closes)
            choch_events = self.detect_choch(swing_highs, swing_lows, closes)
            
            # Generate signals
            signals = self.generate_signals(liquidity_zones, order_blocks, bos_events, 
                                           choch_events, closes, timestamps)
            
            # Get the most recent signal
            latest_signal = signals[-1] if signals else None
            
            result = {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'analysis_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': current_timestamp,
                'total_candles': len(ohlcv_data),
                'current_price': round(current_price, 4),
                'pattern_detected': latest_signal is not None
            }
            
            if latest_signal:
                # Extract signal information
                signal = latest_signal.get('signal', 'HOLD')
                trend = latest_signal.get('trend', 'NEUTRAL')
                confidence = latest_signal.get('confidence', 0)
                bos = latest_signal.get('bos', False)
                choch = latest_signal.get('choch', False)
                ob_hit = latest_signal.get('ob_hit', False)
                liquidity_sweep = latest_signal.get('liquidity_sweep', False)
                
                # Calculate support/resistance levels based on SMC analysis
                if signal == 'BUY':
                    # Find nearest order block or swing low for support
                    support_level = current_price * 0.985  # Conservative support
                    for ob in order_blocks[-5:]:  # Recent order blocks
                        if ob['type'] == 'bullish_ob' and ob['low'] < current_price:
                            support_level = max(support_level, ob['low'])
                    
                    resistance_level = current_price * 1.02
                    stop_zone = support_level * 0.995
                    tp_low = resistance_level
                    tp_high = resistance_level * 1.015
                    
                elif signal == 'SELL':
                    # Find nearest order block or swing high for resistance
                    resistance_level = current_price * 1.015  # Conservative resistance
                    for ob in order_blocks[-5:]:  # Recent order blocks
                        if ob['type'] == 'bearish_ob' and ob['high'] > current_price:
                            resistance_level = min(resistance_level, ob['high'])
                    
                    support_level = current_price * 0.98
                    stop_zone = resistance_level * 1.005
                    tp_low = support_level
                    tp_high = support_level * 0.985
                    
                else:
                    support_level = current_price * 0.99
                    resistance_level = current_price * 1.01
                    stop_zone = current_price * 0.99
                    tp_low = current_price * 1.01
                    tp_high = current_price * 1.02
                
                # Calculate Risk/Reward ratio
                if signal in ['BUY', 'SELL']:
                    risk = abs(current_price - stop_zone)
                    reward = abs(tp_low - current_price) if tp_low != current_price else abs(tp_high - current_price)
                    rr_ratio = reward / risk if risk > 0 else 0
                else:
                    rr_ratio = 0
                
                # Determine entry window based on confluence
                confluence_count = sum([bos, choch, ob_hit, liquidity_sweep])
                if confluence_count >= 3 and confidence > 70:
                    entry_window = "High confluence - optimal entry"
                elif confluence_count >= 2 and confidence > 60:
                    entry_window = "Good setup - enter on pullback"
                elif confluence_count >= 1 and confidence > 50:
                    entry_window = "Wait for additional confirmation"
                else:
                    entry_window = "Weak setup - avoid entry"
                
                # Exit trigger based on SMC principles
                if bos:
                    exit_trigger = "Structure broken - trend continuation expected"
                elif choch:
                    exit_trigger = "Character change - trend reversal possible"
                elif ob_hit:
                    exit_trigger = "Order block reaction - watch for rejection/continuation"
                else:
                    exit_trigger = "Monitor for SMC structure changes"
                
                # Update result with SMC analysis
                result.update({
                    # SMC specific data
                    'bos_detected': bos,
                    'choch_detected': choch,
                    'order_block_hit': ob_hit,
                    'liquidity_sweep': liquidity_sweep,
                    'confluence_count': confluence_count,
                    
                    # Price levels
                    'support_level': round(support_level, 4),
                    'resistance_level': round(resistance_level, 4),
                    'stop_zone': round(stop_zone, 4),
                    'tp_low': round(tp_low, 4),
                    'tp_high': round(tp_high, 4),
                    
                    # Trading analysis
                    'signal': signal,
                    'trend': trend,
                    'confidence_score': confidence,
                    'entry_window': entry_window,
                    'exit_trigger': exit_trigger,
                    'rr_ratio': round(rr_ratio, 1),
                    
                    # SMC structure counts
                    'swing_highs_count': len(swing_highs),
                    'swing_lows_count': len(swing_lows),
                    'liquidity_zones_count': len(liquidity_zones),
                    'order_blocks_count': len(order_blocks),
                    'bos_events_count': len(bos_events),
                    'choch_events_count': len(choch_events),
                    
                    # Raw data
                    'raw_data': {
                        'ohlcv_data': ohlcv_data,
                        'latest_signal': latest_signal,
                        'recent_signals': signals[-5:] if len(signals) >= 5 else signals
                    }
                })
            else:
                # No SMC signals detected
                result.update({
                    'bos_detected': False,
                    'choch_detected': False,
                    'order_block_hit': False,
                    'liquidity_sweep': False,
                    'confluence_count': 0,
                    'signal': 'HOLD',
                    'trend': 'NEUTRAL',
                    'confidence_score': 0,
                    'entry_window': "No SMC signals detected",
                    'exit_trigger': "Wait for structure development",
                    'support_level': round(current_price * 0.99, 4),
                    'resistance_level': round(current_price * 1.01, 4),
                    'rr_ratio': 0,
                    'swing_highs_count': len(swing_highs),
                    'swing_lows_count': len(swing_lows),
                    'liquidity_zones_count': len(liquidity_zones),
                    'order_blocks_count': len(order_blocks),
                    'bos_events_count': len(bos_events),
                    'choch_events_count': len(choch_events)
                })
            
            return result
            
        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'success': False,
                'symbol': symbol,
                'timeframe': timeframe
            }