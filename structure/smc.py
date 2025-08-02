"""
Smart Money Concepts (SMC) analysis implementation for cryptocurrency markets.
Detects liquidity zones, order blocks, break of structure (BOS), and change of character (CHOCH).
"""

from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from data import get_data_collector


class OutputFormatter:
    """Standardized output formatter for trend analysis results."""
    
    @staticmethod
    def format_analysis_output(timestamp: int, metrics: Dict[str, Any], 
                              signal: str, trend: str, 
                              price: Optional[float] = None,
                              symbol: Optional[str] = None,
                              timeframe: Optional[str] = None,
                              multiline: bool = False) -> str:
        """
        Format analysis output with standardized structure.
        
        Args:
            timestamp: Unix timestamp in milliseconds
            metrics: Dictionary of metric names and values
            signal: Trading signal (BUY, SELL, HOLD, NONE, etc.)
            trend: Trend direction (BULLISH, BEARISH, NEUTRAL, etc.)
            price: Optional current price
            symbol: Optional trading symbol (e.g., BTC/USDT)
            timeframe: Optional chart timeframe (e.g., 1h, 4h, 1d)
            multiline: If True, format with better visual hierarchy
            
        Returns:
            Formatted output string (single or multi-line)
        """
        # Convert timestamp to readable format
        dt = datetime.fromtimestamp(timestamp / 1000)
        timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Build context string (symbol/timeframe)
        context_parts = []
        if symbol:
            context_parts.append(symbol)
        if timeframe:
            context_parts.append(timeframe)
        context_str = f"[{' | '.join(context_parts)}] " if context_parts else ""
        
        # Format metrics with angle brackets
        metric_parts = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if abs(value) >= 1000:
                    metric_parts.append(f"<{key}>: {value:.0f}")
                elif abs(value) >= 1:
                    metric_parts.append(f"<{key}>: {value:.2f}")
                else:
                    metric_parts.append(f"<{key}>: {value:.4f}")
            else:
                metric_parts.append(f"<{key}>: {value}")
        
        # Get trend emoji
        trend_emoji = OutputFormatter.get_trend_emoji(trend)
        
        if multiline:
            # Multi-line format with better visual hierarchy
            lines = []
            lines.append(f"┌─ {context_str}[{timestamp_str}]")
            
            # Group metrics by category for better readability
            primary_metrics = []
            secondary_metrics = []
            
            for part in metric_parts:
                key = part.split(":")[0].replace("<", "").replace(">", "")
                if key in ['PRICE', 'SIGNAL', 'CONFIDENCE', 'FIB_LEVEL', 'PATTERN', 'BOS', 'CHOCH']:
                    primary_metrics.append(part)
                else:
                    secondary_metrics.append(part)
            
            if primary_metrics:
                lines.append(f"├─ Primary: {' | '.join(primary_metrics)}")
            if secondary_metrics:
                lines.append(f"├─ Metrics: {' | '.join(secondary_metrics)}")
            
            if price is not None:
                lines.append(f"├─ Price: {price:.4f}")
            
            lines.append(f"└─ Signal: {signal} | {trend_emoji} {trend}")
            
            return "\n".join(lines)
        else:
            # Single-line format (compatible with existing code)
            metrics_str = " | ".join(metric_parts)
            
            # Add price if provided
            if price is not None:
                price_str = f" | <PRICE>: {price:.4f}"
            else:
                price_str = ""
            
            return f"{context_str}[{timestamp_str}] {metrics_str}{price_str} | Signal: {signal} | {trend_emoji} {trend}"
    
    @staticmethod
    def get_trend_emoji(trend: str) -> str:
        """
        Get emoji for trend direction.
        
        Args:
            trend: Trend direction string
            
        Returns:
            Corresponding emoji
        """
        trend_upper = trend.upper()
        
        if trend_upper in ['BULLISH', 'BUY', 'UP', 'LONG']:
            return "📈"
        elif trend_upper in ['BEARISH', 'SELL', 'DOWN', 'SHORT']:
            return "📉"
        elif trend_upper in ['NEUTRAL', 'SIDEWAYS', 'CONSOLIDATION']:
            return "➖"
        elif trend_upper in ['STRONG_BULLISH', 'VERY_BULLISH']:
            return "🚀"
        elif trend_upper in ['STRONG_BEARISH', 'VERY_BEARISH']:
            return "🔻"
        elif trend_upper in ['VOLATILE', 'CHOPPY']:
            return "🌊"
        elif trend_upper in ['UNCERTAIN', 'MIXED']:
            return "❓"
        else:
            return "➖"  # Default to neutral
    
    @staticmethod
    def format_smc_output(timestamp: int, bos: bool, choch: bool, ob_hit: bool,
                         signal: str, trend: str, confidence: int = 0,
                         price: Optional[float] = None, symbol: Optional[str] = None,
                         timeframe: Optional[str] = None, multiline: bool = False) -> str:
        """
        Format SMC (Smart Money Concepts) analysis output.
        
        Args:
            timestamp: Unix timestamp in milliseconds
            bos: Break of Structure detected
            choch: Change of Character detected
            ob_hit: Order Block hit
            signal: Trading signal
            trend: Trend direction
            confidence: Confidence percentage
            price: Optional current price
            symbol: Optional trading symbol
            timeframe: Optional chart timeframe
            multiline: If True, use multi-line format
            
        Returns:
            Formatted output string
        """
        metrics = {
            "BOS": "YES" if bos else "NO",
            "CHOCH": "YES" if choch else "NO",
            "OB_HIT": "YES" if ob_hit else "NO"
        }
        
        if confidence > 0:
            metrics["CONFIDENCE"] = f"{confidence}%"
        
        return OutputFormatter.format_analysis_output(
            timestamp, metrics, signal, trend, price, symbol, timeframe, multiline
        )


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
    
    def analyze(self, symbol: str, timeframe: str, limit: int, 
                zones: bool = False, choch: bool = False) -> str:
        """
        Perform SMC analysis and return results.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            limit: Number of candles to analyze (minimum 200)
            zones: Whether to show liquidity zones details
            choch: Whether to show CHOCH details
            
        Returns:
            Formatted analysis string
        """
        if limit < 200:
            limit = 200
        
        # Fetch OHLCV data
        ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 200:
            return f"Error: Need at least 200 candles for SMC analysis. Got {len(ohlcv_data)}"
        
        # Extract OHLCV data
        timestamps = [candle[0] for candle in ohlcv_data]
        opens = [candle[1] for candle in ohlcv_data]
        highs = [candle[2] for candle in ohlcv_data]
        lows = [candle[3] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        
        # Detect SMC components
        swing_highs, swing_lows = self.find_swing_points(highs, lows)
        liquidity_zones = self.detect_liquidity_zones(swing_highs, swing_lows, highs, lows)
        order_blocks = self.detect_order_blocks(opens, highs, lows, closes)
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
            return "No SMC signals detected in recent data"
        
        # Build output string
        output_parts = []
        
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
            output_parts.append(formatted_output)
        
        # Additional details if requested
        if zones and liquidity_zones:
            output_parts.append(f"\n📍 Liquidity Zones Detected: {len(liquidity_zones)}")
            for lz in liquidity_zones[-3:]:  # Show last 3
                output_parts.append(f"  {lz['type'].upper()}: ${lz['price']:.4f} ({lz['equal_count']} equal levels)")
        
        if choch and choch_events:
            output_parts.append(f"\n🔄 Recent CHOCH Events: {len(choch_events[-5:])}")
            for ch in choch_events[-3:]:  # Show last 3
                dt = datetime.fromtimestamp(timestamps[ch['index']] / 1000)
                output_parts.append(f"  {ch['type'].upper()}: {dt.strftime('%m-%d %H:%M')} at ${ch['break_price']:.4f}")
        
        return "\n".join(output_parts)


