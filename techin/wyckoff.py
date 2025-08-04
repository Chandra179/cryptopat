"""
Wyckoff Method implementation for cryptocurrency market structure analysis.
Identifies accumulation, distribution phases and smart money activity patterns.
"""

from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import statistics
from data import get_data_collector


class WyckoffStrategy:
    """Wyckoff Method strategy for market structure and phase analysis."""
    
    def __init__(self):
        self.collector = get_data_collector()
    
    def calculate_volume_spread_analysis(self, ohlcv_data: List[List]) -> List[Dict]:
        """
        Calculate Volume Spread Analysis (VSA) for each candle.
        
        Args:
            ohlcv_data: OHLCV candle data
            
        Returns:
            List of VSA analysis for each candle
        """
        if len(ohlcv_data) < 10:
            return []
            
        vsa_data = []
        
        # Calculate averages for comparison
        volumes = [candle[5] for candle in ohlcv_data]
        avg_volume = statistics.mean(volumes[-20:]) if len(volumes) >= 20 else statistics.mean(volumes)
        
        for i, candle in enumerate(ohlcv_data):
            timestamp, open_price, high, low, close, volume = candle
            
            # Price spread calculations
            spread = high - low
            close_position = (close - low) / spread if spread > 0 else 0.5
            
            # Volume analysis
            relative_volume = volume / avg_volume if avg_volume > 0 else 1.0
            
            # True Range calculation
            if i > 0:
                prev_close = ohlcv_data[i-1][4]
                true_range = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
            else:
                true_range = high - low
            
            # Effort vs Result analysis
            price_change = abs(close - open_price)
            effort_result_ratio = volume / price_change if price_change > 0 else float('inf')
            
            vsa_data.append({
                'timestamp': timestamp,
                'spread': spread,
                'close_position': close_position,
                'relative_volume': relative_volume,
                'true_range': true_range,
                'effort_result_ratio': effort_result_ratio,
                'volume': volume,
                'close': close,
                'high': high,
                'low': low,
                'open': open_price
            })
        
        return vsa_data
    
    def calculate_accumulation_distribution_line(self, ohlcv_data: List[List]) -> List[float]:
        """
        Calculate Accumulation/Distribution Line.
        
        Formula: AD = Previous AD + ((Close - Low) - (High - Close)) / (High - Low) * Volume
        
        Args:
            ohlcv_data: OHLCV candle data
            
        Returns:
            List of A/D line values
        """
        if not ohlcv_data:
            return []
        
        ad_line = []
        cumulative_ad = 0
        
        for candle in ohlcv_data:
            timestamp, open_price, high, low, close, volume = candle
            
            # Money Flow Multiplier
            if high != low:
                mf_multiplier = ((close - low) - (high - close)) / (high - low)
            else:
                mf_multiplier = 0
            
            # Money Flow Volume
            mf_volume = mf_multiplier * volume
            
            # Add to cumulative A/D line
            cumulative_ad += mf_volume
            ad_line.append(cumulative_ad)
        
        return ad_line
    
    def identify_wyckoff_phases(self, vsa_data: List[Dict], ad_line: List[float]) -> Dict[str, Any]:
        """
        Identify current Wyckoff market phase.
        
        Args:
            vsa_data: Volume Spread Analysis data
            ad_line: Accumulation/Distribution line values
            
        Returns:
            Dictionary with phase analysis
        """
        if len(vsa_data) < 20 or len(ad_line) < 20:
            return {'phase': 'INSUFFICIENT_DATA', 'confidence': 0}
        
        # Analyze recent data (last 20 periods)
        recent_vsa = vsa_data[-20:]
        recent_ad = ad_line[-20:]
        
        # Volume characteristics
        high_volume_count = sum(1 for v in recent_vsa if v['relative_volume'] > 1.5)
        low_volume_count = sum(1 for v in recent_vsa if v['relative_volume'] < 0.7)
        
        # Price spread characteristics
        narrow_spreads = sum(1 for v in recent_vsa if v['spread'] < statistics.mean([x['spread'] for x in recent_vsa]) * 0.8)
        wide_spreads = sum(1 for v in recent_vsa if v['spread'] > statistics.mean([x['spread'] for x in recent_vsa]) * 1.2)
        
        # A/D line trend
        ad_trend = 1 if recent_ad[-1] > recent_ad[0] else -1
        ad_strength = abs(recent_ad[-1] - recent_ad[0]) / abs(recent_ad[0]) if recent_ad[0] != 0 else 0
        
        # Price trend
        price_trend = 1 if recent_vsa[-1]['close'] > recent_vsa[0]['close'] else -1
        
        # Effort vs Result analysis
        high_effort_low_result = sum(1 for v in recent_vsa if v['relative_volume'] > 1.3 and abs(v['close'] - v['open']) < v['spread'] * 0.3)
        low_effort_high_result = sum(1 for v in recent_vsa if v['relative_volume'] < 0.8 and abs(v['close'] - v['open']) > v['spread'] * 0.7)
        
        # Phase identification logic
        phase_scores = {
            'ACCUMULATION': 0,
            'MARKUP': 0,
            'DISTRIBUTION': 0,
            'MARKDOWN': 0
        }
        
        # Accumulation indicators
        if high_volume_count > 8 and narrow_spreads > 10:  # High volume, narrow spreads
            phase_scores['ACCUMULATION'] += 30
        if high_effort_low_result > 5:  # Absorption/support
            phase_scores['ACCUMULATION'] += 25
        if ad_trend > 0 and price_trend <= 0:  # Smart money accumulating while price weak
            phase_scores['ACCUMULATION'] += 20
        if low_volume_count > 8:  # No selling pressure
            phase_scores['ACCUMULATION'] += 15
        
        # Markup indicators  
        if price_trend > 0 and ad_trend > 0:  # Price and A/D rising together
            phase_scores['MARKUP'] += 35
        if low_effort_high_result > 5:  # Easy price movement up
            phase_scores['MARKUP'] += 25
        if wide_spreads > 8 and high_volume_count > 6:  # Wide spreads with volume
            phase_scores['MARKUP'] += 20
        
        # Distribution indicators
        if high_volume_count > 8 and wide_spreads > 8:  # High volume, wide spreads
            phase_scores['DISTRIBUTION'] += 30
        if high_effort_low_result > 6:  # Selling into strength
            phase_scores['DISTRIBUTION'] += 25
        if ad_trend < 0 and price_trend >= 0:  # Smart money distributing while price strong
            phase_scores['DISTRIBUTION'] += 25
        
        # Markdown indicators
        if price_trend < 0 and ad_trend < 0:  # Price and A/D falling together
            phase_scores['MARKDOWN'] += 35
        if low_effort_high_result > 5 and price_trend < 0:  # Easy downward movement
            phase_scores['MARKDOWN'] += 25
        if wide_spreads > 8 and low_volume_count < 5:  # Wide spreads, consistent volume
            phase_scores['MARKDOWN'] += 20
        
        # Determine primary phase
        primary_phase = max(phase_scores, key=phase_scores.get)
        confidence = min(phase_scores[primary_phase], 100)
        
        return {
            'phase': primary_phase,
            'confidence': confidence,
            'phase_scores': phase_scores,
            'volume_characteristics': {
                'high_volume_periods': high_volume_count,
                'low_volume_periods': low_volume_count,
                'avg_relative_volume': statistics.mean([v['relative_volume'] for v in recent_vsa])
            },
            'spread_characteristics': {
                'narrow_spreads': narrow_spreads,
                'wide_spreads': wide_spreads,
                'avg_spread': statistics.mean([v['spread'] for v in recent_vsa])
            },
            'ad_trend': ad_trend,
            'ad_strength': ad_strength,
            'effort_result_divergence': high_effort_low_result
        }
    
    def detect_wyckoff_events(self, vsa_data: List[Dict]) -> List[Dict]:
        """
        Detect specific Wyckoff events (Springs, Upthrusts, Tests, etc.).
        
        Args:
            vsa_data: Volume Spread Analysis data
            
        Returns:
            List of detected events
        """
        if len(vsa_data) < 10:
            return []
        
        events = []
        
        for i in range(5, len(vsa_data) - 1):
            current = vsa_data[i]
            prev_candles = vsa_data[i-5:i]
            
            # Look for support/resistance levels
            recent_lows = [c['low'] for c in prev_candles]
            recent_highs = [c['high'] for c in prev_candles]
            
            support_level = min(recent_lows)
            resistance_level = max(recent_highs)
            
            # Spring detection (false breakdown with reversal)
            if (current['low'] < support_level and 
                current['close'] > current['low'] + (current['high'] - current['low']) * 0.7 and
                current['relative_volume'] > 1.2):
                
                events.append({
                    'type': 'SPRING',
                    'timestamp': current['timestamp'],
                    'price': current['close'],
                    'confidence': min(85, int(current['relative_volume'] * 50)),
                    'description': 'False breakdown followed by recovery - potential accumulation'
                })
            
            # Upthrust detection (false breakout with failure)
            if (current['high'] > resistance_level and 
                current['close'] < current['high'] - (current['high'] - current['low']) * 0.7 and
                current['relative_volume'] > 1.2):
                
                events.append({
                    'type': 'UPTHRUST',
                    'timestamp': current['timestamp'],
                    'price': current['close'],
                    'confidence': min(85, int(current['relative_volume'] * 50)),
                    'description': 'False breakout followed by weakness - potential distribution'
                })
            
            # Test events (low volume retest of levels)
            if (abs(current['low'] - support_level) < support_level * 0.02 and
                current['relative_volume'] < 0.8):
                
                events.append({
                    'type': 'TEST_SUPPORT',
                    'timestamp': current['timestamp'],
                    'price': current['close'],
                    'confidence': 70,
                    'description': 'Low volume test of support - supply absorption'
                })
            
            if (abs(current['high'] - resistance_level) < resistance_level * 0.02 and
                current['relative_volume'] < 0.8):
                
                events.append({
                    'type': 'TEST_RESISTANCE',
                    'timestamp': current['timestamp'],
                    'price': current['close'],
                    'confidence': 70,
                    'description': 'Low volume test of resistance - demand absorption'
                })
        
        return events[-10:]  # Return last 10 events
    
    def calculate_composite_operator_activity(self, vsa_data: List[Dict], ad_line: List[float]) -> Dict[str, Any]:
        """
        Analyze composite operator (smart money) activity.
        
        Args:
            vsa_data: Volume Spread Analysis data
            ad_line: Accumulation/Distribution line
            
        Returns:
            Smart money activity analysis
        """
        if len(vsa_data) < 20:
            return {'activity': 'INSUFFICIENT_DATA', 'confidence': 0}
        
        recent_data = vsa_data[-20:]
        
        # Smart money indicators
        absorption_count = 0
        accumulation_signs = 0
        distribution_signs = 0
        
        for i, candle in enumerate(recent_data[1:], 1):
            prev_candle = recent_data[i-1]
            
            # High volume with narrow spread (absorption)
            if (candle['relative_volume'] > 1.5 and 
                candle['spread'] < statistics.mean([c['spread'] for c in recent_data]) * 0.8):
                absorption_count += 1
            
            # Accumulation signs: Strong close on high volume
            if (candle['close_position'] > 0.7 and 
                candle['relative_volume'] > 1.2):
                accumulation_signs += 1
            
            # Distribution signs: Weak close on high volume
            if (candle['close_position'] < 0.3 and 
                candle['relative_volume'] > 1.2):
                distribution_signs += 1
        
        # Determine smart money activity
        if accumulation_signs > distribution_signs + 2:
            activity = 'ACCUMULATING'
            confidence = min(90, accumulation_signs * 10 + absorption_count * 5)
        elif distribution_signs > accumulation_signs + 2:
            activity = 'DISTRIBUTING'
            confidence = min(90, distribution_signs * 10 + absorption_count * 5)
        elif absorption_count > 5:
            activity = 'ABSORBING'
            confidence = min(80, absorption_count * 10)
        else:
            activity = 'NEUTRAL'
            confidence = 40
        
        return {
            'activity': activity,
            'confidence': confidence,
            'absorption_events': absorption_count,
            'accumulation_signs': accumulation_signs,
            'distribution_signs': distribution_signs,
            'smart_money_strength': (accumulation_signs + absorption_count) - distribution_signs
        }
    
    def analyze(self, symbol: str, timeframe: str, limit: int, ohlcv_data: Optional[List] = None) -> Dict[str, Any]:
        """
        Perform comprehensive Wyckoff Method analysis.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '4h', '1d', '1h')
            limit: Number of candles to analyze
            ohlcv_data: Optional pre-fetched OHLCV data
            
        Returns:
            Dictionary containing Wyckoff analysis results
        """
        # Fetch OHLCV data if not provided
        if ohlcv_data is None:
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 50:
            return {
                'error': f"Need at least 50 candles for Wyckoff analysis. Got {len(ohlcv_data)}",
                'success': False
            }
        
        # Perform VSA calculations
        vsa_data = self.calculate_volume_spread_analysis(ohlcv_data)
        if not vsa_data:
            return {
                'error': "Unable to calculate Volume Spread Analysis",
                'success': False
            }
        
        # Calculate A/D Line
        ad_line = self.calculate_accumulation_distribution_line(ohlcv_data)
        
        # Identify current market phase
        phase_analysis = self.identify_wyckoff_phases(vsa_data, ad_line)
        
        # Detect Wyckoff events
        events = self.detect_wyckoff_events(vsa_data)
        
        # Analyze smart money activity
        smart_money = self.calculate_composite_operator_activity(vsa_data, ad_line)
        
        # Current market data
        latest_candle = ohlcv_data[-1]
        current_price = latest_candle[4]
        latest_vsa = vsa_data[-1]
        
        # Calculate key levels based on recent structure
        recent_data = vsa_data[-20:] if len(vsa_data) >= 20 else vsa_data
        support_level = min([c['low'] for c in recent_data])
        resistance_level = max([c['high'] for c in recent_data])
        
        # Determine signal based on phase and smart money activity
        signal = 'NEUTRAL'
        confidence_score = 0
        
        if phase_analysis['phase'] == 'ACCUMULATION' and smart_money['activity'] == 'ACCUMULATING':
            signal = 'BUY'
            confidence_score = min(90, (phase_analysis['confidence'] + smart_money['confidence']) // 2)
        elif phase_analysis['phase'] == 'DISTRIBUTION' and smart_money['activity'] == 'DISTRIBUTING':
            signal = 'SELL' 
            confidence_score = min(90, (phase_analysis['confidence'] + smart_money['confidence']) // 2)
        elif phase_analysis['phase'] == 'MARKUP':
            signal = 'HOLD'
            confidence_score = phase_analysis['confidence']
        elif phase_analysis['phase'] == 'MARKDOWN':
            signal = 'AVOID'
            confidence_score = phase_analysis['confidence']
        else:
            confidence_score = max(phase_analysis['confidence'], smart_money['confidence']) // 2
        
        # Risk management levels
        if signal == 'BUY':
            stop_zone = support_level * 0.97
            tp_low = resistance_level * 1.02
            tp_high = resistance_level * 1.08
        elif signal == 'SELL':
            stop_zone = resistance_level * 1.03
            tp_low = support_level * 0.98
            tp_high = support_level * 0.92
        else:
            stop_zone = support_level * 0.98
            tp_low = resistance_level * 1.01
            tp_high = resistance_level * 1.05
        
        # Calculate risk/reward ratio
        risk_distance = abs(current_price - stop_zone)
        reward_distance = abs(tp_low - current_price)
        rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
        
        # Summary
        summary = f"Wyckoff Phase: {phase_analysis['phase']} | Smart Money: {smart_money['activity']}"
        if events:
            latest_event = events[-1]
            summary += f" | Recent: {latest_event['type']}"
        
        # Get timestamp
        latest_timestamp = datetime.fromtimestamp(latest_candle[0] / 1000)
        
        return {
            'success': True,
            'analysis_time': latest_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': latest_candle[0],
            
            # Core Wyckoff indicators
            'wyckoff_phase': phase_analysis['phase'],
            'phase_confidence': phase_analysis['confidence'],
            'smart_money_activity': smart_money['activity'],
            'smart_money_confidence': smart_money['confidence'],
            'ad_line_trend': phase_analysis.get('ad_trend', 0),
            'composite_operator_strength': smart_money['smart_money_strength'],
            
            # Volume Spread Analysis
            'relative_volume': latest_vsa['relative_volume'],
            'spread_analysis': latest_vsa['spread'],
            'close_position': latest_vsa['close_position'],
            'effort_result_ratio': latest_vsa['effort_result_ratio'],
            
            # Price levels
            'current_price': round(current_price, 4),
            'support_level': round(support_level, 4),
            'resistance_level': round(resistance_level, 4),
            'stop_zone': round(stop_zone, 4),
            'tp_low': round(tp_low, 4),
            'tp_high': round(tp_high, 4),
            
            # Trading analysis
            'signal': signal,
            'summary': summary,
            'confidence_score': confidence_score,
            'rr_ratio': round(rr_ratio, 1),
            
            # Events and patterns
            'recent_events': events,
            'absorption_events': smart_money['absorption_events'],
            'spring_or_upthrust': any(e['type'] in ['SPRING', 'UPTHRUST'] for e in events[-3:]),
            
            # Phase characteristics
            'volume_characteristics': phase_analysis.get('volume_characteristics', {}),
            'spread_characteristics': phase_analysis.get('spread_characteristics', {}),
            
            # Raw data
            'raw_data': {
                'vsa_data': vsa_data,
                'ad_line': ad_line,
                'phase_scores': phase_analysis.get('phase_scores', {}),
                'ohlcv_data': ohlcv_data
            }
        }