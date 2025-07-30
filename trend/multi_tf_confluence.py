"""
Multi-Timeframe Confluence analysis for cryptocurrency trend detection.
Analyzes signals across multiple timeframes to confirm trend direction and 
filter false signals using EMA alignment, MACD histogram, and RSI divergence.
"""

import sys
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from data import get_data_collector
from trend.ema_9_21 import EMA9_21Strategy
from trend.macd import MACDStrategy
from trend.rsi_14 import RSI14Strategy
from trend.output_formatter import OutputFormatter


class MultiTimeframeConfluence:
    """Multi-timeframe confluence analyzer for enhanced signal validation."""
    
    def __init__(self):
        self.collector = get_data_collector()
        self.ema_strategy = EMA9_21Strategy()
        self.macd_strategy = MACDStrategy()
        self.rsi_strategy = RSI14Strategy()
    
    def analyze_timeframe(self, symbol: str, timeframe: str, limit: int) -> Dict:
        """
        Analyze a single timeframe for EMA, MACD, and RSI signals.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1d', '4h', '1h')
            limit: Number of candles to analyze
            
        Returns:
            Dictionary containing analysis results for the timeframe
        """
        # Fetch OHLCV data
        ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if len(ohlcv_data) < 100:
            return {
                'timeframe': timeframe,
                'error': f'Insufficient data: {len(ohlcv_data)} candles'
            }
        
        # Extract data
        timestamps = [candle[0] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        volumes = [candle[5] for candle in ohlcv_data]
        
        result = {
            'timeframe': timeframe,
            'latest_close': closes[-1],
            'timestamp': timestamps[-1]
        }
        
        try:
            # EMA Analysis
            ema9 = self.ema_strategy.calculate_ema(closes, 9)
            ema21 = self.ema_strategy.calculate_ema(closes, 21)
            
            if ema9 and ema21:
                latest_ema9 = ema9[-1]
                latest_ema21 = ema21[-1]
                
                result['ema'] = {
                    'ema9': latest_ema9,
                    'ema21': latest_ema21,
                    'bullish': latest_ema9 > latest_ema21,
                    'bearish': latest_ema9 < latest_ema21,
                    'above_both': closes[-1] > latest_ema9 and closes[-1] > latest_ema21,
                    'below_both': closes[-1] < latest_ema9 and closes[-1] < latest_ema21
                }
            
            # MACD Analysis
            macd_line, signal_line, histogram = self.macd_strategy.calculate_macd(closes)
            
            if macd_line and signal_line and histogram:
                result['macd'] = {
                    'macd': macd_line[-1],
                    'signal': signal_line[-1],
                    'histogram': histogram[-1],
                    'bullish_cross': macd_line[-1] > signal_line[-1],
                    'bearish_cross': macd_line[-1] < signal_line[-1],
                    'histogram_positive': histogram[-1] > 0,
                    'histogram_negative': histogram[-1] < 0
                }
            
            # RSI Analysis
            rsi_values = self.rsi_strategy.calculate_rsi(closes, 14)
            
            if rsi_values:
                latest_rsi = rsi_values[-1]
                result['rsi'] = {
                    'value': latest_rsi,
                    'overbought': latest_rsi > 70,
                    'oversold': latest_rsi < 30,
                    'bullish_zone': latest_rsi > 50,
                    'bearish_zone': latest_rsi < 50
                }
                
                # Proper divergence detection using swing highs/lows
                if len(rsi_values) >= 20 and len(closes) >= 20:
                    divergence = self._detect_rsi_divergence(closes, rsi_values)
                    result['rsi']['divergence'] = divergence
            
        except Exception as e:
            result['error'] = f'Analysis error: {str(e)}'
        
        return result
    
    def check_ema_alignment(self, tf_results: List[Dict]) -> Dict:
        """
        Check EMA alignment across multiple timeframes.
        
        Args:
            tf_results: List of timeframe analysis results
            
        Returns:
            Dictionary with alignment analysis
        """
        ema_alignment = {
            'all_bullish': True,
            'all_bearish': True,
            'bullish_count': 0,
            'bearish_count': 0,
            'timeframes': {}
        }
        
        for tf_result in tf_results:
            if 'ema' in tf_result:
                tf = tf_result['timeframe']
                ema_data = tf_result['ema']
                
                is_bullish = ema_data['bullish']
                is_bearish = ema_data['bearish']
                
                ema_alignment['timeframes'][tf] = {
                    'bullish': is_bullish,
                    'bearish': is_bearish,
                    'ema9': ema_data['ema9'],
                    'ema21': ema_data['ema21']
                }
                
                if is_bullish:
                    ema_alignment['bullish_count'] += 1
                else:
                    ema_alignment['all_bullish'] = False
                
                if is_bearish:
                    ema_alignment['bearish_count'] += 1
                else:
                    ema_alignment['all_bearish'] = False
        
        return ema_alignment
    
    def check_macd_confluence(self, tf_results: List[Dict]) -> Dict:
        """
        Check MACD histogram confluence across timeframes.
        
        Args:
            tf_results: List of timeframe analysis results
            
        Returns:
            Dictionary with MACD confluence analysis
        """
        macd_confluence = {
            'all_positive': True,
            'all_negative': True,
            'positive_count': 0,
            'negative_count': 0,
            'timeframes': {}
        }
        
        for tf_result in tf_results:
            if 'macd' in tf_result:
                tf = tf_result['timeframe']
                macd_data = tf_result['macd']
                
                is_positive = macd_data['histogram_positive']
                is_negative = macd_data['histogram_negative']
                
                macd_confluence['timeframes'][tf] = {
                    'histogram': macd_data['histogram'],
                    'positive': is_positive,
                    'negative': is_negative,
                    'bullish_cross': macd_data['bullish_cross']
                }
                
                if is_positive:
                    macd_confluence['positive_count'] += 1
                else:
                    macd_confluence['all_positive'] = False
                
                if is_negative:
                    macd_confluence['negative_count'] += 1
                else:
                    macd_confluence['all_negative'] = False
        
        return macd_confluence
    
    def check_rsi_divergence_match(self, tf_results: List[Dict]) -> Dict:
        """
        Check RSI divergence patterns across timeframes.
        
        Args:
            tf_results: List of timeframe analysis results
            
        Returns:
            Dictionary with RSI divergence analysis
        """
        rsi_analysis = {
            'divergence_confirmed': False,
            'bullish_divergence_count': 0,
            'bearish_divergence_count': 0,
            'timeframes': {}
        }
        
        for tf_result in tf_results:
            if 'rsi' in tf_result and 'divergence' in tf_result['rsi']:
                tf = tf_result['timeframe']
                rsi_data = tf_result['rsi']
                
                bullish_div = rsi_data['divergence']['bullish']
                bearish_div = rsi_data['divergence']['bearish']
                
                rsi_analysis['timeframes'][tf] = {
                    'value': rsi_data['value'],
                    'bullish_divergence': bullish_div,
                    'bearish_divergence': bearish_div,
                    'overbought': rsi_data['overbought'],
                    'oversold': rsi_data['oversold']
                }
                
                if bullish_div:
                    rsi_analysis['bullish_divergence_count'] += 1
                if bearish_div:
                    rsi_analysis['bearish_divergence_count'] += 1
        
        # Confirm divergence if detected on higher timeframe
        rsi_analysis['divergence_confirmed'] = (
            rsi_analysis['bullish_divergence_count'] > 0 or 
            rsi_analysis['bearish_divergence_count'] > 0
        )
        
        return rsi_analysis
    
    def _detect_rsi_divergence(self, closes: List[float], rsi_values: List[float]) -> Dict:
        """
        Detect RSI divergence using proper swing high/low identification.
        
        Args:
            closes: Recent closing prices (last 20)
            rsi_values: Recent RSI values (last 20)
            
        Returns:
            Dictionary with divergence analysis
        """
        if len(closes) < 20 or len(rsi_values) < 20:
            return {'bearish': False, 'bullish': False}
        
        # Use last 20 periods for divergence analysis
        price_data = closes[-20:]
        rsi_data = rsi_values[-20:]
        
        # Find swing highs and lows with minimum 3-period separation
        price_highs = self._find_swing_highs(price_data, min_periods=3)
        price_lows = self._find_swing_lows(price_data, min_periods=3)
        rsi_highs = self._find_swing_highs(rsi_data, min_periods=3)
        rsi_lows = self._find_swing_lows(rsi_data, min_periods=3)
        
        bearish_divergence = False
        bullish_divergence = False
        
        # Bearish divergence: Price makes higher high, RSI makes lower high
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            latest_price_high = price_highs[-1]
            prev_price_high = price_highs[-2]
            latest_rsi_high = rsi_highs[-1]
            prev_rsi_high = rsi_highs[-2]
            
            if (price_data[latest_price_high] > price_data[prev_price_high] and 
                rsi_data[latest_rsi_high] < rsi_data[prev_rsi_high]):
                bearish_divergence = True
        
        # Bullish divergence: Price makes lower low, RSI makes higher low
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            latest_price_low = price_lows[-1]
            prev_price_low = price_lows[-2]
            latest_rsi_low = rsi_lows[-1]
            prev_rsi_low = rsi_lows[-2]
            
            if (price_data[latest_price_low] < price_data[prev_price_low] and 
                rsi_data[latest_rsi_low] > rsi_data[prev_rsi_low]):
                bullish_divergence = True
        
        return {
            'bearish': bearish_divergence,
            'bullish': bullish_divergence
        }
    
    def _find_swing_highs(self, data: List[float], min_periods: int = 3) -> List[int]:
        """
        Find swing highs in price/indicator data.
        
        Args:
            data: Price or indicator values
            min_periods: Minimum periods between swings
            
        Returns:
            List of indices where swing highs occur
        """
        swing_highs = []
        
        for i in range(min_periods, len(data) - min_periods):
            is_high = True
            
            # Check if current point is higher than surrounding points
            for j in range(i - min_periods, i + min_periods + 1):
                if j != i and data[j] >= data[i]:
                    is_high = False
                    break
            
            if is_high:
                swing_highs.append(i)
        
        return swing_highs
    
    def _find_swing_lows(self, data: List[float], min_periods: int = 3) -> List[int]:
        """
        Find swing lows in price/indicator data.
        
        Args:
            data: Price or indicator values
            min_periods: Minimum periods between swings
            
        Returns:
            List of indices where swing lows occur
        """
        swing_lows = []
        
        for i in range(min_periods, len(data) - min_periods):
            is_low = True
            
            # Check if current point is lower than surrounding points
            for j in range(i - min_periods, i + min_periods + 1):
                if j != i and data[j] <= data[i]:
                    is_low = False
                    break
            
            if is_low:
                swing_lows.append(i)
        
        return swing_lows
    
    def _get_timeframe_weight(self, timeframe: str) -> float:
        """
        Get weight for timeframe based on standard hierarchy.
        Higher timeframes get more weight.
        
        Args:
            timeframe: Timeframe string (e.g., '1d', '4h', '1h')
            
        Returns:
            Weight multiplier for the timeframe
        """
        timeframe_weights = {
            '1M': 3.0,   # Monthly - highest weight
            '1w': 2.5,   # Weekly
            '3d': 2.2,   # 3-day
            '1d': 2.0,   # Daily
            '12h': 1.8,  # 12-hour
            '8h': 1.6,   # 8-hour
            '6h': 1.5,   # 6-hour
            '4h': 1.3,   # 4-hour
            '2h': 1.1,   # 2-hour
            '1h': 1.0,   # Hourly - base weight
            '30m': 0.8,  # 30-minute
            '15m': 0.6,  # 15-minute
            '5m': 0.4,   # 5-minute
            '3m': 0.3,   # 3-minute
            '1m': 0.2    # 1-minute - lowest weight
        }
        
        return timeframe_weights.get(timeframe, 1.0)
    
    def generate_confluence_signal(self, ema_alignment: Dict, macd_confluence: Dict, 
                                 rsi_analysis: Dict) -> Dict:
        """
        Generate final confluence signal based on multi-timeframe analysis.
        
        Args:
            ema_alignment: EMA alignment analysis
            macd_confluence: MACD confluence analysis  
            rsi_analysis: RSI divergence analysis
            
        Returns:
            Dictionary with final signal and confidence
        """
        signal = {
            'action': 'NONE',
            'trend': 'NEUTRAL',
            'confidence': 0,
            'strength_score': 0,
            'reasons': []
        }
        
        # Calculate weighted strength score using timeframe hierarchy
        timeframes = list(ema_alignment['timeframes'].keys())
        if not timeframes:
            return signal
        
        # Calculate weighted strength score using timeframe hierarchy
        
        # EMA alignment scoring with timeframe weighting (50% of total)
        ema_score = 0
        ema_weight_sum = 0
        for tf in timeframes:
            tf_weight = self._get_timeframe_weight(tf)
            ema_weight_sum += tf_weight
            
            if tf in ema_alignment['timeframes']:
                tf_data = ema_alignment['timeframes'][tf]
                if tf_data['bullish'] or tf_data['bearish']:
                    ema_score += tf_weight
        
        if ema_weight_sum > 0:
            ema_normalized = (ema_score / ema_weight_sum) * 50
            signal['strength_score'] += int(ema_normalized)
        
        if ema_alignment['all_bullish']:
            signal['reasons'].append('All timeframes EMA bullish alignment')
        elif ema_alignment['all_bearish']:
            signal['reasons'].append('All timeframes EMA bearish alignment')
        elif ema_alignment['bullish_count'] > ema_alignment['bearish_count']:
            signal['reasons'].append(f'EMA bullish bias ({ema_alignment["bullish_count"]}/{len(timeframes)} TFs)')
        elif ema_alignment['bearish_count'] > ema_alignment['bullish_count']:
            signal['reasons'].append(f'EMA bearish bias ({ema_alignment["bearish_count"]}/{len(timeframes)} TFs)')
        
        # MACD confluence scoring with timeframe weighting (30% of total)
        macd_score = 0
        macd_weight_sum = 0
        for tf in timeframes:
            tf_weight = self._get_timeframe_weight(tf)
            macd_weight_sum += tf_weight
            
            if tf in macd_confluence['timeframes']:
                tf_data = macd_confluence['timeframes'][tf]
                if tf_data['positive'] or tf_data['negative']:
                    macd_score += tf_weight
        
        if macd_weight_sum > 0:
            macd_normalized = (macd_score / macd_weight_sum) * 30
            signal['strength_score'] += int(macd_normalized)
        
        if macd_confluence['all_positive']:
            signal['reasons'].append('All timeframes MACD positive histogram')
        elif macd_confluence['all_negative']:
            signal['reasons'].append('All timeframes MACD negative histogram')
        
        # RSI divergence scoring with higher timeframe emphasis (20% of total)
        if rsi_analysis['divergence_confirmed']:
            # Higher weight for divergence on higher timeframes
            divergence_score = 0
            divergence_weight_sum = 0
            
            for tf in timeframes:
                if tf in rsi_analysis['timeframes']:
                    tf_data = rsi_analysis['timeframes'][tf]
                    tf_weight = self._get_timeframe_weight(tf)
                    divergence_weight_sum += tf_weight
                    
                    if tf_data.get('bullish_divergence') or tf_data.get('bearish_divergence'):
                        divergence_score += tf_weight * 2  # Double weight for divergence
            
            if divergence_weight_sum > 0:
                divergence_normalized = min(20, (divergence_score / divergence_weight_sum) * 20)
                signal['strength_score'] += int(divergence_normalized)
            
            if rsi_analysis['bullish_divergence_count'] > 0:
                signal['reasons'].append('Bullish RSI divergence detected')
            if rsi_analysis['bearish_divergence_count'] > 0:
                signal['reasons'].append('Bearish RSI divergence detected')
        
        # Determine final signal
        if (ema_alignment['bullish_count'] > ema_alignment['bearish_count'] and 
            macd_confluence['positive_count'] > macd_confluence['negative_count']):
            signal['action'] = 'BUY'
            signal['trend'] = 'BULLISH'
        elif (ema_alignment['bearish_count'] > ema_alignment['bullish_count'] and 
              macd_confluence['negative_count'] > macd_confluence['positive_count']):
            signal['action'] = 'SELL'
            signal['trend'] = 'BEARISH'
        
        # Confidence based on strength score
        if signal['strength_score'] >= 80:
            signal['confidence'] = 'HIGH'
        elif signal['strength_score'] >= 60:
            signal['confidence'] = 'MEDIUM'
        elif signal['strength_score'] >= 40:
            signal['confidence'] = 'LOW'
        else:
            signal['confidence'] = 'VERY_LOW'
        
        return signal
    
    def analyze(self, symbol: str, timeframes: List[str], indicators: List[str], limit: int) -> None:
        """
        Perform multi-timeframe confluence analysis and display results.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframes: List of timeframes to analyze (e.g., ['1d', '4h', '1h'])
            indicators: List of indicators to use (e.g., ['ema9/21', 'macd', 'rsi14'])
            limit: Number of candles to analyze
        """
        print(f"Symbol: {symbol}")
        print(f"Timeframes: {', '.join(timeframes)}")
        print(f"Indicators: {', '.join(indicators)}")
        print(f"Limit: {limit} candles\n")
        
        # Analyze each timeframe
        tf_results = []
        for tf in timeframes:
            try:
                result = self.analyze_timeframe(symbol, tf, limit)
                tf_results.append(result)
                
                if 'error' in result:
                    print(f"⚠️  {tf}: {result['error']}")
                else:
                    print(f"✅ {tf}: Analysis completed")
                    
            except Exception as e:
                print(f"❌ {tf}: Error - {str(e)}")
        
        if not tf_results or all('error' in result for result in tf_results):
            print("❌ Unable to analyze any timeframes")
            return
        
        # Filter successful results
        valid_results = [r for r in tf_results if 'error' not in r]
        
        if not valid_results:
            print("❌ No valid timeframe results")
            return
        
        # Check alignments
        ema_alignment = self.check_ema_alignment(valid_results)
        macd_confluence = self.check_macd_confluence(valid_results)
        rsi_analysis = self.check_rsi_divergence_match(valid_results)
        
        # Generate final signal
        final_signal = self.generate_confluence_signal(ema_alignment, macd_confluence, rsi_analysis)
        
        # Display results
        current_timestamp = int(datetime.now().timestamp() * 1000)
        
        # EMA Alignment
        ema_status = "✅ All Bullish" if ema_alignment['all_bullish'] else "❌ All Bearish" if ema_alignment['all_bearish'] else f"⚠️  Mixed ({ema_alignment['bullish_count']}/{len(valid_results)} Bullish)"
        
        # MACD Confluence  
        macd_status = "✅ All Positive" if macd_confluence['all_positive'] else "❌ All Negative" if macd_confluence['all_negative'] else f"⚠️  Mixed ({macd_confluence['positive_count']}/{len(valid_results)} Positive)"
        
        # RSI Divergence
        rsi_status = "⚠️  Divergence Detected" if rsi_analysis['divergence_confirmed'] else "➖ No Divergence"
        
        # Format metrics for output
        metrics = {
            "EMA": ema_status,
            "MACD": macd_status,
            "RSI_DIV": rsi_status
        }
        
        print(OutputFormatter.format_analysis_output(
            timestamp=current_timestamp,
            metrics=metrics,
            signal=final_signal['action'],
            trend=final_signal['trend']
        ))
        
        # Additional confidence info
        confidence_emoji = "🔥" if final_signal['confidence'] == 'HIGH' else "⚡" if final_signal['confidence'] == 'MEDIUM' else "⚠️" if final_signal['confidence'] == 'LOW' else "❓"
        print(f"Confidence: {confidence_emoji} {final_signal['confidence']} ({final_signal['strength_score']}/100)")
        
        # Detailed breakdown per timeframe
        for result in valid_results:
            tf = result['timeframe']
            dt = datetime.fromtimestamp(result['timestamp'] / 1000)
            
            # EMA status
            if 'ema' in result:
                ema_data = result['ema']
                ema_trend = "📈 Bull" if ema_data['bullish'] else "📉 Bear" if ema_data['bearish'] else "➖ Flat"
                ema_pos = "Above" if ema_data['above_both'] else "Below" if ema_data['below_both'] else "Between"
            else:
                ema_trend = "❌ Error"
                ema_pos = "N/A"
            
            # MACD status
            if 'macd' in result:
                macd_data = result['macd']
                macd_hist = "+" if macd_data['histogram_positive'] else "-" if macd_data['histogram_negative'] else "0"
                macd_cross = "Bull" if macd_data['bullish_cross'] else "Bear"
            else:
                macd_hist = "❌"
                macd_cross = "Error"
            
            # RSI status
            if 'rsi' in result:
                rsi_data = result['rsi']
                rsi_level = f"{rsi_data['value']:.1f}"
                rsi_zone = "OB" if rsi_data['overbought'] else "OS" if rsi_data['oversold'] else "OK"
            else:
                rsi_level = "❌"
                rsi_zone = "Error"
            
            print(f"{tf:>3} | EMA: {ema_trend} ({ema_pos}) | MACD: {macd_cross} ({macd_hist}) | RSI: {rsi_level} ({rsi_zone})")
        
        # Reasons
        if final_signal['reasons']:
            for reason in final_signal['reasons']:
                print(f"• {reason}")


def parse_command(command: str) -> Tuple[str, List[str], List[str], int]:
    """
    Parse terminal command: multi_tf s=BTC/USDT t1=1d t2=4h t3=1h indicators=ema9/21,macd,rsi14 l=200
    
    Args:
        command: Command string
        
    Returns:
        Tuple of (symbol, timeframes, indicators, limit)
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'multi_tf':
        raise ValueError("Invalid command format. Use: multi_tf s=SYMBOL t1=TF1 t2=TF2 t3=TF3 indicators=LIST l=LIMIT")
    
    symbol = None
    timeframes = []
    indicators = ['ema9/21', 'macd', 'rsi14']  # default
    limit = 200  # default
    
    for part in parts[1:]:
        if part.startswith('s='):
            symbol = part[2:]
        elif part.startswith('t1='):
            timeframes.append(part[3:])
        elif part.startswith('t2='):
            timeframes.append(part[3:])
        elif part.startswith('t3='):
            timeframes.append(part[3:])
        elif part.startswith('indicators='):
            indicators = part[11:].split(',')
        elif part.startswith('l='):
            try:
                limit = int(part[2:])
            except ValueError:
                raise ValueError(f"Invalid limit value: {part[2:]}")
    
    if symbol is None:
        raise ValueError("Symbol (s=) is required")
    
    if not timeframes:
        timeframes = ['1d', '4h', '1h']  # default
    
    return symbol, timeframes, indicators, limit


def main():
    """Main entry point for multi-timeframe confluence analysis."""
    if len(sys.argv) < 2:
        print("Usage: python multi_tf_confluence.py s=SYMBOL t1=TF1 t2=TF2 t3=TF3 indicators=LIST l=LIMIT")
        print("Example: python multi_tf_confluence.py s=BTC/USDT t1=1d t2=4h t3=1h indicators=ema9/21,macd,rsi14 l=200")
        return
    
    try:
        # Parse command line arguments
        command = ' '.join(['multi_tf'] + sys.argv[1:])
        symbol, timeframes, indicators, limit = parse_command(command)
        
        # Run analysis
        analyzer = MultiTimeframeConfluence()
        analyzer.analyze(symbol, timeframes, indicators, limit)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()