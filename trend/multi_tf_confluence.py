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
                
                # Check for divergence (simplified - last 5 periods)
                if len(rsi_values) >= 10 and len(closes) >= 10:
                    recent_rsi = rsi_values[-5:]
                    recent_closes = closes[-5:]
                    
                    # Check if price made higher high but RSI made lower high (bearish divergence)
                    price_high_idx = recent_closes.index(max(recent_closes))
                    rsi_high_idx = recent_rsi.index(max(recent_rsi))
                    
                    # Check if price made lower low but RSI made higher low (bullish divergence)
                    price_low_idx = recent_closes.index(min(recent_closes))
                    rsi_low_idx = recent_rsi.index(min(recent_rsi))
                    
                    result['rsi']['divergence'] = {
                        'bearish': price_high_idx != rsi_high_idx and max(recent_closes) > recent_closes[0],
                        'bullish': price_low_idx != rsi_low_idx and min(recent_closes) < recent_closes[0]
                    }
            
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
        
        # Calculate strength score (0-100)
        total_timeframes = len(ema_alignment['timeframes'])
        if total_timeframes == 0:
            return signal
        
        # EMA alignment scoring (40% weight)
        if ema_alignment['all_bullish']:
            signal['strength_score'] += 40
            signal['reasons'].append('All timeframes EMA bullish')
        elif ema_alignment['all_bearish']:
            signal['strength_score'] += 40
            signal['reasons'].append('All timeframes EMA bearish')
        else:
            # Partial alignment
            bullish_ratio = ema_alignment['bullish_count'] / total_timeframes
            bearish_ratio = ema_alignment['bearish_count'] / total_timeframes
            signal['strength_score'] += int(max(bullish_ratio, bearish_ratio) * 40)
        
        # MACD confluence scoring (30% weight)
        if macd_confluence['all_positive']:
            signal['strength_score'] += 30
            signal['reasons'].append('All timeframes MACD positive')
        elif macd_confluence['all_negative']:
            signal['strength_score'] += 30
            signal['reasons'].append('All timeframes MACD negative')
        else:
            # Partial confluence
            pos_ratio = macd_confluence['positive_count'] / total_timeframes
            neg_ratio = macd_confluence['negative_count'] / total_timeframes
            signal['strength_score'] += int(max(pos_ratio, neg_ratio) * 30)
        
        # RSI divergence scoring (30% weight)
        if rsi_analysis['divergence_confirmed']:
            signal['strength_score'] += 30
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
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # EMA Alignment
        ema_status = "✅ All Bullish" if ema_alignment['all_bullish'] else "❌ All Bearish" if ema_alignment['all_bearish'] else f"⚠️  Mixed ({ema_alignment['bullish_count']}/{len(valid_results)} Bullish)"
        
        # MACD Confluence  
        macd_status = "✅ All Positive" if macd_confluence['all_positive'] else "❌ All Negative" if macd_confluence['all_negative'] else f"⚠️  Mixed ({macd_confluence['positive_count']}/{len(valid_results)} Positive)"
        
        # RSI Divergence
        rsi_status = "⚠️  Divergence Detected" if rsi_analysis['divergence_confirmed'] else "➖ No Divergence"
        
        print(f"[{current_time}] EMA: {ema_status} | MACD: {macd_status} | RSI Div: {rsi_status}")
        
        # Final Signal
        signal_emoji = "📈" if final_signal['trend'] == 'BULLISH' else "📉" if final_signal['trend'] == 'BEARISH' else "➖"
        confidence_emoji = "🔥" if final_signal['confidence'] == 'HIGH' else "⚡" if final_signal['confidence'] == 'MEDIUM' else "⚠️" if final_signal['confidence'] == 'LOW' else "❓"
        
        print(f"Signal: {final_signal['action']} | {signal_emoji} {final_signal['trend']} | {confidence_emoji} {final_signal['confidence']} ({final_signal['strength_score']}/100)")
        
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