#!/usr/bin/env python3
"""
CLI Output Formatter for CryptoPat Analysis
Formats analysis results from all modules into a beautiful CLI display
"""

import datetime
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class AnalysisResults:
    """Container for all analysis results"""
    techin_results: Dict[str, Any]
    pattern_results: Dict[str, Any] 
    orderflow_results: Dict[str, Any]
    market_data: Dict[str, Any]


class CLIFormatter:
    """Beautiful CLI output formatter for CryptoPat analysis results"""
    
    def __init__(self):
        self.width = 79
        self.separator = "=" * self.width
        
    def format_price(self, price: float) -> str:
        """Format price with appropriate decimal places and commas"""
        if price >= 1000:
            return f"${price:,.2f}"
        elif price >= 1:
            return f"${price:.4f}"
        else:
            return f"${price:.6f}"
    
    def format_percentage(self, value: float, decimals: int = 1) -> str:
        """Format percentage with + or - sign"""
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.{decimals}f}%"
    
    def format_confidence_bar(self, confidence: int) -> str:
        """Create a visual confidence bar"""
        filled = int(confidence / 10)
        empty = 10 - filled
        bar = "█" * filled + "░" * empty
        return f"{bar} {confidence}%"
    
    def get_signal_emoji(self, signal: str) -> str:
        """Get emoji for signal type"""
        signal_map = {
            'BUY': '🔴',
            'SELL': '🟢', 
            'HOLD': '🟡',
            'NEUTRAL': '⚪',
            'PENDING': '🔵'
        }
        return signal_map.get(signal.upper(), '⚪')
    
    def get_bias_color(self, bias: str) -> str:
        """Get color coding for bias"""
        if bias.upper() in ['BULLISH', 'BUY']:
            return '🟢'
        elif bias.upper() in ['BEARISH', 'SELL']:
            return '🔴'
        else:
            return '🟡'
    
    def calculate_overall_bias(self, results: AnalysisResults) -> tuple:
        """Calculate overall market bias and confidence"""
        signals = []
        confidences = []
        
        # Collect all signals and confidences
        for category in [results.techin_results, results.pattern_results, results.orderflow_results]:
            for name, data in category.items():
                if isinstance(data, dict) and data.get('success', False):
                    signal = data.get('signal', 'NEUTRAL')
                    confidence = data.get('confidence_score', 0)
                    
                    if signal in ['BUY', 'BULLISH']:
                        signals.append(1)
                        confidences.append(confidence)
                    elif signal in ['SELL', 'BEARISH']:
                        signals.append(-1)
                        confidences.append(confidence)
                    else:
                        signals.append(0)
                        confidences.append(confidence)
        
        if not signals:
            return 'NEUTRAL', 0
        
        # Weighted average based on confidence
        weighted_sum = sum(s * c for s, c in zip(signals, confidences))
        total_confidence = sum(confidences)
        
        if total_confidence == 0:
            return 'NEUTRAL', 0
        
        bias_score = weighted_sum / total_confidence
        overall_confidence = total_confidence / len(signals)
        
        if bias_score > 0.3:
            return 'BULLISH', int(overall_confidence)
        elif bias_score < -0.3:
            return 'BEARISH', int(overall_confidence)
        else:
            return 'NEUTRAL', int(overall_confidence)
    
    def format_header(self, symbol: str, timeframe: str, candles: int, current_price: float, price_change_pct: float = 0.0) -> str:
        """Format the header section"""
        analysis_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        price_change_value = current_price * (price_change_pct / 100)
        
        header = f"""
{self.separator}
                        CRYPTOPAT ANALYSIS REPORT                            
{self.separator}
Symbol: {symbol} | Timeframe: {timeframe} | Candles: {candles} | Analysis Time: {analysis_time}
Current Price: {self.format_price(current_price)} | 24h Change: {self.format_percentage(price_change_pct)} ({self.format_price(abs(price_change_value)) if price_change_pct >= 0 else f'-{self.format_price(abs(price_change_value))}'})
{self.separator}
"""
        return header
    
    def format_market_overview(self, results: AnalysisResults) -> str:
        """Format market overview section"""
        overall_bias, confidence = self.calculate_overall_bias(results)
        bias_emoji = self.get_bias_color(overall_bias)
        
        # Get primary signal from highest confidence indicator
        primary_signal = 'NEUTRAL'
        highest_confidence = 0
        
        for category in [results.techin_results, results.pattern_results, results.orderflow_results]:
            for name, data in category.items():
                if isinstance(data, dict) and data.get('success', False):
                    conf = data.get('confidence_score', 0)
                    if conf > highest_confidence:
                        highest_confidence = conf
                        primary_signal = data.get('signal', 'NEUTRAL')
        
        # Calculate average R/R ratio
        rr_ratios = []
        for category in [results.techin_results, results.pattern_results, results.orderflow_results]:
            for name, data in category.items():
                if isinstance(data, dict) and data.get('success', False):
                    rr = data.get('rr_ratio', 0)
                    if rr > 0:
                        rr_ratios.append(rr)
        
        avg_rr = sum(rr_ratios) / len(rr_ratios) if rr_ratios else 0
        
        overview = f"""
📊 MARKET OVERVIEW
Overall Bias: {bias_emoji} {overall_bias} (Confidence: {confidence}%)
Primary Signal: {self.get_signal_emoji(primary_signal)} {primary_signal}
Risk/Reward: 1:{avg_rr:.1f}
Market Regime: {'BULL_MARKET' if overall_bias == 'BULLISH' else 'BEAR_MARKET' if overall_bias == 'BEARISH' else 'SIDEWAYS'} (Moderate Volatility)
"""
        return overview
    
    def format_key_levels(self, results: AnalysisResults) -> str:
        """Format key price levels section"""
        # Collect all support/resistance levels
        supports = []
        resistances = []
        stop_losses = []
        targets = []
        
        for category in [results.techin_results, results.pattern_results, results.orderflow_results]:
            for name, data in category.items():
                if isinstance(data, dict) and data.get('success', False):
                    if 'support_level' in data and data['support_level'] > 0:
                        supports.append(data['support_level'])
                    if 'resistance_level' in data and data['resistance_level'] > 0:
                        resistances.append(data['resistance_level'])
                    if 'stop_zone' in data and data['stop_zone'] > 0:
                        stop_losses.append(data['stop_zone'])
                    if 'tp_low' in data and data['tp_low'] > 0:
                        targets.append(data['tp_low'])
                    if 'tp_high' in data and data['tp_high'] > 0:
                        targets.append(data['tp_high'])
        
        # Calculate average levels
        avg_support = sum(supports) / len(supports) if supports else 0
        avg_resistance = sum(resistances) / len(resistances) if resistances else 0
        avg_stop = sum(stop_losses) / len(stop_losses) if stop_losses else 0
        
        # Determine current price for proper target ordering
        current_price = 0
        for category in [results.techin_results, results.pattern_results, results.orderflow_results]:
            for name, data in category.items():
                if isinstance(data, dict) and data.get('success', False):
                    current_price = data.get('current_price', current_price)
                    break
            if current_price > 0:
                break
        
        # Get overall market bias for intelligent target selection
        overall_bias, confidence = self.calculate_overall_bias(results)
        
        # Filter targets based on market bias and current price
        if targets and current_price > 0:
            if overall_bias == 'BULLISH':
                # For bullish bias, use targets above current price
                valid_targets = [t for t in targets if t > current_price * 1.01]  # At least 1% above
            elif overall_bias == 'BEARISH':
                # For bearish bias, use targets below current price  
                valid_targets = [t for t in targets if t < current_price * 0.99]  # At least 1% below
            else:
                # For neutral bias, prefer targets above current price but allow both
                above_targets = [t for t in targets if t > current_price * 1.01]
                below_targets = [t for t in targets if t < current_price * 0.99]
                # Prefer upside targets for neutral bias (common trading practice)
                valid_targets = above_targets if above_targets else below_targets
            
            if valid_targets:
                valid_targets.sort(key=lambda x: abs(x - current_price))  # Sort by distance from current price
                tp1 = valid_targets[0] if valid_targets else 0
                tp2 = valid_targets[1] if len(valid_targets) > 1 else valid_targets[-1] if valid_targets else 0
            else:
                # Fallback: if no valid targets, use resistance levels
                tp1 = avg_resistance * 1.02 if avg_resistance > current_price else 0
                tp2 = avg_resistance * 1.05 if avg_resistance > current_price else 0
        else:
            tp1 = tp2 = 0
        
        levels = f"""
🎯 KEY LEVELS
Support: {self.format_price(avg_support) if avg_support > 0 else 'N/A'}
Resistance: {self.format_price(avg_resistance) if avg_resistance > 0 else 'N/A'}
Stop Loss: {self.format_price(avg_stop) if avg_stop > 0 else 'N/A'}
Targets: {self.format_price(tp1) if tp1 > 0 else 'N/A'} (TP1){f' | {self.format_price(tp2)} (TP2)' if tp2 > 0 else ''}
"""
        return levels
    
    def format_technical_indicators(self, techin_results: Dict[str, Any]) -> str:
        """Format technical indicators section"""
        section = "\n📈 TECHNICAL INDICATORS\n"
        
        # Main trend indicators
        if 'ema_9_21' in techin_results:
            ema = techin_results['ema_9_21']
            if ema.get('success', False):
                signal = ema.get('signal', 'NEUTRAL')
                confidence = ema.get('confidence_score', 0)
                # Calculate distance from current price to EMA21
                current_price = ema.get('current_price', 0)
                ema21_val = ema.get('ema21_value', 0)
                distance = abs(current_price - ema21_val) if current_price > 0 and ema21_val > 0 else 0
                section += f"EMA 9/21:     {self.get_signal_emoji(signal)} {signal} ({confidence}%)    │ Distance: {self.format_price(distance)}\n"
        
        if 'macd' in techin_results:
            macd = techin_results['macd']
            if macd.get('success', False):
                signal = macd.get('signal', 'NEUTRAL')
                confidence = macd.get('confidence_score', 0)
                histogram = macd.get('histogram', 0)  # Fixed: use 'histogram' not 'histogram_value'
                section += f"MACD:         {self.get_signal_emoji(signal)} {signal} ({confidence}%)    │ Histogram: {histogram:+.1f}\n"
        
        if 'supertrend' in techin_results:
            st = techin_results['supertrend']
            if st.get('success', False):
                signal = st.get('signal', 'NEUTRAL')
                confidence = st.get('confidence_score', 0)
                st_value = st.get('supertrend_value', 0)
                section += f"SuperTrend:   {self.get_signal_emoji(signal)} {signal} ({confidence}%)    │ Trend: {self.format_price(st_value)}\n"
        
        if 'bollinger_bands' in techin_results:
            bb = techin_results['bollinger_bands']
            if bb.get('success', False):
                signal = bb.get('signal', 'NEUTRAL')
                confidence = bb.get('confidence_score', 0)
                position = bb.get('band_position', 0)  # Fixed: use 'band_position' not 'bb_position'
                section += f"Bollinger:    {self.get_signal_emoji(signal)} {signal} ({confidence}%)    │ Position: {position:.2f}\n"
        
        # Momentum indicators
        section += "\nMOMENTUM INDICATORS\n"
        
        if 'rsi_14' in techin_results:
            rsi = techin_results['rsi_14']
            if isinstance(rsi, dict) and rsi.get('rsi_value') is not None:
                rsi_value = rsi.get('rsi_value', 0)
                condition = rsi.get('rsi_condition', 'NORMAL')
                section += f"RSI-14:       {rsi_value:.1f} ({condition})\n"
        
        if 'atr_adx' in techin_results:
            atr_adx = techin_results['atr_adx']
            if isinstance(atr_adx, dict) and atr_adx.get('atr_value') is not None:
                atr = atr_adx.get('atr_value', 0)
                adx = atr_adx.get('adx_strength', 0)  # Fixed: use 'adx_strength' not 'adx_value'
                trend_strength = atr_adx.get('trend_strength', 'WEAK')
                section += f"ATR/ADX:      ATR: {atr:.0f} | ADX: {adx:.1f} ({trend_strength} Trend)\n"
        
        # Volume indicators
        section += "\nVOLUME INDICATORS\n"
        
        if 'obv' in techin_results:
            obv = techin_results['obv']
            if isinstance(obv, dict) and obv.get('obv_value') is not None:
                signal = obv.get('signal', 'NEUTRAL')
                obv_value = obv.get('obv_value', 0)
                # Convert to millions and format
                obv_millions = obv_value / 1000000 if abs(obv_value) >= 1000000 else obv_value / 1000
                unit = 'M' if abs(obv_value) >= 1000000 else 'K'
                trend_conf = obv.get('trend_confirmation', 'NEUTRAL')
                section += f"OBV:          {trend_conf} ({obv_millions:+.1f}{unit})\n"
        
        if 'vwap' in techin_results:
            vwap = techin_results['vwap']
            if isinstance(vwap, dict) and vwap.get('signal') is not None:
                signal = vwap.get('signal', 'NEUTRAL')
                position = vwap.get('price_position', 'NEUTRAL')
                section += f"VWAP:         {self.get_signal_emoji(signal)} {signal} (Price {position})\n"
        
        # Wyckoff Method analysis
        if 'wyckoff' in techin_results:
            wyckoff = techin_results['wyckoff']
            if isinstance(wyckoff, dict) and wyckoff.get('success', False):
                signal = wyckoff.get('signal', 'NEUTRAL')
                phase = wyckoff.get('wyckoff_phase', 'UNKNOWN')
                smart_money = wyckoff.get('smart_money_activity', 'NEUTRAL')
                confidence = wyckoff.get('confidence_score', 0)
                section += f"\nWYCKOFF METHOD\n"
                section += f"Phase:        {phase} ({confidence}% confidence)\n"
                section += f"Smart Money:  {smart_money}\n"
                section += f"Signal:       {self.get_signal_emoji(signal)} {signal}\n"
                
                # Recent events
                events = wyckoff.get('recent_events', [])
                if events:
                    latest_event = events[-1]
                    section += f"Recent Event: {latest_event.get('type', 'NONE')} (Conf: {latest_event.get('confidence', 0)}%)\n"
                
                # Volume characteristics
                vol_chars = wyckoff.get('volume_characteristics', {})
                relative_vol = wyckoff.get('relative_volume', 1.0)
                section += f"Volume:       {relative_vol:.1f}x avg (High: {vol_chars.get('high_volume_periods', 0)}, Low: {vol_chars.get('low_volume_periods', 0)})\n"
        
        # Pivot Points analysis
        if 'pivotpoint' in techin_results:
            pivot = techin_results['pivotpoint']
            if isinstance(pivot, dict) and pivot.get('success', False):
                signal = pivot.get('signal', 'NEUTRAL')
                confidence = pivot.get('pivot_strength', 0)
                bias = pivot.get('analysis', {}).get('bias', 'neutral')
                
                section += f"\nPIVOT POINTS\n"
                section += f"Signal:       {self.get_signal_emoji(signal)} {signal} (Strength: {confidence:.1f}%)\n"
                section += f"Bias:         {self.get_bias_color(bias)} {bias.upper()}\n"
                
                # Display key pivot levels
                standard_pivots = pivot.get('standard_pivots', {})
                current_price = pivot.get('current_price', 0)
                
                if standard_pivots and current_price > 0:
                    pp = standard_pivots.get('pivot', 0)
                    r1 = standard_pivots.get('r1', 0)
                    s1 = standard_pivots.get('s1', 0)
                    
                    section += f"Pivot Point:  {self.format_price(pp)}\n"
                    section += f"Resistance:   {self.format_price(r1)} (R1)\n"
                    section += f"Support:      {self.format_price(s1)} (S1)\n"
                    
                    # Show nearest level info
                    analysis = pivot.get('analysis', {})
                    nearest_support = analysis.get('nearest_support')
                    nearest_resistance = analysis.get('nearest_resistance')
                    support_dist_pct = analysis.get('support_distance_pct')
                    resistance_dist_pct = analysis.get('resistance_distance_pct')
                    
                    if nearest_support and support_dist_pct is not None:
                        section += f"Nearest Support: {self.format_price(nearest_support)} ({support_dist_pct:.2f}% away)\n"
                    if nearest_resistance and resistance_dist_pct is not None:
                        section += f"Nearest Resistance: {self.format_price(nearest_resistance)} ({resistance_dist_pct:.2f}% away)\n"
        
        return section
    
    def format_chart_patterns(self, pattern_results: Dict[str, Any]) -> str:
        """Format chart patterns section"""
        section = "\nCHART PATTERNS\n"
        
        pattern_names = {
            'flag': 'Bull Flag',
            'triangle': 'Triangle',
            'head_and_shoulders': 'Head & Shoulders',
            'inverse_head_and_shoulders': 'Inverse H&S',
            'double_top': 'Double Top',
            'double_bottom': 'Double Bottom',
            'wedge': 'Wedge',
            'shark_pattern': 'Shark Pattern',
            'butterfly_pattern': 'Butterfly',
            'elliott_wave': 'Elliott Wave'
        }
        
        for pattern_key, display_name in pattern_names.items():
            if pattern_key in pattern_results:
                result = pattern_results[pattern_key]
                if result.get('success', False):
                    detected = result.get('pattern_detected', False)
                    confidence = result.get('confidence_score', 0)
                    
                    if detected and confidence > 50:
                        target = result.get('tp_high', result.get('tp_low', 0))
                        status = f"✅ {display_name}: {confidence}% Confidence"
                        if target > 0:
                            status += f" │ Target: {self.format_price(target)}"
                        section += f"{status}\n"
                    else:
                        section += f"❌ {display_name}: Not Detected\n"
        
        return section
    
    def format_orderflow(self, orderflow_results: Dict[str, Any]) -> str:
        """Format order flow analysis section"""
        section = "\nSMART MONEY CONCEPTS\n"
        
        if 'smc' in orderflow_results:
            smc = orderflow_results['smc']
            if smc.get('success', False):
                bos = "✅ Detected" if smc.get('bos_detected', False) else "❌ None"
                choch = "✅ Detected" if smc.get('choch_detected', False) else "❌ None"
                ob_hit = "✅ Active" if smc.get('order_block_hit', False) else "❌ None"
                liq_sweep = "✅ Bullish" if smc.get('liquidity_sweep', False) else "❌ None"
                confluence = smc.get('confluence_count', 0)
                
                section += f"BOS (Break of Structure): {bos}"
                if confluence > 0:
                    section += f" │ Confluence: {confluence} factors"
                section += f"\nCHoCH (Change of Character): {choch}\n"
                section += f"Order Block Hit: {ob_hit} │ Liquidity Sweep: {liq_sweep}\n"
        
        section += "\nVOLUME PROFILE\n"
        
        if 'cvd' in orderflow_results:
            cvd = orderflow_results['cvd']
            if cvd.get('success', False):
                cvd_value = cvd.get('cvd_value', 0)
                bias = cvd.get('bias', 'NEUTRAL')
                section += f"CVD (Cumulative Volume Delta): {cvd_value:+,.0f} ({bias})\n"
        
        if 'imbalance' in orderflow_results:
            imbalance = orderflow_results['imbalance']
            if imbalance.get('success', False):
                pattern_detected = imbalance.get('pattern_detected', False)
                if pattern_detected:
                    section += "Volume Imbalance: Fair Value Gap Detected\n"
                else:
                    section += "Volume Imbalance: No significant gaps\n"
        
        if 'absorption' in orderflow_results:
            absorption = orderflow_results['absorption']
            if absorption.get('success', False):
                pattern_detected = absorption.get('pattern_detected', False)
                if pattern_detected:
                    level = absorption.get('absorption_level', 0)
                    section += f"Absorption: Detected at {self.format_price(level)}\n"
                else:
                    section += "Absorption: No significant levels\n"
        
        if 'stop_sweep' in orderflow_results:
            stop_sweep = orderflow_results['stop_sweep']
            if stop_sweep.get('success', False):
                pattern_detected = stop_sweep.get('pattern_detected', False)
                section += f"Stop Sweep: {'Detected' if pattern_detected else 'None detected'}\n"
        
        return section
    
    def format_alerts(self, results: AnalysisResults, current_price: float) -> str:
        """Format alerts and signals section"""
        section = "\n⚠️  ALERTS & SIGNALS\n"
        
        alerts = []
        
        # Check for high confidence signals
        for category in [results.techin_results, results.pattern_results, results.orderflow_results]:
            for name, data in category.items():
                if isinstance(data, dict) and data.get('success', False):
                    confidence = data.get('confidence_score', 0)
                    signal = data.get('signal', 'NEUTRAL')
                    
                    if confidence >= 80:
                        alerts.append(f"🔴 HIGH:   {name.upper()} showing strong {signal} signal ({confidence}%)")
                    elif confidence >= 60:
                        alerts.append(f"🟡 MEDIUM: {name.upper()} indicating {signal} ({confidence}%)")
        
        # Check proximity to key levels
        for category in [results.techin_results, results.pattern_results, results.orderflow_results]:
            for name, data in category.items():
                if isinstance(data, dict) and data.get('success', False):
                    resistance = data.get('resistance_level', 0)
                    support = data.get('support_level', 0)
                    
                    if resistance > 0 and abs(current_price - resistance) / current_price < 0.02:
                        alerts.append(f"🔴 HIGH:   Price approaching key resistance at {self.format_price(resistance)}")
                    
                    if support > 0 and abs(current_price - support) / current_price < 0.02:
                        alerts.append(f"🟡 MEDIUM: Price near support level at {self.format_price(support)}")
        
        if not alerts:
            alerts.append("🟢 INFO:   No significant alerts at current levels")
        
        # Limit to top 5 alerts
        for alert in alerts[:5]:
            section += f"{alert}\n"
        
        return section
    
    def format_analysis_report(self, results: AnalysisResults, symbol: str, timeframe: str, candles: int) -> str:
        """Format complete analysis report"""
        # Get basic market data
        current_price = 0
        price_change_pct = 0
        
        # Extract price and calculate 24h change from OHLCV data
        for category in [results.techin_results, results.pattern_results, results.orderflow_results]:
            for name, data in category.items():
                if isinstance(data, dict) and data.get('success', False):
                    current_price = data.get('current_price', current_price)
                    
                    # Calculate 24h price change from raw OHLCV data if available
                    raw_data = data.get('raw_data', {})
                    ohlcv_data = raw_data.get('ohlcv_data', [])
                    if ohlcv_data and len(ohlcv_data) >= 24:  # Need at least 24 candles for 24h change
                        current_close = ohlcv_data[-1][4]  # Latest close price
                        past_close = ohlcv_data[-24][4] if len(ohlcv_data) >= 24 else ohlcv_data[0][4]  # 24 candles ago
                        if past_close > 0:
                            price_change_pct = ((current_close - past_close) / past_close) * 100
                    
                    break
            if current_price > 0:
                break
        
        # Build complete report
        report = self.format_header(symbol, timeframe, candles, current_price, price_change_pct)
        report += self.format_market_overview(results)
        report += self.format_key_levels(results)
        report += self.format_technical_indicators(results.techin_results)
        report += self.format_chart_patterns(results.pattern_results)
        report += self.format_orderflow(results.orderflow_results)
        report += self.format_alerts(results, current_price)
        
        return report


def format_analysis_output(techin_results: Dict[str, Any], pattern_results: Dict[str, Any], 
                         orderflow_results: Dict[str, Any], symbol: str, timeframe: str, 
                         candles: int) -> str:
    """
    Main function to format all analysis results into beautiful CLI output
    
    Args:
        techin_results: Results from all technical indicator analyze() functions
        pattern_results: Results from all pattern analyze() functions  
        orderflow_results: Results from all orderflow analyze() functions
        symbol: Trading symbol (e.g. 'BTC/USDT')
        timeframe: Analysis timeframe (e.g. '1h')
        candles: Number of candles analyzed
        
    Returns:
        Formatted CLI output string
    """
    formatter = CLIFormatter()
    
    results = AnalysisResults(
        techin_results=techin_results,
        pattern_results=pattern_results,
        orderflow_results=orderflow_results,
        market_data={}
    )
    
    return formatter.format_analysis_report(results, symbol, timeframe, candles)


if __name__ == "__main__":
    # Example usage
    sample_techin = {
        'ema_9_21': {'success': True, 'signal': 'BUY', 'confidence_score': 85, 'distance_from_ema': 1240.50},
        'macd': {'success': True, 'signal': 'BUY', 'confidence_score': 72, 'histogram_value': 45.2},
        'supertrend': {'success': True, 'signal': 'BUY', 'confidence_score': 88, 'supertrend_value': 66800.00},
        'rsi_14': {'success': True, 'rsi_value': 68.5, 'rsi_condition': 'NEUTRAL'}
    }
    
    sample_patterns = {
        'flag': {'success': True, 'pattern_detected': True, 'confidence_score': 82, 'tp_high': 69500},
        'triangle': {'success': True, 'pattern_detected': True, 'confidence_score': 76},
        'head_and_shoulders': {'success': True, 'pattern_detected': False}
    }
    
    sample_orderflow = {
        'smc': {'success': True, 'bos_detected': True, 'confluence_count': 4, 'liquidity_sweep': True},
        'cvd': {'success': True, 'cvd_value': 2450000, 'bias': 'BULLISH'}
    }
    
    output = format_analysis_output(sample_techin, sample_patterns, sample_orderflow, 
                                  'BTC/USDT', '1h', 100)
    print(output)