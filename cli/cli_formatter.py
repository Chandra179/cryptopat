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
        """Calculate overall market bias and confidence using industry-standard weighted approach"""
        # Industry-standard reliability weights based on statistical performance
        indicator_weights = {
            'macd': 1.2,           # Strong momentum indicator
            'ema_9_21': 1.3,       # Reliable trend indicator
            'supertrend': 1.4,     # High accuracy trend following
            'bollinger_bands': 1.0, # Standard volatility indicator
            'rsi_14': 0.8,         # Lagging momentum
            'atr_adx': 1.1,        # Good trend strength measure
            'obv': 0.9,            # Volume confirmation
            'vwap': 1.0,           # Institutional reference
            'smc': 1.5,            # Smart money concepts (high reliability when present)
            'flag': 1.3,           # Strong continuation pattern
            'triangle': 1.1,       # Reliable breakout pattern
            'head_and_shoulders': 1.4, # High reversal accuracy
            'double_top': 1.2,     # Good reversal signal
            'double_bottom': 1.2,  # Good reversal signal
            'wedge': 1.1,          # Moderate reliability
            'shark_pattern': 1.0,  # Harmonic pattern
            'butterfly_pattern': 1.0, # Harmonic pattern
            'elliott_wave': 0.9,   # Subjective interpretation
            'cvd': 1.2,            # Volume delta analysis
            'imbalance': 1.0,      # Market structure
            'absorption': 1.1,     # Liquidity analysis
            'stop_sweep': 1.3      # Smart money activity
        }
        
        signal_data = []
        
        # Collect signals with weights and confidences
        for category in [results.techin_results, results.pattern_results, results.orderflow_results]:
            for name, data in category.items():
                if isinstance(data, dict) and data.get('success', False):
                    signal = data.get('signal', 'NEUTRAL')
                    confidence = data.get('confidence_score', 0)
                    weight = indicator_weights.get(name, 1.0)
                    
                    # Only include signals with meaningful confidence (>30%)
                    if confidence > 30:
                        if signal in ['BUY', 'BULLISH']:
                            signal_value = 1
                        elif signal in ['SELL', 'BEARISH']:
                            signal_value = -1
                        else:
                            signal_value = 0
                        
                        signal_data.append((signal_value, weight, confidence))
        
        if not signal_data:
            return 'NEUTRAL', 0
        
        try:
            # Calculate weighted bias score: sum(signal * weight) / sum(weight)
            total_weight = sum(weight for _, weight, _ in signal_data)
            if total_weight == 0:
                return 'NEUTRAL', 0
            
            weighted_bias = sum(signal * weight for signal, weight, _ in signal_data) / total_weight
            
            # Calculate average confidence (not weighted to avoid inflating confidence)
            avg_confidence = sum(confidence for _, _, confidence in signal_data) / len(signal_data)
            
            # Apply confidence boost for signal agreement
            bullish_signals = sum(1 for signal, _, _ in signal_data if signal > 0)
            bearish_signals = sum(1 for signal, _, _ in signal_data if signal < 0)
            total_directional = bullish_signals + bearish_signals
            
            if total_directional > 0:
                agreement_ratio = max(bullish_signals, bearish_signals) / total_directional
                # Boost confidence for high agreement (70%+ agreement gets boost)
                if agreement_ratio >= 0.7:
                    confidence_boost = min(15, (agreement_ratio - 0.7) * 50)
                    avg_confidence = min(95, avg_confidence + confidence_boost)
            
            overall_confidence = max(0, min(95, int(avg_confidence)))
            
        except (ValueError, TypeError, ZeroDivisionError):
            return 'NEUTRAL', 0
        
        # Statistical significance thresholds (normalized for -1 to 1 range)
        strong_threshold = 0.4   # 70% directional agreement
        weak_threshold = 0.15    # 57.5% directional agreement
        
        if weighted_bias > strong_threshold:
            return 'BULLISH', overall_confidence
        elif weighted_bias < -strong_threshold:
            return 'BEARISH', overall_confidence
        elif abs(weighted_bias) > weak_threshold:
            bias_type = 'BULLISH' if weighted_bias > 0 else 'BEARISH'
            # Reduce confidence for weak signals
            return bias_type, max(30, int(overall_confidence * 0.7))
        else:
            return 'NEUTRAL', min(60, overall_confidence)
    
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
        
        # Calculate risk/reward using industry-standard expected value approach
        rr_data = []
        for category in [results.techin_results, results.pattern_results, results.orderflow_results]:
            for name, data in category.items():
                if isinstance(data, dict) and data.get('success', False):
                    rr = data.get('rr_ratio', 0)
                    confidence = data.get('confidence_score', 0)
                    
                    # Only include realistic R/R ratios (0.5 to 5.0 range)
                    if 0.5 <= rr <= 5.0 and confidence > 30:
                        # Convert confidence to success probability (conservative estimate)
                        success_prob = min(0.7, confidence / 100)  # Cap at 70% max
                        expected_value = rr * success_prob
                        rr_data.append((rr, expected_value, confidence))
        
        if rr_data:
            # Weight by confidence and expected value
            total_weight = sum(conf for _, _, conf in rr_data)
            if total_weight > 0:
                avg_rr = sum(rr * conf for rr, _, conf in rr_data) / total_weight
                # Ensure reasonable bounds
                avg_rr = max(0.5, min(3.0, avg_rr))
            else:
                avg_rr = 0
        else:
            avg_rr = 0
        
        overview = f"""
📊 MARKET OVERVIEW
Overall Bias: {bias_emoji} {overall_bias} (Confidence: {confidence}%)
Primary Signal: {self.get_signal_emoji(primary_signal)} {primary_signal}
Risk/Reward: 1:{avg_rr:.1f}
Market Regime: {'BULL_MARKET' if overall_bias == 'BULLISH' else 'BEAR_MARKET' if overall_bias == 'BEARISH' else 'SIDEWAYS'} (Moderate Volatility)
"""
        return overview
    
    def format_key_levels(self, results: AnalysisResults) -> str:
        """Format key price levels section using price clustering"""
        # Collect all support/resistance levels with weights
        supports = []
        resistances = []
        stop_losses = []
        targets = []
        
        # Indicator reliability weights for level clustering
        level_weights = {
            'smc': 1.5, 'flag': 1.3, 'head_and_shoulders': 1.4,
            'double_top': 1.2, 'double_bottom': 1.2, 'triangle': 1.1,
            'supertrend': 1.3, 'bollinger_bands': 1.0, 'vwap': 1.1,
            'shark_pattern': 1.0, 'butterfly_pattern': 1.0
        }
        
        for category in [results.techin_results, results.pattern_results, results.orderflow_results]:
            for name, data in category.items():
                if isinstance(data, dict) and data.get('success', False):
                    weight = level_weights.get(name, 1.0)
                    confidence = data.get('confidence_score', 50) / 100  # Convert to 0-1
                    level_strength = weight * confidence
                    
                    if 'support_level' in data and data['support_level'] > 0:
                        supports.append((data['support_level'], level_strength))
                    if 'resistance_level' in data and data['resistance_level'] > 0:
                        resistances.append((data['resistance_level'], level_strength))
                    if 'stop_zone' in data and data['stop_zone'] > 0:
                        stop_losses.append((data['stop_zone'], level_strength))
                    if 'tp_low' in data and data['tp_low'] > 0:
                        targets.append((data['tp_low'], level_strength))
                    if 'tp_high' in data and data['tp_high'] > 0:
                        targets.append((data['tp_high'], level_strength))
        
        # Industry-standard price clustering (within 1.5% considered same level)
        def cluster_levels(levels_with_weights, cluster_threshold=0.015):
            if not levels_with_weights:
                return 0
            
            # Sort by price
            sorted_levels = sorted(levels_with_weights, key=lambda x: x[0])
            clusters = []
            
            for price, weight in sorted_levels:
                # Find if price belongs to existing cluster
                added_to_cluster = False
                for cluster in clusters:
                    cluster_center = sum(p * w for p, w in cluster) / sum(w for p, w in cluster)
                    if abs(price - cluster_center) / cluster_center <= cluster_threshold:
                        cluster.append((price, weight))
                        added_to_cluster = True
                        break
                
                if not added_to_cluster:
                    clusters.append([(price, weight)])
            
            # Find strongest cluster (highest total weight)
            if not clusters:
                return 0
            
            strongest_cluster = max(clusters, key=lambda c: sum(w for p, w in c))
            # Return weighted average of strongest cluster
            total_weight = sum(w for p, w in strongest_cluster)
            return sum(p * w for p, w in strongest_cluster) / total_weight if total_weight > 0 else 0
        
        # Calculate clustered levels
        avg_support = cluster_levels(supports)
        avg_resistance = cluster_levels(resistances)
        avg_stop = cluster_levels(stop_losses)
        
        # Determine current price for proper target ordering
        current_price = 0
        for category in [results.techin_results, results.pattern_results, results.orderflow_results]:
            for name, data in category.items():
                if isinstance(data, dict) and data.get('success', False):
                    current_price = data.get('current_price', current_price)
                    break
            if current_price > 0:
                break
        
        # Determine market bias for proper target ordering
        overall_bias, _ = self.calculate_overall_bias(results)
        
        # Sort targets appropriately based on market bias and current price
        if targets and current_price > 0:
            # Extract just the prices from targets (they have weights)
            target_prices = [price for price, weight in targets]
            
            if overall_bias == 'BULLISH':
                # For bullish bias: targets should be above current price
                valid_targets = [t for t in target_prices if t > current_price * 1.005]  # At least 0.5% above
                valid_targets.sort()  # Ascending order: TP1 closer, TP2 further
            elif overall_bias == 'BEARISH':
                # For bearish bias: targets should be below current price  
                valid_targets = [t for t in target_prices if t < current_price * 0.995]  # At least 0.5% below
                valid_targets.sort(reverse=True)  # Descending order: TP1 closer, TP2 further
            else:
                # Neutral: use closest targets regardless of direction
                valid_targets = [t for t in target_prices if abs(t - current_price) > 0.005 * current_price]
                valid_targets.sort(key=lambda x: abs(x - current_price))
            
            if valid_targets:
                tp1 = valid_targets[0] if len(valid_targets) > 0 else 0
                tp2 = valid_targets[1] if len(valid_targets) > 1 else 0
            else:
                # Fallback: use clustered targets if no valid targets found
                clustered_target = cluster_levels(targets)
                tp1 = clustered_target if clustered_target > 0 else 0
                tp2 = 0
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
                section += f"ATR/ADX:      ATR: {atr:.5f} | ADX: {adx:.5f} ({trend_strength} Trend)\n"
        
        # Volume indicators
        section += "\nVOLUME INDICATORS\n"
        
        if 'obv' in techin_results:
            obv = techin_results['obv']
            if isinstance(obv, dict) and obv.get('obv_value') is not None:
                signal = obv.get('signal', 'NEUTRAL')
                obv_value = obv.get('obv_value', 0)
                # Convert to appropriate units (industry standard formatting)
                if abs(obv_value) >= 1000000:
                    obv_millions = obv_value / 1000000
                    unit = 'M'
                elif abs(obv_value) >= 1000:
                    obv_millions = obv_value / 1000
                    unit = 'K'
                else:
                    obv_millions = obv_value
                    unit = ''
                trend_conf = obv.get('trend_confirmation', 'NEUTRAL')
                # Format based on scale - integers for raw values, decimals for scaled
                if unit == '':
                    section += f"OBV:          {trend_conf} ({obv_millions:+.0f})\n"
                else:
                    section += f"OBV:          {trend_conf} ({obv_millions:+.1f}{unit})\n"
        
        if 'vwap' in techin_results:
            vwap = techin_results['vwap']
            if isinstance(vwap, dict) and vwap.get('success', False):
                # Get latest signal from VWAP analysis
                latest_signal = vwap.get('latest_signal', {})
                if latest_signal:
                    signal = latest_signal.get('signal', 'NEUTRAL')
                    bias = latest_signal.get('bias', 'NEUTRAL')
                    section += f"VWAP:         {self.get_signal_emoji(signal)} {signal} ({bias})\n"
        
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
        
        # Timeframe mapping to minutes (industry standard)
        timeframe_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        
        # Extract price and calculate proper 24h change from OHLCV data
        for category in [results.techin_results, results.pattern_results, results.orderflow_results]:
            for _, data in category.items():
                if isinstance(data, dict) and data.get('success', False):
                    current_price = data.get('current_price', current_price)
                    
                    # Calculate 24h price change with proper timeframe handling
                    raw_data = data.get('raw_data', {})
                    ohlcv_data = raw_data.get('ohlcv_data', [])
                    
                    if ohlcv_data and timeframe in timeframe_minutes:
                        tf_minutes = timeframe_minutes[timeframe]
                        # Calculate how many candles represent 24 hours
                        candles_per_24h = int(1440 / tf_minutes)  # 1440 minutes = 24 hours
                        
                        # Ensure we have enough data for 24h comparison
                        if len(ohlcv_data) > candles_per_24h:
                            current_close = ohlcv_data[-1][4]  # Latest close price
                            past_close = ohlcv_data[-candles_per_24h - 1][4]  # 24h ago
                            
                            if past_close > 0:
                                price_change_pct = ((current_close - past_close) / past_close) * 100
                        elif len(ohlcv_data) > 1:
                            # Fallback: use oldest available data if less than 24h
                            current_close = ohlcv_data[-1][4]
                            past_close = ohlcv_data[0][4]
                            if past_close > 0:
                                # Scale the change to represent approximate 24h equivalent
                                hours_available = (len(ohlcv_data) * tf_minutes) / 60
                                raw_change = ((current_close - past_close) / past_close) * 100
                                price_change_pct = raw_change * (24 / hours_available) if hours_available > 0 else 0
                    
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
        
        report += f"\n{self.separator}\n"
        report += "                    Analysis Complete - Trade Safely!                     \n"
        report += f"{self.separator}\n"
        
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