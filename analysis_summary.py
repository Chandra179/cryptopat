"""
Analysis Summary Engine for CryptoPat
Processes technical indicator results and generates comprehensive market consensus
"""

from typing import Dict, List, Tuple, Any
import statistics
import yaml

class AnalysisSummary:
    def __init__(self, config_path: str = 'summary.yaml'):
        self.config = self._load_config(config_path)
        self.categories = self.config['categories']
        self.signal_mappings = self.config['signal_mappings']
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file '{config_path}' not found")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

    def get_signal_score(self, indicator_name: str, signal: str) -> float:
        """Convert signal string to numerical score (-1 to +1)"""
        if indicator_name in self.signal_mappings:
            return self.signal_mappings[indicator_name].get(signal, 0.0)
        else:
            return self.signal_mappings['generic'].get(signal, 0.0)
    
    def categorize_indicators(self, results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Group indicator results by category"""
        categorized = {category: {} for category in self.categories}
        
        for indicator_name, result in results.items():
            for category, indicators in self.categories.items():
                if indicator_name in indicators:
                    categorized[category][indicator_name] = result
                    break
        
        return categorized
    
    def calculate_category_consensus(self, category_results: Dict[str, Any]) -> Tuple[float, str, str]:
        """Calculate consensus for a category"""
        if not category_results:
            return 0.0, "neutral", "⚪"
        
        scores = []
        for indicator_name, result in category_results.items():
            if result and 'signal' in result:
                score = self.get_signal_score(indicator_name, result['signal'])
                scores.append(score)
        
        if not scores:
            return 0.0, "neutral", "⚪"
        
        avg_score = statistics.mean(scores)
        
        # Determine sentiment and emoji using config thresholds
        bearish_threshold = self.config['category_consensus']['bearish_threshold']
        bullish_threshold = self.config['category_consensus']['bullish_threshold']
        
        if avg_score <= bearish_threshold:
            sentiment = "bearish"
            emoji = self.config['display']['emojis']['bearish']
        elif avg_score >= bullish_threshold:
            sentiment = "bullish" 
            emoji = self.config['display']['emojis']['bullish']
        else:
            sentiment = "neutral"
            emoji = self.config['display']['emojis']['neutral']
        
        return avg_score, sentiment, emoji
    
    def calculate_overall_consensus(self, results: Dict[str, Any]) -> Tuple[float, int, int, int, int]:
        """Calculate overall market consensus"""
        all_scores = []
        bear_count = 0
        neutral_count = 0
        bull_count = 0
        total_signals = 0
        
        for indicator_name, result in results.items():
            if result and 'signal' in result:
                score = self.get_signal_score(indicator_name, result['signal'])
                all_scores.append(score)
                total_signals += 1
                
                bear_threshold = self.config['signal_counting']['bear_threshold']
                bull_threshold = self.config['signal_counting']['bull_threshold']
                
                if score <= bear_threshold:
                    bear_count += 1
                elif score >= bull_threshold:
                    bull_count += 1
                else:
                    neutral_count += 1
        
        overall_score = statistics.mean(all_scores) if all_scores else 0.0
        return overall_score, total_signals, bear_count, neutral_count, bull_count
    
    def get_bias_description(self, score: float) -> Tuple[str, str]:
        """Get bias description and emoji from score using config thresholds"""
        bias_config = self.config['bias_classification']
        
        if score <= bias_config['strongly_bearish']['threshold']:
            bias = bias_config['strongly_bearish']
            return bias['description'], bias['emoji']
        elif score <= bias_config['moderately_bearish']['threshold']:
            bias = bias_config['moderately_bearish']
            return bias['description'], bias['emoji']
        elif score <= bias_config['weakly_bearish']['threshold']:
            bias = bias_config['weakly_bearish']
            return bias['description'], bias['emoji']
        elif score >= bias_config['strongly_bullish']['threshold']:
            bias = bias_config['strongly_bullish']
            return bias['description'], bias['emoji']
        elif score >= bias_config['moderately_bullish']['threshold']:
            bias = bias_config['moderately_bullish']
            return bias['description'], bias['emoji']
        elif score >= bias_config['weakly_bullish']['threshold']:
            bias = bias_config['weakly_bullish']
            return bias['description'], bias['emoji']
        else:
            bias = bias_config['neutral']
            return bias['description'], bias['emoji']
    
    def get_key_drivers(self, results: Dict[str, Any], limit: int = 3) -> List[str]:
        """Identify key signal drivers with specific values"""
        drivers = []
        
        for indicator_name, result in results.items():
            if not result:
                continue
                
            # Extract key values based on indicator type
            if indicator_name == 'Chaikin Money Flow' and 'cmf' in result:
                cmf_val = result['cmf']
                cmf_threshold = self.config['key_drivers']['chaikin_money_flow']['significance_threshold']
                if abs(cmf_val) > cmf_threshold:
                    drivers.append(f"CMF {cmf_val:.3f} ({'selling' if cmf_val < 0 else 'buying'})")
            
            elif indicator_name == 'RSI' and 'rsi' in result:
                rsi_val = result['rsi']
                rsi_config = self.config['key_drivers']['rsi']
                if rsi_val < rsi_config['oversold_threshold']:
                    drivers.append(f"RSI {rsi_val:.1f} (oversold)")
                elif rsi_val > rsi_config['overbought_threshold']:
                    drivers.append(f"RSI {rsi_val:.1f} (overbought)")
            
            elif indicator_name == 'Parabolic SAR' and 'signal' in result:
                if result['signal'] in self.config['key_drivers']['highlight_signals']:
                    drivers.append(f"Parabolic SAR ({result['signal'].replace('_', ' ')})")
            
            elif indicator_name == 'MACD' and 'signal' in result:
                if result['signal'] in self.config['key_drivers']['highlight_signals']:
                    drivers.append(f"MACD ({result['signal'].replace('_', ' ')})")
        
        return drivers[:limit]
    
    def get_current_price(self, results: Dict[str, Any]) -> float:
        """Extract current price from any available indicator result"""
        for result in results.values():
            if result and 'current_price' in result:
                return result['current_price']
        return 0.0
    
    def get_average_support_resistance(self, results: Dict[str, Any], current_price: float) -> Tuple[float, float, str]:
        """Calculate average support and resistance levels from key indicators"""
        supports = []
        resistances = []
        
        # Extract from Pivot Points (most reliable)
        if 'Pivot Point' in results and results['Pivot Point']:
            pivot_result = results['Pivot Point']
            if 'support_1' in pivot_result:
                supports.append(pivot_result['support_1'])
            if 'support_2' in pivot_result:
                supports.append(pivot_result['support_2'])
            if 'resistance_1' in pivot_result:
                resistances.append(pivot_result['resistance_1'])
            if 'resistance_2' in pivot_result:
                resistances.append(pivot_result['resistance_2'])
        
        # Extract from Bollinger Bands
        if 'Bollinger Bands' in results and results['Bollinger Bands']:
            bb_result = results['Bollinger Bands']
            if 'lower_band' in bb_result:
                supports.append(bb_result['lower_band'])
            if 'upper_band' in bb_result:
                resistances.append(bb_result['upper_band'])
        
        # Extract from Ichimoku Cloud
        if 'Ichimoku Cloud' in results and results['Ichimoku Cloud']:
            ichimoku_result = results['Ichimoku Cloud']
            if 'cloud_bottom' in ichimoku_result:
                supports.append(ichimoku_result['cloud_bottom'])
            if 'cloud_top' in ichimoku_result:
                resistances.append(ichimoku_result['cloud_top'])
        
        # Extract from SuperTrend (dynamic S/R)
        if 'SuperTrend' in results and results['SuperTrend']:
            st_result = results['SuperTrend']
            if 'support_resistance' in st_result and 'price_above_supertrend' in st_result:
                sr_level = st_result['support_resistance']
                if st_result['price_above_supertrend']:
                    supports.append(sr_level)  # Acting as support
                else:
                    resistances.append(sr_level)  # Acting as resistance
        
        if not current_price:
            return 0.0, 0.0, "S/R Levels: Insufficient price data"
        
        # Filter levels relative to current price
        valid_supports = [s for s in supports if s < current_price and s > 0]
        valid_resistances = [r for r in resistances if r > current_price and r > 0]
        
        # Calculate averages
        avg_support = statistics.mean(valid_supports) if valid_supports else 0.0
        avg_resistance = statistics.mean(valid_resistances) if valid_resistances else 0.0
        
        # Generate summary string using config formatting
        price_precision = self.config['display']['price_precision']
        support_emoji = self.config['display']['emojis']['support']
        
        sr_lines = []
        if avg_support > 0:
            support_count = len(valid_supports)
            sr_lines.append(f"{support_emoji} Avg Support: ${avg_support:,.{price_precision}f} ({support_count} levels)")
        
        if avg_resistance > 0:
            resistance_count = len(valid_resistances)
            sr_lines.append(f"{support_emoji} Avg Resistance: ${avg_resistance:,.{price_precision}f} ({resistance_count} levels)")
        
        if not sr_lines:
            sr_summary = f"{support_emoji} S/R Levels: No clear levels identified"
        else:
            sr_summary = "\n".join(sr_lines)
        
        return avg_support, avg_resistance, sr_summary
    
    def generate_action_recommendation(self, overall_score: float, current_price: float = None, avg_support: float = 0.0, avg_resistance: float = 0.0) -> str:
        """Generate trading action recommendation with S/R levels using config thresholds"""
        action_config = self.config['action_recommendations']
        price_precision = self.config['display']['price_precision']
        
        if overall_score <= action_config['strong_bearish_threshold']:
            if avg_support > 0:
                return f"Avoid aggressive longs. Short on confirmed breakdown below **${current_price:,.{price_precision}f}** with **volume > avg**. Key support watch: **${avg_support:,.{price_precision}f}**."
            else:
                return "Avoid aggressive longs. Consider shorts on breakdown with volume confirmation."
        
        elif overall_score >= action_config['strong_bullish_threshold']:
            if avg_resistance > 0:
                return f"Favor longs on pullbacks. Target breakout above **${avg_resistance:,.{price_precision}f}** with volume. Avoid shorts unless clear reversal signals."
            else:
                return "Favor long positions on pullbacks. Look for volume-confirmed breakouts."
        
        else:
            if avg_support > 0 and avg_resistance > 0:
                return f"Mixed signals suggest range **${avg_support:,.{price_precision}f}** - **${avg_resistance:,.{price_precision}f}**. Trade breakouts with volume confirmation."
            else:
                return "Mixed signals suggest range-bound action. Trade breakouts with volume confirmation. Avoid large directional bets."

    def generate_summary(self, results: Dict[str, Any], symbol: str = "BTC/USDT", timeframe: str = "1d") -> str:
        """Generate comprehensive technical analysis summary"""
        # Calculate overall consensus
        overall_score, total_signals, bear_count, neutral_count, bull_count = self.calculate_overall_consensus(results)
        
        # Get bias description  
        bias_desc, _ = self.get_bias_description(overall_score)
        
        # Calculate confidence (higher when signals agree)
        confidence_precision = self.config['display']['confidence_precision']
        confidence = max(bear_count, bull_count) / total_signals * 100 if total_signals > 0 else 0
        
        # Categorize results
        categorized = self.categorize_indicators(results)
        
        # Calculate category consensus
        category_summaries = {}
        for category, category_results in categorized.items():
            score, sentiment, emoji = self.calculate_category_consensus(category_results)
            category_summaries[category] = {
                'score': score,
                'sentiment': sentiment, 
                'emoji': emoji,
                'details': self.get_category_details(category, category_results)
            }
        
        # Get key drivers
        key_drivers = self.get_key_drivers(results)
        
        # Get current price for S/R analysis
        current_price = self.get_current_price(results)
        
        # Calculate average support/resistance levels
        avg_support, avg_resistance, sr_summary = self.get_average_support_resistance(results, current_price)

        # Format output using config precision settings
        score_precision = self.config['display']['score_precision']
        timeframe_display = timeframe.upper().replace('M', 'Month').replace('D', 'D').replace('W', 'W')
        analysis_emoji = self.config['display']['emojis']['analysis']
        
        summary = f"""=========================================
{analysis_emoji} {symbol} ({timeframe_display}) — Technical Consensus
=========================================
Overall Bias: {bias_desc}  (Score: {overall_score:+.{score_precision}f} / −1..+1)
Confidence: {confidence:.{confidence_precision}f}% ({total_signals} signals — {bear_count} Bear / {neutral_count} Neutral / {bull_count} Bull)

Trend: {category_summaries['Trend']['emoji']} {category_summaries['Trend']['sentiment'].title()} {category_summaries['Trend']['details']}
Volatility: {category_summaries['Volatility']['emoji']} {category_summaries['Volatility']['sentiment'].title()} {category_summaries['Volatility']['details']}
Volume: {category_summaries['Volume']['emoji']} {category_summaries['Volume']['sentiment'].title()} {category_summaries['Volume']['details']}
Momentum: {category_summaries['Momentum']['emoji']} {category_summaries['Momentum']['sentiment'].title()} {category_summaries['Momentum']['details']}
S/R: {category_summaries['S/R']['emoji']} {category_summaries['S/R']['sentiment'].title()} {category_summaries['S/R']['details']}
Alt Chart: {category_summaries['Alt Chart']['emoji']} {category_summaries['Alt Chart']['sentiment'].title()} {category_summaries['Alt Chart']['details']}

Top drivers: {', '.join(key_drivers) if key_drivers else 'No strong signals detected'}
Signal age: Current analysis (live data)

{sr_summary}

{self.config['display']['emojis']['action']} Action: {self.generate_action_recommendation(overall_score, current_price, avg_support, avg_resistance)}"""

        return summary
    
    def get_category_details(self, category: str, category_results: Dict[str, Any]) -> str:
        """Generate category-specific details"""
        if not category_results:
            return "(no data)"
        
        details = []
        
        if category == 'Trend':
            for indicator, result in category_results.items():
                if not result:
                    continue
                if indicator == 'MACD' and 'signal' in result:
                    if 'weakening' in result['signal']:
                        details.append("MACD weakening")
                    elif 'cross' in result['signal']:
                        details.append(f"MACD {result['signal'].replace('_', ' ')}")
                elif indicator == 'Parabolic SAR' and 'signal' in result:
                    if 'bearish' in result['signal'] or 'sell' in result['signal']:
                        details.append("SAR bearish")
                    elif 'bullish' in result['signal'] or 'buy' in result['signal']:
                        details.append("SAR bullish")
                elif indicator == 'EMA 20/50' and 'signal' in result:
                    details.append("EMA mixed" if result['signal'] == 'neutral' else f"EMA {result['signal']}")
        
        elif category == 'Volume':
            for indicator, result in category_results.items():
                if not result:
                    continue
                if indicator == 'Chaikin Money Flow' and 'cmf' in result:
                    cmf_val = result['cmf']
                    details.append(f"CMF {cmf_val:+.3f}")
                elif indicator == 'OBV' and 'signal' in result:
                    if result['signal'] != 'neutral':
                        details.append(f"OBV {result['signal']}")
                elif indicator == 'VWAP' and 'signal' in result:
                    if 'above' in str(result.get('price_vs_vwap', '')).lower():
                        details.append("VWAP elevated")
                    elif 'below' in str(result.get('price_vs_vwap', '')).lower():
                        details.append("VWAP support")
        
        elif category == 'Momentum':
            for indicator, result in category_results.items():
                if not result:
                    continue
                if indicator == 'RSI' and 'rsi' in result:
                    rsi_val = result['rsi']
                    details.append(f"RSI {rsi_val:.1f}")
        
        return f"({', '.join(details)})" if details else "(mixed signals)"