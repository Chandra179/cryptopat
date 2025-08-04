#!/usr/bin/env python3
"""
Pattern Enhancer - Integration module for enhanced calculations
Updates existing pattern detection with industry-standard support/resistance, stops, and targets
"""

from typing import Dict, List, Any, Optional
import sys
import os

# Add the parent directory to sys.path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from techin.enhanced_levels import EnhancedLevelsCalculator, TradingLevel
from techin.risk_manager import RiskManager, TradeSetup


class PatternEnhancer:
    """Enhances existing pattern analysis with industry-standard calculations"""
    
    def __init__(self, account_balance: float = 10000):
        self.levels_calc = EnhancedLevelsCalculator()
        self.risk_manager = RiskManager(account_balance)
        
    def enhance_pattern_result(self, pattern_result: Dict[str, Any], 
                             ohlcv_data: List[List],
                             symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Enhance existing pattern analysis result with industry-standard calculations
        
        Args:
            pattern_result: Original pattern analysis result
            ohlcv_data: OHLCV data for calculations
            symbol: Trading symbol
            
        Returns:
            Enhanced result with improved support/resistance, stops, and targets
        """
        
        if not pattern_result.get('success', False) or len(ohlcv_data) < 50:
            return pattern_result
        
        # Get basic info from pattern result
        current_price = pattern_result.get('current_price', ohlcv_data[-1][4])
        signal = pattern_result.get('signal', 'HOLD')
        confidence = pattern_result.get('confidence_score', 50)
        
        # Run comprehensive levels analysis
        levels_analysis = self.levels_calc.analyze_comprehensive_levels(
            ohlcv_data, symbol, self.risk_manager.account_balance
        )
        
        if not levels_analysis.get('success', False):
            return pattern_result  # Return original if enhancement fails
        
        # Calculate ATR for risk management
        atr_values = self.levels_calc.calculate_atr(ohlcv_data)
        current_atr = atr_values[-1] if atr_values else current_price * 0.02
        
        # Enhanced support and resistance levels
        enhanced_support = levels_analysis.get('support_level', 0)
        enhanced_resistance = levels_analysis.get('resistance_level', 0)
        
        # Use enhanced levels if they're stronger, otherwise keep pattern-specific levels  
        original_support = pattern_result.get('support_level', 0)
        original_resistance = pattern_result.get('resistance_level', 0)
        
        # Choose the best support/resistance based on strength and proximity
        final_support = self._choose_best_level(
            original_support, enhanced_support, current_price, 'support'
        )
        final_resistance = self._choose_best_level(
            original_resistance, enhanced_resistance, current_price, 'resistance'
        )
        
        # Enhanced stop loss calculation
        enhanced_stop = self._calculate_enhanced_stop_loss(
            current_price, signal, current_atr, final_support, final_resistance,
            pattern_result.get('stop_zone', 0)
        )
        
        # Enhanced take profit calculation
        enhanced_tp1, enhanced_tp2 = self._calculate_enhanced_targets(
            current_price, enhanced_stop, signal, current_atr,
            final_resistance, final_support,
            pattern_result.get('tp_low', 0),
            pattern_result.get('tp_high', 0)
        )
        
        # Create trade setup for risk management
        trade_setup = TradeSetup(
            symbol=symbol,
            entry_price=current_price,
            stop_loss=enhanced_stop,
            take_profit_1=enhanced_tp1,
            take_profit_2=enhanced_tp2,
            signal=signal,
            confidence=confidence,
            atr_value=current_atr,
            timeframe=pattern_result.get('timeframe', '1h')
        )
        
        # Calculate position sizing
        position_calc = self.risk_manager.calculate_position_size(trade_setup)
        
        # Calculate enhanced risk/reward ratio
        risk = abs(current_price - enhanced_stop)
        reward = abs(enhanced_tp1 - current_price)
        enhanced_rr = reward / risk if risk > 0 else 0
        
        # Update the pattern result with enhanced calculations
        enhanced_result = pattern_result.copy()
        
        enhanced_result.update({
            # Enhanced levels
            'support_level': round(final_support, 4),
            'resistance_level': round(final_resistance, 4),
            'support_strength': levels_analysis.get('support_strength', 5),
            'resistance_strength': levels_analysis.get('resistance_strength', 5),
            
            # Enhanced trading levels
            'stop_zone': round(enhanced_stop, 4),
            'tp_low': round(enhanced_tp1, 4),
            'tp_high': round(enhanced_tp2, 4),
            'entry_price': round(current_price, 4),
            
            # Enhanced risk metrics
            'rr_ratio': round(enhanced_rr, 2),
            'atr_value': round(current_atr, 4),
            'atr_percentage': round((current_atr / current_price) * 100, 2),
            
            # Position sizing (industry standard)
            'position_sizing': position_calc['final_position'],
            'risk_management': {
                'recommended_units': position_calc['final_position']['units'],
                'position_value': position_calc['final_position']['position_value'],
                'risk_amount': position_calc['final_position']['risk_amount'],
                'risk_percentage': position_calc['final_position']['risk_percent'],
                'max_drawdown': position_calc['risk_metrics']['max_drawdown_percent']
            },
            
            # Expected returns
            'expected_returns': position_calc['expected_returns'],
            
            # Enhancement metadata
            'enhancement_applied': True,
            'enhancement_details': {
                'levels_analyzed': levels_analysis.get('all_levels_count', 0),
                'confluence_zones': levels_analysis.get('confluence_zones_count', 0),
                'fibonacci_levels': levels_analysis.get('level_details', {}).get('fibonacci_levels', 0),
                'pivot_levels': levels_analysis.get('level_details', {}).get('pivot_levels', 0),
                'volume_levels': levels_analysis.get('level_details', {}).get('volume_levels', 0),
                'enhancement_confidence': self._calculate_enhancement_confidence(levels_analysis)
            },
            
            # Original values for comparison
            'original_values': {
                'support_level': original_support,
                'resistance_level': original_resistance,
                'stop_zone': pattern_result.get('stop_zone', 0),
                'tp_low': pattern_result.get('tp_low', 0),
                'tp_high': pattern_result.get('tp_high', 0),
                'rr_ratio': pattern_result.get('rr_ratio', 0)
            }
        })
        
        return enhanced_result
    
    def _choose_best_level(self, original_level: float, enhanced_level: float,
                          current_price: float, level_type: str) -> float:
        """Choose the best support/resistance level between original and enhanced"""
        
        if original_level == 0:
            return enhanced_level
        if enhanced_level == 0:
            return original_level
        
        # Calculate distances from current price
        original_distance = abs(current_price - original_level) / current_price
        enhanced_distance = abs(current_price - enhanced_level) / current_price
        
        # Prefer levels that are closer but not too close (minimum 0.5% away)
        min_distance = 0.005  # 0.5%
        max_distance = 0.05   # 5%
        
        original_valid = min_distance <= original_distance <= max_distance
        enhanced_valid = min_distance <= enhanced_distance <= max_distance
        
        if level_type == 'support':
            # For support, prefer higher levels (stronger support) if both valid
            if original_valid and enhanced_valid:
                return max(original_level, enhanced_level)
            elif original_valid:
                return original_level
            elif enhanced_valid:
                return enhanced_level
            else:
                # Neither ideal, choose closer one
                return original_level if original_distance < enhanced_distance else enhanced_level
        
        else:  # resistance
            # For resistance, prefer lower levels (closer resistance) if both valid
            if original_valid and enhanced_valid:
                return min(original_level, enhanced_level)
            elif original_valid:
                return original_level
            elif enhanced_valid:
                return enhanced_level
            else:
                # Neither ideal, choose closer one
                return original_level if original_distance < enhanced_distance else enhanced_level
    
    def _calculate_enhanced_stop_loss(self, current_price: float, signal: str,
                                    atr_value: float, support_level: float,
                                    resistance_level: float, original_stop: float) -> float:
        """Calculate enhanced stop loss using multiple methods"""
        
        # Method 1: ATR-based stop
        atr_stop = self.levels_calc.calculate_stop_loss(
            current_price, signal, atr_value, method='atr'
        )
        
        # Method 2: Support/Resistance based stop
        sr_stop = self.levels_calc.calculate_stop_loss(
            current_price, signal, atr_value, support_level, resistance_level,
            method='support_resistance'
        )
        
        # Method 3: Volatility adjusted stop
        vol_stop = self.levels_calc.calculate_stop_loss(
            current_price, signal, atr_value, method='volatility_adjusted'
        )
        
        # Choose the most conservative (furthest from price) stop
        stops = [atr_stop, sr_stop, vol_stop]
        if original_stop != 0:
            stops.append(original_stop)
        
        if signal.upper() in ['BUY', 'LONG']:
            # For long positions, choose highest stop (most conservative)
            final_stop = max(stop for stop in stops if stop < current_price)
        else:
            # For short positions, choose lowest stop (most conservative) 
            final_stop = min(stop for stop in stops if stop > current_price)
        
        return final_stop
    
    def _calculate_enhanced_targets(self, current_price: float, stop_loss: float,
                                  signal: str, atr_value: float, resistance_level: float,
                                  support_level: float, original_tp1: float,
                                  original_tp2: float) -> tuple:
        """Calculate enhanced take profit targets"""
        
        # Method 1: Risk/Reward based targets
        rr_tp1, rr_tp2 = self.levels_calc.calculate_take_profits(
            current_price, stop_loss, signal, atr_value,
            resistance_level, support_level, 'risk_reward'
        )
        
        # Method 2: ATR multiple targets
        atr_tp1, atr_tp2 = self.levels_calc.calculate_take_profits(
            current_price, stop_loss, signal, atr_value,
            resistance_level, support_level, 'atr_multiple'
        )
        
        # Method 3: Support/Resistance targets
        sr_tp1, sr_tp2 = self.levels_calc.calculate_take_profits(
            current_price, stop_loss, signal, atr_value,
            resistance_level, support_level, 'support_resistance'
        )
        
        # Method 4: Fibonacci projection targets
        fib_tp1, fib_tp2 = self.levels_calc.calculate_take_profits(
            current_price, stop_loss, signal, atr_value,
            resistance_level, support_level, 'fibonacci_projection'
        )
        
        # Choose targets based on confluence and conservatism
        tp1_candidates = [rr_tp1, atr_tp1, sr_tp1, fib_tp1]
        tp2_candidates = [rr_tp2, atr_tp2, sr_tp2, fib_tp2]
        
        if original_tp1 != 0:
            tp1_candidates.append(original_tp1)
        if original_tp2 != 0:
            tp2_candidates.append(original_tp2)
        
        # For TP1, choose the most conservative (closest to current price)
        if signal.upper() in ['BUY', 'LONG']:
            final_tp1 = min(tp for tp in tp1_candidates if tp > current_price)
            final_tp2 = min(tp for tp in tp2_candidates if tp > final_tp1)
        else:
            final_tp1 = max(tp for tp in tp1_candidates if tp < current_price)
            final_tp2 = max(tp for tp in tp2_candidates if tp < final_tp1)
        
        return final_tp1, final_tp2
    
    def _calculate_enhancement_confidence(self, levels_analysis: Dict[str, Any]) -> int:
        """Calculate confidence in the enhancement based on level quality"""
        
        base_confidence = 50
        
        # Add confidence based on number of levels analyzed
        levels_count = levels_analysis.get('all_levels_count', 0)
        base_confidence += min(20, levels_count * 2)
        
        # Add confidence based on confluence zones
        confluence_count = levels_analysis.get('confluence_zones_count', 0)
        base_confidence += min(15, confluence_count * 5)
        
        # Add confidence based on level strength
        support_strength = levels_analysis.get('support_strength', 0)
        resistance_strength = levels_analysis.get('resistance_strength', 0)
        avg_strength = (support_strength + resistance_strength) / 2
        base_confidence += min(15, avg_strength)
        
        return min(100, max(0, base_confidence))
    
    def batch_enhance_patterns(self, pattern_results: List[Dict[str, Any]],
                             ohlcv_data_dict: Dict[str, List[List]]) -> List[Dict[str, Any]]:
        """Enhance multiple pattern results in batch"""
        
        enhanced_results = []
        
        for pattern_result in pattern_results:
            symbol = pattern_result.get('symbol', 'UNKNOWN')
            ohlcv_data = ohlcv_data_dict.get(symbol, [])
            
            if ohlcv_data:
                enhanced_result = self.enhance_pattern_result(pattern_result, ohlcv_data, symbol)
                enhanced_results.append(enhanced_result)
            else:
                # If no OHLCV data available, return original result
                enhanced_results.append(pattern_result)
        
        return enhanced_results
    
    def generate_enhancement_report(self, original_result: Dict[str, Any],
                                  enhanced_result: Dict[str, Any]) -> str:
        """Generate a report comparing original vs enhanced calculations"""
        
        if not enhanced_result.get('enhancement_applied', False):
            return "No enhancement applied."
        
        orig = enhanced_result.get('original_values', {})
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        PATTERN ENHANCEMENT REPORT                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Symbol: {enhanced_result.get('symbol', 'N/A'):<10} │ Pattern: {enhanced_result.get('pattern_type', 'N/A'):<20}        ║
║ Enhancement Confidence: {enhanced_result.get('enhancement_details', {}).get('enhancement_confidence', 0)}%                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                           LEVEL COMPARISON                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Support   - Original: ${orig.get('support_level', 0):<10.2f} → Enhanced: ${enhanced_result.get('support_level', 0):<10.2f} ║
║ Resistance- Original: ${orig.get('resistance_level', 0):<10.2f} → Enhanced: ${enhanced_result.get('resistance_level', 0):<10.2f} ║
║ Stop Loss - Original: ${orig.get('stop_zone', 0):<10.2f} → Enhanced: ${enhanced_result.get('stop_zone', 0):<10.2f} ║
║ TP1       - Original: ${orig.get('tp_low', 0):<10.2f} → Enhanced: ${enhanced_result.get('tp_low', 0):<10.2f} ║
║ TP2       - Original: ${orig.get('tp_high', 0):<10.2f} → Enhanced: ${enhanced_result.get('tp_high', 0):<10.2f} ║
║ R/R Ratio - Original: 1:{orig.get('rr_ratio', 0):<8.1f} → Enhanced: 1:{enhanced_result.get('rr_ratio', 0):<8.1f}   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                       ENHANCEMENT DETAILS                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Total Levels Analyzed: {enhanced_result.get('enhancement_details', {}).get('levels_analyzed', 0):<8}                            ║
║ Fibonacci Levels: {enhanced_result.get('enhancement_details', {}).get('fibonacci_levels', 0):<12}                            ║
║ Pivot Points: {enhanced_result.get('enhancement_details', {}).get('pivot_levels', 0):<16}                            ║
║ Volume Levels: {enhanced_result.get('enhancement_details', {}).get('volume_levels', 0):<15}                            ║
║ Confluence Zones: {enhanced_result.get('enhancement_details', {}).get('confluence_zones', 0):<12}                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                      RISK MANAGEMENT                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Position Size: ${enhanced_result.get('risk_management', {}).get('position_value', 0):<12.2f} ({enhanced_result.get('risk_management', {}).get('risk_percentage', 0):<5.1f}% risk)        ║
║ Units to Trade: {enhanced_result.get('risk_management', {}).get('recommended_units', 0):<12.8f}                        ║
║ Expected Return: ${enhanced_result.get('expected_returns', {}).get('expected_value', 0):<10.2f}                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        
        return report


if __name__ == "__main__":
    # Example usage
    enhancer = PatternEnhancer(account_balance=10000)
    
    # Mock pattern result
    mock_pattern_result = {
        'success': True,
        'symbol': 'BTC/USDT',
        'pattern_type': 'Triangle',
        'signal': 'BUY',
        'confidence_score': 75,
        'current_price': 50000,
        'support_level': 49000,
        'resistance_level': 51000,
        'stop_zone': 48500,
        'tp_low': 51500,
        'tp_high': 52000,
        'rr_ratio': 1.5
    }
    
    # Mock OHLCV data
    import random
    base_price = 50000
    mock_ohlcv = []
    
    for i in range(100):
        open_price = base_price + random.uniform(-100, 100)
        close_price = open_price + random.uniform(-50, 50)
        high_price = max(open_price, close_price) + random.uniform(0, 30)
        low_price = min(open_price, close_price) - random.uniform(0, 30)
        volume = random.uniform(1000, 5000)
        
        mock_ohlcv.append([i, open_price, high_price, low_price, close_price, volume])
        base_price = close_price
    
    # Enhance the pattern result
    enhanced = enhancer.enhance_pattern_result(mock_pattern_result, mock_ohlcv, 'BTC/USDT')
    
    # Generate comparison report
    report = enhancer.generate_enhancement_report(mock_pattern_result, enhanced)
    print(report)