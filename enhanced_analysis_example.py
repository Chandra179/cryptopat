#!/usr/bin/env python3
"""
Example script showing how to use enhanced support/resistance, stop loss, and target calculations
This demonstrates integration with existing pattern analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import get_data_collector
from techin.enhanced_levels import EnhancedLevelsCalculator
from techin.risk_manager import RiskManager, TradeSetup
from techin.pattern_enhancer import PatternEnhancer

# Import some existing pattern analyzers for demonstration
try:
    from pattern.triangle import TriangleStrategy
    from pattern.double_bottom import DoubleBottomStrategy
    from pattern.flag import FlagStrategy
except ImportError as e:
    print(f"Warning: Could not import pattern strategies: {e}")
    TriangleStrategy = None
    DoubleBottomStrategy = None
    FlagStrategy = None


def analyze_symbol_enhanced(symbol: str, timeframe: str, limit: int = 100, 
                          account_balance: float = 10000):
    """
    Comprehensive enhanced analysis combining pattern detection with industry-standard calculations
    """
    print(f"\n{'='*80}")
    print(f"ENHANCED ANALYSIS: {symbol} ({timeframe})")
    print(f"{'='*80}")
    
    try:
        # Get market data
        collector = get_data_collector()
        ohlcv_data = collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if not ohlcv_data:
            print(f"❌ Failed to fetch data for {symbol}")
            return
        
        print(f"✅ Fetched {len(ohlcv_data)} candles")
        
        # Initialize enhanced calculators
        levels_calc = EnhancedLevelsCalculator()
        risk_manager = RiskManager(account_balance)
        pattern_enhancer = PatternEnhancer(account_balance)
        
        # 1. Run comprehensive levels analysis
        print("\n🔍 Running comprehensive levels analysis...")
        levels_analysis = levels_calc.analyze_comprehensive_levels(ohlcv_data, symbol, account_balance)
        
        if levels_analysis['success']:
            print(f"✅ Levels Analysis Complete")
            print(f"   Signal: {levels_analysis['signal']} ({levels_analysis['bias']})")
            print(f"   Confidence: {levels_analysis['confidence_score']}%")
            print(f"   Support: ${levels_analysis['support_level']:.2f} (Strength: {levels_analysis['support_strength']})")
            print(f"   Resistance: ${levels_analysis['resistance_level']:.2f} (Strength: {levels_analysis['resistance_strength']})")
            print(f"   Total Levels: {levels_analysis['all_levels_count']}")
            print(f"   Confluence Zones: {levels_analysis['confluence_zones_count']}")
        
        # 2. Run pattern analysis (if available)
        pattern_results = {}
        
        if TriangleStrategy:
            print("\n📈 Running Triangle pattern analysis...")
            triangle = TriangleStrategy()
            triangle_result = triangle.analyze(symbol, timeframe, limit)
            if triangle_result['success']:
                enhanced_triangle = pattern_enhancer.enhance_pattern_result(triangle_result, ohlcv_data, symbol)
                pattern_results['triangle'] = enhanced_triangle
                print(f"✅ Triangle: {enhanced_triangle['signal']} ({enhanced_triangle['confidence_score']}%)")
        
        if DoubleBottomStrategy:
            print("📈 Running Double Bottom pattern analysis...")
            double_bottom = DoubleBottomStrategy()
            db_result = double_bottom.analyze(symbol, timeframe, limit)
            if db_result['success']:
                enhanced_db = pattern_enhancer.enhance_pattern_result(db_result, ohlcv_data, symbol)
                pattern_results['double_bottom'] = enhanced_db
                print(f"✅ Double Bottom: {enhanced_db['signal']} ({enhanced_db['confidence_score']}%)")
        
        if FlagStrategy:
            print("📈 Running Flag pattern analysis...")
            flag = FlagStrategy()
            flag_result = flag.analyze(symbol, timeframe, limit)
            if flag_result['success']:
                enhanced_flag = pattern_enhancer.enhance_pattern_result(flag_result, ohlcv_data, symbol)
                pattern_results['flag'] = enhanced_flag
                print(f"✅ Flag: {enhanced_flag['signal']} ({enhanced_flag['confidence_score']}%)")
        
        # 3. Display comprehensive results
        current_price = ohlcv_data[-1][4]
        print(f"\n💰 Current Price: ${current_price:.2f}")
        
        # Show best trade setup
        best_setup = None
        best_confidence = 0
        
        # Check levels analysis
        if levels_analysis['success'] and levels_analysis['confidence_score'] > best_confidence:
            best_confidence = levels_analysis['confidence_score']
            best_setup = {
                'source': 'Enhanced Levels',
                'signal': levels_analysis['signal'],
                'confidence': levels_analysis['confidence_score'],
                'entry': current_price,
                'stop': levels_analysis['stop_zone'],
                'tp1': levels_analysis['tp_low'],
                'tp2': levels_analysis['tp_high'],
                'support': levels_analysis['support_level'],
                'resistance': levels_analysis['resistance_level'],
                'rr_ratio': levels_analysis['rr_ratio']
            }
        
        # Check pattern results
        for pattern_name, pattern_result in pattern_results.items():
            if pattern_result['confidence_score'] > best_confidence:
                best_confidence = pattern_result['confidence_score']
                best_setup = {
                    'source': f'Enhanced {pattern_name.title()}',
                    'signal': pattern_result['signal'],
                    'confidence': pattern_result['confidence_score'],
                    'entry': pattern_result['current_price'],
                    'stop': pattern_result['stop_zone'],
                    'tp1': pattern_result['tp_low'],
                    'tp2': pattern_result['tp_high'],
                    'support': pattern_result['support_level'],
                    'resistance': pattern_result['resistance_level'],
                    'rr_ratio': pattern_result['rr_ratio']
                }
        
        if best_setup:
            print(f"\n🎯 BEST SETUP ({best_setup['source']}):")
            print(f"   Signal: {best_setup['signal']} (Confidence: {best_setup['confidence']}%)")
            print(f"   Entry: ${best_setup['entry']:.2f}")
            print(f"   Stop Loss: ${best_setup['stop']:.2f}")
            print(f"   TP1: ${best_setup['tp1']:.2f}")
            print(f"   TP2: ${best_setup['tp2']:.2f}")
            print(f"   Support: ${best_setup['support']:.2f}")
            print(f"   Resistance: ${best_setup['resistance']:.2f}")
            print(f"   Risk/Reward: 1:{best_setup['rr_ratio']:.1f}")
            
            # Calculate position sizing
            atr_values = levels_calc.calculate_atr(ohlcv_data)
            current_atr = atr_values[-1] if atr_values else current_price * 0.02
            
            trade_setup = TradeSetup(
                symbol=symbol,
                entry_price=best_setup['entry'],
                stop_loss=best_setup['stop'],
                take_profit_1=best_setup['tp1'],
                take_profit_2=best_setup['tp2'],
                signal=best_setup['signal'],
                confidence=best_setup['confidence'],
                atr_value=current_atr,
                timeframe=timeframe
            )
            
            # Generate risk management report
            risk_report = risk_manager.generate_risk_report(trade_setup)
            print(risk_report)
            
            # Show enhancement comparison if pattern was enhanced
            if best_setup['source'].startswith('Enhanced') and pattern_results:
                for pattern_name, enhanced_result in pattern_results.items():
                    if enhanced_result.get('enhancement_applied', False):
                        print("\n" + "="*50)
                        print("ENHANCEMENT COMPARISON")
                        print("="*50)
                        
                        orig = enhanced_result.get('original_values', {})
                        print(f"Original vs Enhanced Values:")
                        print(f"Support:    ${orig.get('support_level', 0):.2f} → ${enhanced_result['support_level']:.2f}")
                        print(f"Resistance: ${orig.get('resistance_level', 0):.2f} → ${enhanced_result['resistance_level']:.2f}")
                        print(f"Stop Loss:  ${orig.get('stop_zone', 0):.2f} → ${enhanced_result['stop_zone']:.2f}")
                        print(f"TP1:        ${orig.get('tp_low', 0):.2f} → ${enhanced_result['tp_low']:.2f}")
                        print(f"TP2:        ${orig.get('tp_high', 0):.2f} → ${enhanced_result['tp_high']:.2f}")
                        print(f"R/R Ratio:  1:{orig.get('rr_ratio', 0):.1f} → 1:{enhanced_result['rr_ratio']:.1f}")
                        
                        enhancement_details = enhanced_result.get('enhancement_details', {})
                        print(f"\nEnhancement used:")
                        print(f"- {enhancement_details.get('fibonacci_levels', 0)} Fibonacci levels")
                        print(f"- {enhancement_details.get('pivot_levels', 0)} Pivot points")
                        print(f"- {enhancement_details.get('volume_levels', 0)} Volume levels")
                        print(f"- {enhancement_details.get('confluence_zones', 0)} Confluence zones")
                        print(f"- Enhancement confidence: {enhancement_details.get('enhancement_confidence', 0)}%")
                        break
        else:
            print("\n❌ No viable setups found")
            
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run enhanced analysis examples"""
    
    print("🚀 CRYPTOPAT ENHANCED ANALYSIS SYSTEM")
    print("Industry-standard support/resistance, stop loss, and target calculations")
    print("Enhanced with Fibonacci, Pivot Points, Volume Profile, and Risk Management")
    
    # Test symbols and timeframes
    test_cases = [
        ('BTC/USDT', '1h', 100),
        ('ETH/USDT', '4h', 150),
        ('SOL/USDT', '1h', 100),
    ]
    
    account_balance = 10000  # $10,000 demo account
    
    for symbol, timeframe, limit in test_cases:
        try:
            analyze_symbol_enhanced(symbol, timeframe, limit, account_balance)
        except KeyboardInterrupt:
            print("\n⏹️  Analysis interrupted by user")
            break
        except Exception as e:
            print(f"❌ Failed to analyze {symbol}: {str(e)}")
            continue
        
        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()