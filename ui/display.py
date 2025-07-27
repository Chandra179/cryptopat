#!/usr/bin/env python3

from typing import List, Dict

def display_results(results: List[Dict], methods: List[str] = ['sma', 'ema']):
    """Display analysis results in a formatted way."""
    print("\n" + "="*80)
    methods_str = ', '.join(m.upper() for m in methods)
    print(f"CRYPTOCURRENCY TREND ANALYSIS RESULTS ({methods_str})")
    print("="*80)
    
    for result in results:
        if 'error' in result:
            print(f"\n❌ {result.get('symbol', 'Unknown')}: Error - {result['error']}")
            continue
        
        symbol = result['symbol']
        trend = result['trend'].upper()
        bullish = result['bullish_confidence']
        bearish = result['bearish_confidence']
        current_price = result['current_price']
        
        trend_emoji = "🔴" if trend == "BEARISH" else "🟢"
        
        print(f"\n{trend_emoji} {symbol}")
        print(f"   Current Price: ${current_price:,.2f}")
        print(f"   Prediction: {trend} for next {result['prediction_days']} days")
        print(f"   Combined Confidence: {bullish:.1f}% Bullish | {bearish:.1f}% Bearish")
        
        # Show special EMA crossover details if it's the only method or has special fields
        if 'pattern' in result:  # Single ema_cross or ema_500_200 method
            pattern = result.get('pattern', 'unknown')
            days_since = result.get('days_since_crossover')
            print(f"   EMA Pattern: {pattern.replace('_', ' ').title()}")
            
            # Find EMA values dynamically
            ema_keys = [k for k in result.keys() if k.startswith('ema_') and k.endswith(('_current', '50', '200', '500', '20'))]
            ema_values = []
            for key in ema_keys:
                if key in result:
                    period = key.replace('ema_', '').replace('_current', '')
                    ema_values.append((period, result[key]))
            
            if ema_values:
                ema_display = " | ".join([f"{period} EMA: ${value:,.2f}" for period, value in sorted(ema_values)])
                print(f"   {ema_display}")
            
            if days_since is not None:
                print(f"   Days Since Crossover: {days_since}")
            print(f"   Crossover Strength: {result.get('crossover_strength', 0):.1f}%")
        
        # Show individual analysis results if multiple methods were used
        if len(methods) > 1:
            for method in methods:
                method_key = f'{method}_analysis'
                if method_key in result:
                    method_result = result[method_key]
                    if method == 'ema_cross':
                        pattern = method_result.get('pattern', 'unknown')
                        method_display = method.replace('_', ' ').upper()
                        print(f"   {method_display} Analysis: {method_result['bullish_confidence']:.1f}% Bullish | {method_result['bearish_confidence']:.1f}% Bearish ({method_result['trend'].upper()}) - Pattern: {pattern.replace('_', ' ').title()}")
                    else:
                        print(f"   {method.upper()} Analysis: {method_result['bullish_confidence']:.1f}% Bullish | {method_result['bearish_confidence']:.1f}% Bearish ({method_result['trend'].upper()})")
        
        print(f"   Analysis based on {result['analysis_days']} days ({result['data_points']} data points)")