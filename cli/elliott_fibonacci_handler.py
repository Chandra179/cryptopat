from trend.elliott_fibonacci import ElliottFibonacciAnalyzer, parse_command


def handle_elliott_fibonacci_command(command: str):
    """Handle Elliott Wave + Fibonacci analysis command"""
    try:
        symbol, timeframe, limit, zigzag_threshold = parse_command(command)
        
        analyzer = ElliottFibonacciAnalyzer()
        result = analyzer.analyze(symbol, timeframe, limit, zigzag_threshold)
        
        if "error" in result:
            print(f"❌ {result['error']}")
            return
        
        _format_elliott_output(result)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Usage: elliott_fibonacci s=SYMBOL t=TIMEFRAME l=LIMIT zz=THRESHOLD")
        print("Example: elliott_fibonacci s=XRP/USDT t=4h l=150 zz=4")


def _format_elliott_output(result):
    """Format Elliott Wave analysis output"""
    print("\n" + "="*60)
    print("[ELLIOTT + FIBONACCI STRUCTURE]")
    print("="*60)
    
    print(f"Symbol: {result['symbol']}")
    print(f"Timeframe: {result['timeframe']}")
    print(f"Pattern: {result['pattern_type']} Wave")
    print(f"Confluence: {result['confluence_strength']} ({result['confluence_score']})")
    
    print("\n📊 WAVE STRUCTURE:")
    print("-" * 50)
    
    for wave_data in result['waves']:
        wave = wave_data['wave']
        start = wave_data['start_price']
        end = wave_data['end_price']
        length = wave_data['length']
        confidence = wave_data['confidence']
        
        direction = "↗️" if end > start else "↘️"
        print(f"Wave {wave}: {start:.4f} → {end:.4f} {direction} (Length: {length:.4f}, Confidence: {confidence})")
    
    if result.get('current_wave'):
        print(f"🎯 Current Status: Wave {result['current_wave']} in progress")
    
    if result.get('next_targets'):
        print("🎯 FIBONACCI TARGETS:")
        print("-" * 30)
        for target_name, target_price in result['next_targets'].items():
            ratio = target_name.split('_')[-1] if '_' in target_name else 'N/A'
            wave = target_name.split('_')[1] if '_' in target_name else 'Next'
            print(f"{wave} ({ratio}): {target_price:.4f}")
    
    _print_elliott_rules_summary(result)


def _print_elliott_rules_summary(result):
    """Print Elliott Wave rules validation summary"""
    print("\n📋 ELLIOTT WAVE RULES CHECK:")
    print("-" * 35)
    
    if result['pattern_type'].lower() == 'impulse':
        waves = result['waves']
        if len(waves) >= 5:
            wave1_len = waves[0]['length']
            wave3_len = waves[2]['length']
            wave5_len = waves[4]['length']
            
            # Rule 1: Wave 3 is not the shortest
            rule1_ok = not (wave3_len < wave1_len and wave3_len < wave5_len)
            print(f"✅ Rule 1: Wave 3 not shortest" if rule1_ok else f"❌ Rule 1: Wave 3 is shortest")
            
            # Wave 2 retracement analysis
            if len(waves) >= 2:
                wave1_start = waves[0]['start_price']
                wave1_end = waves[0]['end_price']
                wave2_end = waves[1]['end_price']
                
                retrace_pct = abs(wave2_end - wave1_start) / abs(wave1_end - wave1_start)
                print(f"📏 Wave 2 retracement: {retrace_pct:.1%}")
                
                if 0.5 <= retrace_pct <= 0.786:
                    print("✅ Wave 2 in valid range (50%-78.6%)")
                else:
                    print("⚠️  Wave 2 outside typical range")
            
            # Wave 3 extension analysis
            if len(waves) >= 3:
                wave3_ratio = wave3_len / wave1_len
                print(f"📏 Wave 3/Wave 1 ratio: {wave3_ratio:.2f}")
                
                if 1.4 <= wave3_ratio <= 1.8:
                    print("✅ Wave 3 in extension range (1.4-1.8x)")
                else:
                    print("⚠️  Wave 3 outside typical extension")
    
    elif result['pattern_type'].lower() == 'corrective':
        _print_corrective_rules_summary(result)
    
    print("\n" + "="*60)


def _print_corrective_rules_summary(result):
    """Print A-B-C corrective wave rules validation"""
    if 'corrective_rules' not in result:
        print("⚠️  No corrective rules validation available")
        return
        
    rules = result['corrective_rules']
    waves = result['waves']
    
    if len(waves) >= 3:
        wave_a_len = waves[0]['length']
        wave_b_len = waves[1]['length']
        wave_c_len = waves[2]['length']
        
        # Wave B retracement analysis
        wave_b_retrace = wave_b_len / wave_a_len
        print(f"📏 Wave B retracement: {wave_b_retrace:.1%} of Wave A")
        
        if rules.get('wave_b_valid_retrace', False):
            print("✅ Wave B in valid range (38.2%-78.6%)")
        else:
            print("⚠️  Wave B outside typical range")
        
        # Wave C analysis
        wave_c_ratio = wave_c_len / wave_a_len
        print(f"📏 Wave C/Wave A ratio: {wave_c_ratio:.2f}")
        
        if rules.get('wave_c_adequate_length', False):
            print("✅ Wave C adequate length (≥61.8% of A)")
        else:
            print("❌ Wave C too short")
            
        if rules.get('wave_c_fibonacci_ratio', False):
            print("✅ Wave C at Fibonacci ratio (1.0x or 1.618x)")
        else:
            print("⚠️  Wave C not at typical Fibonacci ratio")
            
        if rules.get('wave_c_correct_direction', False):
            print("✅ Wave C moves in same direction as Wave A")
        else:
            print("❌ Wave C direction incorrect")