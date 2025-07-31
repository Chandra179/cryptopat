"""
CLI handler for Order Flow Imbalance Analysis.
"""

import asyncio
from typing import Dict
from orderflow.imbalance import run_imbalance_analysis, format_imbalance_results

async def handle_imbalance_analysis(symbol: str = "XRP/USDT", duration: int = 30):
    """
    Handle order flow imbalance analysis command.
    
    Args:
        symbol: Trading pair symbol (default: XRP/USDT)
        duration: Analysis duration in seconds (default: 30)
    """
    print(f"\n🔄 Starting Order Flow Imbalance Analysis for {symbol}")
    print(f"Duration: {duration} seconds")
    print("This will analyze real-time order flow patterns...\n")
    
    try:
        # Run the analysis
        results = await run_imbalance_analysis(symbol, duration)
        
        # Format and display results
        formatted_output = format_imbalance_results(results)
        print(formatted_output)
        
        # Export to CSV if there are results
        if 'timeframe_results' in results and results['timeframe_results']:
            await export_imbalance_to_csv(results)
        
    except Exception as e:
        print(f"❌ Error during imbalance analysis: {e}")

async def export_imbalance_to_csv(results: Dict):
    """Export imbalance analysis results to CSV files."""
    import csv
    import os
    from datetime import datetime
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol_clean = results['symbol'].replace('/', '_')
    
    # Ensure csv_exports directory exists
    csv_dir = "data/csv_exports"
    os.makedirs(csv_dir, exist_ok=True)
    
    # Export snapshots for each timeframe
    for timeframe, data in results.get('timeframe_results', {}).items():
        latest_snapshot = data.get('latest_snapshot')
        if latest_snapshot:
            filename = f"{csv_dir}/imbalance_analysis_{symbol_clean}_{timestamp}_{timeframe}_snapshot.csv"
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'timestamp', 'symbol', 'timeframe_ms', 'buy_volume', 'sell_volume', 
                    'total_volume', 'basic_imbalance', 'volume_weighted_imbalance',
                    'statistical_significance', 'aggressive_ratio', 'mid_price'
                ])
                writer.writerow([
                    latest_snapshot.timestamp, latest_snapshot.symbol, latest_snapshot.timeframe_ms,
                    latest_snapshot.buy_volume, latest_snapshot.sell_volume, latest_snapshot.total_volume,
                    latest_snapshot.basic_imbalance, latest_snapshot.volume_weighted_imbalance,
                    latest_snapshot.statistical_significance, latest_snapshot.aggressive_ratio,
                    latest_snapshot.mid_price
                ])
        
        # Export signals
        recent_signals = data.get('recent_signals', [])
        if recent_signals:
            filename = f"{csv_dir}/imbalance_analysis_{symbol_clean}_{timestamp}_{timeframe}_signals.csv"
            
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'timestamp', 'symbol', 'timeframe_ms', 'signal_type', 'direction',
                    'strength', 'confidence', 'imbalance_value', 'z_score'
                ])
                for signal in recent_signals:
                    writer.writerow([
                        signal.timestamp, signal.symbol, signal.timeframe_ms,
                        signal.signal_type, signal.direction, signal.strength,
                        signal.confidence, signal.imbalance_value, signal.z_score
                    ])
    
    print(f"\n📊 Analysis results exported to CSV files in {csv_dir}/")

if __name__ == "__main__":
    import sys
    
    symbol = sys.argv[1] if len(sys.argv) > 1 else "XRP/USDT"
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    asyncio.run(handle_imbalance_analysis(symbol, duration))