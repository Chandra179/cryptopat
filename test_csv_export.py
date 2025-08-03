#!/usr/bin/env python3
"""
Test script for CSV export functionality.
Exports OHLCV data for various symbols and timeframes to CSV files.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.collector import DataCollector

def main():
    """Test CSV export functionality."""
    try:
        # Initialize data collector
        collector = DataCollector()
        
        # Test cases - different symbols and timeframes
        test_cases = [
            ('BTC/USDT', '1d', 100),
            ('ETH/USDT', '4h', 200),
            ('SOL/USDT', '1h', 50),
        ]
        
        print("Starting OHLCV CSV export tests...")
        print("=" * 50)
        
        for symbol, timeframe, limit in test_cases:
            print(f"\nExporting {symbol} {timeframe} data ({limit} candles)...")
            
            try:
                csv_path = collector.export_ohlcv_to_csv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit
                )
                print(f"✅ Success: {csv_path}")
                
                # Check if file exists and get size
                if os.path.exists(csv_path):
                    file_size = os.path.getsize(csv_path)
                    print(f"   File size: {file_size:,} bytes")
                    
                    # Show first few lines
                    with open(csv_path, 'r') as f:
                        lines = f.readlines()[:3]  # Header + 2 data rows
                        print(f"   Preview:")
                        for line in lines:
                            print(f"   {line.strip()}")
                else:
                    print(f"❌ File not found: {csv_path}")
                    
            except Exception as e:
                print(f"❌ Error exporting {symbol}: {e}")
        
        print("\n" + "=" * 50)
        print("CSV export test completed!")
        
        # List all created files
        csv_dir = 'data/csv_exports'
        if os.path.exists(csv_dir):
            print(f"\nFiles in {csv_dir}:")
            for file in os.listdir(csv_dir):
                if file.endswith('.csv'):
                    filepath = os.path.join(csv_dir, file)
                    size = os.path.getsize(filepath)
                    print(f"  {file} ({size:,} bytes)")
                    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())