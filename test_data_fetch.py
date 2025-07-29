#!/usr/bin/env python3
"""
Simple script to test data fetching and export to CSV.
Tests the DataCollector class and exports OHLCV data to CSV files.
"""

import os
import pandas as pd
from datetime import datetime
from data import get_data_collector

def main():
    """Test data fetching and export to CSV."""
    print("Testing cryptocurrency data collection...")
    
    # Initialize data collector
    try:
        collector = get_data_collector()
        print(f"✓ Connected to {collector.exchange_name} exchange")
    except Exception as e:
        print(f"✗ Failed to initialize collector: {e}")
        return
    
    # Test symbols (subset for quick testing)
    test_symbols = ['XRP/USDT']
    timeframe = '3d'
    limit = 30  # 30 candles in 4 hour timeframe
    
    print(f"\nFetching {timeframe} data for {len(test_symbols)} symbols...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data/csv_exports', exist_ok=True)
    
    for symbol in test_symbols:
        try:
            print(f"\nProcessing {symbol}...")
            
            # Fetch OHLCV data
            ohlcv_data = collector.fetch_ohlcv_data(symbol, timeframe, limit)
            
            if not ohlcv_data:
                print(f"✗ No data retrieved for {symbol}")
                continue
            
            print(f"✓ Retrieved {len(ohlcv_data)} candles for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convert timestamp to readable datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Reorder columns
            df = df[['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Generate filename
            safe_symbol = symbol.replace('/', '_')
            filename = f"data/csv_exports/{safe_symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
            
            # Export to CSV
            df.to_csv(filename, index=False)
            print(f"✓ Exported to {filename}")
            
            # Show sample data
            print(f"Sample data (first 3 rows):")
            print(df.head(3).to_string(index=False))
            
        except Exception as e:
            print(f"✗ Error processing {symbol}: {e}")
    
    print(f"\n✓ Data collection test completed!")
    print(f"CSV files saved in: data/csv_exports/")

if __name__ == "__main__":
    main()