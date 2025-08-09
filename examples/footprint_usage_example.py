#!/usr/bin/env python3
"""
Example of how to use the new footprint time window segmentation.

This shows the proper way to use create_footprint_bars() instead of 
calling FootprintStrategy.calculate() directly with all trades.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orderflow.footprint import create_footprint_bars
from data import get_data_collector
import logging
import time

logging.basicConfig(level=logging.INFO)

def create_mock_trades(num_trades: int = 100, time_span_minutes: int = 5) -> list:
    """Create mock trade data for testing."""
    current_time = int(time.time() * 1000)  # Current time in milliseconds
    time_span_ms = time_span_minutes * 60 * 1000
    
    trades = []
    for i in range(num_trades):
        # Distribute trades across the time span
        timestamp = current_time - time_span_ms + (i * time_span_ms // num_trades)
        
        # Create realistic trade data
        base_price = 50000.0
        price_variation = (i % 20 - 10) * 5  # Price moves up and down
        price = base_price + price_variation
        
        amount = 0.1 + (i % 10) * 0.05  # Variable amounts
        
        trade = {
            'timestamp': timestamp,
            'price': price,
            'amount': amount,
            'id': f'trade_{i}',
        }
        trades.append(trade)
    
    return trades

def analyze_footprint_with_time_windows():
    """
    Example showing how to create proper footprint bars with time windows.
    
    Before the fix: FootprintStrategy processed all trades as one large chunk
    After the fix: create_footprint_bars() segments trades into time windows
    """
    
    print("🦶 Footprint Analysis with Time Windows")
    print("=" * 50)
    
    # Get data collector
    collector = get_data_collector()
    symbol = "BTC/USDT"
    
    try:
        # Try to fetch real trades data, fallback to mock data if unavailable
        print(f"📡 Attempting to fetch recent trades for {symbol}...")
        try:
            trades = collector.fetch_recent_trades(symbol, limit=1000)
        except Exception as e:
            print(f"⚠️ Could not fetch real data ({e}), using mock data for demonstration")
            trades = create_mock_trades(200, 10)  # 200 trades over 10 minutes
        
        if not trades:
            print("❌ No trades data available")
            return
        
        print(f"📊 Fetched {len(trades)} trades")
        
        # OLD WAY (processes all trades as one chunk - not a true footprint)
        print(f"\n❌ OLD WAY - Single large chunk:")
        from orderflow.footprint import FootprintStrategy
        old_strategy = FootprintStrategy(
            symbol=symbol,
            timeframe="1m",
            limit=len(trades),
            ob=None,
            ohlcv=None,
            trades=trades
        )
        old_result = old_strategy.calculate()
        print(f"   Result: {len(old_result.get('footprint_data', {}))} price levels from ALL trades")
        
        # NEW WAY (proper footprint bars with time windows)
        print(f"\n✅ NEW WAY - Time window segmentation:")
        
        # Create 1-minute footprint bars
        footprint_bars_1m = create_footprint_bars(
            symbol=symbol,
            timeframe="1m", 
            trades=trades,
            window_seconds=60,  # 1-minute bars
            limit=10
        )
        
        print(f"   Created {len(footprint_bars_1m)} 1-minute footprint bars")
        
        # Create 30-second footprint bars
        footprint_bars_30s = create_footprint_bars(
            symbol=symbol,
            timeframe="30s",
            trades=trades, 
            window_seconds=30,  # 30-second bars
            limit=20
        )
        
        print(f"   Created {len(footprint_bars_30s)} 30-second footprint bars")
        
        # Analyze the results
        print(f"\n📈 Analysis Results:")
        
        if footprint_bars_1m:
            print(f"   1-minute bars:")
            for i, bar in enumerate(footprint_bars_1m[:3]):  # Show first 3
                summary = bar.get('summary', {})
                window_meta = bar.get('window_metadata', {})
                print(f"     Bar {i+1}: {window_meta.get('trades_in_window', 0)} trades, "
                      f"Volume: {summary.get('total_volume', 0):.2f}, "
                      f"Delta: {summary.get('cumulative_delta', 0):.2f}")
        
        if footprint_bars_30s:
            print(f"   30-second bars:")
            for i, bar in enumerate(footprint_bars_30s[:3]):  # Show first 3  
                summary = bar.get('summary', {})
                window_meta = bar.get('window_metadata', {})
                print(f"     Bar {i+1}: {window_meta.get('trades_in_window', 0)} trades, "
                      f"Volume: {summary.get('total_volume', 0):.2f}, "
                      f"Delta: {summary.get('cumulative_delta', 0):.2f}")
        
        # Show the power of time segmentation
        print(f"\n🎯 Key Benefits:")
        print(f"   ✅ Each bar represents actual time window (e.g., 1-minute candle period)")
        print(f"   ✅ Can track order flow changes over time")
        print(f"   ✅ Enables proper footprint chart visualization")
        print(f"   ✅ Better for real-time trading applications")
        
        return footprint_bars_1m
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def advanced_footprint_analysis():
    """
    Advanced example showing different time window configurations.
    """
    print(f"\n🔬 Advanced Time Window Analysis")
    print("=" * 30)
    
    # Mock trades for consistent testing
    import time
    current_time = int(time.time() * 1000)
    mock_trades = []
    
    for i in range(200):
        # 10 minutes of trades
        timestamp = current_time - (10 * 60 * 1000) + (i * 3000)  # Every 3 seconds
        mock_trades.append({
            'timestamp': timestamp,
            'price': 50000 + (i % 40 - 20),  # Price oscillation
            'amount': 0.1 + (i % 5) * 0.1,
            'id': f'mock_{i}'
        })
    
    # Different time window configurations
    configurations = [
        (15, "15s", "Scalping"),
        (30, "30s", "Short-term"), 
        (60, "1m", "Standard"),
        (300, "5m", "Swing")
    ]
    
    for window_sec, label, description in configurations:
        bars = create_footprint_bars(
            symbol="BTC/USDT",
            timeframe=label,
            trades=mock_trades,
            window_seconds=window_sec,
            limit=5
        )
        
        avg_trades_per_bar = sum(
            bar['window_metadata']['trades_in_window'] 
            for bar in bars
        ) / len(bars) if bars else 0
        
        print(f"   {label} ({description}): {len(bars)} bars, "
              f"avg {avg_trades_per_bar:.1f} trades/bar")

if __name__ == "__main__":
    # Run the main example
    result = analyze_footprint_with_time_windows()
    
    # Run advanced analysis
    advanced_footprint_analysis()
    
    print(f"\n✅ Example completed!")