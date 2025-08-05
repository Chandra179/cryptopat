#!/usr/bin/env python3

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from data import get_data_collector

class ParabolicSARStrategy:
    def __init__(self, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.20):
        self.af_start = af_start
        self.af_increment = af_increment
        self.af_max = af_max
        self.data_collector = get_data_collector()

    def calculate_parabolic_sar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Parabolic SAR values"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        length = len(df)
        sar = np.zeros(length)
        trend = np.zeros(length)  # 1 for uptrend, -1 for downtrend
        af = np.zeros(length)
        ep = np.zeros(length)  # Extreme Point
        
        # Initialize first values
        sar[0] = low[0]
        trend[0] = 1
        af[0] = self.af_start
        ep[0] = high[0]
        
        for i in range(1, length):
            # Previous values
            prev_sar = sar[i-1]
            prev_trend = trend[i-1]
            prev_af = af[i-1]
            prev_ep = ep[i-1]
            
            # Calculate SAR
            if prev_trend == 1:  # Uptrend
                sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
                
                # SAR should not be above previous two lows
                if i >= 2:
                    sar[i] = min(sar[i], low[i-1], low[i-2])
                else:
                    sar[i] = min(sar[i], low[i-1])
                
                # Check for trend reversal
                if low[i] <= sar[i]:
                    # Trend reversal to downtrend
                    trend[i] = -1
                    sar[i] = prev_ep
                    ep[i] = low[i]
                    af[i] = self.af_start
                else:
                    # Continue uptrend
                    trend[i] = 1
                    if high[i] > prev_ep:
                        ep[i] = high[i]
                        af[i] = min(prev_af + self.af_increment, self.af_max)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af
                        
            else:  # Downtrend
                sar[i] = prev_sar - prev_af * (prev_sar - prev_ep)
                
                # SAR should not be below previous two highs
                if i >= 2:
                    sar[i] = max(sar[i], high[i-1], high[i-2])
                else:
                    sar[i] = max(sar[i], high[i-1])
                
                # Check for trend reversal
                if high[i] >= sar[i]:
                    # Trend reversal to uptrend
                    trend[i] = 1
                    sar[i] = prev_ep
                    ep[i] = high[i]
                    af[i] = self.af_start
                else:
                    # Continue downtrend
                    trend[i] = -1
                    if low[i] < prev_ep:
                        ep[i] = low[i]
                        af[i] = min(prev_af + self.af_increment, self.af_max)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af
        
        df = df.copy()
        df['sar'] = sar
        df['sar_trend'] = trend
        df['sar_af'] = af
        df['sar_ep'] = ep
        
        return df

    def analyze_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Parabolic SAR signals"""
        if len(df) < 10:
            return {"error": "Insufficient data for analysis"}
        
        current_sar = df['sar'].iloc[-1]
        current_price = df['close'].iloc[-1]
        current_trend = df['sar_trend'].iloc[-1]
        current_af = df['sar_af'].iloc[-1]
        
        # Find recent trend changes
        trend_changes = []
        for i in range(1, len(df)):
            if df['sar_trend'].iloc[i] != df['sar_trend'].iloc[i-1]:
                trend_changes.append({
                    'index': i,
                    'date': df.index[i],
                    'price': df['close'].iloc[i],
                    'from_trend': 'up' if df['sar_trend'].iloc[i-1] == 1 else 'down',
                    'to_trend': 'up' if df['sar_trend'].iloc[i] == 1 else 'down'
                })
        
        # Get last 5 trend changes
        recent_changes = trend_changes[-5:] if len(trend_changes) >= 5 else trend_changes
        
        # Calculate distance from SAR
        sar_distance = abs(current_price - current_sar) / current_price * 100
        
        # Determine signal strength based on acceleration factor
        if current_af >= self.af_max * 0.8:
            strength = "Strong"
        elif current_af >= self.af_max * 0.5:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        # Generate signal
        if current_trend == 1:
            signal = "BULLISH"
            position = "below"
        else:
            signal = "BEARISH"
            position = "above"
        
        return {
            "success": True,
            "signal": signal,
            "strength": strength,
            "current_sar": current_sar,
            "current_price": current_price,
            "sar_distance_pct": sar_distance,
            "position": position,
            "acceleration_factor": current_af,
            "trend_changes": recent_changes,
            "interpretation": self._get_interpretation(signal, strength, sar_distance, current_af)
        }

    def _get_interpretation(self, signal: str, strength: str, distance: float, af: float) -> str:
        """Generate interpretation of the Parabolic SAR signal"""
        base_msg = f"{signal.lower()} trend with {strength.lower()} momentum"
        
        if distance < 1:
            proximity = "very close to SAR - potential reversal zone"
        elif distance < 3:
            proximity = "close to SAR - watch for reversal"
        elif distance < 5:
            proximity = "moderate distance from SAR"
        else:
            proximity = "far from SAR - strong trend continuation"
        
        if af >= self.af_max * 0.8:
            momentum = "High acceleration factor indicates strong trending market"
        elif af <= self.af_start * 2:
            momentum = "Low acceleration factor suggests early trend or consolidation"
        else:
            momentum = "Moderate acceleration factor"
        
        return f"{base_msg.capitalize()}. {proximity.capitalize()}. {momentum}."

    def analyze(self, symbol: str, timeframe: str, limit: int = 100, ohlcv_data: list = None) -> Dict[str, Any]:
        """Main analysis function"""
        try:
            # Use provided data or fetch new data
            if ohlcv_data is None:
                ohlcv_data = self.data_collector.fetch_ohlcv_data(symbol, timeframe, limit)
            
            if not ohlcv_data:
                return {"success": False, "error": "No data available"}
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            if df.empty:
                return {"success": False, "error": "No data available"}
            
            # Calculate Parabolic SAR
            df_with_sar = self.calculate_parabolic_sar(df)
            
            # Analyze signals
            analysis = self.analyze_signals(df_with_sar)
            
            if not analysis.get("success", False):
                return analysis
            
            # Add metadata
            analysis.update({
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": df.index[-1],
                "data_points": len(df),
                "af_settings": {
                    "start": self.af_start,
                    "increment": self.af_increment,
                    "max": self.af_max
                }
            })
            
            return analysis
            
        except Exception as e:
            return {"success": False, "error": f"Analysis failed: {str(e)}"}

def main():
    """CLI interface for Parabolic SAR analysis"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python parabolicsar.py <symbol> [timeframe] [limit]")
        print("Example: python parabolicsar.py BTC/USDT 1h 100")
        return
    
    symbol = sys.argv[1]
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "1h"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    strategy = ParabolicSARStrategy()
    result = strategy.analyze(symbol, timeframe, limit)
    
    if result.get("success"):
        print(f"\n=== Parabolic SAR Analysis for {symbol} ({timeframe}) ===")
        print(f"Signal: {result['signal']} ({result['strength']})")
        print(f"Current Price: ${result['current_price']:.4f}")
        print(f"Current SAR: ${result['current_sar']:.4f} ({result['position']} price)")
        print(f"Distance from SAR: {result['sar_distance_pct']:.2f}%")
        print(f"Acceleration Factor: {result['acceleration_factor']:.3f}")
        print(f"\nInterpretation: {result['interpretation']}")
        
        if result['trend_changes']:
            print(f"\nRecent Trend Changes ({len(result['trend_changes'])}):")
            for change in result['trend_changes'][-3:]:
                print(f"  {change['date']}: {change['from_trend']} → {change['to_trend']} at ${change['price']:.4f}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()