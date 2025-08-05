#!/usr/bin/env python3

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from data import get_data_collector

class DonchianChannelStrategy:
    def __init__(self, period: int = 20):
        self.period = period
        self.data_collector = get_data_collector()

    def calculate_donchian_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Donchian Channel values"""
        if len(df) < self.period:
            raise ValueError(f"Insufficient data. Need at least {self.period} periods")
        
        # Calculate rolling highest high and lowest low
        df = df.copy()
        df['dc_upper'] = df['high'].rolling(window=self.period).max()
        df['dc_lower'] = df['low'].rolling(window=self.period).min()
        df['dc_middle'] = (df['dc_upper'] + df['dc_lower']) / 2
        
        # Calculate channel width and position
        df['dc_width'] = df['dc_upper'] - df['dc_lower']
        df['dc_width_pct'] = (df['dc_width'] / df['close']) * 100
        
        # Calculate position within channel (0 = at lower, 1 = at upper)
        df['dc_position'] = (df['close'] - df['dc_lower']) / df['dc_width']
        df['dc_position'] = df['dc_position'].clip(0, 1)  # Clamp between 0 and 1
        
        return df

    def analyze_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Donchian Channel signals"""
        if len(df) < self.period + 5:
            return {"error": "Insufficient data for analysis"}
        
        current_price = df['close'].iloc[-1]
        current_upper = df['dc_upper'].iloc[-1]
        current_lower = df['dc_lower'].iloc[-1]
        current_middle = df['dc_middle'].iloc[-1]
        current_position = df['dc_position'].iloc[-1]
        current_width_pct = df['dc_width_pct'].iloc[-1]
        
        # Previous values for trend analysis
        prev_upper = df['dc_upper'].iloc[-2]
        prev_lower = df['dc_lower'].iloc[-2]
        prev_position = df['dc_position'].iloc[-2]
        
        # Channel trend analysis
        upper_trend = "expanding" if current_upper > prev_upper else "contracting" if current_upper < prev_upper else "flat"
        lower_trend = "expanding" if current_lower < prev_lower else "contracting" if current_lower > prev_lower else "flat"
        
        # Breakouts detection
        breakout_signals = []
        
        # Check for upper breakout
        if current_price >= current_upper:
            breakout_signals.append("UPPER_BREAKOUT")
        
        # Check for lower breakout
        if current_price <= current_lower:
            breakout_signals.append("LOWER_BREAKOUT")
        
        # Position analysis
        if current_position >= 0.8:
            position_signal = "NEAR_UPPER"
            position_desc = "near upper channel"
        elif current_position <= 0.2:
            position_signal = "NEAR_LOWER" 
            position_desc = "near lower channel"
        elif 0.4 <= current_position <= 0.6:
            position_signal = "MIDDLE"
            position_desc = "in middle channel"
        else:
            position_signal = "NEUTRAL"
            position_desc = "in neutral zone"
        
        # Channel width analysis (volatility)
        avg_width = df['dc_width_pct'].tail(10).mean()
        if current_width_pct > avg_width * 1.2:
            volatility = "HIGH"
            volatility_desc = "high volatility"
        elif current_width_pct < avg_width * 0.8:
            volatility = "LOW"
            volatility_desc = "low volatility"
        else:
            volatility = "NORMAL"
            volatility_desc = "normal volatility"
        
        # Generate primary signal
        if breakout_signals:
            if "UPPER_BREAKOUT" in breakout_signals:
                signal = "BULLISH_BREAKOUT"
                strength = "Strong"
            else:
                signal = "BEARISH_BREAKOUT"
                strength = "Strong"
        elif position_signal == "NEAR_UPPER":
            signal = "BULLISH"
            strength = "Moderate"
        elif position_signal == "NEAR_LOWER":
            signal = "BEARISH"
            strength = "Moderate"
        else:
            signal = "NEUTRAL"
            strength = "Weak"
        
        # Calculate support and resistance levels
        recent_highs = df['dc_upper'].tail(5)
        recent_lows = df['dc_lower'].tail(5)
        
        return {
            "success": True,
            "signal": signal,
            "strength": strength,
            "current_price": current_price,
            "upper_channel": current_upper,
            "lower_channel": current_lower,
            "middle_channel": current_middle,
            "position_in_channel": current_position,
            "position_signal": position_signal,
            "position_description": position_desc,
            "channel_width_pct": current_width_pct,
            "volatility": volatility,
            "volatility_description": volatility_desc,
            "breakout_signals": breakout_signals,
            "channel_trends": {
                "upper": upper_trend,
                "lower": lower_trend
            },
            "support_level": current_lower,
            "resistance_level": current_upper,
            "interpretation": self._get_interpretation(signal, strength, position_desc, volatility_desc, breakout_signals)
        }

    def _get_interpretation(self, signal: str, strength: str, position: str, volatility: str, breakouts: list) -> str:
        """Generate interpretation of the Donchian Channel signal"""
        if breakouts:
            if "UPPER_BREAKOUT" in breakouts:
                return f"Strong bullish breakout above upper channel. Price momentum is very strong upward. Consider bullish positions with stop below middle channel."
            else:
                return f"Strong bearish breakout below lower channel. Price momentum is very strong downward. Consider bearish positions with stop above middle channel."
        
        base_msg = f"{signal.lower()} signal with {strength.lower()} conviction"
        
        if "near upper" in position:
            position_msg = "Price is testing resistance at upper channel. Watch for breakout or rejection."
        elif "near lower" in position:
            position_msg = "Price is testing support at lower channel. Watch for breakdown or bounce."
        elif "middle" in position:
            position_msg = "Price is in equilibrium zone. Wait for directional move toward channels."
        else:
            position_msg = "Price is in neutral territory within the channel."
        
        volatility_msg = f"Market showing {volatility} indicating {'potential major moves ahead' if volatility == 'low volatility' else 'active price action'}"
        
        return f"{base_msg.capitalize()}. {position_msg} {volatility_msg}."

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
            
            # Calculate Donchian Channels
            df_with_dc = self.calculate_donchian_channels(df)
            
            # Analyze signals
            analysis = self.analyze_signals(df_with_dc)
            
            if not analysis.get("success", False):
                return analysis
            
            # Add metadata
            analysis.update({
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": df.index[-1],
                "data_points": len(df),
                "period": self.period
            })
            
            return analysis
            
        except Exception as e:
            return {"success": False, "error": f"Analysis failed: {str(e)}"}

def main():
    """CLI interface for Donchian Channel analysis"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python donchain.py <symbol> [timeframe] [limit] [period]")
        print("Example: python donchain.py BTC/USDT 1h 100 20")
        return
    
    symbol = sys.argv[1]
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "1h"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    period = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    
    strategy = DonchianChannelStrategy(period=period)
    result = strategy.analyze(symbol, timeframe, limit)
    
    if result.get("success"):
        print(f"\n=== Donchian Channel Analysis for {symbol} ({timeframe}) ===")
        print(f"Period: {result['period']}")
        print(f"Signal: {result['signal']} ({result['strength']})")
        print(f"Current Price: ${result['current_price']:.4f}")
        print(f"Upper Channel: ${result['upper_channel']:.4f}")
        print(f"Lower Channel: ${result['lower_channel']:.4f}")
        print(f"Middle Channel: ${result['middle_channel']:.4f}")
        print(f"Position in Channel: {result['position_in_channel']:.1%} ({result['position_description']})")
        print(f"Channel Width: {result['channel_width_pct']:.2f}% ({result['volatility_description']})")
        
        if result['breakout_signals']:
            print(f"Breakout Signals: {', '.join(result['breakout_signals'])}")
        
        print(f"Support Level: ${result['support_level']:.4f}")
        print(f"Resistance Level: ${result['resistance_level']:.4f}")
        print(f"\nInterpretation: {result['interpretation']}")
        
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()