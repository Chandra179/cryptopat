#!/usr/bin/env python3

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_data_collector


class IchimokuAnalysis:
    def __init__(self):
        self.data_collector = get_data_collector()
        
    def calculate_ichimoku(self, df: pd.DataFrame, 
                          tenkan_period: int = 9,
                          kijun_period: int = 26, 
                          senkou_b_period: int = 52,
                          displacement: int = 26) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud components
        
        Args:
            df: DataFrame with OHLCV data
            tenkan_period: Period for Tenkan-sen (default: 9)
            kijun_period: Period for Kijun-sen (default: 26)
            senkou_b_period: Period for Senkou Span B (default: 52)
            displacement: Future displacement for spans (default: 26)
            
        Returns:
            DataFrame with Ichimoku components added
        """
        df = df.copy()
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        tenkan_high = df['high'].rolling(window=tenkan_period).max()
        tenkan_low = df['low'].rolling(window=tenkan_period).min()
        df['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        kijun_high = df['high'].rolling(window=kijun_period).max()
        kijun_low = df['low'].rolling(window=kijun_period).min()
        df['kijun_sen'] = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted forward
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted forward
        senkou_b_high = df['high'].rolling(window=senkou_b_period).max()
        senkou_b_low = df['low'].rolling(window=senkou_b_period).min()
        df['senkou_span_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span): Current close shifted backward
        df['chikou_span'] = df['close'].shift(-displacement)
        
        # Cloud color (1 for bullish/green, -1 for bearish/red)
        df['cloud_color'] = np.where(df['senkou_span_a'] > df['senkou_span_b'], 1, -1)
        
        # Price position relative to cloud
        df['price_vs_cloud'] = np.where(
            df['close'] > df[['senkou_span_a', 'senkou_span_b']].max(axis=1), 1,  # Above cloud
            np.where(df['close'] < df[['senkou_span_a', 'senkou_span_b']].min(axis=1), -1, 0)  # Below cloud / In cloud
        )
        
        return df
    
    def get_ichimoku_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate Ichimoku trading signals
        
        Args:
            df: DataFrame with Ichimoku components calculated
            
        Returns:
            Dictionary with signal analysis
        """
        if len(df) < 2:
            return {"error": "Insufficient data for signal analysis"}
            
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        signals = {
            "tenkan_kijun_cross": None,
            "price_cloud_breakout": None,
            "chikou_confirmation": None,
            "overall_trend": "neutral",
            "cloud_trend": "neutral",
            "price_position": "in_cloud"
        }
        
        # Tenkan-Kijun Cross
        if not pd.isna(current['tenkan_sen']) and not pd.isna(current['kijun_sen']):
            if not pd.isna(previous['tenkan_sen']) and not pd.isna(previous['kijun_sen']):
                if previous['tenkan_sen'] <= previous['kijun_sen'] and current['tenkan_sen'] > current['kijun_sen']:
                    signals["tenkan_kijun_cross"] = "bullish"
                elif previous['tenkan_sen'] >= previous['kijun_sen'] and current['tenkan_sen'] < current['kijun_sen']:
                    signals["tenkan_kijun_cross"] = "bearish"
        
        # Price vs Cloud position
        if current['price_vs_cloud'] == 1:
            signals["price_position"] = "above_cloud"
        elif current['price_vs_cloud'] == -1:
            signals["price_position"] = "below_cloud"
        else:
            signals["price_position"] = "in_cloud"
            
        # Price-Cloud breakout
        if previous['price_vs_cloud'] != current['price_vs_cloud']:
            if current['price_vs_cloud'] == 1:
                signals["price_cloud_breakout"] = "bullish_breakout"
            elif current['price_vs_cloud'] == -1:
                signals["price_cloud_breakout"] = "bearish_breakout"
        
        # Cloud trend
        if not pd.isna(current['cloud_color']):
            signals["cloud_trend"] = "bullish" if current['cloud_color'] == 1 else "bearish"
        
        # Chikou Span confirmation
        if not pd.isna(current['chikou_span']) and len(df) > 26:
            chikou_price = df.iloc[-27]['close'] if len(df) > 26 else None
            if chikou_price is not None:
                if current['chikou_span'] > chikou_price:
                    signals["chikou_confirmation"] = "bullish"
                elif current['chikou_span'] < chikou_price:
                    signals["chikou_confirmation"] = "bearish"
        
        # Overall trend assessment
        bullish_signals = sum([
            signals["tenkan_kijun_cross"] == "bullish",
            signals["price_position"] == "above_cloud",
            signals["cloud_trend"] == "bullish",
            signals["chikou_confirmation"] == "bullish"
        ])
        
        bearish_signals = sum([
            signals["tenkan_kijun_cross"] == "bearish", 
            signals["price_position"] == "below_cloud",
            signals["cloud_trend"] == "bearish",
            signals["chikou_confirmation"] == "bearish"
        ])
        
        if bullish_signals >= 3:
            signals["overall_trend"] = "strong_bullish"
        elif bullish_signals >= 2:
            signals["overall_trend"] = "bullish"
        elif bearish_signals >= 3:
            signals["overall_trend"] = "strong_bearish"
        elif bearish_signals >= 2:
            signals["overall_trend"] = "bearish"
        
        return signals
    
    def analyze(self, symbol: str, timeframe: str, limit: int = 100) -> Dict[str, Any]:
        """
        Perform complete Ichimoku analysis
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe for analysis
            limit: Number of data points to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Fetch OHLCV data
            ohlcv_data = self.data_collector.fetch_ohlcv_data(symbol, timeframe, limit + 60)  # Extra data for calculations
            
            if ohlcv_data is None or len(ohlcv_data) < 60:
                return {
                    "success": False,
                    "error": f"Insufficient data for {symbol} on {timeframe}"
                }
            
            # Convert OHLCV list to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate Ichimoku components
            df_ichimoku = self.calculate_ichimoku(df)
            
            # Get current values
            current = df_ichimoku.iloc[-1]
            
            # Get signals
            signals = self.get_ichimoku_signals(df_ichimoku)
            
            # Calculate support/resistance from cloud
            cloud_top = max(current['senkou_span_a'], current['senkou_span_b']) if not pd.isna(current['senkou_span_a']) and not pd.isna(current['senkou_span_b']) else None
            cloud_bottom = min(current['senkou_span_a'], current['senkou_span_b']) if not pd.isna(current['senkou_span_a']) and not pd.isna(current['senkou_span_b']) else None
            
            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": current['timestamp'] if 'timestamp' in current else None,
                
                # Current values
                "current_price": float(current['close']),
                "tenkan_sen": float(current['tenkan_sen']) if not pd.isna(current['tenkan_sen']) else None,
                "kijun_sen": float(current['kijun_sen']) if not pd.isna(current['kijun_sen']) else None,
                "senkou_span_a": float(current['senkou_span_a']) if not pd.isna(current['senkou_span_a']) else None,
                "senkou_span_b": float(current['senkou_span_b']) if not pd.isna(current['senkou_span_b']) else None,
                "chikou_span": float(current['chikou_span']) if not pd.isna(current['chikou_span']) else None,
                
                # Cloud levels
                "cloud_top": float(cloud_top) if cloud_top is not None else None,
                "cloud_bottom": float(cloud_bottom) if cloud_bottom is not None else None,
                "cloud_thickness": float(abs(current['senkou_span_a'] - current['senkou_span_b'])) if not pd.isna(current['senkou_span_a']) and not pd.isna(current['senkou_span_b']) else None,
                
                # Signals and analysis
                "signals": signals,
                "trend_strength": self._calculate_trend_strength(df_ichimoku),
                "recommendation": self._get_recommendation(signals)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Analysis failed for {symbol}: {str(e)}"
            }
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> str:
        """Calculate trend strength based on multiple factors"""
        if len(df) < 10:
            return "insufficient_data"
            
        recent_data = df.tail(10)
        
        # Check price vs cloud consistency
        above_cloud_count = (recent_data['price_vs_cloud'] == 1).sum()
        below_cloud_count = (recent_data['price_vs_cloud'] == -1).sum()
        
        # Check cloud color consistency
        bullish_cloud_count = (recent_data['cloud_color'] == 1).sum()
        bearish_cloud_count = (recent_data['cloud_color'] == -1).sum()
        
        if above_cloud_count >= 8 and bullish_cloud_count >= 7:
            return "very_strong_bullish"
        elif above_cloud_count >= 6 and bullish_cloud_count >= 5:
            return "strong_bullish"
        elif below_cloud_count >= 8 and bearish_cloud_count >= 7:
            return "very_strong_bearish"
        elif below_cloud_count >= 6 and bearish_cloud_count >= 5:
            return "strong_bearish"
        else:
            return "weak_or_consolidating"
    
    def _get_recommendation(self, signals: Dict[str, Any]) -> str:
        """Generate trading recommendation based on signals"""
        overall_trend = signals.get("overall_trend", "neutral")
        price_position = signals.get("price_position", "in_cloud")
        
        if overall_trend == "strong_bullish" and price_position == "above_cloud":
            return "strong_buy"
        elif overall_trend == "bullish":
            return "buy"
        elif overall_trend == "strong_bearish" and price_position == "below_cloud":
            return "strong_sell"
        elif overall_trend == "bearish":
            return "sell"
        else:
            return "hold"


def main():
    """CLI interface for Ichimoku analysis"""
    if len(sys.argv) < 4:
        print("Usage: python ichimoku.py <symbol> <timeframe> <limit>")
        print("Example: python ichimoku.py BTC/USDT 1h 100")
        return
    
    symbol = sys.argv[1]
    timeframe = sys.argv[2]
    limit = int(sys.argv[3])
    
    analysis = IchimokuAnalysis()
    result = analysis.analyze(symbol, timeframe, limit)
    
    if result["success"]:
        print(f"\n=== Ichimoku Cloud Analysis for {symbol} ({timeframe}) ===")
        print(f"Current Price: ${result['current_price']:.4f}")
        print(f"Tenkan-sen (Conversion): ${result['tenkan_sen']:.4f}" if result['tenkan_sen'] else "Tenkan-sen: N/A")
        print(f"Kijun-sen (Base): ${result['kijun_sen']:.4f}" if result['kijun_sen'] else "Kijun-sen: N/A")
        print(f"Senkou Span A: ${result['senkou_span_a']:.4f}" if result['senkou_span_a'] else "Senkou Span A: N/A")
        print(f"Senkou Span B: ${result['senkou_span_b']:.4f}" if result['senkou_span_b'] else "Senkou Span B: N/A")
        print(f"Cloud Top: ${result['cloud_top']:.4f}" if result['cloud_top'] else "Cloud Top: N/A")
        print(f"Cloud Bottom: ${result['cloud_bottom']:.4f}" if result['cloud_bottom'] else "Cloud Bottom: N/A")
        
        signals = result['signals']
        print(f"\n--- Signals ---")
        print(f"Price Position: {signals['price_position'].replace('_', ' ').title()}")
        print(f"Cloud Trend: {signals['cloud_trend'].title()}")
        print(f"Overall Trend: {signals['overall_trend'].replace('_', ' ').title()}")
        print(f"Trend Strength: {result['trend_strength'].replace('_', ' ').title()}")
        print(f"Recommendation: {result['recommendation'].replace('_', ' ').upper()}")
        
        if signals['tenkan_kijun_cross']:
            print(f"Tenkan-Kijun Cross: {signals['tenkan_kijun_cross'].title()}")
        if signals['price_cloud_breakout']:
            print(f"Cloud Breakout: {signals['price_cloud_breakout'].replace('_', ' ').title()}")
        if signals['chikou_confirmation']:
            print(f"Chikou Confirmation: {signals['chikou_confirmation'].title()}")
            
    else:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()