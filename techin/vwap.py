from typing import List, Dict
import pandas as pd
import numpy as np

class VWAP:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.param = {
            "period_length": 20,  # VWAP calculation period
            "deviation_threshold": 0.02,  # 2% deviation from VWAP considered significant
            "volume_weight_formula": lambda price, volume: price * volume
        }
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
    
    def calculate(self):
        """
        Calculate VWAP (Volume Weighted Average Price) according to TradingView methodology.
        """
        if not self.ohlcv or len(self.ohlcv) == 0:
            result = {"error": "No OHLCV data available for VWAP calculation"}
            self.print_output(result)
            return
        
        # Convert OHLCV to DataFrame for easier manipulation
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Calculate typical price (HLC/3)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate VWAP
        df['price_volume'] = df['typical_price'] * df['volume']
        df['cumulative_pv'] = df['price_volume'].cumsum()
        df['cumulative_volume'] = df['volume'].cumsum()
        df['vwap'] = df['cumulative_pv'] / df['cumulative_volume']
        
        # Calculate rolling VWAP for specified period
        period = self.param["period_length"]
        df['rolling_pv'] = df['price_volume'].rolling(window=period).sum()
        df['rolling_volume'] = df['volume'].rolling(window=period).sum()
        df['rolling_vwap'] = df['rolling_pv'] / df['rolling_volume']
        
        # Calculate deviations
        df['price_to_vwap_ratio'] = df['close'] / df['vwap']
        df['deviation_from_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        df['rolling_deviation'] = (df['close'] - df['rolling_vwap']) / df['rolling_vwap']
        
        # Identify significant deviations
        threshold = self.param["deviation_threshold"]
        df['above_vwap_threshold'] = df['deviation_from_vwap'] > threshold
        df['below_vwap_threshold'] = df['deviation_from_vwap'] < -threshold
        
        # Get current values (last candle)
        current = df.iloc[-1]
        
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": float(current['close']),
            "current_vwap": float(current['vwap']),
            "rolling_vwap": float(current['rolling_vwap']) if not pd.isna(current['rolling_vwap']) else None,
            "deviation_from_vwap": float(current['deviation_from_vwap']),
            "rolling_deviation": float(current['rolling_deviation']) if not pd.isna(current['rolling_deviation']) else None,
            "price_to_vwap_ratio": float(current['price_to_vwap_ratio']),
            "above_threshold": bool(current['above_vwap_threshold']),
            "below_threshold": bool(current['below_vwap_threshold']),
            "total_volume": float(df['volume'].sum()),
            "avg_volume": float(df['volume'].mean()),
            "vwap_trend": "bullish" if current['close'] > current['vwap'] else "bearish",
            "period_used": period,
            "data_points": len(df)
        }
        
        self.print_output(result)
        return result
    
    def print_output(self, result: dict):
        """Print the VWAP analysis output"""
        if "error" in result:
            print(f"VWAP Error: {result['error']}")
            return
            
        print(f"\n{'='*50}")
        print(f"VWAP ANALYSIS")
        print(f"{'='*50}")
        print(f"Current Price: ${result['current_price']:.4f}")
        print(f"Current VWAP: ${result['current_vwap']:.4f}")
        if result['rolling_vwap']:
            print(f"Rolling VWAP ({result['period_used']}): ${result['rolling_vwap']:.4f}")
        
        print(f"Deviation from VWAP: {result['deviation_from_vwap']:.4%}")
        if result['rolling_deviation']:
            print(f"Rolling Deviation: {result['rolling_deviation']:.4%}")
        
        print(f"Price/VWAP Ratio: {result['price_to_vwap_ratio']:.4f}")
        print(f"VWAP Trend: {result['vwap_trend'].upper()}")
        
        if result['above_threshold']:
            print("⚠️  Price significantly ABOVE VWAP (potential resistance)")
        elif result['below_threshold']:
            print("⚠️  Price significantly BELOW VWAP (potential support)")
        else:
            print("✅ Price near VWAP (fair value zone)")
        
        print(f"Total Volume: {result['total_volume']:,.0f}")
        print(f"Average Volume: {result['avg_volume']:,.0f}")
        print(f"Data Points: {result['data_points']}")