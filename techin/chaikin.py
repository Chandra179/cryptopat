"""
Chaikin Money Flow (CMF) Analysis Module
Calculates Chaikin Money Flow indicator to measure buying/selling pressure using price and volume
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from data import get_data_collector


class ChaikinMoneyFlowAnalyzer:
    def __init__(self):
        self.data_collector = get_data_collector()
    
    def calculate_money_flow_multiplier(self, high: float, low: float, close: float) -> float:
        """Calculate Money Flow Multiplier (MFM)"""
        if high == low:
            return 0.0
        return ((close - low) - (high - close)) / (high - low)
    
    def calculate_money_flow_volume(self, mfm: float, volume: float) -> float:
        """Calculate Money Flow Volume (MFV)"""
        return mfm * volume
    
    def calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow over specified period"""
        # Calculate Money Flow Multiplier
        df['mfm'] = df.apply(
            lambda row: self.calculate_money_flow_multiplier(row['high'], row['low'], row['close']), 
            axis=1
        )
        
        # Calculate Money Flow Volume
        df['mfv'] = df['mfm'] * df['volume']
        
        # Calculate CMF as rolling sum of MFV / rolling sum of Volume
        mfv_sum = df['mfv'].rolling(window=period).sum()
        volume_sum = df['volume'].rolling(window=period).sum()
        
        # Avoid division by zero
        cmf = np.where(volume_sum != 0, mfv_sum / volume_sum, 0)
        
        return pd.Series(cmf, index=df.index)
    
    def interpret_cmf(self, cmf_value: float) -> Dict[str, Any]:
        """Interpret CMF value and provide analysis"""
        if cmf_value > 0.25:
            strength = "Very Strong"
            bias = "Bullish"
            description = "Strong buying pressure - accumulation phase"
        elif cmf_value > 0.1:
            strength = "Strong"
            bias = "Bullish"
            description = "Moderate buying pressure - bullish momentum"
        elif cmf_value > 0:
            strength = "Weak"
            bias = "Bullish"
            description = "Slight buying pressure - weak bullish sentiment"
        elif cmf_value > -0.1:
            strength = "Weak"
            bias = "Bearish"
            description = "Slight selling pressure - weak bearish sentiment"
        elif cmf_value > -0.25:
            strength = "Strong"
            bias = "Bearish"
            description = "Moderate selling pressure - bearish momentum"
        else:
            strength = "Very Strong"
            bias = "Bearish"
            description = "Strong selling pressure - distribution phase"
        
        return {
            'bias': bias,
            'strength': strength,
            'description': description,
            'pressure_type': "Buying" if cmf_value > 0 else "Selling"
        }
    
    def detect_divergences(self, df: pd.DataFrame, cmf_series: pd.Series, lookback: int = 20) -> List[Dict[str, Any]]:
        """Detect bullish and bearish divergences between price and CMF"""
        divergences = []
        
        if len(df) < lookback * 2:
            return divergences
        
        # Get recent highs and lows for price and CMF
        price_highs = df['high'].rolling(window=lookback).max()
        price_lows = df['low'].rolling(window=lookback).min()
        cmf_highs = cmf_series.rolling(window=lookback).max()
        cmf_lows = cmf_series.rolling(window=lookback).min()
        
        # Check for divergences in the last few periods
        for i in range(-10, 0):  # Check last 10 periods
            if i + len(df) < lookback:
                continue
                
            idx = len(df) + i
            
            # Bullish divergence: price makes lower low, CMF makes higher low
            if (df['low'].iloc[idx] < price_lows.iloc[idx-lookback] and 
                cmf_series.iloc[idx] > cmf_lows.iloc[idx-lookback]):
                
                divergences.append({
                    'type': 'Bullish',
                    'period': idx,
                    'description': 'Price lower low, CMF higher low - potential reversal up'
                })
            
            # Bearish divergence: price makes higher high, CMF makes lower high
            elif (df['high'].iloc[idx] > price_highs.iloc[idx-lookback] and 
                  cmf_series.iloc[idx] < cmf_highs.iloc[idx-lookback]):
                
                divergences.append({
                    'type': 'Bearish',
                    'period': idx,
                    'description': 'Price higher high, CMF lower high - potential reversal down'
                })
        
        return divergences
    
    def analyze(self, symbol: str, timeframe: str, limit: int = 100, period: int = 20, ohlcv_data=None) -> Dict[str, Any]:
        """
        Perform Chaikin Money Flow analysis
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe for analysis (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch
            period: CMF calculation period (default 20)
            
        Returns:
            Dictionary containing CMF analysis results
        """
        try:
            # Use provided OHLCV data or fetch new data
            if ohlcv_data is not None:
                raw_ohlcv = ohlcv_data
            else:
                raw_ohlcv = self.data_collector.fetch_ohlcv_data(symbol, timeframe, limit)
            
            if not raw_ohlcv or len(raw_ohlcv) < period:
                return {'success': False, 'error': f'Insufficient data (need at least {period} periods)'}
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(raw_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate CMF
            cmf_series = self.calculate_cmf(df, period)
            
            # Get current and recent values
            current_cmf = cmf_series.iloc[-1]
            prev_cmf = cmf_series.iloc[-2] if len(cmf_series) > 1 else current_cmf
            cmf_change = current_cmf - prev_cmf
            
            # Get interpretation
            interpretation = self.interpret_cmf(current_cmf)
            
            # Detect divergences
            divergences = self.detect_divergences(df, cmf_series)
            
            # Calculate trend (last 5 periods)
            recent_cmf = cmf_series.tail(5)
            if len(recent_cmf) >= 2:
                trend = "Rising" if recent_cmf.iloc[-1] > recent_cmf.iloc[0] else "Falling"
            else:
                trend = "Neutral"
            
            # Generate signal
            signal = self._generate_signal(current_cmf, cmf_change, trend, divergences)
            
            # Calculate statistics
            cmf_stats = {
                'mean': float(cmf_series.dropna().mean()),
                'std': float(cmf_series.dropna().std()),
                'min': float(cmf_series.dropna().min()),
                'max': float(cmf_series.dropna().max())
            }
            
            return {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'period': period,
                'current_cmf': float(current_cmf),
                'previous_cmf': float(prev_cmf),
                'cmf_change': float(cmf_change),
                'trend': trend,
                'interpretation': interpretation,
                'divergences': divergences,
                'signal': signal,
                'statistics': cmf_stats,
                'cmf_values': cmf_series.dropna().tail(10).tolist()  # Last 10 values
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_signal(self, current_cmf: float, cmf_change: float, trend: str, divergences: List) -> str:
        """Generate trading signal based on CMF analysis"""
        
        # Check for divergences first
        if divergences:
            latest_div = divergences[-1]
            if latest_div['type'] == 'Bullish':
                return "BUY - Bullish Divergence Detected"
            else:
                return "SELL - Bearish Divergence Detected"
        
        # Strong signals
        if current_cmf > 0.25 and cmf_change > 0:
            return "STRONG BUY - Very Strong Buying Pressure"
        elif current_cmf < -0.25 and cmf_change < 0:
            return "STRONG SELL - Very Strong Selling Pressure"
        
        # Moderate signals
        elif current_cmf > 0.1 and trend == "Rising":
            return "BUY - Rising Buying Pressure"
        elif current_cmf < -0.1 and trend == "Falling":
            return "SELL - Rising Selling Pressure"
        
        # Weak signals
        elif current_cmf > 0 and cmf_change > 0.05:
            return "WEAK BUY - Increasing Buying Pressure"
        elif current_cmf < 0 and cmf_change < -0.05:
            return "WEAK SELL - Increasing Selling Pressure"
        
        # Neutral/trend change signals
        elif abs(current_cmf) < 0.05:
            return "NEUTRAL - Balanced Money Flow"
        elif current_cmf > 0 and cmf_change < 0:
            return "CAUTION - Weakening Buying Pressure"
        elif current_cmf < 0 and cmf_change > 0:
            return "CAUTION - Weakening Selling Pressure"
        
        else:
            return "HOLD - No Clear Signal"


def analyze_chaikin_money_flow(symbol: str, timeframe: str, limit: int = 100, period: int = 20) -> Dict[str, Any]:
    """Convenience function for Chaikin Money Flow analysis"""
    analyzer = ChaikinMoneyFlowAnalyzer()
    return analyzer.analyze(symbol, timeframe, limit, period)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python techin/chaikin.py <symbol> [timeframe] [limit] [period]")
        print("Example: python techin/chaikin.py BTC/USDT 1h 100 20")
        sys.exit(1)
    
    symbol = sys.argv[1]
    timeframe = sys.argv[2] if len(sys.argv) > 2 else '1h'
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    period = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    
    result = analyze_chaikin_money_flow(symbol, timeframe, limit, period)
    
    if result['success']:
        print(f"\n=== CHAIKIN MONEY FLOW ANALYSIS: {symbol} ({timeframe}) ===")
        print(f"Period: {result['period']}")
        print(f"Current CMF: {result['current_cmf']:.4f}")
        print(f"Previous CMF: {result['previous_cmf']:.4f}")
        print(f"Change: {result['cmf_change']:+.4f}")
        print(f"Trend: {result['trend']}")
        
        print(f"\n--- Interpretation ---")
        interp = result['interpretation']
        print(f"Bias: {interp['bias']}")
        print(f"Strength: {interp['strength']}")
        print(f"Pressure Type: {interp['pressure_type']}")
        print(f"Description: {interp['description']}")
        
        print(f"\n--- Signal ---")
        print(f"{result['signal']}")
        
        if result['divergences']:
            print(f"\n--- Divergences ---")
            for div in result['divergences']:
                print(f"{div['type']}: {div['description']}")
        
        print(f"\n--- Statistics ---")
        stats = result['statistics']
        print(f"Mean CMF: {stats['mean']:.4f}")
        print(f"Min/Max: {stats['min']:.4f} / {stats['max']:.4f}")
        print(f"Std Dev: {stats['std']:.4f}")
        
    else:
        print(f"Error: {result['error']}")