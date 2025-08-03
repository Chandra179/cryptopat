"""
ATR-based Risk Management System
Implements dynamic support/resistance, stop-loss, and profit targets based on volatility
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from data.collector import DataCollector


class ATRRiskManager:
    """
    ATR-based risk management system implementing the goals from logs.txt:
    1. Dynamic Support & Resistance calculation
    2. ATR-based volatility measurement
    3. Systematic stop-loss placement
    4. Multiple profit targets
    5. Risk/reward validation
    """
    
    def __init__(self, 
                 lookback_period: int = 20,
                 atr_period: int = 14,
                 sl_buffer: float = 1.1,
                 r1_multiple: float = 1.5,
                 r2_multiple: float = 2.0):
        """
        Initialize ATR Risk Manager
        
        Args:
            lookback_period (N): Period for support/resistance calculation
            atr_period (M): Period for ATR calculation
            sl_buffer: Stop loss buffer multiplier
            r1_multiple: First target reward multiple
            r2_multiple: Second target reward multiple
        """
        self.lookback_period = lookback_period
        self.atr_period = atr_period
        self.sl_buffer = sl_buffer
        self.r1_multiple = r1_multiple
        self.r2_multiple = r2_multiple
        self.collector = DataCollector()
    
    def calculate_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Calculate True Range (TR) for ATR computation
        TR = max(high - low, |high - prev_close|, |low - prev_close|)
        """
        # Shift close by 1 to get previous close
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # Handle first value
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        return true_range
    
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """
        Calculate Average True Range (ATR)
        ATR = rolling_mean(TR, over M bars)
        """
        true_range = self.calculate_true_range(high, low, close)
        
        # Calculate rolling mean for ATR period
        if len(true_range) >= self.atr_period:
            atr = np.mean(true_range[-self.atr_period:])
        else:
            atr = np.mean(true_range)
        
        return atr
    
    def calculate_support_resistance(self, high: np.ndarray, low: np.ndarray) -> Tuple[float, float]:
        """
        Calculate dynamic support and resistance levels
        Support = min(low over last N bars)
        Resistance = max(high over last N bars)
        """
        if len(low) >= self.lookback_period:
            support = np.min(low[-self.lookback_period:])
            resistance = np.max(high[-self.lookback_period:])
        else:
            support = np.min(low)
            resistance = np.max(high)
        
        return support, resistance
    
    def calculate_levels(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Dict:
        """
        Calculate all risk management levels for a given symbol
        
        Returns:
            Dict containing support, resistance, ATR, stop loss, targets, and R:R ratios
        """
        try:
            # Fetch OHLCV data
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
            
            if len(ohlcv_data) < max(self.lookback_period, self.atr_period):
                raise ValueError(f"Insufficient data: need at least {max(self.lookback_period, self.atr_period)} candles")
            
            # Convert to numpy arrays
            timestamps = np.array([candle[0] for candle in ohlcv_data])
            opens = np.array([candle[1] for candle in ohlcv_data])
            highs = np.array([candle[2] for candle in ohlcv_data])
            lows = np.array([candle[3] for candle in ohlcv_data])
            closes = np.array([candle[4] for candle in ohlcv_data])
            volumes = np.array([candle[5] for candle in ohlcv_data])
            
            # Current price (last close)
            current_price = closes[-1]
            
            # Calculate support and resistance
            support, resistance = self.calculate_support_resistance(highs, lows)
            
            # Calculate ATR
            atr = self.calculate_atr(highs, lows, closes)
            
            # Calculate stop loss (for long position)
            stop_loss = current_price - (self.sl_buffer * atr)
            
            # Calculate profit targets
            tp1 = current_price + (self.r1_multiple * atr)
            tp2 = current_price + (self.r2_multiple * atr)
            
            # Calculate risk and rewards
            risk = current_price - stop_loss
            reward1 = tp1 - current_price
            reward2 = tp2 - current_price
            
            # Calculate R:R ratios
            rr1 = reward1 / risk if risk > 0 else 0
            rr2 = reward2 / risk if risk > 0 else 0
            
            # Risk percentages
            risk_pct = (risk / current_price) * 100
            reward1_pct = (reward1 / current_price) * 100
            reward2_pct = (reward2 / current_price) * 100
            
            # Market structure analysis
            recent_high = np.max(highs[-5:])  # Last 5 candles high
            recent_low = np.min(lows[-5:])    # Last 5 candles low
            
            # Validation checks
            validation = {
                'targets_above_current': tp1 > current_price and tp2 > current_price,
                'stop_below_current': stop_loss < current_price,
                'positive_rr': rr1 > 0 and rr2 > 0,
                'support_relevance': abs(support - recent_low) / current_price < 0.05,  # Support within 5% of recent low
                'resistance_relevance': abs(resistance - recent_high) / current_price < 0.05  # Resistance within 5% of recent high
            }
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': timestamps[-1],
                'current_price': current_price,
                'atr': atr,
                'atr_pct': (atr / current_price) * 100,
                'support': support,
                'resistance': resistance,
                'stop_loss': stop_loss,
                'tp1': tp1,
                'tp2': tp2,
                'risk': risk,
                'reward1': reward1,
                'reward2': reward2,
                'risk_pct': risk_pct,
                'reward1_pct': reward1_pct,
                'reward2_pct': reward2_pct,
                'rr1': rr1,
                'rr2': rr2,
                'recent_high': recent_high,
                'recent_low': recent_low,
                'validation': validation,
                'parameters': {
                    'lookback_period': self.lookback_period,
                    'atr_period': self.atr_period,
                    'sl_buffer': self.sl_buffer,
                    'r1_multiple': self.r1_multiple,
                    'r2_multiple': self.r2_multiple
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error calculating levels for {symbol}: {str(e)}"
            }
    
    def format_output(self, levels: Dict) -> str:
        """
        Format the calculated levels into a readable output
        """
        if not levels.get('success', True):
            return f"❌ {levels.get('error', 'Unknown error')}"
        
        current = levels['current_price']
        
        output = f"""
🎯 KEY LEVELS - {levels['symbol']} ({levels['timeframe']})
Current Price: ${current:,.2f}

📊 VOLATILITY & STRUCTURE
ATR: ${levels['atr']:.2f} ({levels['atr_pct']:.2f}%)
Support: ${levels['support']:,.2f}
Resistance: ${levels['resistance']:,.2f}

🛑 RISK MANAGEMENT
Stop Loss: ${levels['stop_loss']:,.2f}
Risk: ${levels['risk']:.2f} ({levels['risk_pct']:.2f}%)

🎯 PROFIT TARGETS
TP1: ${levels['tp1']:,.2f} (+{levels['reward1_pct']:.2f}%) | R:R = {levels['rr1']:.2f}
TP2: ${levels['tp2']:,.2f} (+{levels['reward2_pct']:.2f}%) | R:R = {levels['rr2']:.2f}

📈 RECENT STRUCTURE
Recent High: ${levels['recent_high']:,.2f}
Recent Low: ${levels['recent_low']:,.2f}

✅ VALIDATION
"""
        
        validation = levels['validation']
        for check, passed in validation.items():
            status = "✅" if passed else "❌"
            check_name = check.replace('_', ' ').title()
            output += f"{status} {check_name}\n"
        
        # Add recommendations based on validation
        if not validation['targets_above_current']:
            output += "\n⚠️  WARNING: Target prices are below current price!"
        if not validation['stop_below_current']:
            output += "\n⚠️  WARNING: Stop loss is above current price!"
        if not validation['support_relevance']:
            output += f"\n💡 SUGGESTION: Support (${levels['support']:,.2f}) is far from recent lows (${levels['recent_low']:,.2f}). Consider using recent low as support."
        if not validation['resistance_relevance']:
            output += f"\n💡 SUGGESTION: Resistance (${levels['resistance']:,.2f}) is far from recent highs (${levels['recent_high']:,.2f}). Consider using recent high as resistance."
        
        return output.strip()


def main():
    """Test the ATR Risk Manager with BTC/USDT"""
    import sys
    
    # Parse command line arguments
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTC/USDT'
    timeframe = sys.argv[2] if len(sys.argv) > 2 else '1h'
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    # Create risk manager with default parameters
    risk_manager = ATRRiskManager()
    
    # Calculate levels
    levels = risk_manager.calculate_levels(symbol, timeframe, limit)
    
    # Format and print output
    output = risk_manager.format_output(levels)
    print(output)


if __name__ == "__main__":
    main()