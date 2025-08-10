import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EMA_20_50:
    
    def __init__(self, 
                 symbol: str,
                 timeframe: str,
                 limit: int,
                 ob: dict,
                 ticker: dict,            
                 ohlcv: List[List],       
                 trades: List[Dict]):    
        self.rules = {
            "ema_short_period": 20,
            "ema_long_period": 50,
            "trend_strength_threshold": 0.02,  # 2% difference for strong trend
            "signal_confirmation_periods": 3,   # periods to confirm trend change
        }
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
    
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return []
        
        # Convert to pandas Series for easier calculation
        price_series = pd.Series(prices)
        ema = price_series.ewm(span=period, adjust=False).mean()
        return ema.tolist()
    
    def get_trend_signal(self, ema_20: float, ema_50: float, price: float) -> str:
        """Determine trend signal based on EMA relationship"""
        if ema_20 > ema_50 and price > ema_20:
            return "STRONG_BULLISH"
        elif ema_20 > ema_50 and price > ema_50:
            return "BULLISH"
        elif ema_20 < ema_50 and price < ema_20:
            return "STRONG_BEARISH"
        elif ema_20 < ema_50 and price < ema_50:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def detect_crossover(self, ema_20_values: List[float], ema_50_values: List[float]) -> str:
        """Detect EMA crossover signals"""
        if len(ema_20_values) < 2 or len(ema_50_values) < 2:
            return "NO_SIGNAL"
        
        # Current and previous values
        current_20, previous_20 = ema_20_values[-1], ema_20_values[-2]
        current_50, previous_50 = ema_50_values[-1], ema_50_values[-2]
        
        # Golden Cross: EMA 20 crosses above EMA 50
        if previous_20 <= previous_50 and current_20 > current_50:
            return "GOLDEN_CROSS"
        
        # Death Cross: EMA 20 crosses below EMA 50
        if previous_20 >= previous_50 and current_20 < current_50:
            return "DEATH_CROSS"
        
        return "NO_CROSSOVER"
    
    def calculate_trend_strength(self, ema_20: float, ema_50: float) -> float:
        """Calculate trend strength as percentage difference between EMAs"""
        if ema_50 == 0:
            return 0
        return abs((ema_20 - ema_50) / ema_50) * 100
    
    def calculate(self):
        """
        Calculate EMA 20/50 analysis according to TradingView methodology.
        """
        if not self.ohlcv or len(self.ohlcv) < self.rules["ema_long_period"]:
            logger.warning(f"Insufficient data for EMA calculation. Need at least {self.rules['ema_long_period']} periods")
            return
        
        # Extract closing prices
        closes = [float(candle[4]) for candle in self.ohlcv]
        current_price = closes[-1]
        
        # Calculate EMAs
        ema_20_values = self.calculate_ema(closes, self.rules["ema_short_period"])
        ema_50_values = self.calculate_ema(closes, self.rules["ema_long_period"])
        
        if not ema_20_values or not ema_50_values:
            logger.warning("Unable to calculate EMAs")
            return
        
        # Current EMA values
        current_ema_20 = ema_20_values[-1]
        current_ema_50 = ema_50_values[-1]
        
        # Analysis
        trend_signal = self.get_trend_signal(current_ema_20, current_ema_50, current_price)
        crossover_signal = self.detect_crossover(ema_20_values, ema_50_values)
        trend_strength = self.calculate_trend_strength(current_ema_20, current_ema_50)
        
        # Support/Resistance levels
        support_level = min(current_ema_20, current_ema_50)
        resistance_level = max(current_ema_20, current_ema_50)
        
        # Distance from EMAs
        distance_from_ema20 = ((current_price - current_ema_20) / current_ema_20) * 100
        distance_from_ema50 = ((current_price - current_ema_50) / current_ema_50) * 100
        
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": round(current_price, 6),
            "ema_20": round(current_ema_20, 6),
            "ema_50": round(current_ema_50, 6),
            "trend_signal": trend_signal,
            "crossover_signal": crossover_signal,
            "trend_strength_percent": round(trend_strength, 2),
            "support_level": round(support_level, 6),
            "resistance_level": round(resistance_level, 6),
            "distance_from_ema20_percent": round(distance_from_ema20, 2),
            "distance_from_ema50_percent": round(distance_from_ema50, 2),
            "is_strong_trend": trend_strength > self.rules["trend_strength_threshold"]
        }
        
        self.print_output(result)
        return result
    
    def print_output(self, result: dict):
        """Print the EMA 20/50 analysis output"""
        print("\n" + "="*60)
        print(f"📈 EMA 20/50 ANALYSIS - {result['symbol']} ({result['timeframe']})")
        print("="*60)
        print(f"Current Price:     ${result['current_price']}")
        print(f"EMA 20:           ${result['ema_20']}")
        print(f"EMA 50:           ${result['ema_50']}")
        print("-"*60)
        print(f"Trend Signal:      {result['trend_signal']}")
        print(f"Crossover Signal:  {result['crossover_signal']}")
        print(f"Trend Strength:    {result['trend_strength_percent']}%")
        print("-"*60)
        print(f"Support Level:     ${result['support_level']}")
        print(f"Resistance Level:  ${result['resistance_level']}")
        print("-"*60)
        print(f"Distance from EMA20: {result['distance_from_ema20_percent']:+.2f}%")
        print(f"Distance from EMA50: {result['distance_from_ema50_percent']:+.2f}%")
        
        # Trading interpretation
        print("\n📊 INTERPRETATION:")
        if result['crossover_signal'] == 'GOLDEN_CROSS':
            print("🟢 GOLDEN CROSS detected - Potential bullish momentum")
        elif result['crossover_signal'] == 'DEATH_CROSS':
            print("🔴 DEATH CROSS detected - Potential bearish momentum")
        
        if result['trend_signal'] == 'STRONG_BULLISH':
            print("🚀 Strong bullish trend - Price above both EMAs")
        elif result['trend_signal'] == 'STRONG_BEARISH':
            print("📉 Strong bearish trend - Price below both EMAs")
        elif result['trend_signal'] == 'NEUTRAL':
            print("⚖️  Neutral/Consolidation phase")
        
        if result['is_strong_trend']:
            print(f"💪 Strong trend confirmed ({result['trend_strength_percent']}% separation)")