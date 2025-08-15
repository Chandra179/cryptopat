"""
On-Balance Volume (OBV) Strategy Analysis

OBV is a technical analysis indicator that uses volume flow to predict changes in price.
- When close > previous close: OBV = previous OBV + volume
- When close < previous close: OBV = previous OBV - volume  
- When close = previous close: OBV = previous OBV
"""

from typing import List, Dict

class OBV:
    
    def __init__(self, 
                 symbol: str,
                 timeframe: str,
                 limit: int,
                 ob: dict,
                 ticker: dict,            
                 ohlcv: List[List],       
                 trades: List[Dict]):    
        self.param = {
            "obv_divergence_threshold": 0.1,  # 10% divergence threshold
            "trend_confirmation_periods": 5,   # periods to confirm trend
            "volume_significance_multiplier": 1.5,  # volume must be X times average
        }
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.ticker = ticker
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
    
    def calculate(self):
        """
        Calculate OBV (On-Balance Volume) according to TradingView methodology.
        """
        if len(self.ohlcv) < 2:
            result = {"error": "Insufficient data for OBV calculation"}
            self.print_output(result)
            return
            
        obv_values = []
        obv = 0
        
        # Calculate OBV for each candle
        for i, candle in enumerate(self.ohlcv):
            timestamp, open_price, high, low, close, volume = candle
            
            if i == 0:
                # First candle, set initial OBV to volume
                obv = volume
            else:
                prev_close = self.ohlcv[i-1][4]  # Previous close price
                
                if close > prev_close:
                    obv += volume  # Accumulation
                elif close < prev_close:  
                    obv -= volume  # Distribution
                # If close == prev_close, OBV remains unchanged
                
            obv_values.append({
                'timestamp': timestamp,
                'close': close,
                'volume': volume,
                'obv': obv
            })
        
        # Analysis
        current_obv = obv_values[-1]['obv']
        current_price = obv_values[-1]['close']
        
        # Calculate OBV trend (last 5 periods)
        trend_periods = min(self.param["trend_confirmation_periods"], len(obv_values))
        recent_obv = [item['obv'] for item in obv_values[-trend_periods:]]
        recent_prices = [item['close'] for item in obv_values[-trend_periods:]]
        
        obv_trend = "neutral"
        price_trend = "neutral"
        
        if len(recent_obv) >= 2:
            obv_change = (recent_obv[-1] - recent_obv[0]) / abs(recent_obv[0]) if recent_obv[0] != 0 else 0
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] != 0 else 0
            
            obv_trend = "bullish" if obv_change > 0.05 else "bearish" if obv_change < -0.05 else "neutral"
            price_trend = "bullish" if price_change > 0.02 else "bearish" if price_change < -0.02 else "neutral"
        
        # Detect divergence
        divergence = "none"
        if obv_trend == "bullish" and price_trend == "bearish":
            divergence = "bullish_divergence"  # OBV rising, price falling
        elif obv_trend == "bearish" and price_trend == "bullish":
            divergence = "bearish_divergence"  # OBV falling, price rising
        
        # Volume significance
        avg_volume = sum([item['volume'] for item in obv_values]) / len(obv_values)
        current_volume = obv_values[-1]['volume']
        volume_significant = current_volume > (avg_volume * self.param["volume_significance_multiplier"])
        
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_obv": current_obv,
            "current_price": current_price,
            "obv_trend": obv_trend,
            "price_trend": price_trend,
            "divergence": divergence,
            "volume_significant": volume_significant,
            "avg_volume": avg_volume,
            "current_volume": current_volume,
            "analysis_periods": len(obv_values),
            "signal": self._generate_signal(obv_trend, price_trend, divergence, volume_significant)
        }
        
        self.print_output(result)
        return result
    
    def _generate_signal(self, obv_trend, price_trend, divergence, volume_significant):
        """Generate trading signal based on OBV analysis"""
        if divergence == "bullish_divergence" and volume_significant:
            return "STRONG_BUY"
        elif divergence == "bearish_divergence" and volume_significant:
            return "STRONG_SELL"
        elif obv_trend == "bullish" and price_trend == "bullish":
            return "BUY"
        elif obv_trend == "bearish" and price_trend == "bearish":
            return "SELL"
        elif divergence == "bullish_divergence":
            return "WEAK_BUY"
        elif divergence == "bearish_divergence":
            return "WEAK_SELL"
        else:
            return "NEUTRAL"
    
    def print_output(self, result: dict):
        """Print the OBV analysis output"""
        if "error" in result:
            print(f"❌ OBV Error: {result['error']}")
            return
            
        print("\n" + "="*50)
        print("📊 ON-BALANCE VOLUME (OBV) ANALYSIS")
        print("="*50)
        print(f"Symbol: {result['symbol']}")
        print(f"Timeframe: {result['timeframe']}")
        print(f"Analysis Periods: {result['analysis_periods']}")
        print()
        print(f"Current Price: ${result['current_price']:.6f}")
        print(f"Current OBV: {result['current_obv']:,.0f}")
        print()
        print(f"OBV Trend: {result['obv_trend'].upper()}")
        print(f"Price Trend: {result['price_trend'].upper()}")
        print(f"Divergence: {result['divergence'].replace('_', ' ').upper()}")
        print()
        print(f"Current Volume: {result['current_volume']:,.0f}")
        print(f"Average Volume: {result['avg_volume']:,.0f}")
        print(f"Volume Significant: {'✅ YES' if result['volume_significant'] else '❌ NO'}")
        print()
        
        # Signal output with colors/emojis
        signal = result['signal']
        signal_display = {
            'STRONG_BUY': '🟢 STRONG BUY',
            'BUY': '🟢 BUY', 
            'WEAK_BUY': '🟡 WEAK BUY',
            'NEUTRAL': '⚪ NEUTRAL',
            'WEAK_SELL': '🟡 WEAK SELL',
            'SELL': '🔴 SELL',
            'STRONG_SELL': '🔴 STRONG SELL'
        }
        
        print(f"📊 SIGNAL: {signal_display.get(signal, signal)}")