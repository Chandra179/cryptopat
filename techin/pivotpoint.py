from typing import List, Dict

class PivotPoint:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.rules = {
            "pivot_types": ["standard", "fibonacci", "woodie", "camarilla", "demark"],
            "fibonacci_ratios": {
                "r3": 1.000,
                "r2": 0.618,
                "r1": 0.382,
                "s1": 0.382,
                "s2": 0.618,
                "s3": 1.000
            },
            "camarilla_multiplier": 1.1 / 12,
            "woodie_weight": {
                "close": 2,
                "high": 1,
                "low": 1
            }
        }
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
    
    def calculate(self):
        """
        Calculate Pivot Point levels according to TradingView methodology.
        """
        if not self.ohlcv or len(self.ohlcv) == 0:
            print("No OHLCV data available for pivot point calculation")
            return
            
        latest_candle = self.ohlcv[-1]
        high = float(latest_candle[2])
        low = float(latest_candle[3])
        close = float(latest_candle[4])
        
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "pivot_levels": {},
            "current_price": close
        }
        
        # Standard Pivot Points
        pp = (high + low + close) / 3
        result["pivot_levels"]["standard"] = {
            "pivot": round(pp, 6),
            "r3": round(pp + 2 * (high - low), 6),
            "r2": round(pp + (high - low), 6),
            "r1": round((2 * pp) - low, 6),
            "s1": round((2 * pp) - high, 6),
            "s2": round(pp - (high - low), 6),
            "s3": round(pp - 2 * (high - low), 6)
        }
        
        # Fibonacci Pivot Points
        range_hl = high - low
        result["pivot_levels"]["fibonacci"] = {
            "pivot": round(pp, 6),
            "r3": round(pp + (range_hl * self.rules["fibonacci_ratios"]["r3"]), 6),
            "r2": round(pp + (range_hl * self.rules["fibonacci_ratios"]["r2"]), 6),
            "r1": round(pp + (range_hl * self.rules["fibonacci_ratios"]["r1"]), 6),
            "s1": round(pp - (range_hl * self.rules["fibonacci_ratios"]["s1"]), 6),
            "s2": round(pp - (range_hl * self.rules["fibonacci_ratios"]["s2"]), 6),
            "s3": round(pp - (range_hl * self.rules["fibonacci_ratios"]["s3"]), 6)
        }
        
        # Woodie Pivot Points
        woodie_pp = (high + low + 2 * close) / 4
        result["pivot_levels"]["woodie"] = {
            "pivot": round(woodie_pp, 6),
            "r2": round(woodie_pp + (high - low), 6),
            "r1": round((2 * woodie_pp) - low, 6),
            "s1": round((2 * woodie_pp) - high, 6),
            "s2": round(woodie_pp - (high - low), 6)
        }
        
        # Camarilla Pivot Points
        cam_mult = self.rules["camarilla_multiplier"]
        result["pivot_levels"]["camarilla"] = {
            "pivot": round(close, 6),
            "r4": round(close + ((high - low) * 0.55), 6),
            "r3": round(close + ((high - low) * cam_mult * 4), 6),
            "r2": round(close + ((high - low) * cam_mult * 2), 6),
            "r1": round(close + ((high - low) * cam_mult), 6),
            "s1": round(close - ((high - low) * cam_mult), 6),
            "s2": round(close - ((high - low) * cam_mult * 2), 6),
            "s3": round(close - ((high - low) * cam_mult * 4), 6),
            "s4": round(close - ((high - low) * 0.55), 6)
        }
        
        # DeMark Pivot Points
        if close < low:
            x = high + 2 * low + close
        elif close > high:
            x = 2 * high + low + close
        else:
            x = high + low + 2 * close
            
        demark_pp = x / 4
        result["pivot_levels"]["demark"] = {
            "pivot": round(demark_pp, 6),
            "r1": round(x / 2 - low, 6),
            "s1": round(x / 2 - high, 6)
        }
        
        # Determine current market position relative to pivot
        standard_pivot = result["pivot_levels"]["standard"]["pivot"]
        if close > standard_pivot:
            result["market_bias"] = "bullish"
            result["nearest_resistance"] = self._find_nearest_resistance(close, result["pivot_levels"]["standard"])
            result["nearest_support"] = standard_pivot
        else:
            result["market_bias"] = "bearish"
            result["nearest_support"] = self._find_nearest_support(close, result["pivot_levels"]["standard"])
            result["nearest_resistance"] = standard_pivot
            
        self.print_output(result)
        return result
    
    def _find_nearest_resistance(self, price, levels):
        resistances = [levels["r1"], levels["r2"], levels["r3"]]
        for r in resistances:
            if r > price:
                return r
        return resistances[-1]
    
    def _find_nearest_support(self, price, levels):
        supports = [levels["s1"], levels["s2"], levels["s3"]]
        for s in reversed(supports):
            if s < price:
                return s
        return supports[-1]
    
    def print_output(self, result: dict):
        """Print the pivot point analysis output"""
        print("\n" + "="*50)
        print("PIVOT POINT ANALYSIS")
        print("="*50)
        print(f"Symbol: {result['symbol']}")
        print(f"Timeframe: {result['timeframe']}")
        print(f"Current Price: {result['current_price']}")
        print(f"Market Bias: {result['market_bias'].upper()}")
        print(f"Nearest Support: {result['nearest_support']}")
        print(f"Nearest Resistance: {result['nearest_resistance']}")
        
        for pivot_type, levels in result["pivot_levels"].items():
            print(f"\n{pivot_type.upper()} PIVOT POINTS:")
            for level, value in levels.items():
                print(f"  {level.upper()}: {value}")