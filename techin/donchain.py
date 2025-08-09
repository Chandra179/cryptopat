from typing import List, Dict

class DonchianChannel:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.rules = {
            "period": 20,
            "breakout_threshold": 0.001
        }
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
    
    def calculate(self):
        """
        Calculate Donchian Channel according to TradingView methodology.
        """
        if len(self.ohlcv) < self.rules["period"]:
            result = {
                "error": f"Not enough data. Need at least {self.rules['period']} candles, got {len(self.ohlcv)}"
            }
            self.print_output(result)
            return
        
        period = self.rules["period"]
        upper_band = []
        lower_band = []
        middle_band = []
        
        for i in range(period - 1, len(self.ohlcv)):
            window = self.ohlcv[i - period + 1:i + 1]
            
            highest_high = max([candle[2] for candle in window])
            lowest_low = min([candle[3] for candle in window])
            middle = (highest_high + lowest_low) / 2
            
            upper_band.append(highest_high)
            lower_band.append(lowest_low)
            middle_band.append(middle)
        
        current_price = self.ohlcv[-1][4]
        current_upper = upper_band[-1] if upper_band else None
        current_lower = lower_band[-1] if lower_band else None
        current_middle = middle_band[-1] if middle_band else None
        
        signal = "NEUTRAL"
        if current_price and current_upper and current_lower:
            if current_price >= current_upper:
                signal = "BREAKOUT_UP"
            elif current_price <= current_lower:
                signal = "BREAKOUT_DOWN"
            elif current_price > current_middle:
                signal = "ABOVE_MIDDLE"
            elif current_price < current_middle:
                signal = "BELOW_MIDDLE"
        
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "period": period,
            "current_price": current_price,
            "upper_band": current_upper,
            "lower_band": current_lower,
            "middle_band": current_middle,
            "signal": signal,
            "band_width": current_upper - current_lower if current_upper and current_lower else None,
            "price_position_pct": ((current_price - current_lower) / (current_upper - current_lower) * 100) if current_upper and current_lower and current_price else None
        }
        
        self.print_output(result)
    
    def print_output(self, result: dict):
        print("\n" + "="*50)
        print("DONCHIAN CHANNEL ANALYSIS")
        print("="*50)
        
        if "error" in result:
            print(f"ERROR: {result['error']}")
            return
        
        print(f"Symbol: {result['symbol']}")
        print(f"Timeframe: {result['timeframe']}")
        print(f"Period: {result['period']}")
        print(f"Current Price: ${result['current_price']:.4f}")
        print(f"Upper Band: ${result['upper_band']:.4f}")
        print(f"Lower Band: ${result['lower_band']:.4f}")
        print(f"Middle Band: ${result['middle_band']:.4f}")
        print(f"Band Width: ${result['band_width']:.4f}")
        print(f"Price Position: {result['price_position_pct']:.2f}%")
        print(f"Signal: {result['signal']}")