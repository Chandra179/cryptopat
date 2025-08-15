from typing import List, Dict

class ChaikinMoneyFlow:
    
    def __init__(self, 
                 symbol: str,
                 timeframe: str,
                 limit: int,
                 ob: dict,
                 ticker: dict,            
                 ohlcv: List[List],       
                 trades: List[Dict]):    
        self.param = {
            "cmf_period": 20,  # Standard CMF period
            "strong_buying_threshold": 0.2,  # CMF > 0.2 indicates strong buying pressure
            "strong_selling_threshold": -0.2,  # CMF < -0.2 indicates strong selling pressure
            "money_flow_multiplier_formula": lambda high, low, close: ((close - low) - (high - close)) / (high - low) if (high - low) != 0 else 0,
            "cmf_formula": lambda mf_volume_sum, volume_sum: mf_volume_sum / volume_sum if volume_sum != 0 else 0
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
        Calculate Chaikin Money Flow according to TradingView methodology.
        """
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "cmf_values": [],
            "money_flow_multipliers": [],
            "money_flow_volumes": [],
            "interpretation": ""
        }
        
        if len(self.ohlcv) < self.param["cmf_period"]:
            result["error"] = f"Insufficient data: need at least {self.param['cmf_period']} candles, got {len(self.ohlcv)}"
            self.print_output(result)
            return result
        
        # Calculate Money Flow Multiplier and Money Flow Volume for each candle
        money_flow_multipliers = []
        money_flow_volumes = []
        
        for candle in self.ohlcv:
            timestamp, open_price, high, low, close, volume = candle
            
            # Calculate Money Flow Multiplier
            mf_multiplier = self.param["money_flow_multiplier_formula"](high, low, close)
            money_flow_multipliers.append(mf_multiplier)
            
            # Calculate Money Flow Volume
            mf_volume = mf_multiplier * volume
            money_flow_volumes.append(mf_volume)
        
        result["money_flow_multipliers"] = money_flow_multipliers
        result["money_flow_volumes"] = money_flow_volumes
        
        # Calculate CMF for each period
        cmf_values = []
        period = self.param["cmf_period"]
        
        for i in range(period - 1, len(self.ohlcv)):
            # Sum of Money Flow Volume over the period
            mf_volume_sum = sum(money_flow_volumes[i - period + 1:i + 1])
            
            # Sum of Volume over the period
            volume_sum = sum([candle[5] for candle in self.ohlcv[i - period + 1:i + 1]])
            
            # Calculate CMF
            cmf = self.param["cmf_formula"](mf_volume_sum, volume_sum)
            cmf_values.append({
                "timestamp": self.ohlcv[i][0],
                "cmf": round(cmf, 6)
            })
        
        result["cmf_values"] = cmf_values
        
        # Get latest CMF value for interpretation
        if cmf_values:
            latest_cmf = cmf_values[-1]["cmf"]
            
            if latest_cmf > self.param["strong_buying_threshold"]:
                result["interpretation"] = f"Strong buying pressure (CMF: {latest_cmf:.4f})"
            elif latest_cmf < self.param["strong_selling_threshold"]:
                result["interpretation"] = f"Strong selling pressure (CMF: {latest_cmf:.4f})"
            elif latest_cmf > 0:
                result["interpretation"] = f"Moderate buying pressure (CMF: {latest_cmf:.4f})"
            elif latest_cmf < 0:
                result["interpretation"] = f"Moderate selling pressure (CMF: {latest_cmf:.4f})"
            else:
                result["interpretation"] = f"Neutral (CMF: {latest_cmf:.4f})"
            
            result["latest_cmf"] = latest_cmf
        
        self.print_output(result)
        return result
    
    def print_output(self, result: dict):
        """Print the CMF analysis output"""
        print("\n" + "="*60)
        print(f"CHAIKIN MONEY FLOW ANALYSIS")
        print("="*60)
        print(f"Symbol: {result.get('symbol', 'N/A')}")
        print(f"Timeframe: {result.get('timeframe', 'N/A')}")
        print(f"CMF Period: {self.param['cmf_period']}")
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"Total Data Points: {len(self.ohlcv)}")
        print(f"CMF Calculations: {len(result.get('cmf_values', []))}")
        
        if result.get("cmf_values"):
            print(f"\nLatest CMF: {result.get('latest_cmf', 0):.6f}")
            print(f"Interpretation: {result.get('interpretation', 'N/A')}")
            
            print(f"\nLast 5 CMF Values:")
            for cmf_data in result["cmf_values"][-5:]:
                timestamp = cmf_data["timestamp"]
                cmf_val = cmf_data["cmf"]
                print(f"  {timestamp}: {cmf_val:.6f}")
        
        print(f"\nCMF Interpretation Guide:")
        print(f"  > {self.param['strong_buying_threshold']}: Strong buying pressure")
        print(f"  0 to {self.param['strong_buying_threshold']}: Moderate buying pressure")
        print(f"  {self.param['strong_selling_threshold']} to 0: Moderate selling pressure")
        print(f"  < {self.param['strong_selling_threshold']}: Strong selling pressure")