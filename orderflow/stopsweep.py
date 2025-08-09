from typing import List, Dict

class StopSweep:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.rules = {
            "min_volume_spike_ratio": 2.0,
            "min_price_deviation": 0.002,
            "liquidity_threshold": 50000.0,
            "sweep_confirmation_trades": 5,
            "price_movement_formula": lambda price_before, price_after: abs(
                (price_after - price_before) / price_before
            ),
            "volume_spike_formula": lambda current_vol, avg_vol: current_vol / avg_vol if avg_vol > 0 else 0
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
        Calculate Stop Run & Liquidity Sweep Detection according to TradingView methodology.
        """
        result = {
            "stop_runs": [],
            "liquidity_sweeps": [],
            "analysis_summary": {}
        }
        
        if len(self.ohlcv) < 20:
            result["analysis_summary"]["error"] = "Insufficient data for analysis"
            self.print_output(result)
            return
        
        # Calculate average volume for comparison
        volumes = [candle[5] for candle in self.ohlcv[-20:]]
        avg_volume = sum(volumes) / len(volumes)
        
        # Detect stop runs and liquidity sweeps
        stop_runs = self._detect_stop_runs(avg_volume)
        liquidity_sweeps = self._detect_liquidity_sweeps()
        
        result["stop_runs"] = stop_runs
        result["liquidity_sweeps"] = liquidity_sweeps
        result["analysis_summary"] = {
            "total_stop_runs": len(stop_runs),
            "total_liquidity_sweeps": len(liquidity_sweeps),
            "avg_volume_baseline": avg_volume,
            "current_bid_ask_spread": self._calculate_spread(),
            "market_conditions": self._assess_market_conditions()
        }
        
        self.print_output(result)
    
    def _detect_stop_runs(self, avg_volume):
        """Detect potential stop runs based on volume spikes and price movements"""
        stop_runs = []
        
        for i in range(len(self.ohlcv) - 1):
            current_candle = self.ohlcv[i]
            next_candle = self.ohlcv[i + 1] if i + 1 < len(self.ohlcv) else None
            
            if not next_candle:
                continue
            
            timestamp, open_price, high, low, close, volume = current_candle
            next_open, next_high, next_low, next_close = next_candle[1:5]
            
            # Check for volume spike
            volume_ratio = self.rules["volume_spike_formula"](volume, avg_volume)
            
            # Check for significant price movement beyond normal range
            price_deviation = self.rules["price_movement_formula"](close, next_open)
            
            # Detect stop run conditions
            if (volume_ratio >= self.rules["min_volume_spike_ratio"] and 
                price_deviation >= self.rules["min_price_deviation"]):
                
                # Check if price breaks key levels then reverses
                breakout_high = max(high, next_high)
                breakout_low = min(low, next_low)
                
                # Potential stop run if price spikes then reverses
                if ((next_high > high and next_close < high) or 
                    (next_low < low and next_close > low)):
                    
                    stop_runs.append({
                        "timestamp": timestamp,
                        "type": "bullish_stop_run" if next_low < low else "bearish_stop_run",
                        "trigger_price": breakout_low if next_low < low else breakout_high,
                        "volume_spike": volume_ratio,
                        "price_deviation": price_deviation,
                        "reversal_close": next_close
                    })
        
        return stop_runs
    
    def _detect_liquidity_sweeps(self):
        """Detect liquidity sweeps using order book and trade data"""
        sweeps = []
        
        if not self.trades or len(self.trades) < self.rules["sweep_confirmation_trades"]:
            return sweeps
        
        # Analyze recent large trades that might indicate liquidity sweeps
        large_trades = [trade for trade in self.trades 
                       if trade.get('cost', 0) >= self.rules["liquidity_threshold"]]
        
        # Group trades by time proximity and analyze for sweep patterns
        for i, trade in enumerate(large_trades):
            if i < len(large_trades) - 1:
                next_trade = large_trades[i + 1]
                
                # Check if consecutive large trades move price significantly
                price_diff = abs(trade['price'] - next_trade['price'])
                price_impact = price_diff / trade['price']
                
                if price_impact >= self.rules["min_price_deviation"]:
                    sweeps.append({
                        "timestamp": trade['timestamp'],
                        "sweep_type": "buy_sweep" if trade['side'] == 'buy' else "sell_sweep",
                        "price": trade['price'],
                        "volume": trade['amount'],
                        "cost": trade['cost'],
                        "price_impact": price_impact
                    })
        
        return sweeps
    
    def _calculate_spread(self):
        """Calculate current bid-ask spread from order book"""
        if not self.ob or 'bids' not in self.ob or 'asks' not in self.ob:
            return 0
        
        if not self.ob['bids'] or not self.ob['asks']:
            return 0
        
        best_bid = self.ob['bids'][0][0] if self.ob['bids'] else 0
        best_ask = self.ob['asks'][0][0] if self.ob['asks'] else 0
        
        if best_bid > 0 and best_ask > 0:
            return (best_ask - best_bid) / best_bid
        
        return 0
    
    def _assess_market_conditions(self):
        """Assess current market conditions based on recent price action"""
        if len(self.ohlcv) < 5:
            return "insufficient_data"
        
        recent_candles = self.ohlcv[-5:]
        closes = [candle[4] for candle in recent_candles]
        
        # Simple trend assessment
        if closes[-1] > closes[0]:
            return "bullish_trend"
        elif closes[-1] < closes[0]:
            return "bearish_trend"
        else:
            return "sideways"
    
    def print_output(self, result: dict):
        """Print the output"""
        print("\n" + "="*50)
        print(f"STOP RUN & LIQUIDITY SWEEP ANALYSIS")
        print("="*50)
        print(f"Symbol: {self.symbol} | Timeframe: {self.timeframe}")

        if "error" in result["analysis_summary"]:
            print(f"Error: {result['analysis_summary']['error']}")
            return
        
        summary = result["analysis_summary"]
        print(f"Total Stop Runs Detected: {summary['total_stop_runs']}")
        print(f"Total Liquidity Sweeps: {summary['total_liquidity_sweeps']}")
        print(f"Average Volume Baseline: {summary['avg_volume_baseline']:.2f}")
        print(f"Current Bid-Ask Spread: {summary['current_bid_ask_spread']:.4f}%")
        print(f"Market Conditions: {summary['market_conditions']}")
        
        if result["stop_runs"]:
            print(f"\nSTOP RUNS:")
            for i, stop_run in enumerate(result["stop_runs"][-3:], 1):
                print(f"  {i}. Type: {stop_run['type']}")
                print(f"     Trigger Price: ${stop_run['trigger_price']:.4f}")
                print(f"     Volume Spike: {stop_run['volume_spike']:.2f}x")
                print(f"     Price Deviation: {stop_run['price_deviation']:.4f}")
        
        if result["liquidity_sweeps"]:
            print(f"\nLIQUIDITY SWEEPS:")
            for i, sweep in enumerate(result["liquidity_sweeps"][-3:], 1):
                print(f"  {i}. Type: {sweep['sweep_type']}")
                print(f"     Price: ${sweep['price']:.4f}")
                print(f"     Volume: {sweep['volume']:.4f}")
                print(f"     Price Impact: {sweep['price_impact']:.4f}")