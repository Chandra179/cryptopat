"""
Cumulative Volume Delta (CVD) calculation module.
Analyzes order flow to detect aggressive buying vs selling pressure.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class CVDStrategy:
    """Cumulative Volume Delta analyzer for order flow analysis."""
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,             
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.rules = {
            "min_trade_volume": 0.1,  # Ignore micro trades (dust)
            "max_trade_volume": None, # No cap unless filtering anomalies
            "cvd_time_window_sec": 60,
            "cvd_side_inference": True,
            "cvd_smoothing_period": 3,  # EMA smoothing to reduce noise

            # Formula to infer trade side if not explicitly provided
            "infer_trade_side_formula": lambda price, best_bid, best_ask: (
                "buy" if price >= best_ask else "sell" if price <= best_bid else "unknown"
            ),

            # Formula for per-trade delta
            "trade_delta_formula": lambda volume, side: (
                volume if side == "buy" else -volume if side == "sell" else 0.0
            ),

            # Formula for updating cumulative volume delta
            "cvd_update_formula": lambda cvd_prev, trade_delta, side: cvd_prev + trade_delta,

            # Optional normalization
            "cvd_normalization_formula": lambda cvd_value, total_volume: (
                cvd_value / total_volume if total_volume and total_volume != 0 else 0.0
            ),
        }
        self.ob = ob
        self.ticker = ticker
        self.ohlcv = ohlcv
        self.trades = trades
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
    
    def calculate(self):
        if not self.trades:
            logger.warning(f"No trades data available for {self.symbol}")
            return {"cvd": 0.0, "total_volume": 0.0, "buy_volume": 0.0, "sell_volume": 0.0}
        
        # Get best bid/ask from order book
        best_bid = self.ob.get('bids', [[0]])[0][0] if self.ob.get('bids') else 0
        best_ask = self.ob.get('asks', [[0]])[0][0] if self.ob.get('asks') else 0
        
        cvd = 0.0
        total_volume = 0.0
        buy_volume = 0.0
        sell_volume = 0.0
        cvd_history = []
        
        for trade in self.trades:
            # Extract trade data
            price = float(trade.get('price', 0))
            volume = float(trade.get('amount', 0))
            
            # Filter out dust trades
            if volume < self.rules["min_trade_volume"]:
                continue
                
            # Filter out anomalously large trades if max volume is set
            if self.rules["max_trade_volume"] and volume > self.rules["max_trade_volume"]:
                continue
            
            # Infer trade side if not provided
            trade_side = trade.get('side')
            if not trade_side and self.rules["cvd_side_inference"]:
                trade_side = self.rules["infer_trade_side_formula"](price, best_bid, best_ask)
            
            # Calculate trade delta
            trade_delta = self.rules["trade_delta_formula"](volume, trade_side)
            
            # Update CVD
            cvd = self.rules["cvd_update_formula"](cvd, trade_delta, trade_side)
            
            # Track volume by side
            total_volume += volume
            if trade_side == "buy":
                buy_volume += volume
            elif trade_side == "sell":
                sell_volume += volume
                
            cvd_history.append({
                'timestamp': trade.get('timestamp'),
                'price': price,
                'volume': volume,
                'side': trade_side,
                'delta': trade_delta,
                'cvd': cvd
            })
        
        # Apply smoothing if enabled
        if self.rules["cvd_smoothing_period"] > 1 and len(cvd_history) > 1:
            cvd = self._apply_ema_smoothing(cvd_history, self.rules["cvd_smoothing_period"])
        
        # Normalize CVD if enabled
        normalized_cvd = self.rules["cvd_normalization_formula"](cvd, total_volume)
        
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "cvd": cvd,
            "normalized_cvd": normalized_cvd,
            "total_volume": total_volume,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_sell_ratio": buy_volume / sell_volume if sell_volume > 0 else float('inf'),
            "trade_count": len([t for t in self.trades if float(t.get('amount', 0)) >= self.rules["min_trade_volume"]]),
            "cvd_history": cvd_history[-10:] if cvd_history else []  # Last 10 trades
        }
        
        self.print_output(result)
        return result
    
    def _apply_ema_smoothing(self, cvd_history: List[Dict], period: int) -> float:
        """Apply EMA smoothing to CVD values."""
        if not cvd_history or len(cvd_history) < 2:
            return cvd_history[-1]['cvd'] if cvd_history else 0.0
            
        alpha = 2.0 / (period + 1)
        ema = cvd_history[0]['cvd']
        
        for i in range(1, len(cvd_history)):
            ema = alpha * cvd_history[i]['cvd'] + (1 - alpha) * ema
            
        return ema
    
    def print_output(self, result: dict):
        """Print CVD analysis results."""
        if not result:
            return
            
        print(f"\nCVD Analysis for {result.get('symbol', 'Unknown')} ({result.get('timeframe', 'Unknown')})")
        print(f"CVD: {result.get('cvd', 0):.4f}")
        print(f"Normalized CVD: {result.get('normalized_cvd', 0):.4f}")
        print(f"Total Volume: {result.get('total_volume', 0):.4f}")
        print(f"Buy Volume: {result.get('buy_volume', 0):.4f}")
        print(f"Sell Volume: {result.get('sell_volume', 0):.4f}")
        print(f"Buy/Sell Ratio: {result.get('buy_sell_ratio', 0):.4f}")
        print(f"Trade Count: {result.get('trade_count', 0)}")
        
        # Interpret CVD signal
        cvd_value = result.get('cvd', 0)
        if cvd_value > 0:
            signal = "BULLISH (More buying pressure)"
        elif cvd_value < 0:
            signal = "BEARISH (More selling pressure)"
        else:
            signal = "NEUTRAL (Balanced flow)"
            
        print(f"CVD Signal: {signal}")
