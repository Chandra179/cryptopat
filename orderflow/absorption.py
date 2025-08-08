"""
Absorption Detection Module for CryptoPat.

This module implements absorption detection using historical data
to identify when large volumes are absorbed with minimal price movement,
indicating potential support/resistance levels.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class AbsorptionStrategy:
    """Absorption pattern detection strategy for cryptocurrency order flow analysis."""
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,        
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.rules = {
           "min_large_trade_volume": 1000.0,
            "max_price_movement": 0.001,
            "absorption_time_window_sec": 60,
            "min_liquidity_at_best_price": 5000.0,
            "price_movement_formula": lambda price_before, price_after: abs(
                (price_after - price_before) / price_before
            )
            if price_before and price_before != 0
            else float("inf"),
            "absorption_ratio_formula": lambda volume_absorbed, large_trade_volume: (
                volume_absorbed / large_trade_volume
                if large_trade_volume and large_trade_volume != 0
                else 0.0
            ),
            "min_absorption_ratio": 0.8,
            "max_levels_to_check": 10,
        }
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit

    def calculate(self):
        """Run absorption detection and return structured results."""
        result = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'absorption_events': [],
            'summary': {
                'total_events': 0,
                'avg_absorption_ratio': 0.0,
                'strongest_absorption': 0.0
            }
        }
        
        if not self.trades or len(self.trades) < 2:
            logger.warning("Insufficient trade data for absorption analysis")
            self.print_output(result)
            return result
            
        # Detect absorption events
        absorption_events = []
        absorption_ratios = []
        
        for i, trade in enumerate(self.trades[:-1]):
            if self._is_large_trade(trade):
                next_trade = self.trades[i + 1]
                
                # Check if within time window
                if self._within_time_window(trade, next_trade):
                    price_movement = self.rules["price_movement_formula"](
                        trade.get('price', 0), 
                        next_trade.get('price', 0)
                    )
                    
                    # Check for minimal price movement (absorption)
                    if price_movement <= self.rules["max_price_movement"]:
                        volume_absorbed = self._calculate_absorbed_volume(trade, next_trade)
                        absorption_ratio = self.rules["absorption_ratio_formula"](
                            volume_absorbed, 
                            trade.get('amount', 0)
                        )
                        
                        if absorption_ratio >= self.rules["min_absorption_ratio"]:
                            event = {
                                'timestamp': trade.get('timestamp', 0),
                                'price': trade.get('price', 0),
                                'large_trade_volume': trade.get('amount', 0),
                                'volume_absorbed': volume_absorbed,
                                'absorption_ratio': absorption_ratio,
                                'price_movement': price_movement
                            }
                            absorption_events.append(event)
                            absorption_ratios.append(absorption_ratio)
        
        result['absorption_events'] = absorption_events
        result['summary']['total_events'] = len(absorption_events)
        if absorption_ratios:
            result['summary']['avg_absorption_ratio'] = sum(absorption_ratios) / len(absorption_ratios)
            result['summary']['strongest_absorption'] = max(absorption_ratios)
        
        self.print_output(result)
        return result

    def print_output(self, result: dict):
        """Print a compact summary of detected absorption events."""
        print(f"\n=== ABSORPTION ANALYSIS: {result['symbol']} ({result['timeframe']}) ===")
        print(f"Total Absorption Events: {result['summary']['total_events']}")
        
        if result['summary']['total_events'] > 0:
            print(f"Average Absorption Ratio: {result['summary']['avg_absorption_ratio']:.3f}")
            print(f"Strongest Absorption: {result['summary']['strongest_absorption']:.3f}")
            print("\nTop Absorption Events:")
            
            # Sort events by absorption ratio
            sorted_events = sorted(result['absorption_events'], 
                                 key=lambda x: x['absorption_ratio'], reverse=True)[:5]
            
            for i, event in enumerate(sorted_events, 1):
                print(f"{i}. Price: ${event['price']:.4f} | "
                      f"Volume: {event['large_trade_volume']:.2f} | "
                      f"Absorption: {event['absorption_ratio']:.3f} | "
                      f"Price Movement: {event['price_movement']:.4f}%")
        else:
            print("No absorption events detected with current parameters.")
        print("=" * 60)

    def _is_large_trade(self, trade: Dict) -> bool:
        """Check if trade volume exceeds minimum threshold."""
        return trade.get('amount', 0) >= self.rules["min_large_trade_volume"]
    
    def _within_time_window(self, trade1: Dict, trade2: Dict) -> bool:
        """Check if trades are within absorption time window."""
        time_diff = abs(trade2.get('timestamp', 0) - trade1.get('timestamp', 0))
        return time_diff <= self.rules["absorption_time_window_sec"]
    
    def _calculate_absorbed_volume(self, trade1: Dict, trade2: Dict) -> float:
        """Calculate volume absorbed between two trades."""
        # Simplified calculation - in real implementation would use order book data
        return min(trade1.get('amount', 0), trade2.get('amount', 0))