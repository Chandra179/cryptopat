"""
Volume Footprint Chart analysis strategy for order flow analysis.
Analyzes the distribution of volume at different price levels to identify
buying and selling pressure, support/resistance levels, and market imbalances.
"""

from typing import List, Dict
import pandas as pd


class VolumeFootprint:
    
    def __init__(self, 
             symbol: str,
             timeframe: str,
             limit: int,
             ob: dict,
             ticker: dict,            
             ohlcv: List[List],       
             trades: List[Dict]):    
        self.rules = {
            "min_volume_threshold": 0.1,  # Minimum volume to consider significant
            "price_level_precision": 4,   # Decimal places for price level grouping
            "imbalance_ratio": 2.0,       # Ratio to identify buy/sell imbalances
            "volume_cluster_min": 0.05,   # Minimum % of total volume to be a cluster
            "poc_calculation": "volume",   # Point of Control calculation method
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
        Calculate Volume Footprint Chart according to TradingView methodology.
        """
        if not self.trades:
            print("No trades data available for footprint analysis")
            return {}
            
        # Convert trades to DataFrame for easier manipulation
        trades_df = pd.DataFrame(self.trades)
        
        # Group trades by price levels
        price_levels = self._group_by_price_levels(trades_df)
        
        # Calculate buy/sell volume at each price level
        footprint_data = self._calculate_buy_sell_volume(price_levels)
        
        # Identify key levels
        poc = self._find_point_of_control(footprint_data)
        value_area = self._calculate_value_area(footprint_data)
        imbalances = self._find_volume_imbalances(footprint_data)
        clusters = self._identify_volume_clusters(footprint_data)
        
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "footprint_data": footprint_data,
            "point_of_control": poc,
            "value_area": value_area,
            "volume_imbalances": imbalances,
            "volume_clusters": clusters,
            "total_volume": sum([level["total_volume"] for level in footprint_data.values()]),
            "total_trades": len(self.trades)
        }
        
        self.print_output(result)
        return result
    
    def _group_by_price_levels(self, trades_df: pd.DataFrame) -> Dict:
        """Group trades by rounded price levels"""
        precision = self.rules["price_level_precision"]
        trades_df['price_level'] = trades_df['price'].round(precision)
        return trades_df.groupby('price_level')
    
    def _calculate_buy_sell_volume(self, price_groups) -> Dict:
        """Calculate buy and sell volume for each price level"""
        footprint_data = {}
        
        for price_level, group in price_groups:
            buy_volume = group[group['side'] == 'buy']['amount'].sum()
            sell_volume = group[group['side'] == 'sell']['amount'].sum()
            total_volume = buy_volume + sell_volume
            
            if total_volume >= self.rules["min_volume_threshold"]:
                footprint_data[price_level] = {
                    "buy_volume": buy_volume,
                    "sell_volume": sell_volume,
                    "total_volume": total_volume,
                    "trade_count": len(group),
                    "buy_trades": len(group[group['side'] == 'buy']),
                    "sell_trades": len(group[group['side'] == 'sell']),
                    "volume_delta": buy_volume - sell_volume,
                    "buy_sell_ratio": buy_volume / sell_volume if sell_volume > 0 else float('inf')
                }
        
        return footprint_data
    
    def _find_point_of_control(self, footprint_data: Dict) -> Dict:
        """Find the Point of Control (price level with highest volume)"""
        if not footprint_data:
            return {}
            
        max_volume = 0
        poc_price = None
        
        for price_level, data in footprint_data.items():
            if data["total_volume"] > max_volume:
                max_volume = data["total_volume"]
                poc_price = price_level
        
        return {
            "price": poc_price,
            "volume": max_volume,
            "data": footprint_data.get(poc_price, {})
        }
    
    def _calculate_value_area(self, footprint_data: Dict, percentage: float = 0.68) -> Dict:
        """Calculate Value Area (price range containing specified % of volume)"""
        if not footprint_data:
            return {}
            
        # Sort price levels by volume (descending)
        sorted_levels = sorted(footprint_data.items(), key=lambda x: x[1]["total_volume"], reverse=True)
        
        total_volume = sum([data["total_volume"] for data in footprint_data.values()])
        target_volume = total_volume * percentage
        
        value_area_volume = 0
        value_area_prices = []
        
        for price_level, data in sorted_levels:
            value_area_volume += data["total_volume"]
            value_area_prices.append(price_level)
            
            if value_area_volume >= target_volume:
                break
        
        return {
            "high": max(value_area_prices) if value_area_prices else None,
            "low": min(value_area_prices) if value_area_prices else None,
            "volume": value_area_volume,
            "percentage_of_total": (value_area_volume / total_volume) * 100 if total_volume > 0 else 0,
            "price_levels": sorted(value_area_prices)
        }
    
    def _find_volume_imbalances(self, footprint_data: Dict) -> List[Dict]:
        """Identify price levels with significant buy/sell imbalances"""
        imbalances = []
        imbalance_ratio = self.rules["imbalance_ratio"]
        
        for price_level, data in footprint_data.items():
            ratio = data["buy_sell_ratio"]
            
            if ratio >= imbalance_ratio:
                imbalances.append({
                    "price": price_level,
                    "type": "buy_imbalance",
                    "ratio": ratio,
                    "buy_volume": data["buy_volume"],
                    "sell_volume": data["sell_volume"],
                    "strength": "strong" if ratio >= imbalance_ratio * 2 else "moderate"
                })
            elif data["sell_volume"] > 0 and data["buy_volume"] / data["sell_volume"] <= 1 / imbalance_ratio:
                imbalances.append({
                    "price": price_level,
                    "type": "sell_imbalance",
                    "ratio": 1 / ratio if ratio > 0 else float('inf'),
                    "buy_volume": data["buy_volume"],
                    "sell_volume": data["sell_volume"],
                    "strength": "strong" if ratio <= 1 / (imbalance_ratio * 2) else "moderate"
                })
        
        return sorted(imbalances, key=lambda x: x["ratio"], reverse=True)
    
    def _identify_volume_clusters(self, footprint_data: Dict) -> List[Dict]:
        """Identify significant volume clusters"""
        if not footprint_data:
            return []
            
        total_volume = sum([data["total_volume"] for data in footprint_data.values()])
        min_cluster_volume = total_volume * self.rules["volume_cluster_min"]
        
        clusters = []
        for price_level, data in footprint_data.items():
            if data["total_volume"] >= min_cluster_volume:
                clusters.append({
                    "price": price_level,
                    "volume": data["total_volume"],
                    "percentage_of_total": (data["total_volume"] / total_volume) * 100,
                    "buy_percentage": (data["buy_volume"] / data["total_volume"]) * 100,
                    "sell_percentage": (data["sell_volume"] / data["total_volume"]) * 100,
                    "trade_count": data["trade_count"]
                })
        
        return sorted(clusters, key=lambda x: x["volume"], reverse=True)
    
    def print_output(self, result: dict):
        """Print the footprint analysis output"""
        print("\n" + "="*50)
        print(f"Volume Footprint Analysis")
        print("="*50)
        print(f"Total Volume: {result['total_volume']:.4f}")
        print(f"Total Trades: {result['total_trades']}")
        
        # Point of Control
        poc = result['point_of_control']
        if poc:
            print(f"\nPoint of Control:")
            print(f"  Price: {poc['price']}")
            print(f"  Volume: {poc['volume']:.4f}")
            if poc['data']:
                print(f"  Buy Volume: {poc['data']['buy_volume']:.4f}")
                print(f"  Sell Volume: {poc['data']['sell_volume']:.4f}")
                print(f"  Delta: {poc['data']['volume_delta']:.4f}")
        
        # Value Area
        va = result['value_area']
        if va and va['high'] is not None:
            print(f"\nValue Area (68% of volume):")
            print(f"  High: {va['high']}")
            print(f"  Low: {va['low']}")
            print(f"  Volume: {va['volume']:.4f} ({va['percentage_of_total']:.1f}%)")
        
        # Volume Imbalances
        imbalances = result['volume_imbalances']
        if imbalances:
            print(f"\nTop Volume Imbalances:")
            for i, imb in enumerate(imbalances[:5], 1):
                print(f"  {i}. {imb['type']} at {imb['price']} (ratio: {imb['ratio']:.2f}, {imb['strength']})")
        
        # Volume Clusters
        clusters = result['volume_clusters']
        if clusters:
            print(f"\nTop Volume Clusters:")
            for i, cluster in enumerate(clusters[:5], 1):
                print(f"  {i}. Price: {cluster['price']}, Volume: {cluster['volume']:.4f} ({cluster['percentage_of_total']:.1f}%)")