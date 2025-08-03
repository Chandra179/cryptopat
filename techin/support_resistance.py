"""
Support and Resistance Zone Detection

Implements comprehensive S/R analysis using:
1. Price clusters & historical touchpoints
2. Volume profile integration  
3. Order book analysis
4. Fibonacci zones
5. Swing highs/lows with buffers
6. Moving average confluences

Returns zones (blocks) instead of single lines for actionable trading levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from data import get_data_collector


class SupportResistanceZones:
    """
    Advanced Support/Resistance zone detection using multiple confluence factors.
    Implements zone-based approach as specified in logs.txt requirements.
    """
    
    def __init__(self, zone_buffer_pct: float = 0.5):
        """
        Initialize S/R zone detector.
        
        Args:
            zone_buffer_pct: Percentage buffer around key levels to form zones
        """
        self.data_collector = get_data_collector()
        self.zone_buffer = zone_buffer_pct / 100  # Convert to decimal
        
    def analyze(self, symbol: str, timeframe: str, limit: int = 200) -> Dict[str, Any]:
        """
        Comprehensive S/R zone analysis.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Chart timeframe (e.g., '1h', '4h', '1d')
            limit: Number of candles to analyze
            
        Returns:
            Dict containing zone analysis results
        """
        try:
            # Fetch OHLCV data
            ohlcv_data = self.data_collector.fetch_ohlcv_data(symbol, timeframe, limit)
            if not ohlcv_data:
                return {'success': False, 'error': 'Failed to fetch OHLCV data'}
                
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Fetch order book for liquidity analysis
            order_book = self.data_collector.fetch_order_book(symbol, limit=50)
            
            # Get current ticker
            ticker = self.data_collector.fetch_ticker(symbol)
            current_price = ticker.get('last', df['close'].iloc[-1])
            
            # 1. Price cluster detection
            price_clusters = self._detect_price_clusters(df)
            
            # 2. Volume profile zones
            volume_zones = self._calculate_volume_profile_zones(df)
            
            # 3. Order book liquidity zones
            liquidity_zones = self._analyze_order_book_zones(order_book, current_price)
            
            # 4. Fibonacci zones
            fibonacci_zones = self._calculate_fibonacci_zones(df)
            
            # 5. Swing high/low zones with buffers
            swing_zones = self._detect_swing_zones(df)
            
            # 6. Moving average confluence zones
            ma_zones = self._calculate_ma_confluence_zones(df)
            
            # Combine all zones and calculate strength scores
            all_zones = self._combine_and_score_zones(
                price_clusters, volume_zones, liquidity_zones,
                fibonacci_zones, swing_zones, ma_zones, current_price=current_price
            )
            
            # Filter and rank zones
            significant_zones = self._filter_significant_zones(all_zones, current_price)
            
            # Generate trading signals based on zones
            signals = self._generate_zone_signals(significant_zones, current_price, df)
            
            return {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': round(current_price, 6),
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'support_zones': [z for z in significant_zones if z['type'] == 'support'],
                'resistance_zones': [z for z in significant_zones if z['type'] == 'resistance'],
                'zone_count': len(significant_zones),
                'signals': signals,
                'zone_methodology': {
                    'price_clusters': len(price_clusters),
                    'volume_zones': len(volume_zones),
                    'liquidity_zones': len(liquidity_zones),
                    'fibonacci_zones': len(fibonacci_zones),
                    'swing_zones': len(swing_zones),
                    'ma_confluence_zones': len(ma_zones)
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Analysis failed: {str(e)}'}
    
    def _detect_price_clusters(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect price clusters where multiple candles touch/bounce.
        Implementation of requirement #1 from logs.txt.
        """
        clusters = []
        
        # Combine wicks and bodies for cluster detection
        all_prices = []
        for _, row in df.iterrows():
            all_prices.extend([row['high'], row['low'], row['open'], row['close']])
            
        # Create price bins for clustering
        price_range = max(all_prices) - min(all_prices)
        bin_size = price_range * 0.005  # 0.5% bins
        
        price_counts = {}
        for price in all_prices:
            bin_key = round(price / bin_size) * bin_size
            price_counts[bin_key] = price_counts.get(bin_key, 0) + 1
        
        # Find significant clusters (more than 3 touches)
        for price_level, count in price_counts.items():
            if count >= 4:  # Minimum 4 touches for a cluster
                clusters.append({
                    'center_price': price_level,
                    'upper_bound': price_level * (1 + self.zone_buffer),
                    'lower_bound': price_level * (1 - self.zone_buffer),
                    'touch_count': count,
                    'type': 'cluster',
                    'strength': min(count / 2, 5.0)  # Cap strength at 5.0
                })
        
        return clusters
    
    def _calculate_volume_profile_zones(self, df: pd.DataFrame) -> List[Dict]:
        """
        Calculate Volume at Price (VAP) zones.
        Implementation of requirement #2 from logs.txt.
        """
        zones = []
        
        # Create price-volume profile
        price_volume_map = {}
        
        for _, row in df.iterrows():
            # Distribute volume across OHLC prices
            avg_price = (row['open'] + row['high'] + row['low'] + row['close']) / 4
            volume = row['volume']
            
            # Create price bins
            price_range = row['high'] - row['low']
            if price_range > 0:
                bin_size = price_range / 10  # Divide range into 10 bins
                for i in range(10):
                    bin_price = row['low'] + (i * bin_size)
                    price_volume_map[bin_price] = price_volume_map.get(bin_price, 0) + (volume / 10)
            else:
                price_volume_map[avg_price] = price_volume_map.get(avg_price, 0) + volume
        
        # Find high volume zones
        avg_volume = np.mean(list(price_volume_map.values()))
        threshold = avg_volume * 1.5  # 1.5x average volume
        
        for price, volume in price_volume_map.items():
            if volume >= threshold:
                zones.append({
                    'center_price': price,
                    'upper_bound': price * (1 + self.zone_buffer),
                    'lower_bound': price * (1 - self.zone_buffer),
                    'volume': volume,
                    'type': 'volume_profile',
                    'strength': min(volume / avg_volume, 5.0)
                })
        
        return zones
    
    def _analyze_order_book_zones(self, order_book: Dict, current_price: float) -> List[Dict]:
        """
        Analyze order book for liquidity pools.
        Implementation of requirement #3 from logs.txt.
        """
        zones = []
        
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return zones
        
        # Analyze bid clusters (support zones)
        bids = order_book['bids']
        if bids:
            # Group nearby bids
            bid_clusters = self._cluster_orders(bids, 'bid')
            for cluster in bid_clusters:
                zones.append({
                    'center_price': cluster['price'],
                    'upper_bound': cluster['price'] * (1 + self.zone_buffer),
                    'lower_bound': cluster['price'] * (1 - self.zone_buffer),
                    'liquidity': cluster['volume'],
                    'type': 'liquidity_support',
                    'strength': min(cluster['volume'] / 100, 3.0)  # Scale liquidity strength
                })
        
        # Analyze ask clusters (resistance zones)
        asks = order_book['asks']
        if asks:
            ask_clusters = self._cluster_orders(asks, 'ask')
            for cluster in ask_clusters:
                zones.append({
                    'center_price': cluster['price'],
                    'upper_bound': cluster['price'] * (1 + self.zone_buffer),
                    'lower_bound': cluster['price'] * (1 - self.zone_buffer),
                    'liquidity': cluster['volume'],
                    'type': 'liquidity_resistance',
                    'strength': min(cluster['volume'] / 100, 3.0)
                })
        
        return zones
    
    def _cluster_orders(self, orders: List, order_type: str) -> List[Dict]:
        """Helper to cluster nearby orders in order book."""
        if not orders:
            return []
            
        clusters = []
        sorted_orders = sorted(orders, key=lambda x: x[0])  # Sort by price
        
        current_cluster = {'price': sorted_orders[0][0], 'volume': sorted_orders[0][1]}
        price_threshold = sorted_orders[0][0] * 0.002  # 0.2% clustering threshold
        
        for price, volume in sorted_orders[1:]:
            if abs(price - current_cluster['price']) <= price_threshold:
                # Add to current cluster
                total_volume = current_cluster['volume'] + volume
                weighted_price = (current_cluster['price'] * current_cluster['volume'] + price * volume) / total_volume
                current_cluster = {'price': weighted_price, 'volume': total_volume}
            else:
                # Save current cluster if significant
                if current_cluster['volume'] > 50:  # Minimum volume threshold
                    clusters.append(current_cluster)
                # Start new cluster
                current_cluster = {'price': price, 'volume': volume}
                price_threshold = price * 0.002
        
        # Don't forget the last cluster
        if current_cluster['volume'] > 50:
            clusters.append(current_cluster)
            
        return clusters
    
    def _calculate_fibonacci_zones(self, df: pd.DataFrame) -> List[Dict]:
        """
        Calculate Fibonacci retracement zones from major swings.
        Implementation of requirement #4 from logs.txt.
        """
        zones = []
        
        # Find major swing highs and lows
        swings = self._find_major_swings(df)
        if len(swings) < 2:
            return zones
        
        # Calculate Fibonacci levels for recent major swings
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for i in range(len(swings) - 1):
            swing_high = swings[i]
            swing_low = swings[i + 1]
            
            if swing_high['type'] == 'high' and swing_low['type'] == 'low':
                price_range = swing_high['price'] - swing_low['price']
                
                for level in fib_levels:
                    fib_price = swing_low['price'] + (price_range * level)
                    zones.append({
                        'center_price': fib_price,
                        'upper_bound': fib_price * (1 + self.zone_buffer),
                        'lower_bound': fib_price * (1 - self.zone_buffer),
                        'fibonacci_level': level,
                        'type': 'fibonacci',
                        'strength': 2.0 + (0.618 - abs(level - 0.618)) * 2  # Stronger near golden ratio
                    })
        
        return zones
    
    def _find_major_swings(self, df: pd.DataFrame, lookback: int = 20) -> List[Dict]:
        """Find major swing highs and lows."""
        swings = []
        
        for i in range(lookback, len(df) - lookback):
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            # Check for swing high
            is_swing_high = all(current_high >= df.iloc[j]['high'] for j in range(i - lookback, i + lookback + 1) if j != i)
            if is_swing_high:
                swings.append({'price': current_high, 'index': i, 'type': 'high'})
            
            # Check for swing low
            is_swing_low = all(current_low <= df.iloc[j]['low'] for j in range(i - lookback, i + lookback + 1) if j != i)
            if is_swing_low:
                swings.append({'price': current_low, 'index': i, 'type': 'low'})
        
        return sorted(swings, key=lambda x: x['index'])
    
    def _detect_swing_zones(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect swing high/low zones with buffers.
        Implementation of requirement #5 from logs.txt.
        """
        zones = []
        swings = self._find_major_swings(df)
        
        for swing in swings:
            zone_type = 'resistance' if swing['type'] == 'high' else 'support'
            zones.append({
                'center_price': swing['price'],
                'upper_bound': swing['price'] * (1 + self.zone_buffer),
                'lower_bound': swing['price'] * (1 - self.zone_buffer),
                'swing_type': swing['type'],
                'type': f'swing_{zone_type}',
                'strength': 2.5
            })
        
        return zones
    
    def _calculate_ma_confluence_zones(self, df: pd.DataFrame) -> List[Dict]:
        """
        Calculate moving average confluence zones.
        Implementation of requirement #6 from logs.txt.
        """
        zones = []
        
        # Calculate popular MAs using pandas rolling mean
        ma_periods = [20, 50, 100, 200]
        mas = {}
        
        for period in ma_periods:
            if len(df) >= period:
                mas[period] = df['close'].rolling(window=period).mean().values
        
        if not mas:
            return zones
        
        # Find MA confluences (where multiple MAs cluster)
        latest_idx = len(df) - 1
        ma_values = []
        
        for period, ma_array in mas.items():
            if not np.isnan(ma_array[latest_idx]):
                ma_values.append(ma_array[latest_idx])
        
        if len(ma_values) >= 2:
            # Check for MA clustering
            ma_values.sort()
            
            for i in range(len(ma_values) - 1):
                price_diff_pct = abs(ma_values[i+1] - ma_values[i]) / ma_values[i]
                
                if price_diff_pct <= 0.01:  # MAs within 1% = confluence
                    avg_ma = (ma_values[i] + ma_values[i+1]) / 2
                    zones.append({
                        'center_price': avg_ma,
                        'upper_bound': avg_ma * (1 + self.zone_buffer),
                        'lower_bound': avg_ma * (1 - self.zone_buffer),
                        'ma_confluence': True,
                        'type': 'ma_confluence',
                        'strength': 2.0
                    })
        
        return zones
    
    def _combine_and_score_zones(self, *zone_lists, current_price: float) -> List[Dict]:
        """Combine all zone types and calculate composite strength scores."""
        all_zones = []
        
        # Flatten all zone lists
        for zone_list in zone_lists[:-1]:  # Exclude current_price
            all_zones.extend(zone_list)
        
        # Merge overlapping zones and boost strength
        merged_zones = []
        
        for zone in all_zones:
            merged = False
            
            for existing_zone in merged_zones:
                # Check for overlap (more strict criteria)
                overlap_threshold = 0.3  # 30% overlap required
                zone_range = zone['upper_bound'] - zone['lower_bound']
                existing_range = existing_zone['upper_bound'] - existing_zone['lower_bound']
                
                overlap_start = max(zone['lower_bound'], existing_zone['lower_bound'])
                overlap_end = min(zone['upper_bound'], existing_zone['upper_bound'])
                overlap_size = max(0, overlap_end - overlap_start)
                
                zone_overlap_pct = overlap_size / zone_range if zone_range > 0 else 0
                existing_overlap_pct = overlap_size / existing_range if existing_range > 0 else 0
                
                if zone_overlap_pct >= overlap_threshold or existing_overlap_pct >= overlap_threshold:
                    # Merge zones with weighted average
                    total_strength = existing_zone['strength'] + zone['strength']
                    weight1 = existing_zone['strength'] / total_strength
                    weight2 = zone['strength'] / total_strength
                    
                    new_center = (existing_zone['center_price'] * weight1 + zone['center_price'] * weight2)
                    
                    # Collect unique zone types
                    existing_types = set(existing_zone.get('merged_types', [existing_zone['type']]))
                    new_types = existing_types.union({zone['type']})
                    
                    existing_zone.update({
                        'center_price': new_center,
                        'upper_bound': max(existing_zone['upper_bound'], zone['upper_bound']),
                        'lower_bound': min(existing_zone['lower_bound'], zone['lower_bound']),
                        'strength': min(total_strength, 10.0),  # Cap at 10.0
                        'confluence_count': len(new_types),
                        'merged_types': list(new_types)
                    })
                    merged = True
                    break
            
            if not merged:
                zone['confluence_count'] = 1
                zone['merged_types'] = [zone['type']]
                merged_zones.append(zone)
        
        # Determine support/resistance type based on current price
        for zone in merged_zones:
            if zone['center_price'] < current_price:
                zone['type'] = 'support'
            else:
                zone['type'] = 'resistance'
        
        return merged_zones
    
    def _filter_significant_zones(self, zones: List[Dict], current_price: float) -> List[Dict]:
        """Filter and rank zones by significance."""
        # Filter by minimum strength
        significant_zones = [z for z in zones if z['strength'] >= 1.5]
        
        # Sort by strength (descending)
        significant_zones.sort(key=lambda x: x['strength'], reverse=True)
        
        # Limit to top zones
        return significant_zones[:10]
    
    def _generate_zone_signals(self, zones: List[Dict], current_price: float, df: pd.DataFrame) -> Dict:
        """Generate trading signals based on zone analysis."""
        signals = {
            'signal': 'HOLD',
            'confidence': 0.0,
            'reason': '',
            'nearest_support': None,
            'nearest_resistance': None,
            'zone_analysis': ''
        }
        
        support_zones = [z for z in zones if z['type'] == 'support']
        resistance_zones = [z for z in zones if z['type'] == 'resistance']
        
        # Find nearest zones
        if support_zones:
            nearest_support = max(support_zones, key=lambda x: x['center_price'])
            signals['nearest_support'] = {
                'price': round(nearest_support['center_price'], 6),
                'strength': round(nearest_support['strength'], 2),
                'distance_pct': round(((current_price - nearest_support['center_price']) / current_price) * 100, 2)
            }
        
        if resistance_zones:
            nearest_resistance = min(resistance_zones, key=lambda x: x['center_price'])
            signals['nearest_resistance'] = {
                'price': round(nearest_resistance['center_price'], 6),
                'strength': round(nearest_resistance['strength'], 2),
                'distance_pct': round(((nearest_resistance['center_price'] - current_price) / current_price) * 100, 2)
            }
        
        # Generate signal logic
        if signals['nearest_support'] and signals['nearest_resistance']:
            support_dist = abs(signals['nearest_support']['distance_pct'])
            resistance_dist = abs(signals['nearest_resistance']['distance_pct'])
            
            if support_dist < 2.0:  # Within 2% of strong support
                signals['signal'] = 'BUY'
                signals['confidence'] = min(nearest_support['strength'] / 5.0, 1.0) * 100
                signals['reason'] = f"Price near strong support zone (strength: {nearest_support['strength']:.1f})"
                
            elif resistance_dist < 2.0:  # Within 2% of strong resistance
                signals['signal'] = 'SELL'
                signals['confidence'] = min(nearest_resistance['strength'] / 5.0, 1.0) * 100
                signals['reason'] = f"Price near strong resistance zone (strength: {nearest_resistance['strength']:.1f})"
                
            else:
                signals['reason'] = f"Price between zones (S: {support_dist:.1f}% | R: {resistance_dist:.1f}%)"
        
        # Zone analysis summary
        strong_zones = len([z for z in zones if z['strength'] >= 3.0])
        signals['zone_analysis'] = f"Detected {len(zones)} significant zones ({strong_zones} strong)"
        
        return signals


def analyze_support_resistance(symbol: str, timeframe: str = '1h', limit: int = 200) -> Dict[str, Any]:
    """
    Convenience function for S/R zone analysis.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Chart timeframe (e.g., '1h', '4h', '1d')
        limit: Number of candles to analyze
        
    Returns:
        Dict containing complete S/R zone analysis
    """
    analyzer = SupportResistanceZones()
    return analyzer.analyze(symbol, timeframe, limit)


if __name__ == "__main__":
    import sys
    
    # Command line usage
    if len(sys.argv) >= 2:
        symbol = sys.argv[1]
        timeframe = sys.argv[2] if len(sys.argv) > 2 else '1h'
        limit = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        
        result = analyze_support_resistance(symbol, timeframe, limit)
        
        if result['success']:
            print(f"\n=== Support/Resistance Zone Analysis: {symbol} ({timeframe}) ===")
            print(f"Current Price: ${result['current_price']}")
            print(f"Analysis Time: {result['analysis_timestamp']}")
            print(f"\nZone Detection Summary:")
            print(f"- Total Significant Zones: {result['zone_count']}")
            print(f"- Support Zones: {len(result['support_zones'])}")
            print(f"- Resistance Zones: {len(result['resistance_zones'])}")
            
            print(f"\nMethodology Breakdown:")
            for method, count in result['zone_methodology'].items():
                print(f"- {method.replace('_', ' ').title()}: {count}")
            
            print(f"\n--- TRADING SIGNALS ---")
            signals = result['signals']
            print(f"Signal: {signals['signal']}")
            print(f"Confidence: {signals['confidence']:.1f}%")
            print(f"Reason: {signals['reason']}")
            print(f"Zone Analysis: {signals['zone_analysis']}")
            
            if signals['nearest_support']:
                sup = signals['nearest_support']
                print(f"\nNearest Support: ${sup['price']} (Strength: {sup['strength']}, Distance: {sup['distance_pct']:.2f}%)")
                
            if signals['nearest_resistance']:
                res = signals['nearest_resistance']
                print(f"Nearest Resistance: ${res['price']} (Strength: {res['strength']}, Distance: {res['distance_pct']:.2f}%)")
            
            print(f"\n--- TOP SUPPORT ZONES ---")
            for i, zone in enumerate(result['support_zones'][:3], 1):
                print(f"{i}. ${zone['lower_bound']:.6f} - ${zone['upper_bound']:.6f} (Center: ${zone['center_price']:.6f})")
                print(f"   Strength: {zone['strength']:.1f} | Confluence: {zone['confluence_count']} | Types: {', '.join(zone['merged_types'])}")
            
            print(f"\n--- TOP RESISTANCE ZONES ---")
            for i, zone in enumerate(result['resistance_zones'][:3], 1):
                print(f"{i}. ${zone['lower_bound']:.6f} - ${zone['upper_bound']:.6f} (Center: ${zone['center_price']:.6f})")
                print(f"   Strength: {zone['strength']:.1f} | Confluence: {zone['confluence_count']} | Types: {', '.join(zone['merged_types'])}")
                
        else:
            print(f"Error: {result['error']}")
    else:
        print("Usage: python support_resistance.py <symbol> [timeframe] [limit]")
        print("Example: python support_resistance.py BTC/USDT 1h 200")