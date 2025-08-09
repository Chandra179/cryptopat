"""
Smart Money Concepts (SMC) Analysis Strategy

SMC focuses on identifying institutional order flow through:
- Break of Structure (BOS) and Change of Character (CHoCH)
- Fair Value Gaps (FVG)
- Order Blocks (OB) - last supply/demand zone before price moves
- Liquidity sweeps and inducement
- Premium/Discount zones using Fibonacci levels
"""

from typing import List, Dict
import pandas as pd


class SmartMoneyConcepts:
    
    def __init__(self, 
                 symbol: str,
                 timeframe: str,
                 limit: int,
                 ob: dict,
                 ticker: dict,            
                 ohlcv: List[List],       
                 trades: List[Dict]):    
        self.rules = {
            "structure_break_threshold": 0.001,  # 0.1% minimum move for BOS
            "fvg_threshold": 0.0005,  # 0.05% minimum gap size
            "liquidity_wick_ratio": 0.6,  # Wick must be 60% of candle range
            "orderblock_lookback": 5,  # Look back 5 candles for order blocks
            "premium_threshold": 0.618,  # 61.8% Fibonacci level
            "discount_threshold": 0.382,  # 38.2% Fibonacci level
            "volume_multiplier": 1.5,  # Volume must be 1.5x average for significance
        }
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.ticker = ticker
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.df = self._create_dataframe()
    
    def _create_dataframe(self):
        """Convert OHLCV data to pandas DataFrame"""
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.sort_values('timestamp').reset_index(drop=True)
    
    def calculate(self):
        """Calculate Smart Money Concepts analysis"""
        result = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'analysis': 'Smart Money Concepts (SMC)',
            'market_structure': self._analyze_market_structure(),
            'fair_value_gaps': self._identify_fair_value_gaps(),
            'order_blocks': self._identify_order_blocks(),
            'liquidity_zones': self._identify_liquidity_zones(),
            'premium_discount': self._calculate_premium_discount(),
            'current_bias': self._determine_bias()
        }
        self.print_output(result)
        return result
    
    def _analyze_market_structure(self):
        """Identify Break of Structure (BOS) and Change of Character (CHoCH)"""
        structure = {
            'trend': 'sideways',
            'last_bos': None,
            'choch_signals': []
        }
        
        if len(self.df) < 10:
            return structure
            
        # Calculate swing highs and lows
        highs = []
        lows = []
        
        for i in range(2, len(self.df) - 2):
            # Swing high: higher than previous 2 and next 2 candles
            if (self.df.loc[i, 'high'] > self.df.loc[i-1, 'high'] and 
                self.df.loc[i, 'high'] > self.df.loc[i-2, 'high'] and
                self.df.loc[i, 'high'] > self.df.loc[i+1, 'high'] and
                self.df.loc[i, 'high'] > self.df.loc[i+2, 'high']):
                highs.append((i, self.df.loc[i, 'high']))
                
            # Swing low: lower than previous 2 and next 2 candles
            if (self.df.loc[i, 'low'] < self.df.loc[i-1, 'low'] and 
                self.df.loc[i, 'low'] < self.df.loc[i-2, 'low'] and
                self.df.loc[i, 'low'] < self.df.loc[i+1, 'low'] and
                self.df.loc[i, 'low'] < self.df.loc[i+2, 'low']):
                lows.append((i, self.df.loc[i, 'low']))
        
        # Determine trend based on structure
        if len(highs) >= 2 and len(lows) >= 2:
            recent_highs = highs[-2:]
            recent_lows = lows[-2:]
            
            if (recent_highs[1][1] > recent_highs[0][1] and 
                recent_lows[1][1] > recent_lows[0][1]):
                structure['trend'] = 'bullish'
            elif (recent_highs[1][1] < recent_highs[0][1] and 
                  recent_lows[1][1] < recent_lows[0][1]):
                structure['trend'] = 'bearish'
        
        return structure
    
    def _identify_fair_value_gaps(self):
        """Identify Fair Value Gaps (imbalances in price)"""
        gaps = []
        
        for i in range(1, len(self.df) - 1):
            # Bullish FVG: Gap between previous candle high and next candle low
            if (self.df.loc[i+1, 'low'] > self.df.loc[i-1, 'high'] and
                (self.df.loc[i+1, 'low'] - self.df.loc[i-1, 'high']) / self.df.loc[i, 'close'] > self.rules['fvg_threshold']):
                gaps.append({
                    'type': 'bullish',
                    'index': i,
                    'top': self.df.loc[i+1, 'low'],
                    'bottom': self.df.loc[i-1, 'high'],
                    'size': self.df.loc[i+1, 'low'] - self.df.loc[i-1, 'high']
                })
            
            # Bearish FVG: Gap between previous candle low and next candle high  
            elif (self.df.loc[i-1, 'low'] > self.df.loc[i+1, 'high'] and
                  (self.df.loc[i-1, 'low'] - self.df.loc[i+1, 'high']) / self.df.loc[i, 'close'] > self.rules['fvg_threshold']):
                gaps.append({
                    'type': 'bearish',
                    'index': i,
                    'top': self.df.loc[i-1, 'low'],
                    'bottom': self.df.loc[i+1, 'high'],
                    'size': self.df.loc[i-1, 'low'] - self.df.loc[i+1, 'high']
                })
        
        return gaps
    
    def _identify_order_blocks(self):
        """Identify Order Blocks - last opposite candle before strong move"""
        order_blocks = []
        avg_volume = self.df['volume'].rolling(10).mean()
        
        for i in range(self.rules['orderblock_lookback'], len(self.df)):
            current_volume = self.df.loc[i, 'volume']
            
            # High volume bullish candle
            if (current_volume > avg_volume.iloc[i] * self.rules['volume_multiplier'] and
                self.df.loc[i, 'close'] > self.df.loc[i, 'open']):
                
                # Look for last bearish candle before this move
                for j in range(i-1, max(0, i-self.rules['orderblock_lookback']), -1):
                    if self.df.loc[j, 'close'] < self.df.loc[j, 'open']:
                        order_blocks.append({
                            'type': 'bullish_ob',
                            'index': j,
                            'high': self.df.loc[j, 'high'],
                            'low': self.df.loc[j, 'low'],
                            'trigger_candle': i
                        })
                        break
            
            # High volume bearish candle
            elif (current_volume > avg_volume.iloc[i] * self.rules['volume_multiplier'] and
                  self.df.loc[i, 'close'] < self.df.loc[i, 'open']):
                
                # Look for last bullish candle before this move
                for j in range(i-1, max(0, i-self.rules['orderblock_lookback']), -1):
                    if self.df.loc[j, 'close'] > self.df.loc[j, 'open']:
                        order_blocks.append({
                            'type': 'bearish_ob',
                            'index': j,
                            'high': self.df.loc[j, 'high'],
                            'low': self.df.loc[j, 'low'],
                            'trigger_candle': i
                        })
                        break
        
        return order_blocks
    
    def _identify_liquidity_zones(self):
        """Identify liquidity sweep areas and equal highs/lows"""
        liquidity = {
            'equal_highs': [],
            'equal_lows': [],
            'potential_sweeps': []
        }
        
        # Find equal highs and lows (within 0.1% of each other)
        tolerance = 0.001
        
        for i in range(len(self.df) - 1):
            for j in range(i + 3, min(i + 20, len(self.df))):  # Look ahead 3-20 candles
                # Equal highs
                if abs(self.df.loc[i, 'high'] - self.df.loc[j, 'high']) / self.df.loc[i, 'high'] < tolerance:
                    liquidity['equal_highs'].append({
                        'price': (self.df.loc[i, 'high'] + self.df.loc[j, 'high']) / 2,
                        'indices': [i, j],
                        'strength': 2
                    })
                
                # Equal lows
                if abs(self.df.loc[i, 'low'] - self.df.loc[j, 'low']) / self.df.loc[i, 'low'] < tolerance:
                    liquidity['equal_lows'].append({
                        'price': (self.df.loc[i, 'low'] + self.df.loc[j, 'low']) / 2,
                        'indices': [i, j],
                        'strength': 2
                    })
        
        return liquidity
    
    def _calculate_premium_discount(self):
        """Calculate if price is in premium, discount, or equilibrium zone"""
        if len(self.df) < 20:
            return {'zone': 'insufficient_data'}
            
        # Use recent 20-period range
        recent_high = self.df['high'].tail(20).max()
        recent_low = self.df['low'].tail(20).min()
        current_price = self.df['close'].iloc[-1]
        
        range_size = recent_high - recent_low
        price_position = (current_price - recent_low) / range_size
        
        if price_position > self.rules['premium_threshold']:
            zone = 'premium'
        elif price_position < self.rules['discount_threshold']:
            zone = 'discount'
        else:
            zone = 'equilibrium'
            
        return {
            'zone': zone,
            'position_pct': price_position * 100,
            'range_high': recent_high,
            'range_low': recent_low,
            'current_price': current_price
        }
    
    def _determine_bias(self):
        """Determine overall market bias based on SMC factors"""
        structure = self._analyze_market_structure()
        premium_discount = self._calculate_premium_discount()
        
        bias_score = 0
        factors = []
        
        # Structure bias
        if structure['trend'] == 'bullish':
            bias_score += 2
            factors.append('Bullish structure')
        elif structure['trend'] == 'bearish':
            bias_score -= 2
            factors.append('Bearish structure')
        
        # Premium/Discount bias
        if premium_discount['zone'] == 'discount':
            bias_score += 1
            factors.append('In discount zone (bullish for continuation)')
        elif premium_discount['zone'] == 'premium':
            bias_score -= 1
            factors.append('In premium zone (bearish for continuation)')
        
        # Current orderbook bias
        if self.ob and 'bids' in self.ob and 'asks' in self.ob:
            bid_volume = sum([float(bid[1]) for bid in self.ob['bids'][:10]])
            ask_volume = sum([float(ask[1]) for ask in self.ob['asks'][:10]])
            
            if bid_volume > ask_volume * 1.2:
                bias_score += 1
                factors.append('Strong bid support in orderbook')
            elif ask_volume > bid_volume * 1.2:
                bias_score -= 1
                factors.append('Heavy ask pressure in orderbook')
        
        if bias_score > 1:
            bias = 'bullish'
        elif bias_score < -1:
            bias = 'bearish'
        else:
            bias = 'neutral'
            
        return {
            'bias': bias,
            'score': bias_score,
            'factors': factors
        }
    
    def print_output(self, result: dict):
        """Print the SMC analysis output"""
        print("\n" + "="*60)
        print(f"SMART MONEY CONCEPTS ANALYSIS - {result['symbol']} ({result['timeframe']})")
        print("="*60)
        
        print(f"MARKET STRUCTURE:")
        structure = result['market_structure']
        print(f"  Trend: {structure['trend'].upper()}")
        
        print(f"\nFAIR VALUE GAPS ({len(result['fair_value_gaps'])} found):")
        for gap in result['fair_value_gaps'][-3:]:  # Show last 3
            print(f"  {gap['type'].upper()} FVG: {gap['bottom']:.4f} - {gap['top']:.4f} (Size: {gap['size']:.4f})")
        
        print(f"\nORDER BLOCKS ({len(result['order_blocks'])} found):")
        for ob in result['order_blocks'][-3:]:  # Show last 3
            print(f"  {ob['type'].upper()}: {ob['low']:.4f} - {ob['high']:.4f}")
        
        print(f"\nPREMIUM/DISCOUNT ANALYSIS:")
        pd_info = result['premium_discount']
        print(f"  Current Zone: {pd_info['zone'].upper()}")
        print(f"  Position: {pd_info['position_pct']:.1f}% of range")
        print(f"  Range: {pd_info['range_low']:.4f} - {pd_info['range_high']:.4f}")
        
        print(f"\nLIQUIDITY ZONES:")
        liq = result['liquidity_zones']
        print(f"  Equal Highs: {len(liq['equal_highs'])}")
        print(f"  Equal Lows: {len(liq['equal_lows'])}")
        
        print(f"\nOVERALL BIAS:")
        bias = result['current_bias']
        print(f"  Bias: {bias['bias'].upper()} (Score: {bias['score']})")
        print(f"  Key Factors:")
        for factor in bias['factors']:
            print(f"    - {factor}")