#!/usr/bin/env python3
"""
Enhanced Support/Resistance, Stop Loss, and Target Calculator
Industry-standard calculations with Fibonacci, Pivot Points, ATR, and Volume Profile
"""

import statistics
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class TradingLevel:
    """Represents a trading level with strength and type information"""
    price: float
    level_type: str  # 'support', 'resistance', 'pivot'
    strength: int    # 1-10 scale
    method: str      # 'fibonacci', 'pivot', 'atr', 'volume', 'pattern'
    confluence_count: int = 1
    test_count: int = 0  # How many times price tested this level


@dataclass
class RiskManagement:
    """Risk management parameters"""
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_ratio: float
    position_size_percent: float
    max_drawdown_percent: float


class EnhancedLevelsCalculator:
    """Enhanced calculator for support/resistance, stops, and targets using industry standards"""
    
    def __init__(self):
        # Industry standard parameters
        self.atr_period = 14
        self.fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.pivot_methods = ['classic', 'woodie', 'camarilla', 'demark', 'fibonacci']
        self.min_confluence_distance = 0.002  # 0.2% minimum distance between levels
        
    def calculate_atr(self, ohlcv_data: List[List], period: int = 14) -> List[float]:
        """Calculate Average True Range with industry standard smoothing"""
        if len(ohlcv_data) < period + 1:
            return [0.0] * len(ohlcv_data)
        
        true_ranges = []
        for i in range(1, len(ohlcv_data)):
            high = ohlcv_data[i][2]
            low = ohlcv_data[i][3]
            prev_close = ohlcv_data[i-1][4]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        # Wilder's smoothing method (industry standard)
        atr_values = [0.0]  # First value is 0
        
        if len(true_ranges) >= period:
            # Initial ATR is simple average
            first_atr = sum(true_ranges[:period]) / period
            atr_values.append(first_atr)
            
            # Subsequent ATR values use Wilder's smoothing
            for i in range(period, len(true_ranges)):
                atr = (atr_values[-1] * (period - 1) + true_ranges[i]) / period
                atr_values.append(atr)
        
        # Extend to match input length
        while len(atr_values) < len(ohlcv_data):
            atr_values.append(atr_values[-1] if atr_values else 0.0)
            
        return atr_values
    
    def calculate_fibonacci_levels(self, swing_high: float, swing_low: float, 
                                 trend_direction: str = 'up') -> Dict[str, TradingLevel]:
        """Calculate Fibonacci retracement/extension levels"""
        levels = {}
        price_range = swing_high - swing_low
        
        if trend_direction.lower() == 'up':
            # Uptrend: retracements from high
            base_price = swing_high
            for level in self.fibonacci_levels:
                price = base_price - (price_range * level)
                levels[f'fib_{level:.1%}'] = TradingLevel(
                    price=price,
                    level_type='support',
                    strength=self._get_fibonacci_strength(level),
                    method='fibonacci'
                )
            
            # Extensions above swing high
            extensions = [1.272, 1.414, 1.618, 2.0, 2.618]
            for ext in extensions:
                price = swing_high + (price_range * (ext - 1))
                levels[f'fib_ext_{ext:.1%}'] = TradingLevel(
                    price=price,
                    level_type='resistance',
                    strength=7 if ext == 1.618 else 5,
                    method='fibonacci'
                )
        else:
            # Downtrend: retracements from low
            base_price = swing_low
            for level in self.fibonacci_levels:
                price = base_price + (price_range * level)
                levels[f'fib_{level:.1%}'] = TradingLevel(
                    price=price,
                    level_type='resistance',
                    strength=self._get_fibonacci_strength(level),
                    method='fibonacci'
                )
            
            # Extensions below swing low
            extensions = [1.272, 1.414, 1.618, 2.0, 2.618]
            for ext in extensions:
                price = swing_low - (price_range * (ext - 1))
                levels[f'fib_ext_{ext:.1%}'] = TradingLevel(
                    price=price,
                    level_type='support',
                    strength=7 if ext == 1.618 else 5,
                    method='fibonacci'
                )
        
        return levels
    
    def _get_fibonacci_strength(self, level: float) -> int:
        """Get strength rating for Fibonacci levels"""
        strength_map = {
            0.236: 4,
            0.382: 6,
            0.5: 7,
            0.618: 9,  # Golden ratio - strongest
            0.786: 5
        }
        return strength_map.get(level, 5)
    
    def calculate_pivot_points(self, high: float, low: float, close: float, 
                             open_price: float = None, method: str = 'classic') -> Dict[str, TradingLevel]:
        """Calculate pivot points using various industry methods"""
        levels = {}
        
        if method == 'classic':
            pp = (high + low + close) / 3
            r1 = (2 * pp) - low
            r2 = pp + (high - low)
            r3 = high + 2 * (pp - low)
            s1 = (2 * pp) - high
            s2 = pp - (high - low)
            s3 = low - 2 * (high - pp)
            
        elif method == 'woodie':
            pp = (high + low + 2 * close) / 4
            r1 = (2 * pp) - low
            r2 = pp + (high - low)
            s1 = (2 * pp) - high
            s2 = pp - (high - low)
            r3 = high + 2 * (pp - low)
            s3 = low - 2 * (high - pp)
            
        elif method == 'camarilla':
            pp = (high + low + close) / 3
            r1 = close + (high - low) * 1.1 / 12
            r2 = close + (high - low) * 1.1 / 6
            r3 = close + (high - low) * 1.1 / 4
            s1 = close - (high - low) * 1.1 / 12
            s2 = close - (high - low) * 1.1 / 6
            s3 = close - (high - low) * 1.1 / 4
            
        elif method == 'demark':
            # Use open price if available, otherwise use close
            open_val = open_price if open_price is not None else close
            
            if close < open_val:
                x = high + 2 * low + close
            elif close > open_val:
                x = 2 * high + low + close
            else:
                x = high + low + 2 * close
            
            pp = x / 4
            r1 = x / 2 - low
            s1 = x / 2 - high
            r2 = r3 = s2 = s3 = pp  # DeMark only has R1/S1
            
        elif method == 'fibonacci':
            pp = (high + low + close) / 3
            r1 = pp + 0.382 * (high - low)
            r2 = pp + 0.618 * (high - low)
            r3 = pp + 1.000 * (high - low)
            s1 = pp - 0.382 * (high - low)
            s2 = pp - 0.618 * (high - low)
            s3 = pp - 1.000 * (high - low)
        
        # Create TradingLevel objects
        levels.update({
            f'{method}_pp': TradingLevel(pp, 'pivot', 8, f'pivot_{method}'),
            f'{method}_r1': TradingLevel(r1, 'resistance', 6, f'pivot_{method}'),
            f'{method}_r2': TradingLevel(r2, 'resistance', 7, f'pivot_{method}'),
            f'{method}_r3': TradingLevel(r3, 'resistance', 5, f'pivot_{method}'),
            f'{method}_s1': TradingLevel(s1, 'support', 6, f'pivot_{method}'),
            f'{method}_s2': TradingLevel(s2, 'support', 7, f'pivot_{method}'),
            f'{method}_s3': TradingLevel(s3, 'support', 5, f'pivot_{method}')
        })
        
        return levels
    
    def calculate_volume_levels(self, ohlcv_data: List[List], bins: int = 50) -> Dict[str, TradingLevel]:
        """Calculate volume profile support/resistance levels"""
        if len(ohlcv_data) < 20:
            return {}
        
        # Extract price and volume data
        prices = []
        volumes = []
        
        for candle in ohlcv_data:
            high, low, volume = candle[2], candle[3], candle[5]
            # Distribute volume across the high-low range
            price_range = high - low
            if price_range > 0:
                for i in range(10):  # 10 price points per candle
                    price = low + (price_range * i / 9)
                    prices.append(price)
                    volumes.append(volume / 10)
        
        if not prices:
            return {}
        
        # Create price bins and sum volumes
        min_price, max_price = min(prices), max(prices)
        price_range = max_price - min_price
        bin_size = price_range / bins
        
        volume_profile = {}
        for i in range(bins):
            bin_low = min_price + (i * bin_size)
            bin_high = bin_low + bin_size
            bin_center = (bin_low + bin_high) / 2
            
            total_volume = sum(vol for price, vol in zip(prices, volumes) 
                             if bin_low <= price < bin_high)
            
            if total_volume > 0:
                volume_profile[bin_center] = total_volume
        
        # Find high volume areas (top 20%)
        if not volume_profile:
            return {}
        
        sorted_volumes = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        top_levels = sorted_volumes[:max(1, len(sorted_volumes) // 5)]
        
        levels = {}
        for i, (price, volume) in enumerate(top_levels):
            strength = max(5, min(10, int(9 - (i * 0.5))))  # Decrease strength with rank
            level_type = self._determine_level_type(price, ohlcv_data[-20:])
            
            levels[f'volume_{i+1}'] = TradingLevel(
                price=price,
                level_type=level_type,
                strength=strength,
                method='volume'
            )
        
        return levels
    
    def _determine_level_type(self, price: float, recent_candles: List[List]) -> str:
        """Determine if a price level acts as support or resistance"""
        current_price = recent_candles[-1][4]  # Current close
        
        if price < current_price:
            return 'support'
        else:
            return 'resistance'
    
    def calculate_swing_levels(self, ohlcv_data: List[List], lookback: int = 20) -> Dict[str, TradingLevel]:
        """Calculate swing high/low support and resistance levels"""
        if len(ohlcv_data) < lookback * 2:
            return {}
        
        highs = [candle[2] for candle in ohlcv_data]
        lows = [candle[3] for candle in ohlcv_data]
        
        swing_highs = []
        swing_lows = []
        
        # Find swing highs and lows
        for i in range(lookback, len(ohlcv_data) - lookback):
            # Check for swing high
            is_swing_high = all(highs[i] >= highs[j] for j in range(i - lookback, i + lookback + 1) if j != i)
            if is_swing_high:
                swing_highs.append((i, highs[i]))
            
            # Check for swing low
            is_swing_low = all(lows[i] <= lows[j] for j in range(i - lookback, i + lookback + 1) if j != i)
            if is_swing_low:
                swing_lows.append((i, lows[i]))
        
        levels = {}
        
        # Recent swing highs (resistance)
        recent_swing_highs = sorted(swing_highs, key=lambda x: x[0], reverse=True)[:5]
        for i, (idx, price) in enumerate(recent_swing_highs):
            age_factor = max(5, 10 - i)  # Newer swings get higher strength
            levels[f'swing_high_{i+1}'] = TradingLevel(
                price=price,
                level_type='resistance',
                strength=age_factor,
                method='swing'
            )
        
        # Recent swing lows (support)
        recent_swing_lows = sorted(swing_lows, key=lambda x: x[0], reverse=True)[:5]
        for i, (idx, price) in enumerate(recent_swing_lows):
            age_factor = max(5, 10 - i)  # Newer swings get higher strength
            levels[f'swing_low_{i+1}'] = TradingLevel(
                price=price,
                level_type='support',
                strength=age_factor,
                method='swing'
            )
        
        return levels
    
    def calculate_confluence_levels(self, all_levels: Dict[str, TradingLevel], 
                                  current_price: float) -> Dict[str, TradingLevel]:
        """Identify confluence areas where multiple levels cluster"""
        if not all_levels:
            return {}
        
        confluence_zones = {}
        processed_levels = set()
        
        for name1, level1 in all_levels.items():
            if name1 in processed_levels:
                continue
            
            # Find nearby levels (within min_confluence_distance)
            confluent_levels = [level1]
            confluent_names = [name1]
            
            for name2, level2 in all_levels.items():
                if name2 != name1 and name2 not in processed_levels:
                    price_diff = abs(level1.price - level2.price) / current_price
                    if price_diff <= self.min_confluence_distance:
                        confluent_levels.append(level2)
                        confluent_names.append(name2)
            
            if len(confluent_levels) > 1:  # Confluence found
                # Calculate weighted average price
                total_strength = sum(level.strength for level in confluent_levels)
                weighted_price = sum(level.price * level.strength for level in confluent_levels) / total_strength
                
                # Determine level type (majority wins)
                support_count = sum(1 for level in confluent_levels if level.level_type == 'support')
                resistance_count = sum(1 for level in confluent_levels if level.level_type == 'resistance')
                pivot_count = sum(1 for level in confluent_levels if level.level_type == 'pivot')
                
                if pivot_count > 0:
                    level_type = 'pivot'
                elif support_count > resistance_count:
                    level_type = 'support'
                else:
                    level_type = 'resistance'
                
                # Calculate confluence strength
                confluence_strength = min(10, max(level.strength for level in confluent_levels) + len(confluent_levels))
                methods = list(set(level.method for level in confluent_levels))
                
                confluence_zones[f'confluence_{len(confluence_zones)+1}'] = TradingLevel(
                    price=weighted_price,
                    level_type=level_type,
                    strength=confluence_strength,
                    method=f"confluence_{'_'.join(methods)}",
                    confluence_count=len(confluent_levels)
                )
                
                # Mark levels as processed
                processed_levels.update(confluent_names)
        
        return confluence_zones
    
    def calculate_stop_loss(self, entry_price: float, signal: str, atr_value: float,
                           support_level: float = 0, resistance_level: float = 0,
                           method: str = 'atr') -> float:
        """Calculate stop loss using industry standard methods"""
        
        if method == 'atr':
            # ATR-based stops (industry standard: 1.5-2x ATR)
            if signal.upper() in ['BUY', 'LONG']:
                return entry_price - (atr_value * 2.0)
            else:
                return entry_price + (atr_value * 2.0)
        
        elif method == 'percentage':
            # Fixed percentage stops (2-3% typical)
            stop_percentage = 0.02  # 2%
            if signal.upper() in ['BUY', 'LONG']:
                return entry_price * (1 - stop_percentage)
            else:
                return entry_price * (1 + stop_percentage)
        
        elif method == 'support_resistance':
            # Stop beyond support/resistance with buffer
            buffer = atr_value * 0.5  # Half ATR buffer
            if signal.upper() in ['BUY', 'LONG'] and support_level > 0:
                return support_level - buffer
            elif signal.upper() in ['SELL', 'SHORT'] and resistance_level > 0:
                return resistance_level + buffer
            else:
                # Fallback to ATR method
                return self.calculate_stop_loss(entry_price, signal, atr_value, method='atr')
        
        elif method == 'volatility_adjusted':
            # Adjust stop based on recent volatility
            volatility_multiplier = min(3.0, max(1.5, atr_value / entry_price * 100))  # 1.5x to 3x based on volatility
            if signal.upper() in ['BUY', 'LONG']:
                return entry_price - (atr_value * volatility_multiplier)
            else:
                return entry_price + (atr_value * volatility_multiplier)
        
        # Default fallback
        return self.calculate_stop_loss(entry_price, signal, atr_value, method='atr')
    
    def calculate_take_profits(self, entry_price: float, stop_loss: float, signal: str,
                             atr_value: float, resistance_level: float = 0, support_level: float = 0,
                             target_method: str = 'risk_reward') -> Tuple[float, float]:
        """Calculate TP1 and TP2 using industry standard methods"""
        
        risk = abs(entry_price - stop_loss)
        
        if target_method == 'risk_reward':
            # Risk/reward ratio targets (industry standard: 2:1 and 3:1)
            if signal.upper() in ['BUY', 'LONG']:
                tp1 = entry_price + (risk * 2.0)  # 2:1 R/R
                tp2 = entry_price + (risk * 3.0)  # 3:1 R/R
            else:
                tp1 = entry_price - (risk * 2.0)
                tp2 = entry_price - (risk * 3.0)
        
        elif target_method == 'atr_multiple':
            # ATR-based targets
            if signal.upper() in ['BUY', 'LONG']:
                tp1 = entry_price + (atr_value * 3.0)
                tp2 = entry_price + (atr_value * 5.0)
            else:
                tp1 = entry_price - (atr_value * 3.0)
                tp2 = entry_price - (atr_value * 5.0)
        
        elif target_method == 'support_resistance':
            # Target nearest significant levels
            if signal.upper() in ['BUY', 'LONG']:
                tp1 = resistance_level if resistance_level > entry_price else entry_price + (risk * 2.0)
                tp2 = tp1 + (atr_value * 2.0)  # Extended target
            else:
                tp1 = support_level if support_level < entry_price else entry_price - (risk * 2.0)
                tp2 = tp1 - (atr_value * 2.0)  # Extended target
        
        elif target_method == 'fibonacci_projection':
            # Fibonacci projection targets
            if signal.upper() in ['BUY', 'LONG']:
                tp1 = entry_price + (risk * 1.618)  # Golden ratio
                tp2 = entry_price + (risk * 2.618)  # Fibonacci extension
            else:
                tp1 = entry_price - (risk * 1.618)
                tp2 = entry_price - (risk * 2.618)
        
        else:
            # Default to risk/reward method
            return self.calculate_take_profits(entry_price, stop_loss, signal, atr_value, 
                                             resistance_level, support_level, 'risk_reward')
        
        return tp1, tp2
    
    def calculate_position_size(self, account_balance: float, risk_percentage: float,
                              entry_price: float, stop_loss: float) -> Dict[str, float]:
        """Calculate position size based on risk management rules"""
        
        # Risk amount in currency
        risk_amount = account_balance * (risk_percentage / 100)
        
        # Risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return {'position_size': 0, 'units': 0, 'risk_amount': 0}
        
        # Units to trade
        units = risk_amount / risk_per_unit
        
        # Position size in currency
        position_size = units * entry_price
        
        # Position size as percentage of account
        position_size_percent = (position_size / account_balance) * 100
        
        return {
            'units': round(units, 8),
            'position_size': round(position_size, 2),
            'position_size_percent': round(position_size_percent, 2),
            'risk_amount': round(risk_amount, 2),
            'risk_per_unit': round(risk_per_unit, 4)
        }
    
    def analyze_comprehensive_levels(self, ohlcv_data: List[List], symbol: str,
                                   account_balance: float = 10000,
                                   risk_percentage: float = 1.0) -> Dict[str, Any]:
        """Comprehensive analysis using all level calculation methods"""
        
        if len(ohlcv_data) < 50:
            return {'success': False, 'error': 'Insufficient data'}
        
        current_price = ohlcv_data[-1][4]
        current_high = ohlcv_data[-1][2]
        current_low = ohlcv_data[-1][3]
        
        # Calculate ATR
        atr_values = self.calculate_atr(ohlcv_data)
        current_atr = atr_values[-1]
        
        # Find swing points for Fibonacci
        highs = [candle[2] for candle in ohlcv_data[-50:]]
        lows = [candle[3] for candle in ohlcv_data[-50:]]
        swing_high = max(highs)
        swing_low = min(lows)
        
        # Calculate all level types
        all_levels = {}
        
        # 1. Fibonacci levels
        fib_levels_up = self.calculate_fibonacci_levels(swing_high, swing_low, 'up')
        fib_levels_down = self.calculate_fibonacci_levels(swing_high, swing_low, 'down')
        all_levels.update(fib_levels_up)
        all_levels.update(fib_levels_down)
        
        # 2. Pivot points (multiple methods)
        prev_candle = ohlcv_data[-2]  # Previous day's data
        for method in self.pivot_methods:
            pivot_levels = self.calculate_pivot_points(
                prev_candle[2], prev_candle[3], prev_candle[4], prev_candle[1], method
            )
            all_levels.update(pivot_levels)
        
        # 3. Volume profile levels
        volume_levels = self.calculate_volume_levels(ohlcv_data)
        all_levels.update(volume_levels)
        
        # 4. Swing levels
        swing_levels = self.calculate_swing_levels(ohlcv_data)
        all_levels.update(swing_levels)
        
        # 5. Find confluence zones
        confluence_levels = self.calculate_confluence_levels(all_levels, current_price)
        
        # Separate levels by type and proximity to current price
        nearby_support = []
        nearby_resistance = []
        
        for name, level in {**all_levels, **confluence_levels}.items():
            distance_pct = abs(level.price - current_price) / current_price
            if distance_pct <= 0.05:  # Within 5% of current price
                if level.level_type == 'support' and level.price < current_price:
                    nearby_support.append((name, level))
                elif level.level_type == 'resistance' and level.price > current_price:
                    nearby_resistance.append((name, level))
        
        # Sort by strength and proximity
        nearby_support.sort(key=lambda x: (x[1].strength, -abs(x[1].price - current_price)), reverse=True)
        nearby_resistance.sort(key=lambda x: (x[1].strength, -abs(x[1].price - current_price)), reverse=True)
        
        # Get strongest levels
        strongest_support = nearby_support[0][1] if nearby_support else None
        strongest_resistance = nearby_resistance[0][1] if nearby_resistance else None
        
        # Determine market bias and signal
        if strongest_support and strongest_resistance:
            support_distance = (current_price - strongest_support.price) / current_price
            resistance_distance = (strongest_resistance.price - current_price) / current_price
            
            if support_distance < resistance_distance and support_distance < 0.02:
                signal = 'BUY'
                bias = 'BULLISH'
                confidence = min(90, strongest_support.strength * 10)
            elif resistance_distance < support_distance and resistance_distance < 0.02:
                signal = 'SELL'
                bias = 'BEARISH'
                confidence = min(90, strongest_resistance.strength * 10)
            else:
                signal = 'HOLD'
                bias = 'NEUTRAL'
                confidence = 50
        else:
            signal = 'HOLD'
            bias = 'NEUTRAL'
            confidence = 50
        
        # Calculate trading levels
        entry_price = current_price
        
        # Calculate stop loss using multiple methods
        stop_atr = self.calculate_stop_loss(entry_price, signal, current_atr, method='atr')
        stop_sr = self.calculate_stop_loss(
            entry_price, signal, current_atr,
            strongest_support.price if strongest_support else 0,
            strongest_resistance.price if strongest_resistance else 0,
            method='support_resistance'
        )
        
        # Use the more conservative stop
        if signal == 'BUY':
            final_stop = max(stop_atr, stop_sr)
        elif signal == 'SELL':
            final_stop = min(stop_atr, stop_sr)
        else:
            final_stop = stop_atr
        
        # Calculate take profits
        tp1, tp2 = self.calculate_take_profits(
            entry_price, final_stop, signal, current_atr,
            strongest_resistance.price if strongest_resistance else 0,
            strongest_support.price if strongest_support else 0,
            'risk_reward'
        )
        
        # Calculate position sizing
        position_info = self.calculate_position_size(
            account_balance, risk_percentage, entry_price, final_stop
        )
        
        # Calculate risk/reward ratio
        risk = abs(entry_price - final_stop)
        reward = abs(tp1 - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        return {
            'success': True,
            'symbol': symbol,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': round(current_price, 4),
            'atr_value': round(current_atr, 4),
            'signal': signal,
            'bias': bias,
            'confidence_score': confidence,
            
            # Key levels
            'support_level': round(strongest_support.price, 4) if strongest_support else 0,
            'resistance_level': round(strongest_resistance.price, 4) if strongest_resistance else 0,
            'support_strength': strongest_support.strength if strongest_support else 0,
            'resistance_strength': strongest_resistance.strength if strongest_resistance else 0,
            
            # Trading levels
            'stop_zone': round(final_stop, 4),
            'tp_low': round(tp1, 4),
            'tp_high': round(tp2, 4),
            'rr_ratio': round(rr_ratio, 2),
            'entry_price': round(entry_price, 4),
            
            # Risk management
            'position_sizing': position_info,
            'max_drawdown_percent': round((current_atr / current_price) * 100 * 2, 2),
            
            # Level details
            'all_levels_count': len(all_levels),
            'confluence_zones_count': len(confluence_levels),
            'nearby_support_count': len(nearby_support),
            'nearby_resistance_count': len(nearby_resistance),
            
            # Raw data for debugging
            'level_details': {
                'fibonacci_levels': len([l for l in all_levels.values() if l.method == 'fibonacci']),
                'pivot_levels': len([l for l in all_levels.values() if 'pivot' in l.method]),
                'volume_levels': len([l for l in all_levels.values() if l.method == 'volume']),
                'swing_levels': len([l for l in all_levels.values() if l.method == 'swing']),
                'confluence_zones': len(confluence_levels)
            }
        }


if __name__ == "__main__":
    # Example usage and testing
    calc = EnhancedLevelsCalculator()
    
    # Mock OHLCV data for testing
    import random
    base_price = 50000
    mock_data = []
    
    for i in range(100):
        # Generate realistic OHLCV data
        open_price = base_price + random.uniform(-100, 100)
        close_price = open_price + random.uniform(-50, 50)
        high_price = max(open_price, close_price) + random.uniform(0, 30)
        low_price = min(open_price, close_price) - random.uniform(0, 30)
        volume = random.uniform(1000, 5000)
        
        mock_data.append([i, open_price, high_price, low_price, close_price, volume])
        base_price = close_price
    
    # Test comprehensive analysis
    result = calc.analyze_comprehensive_levels(mock_data, 'BTC/USDT')
    
    print("Enhanced Levels Analysis Results:")
    print(f"Signal: {result['signal']}")
    print(f"Bias: {result['bias']}")
    print(f"Confidence: {result['confidence_score']}%")
    print(f"Support: ${result['support_level']:.2f} (Strength: {result['support_strength']})")
    print(f"Resistance: ${result['resistance_level']:.2f} (Strength: {result['resistance_strength']})")
    print(f"Stop Loss: ${result['stop_zone']:.2f}")
    print(f"TP1: ${result['tp_low']:.2f}")
    print(f"TP2: ${result['tp_high']:.2f}")
    print(f"Risk/Reward: 1:{result['rr_ratio']:.1f}")
    print(f"Total Levels Analyzed: {result['all_levels_count']}")
    print(f"Confluence Zones: {result['confluence_zones_count']}")