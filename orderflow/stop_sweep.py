"""
Stop Run & Liquidity Sweep Detection Implementation

Professional-grade stop run and liquidity sweep detection using:
- Real-time L2 market depth analysis
- Tick trade data processing
- ZigZag/fractal support/resistance level identification
- Volume spike and reversal pattern detection
"""

import logging
import time
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import statistics
from data import get_data_collector

logger = logging.getLogger(__name__)

@dataclass
class SwingPoint:
    """Swing high/low point identified by ZigZag analysis."""
    timestamp: float
    price: float
    point_type: str  # "swing_high" or "swing_low"
    volume: float
    strength: float  # 0.0 to 1.0 based on significance
    confirmed: bool = False

@dataclass
class SupportResistanceLevel:
    """Support or resistance level for stop run detection."""
    price: float
    level_type: str  # "support" or "resistance"
    strength: float  # 0.0 to 1.0
    timestamp_created: float
    touches: int = 1
    volume_at_level: float = 0.0
    swing_points: List[SwingPoint] = None

@dataclass
class DeltaAnalysis:
    """Order flow delta analysis data."""
    cumulative_delta: float
    delta_divergence: bool
    buy_pressure: float
    sell_pressure: float
    delta_momentum: float
    
@dataclass
class ATRData:
    """Average True Range calculation data."""
    current_atr: float
    atr_multiplier: float = 2.0
    lookback_period: int = 14

@dataclass
class MarketRegimeData:
    """Market regime analysis data."""
    volatility_percentile: float
    trend_strength: float
    volume_profile: str  # "high", "medium", "low"
    session_type: str  # "asian", "london", "ny", "overlap"
    is_favorable: bool

@dataclass
class HistoricalPerformance:
    """Historical success rate tracking."""
    total_signals: int = 0
    successful_signals: int = 0
    success_rate: float = 0.0
    avg_profit_loss: float = 0.0
    confidence_breakdown: Dict[str, Dict] = field(default_factory=dict)

@dataclass
class StopRunEvent:
    """Enhanced stop run event with all relevant data."""
    timestamp: float
    symbol: str
    level: SupportResistanceLevel
    probe_price: float
    ticks_beyond: float
    volume_spike: float
    volume_threshold: float
    reversal_confirmed: bool
    reversal_time_seconds: float
    signal_type: str  # "bullish_stop_run" or "bearish_stop_run"
    confidence: str  # "LOW", "MEDIUM", "HIGH"
    entry_price: float
    stop_loss: float
    take_profit: float
    delta_analysis: Optional[DeltaAnalysis] = None
    atr_data: Optional[ATRData] = None
    market_regime: Optional[MarketRegimeData] = None
    confluence_score: float = 0.0
    historical_success_rate: float = 0.0

@dataclass
class LiquiditySweepEvent:
    """Enhanced liquidity sweep event."""
    timestamp: float
    symbol: str
    level: SupportResistanceLevel
    wall_volume_consumed: float
    wall_volume_threshold: float
    trade_volume_at_price: float
    reversal_confirmed: bool
    reversal_time_seconds: float
    signal_type: str  # "bullish_sweep" or "bearish_sweep"
    confidence: str
    entry_price: float
    stop_loss: float
    take_profit: float
    delta_analysis: Optional[DeltaAnalysis] = None
    atr_data: Optional[ATRData] = None
    market_regime: Optional[MarketRegimeData] = None
    confluence_score: float = 0.0
    historical_success_rate: float = 0.0

class ZigZagDetector:
    """ZigZag pattern detection for identifying swing points."""
    
    def __init__(self, threshold_percent: float = 3.0):
        """
        Initialize ZigZag detector.
        
        Args:
            threshold_percent: Minimum percentage move to register swing
        """
        self.threshold = threshold_percent / 100.0
        
    def detect_swing_points(self, ohlcv_data: List[List]) -> List[SwingPoint]:
        """
        Detect swing highs and lows using ZigZag logic.
        
        Args:
            ohlcv_data: OHLCV candlestick data
            
        Returns:
            List of SwingPoint objects
        """
        if len(ohlcv_data) < 3:
            return []
            
        swing_points = []
        highs = [candle[2] for candle in ohlcv_data]  # High prices
        lows = [candle[3] for candle in ohlcv_data]   # Low prices
        volumes = [candle[5] for candle in ohlcv_data]
        timestamps = [candle[0] for candle in ohlcv_data]
        
        # Find initial direction
        current_high = highs[0]
        current_low = lows[0]
        current_trend = None
        last_swing_idx = 0
        
        for i in range(1, len(highs)):
            high_change = (highs[i] - current_high) / current_high
            low_change = (current_low - lows[i]) / current_low
            
            # Check for swing high
            if high_change >= self.threshold:
                if current_trend != "up":
                    # Add previous swing low if we were going down
                    if current_trend == "down" and last_swing_idx < i:
                        swing_points.append(SwingPoint(
                            timestamp=timestamps[last_swing_idx],
                            price=current_low,
                            point_type="swing_low",
                            volume=volumes[last_swing_idx],
                            strength=self._calculate_swing_strength(current_low, lows, last_swing_idx),
                            confirmed=True
                        ))
                    current_trend = "up"
                    last_swing_idx = i
                
                current_high = highs[i]
                
            # Check for swing low  
            elif low_change >= self.threshold:
                if current_trend != "down":
                    # Add previous swing high if we were going up
                    if current_trend == "up" and last_swing_idx < i:
                        swing_points.append(SwingPoint(
                            timestamp=timestamps[last_swing_idx],
                            price=current_high,
                            point_type="swing_high", 
                            volume=volumes[last_swing_idx],
                            strength=self._calculate_swing_strength(current_high, highs, last_swing_idx),
                            confirmed=True
                        ))
                    current_trend = "down"
                    last_swing_idx = i
                    
                current_low = lows[i]
        
        return swing_points
    
    def _calculate_swing_strength(self, price: float, price_series: List[float], 
                                index: int, lookback: int = 10) -> float:
        """Calculate swing point strength based on local extremes."""
        start_idx = max(0, index - lookback)
        end_idx = min(len(price_series), index + lookback + 1)
        local_prices = price_series[start_idx:end_idx]
        
        if not local_prices:
            return 0.5
            
        price_range = max(local_prices) - min(local_prices)
        if price_range == 0:
            return 0.5
            
        # Strength based on how extreme the price is in local context
        local_min = min(local_prices)
        local_max = max(local_prices)
        
        if price == local_max:
            return 1.0  # Perfect high
        elif price == local_min:
            return 1.0  # Perfect low
        else:
            # Intermediate strength
            return 0.3 + 0.4 * abs(price - (local_min + local_max) / 2) / (price_range / 2)

class EnhancedStopSweepDetector:
    """Enhanced professional stop run and liquidity sweep detector with advanced analytics."""
    
    def __init__(self):
        """Initialize the enhanced detector."""
        self.collector = get_data_collector()
        self.zigzag = ZigZagDetector()
        self.performance_tracker = HistoricalPerformance()
        self._load_historical_performance()
        
    def _load_historical_performance(self):
        """Load historical performance data from file."""
        try:
            with open('performance_data.json', 'r') as f:
                data = json.load(f)
                self.performance_tracker = HistoricalPerformance(**data)
        except FileNotFoundError:
            logger.info("No historical performance data found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
    
    def _save_historical_performance(self):
        """Save historical performance data to file."""
        try:
            with open('performance_data.json', 'w') as f:
                json.dump({
                    'total_signals': self.performance_tracker.total_signals,
                    'successful_signals': self.performance_tracker.successful_signals,
                    'success_rate': self.performance_tracker.success_rate,
                    'avg_profit_loss': self.performance_tracker.avg_profit_loss,
                    'confidence_breakdown': self.performance_tracker.confidence_breakdown
                }, f)
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
        
    def identify_sr_levels(self, ohlcv_data: List[List], 
                          zigzag_threshold: float = 3.0) -> List[SupportResistanceLevel]:
        """
        Identify support and resistance levels from swing points.
        
        Args:
            ohlcv_data: OHLCV candlestick data
            zigzag_threshold: ZigZag threshold percentage
            
        Returns:
            List of SupportResistanceLevel objects
        """
        self.zigzag.threshold = zigzag_threshold / 100.0
        swing_points = self.zigzag.detect_swing_points(ohlcv_data)
        
        if not swing_points:
            return []
            
        levels = []
        
        # Group swing points by similar price levels (within 0.1% tolerance)
        price_tolerance = 0.001  # 0.1%
        
        for swing in swing_points:
            # Find existing level near this price
            matching_level = None
            for level in levels:
                if abs(swing.price - level.price) / level.price <= price_tolerance:
                    matching_level = level
                    break
            
            if matching_level:
                # Strengthen existing level
                matching_level.touches += 1
                matching_level.strength = min(1.0, matching_level.strength + 0.2)
                matching_level.volume_at_level += swing.volume
                if matching_level.swing_points is None:
                    matching_level.swing_points = []
                matching_level.swing_points.append(swing)
            else:
                # Create new level
                level_type = "resistance" if swing.point_type == "swing_high" else "support"
                levels.append(SupportResistanceLevel(
                    price=swing.price,
                    level_type=level_type,
                    strength=swing.strength,
                    timestamp_created=swing.timestamp,
                    touches=1,
                    volume_at_level=swing.volume,
                    swing_points=[swing]
                ))
        
        # Sort by strength (strongest first)
        levels.sort(key=lambda x: x.strength, reverse=True)
        
        return levels[:10]  # Return top 10 strongest levels
    
    def detect_price_probe(self, current_price: float, level: SupportResistanceLevel, 
                          tick_size: float = 0.0001) -> Tuple[bool, float]:
        """
        Detect if price has probed beyond a support/resistance level.
        
        Args:
            current_price: Current market price
            level: Support/resistance level to check
            tick_size: Minimum price movement (tick size)
            
        Returns:
            Tuple of (is_probe, ticks_beyond)
        """
        if level.level_type == "support":
            # Price below support = probe
            if current_price < level.price:
                ticks_beyond = (level.price - current_price) / tick_size
                return True, ticks_beyond
        elif level.level_type == "resistance":
            # Price above resistance = probe  
            if current_price > level.price:
                ticks_beyond = (current_price - level.price) / tick_size
                return True, ticks_beyond
                
        return False, 0.0
    
    def detect_volume_spike(self, trades_data: List[Dict], window_seconds: int = 5,
                          volume_threshold: float = 5000) -> Tuple[bool, float]:
        """
        Detect volume spike in recent trades.
        
        Args:
            trades_data: Recent trade data
            window_seconds: Time window to check for spike
            volume_threshold: Minimum volume for spike
            
        Returns:
            Tuple of (is_spike, actual_volume)
        """
        if not trades_data:
            return False, 0.0
            
        current_time = time.time() * 1000  # Convert to milliseconds
        window_start = current_time - (window_seconds * 1000)
        
        recent_volume = 0.0
        for trade in trades_data:
            trade_time = trade.get('timestamp', 0)
            if trade_time >= window_start:
                recent_volume += float(trade.get('amount', 0))
        
        return recent_volume >= volume_threshold, recent_volume
    
    def detect_liquidity_wall_consumption(self, orderbook_data: Dict, 
                                        level: SupportResistanceLevel,
                                        wall_volume_threshold: float = 10000) -> Tuple[bool, float]:
        """
        Detect if large liquidity wall has been consumed at level.
        
        Args:
            orderbook_data: L2 orderbook data
            level: Support/resistance level
            wall_volume_threshold: Minimum wall size
            
        Returns:
            Tuple of (wall_consumed, wall_volume)
        """
        if not orderbook_data or 'bids' not in orderbook_data or 'asks' not in orderbook_data:
            return False, 0.0
            
        price_tolerance = 0.0005  # 0.05% tolerance for level matching
        wall_volume = 0.0
        
        if level.level_type == "support":
            # Check bid side for support wall
            for bid_price, bid_volume in orderbook_data['bids']:
                if abs(float(bid_price) - level.price) / level.price <= price_tolerance:
                    wall_volume += float(bid_volume)
        elif level.level_type == "resistance":
            # Check ask side for resistance wall
            for ask_price, ask_volume in orderbook_data['asks']:
                if abs(float(ask_price) - level.price) / level.price <= price_tolerance:
                    wall_volume += float(ask_volume)
        
        # Wall consumed if volume below threshold (was consumed)
        wall_consumed = wall_volume < (wall_volume_threshold * 0.3)  # 30% remaining = consumed
        
        return wall_consumed, wall_volume
    
    def detect_reversal(self, symbol: str, level: SupportResistanceLevel,
                       sustain_seconds: int = 3) -> Tuple[bool, float]:
        """
        Detect if price has reversed back inside the level.
        
        Args:
            symbol: Trading symbol
            level: Support/resistance level
            sustain_seconds: Time to confirm reversal
            
        Returns:
            Tuple of (reversal_confirmed, time_taken)
        """
        try:
            start_time = time.time()
            end_time = start_time + sustain_seconds
            
            while time.time() < end_time:
                # Get current ticker
                ticker = self.collector.fetch_ticker(symbol)
                current_price = float(ticker.get('last', 0))
                
                if level.level_type == "support":
                    # Reversal = price back above support
                    if current_price >= level.price:
                        return True, time.time() - start_time
                elif level.level_type == "resistance":
                    # Reversal = price back below resistance
                    if current_price <= level.price:
                        return True, time.time() - start_time
                
                time.sleep(0.5)  # Check every 500ms
            
            return False, sustain_seconds
            
        except Exception as e:
            logger.error(f"Error detecting reversal: {e}")
            return False, sustain_seconds
    
    def calculate_atr(self, ohlcv_data: List[List], period: int = 14) -> ATRData:
        """Calculate Average True Range for dynamic stop losses."""
        if len(ohlcv_data) < period + 1:
            return ATRData(current_atr=0.01)  # Default fallback
            
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
        
        recent_trs = true_ranges[-period:] if len(true_ranges) >= period else true_ranges
        current_atr = sum(recent_trs) / len(recent_trs) if recent_trs else 0.01
        
        return ATRData(current_atr=current_atr)
    
    def analyze_delta_divergence(self, trades_data: List[Dict], 
                                current_price: float, level: SupportResistanceLevel) -> DeltaAnalysis:
        """Analyze order flow delta for divergence confirmation."""
        if not trades_data:
            return DeltaAnalysis(0.0, False, 0.0, 0.0, 0.0)
        
        buy_volume = 0.0
        sell_volume = 0.0
        total_volume = 0.0
        
        # Calculate delta from recent trades
        for trade in trades_data[-100:]:  # Last 100 trades
            volume = float(trade.get('amount', 0))
            price = float(trade.get('price', 0))
            side = trade.get('side', 'unknown')
            
            total_volume += volume
            
            if side == 'buy' or (side == 'unknown' and price >= current_price):
                buy_volume += volume
            else:
                sell_volume += volume
        
        cumulative_delta = buy_volume - sell_volume
        buy_pressure = buy_volume / total_volume if total_volume > 0 else 0.5
        sell_pressure = sell_volume / total_volume if total_volume > 0 else 0.5
        
        # Delta momentum (simplified)
        delta_momentum = cumulative_delta / total_volume if total_volume > 0 else 0.0
        
        # Detect divergence
        expected_delta_direction = 1.0 if level.level_type == "support" else -1.0
        actual_delta_direction = 1.0 if cumulative_delta > 0 else -1.0
        delta_divergence = expected_delta_direction == actual_delta_direction
        
        return DeltaAnalysis(
            cumulative_delta=cumulative_delta,
            delta_divergence=delta_divergence,
            buy_pressure=buy_pressure,
            sell_pressure=sell_pressure,
            delta_momentum=delta_momentum
        )
    
    def assess_market_regime(self, ohlcv_data: List[List], current_time: datetime) -> MarketRegimeData:
        """Assess current market regime for signal filtering."""
        if len(ohlcv_data) < 20:
            return MarketRegimeData(50.0, 0.5, "medium", "unknown", True)
        
        # Calculate volatility percentile
        closes = [float(candle[4]) for candle in ohlcv_data[-20:]]
        returns = [abs(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        current_volatility = statistics.mean(returns) if returns else 0.0
        volatility_percentile = min(100.0, current_volatility * 10000)  # Simplified percentile
        
        # Trend strength (simplified momentum)
        trend_strength = abs(closes[-1] - closes[0]) / closes[0] if closes else 0.0
        
        # Volume profile assessment
        volumes = [float(candle[5]) for candle in ohlcv_data[-10:]]
        avg_volume = statistics.mean(volumes) if volumes else 0.0
        recent_avg = statistics.mean(volumes[-3:]) if len(volumes) >= 3 else avg_volume
        
        if recent_avg > avg_volume * 1.5:
            volume_profile = "high"
        elif recent_avg < avg_volume * 0.7:
            volume_profile = "low"
        else:
            volume_profile = "medium"
        
        # Session detection (simplified UTC-based)
        hour = current_time.hour
        if 0 <= hour < 8:
            session_type = "asian"
        elif 8 <= hour < 13:
            session_type = "london"
        elif 13 <= hour < 22:
            session_type = "ny"
        else:
            session_type = "overlap"
        
        # Determine if regime is favorable
        is_favorable = (
            volatility_percentile >= 20.0 and  # Minimum volatility
            volume_profile != "low" and
            session_type in ["london", "ny", "overlap"]
        )
        
        return MarketRegimeData(
            volatility_percentile=volatility_percentile,
            trend_strength=trend_strength,
            volume_profile=volume_profile,
            session_type=session_type,
            is_favorable=is_favorable
        )
    
    def calculate_confluence_score(self, level: SupportResistanceLevel, 
                                 all_levels: List[SupportResistanceLevel]) -> float:
        """Calculate confluence score based on nearby levels."""
        confluence_score = 0.0
        price_tolerance = 0.005  # 0.5% tolerance
        
        # Base score from level strength
        confluence_score += level.strength * 0.3
        
        # Points for multiple touches
        confluence_score += min(0.2, level.touches * 0.05)
        
        # Points for nearby levels (confluence)
        nearby_levels = 0
        for other_level in all_levels:
            if other_level != level:
                price_diff = abs(level.price - other_level.price) / level.price
                if price_diff <= price_tolerance:
                    nearby_levels += 1
                    confluence_score += 0.1
        
        return min(1.0, confluence_score)
    
    def calculate_enhanced_entry_targets(self, level: SupportResistanceLevel, 
                                       signal_type: str, atr_data: ATRData,
                                       confluence_score: float) -> Tuple[float, float, float]:
        """Calculate enhanced entry, stop loss, and take profit using ATR."""
        risk_reward_ratio = 2.0 + (confluence_score * 0.5)  # Higher R:R for better confluence
        atr_multiplier = atr_data.atr_multiplier * (1.0 + confluence_score * 0.3)
        
        if "bullish" in signal_type:
            entry_price = level.price
            stop_loss = level.price - (atr_data.current_atr * atr_multiplier)
            take_profit = entry_price + (abs(entry_price - stop_loss) * risk_reward_ratio)
        else:  # bearish
            entry_price = level.price
            stop_loss = level.price + (atr_data.current_atr * atr_multiplier)
            take_profit = entry_price - (abs(stop_loss - entry_price) * risk_reward_ratio)
        
        return entry_price, stop_loss, take_profit
    
    def analyze_enhanced_stop_sweep(self, symbol: str, timeframe: str = '1m',
                                  zigzag_threshold: float = 3.0, volume_threshold: float = 5000,
                                  wall_volume_threshold: float = 10000, spike_window: int = 5,
                                  sustain_seconds: int = 3, limit: int = 100, ohlcv_data: Optional[List] = None) -> str:
        """
        Complete enhanced stop run and liquidity sweep analysis with advanced features.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for swing detection
            zigzag_threshold: ZigZag threshold percentage
            volume_threshold: Minimum volume for spike detection
            wall_volume_threshold: Minimum wall volume for sweep detection
            spike_window: Window in seconds for volume spike detection
            sustain_seconds: Time to confirm reversal
            limit: Number of candles for analysis
            
        Returns:
            Enhanced formatted analysis output with delta, ATR, confluence, and regime data
        """
        try:
            # Fetch data if not provided
            if ohlcv_data is None:
                ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
            if len(ohlcv_data) < 20:
                return f"Insufficient OHLCV data for {symbol}"
            
            # Identify support/resistance levels
            sr_levels = self.identify_sr_levels(ohlcv_data, zigzag_threshold)
            if not sr_levels:
                return f"No significant S/R levels found for {symbol}"
            
            # Get current market data
            ticker = self.collector.fetch_ticker(symbol)
            current_price = float(ticker.get('last', 0))
            
            trades_data = self.collector.fetch_trades_stream(symbol, limit=500)
            orderbook_data = self.collector.fetch_order_book(symbol)
            
            # Calculate enhanced analytics
            atr_data = self.calculate_atr(ohlcv_data)
            market_regime = self.assess_market_regime(ohlcv_data, datetime.now())
            
            # Market regime filter - skip if unfavorable conditions
            if not market_regime.is_favorable:
                return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {symbol} | Market regime unfavorable for signals\n" + \
                       f"Volatility: {market_regime.volatility_percentile:.1f}% | Volume: {market_regime.volume_profile} | Session: {market_regime.session_type}"
            
            output = []
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            output.append(f"[{timestamp_str}] {symbol} | TF: {timeframe} | Enhanced Analysis")
            output.append(f"Current Price: {current_price:.4f} | ATR: {atr_data.current_atr:.6f}")
            output.append(f"Market Regime: {market_regime.session_type.upper()} | Vol: {market_regime.volatility_percentile:.1f}% | {market_regime.volume_profile.upper()}")
            output.append("")
            
            detected_events = []
            
            # Check each S/R level for stop runs and sweeps
            for level in sr_levels[:5]:  # Check top 5 levels
                is_probe, ticks_beyond = self.detect_price_probe(current_price, level)
                
                if is_probe and ticks_beyond >= 1.0:  # At least 1 tick beyond
                    # Enhanced analysis
                    confluence_score = self.calculate_confluence_score(level, sr_levels)
                    delta_analysis = self.analyze_delta_divergence(trades_data, current_price, level)
                    
                    # Check for volume spike
                    is_volume_spike, actual_volume = self.detect_volume_spike(
                        trades_data, spike_window, volume_threshold
                    )
                    
                    # Check for liquidity wall consumption
                    wall_consumed, wall_volume = self.detect_liquidity_wall_consumption(
                        orderbook_data, level, wall_volume_threshold
                    )
                    
                    level_type = level.level_type.title()
                    output.append(f"Level: {level_type} @ {level.price:.4f} (strength: {level.strength:.2f}, confluence: {confluence_score:.2f})")
                    
                    if is_volume_spike or wall_consumed:
                        # Potential stop run or sweep detected
                        output.append(f"Stop Run Detected: Price → {current_price:.4f} (+{ticks_beyond:.1f} ticks)")
                        
                        if is_volume_spike:
                            output.append(f"• Volume Spike: {actual_volume:.0f} in {spike_window}s (vol_th={volume_threshold})")
                        
                        if wall_consumed:
                            output.append(f"• Liquidity Wall ({wall_volume:.0f}) consumed @ {level.price:.4f}")
                        
                        # Check for reversal
                        reversal_confirmed, reversal_time = self.detect_reversal(
                            symbol, level, sustain_seconds
                        )
                        
                        if reversal_confirmed:
                            output.append(f"• Reversal: closed back inside level in {reversal_time:.1f}s → ✅ Confirmed")
                            
                            # Determine signal type
                            if level.level_type == "support":
                                signal_type = "bullish_stop_run"
                                direction = "LONG"
                            else:
                                signal_type = "bearish_stop_run"
                                direction = "SHORT"
                            
                            # Calculate enhanced targets
                            entry, stop_loss, take_profit = self.calculate_enhanced_entry_targets(
                                level, signal_type, atr_data, confluence_score
                            )
                            
                            # Enhanced confidence calculation
                            confidence_score = 0.0
                            if is_volume_spike:
                                confidence_score += 0.25
                            if wall_consumed:
                                confidence_score += 0.25
                            if level.strength >= 0.8:
                                confidence_score += 0.15
                            if delta_analysis.delta_divergence:
                                confidence_score += 0.2  # Delta confirmation
                            if confluence_score >= 0.7:
                                confidence_score += 0.15  # High confluence
                            
                            confidence = "HIGH" if confidence_score >= 0.8 else "MEDIUM" if confidence_score >= 0.5 else "LOW"
                            
                            # Get historical success rate for this confidence level
                            historical_success = self.performance_tracker.confidence_breakdown.get(
                                confidence, {'success_rate': 0.0}
                            ).get('success_rate', 0.0)
                            
                            output.append("")
                            output.append(f"📊 ENHANCED SIGNAL ANALYSIS:")
                            output.append(f"• Delta: {delta_analysis.cumulative_delta:+.0f} | Divergence: {'✅' if delta_analysis.delta_divergence else '❌'}")
                            output.append(f"• Buy Pressure: {delta_analysis.buy_pressure:.1%} | Sell Pressure: {delta_analysis.sell_pressure:.1%}")
                            output.append(f"• ATR Stop: {atr_data.current_atr:.6f} (x{atr_data.atr_multiplier:.1f}) | R:R: {abs(take_profit-entry)/abs(entry-stop_loss):.1f}:1")
                            output.append("")
                            output.append(f"Signal: {direction} entry at {entry:.4f} | SL: {stop_loss:.4f} | TP: {take_profit:.4f}")
                            output.append(f"Confidence: {confidence} ({confidence_score:.1%}) | Historical: {historical_success:.1%} success")
                            
                            # Create enhanced event
                            event = StopRunEvent(
                                timestamp=time.time(),
                                symbol=symbol,
                                level=level,
                                probe_price=current_price,
                                ticks_beyond=ticks_beyond,
                                volume_spike=actual_volume,
                                volume_threshold=volume_threshold,
                                reversal_confirmed=reversal_confirmed,
                                reversal_time_seconds=reversal_time,
                                signal_type=signal_type,
                                confidence=confidence,
                                entry_price=entry,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                delta_analysis=delta_analysis,
                                atr_data=atr_data,
                                market_regime=market_regime,
                                confluence_score=confluence_score,
                                historical_success_rate=historical_success
                            )
                            
                            detected_events.append(event)
                            
                            # Update performance tracking
                            self._update_performance_tracking(confidence)
                        else:
                            output.append(f"• Reversal: NOT confirmed within {sustain_seconds}s → ❌ Failed")
                        
                        output.append("")
            
            if not detected_events:
                output.append("No stop runs or liquidity sweeps detected at current levels.")
                output.append("\nKey Support/Resistance Levels:")
                for i, level in enumerate(sr_levels[:3], 1):
                    level_type = level.level_type.title()
                    output.append(f"{i}. {level_type} @ {level.price:.4f} (strength: {level.strength:.2f}, touches: {level.touches})")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Error in enhanced stop sweep analysis: {e}")
            return f"Error analyzing enhanced stop sweeps for {symbol}: {str(e)}"

    def _update_performance_tracking(self, confidence: str):
        """Update performance tracking statistics."""
        self.performance_tracker.total_signals += 1
        
        if confidence not in self.performance_tracker.confidence_breakdown:
            self.performance_tracker.confidence_breakdown[confidence] = {
                'total': 0, 'successful': 0, 'success_rate': 0.0
            }
        
        self.performance_tracker.confidence_breakdown[confidence]['total'] += 1
        
        # Save updated performance data
        self._save_historical_performance()

# Legacy compatibility
class StopSweepDetector(EnhancedStopSweepDetector):
    """Legacy wrapper for backward compatibility."""
    def analyze_stop_sweep(self, *args, **kwargs):
        return self.analyze_enhanced_stop_sweep(*args, **kwargs)

class StopSweepStrategy:
    """Stop run and liquidity sweep detection strategy."""
    
    def __init__(self):
        """Initialize strategy."""
        self.detector = EnhancedStopSweepDetector()
    
    def analyze(self, symbol: str, timeframe: str = '1m', limit: int = 100, ohlcv_data: Optional[List] = None) -> Dict:
        """
        Analyze stop run and liquidity sweep patterns.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for analysis
            limit: Number of candles to analyze
            
        Returns:
            Dict with analysis results
        """
        try:
            result = self.detector.analyze_enhanced_stop_sweep(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                ohlcv_data=ohlcv_data
            )
            
            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "analysis": result,
                "signal": "NEUTRAL",  # Would be extracted from actual analysis
                "pattern_detected": "stop_run" in result.lower(),
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "timeframe": timeframe
            }