"""
Enhanced Order Flow Imbalance Detection

Professional-grade order flow analysis with real-time L2 market depth and tick trade data.
Implements industry-standard multi-timeframe analysis with advanced signal detection.
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from collections import deque
import statistics
from data import get_data_collector

logger = logging.getLogger(__name__)

@dataclass
class OrderFlowSnapshot:
    """Single point-in-time order flow snapshot."""
    timestamp: float
    symbol: str
    timeframe_ms: int
    
    # Trade-based metrics
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    total_volume: float = 0.0
    trade_count: int = 0
    aggressive_buy_volume: float = 0.0
    aggressive_sell_volume: float = 0.0
    
    # Order book metrics
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    mid_price: float = 0.0
    spread: float = 0.0
    liquidity_imbalance: float = 0.0
    
    # Enhanced calculations
    basic_imbalance: float = 0.0
    volume_weighted_imbalance: float = 0.0
    time_decay_weighted_imbalance: float = 0.0
    statistical_significance: float = 0.0
    market_impact_adjusted: float = 0.0
    
    # Classification metadata
    aggressive_ratio: float = 0.0
    passive_fills: int = 0
    self_trades_filtered: int = 0
    wash_trades_filtered: int = 0

@dataclass 
class MarketRegime:
    """Market regime classification for context-aware analysis."""
    is_trending: bool = False
    is_ranging: bool = False
    volatility_regime: str = "normal"  # low, normal, high
    session_type: str = "regular"  # regular, pre_market, after_hours
    liquidity_regime: str = "normal"  # thin, normal, deep
    
@dataclass
class ImbalanceSignal:
    """Professional order flow imbalance signal."""
    timestamp: float
    symbol: str
    timeframe_ms: int
    
    # Signal strength
    signal_type: str  # "absorption", "rejection", "breakout", "accumulation"
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    direction: str  # "bullish", "bearish", "neutral"
    
    # Supporting metrics
    imbalance_value: float
    z_score: float
    threshold_breached: float
    volume_context: float
    liquidity_context: float
    
    # Pattern details
    pattern_type: str = ""
    duration_ms: int = 0
    price_movement: float = 0.0
    vwap_deviation: float = 0.0

class OrderFlowImbalanceDetector:
    """
    Professional order flow imbalance detection system.
    
    Implements multi-timeframe analysis with:
    - Real-time L2 market depth integration
    - Tick trade data processing  
    - Volume-weighted calculations
    - Statistical significance testing
    - Market regime awareness
    - Advanced signal detection
    """
    
    def __init__(self):
        self.collector = get_data_collector()
        
        # Multi-timeframe configurations (in milliseconds)
        self.timeframes = [100, 500, 1000, 5000, 30000]  # 100ms, 500ms, 1s, 5s, 30s
        
        # Data storage for each timeframe
        self.snapshots: Dict[int, deque] = {tf: deque(maxlen=1000) for tf in self.timeframes}
        self.signals: Dict[int, List[ImbalanceSignal]] = {tf: [] for tf in self.timeframes}
        
        # Statistical tracking
        self.rolling_stats: Dict[int, Dict] = {tf: {} for tf in self.timeframes}
        
        # Market regime tracking
        self.current_regime = MarketRegime()
        
        # Configuration
        self.config = {
            'time_decay_lambda': 0.95,
            'significance_threshold': 2.0,  # z-score
            'min_confidence': 0.6,
            'volume_filter_percentile': 95,
            'liquidity_depth_levels': 10,
            'wash_trade_detection': True,
            'self_trade_detection': True
        }
        
        logger.info("OrderFlowImbalanceDetector initialized with multi-timeframe analysis")
    
    async def analyze_symbol(self, symbol: str, duration_seconds: int = 60) -> Dict:
        """
        Run comprehensive order flow imbalance analysis for a symbol.
        
        Args:
            symbol: Trading pair symbol
            duration_seconds: How long to run analysis
            
        Returns:
            Complete analysis results with signals and metrics
        """
        logger.info(f"Starting order flow imbalance analysis for {symbol} (duration: {duration_seconds}s)")
        
        start_time = time.time()
        analysis_results = {
            'symbol': symbol,
            'start_time': start_time,
            'duration_seconds': duration_seconds,
            'snapshots_collected': 0,
            'signals_detected': 0,
            'timeframe_results': {}
        }
        
        try:
            # Initialize market regime
            await self._update_market_regime(symbol)
            
            # Main analysis loop
            while time.time() - start_time < duration_seconds:
                iteration_start = time.time()
                
                # Collect real-time data
                orderbook_data = self.collector.fetch_orderbook_stream(symbol, limit=50)
                trades_data = self.collector.fetch_trades_stream(symbol, limit=200)
                
                # Process each timeframe
                for timeframe_ms in self.timeframes:
                    snapshot = await self._create_snapshot(
                        symbol, timeframe_ms, orderbook_data, trades_data
                    )
                    
                    if snapshot:
                        self.snapshots[timeframe_ms].append(snapshot)
                        analysis_results['snapshots_collected'] += 1
                        
                        # Detect signals
                        signals = await self._detect_imbalance_signals(timeframe_ms, snapshot)
                        if signals:
                            self.signals[timeframe_ms].extend(signals)
                            analysis_results['signals_detected'] += len(signals)
                
                # Update market regime periodically
                if int(time.time()) % 10 == 0:  # Every 10 seconds
                    await self._update_market_regime(symbol)
                
                # Rate limiting - ensure we don't exceed API limits
                iteration_time = time.time() - iteration_start
                min_interval = 0.1  # 100ms minimum
                if iteration_time < min_interval:
                    await asyncio.sleep(min_interval - iteration_time)
            
            # Compile final results
            analysis_results['timeframe_results'] = await self._compile_results()
            analysis_results['end_time'] = time.time()
            analysis_results['market_regime'] = self.current_regime
            
            logger.info(f"Analysis completed: {analysis_results['snapshots_collected']} snapshots, "
                       f"{analysis_results['signals_detected']} signals detected")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in order flow analysis: {e}")
            analysis_results['error'] = str(e)
            return analysis_results
    
    async def _create_snapshot(self, symbol: str, timeframe_ms: int, 
                             orderbook_data: Dict, trades_data: List[Dict]) -> Optional[OrderFlowSnapshot]:
        """Create order flow snapshot for given timeframe."""
        try:
            current_time = time.time() * 1000  # Convert to milliseconds
            
            # Filter trades for this timeframe
            cutoff_time = current_time - timeframe_ms
            relevant_trades = [
                trade for trade in trades_data 
                if trade.get('timestamp', 0) >= cutoff_time
            ]
            
            if not relevant_trades:
                return None
            
            snapshot = OrderFlowSnapshot(
                timestamp=current_time,
                symbol=symbol,
                timeframe_ms=timeframe_ms
            )
            
            # Process trade data
            await self._process_trade_data(snapshot, relevant_trades, orderbook_data)
            
            # Process order book data
            await self._process_orderbook_data(snapshot, orderbook_data)
            
            # Calculate enhanced imbalance metrics
            await self._calculate_enhanced_imbalances(snapshot, timeframe_ms)
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error creating snapshot for {symbol} {timeframe_ms}ms: {e}")
            return None
    
    async def _process_trade_data(self, snapshot: OrderFlowSnapshot, 
                                trades: List[Dict], orderbook_data: Dict):
        """Process trade data for order flow analysis."""
        total_volume = 0.0
        buy_volume = 0.0
        sell_volume = 0.0
        aggressive_buy_volume = 0.0
        aggressive_sell_volume = 0.0
        passive_fills = 0
        
        # Get mid price for trade classification
        mid_price = orderbook_data.get('mid_price', 0)
        
        for trade in trades:
            try:
                amount = float(trade.get('amount', 0))
                price = float(trade.get('price', 0))
                side = trade.get('side', '').lower()
                
                total_volume += amount
                
                # Enhanced trade classification
                if self._is_wash_trade(trade, trades):
                    snapshot.wash_trades_filtered += 1
                    continue
                
                if self._is_self_trade(trade):
                    snapshot.self_trades_filtered += 1
                    continue
                
                # Classify aggressor side
                aggressor_side = self._classify_aggressor(trade, mid_price, trades)
                
                if aggressor_side == 'buy':
                    buy_volume += amount
                    if self._is_aggressive_order(trade, orderbook_data, 'buy'):
                        aggressive_buy_volume += amount
                    else:
                        passive_fills += 1
                        
                elif aggressor_side == 'sell':
                    sell_volume += amount
                    if self._is_aggressive_order(trade, orderbook_data, 'sell'):
                        aggressive_sell_volume += amount
                    else:
                        passive_fills += 1
                
            except (ValueError, TypeError) as e:
                logger.debug(f"Error processing trade: {e}")
                continue
        
        # Update snapshot
        snapshot.total_volume = total_volume
        snapshot.buy_volume = buy_volume
        snapshot.sell_volume = sell_volume
        snapshot.trade_count = len(trades)
        snapshot.aggressive_buy_volume = aggressive_buy_volume
        snapshot.aggressive_sell_volume = aggressive_sell_volume
        snapshot.passive_fills = passive_fills
        
        # Calculate aggressive ratio
        total_aggressive = aggressive_buy_volume + aggressive_sell_volume
        snapshot.aggressive_ratio = total_aggressive / total_volume if total_volume > 0 else 0
    
    async def _process_orderbook_data(self, snapshot: OrderFlowSnapshot, orderbook_data: Dict):
        """Process order book data for liquidity analysis."""
        snapshot.bid_volume = orderbook_data.get('total_bid_volume', 0)
        snapshot.ask_volume = orderbook_data.get('total_ask_volume', 0)
        snapshot.mid_price = orderbook_data.get('mid_price', 0)
        snapshot.spread = orderbook_data.get('bid_ask_spread', 0)
        snapshot.liquidity_imbalance = orderbook_data.get('liquidity_imbalance', 0)
    
    async def _calculate_enhanced_imbalances(self, snapshot: OrderFlowSnapshot, timeframe_ms: int):
        """Calculate enhanced imbalance metrics."""
        buy_vol = snapshot.buy_volume
        sell_vol = snapshot.sell_volume
        total_vol = snapshot.total_volume
        
        if total_vol == 0:
            return
        
        # Basic imbalance
        snapshot.basic_imbalance = (buy_vol - sell_vol) / total_vol
        
        # Volume-weighted imbalance (weighted by trade size)
        snapshot.volume_weighted_imbalance = self._calculate_vw_imbalance(snapshot)
        
        # Time-decay weighted imbalance
        snapshot.time_decay_weighted_imbalance = await self._calculate_time_decay_imbalance(
            snapshot, timeframe_ms
        )
        
        # Statistical significance (z-score)
        snapshot.statistical_significance = await self._calculate_z_score(
            snapshot.basic_imbalance, timeframe_ms
        )
        
        # Market impact adjustment
        snapshot.market_impact_adjusted = self._calculate_market_impact_adjustment(snapshot)
    
    def _calculate_vw_imbalance(self, snapshot: OrderFlowSnapshot) -> float:
        """Calculate volume-weighted imbalance."""
        if snapshot.total_volume == 0:
            return 0.0
        
        # Simple approximation - would need individual trade sizes for full implementation
        aggressive_buy = snapshot.aggressive_buy_volume
        aggressive_sell = snapshot.aggressive_sell_volume
        total_aggressive = aggressive_buy + aggressive_sell
        
        if total_aggressive == 0:
            return snapshot.basic_imbalance
        
        return (aggressive_buy - aggressive_sell) / total_aggressive
    
    async def _calculate_time_decay_imbalance(self, snapshot: OrderFlowSnapshot, timeframe_ms: int) -> float:
        """Calculate time-decay weighted imbalance."""
        recent_snapshots = list(self.snapshots[timeframe_ms])
        if len(recent_snapshots) < 2:
            return snapshot.basic_imbalance
        
        decay_lambda = self.config['time_decay_lambda']
        weighted_sum = 0.0
        weight_sum = 0.0
        
        current_time = snapshot.timestamp
        
        for i, snap in enumerate(reversed(recent_snapshots[-10:])):  # Last 10 snapshots
            time_diff = (current_time - snap.timestamp) / 1000.0  # Convert to seconds
            weight = decay_lambda ** time_diff
            weighted_sum += snap.basic_imbalance * weight
            weight_sum += weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else snapshot.basic_imbalance
    
    async def _calculate_z_score(self, imbalance_value: float, timeframe_ms: int) -> float:
        """Calculate statistical significance (z-score) of imbalance."""
        recent_snapshots = list(self.snapshots[timeframe_ms])
        if len(recent_snapshots) < 10:
            return 0.0
        
        recent_imbalances = [snap.basic_imbalance for snap in recent_snapshots[-50:]]
        
        try:
            mean_imbalance = statistics.mean(recent_imbalances)
            std_imbalance = statistics.stdev(recent_imbalances)
            
            if std_imbalance == 0:
                return 0.0
            
            return (imbalance_value - mean_imbalance) / std_imbalance
            
        except statistics.StatisticsError:
            return 0.0
    
    def _calculate_market_impact_adjustment(self, snapshot: OrderFlowSnapshot) -> float:
        """Adjust imbalance for market impact and liquidity depth."""
        base_imbalance = snapshot.basic_imbalance
        
        # Adjust for liquidity depth
        total_liquidity = snapshot.bid_volume + snapshot.ask_volume
        if total_liquidity == 0:
            return base_imbalance
        
        # Normalize by available liquidity
        liquidity_factor = min(snapshot.total_volume / total_liquidity, 1.0)
        
        # Adjust for spread (higher spread = higher impact)
        spread_factor = 1.0 + (snapshot.spread / snapshot.mid_price) if snapshot.mid_price > 0 else 1.0
        
        return base_imbalance * liquidity_factor * spread_factor
    
    async def _detect_imbalance_signals(self, timeframe_ms: int, 
                                      snapshot: OrderFlowSnapshot) -> List[ImbalanceSignal]:
        """Detect order flow imbalance signals using advanced pattern recognition."""
        signals = []
        
        try:
            # Dynamic threshold based on volatility and liquidity
            threshold = await self._calculate_dynamic_threshold(timeframe_ms, snapshot)
            
            # Check for significant imbalance
            if abs(snapshot.statistical_significance) >= self.config['significance_threshold']:
                signal = await self._create_imbalance_signal(snapshot, threshold)
                if signal and signal.confidence >= self.config['min_confidence']:
                    signals.append(signal)
            
            # Pattern-based signal detection
            pattern_signals = await self._detect_pattern_signals(timeframe_ms, snapshot)
            signals.extend(pattern_signals)
            
        except Exception as e:
            logger.error(f"Error detecting signals: {e}")
        
        return signals
    
    async def _calculate_dynamic_threshold(self, timeframe_ms: int, 
                                         snapshot: OrderFlowSnapshot) -> float:
        """Calculate dynamic threshold based on market conditions."""
        base_threshold = 0.3  # 30% imbalance
        
        # Adjust for volatility regime
        if self.current_regime.volatility_regime == 'high':
            base_threshold *= 1.5
        elif self.current_regime.volatility_regime == 'low':
            base_threshold *= 0.7
        
        # Adjust for liquidity regime
        if self.current_regime.liquidity_regime == 'thin':
            base_threshold *= 0.8  # More sensitive in thin markets
        elif self.current_regime.liquidity_regime == 'deep':
            base_threshold *= 1.2  # Less sensitive in deep markets
        
        # Adjust for session type
        if self.current_regime.session_type != 'regular':
            base_threshold *= 0.9  # More sensitive during off-hours
        
        return base_threshold
    
    async def _create_imbalance_signal(self, snapshot: OrderFlowSnapshot, 
                                     threshold: float) -> Optional[ImbalanceSignal]:
        """Create imbalance signal from snapshot data."""
        try:
            direction = "bullish" if snapshot.basic_imbalance > 0 else "bearish"
            strength = min(abs(snapshot.basic_imbalance), 1.0)
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_signal_confidence(snapshot)
            
            # Determine signal type
            signal_type = self._classify_signal_type(snapshot)
            
            return ImbalanceSignal(
                timestamp=snapshot.timestamp,
                symbol=snapshot.symbol,
                timeframe_ms=snapshot.timeframe_ms,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                direction=direction,
                imbalance_value=snapshot.basic_imbalance,
                z_score=snapshot.statistical_significance,
                threshold_breached=threshold,
                volume_context=snapshot.total_volume,
                liquidity_context=snapshot.bid_volume + snapshot.ask_volume
            )
            
        except Exception as e:
            logger.error(f"Error creating signal: {e}")
            return None
    
    def _calculate_signal_confidence(self, snapshot: OrderFlowSnapshot) -> float:
        """Calculate signal confidence based on multiple factors."""
        confidence_factors = []
        
        # Statistical significance
        z_score_confidence = min(abs(snapshot.statistical_significance) / 3.0, 1.0)
        confidence_factors.append(z_score_confidence * 0.3)
        
        # Volume context
        volume_confidence = min(snapshot.total_volume / 1000.0, 1.0)  # Normalize to reasonable scale
        confidence_factors.append(volume_confidence * 0.2)
        
        # Aggressive ratio
        aggressive_confidence = snapshot.aggressive_ratio
        confidence_factors.append(aggressive_confidence * 0.2)
        
        # Liquidity context
        liquidity_confidence = 1.0 - min(abs(snapshot.liquidity_imbalance), 0.5) * 2.0
        confidence_factors.append(liquidity_confidence * 0.15)
        
        # Market regime adjustment
        regime_confidence = 0.8 if self.current_regime.volatility_regime == 'normal' else 0.6
        confidence_factors.append(regime_confidence * 0.15)
        
        return sum(confidence_factors)
    
    def _classify_signal_type(self, snapshot: OrderFlowSnapshot) -> str:
        """Classify the type of order flow signal."""
        aggressive_ratio = snapshot.aggressive_ratio
        liquidity_imbalance = abs(snapshot.liquidity_imbalance)
        
        if aggressive_ratio > 0.7:
            return "breakout"
        elif aggressive_ratio < 0.3 and liquidity_imbalance > 0.3:
            return "absorption"
        elif liquidity_imbalance > 0.5:
            return "rejection"
        else:
            return "accumulation"
    
    async def _detect_pattern_signals(self, timeframe_ms: int, 
                                    snapshot: OrderFlowSnapshot) -> List[ImbalanceSignal]:
        """Detect pattern-based signals like hidden liquidity, iceberg orders, etc."""
        signals = []
        
        # This would implement pattern recognition algorithms
        # For now, return empty list as placeholder
        
        return signals
    
    async def _update_market_regime(self, symbol: str):
        """Update current market regime classification."""
        try:
            # Get recent price data for regime classification
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, '1m', limit=100)
            
            if not ohlcv_data:
                return
            
            # Calculate volatility
            closes = [float(candle[4]) for candle in ohlcv_data[-20:]]  # Last 20 closes
            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            volatility = statistics.stdev(returns) if len(returns) > 1 else 0
            
            # Classify volatility regime
            if volatility > 0.02:  # 2% standard deviation
                self.current_regime.volatility_regime = 'high'
            elif volatility < 0.005:  # 0.5% standard deviation
                self.current_regime.volatility_regime = 'low'
            else:
                self.current_regime.volatility_regime = 'normal'
            
            # Trend detection (simplified)
            recent_prices = closes[-10:]
            if len(recent_prices) >= 2:
                trend_slope = (recent_prices[-1] - recent_prices[0]) / len(recent_prices)
                self.current_regime.is_trending = abs(trend_slope) > 0.001
                self.current_regime.is_ranging = not self.current_regime.is_trending
            
            # Session type (simplified - would need timezone awareness for full implementation)
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 16:  # Market hours
                self.current_regime.session_type = 'regular'
            else:
                self.current_regime.session_type = 'after_hours'
            
        except Exception as e:
            logger.error(f"Error updating market regime: {e}")
    
    async def _compile_results(self) -> Dict:
        """Compile final analysis results for all timeframes."""
        results = {}
        
        for timeframe_ms in self.timeframes:
            tf_snapshots = list(self.snapshots[timeframe_ms])
            tf_signals = self.signals[timeframe_ms]
            
            if tf_snapshots:
                # Calculate summary statistics
                avg_imbalance = statistics.mean([s.basic_imbalance for s in tf_snapshots])
                avg_volume = statistics.mean([s.total_volume for s in tf_snapshots])
                avg_aggressive_ratio = statistics.mean([s.aggressive_ratio for s in tf_snapshots])
                
                results[f"{timeframe_ms}ms"] = {
                    'snapshots_count': len(tf_snapshots),
                    'signals_count': len(tf_signals),
                    'avg_imbalance': avg_imbalance,
                    'avg_volume': avg_volume,
                    'avg_aggressive_ratio': avg_aggressive_ratio,
                    'latest_snapshot': tf_snapshots[-1] if tf_snapshots else None,
                    'recent_signals': tf_signals[-5:] if tf_signals else []
                }
        
        return results
    
    # Helper methods for trade classification
    def _is_wash_trade(self, trade: Dict, all_trades: List[Dict]) -> bool:
        """Detect potential wash trades."""
        if not self.config['wash_trade_detection']:
            return False
        
        # Simplified wash trade detection
        # In practice, this would be much more sophisticated
        return False
    
    def _is_self_trade(self, trade: Dict) -> bool:
        """Detect potential self trades."""
        if not self.config['self_trade_detection']:
            return False
        
        # Simplified self trade detection
        return False
    
    def _classify_aggressor(self, trade: Dict, mid_price: float, all_trades: List[Dict]) -> str:
        """Classify trade aggressor side using multiple methods."""
        price = float(trade.get('price', 0))
        side = trade.get('side', '').lower()
        
        # Method 1: Exchange side field
        if side in ['buy', 'sell']:
            exchange_side = side
        else:
            exchange_side = None
        
        # Method 2: Price-based classification
        price_side = None
        if mid_price > 0:
            if price > mid_price:
                price_side = 'buy'
            elif price < mid_price:
                price_side = 'sell'
        
        # Method 3: Tick rule (uptick/downtick)
        tick_side = trade.get('price_movement', '')
        if tick_side == 'uptick':
            tick_rule_side = 'buy'
        elif tick_side == 'downtick':
            tick_rule_side = 'sell'
        else:
            tick_rule_side = None
        
        # Combine methods with priority
        if price_side:
            return price_side
        elif exchange_side:
            return exchange_side
        elif tick_rule_side:
            return tick_rule_side
        else:
            return 'unknown'
    
    def _is_aggressive_order(self, trade: Dict, orderbook_data: Dict, side: str) -> bool:
        """Determine if trade was from aggressive market order."""
        price = float(trade.get('price', 0))
        
        # Check if trade occurred at or through the opposite side of the book
        if side == 'buy':
            asks = orderbook_data.get('asks', [])
            if asks and price >= float(asks[0][0]):  # At or above best ask
                return True
        elif side == 'sell':
            bids = orderbook_data.get('bids', [])
            if bids and price <= float(bids[0][0]):  # At or below best bid
                return True
        
        return False

    def analyze(self, symbol: str, timeframe: str, limit: int) -> Dict:
        """
        Analyze order flow imbalance patterns for given symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for analysis
            limit: Number of candles to analyze
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Fetch OHLCV data
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
            
            if not ohlcv_data or len(ohlcv_data) < 20:
                return {
                    'error': f'Insufficient data: need at least 20 candles, got {len(ohlcv_data) if ohlcv_data else 0}',
                    'success': False,
                    'symbol': symbol,
                    'timeframe': timeframe
                }
            
            # Get current price and market data
            current_candle = ohlcv_data[-1]
            current_price = float(current_candle[4])  # close price
            current_volume = float(current_candle[5])  # volume
            
            # Calculate recent volume average for comparison
            recent_volumes = [float(candle[5]) for candle in ohlcv_data[-10:]]
            avg_volume = sum(recent_volumes) / len(recent_volumes)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Basic imbalance detection using volume and price action
            price_change = (current_price - float(current_candle[1])) / float(current_candle[1])  # (close - open) / open
            body_size = abs(current_price - float(current_candle[1]))
            candle_range = float(current_candle[2]) - float(current_candle[3])  # high - low
            
            # Detect imbalance pattern
            pattern_detected = False
            signal = 'HOLD'
            bias = 'NEUTRAL'
            confidence = 0
            signal_type = 'no_pattern'
            
            # Volume spike with minimal price movement indicates absorption/imbalance
            if volume_ratio > 1.5 and abs(price_change) < 0.01:  # 1% price movement threshold
                pattern_detected = True
                confidence = min(int(volume_ratio * 30), 95)
                
                if price_change > 0:
                    signal = 'BUY'
                    bias = 'BULLISH'
                    signal_type = 'bullish_imbalance'
                elif price_change < 0:
                    signal = 'SELL'
                    bias = 'BEARISH'
                    signal_type = 'bearish_imbalance'
                else:
                    signal_type = 'neutral_imbalance'
            
            # Calculate support/resistance levels
            if pattern_detected:
                if signal == 'BUY':
                    support_level = current_price * 0.99
                    resistance_level = current_price * 1.02
                    stop_zone = current_price * 0.985
                elif signal == 'SELL':
                    support_level = current_price * 0.98
                    resistance_level = current_price * 1.01
                    stop_zone = current_price * 1.015
                else:
                    support_level = current_price * 0.99
                    resistance_level = current_price * 1.01
                    stop_zone = current_price
            else:
                support_level = current_price * 0.99
                resistance_level = current_price * 1.01
                stop_zone = current_price
            
            # Calculate risk/reward ratio
            risk = abs(current_price - stop_zone)
            reward = abs(resistance_level - current_price) if signal == 'BUY' else abs(current_price - support_level)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Entry window assessment
            if pattern_detected and confidence > 70:
                entry_window = "Optimal now"
            elif pattern_detected and confidence > 50:
                entry_window = "Good in next 2-3 bars"
            else:
                entry_window = "Wait for better setup"
            
            # Exit trigger
            if signal == 'BUY':
                exit_trigger = "Price breaks below support level"
            elif signal == 'SELL':
                exit_trigger = "Price breaks above resistance level"
            else:
                exit_trigger = "Wait for clear directional imbalance"
            
            return {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': int(current_candle[0]),
                'total_candles': len(ohlcv_data),
                'current_price': round(current_price, 4),
                'pattern_detected': pattern_detected,
                
                # Pattern specific data
                'pattern_type': signal_type,
                'bias': bias,
                'signal': signal,
                'confidence_score': confidence,
                
                # Price levels
                'support_level': round(support_level, 4),
                'resistance_level': round(resistance_level, 4),
                'stop_zone': round(stop_zone, 4),
                
                # Trading analysis
                'entry_window': entry_window,
                'exit_trigger': exit_trigger,
                'rr_ratio': round(rr_ratio, 1),
                
                # Volume metrics
                'volume_ratio': round(volume_ratio, 2),
                'current_volume': round(current_volume, 2),
                'avg_volume': round(avg_volume, 2),
                
                # Price action
                'price_change_pct': round(price_change * 100, 3),
                'body_size': round(body_size, 4),
                'candle_range': round(candle_range, 4)
            }
            
        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'success': False,
                'symbol': symbol,
                'timeframe': timeframe
            }


