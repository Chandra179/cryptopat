"""
Advanced Order Book Heatmap & Market Microstructure Analysis

This module implements real-time Level 2 market depth analysis with:
- High-frequency order book snapshots (100-500ms intervals)
- Industry-standard microstructure analytics (VWAP, CVD, OFI)
- Multi-dimensional time-series matrix storage
- Advanced signal detection for institutional patterns
- Market regime classification
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import os
from collections import deque
import time
from threading import Thread, Event
import ccxt.pro as ccxtpro

from data import get_data_collector

logger = logging.getLogger(__name__)

@dataclass
class OrderBookSnapshot:
    """Single order book snapshot with metadata"""
    timestamp: int
    symbol: str
    bids: List[List[float]]  # [[price, quantity], ...]
    asks: List[List[float]]  # [[price, quantity], ...]
    mid_price: float
    spread: float
    total_bid_volume: float
    total_ask_volume: float
    vwap: float = 0.0
    cvd: float = 0.0
    ofi: float = 0.0
    liquidity_depth: float = 0.0

@dataclass
class MarketMicrostructureMetrics:
    """Market microstructure analytics container"""
    timestamp: int
    symbol: str
    vwap: float
    cvd: float  # Cumulative Volume Delta
    ofi: float  # Order Flow Imbalance
    bid_ask_imbalance: float
    liquidity_depth: float
    depth_volatility: float
    liquidity_concentration: float
    market_maker_ratio: float
    spread_volatility: float
    volume_weighted_spread: float

@dataclass
class InstitutionalSignal:
    """Detected institutional trading pattern"""
    timestamp: int
    signal_type: str  # 'iceberg', 'spoofing', 'liquidity_hunting', 'hidden_liquidity'
    confidence: float
    price_level: float
    volume: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class OrderBookHeatmap:
    """
    Advanced order book analysis with real-time streaming and microstructure analytics.
    
    Features:
    - Real-time L2 market depth streaming
    - High-frequency snapshots (100-500ms)
    - VWAP, CVD, OFI calculations
    - Multi-dimensional time-series storage
    - Institutional pattern detection
    - Market regime classification
    """
    
    def __init__(self, 
                 symbol: str = 'XRP/USDT',
                 snapshot_interval_ms: int = 250,
                 depth_levels: int = 50,
                 history_minutes: int = 60):
        """
        Initialize OrderBookHeatmap analyzer.
        
        Args:
            symbol: Trading pair to analyze
            snapshot_interval_ms: Interval between snapshots (100-500ms)
            depth_levels: Number of price levels to capture
            history_minutes: Minutes of history to maintain
        """
        self.symbol = symbol
        self.snapshot_interval_ms = snapshot_interval_ms
        self.depth_levels = depth_levels
        self.history_minutes = history_minutes
        
        # Data storage
        self.snapshots = deque(maxlen=int(history_minutes * 60 * 1000 / snapshot_interval_ms))
        self.metrics_history = deque(maxlen=int(history_minutes * 60 * 1000 / snapshot_interval_ms))
        self.signals_history = deque(maxlen=1000)
        
        # Multi-dimensional matrix for heatmap
        self.price_matrix = None  # Shape: (time_steps, price_levels)
        self.volume_matrix = None
        self.order_count_matrix = None
        self.cvd_matrix = None
        self.ofi_matrix = None
        
        # Streaming state
        self.is_streaming = False
        self.stream_event = Event()
        self.exchange_pro = None
        
        # Analytics state
        self.last_trades = deque(maxlen=1000)
        self.vwap_window = deque(maxlen=100)
        self.cvd_accumulator = 0.0
        
        # Pattern detection state
        self.order_flow_history = deque(maxlen=200)
        self.liquidity_events = deque(maxlen=100)
        
        # Initialize data collector
        self.data_collector = get_data_collector()
        
        logger.info(f"OrderBookHeatmap initialized for {symbol} with {snapshot_interval_ms}ms intervals")
    
    async def start_streaming(self):
        """Start real-time order book streaming with high-frequency snapshots."""
        try:
            # Initialize Pro exchange for streaming
            self.exchange_pro = ccxtpro.binance({
                'apiKey': '',  # Not needed for public data
                'secret': '',
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            self.is_streaming = True
            self.stream_event.set()
            
            logger.info(f"Starting order book stream for {self.symbol}")
            
            # Start concurrent tasks with proper exception handling
            tasks = [
                asyncio.create_task(self._stream_orderbook()),
                asyncio.create_task(self._stream_trades()),
                asyncio.create_task(self._process_snapshots()),
                asyncio.create_task(self._detect_patterns())
            ]
            
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                logger.info("Streaming tasks cancelled")
            finally:
                # Cancel any remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
            
        except Exception as e:
            logger.error(f"Error starting streaming: {e}")
            self.is_streaming = False
            raise
    
    async def _stream_orderbook(self):
        """Stream real-time order book updates."""
        while self.is_streaming:
            try:
                orderbook = await self.exchange_pro.watch_order_book(self.symbol, self.depth_levels)
                
                # Create snapshot
                snapshot = self._create_snapshot(orderbook)
                self.snapshots.append(snapshot)
                
                # Maintain matrix dimensions
                self._update_matrices(snapshot)
                
                await asyncio.sleep(self.snapshot_interval_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Error in order book stream: {e}")
                await asyncio.sleep(1)
    
    async def _stream_trades(self):
        """Stream real-time trades for CVD and flow analysis."""
        while self.is_streaming:
            try:
                trades = await self.exchange_pro.watch_trades(self.symbol)
                
                for trade in trades:
                    self.last_trades.append({
                        'timestamp': trade['timestamp'],
                        'price': trade['price'],
                        'amount': trade['amount'],
                        'side': trade['side'],
                        'cost': trade['cost']
                    })
                
                await asyncio.sleep(0.1)  # 100ms for trade updates
                
            except Exception as e:
                logger.error(f"Error in trades stream: {e}")
                await asyncio.sleep(1)
    
    def _create_snapshot(self, orderbook: Dict) -> OrderBookSnapshot:
        """Create structured snapshot from raw order book data."""
        bids = orderbook['bids'][:self.depth_levels]
        asks = orderbook['asks'][:self.depth_levels]
        
        if not bids or not asks:
            return None
        
        # Calculate basic metrics
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        total_bid_volume = sum(bid[1] for bid in bids)
        total_ask_volume = sum(ask[1] for ask in asks)
        
        # Calculate VWAP from recent trades
        vwap = self._calculate_vwap()
        
        # Calculate CVD
        cvd = self._calculate_cvd()
        
        # Calculate OFI
        ofi = self._calculate_ofi(total_bid_volume, total_ask_volume)
        
        # Calculate liquidity depth
        liquidity_depth = self._calculate_liquidity_depth(bids, asks, mid_price)
        
        return OrderBookSnapshot(
            timestamp=int(time.time() * 1000),
            symbol=self.symbol,
            bids=bids,
            asks=asks,
            mid_price=mid_price,
            spread=spread,
            total_bid_volume=total_bid_volume,
            total_ask_volume=total_ask_volume,
            vwap=vwap,
            cvd=cvd,
            ofi=ofi,
            liquidity_depth=liquidity_depth
        )
    
    def _calculate_vwap(self) -> float:
        """Calculate Volume Weighted Average Price from recent trades."""
        if len(self.last_trades) < 10:
            return 0.0
        
        # Use trades from last 5 minutes
        cutoff_time = time.time() * 1000 - 5 * 60 * 1000
        recent_trades = [t for t in self.last_trades if t['timestamp'] >= cutoff_time]
        
        if not recent_trades:
            return 0.0
        
        total_volume = sum(t['amount'] for t in recent_trades)
        if total_volume == 0:
            return 0.0
        
        vwap = sum(t['price'] * t['amount'] for t in recent_trades) / total_volume
        return vwap
    
    def _calculate_cvd(self) -> float:
        """Calculate Cumulative Volume Delta (buyer vs seller aggression)."""
        if len(self.last_trades) < 2:
            return self.cvd_accumulator
        
        # Get trades since last calculation
        last_timestamp = getattr(self, '_last_cvd_timestamp', 0)
        new_trades = [t for t in self.last_trades if t['timestamp'] > last_timestamp]
        
        for trade in new_trades:
            if trade['side'] == 'buy':
                self.cvd_accumulator += trade['amount']
            else:
                self.cvd_accumulator -= trade['amount']
        
        if new_trades:
            self._last_cvd_timestamp = max(t['timestamp'] for t in new_trades)
        
        return self.cvd_accumulator
    
    def _calculate_ofi(self, bid_volume: float, ask_volume: float) -> float:
        """Calculate Order Flow Imbalance."""
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total_volume
    
    def _calculate_liquidity_depth(self, bids: List, asks: List, mid_price: float) -> float:
        """Calculate average liquidity depth around mid price."""
        depth_range = mid_price * 0.001  # 0.1% from mid price
        
        relevant_bids = [bid for bid in bids if bid[0] >= mid_price - depth_range]
        relevant_asks = [ask for ask in asks if ask[0] <= mid_price + depth_range]
        
        total_depth = sum(bid[1] for bid in relevant_bids) + sum(ask[1] for ask in relevant_asks)
        return total_depth
    
    def _update_matrices(self, snapshot: OrderBookSnapshot):
        """Update multi-dimensional time-series matrices."""
        if not snapshot:
            return
        
        # Initialize matrices if needed
        if self.price_matrix is None:
            max_history = int(self.history_minutes * 60 * 1000 / self.snapshot_interval_ms)
            self.price_matrix = np.zeros((max_history, self.depth_levels * 2))
            self.volume_matrix = np.zeros((max_history, self.depth_levels * 2))
            self.order_count_matrix = np.zeros((max_history, self.depth_levels * 2))
            self.cvd_matrix = np.zeros(max_history)
            self.ofi_matrix = np.zeros(max_history)
        
        # Shift matrices and add new data
        self.price_matrix = np.roll(self.price_matrix, -1, axis=0)
        self.volume_matrix = np.roll(self.volume_matrix, -1, axis=0)
        self.order_count_matrix = np.roll(self.order_count_matrix, -1, axis=0)
        self.cvd_matrix = np.roll(self.cvd_matrix, -1)
        self.ofi_matrix = np.roll(self.ofi_matrix, -1)
        
        # Normalize price levels as percentage from mid price
        mid_price = snapshot.mid_price
        
        # Fill bid data (negative indices from mid price)
        for i, (price, volume) in enumerate(snapshot.bids[:self.depth_levels]):
            if i < self.depth_levels:
                price_pct = (price - mid_price) / mid_price
                self.price_matrix[-1, i] = price_pct
                self.volume_matrix[-1, i] = volume
                self.order_count_matrix[-1, i] = 1  # Assume 1 order per level
        
        # Fill ask data (positive indices from mid price)
        for i, (price, volume) in enumerate(snapshot.asks[:self.depth_levels]):
            if i < self.depth_levels:
                price_pct = (price - mid_price) / mid_price
                self.price_matrix[-1, self.depth_levels + i] = price_pct
                self.volume_matrix[-1, self.depth_levels + i] = volume
                self.order_count_matrix[-1, self.depth_levels + i] = 1
        
        # Update flow metrics
        self.cvd_matrix[-1] = snapshot.cvd
        self.ofi_matrix[-1] = snapshot.ofi
    
    async def _process_snapshots(self):
        """Process snapshots to generate microstructure metrics."""
        while self.is_streaming:
            try:
                if len(self.snapshots) >= 10:
                    latest_snapshots = list(self.snapshots)[-10:]
                    metrics = self._calculate_microstructure_metrics(latest_snapshots)
                    
                    if metrics:
                        self.metrics_history.append(metrics)
                
                await asyncio.sleep(1.0)  # Process every second
                
            except Exception as e:
                logger.error(f"Error processing snapshots: {e}")
                await asyncio.sleep(1)
    
    def _calculate_microstructure_metrics(self, snapshots: List[OrderBookSnapshot]) -> MarketMicrostructureMetrics:
        """Calculate comprehensive microstructure analytics."""
        if not snapshots:
            return None
        
        latest = snapshots[-1]
        
        # Calculate volatility metrics
        spreads = [s.spread for s in snapshots]
        spread_volatility = np.std(spreads) if len(spreads) > 1 else 0.0
        
        depths = [s.liquidity_depth for s in snapshots]
        depth_volatility = np.std(depths) if len(depths) > 1 else 0.0
        
        # Calculate liquidity concentration (Herfindahl index)
        bid_volumes = [bid[1] for bid in latest.bids[:10]]
        ask_volumes = [ask[1] for ask in latest.asks[:10]]
        all_volumes = bid_volumes + ask_volumes
        total_volume = sum(all_volumes)
        
        if total_volume > 0:
            concentration_ratios = [(v / total_volume) ** 2 for v in all_volumes]
            liquidity_concentration = sum(concentration_ratios)
        else:
            liquidity_concentration = 0.0
        
        # Calculate bid-ask imbalance
        bid_ask_imbalance = (latest.total_bid_volume - latest.total_ask_volume) / (latest.total_bid_volume + latest.total_ask_volume) if (latest.total_bid_volume + latest.total_ask_volume) > 0 else 0.0
        
        # Estimate market maker ratio from trade flow
        market_maker_ratio = self._estimate_market_maker_ratio()
        
        # Volume weighted spread
        volume_weighted_spread = latest.spread * (latest.total_bid_volume + latest.total_ask_volume)
        
        return MarketMicrostructureMetrics(
            timestamp=latest.timestamp,
            symbol=latest.symbol,
            vwap=latest.vwap,
            cvd=latest.cvd,
            ofi=latest.ofi,
            bid_ask_imbalance=bid_ask_imbalance,
            liquidity_depth=latest.liquidity_depth,
            depth_volatility=depth_volatility,
            liquidity_concentration=liquidity_concentration,
            market_maker_ratio=market_maker_ratio,
            spread_volatility=spread_volatility,
            volume_weighted_spread=volume_weighted_spread
        )
    
    def _estimate_market_maker_ratio(self) -> float:
        """Estimate market maker vs taker ratio from trade patterns."""
        if len(self.last_trades) < 10:
            return 0.5
        
        # Simple heuristic: trades at bid/ask vs between spread
        recent_trades = list(self.last_trades)[-50:]
        
        # Get recent order book for reference
        if not self.snapshots:
            return 0.5
        
        latest_snapshot = self.snapshots[-1]
        best_bid = latest_snapshot.bids[0][0] if latest_snapshot.bids else 0
        best_ask = latest_snapshot.asks[0][0] if latest_snapshot.asks else 0
        
        if best_bid == 0 or best_ask == 0:
            return 0.5
        
        # Count trades at or near best bid/ask (likely market takers)
        taker_trades = 0
        for trade in recent_trades:
            price = trade['price']
            if abs(price - best_bid) < best_bid * 0.0001 or abs(price - best_ask) < best_ask * 0.0001:
                taker_trades += 1
        
        taker_ratio = taker_trades / len(recent_trades)
        market_maker_ratio = 1.0 - taker_ratio
        
        return max(0.0, min(1.0, market_maker_ratio))
    
    async def _detect_patterns(self):
        """Detect institutional trading patterns and market anomalies."""
        while self.is_streaming:
            try:
                if len(self.snapshots) >= 20:
                    # Detect various patterns
                    signals = []
                    
                    # Iceberg order detection
                    iceberg_signals = self._detect_iceberg_orders()
                    signals.extend(iceberg_signals)
                    
                    # Spoofing detection
                    spoofing_signals = self._detect_spoofing()
                    signals.extend(spoofing_signals)
                    
                    # Liquidity hunting detection
                    hunting_signals = self._detect_liquidity_hunting()
                    signals.extend(hunting_signals)
                    
                    # Hidden liquidity detection
                    hidden_signals = self._detect_hidden_liquidity()
                    signals.extend(hidden_signals)
                    
                    # Add signals to history
                    for signal in signals:
                        self.signals_history.append(signal)
                        logger.info(f"Detected {signal.signal_type}: {signal.description}")
                
                await asyncio.sleep(2.0)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Error detecting patterns: {e}")
                await asyncio.sleep(2)
    
    def _detect_iceberg_orders(self) -> List[InstitutionalSignal]:
        """Detect iceberg orders through order refill patterns."""
        signals = []
        
        if len(self.snapshots) < 10:
            return signals
        
        recent_snapshots = list(self.snapshots)[-10:]
        
        # Look for consistent large orders at same price level that refill
        price_volume_history = {}
        
        for snapshot in recent_snapshots:
            # Check bids
            for price, volume in snapshot.bids[:5]:
                price_key = round(price, 8)
                if price_key not in price_volume_history:
                    price_volume_history[price_key] = []
                price_volume_history[price_key].append(volume)
        
        # Detect refill patterns
        for price, volumes in price_volume_history.items():
            if len(volumes) >= 5:
                # Look for pattern: large -> small -> large (refill)
                avg_volume = np.mean(volumes)
                max_volume = max(volumes)
                
                if max_volume > avg_volume * 3:  # Large order threshold
                    # Check for refill pattern
                    refill_count = 0
                    for i in range(1, len(volumes)):
                        if volumes[i] > volumes[i-1] * 2:  # Significant refill
                            refill_count += 1
                    
                    if refill_count >= 2:
                        signals.append(InstitutionalSignal(
                            timestamp=int(time.time() * 1000),
                            signal_type='iceberg',
                            confidence=min(0.9, refill_count * 0.3),
                            price_level=price,
                            volume=max_volume,
                            description=f"Iceberg order detected at {price} with {refill_count} refills",
                            metadata={'refill_count': refill_count, 'avg_volume': avg_volume}
                        ))
        
        return signals
    
    def _detect_spoofing(self) -> List[InstitutionalSignal]:
        """Detect spoofing through rapid place/cancel patterns."""
        signals = []
        
        if len(self.snapshots) < 8:
            return signals
        
        recent_snapshots = list(self.snapshots)[-8:]
        
        # Track large orders that appear and disappear quickly
        for level in range(min(3, len(recent_snapshots[0].bids))):
            # Check if large order appears and disappears
            volumes = []
            prices = []
            
            for snapshot in recent_snapshots:
                if level < len(snapshot.bids):
                    prices.append(snapshot.bids[level][0])
                    volumes.append(snapshot.bids[level][1])
                else:
                    volumes.append(0)
                    prices.append(0)
            
            # Look for spoof pattern: 0 -> large -> 0
            for i in range(1, len(volumes) - 1):
                if (volumes[i-1] < volumes[i] * 0.3 and 
                    volumes[i+1] < volumes[i] * 0.3 and 
                    volumes[i] > np.mean([v for v in volumes if v > 0]) * 2):
                    
                    signals.append(InstitutionalSignal(
                        timestamp=int(time.time() * 1000),
                        signal_type='spoofing',
                        confidence=0.7,
                        price_level=prices[i],
                        volume=volumes[i],
                        description=f"Potential spoofing: large order at {prices[i]} quickly removed",
                        metadata={'duration_snapshots': 1, 'volume_ratio': volumes[i] / np.mean([v for v in volumes if v > 0])}
                    ))
        
        return signals
    
    def _detect_liquidity_hunting(self) -> List[InstitutionalSignal]:
        """Detect liquidity hunting through price movements toward large orders."""
        signals = []
        
        if len(self.snapshots) < 15:
            return signals
        
        recent_snapshots = list(self.snapshots)[-15:]
        
        # Identify large orders
        large_orders = []
        for i, snapshot in enumerate(recent_snapshots):
            avg_bid_volume = np.mean([bid[1] for bid in snapshot.bids[:5]])
            avg_ask_volume = np.mean([ask[1] for ask in snapshot.asks[:5]])
            
            # Find unusually large orders
            for price, volume in snapshot.bids[:3]:
                if volume > avg_bid_volume * 3:
                    large_orders.append({'timestamp': i, 'price': price, 'volume': volume, 'side': 'bid'})
            
            for price, volume in snapshot.asks[:3]:
                if volume > avg_ask_volume * 3:
                    large_orders.append({'timestamp': i, 'price': price, 'volume': volume, 'side': 'ask'})
        
        # Check if price moved toward large orders
        for order in large_orders:
            start_idx = order['timestamp']
            if start_idx < len(recent_snapshots) - 5:
                start_price = recent_snapshots[start_idx].mid_price
                end_price = recent_snapshots[-1].mid_price
                order_price = order['price']
                
                # Check if price moved toward the large order
                if order['side'] == 'bid' and end_price < start_price and end_price > order_price * 0.999:
                    signals.append(InstitutionalSignal(
                        timestamp=int(time.time() * 1000),
                        signal_type='liquidity_hunting',
                        confidence=0.6,
                        price_level=order_price,
                        volume=order['volume'],
                        description=f"Price moved toward large bid at {order_price}",
                        metadata={'price_movement': end_price - start_price, 'side': order['side']}
                    ))
                elif order['side'] == 'ask' and end_price > start_price and end_price < order_price * 1.001:
                    signals.append(InstitutionalSignal(
                        timestamp=int(time.time() * 1000),
                        signal_type='liquidity_hunting',
                        confidence=0.6,
                        price_level=order_price,
                        volume=order['volume'],
                        description=f"Price moved toward large ask at {order_price}",
                        metadata={'price_movement': end_price - start_price, 'side': order['side']}
                    ))
        
        return signals
    
    def _detect_hidden_liquidity(self) -> List[InstitutionalSignal]:
        """Detect hidden liquidity through execution vs visible depth mismatches."""
        signals = []
        
        if len(self.last_trades) < 20 or len(self.snapshots) < 5:
            return signals
        
        # Compare recent trade volumes with visible liquidity
        recent_trades = list(self.last_trades)[-20:]
        recent_snapshot = self.snapshots[-1]
        
        # Calculate total traded volume in recent period
        total_traded = sum(trade['amount'] for trade in recent_trades)
        
        # Calculate visible liquidity around current price
        mid_price = recent_snapshot.mid_price
        visible_liquidity = 0
        
        # Sum liquidity within 0.1% of mid price
        price_range = mid_price * 0.001
        for price, volume in recent_snapshot.bids:
            if price >= mid_price - price_range:
                visible_liquidity += volume
        
        for price, volume in recent_snapshot.asks:
            if price <= mid_price + price_range:
                visible_liquidity += volume
        
        # If traded volume significantly exceeds visible liquidity
        if total_traded > visible_liquidity * 2 and visible_liquidity > 0:
            signals.append(InstitutionalSignal(
                timestamp=int(time.time() * 1000),
                signal_type='hidden_liquidity',
                confidence=min(0.8, total_traded / visible_liquidity * 0.2),
                price_level=mid_price,
                volume=total_traded - visible_liquidity,
                description=f"Hidden liquidity detected: traded {total_traded:.2f} vs visible {visible_liquidity:.2f}",
                metadata={'visible_liquidity': visible_liquidity, 'traded_volume': total_traded}
            ))
        
        return signals
    
    def classify_market_regime(self) -> str:
        """Classify current market regime based on microstructure metrics."""
        if len(self.metrics_history) < 10:
            return "insufficient_data"
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Calculate regime indicators
        avg_spread_volatility = np.mean([m.spread_volatility for m in recent_metrics])
        avg_depth_volatility = np.mean([m.depth_volatility for m in recent_metrics])
        avg_ofi = np.mean([abs(m.ofi) for m in recent_metrics])
        avg_concentration = np.mean([m.liquidity_concentration for m in recent_metrics])
        
        # Classification thresholds (can be tuned based on market data)
        high_volatility_threshold = 0.001  # 0.1% spread volatility
        high_imbalance_threshold = 0.3
        high_concentration_threshold = 0.5
        
        # Classify regime
        if (avg_spread_volatility > high_volatility_threshold and 
            avg_depth_volatility > np.std([m.liquidity_depth for m in recent_metrics]) * 2):
            return "stress_conditions"
        elif avg_ofi > high_imbalance_threshold:
            return "directional_pressure"
        elif avg_concentration > high_concentration_threshold:
            return "low_liquidity"
        else:
            return "normal_trading"
    
    def get_heatmap_data(self, lookback_minutes: int = 30) -> Dict[str, Any]:
        """
        Generate heatmap data for visualization.
        
        Args:
            lookback_minutes: Minutes of history to include
            
        Returns:
            Dictionary with heatmap matrices and metadata
        """
        if self.price_matrix is None:
            return {"error": "No data available"}
        
        # Calculate lookback samples
        lookback_samples = int(lookback_minutes * 60 * 1000 / self.snapshot_interval_ms)
        
        # Extract relevant data
        price_data = self.price_matrix[-lookback_samples:] if lookback_samples < len(self.price_matrix) else self.price_matrix
        volume_data = self.volume_matrix[-lookback_samples:] if lookback_samples < len(self.volume_matrix) else self.volume_matrix
        cvd_data = self.cvd_matrix[-lookback_samples:] if lookback_samples < len(self.cvd_matrix) else self.cvd_matrix
        ofi_data = self.ofi_matrix[-lookback_samples:] if lookback_samples < len(self.ofi_matrix) else self.ofi_matrix
        
        # Generate timestamps
        current_time = int(time.time() * 1000)
        timestamps = [current_time - (lookback_samples - i) * self.snapshot_interval_ms for i in range(len(price_data))]
        
        return {
            "timestamps": timestamps,
            "price_levels": price_data.tolist(),
            "volumes": volume_data.tolist(),
            "cvd_series": cvd_data.tolist(),
            "ofi_series": ofi_data.tolist(),
            "current_regime": self.classify_market_regime(),
            "recent_signals": [
                {
                    "timestamp": signal.timestamp,
                    "type": signal.signal_type,
                    "confidence": signal.confidence,
                    "price": signal.price_level,
                    "description": signal.description
                }
                for signal in list(self.signals_history)[-10:]
            ],
            "metadata": {
                "symbol": self.symbol,
                "snapshot_interval_ms": self.snapshot_interval_ms,
                "depth_levels": self.depth_levels,
                "total_snapshots": len(self.snapshots)
            }
        }
    
    def export_to_csv(self, filepath: str = None):
        """Export analysis data to CSV files."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"orderflow_analysis_{self.symbol.replace('/', '_')}_{timestamp}"
        
        os.makedirs("data/csv_exports", exist_ok=True)
        base_path = f"data/csv_exports/{filepath}"
        
        # Export snapshots
        if self.snapshots:
            snapshots_data = []
            for snapshot in self.snapshots:
                snapshots_data.append({
                    'timestamp': snapshot.timestamp,
                    'datetime': pd.to_datetime(snapshot.timestamp, unit='ms'),
                    'symbol': snapshot.symbol,
                    'mid_price': snapshot.mid_price,
                    'spread': snapshot.spread,
                    'total_bid_volume': snapshot.total_bid_volume,
                    'total_ask_volume': snapshot.total_ask_volume,
                    'vwap': snapshot.vwap,
                    'cvd': snapshot.cvd,
                    'ofi': snapshot.ofi,
                    'liquidity_depth': snapshot.liquidity_depth
                })
            
            df_snapshots = pd.DataFrame(snapshots_data)
            df_snapshots.to_csv(f"{base_path}_snapshots.csv", index=False)
        
        # Export metrics
        if self.metrics_history:
            metrics_data = []
            for metric in self.metrics_history:
                metrics_data.append({
                    'timestamp': metric.timestamp,
                    'datetime': pd.to_datetime(metric.timestamp, unit='ms'),
                    'symbol': metric.symbol,
                    'vwap': metric.vwap,
                    'cvd': metric.cvd,
                    'ofi': metric.ofi,
                    'bid_ask_imbalance': metric.bid_ask_imbalance,
                    'liquidity_depth': metric.liquidity_depth,
                    'depth_volatility': metric.depth_volatility,
                    'liquidity_concentration': metric.liquidity_concentration,
                    'market_maker_ratio': metric.market_maker_ratio,
                    'spread_volatility': metric.spread_volatility,
                    'volume_weighted_spread': metric.volume_weighted_spread
                })
            
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_csv(f"{base_path}_metrics.csv", index=False)
        
        # Export signals
        if self.signals_history:
            signals_data = []
            for signal in self.signals_history:
                signals_data.append({
                    'timestamp': signal.timestamp,
                    'datetime': pd.to_datetime(signal.timestamp, unit='ms'),
                    'signal_type': signal.signal_type,
                    'confidence': signal.confidence,
                    'price_level': signal.price_level,
                    'volume': signal.volume,
                    'description': signal.description,
                    'metadata': json.dumps(signal.metadata)
                })
            
            df_signals = pd.DataFrame(signals_data)
            df_signals.to_csv(f"{base_path}_signals.csv", index=False)
        
        logger.info(f"Analysis data exported to {base_path}_*.csv")
        return base_path
    
    def stop_streaming(self):
        """Stop the streaming and cleanup resources."""
        self.is_streaming = False
        self.stream_event.clear()
        
        if self.exchange_pro:
            try:
                # Properly close the exchange connection
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._close_exchange())
                else:
                    loop.run_until_complete(self.exchange_pro.close())
            except Exception as e:
                logger.warning(f"Error closing exchange connection: {e}")
        
        logger.info("Order book streaming stopped")
    
    async def _close_exchange(self):
        """Safely close the exchange connection."""
        if self.exchange_pro:
            try:
                await self.exchange_pro.close()
                self.exchange_pro = None
            except Exception as e:
                logger.warning(f"Error during exchange cleanup: {e}")

# Utility functions for external access
def create_heatmap_analyzer(symbol: str = 'XRP/USDT', **kwargs) -> OrderBookHeatmap:
    """Create and return a new OrderBookHeatmap analyzer."""
    return OrderBookHeatmap(symbol=symbol, **kwargs)

async def run_analysis(symbol: str = 'XRP/USDT', duration_minutes: int = 60, **kwargs):
    """
    Run order book analysis for specified duration.
    
    Args:
        symbol: Trading pair to analyze
        duration_minutes: How long to run analysis
        **kwargs: Additional parameters for OrderBookHeatmap
    """
    analyzer = create_heatmap_analyzer(symbol, **kwargs)
    
    try:
        # Start streaming in background
        stream_task = asyncio.create_task(analyzer.start_streaming())
        
        # Wait for specified duration
        await asyncio.sleep(duration_minutes * 15)
        
        # Stop streaming
        analyzer.stop_streaming()
        
        # Cancel and wait for stream task to complete
        if not stream_task.done():
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass
        
        # Export results
        export_path = analyzer.export_to_csv()
        
        # Get final heatmap data
        heatmap_data = analyzer.get_heatmap_data()
        
        logger.info(f"Analysis completed for {symbol}. Data exported to {export_path}")
        return {
            "export_path": export_path,
            "heatmap_data": heatmap_data,
            "final_regime": analyzer.classify_market_regime(),
            "total_signals": len(analyzer.signals_history),
            "total_snapshots": len(analyzer.snapshots)
        }
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        analyzer.stop_streaming()
        
        # Ensure all tasks are cancelled
        if 'stream_task' in locals() and not stream_task.done():
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass
        
        raise

