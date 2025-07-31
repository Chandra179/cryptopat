"""
Absorption Detection Module for CryptoPat.

This module implements real-time absorption detection using Level 2 market depth
and tick trade data to identify when large volumes are absorbed with minimal
price movement, indicating potential support/resistance levels.
"""

import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from data import get_data_collector

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AbsorptionEvent:
    """Data class representing an absorption event."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    price_level: float
    aggressive_volume: float
    price_delta: float
    confidence: str  # 'LOW', 'MEDIUM', 'HIGH'
    signal_type: str  # 'bullish_absorption', 'bearish_absorption', 'hidden_absorption'

@dataclass
class WindowData:
    """Data class for sliding window analysis."""
    timestamp: datetime
    trades: List[Dict]
    orderbook_snapshots: List[Dict]
    start_price: float
    end_price: float
    buy_volume: float
    sell_volume: float

class AbsorptionDetector:
    """
    Real-time absorption detection system using order flow analysis.
    
    Identifies absorption events when large aggressive volume hits one side
    but price movement remains minimal, indicating strong support/resistance.
    """
    
    def __init__(self, symbol: str, window_seconds: int = 10, 
                 volume_threshold: float = 5000, price_threshold: float = 2):
        """
        Initialize the absorption detector.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            window_seconds: Window length for aggregation (default: 10)
            volume_threshold: Minimum volume to consider absorption (default: 5000)
            price_threshold: Maximum allowed price movement in ticks (default: 2)
        """
        self.symbol = symbol
        self.window_seconds = window_seconds
        self.volume_threshold = volume_threshold
        self.price_threshold = price_threshold
        
        # Data storage
        self.trade_window = deque(maxlen=1000)  # Recent trades
        self.orderbook_window = deque(maxlen=100)  # Recent orderbook snapshots
        self.absorption_events = []
        
        # Analysis state
        self.last_analysis_time = None
        self.tick_size = None
        
        # Initialize data collector
        self.collector = get_data_collector()
        
        logger.info(f"Initialized AbsorptionDetector for {symbol}")
        logger.info(f"Window: {window_seconds}s, Vol threshold: {volume_threshold}, Price threshold: {price_threshold}")
    
    def _get_tick_size(self) -> float:
        """Get the tick size for the symbol."""
        if self.tick_size is None:
            try:
                market_info = self.collector.get_market_info(self.symbol)
                self.tick_size = market_info.get('precision', {}).get('price', 0.01)
                logger.info(f"Tick size for {self.symbol}: {self.tick_size}")
            except Exception as e:
                logger.warning(f"Could not get tick size, using default 0.01: {e}")
                self.tick_size = 0.01
        return self.tick_size
    
    def _fetch_market_data(self) -> Tuple[List[Dict], Dict]:
        """
        Fetch recent trades and current orderbook.
        
        Returns:
            Tuple of (trades, orderbook)
        """
        try:
            # Fetch recent trades
            trades = self.collector.fetch_trades_stream(self.symbol, limit=100)
            
            # Fetch current orderbook
            orderbook = self.collector.fetch_orderbook_stream(self.symbol, limit=50)
            
            return trades, orderbook
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return [], {}
    
    def _update_data_windows(self, trades: List[Dict], orderbook: Dict):
        """Update sliding windows with new market data."""
        current_time = datetime.now()
        
        # Add trades to window
        for trade in trades:
            trade['fetch_timestamp'] = current_time
            self.trade_window.append(trade)
        
        # Add orderbook snapshot to window
        if orderbook:
            orderbook['fetch_timestamp'] = current_time
            self.orderbook_window.append(orderbook)
        
        # Clean old data from windows
        cutoff_time = current_time - timedelta(seconds=self.window_seconds * 2)
        
        # Clean trade window
        while (self.trade_window and 
               self.trade_window[0].get('fetch_timestamp', current_time) < cutoff_time):
            self.trade_window.popleft()
        
        # Clean orderbook window
        while (self.orderbook_window and 
               self.orderbook_window[0].get('fetch_timestamp', current_time) < cutoff_time):
            self.orderbook_window.popleft()
    
    def _analyze_window_data(self) -> Optional[WindowData]:
        """
        Analyze current window data for absorption patterns.
        
        Returns:
            WindowData object if sufficient data available, None otherwise
        """
        if not self.trade_window or len(self.trade_window) < 2:
            return None
        
        current_time = datetime.now()
        window_start = current_time - timedelta(seconds=self.window_seconds)
        
        # Filter trades within the analysis window
        window_trades = [
            trade for trade in self.trade_window
            if trade.get('fetch_timestamp', current_time) >= window_start
        ]
        
        if len(window_trades) < 2:
            return None
        
        # Filter orderbook snapshots within the window
        window_orderbooks = [
            ob for ob in self.orderbook_window
            if ob.get('fetch_timestamp', current_time) >= window_start
        ]
        
        # Calculate price movement
        start_price = float(window_trades[0].get('price', 0))
        end_price = float(window_trades[-1].get('price', 0))
        
        # Calculate aggressive volume by side
        buy_volume = 0.0
        sell_volume = 0.0
        
        for trade in window_trades:
            volume = float(trade.get('amount', 0))
            side = trade.get('side', '')
            
            if side == 'buy':
                buy_volume += volume
            elif side == 'sell':
                sell_volume += volume
        
        return WindowData(
            timestamp=current_time,
            trades=window_trades,
            orderbook_snapshots=window_orderbooks,
            start_price=start_price,
            end_price=end_price,
            buy_volume=buy_volume,
            sell_volume=sell_volume
        )
    
    def _detect_absorption(self, window_data: WindowData) -> Optional[AbsorptionEvent]:
        """
        Detect absorption events from window data.
        
        Args:
            window_data: Analyzed window data
            
        Returns:
            AbsorptionEvent if absorption detected, None otherwise
        """
        price_delta = abs(window_data.end_price - window_data.start_price)
        tick_size = self._get_tick_size()
        price_delta_ticks = price_delta / tick_size
        
        # Check if price movement is within threshold
        if price_delta_ticks > self.price_threshold:
            return None
        
        # Determine dominant side and volume
        total_volume = window_data.buy_volume + window_data.sell_volume
        dominant_side = 'buy' if window_data.buy_volume > window_data.sell_volume else 'sell'
        dominant_volume = max(window_data.buy_volume, window_data.sell_volume)
        
        # Check if dominant volume exceeds threshold
        if dominant_volume < self.volume_threshold:
            return None
        
        # Determine absorption type and confidence
        volume_ratio = dominant_volume / total_volume if total_volume > 0 else 0
        price_level = (window_data.start_price + window_data.end_price) / 2
        
        # Determine signal type
        if dominant_side == 'sell' and price_delta_ticks <= 1:
            signal_type = 'bullish_absorption'  # Sellers absorbed, buyers defending
        elif dominant_side == 'buy' and price_delta_ticks <= 1:
            signal_type = 'bearish_absorption'  # Buyers absorbed, sellers defending
        else:
            signal_type = 'hidden_absorption'   # High volume but price flat
        
        # Determine confidence level
        if volume_ratio > 0.8 and price_delta_ticks <= 0.5:
            confidence = 'HIGH'
        elif volume_ratio > 0.65 and price_delta_ticks <= 1:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        return AbsorptionEvent(
            timestamp=window_data.timestamp,
            symbol=self.symbol,
            side=dominant_side,
            price_level=price_level,
            aggressive_volume=dominant_volume,
            price_delta=price_delta,
            confidence=confidence,
            signal_type=signal_type
        )
    
    def _format_absorption_output(self, event: AbsorptionEvent, window_data: WindowData) -> str:
        """
        Format absorption event for terminal output.
        
        Args:
            event: Detected absorption event
            window_data: Window data used for detection
            
        Returns:
            Formatted string for terminal display
        """
        timestamp_str = event.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        price_str = f"{event.price_level:,.2f}"
        volume_str = f"{event.aggressive_volume:,.0f}"
        
        # Build output message
        output_lines = []
        output_lines.append(f"[{timestamp_str}] {event.symbol} | Window: {self.window_seconds}s")
        output_lines.append(f"Price stayed within {self.price_threshold} ticks")
        
        if event.signal_type == 'bullish_absorption':
            output_lines.append(f"→ Bullish Absorption @ {price_str} | Volume: {volume_str}")
            output_lines.append(f"Signal: Support holding — consider LONG entry | Confidence: {event.confidence}")
        elif event.signal_type == 'bearish_absorption':
            output_lines.append(f"→ Bearish Absorption @ {price_str} | Volume: {volume_str}")
            output_lines.append(f"Signal: Resistance holding — consider SHORT entry | Confidence: {event.confidence}")
        else:  # hidden_absorption
            output_lines.append(f"→ Hidden Absorption @ {price_str} | Volume: {volume_str}")
            output_lines.append(f"Signal: Smart money activity detected | Confidence: {event.confidence}")
        
        # Add volume breakdown
        total_volume = window_data.buy_volume + window_data.sell_volume
        buy_pct = (window_data.buy_volume / total_volume * 100) if total_volume > 0 else 0
        sell_pct = (window_data.sell_volume / total_volume * 100) if total_volume > 0 else 0
        
        output_lines.append(f"Volume breakdown: {window_data.buy_volume:,.0f} buy ({buy_pct:.1f}%) | "
                           f"{window_data.sell_volume:,.0f} sell ({sell_pct:.1f}%)")
        
        return '\n'.join(output_lines)
    
    def run_analysis(self, duration_seconds: int = 60, snapshot_interval: float = 1.0) -> List[AbsorptionEvent]:
        """
        Run real-time absorption analysis for specified duration.
        
        Args:
            duration_seconds: How long to run analysis (default: 60 seconds)
            snapshot_interval: Time between data snapshots in seconds (default: 1.0)
            
        Returns:
            List of detected absorption events
        """
        logger.info(f"Starting absorption analysis for {self.symbol}")
        logger.info(f"Duration: {duration_seconds}s, Snapshot interval: {snapshot_interval}s")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        detected_events = []
        
        print(f"\n🔍 Absorption Analysis Started for {self.symbol}")
        print(f"Parameters: Window={self.window_seconds}s, Vol≥{self.volume_threshold:,.0f}, Price≤{self.price_threshold} ticks")
        print("=" * 80)
        
        try:
            while time.time() < end_time:
                loop_start = time.time()
                
                # Fetch market data
                trades, orderbook = self._fetch_market_data()
                
                # Update data windows
                self._update_data_windows(trades, orderbook)
                
                # Analyze for absorption
                window_data = self._analyze_window_data()
                if window_data:
                    absorption_event = self._detect_absorption(window_data)
                    if absorption_event:
                        detected_events.append(absorption_event)
                        self.absorption_events.append(absorption_event)
                        
                        # Print absorption event
                        output = self._format_absorption_output(absorption_event, window_data)
                        print(f"\n{output}")
                        print("=" * 80)
                
                # Sleep until next snapshot
                elapsed = time.time() - loop_start
                sleep_time = max(0, snapshot_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print(f"\n\n⏹️  Analysis stopped by user")
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            print(f"\n❌ Analysis error: {e}")
        
        # Summary
        print(f"\n📊 Analysis Summary:")
        print(f"Duration: {time.time() - start_time:.1f} seconds")
        print(f"Absorption events detected: {len(detected_events)}")
        
        if detected_events:
            bullish = sum(1 for e in detected_events if e.signal_type == 'bullish_absorption')
            bearish = sum(1 for e in detected_events if e.signal_type == 'bearish_absorption')
            hidden = sum(1 for e in detected_events if e.signal_type == 'hidden_absorption')
            
            print(f"  - Bullish absorptions: {bullish}")
            print(f"  - Bearish absorptions: {bearish}")
            print(f"  - Hidden absorptions: {hidden}")
        
        return detected_events

def run_absorption_analysis(symbol: str, window_seconds: int = 10, 
                          volume_threshold: float = 5000, price_threshold: float = 2,
                          duration: int = 60, snapshot_interval: float = 1.0) -> List[AbsorptionEvent]:
    """
    Convenience function to run absorption analysis with specified parameters.
    
    Args:
        symbol: Trading pair symbol
        window_seconds: Window length for aggregation
        volume_threshold: Minimum volume to consider absorption
        price_threshold: Maximum allowed price movement in ticks
        duration: Analysis duration in seconds
        snapshot_interval: Time between snapshots
        
    Returns:
        List of detected absorption events
    """
    detector = AbsorptionDetector(
        symbol=symbol,
        window_seconds=window_seconds,
        volume_threshold=volume_threshold,
        price_threshold=price_threshold
    )
    
    return detector.run_analysis(duration, snapshot_interval)