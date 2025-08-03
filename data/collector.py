"""
Data collector module for fetching cryptocurrency market data from exchanges.
Handles OHLCV data, order books, and ticker information using CCXT library.
"""

import ccxt
import logging
import csv
import os
from datetime import datetime
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollectorSingleton:
    """Singleton DataCollector to ensure only one instance per application run."""
    _instance = None
    _initialized = False
    
    def __new__(cls, exchange_name: str = 'binance'):
        if cls._instance is None:
            cls._instance = super(DataCollectorSingleton, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, exchange_name: str = 'binance'):
        if not self._initialized:
            self._init_collector(exchange_name)
            self._initialized = True
    
    def _init_collector(self, exchange_name: str):
        """Initialize the collector - only called once."""
        self.exchange_name = exchange_name
        self.exchange = self._initialize_exchange(exchange_name)
        logger.info(f"DataCollector singleton initialized with {exchange_name}")


class DataCollector(DataCollectorSingleton):
    """Main DataCollector class that implements singleton pattern."""
    
    def __init__(self, exchange_name: str = 'binance'):
        """
        Initialize the data collector with specified exchange.
        Uses singleton pattern to ensure single instance.
        
        Args:
            exchange_name: Name of the exchange to use (default: binance)
        """
        super().__init__(exchange_name)
        
    def _initialize_exchange(self, exchange_name: str):
        """Initialize exchange connection with rate limiting."""
        try:
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({
                'rateLimit': 1200,  # milliseconds between requests
                'enableRateLimit': True,
                'sandbox': False,
            })
            logger.info(f"Initialized {exchange_name} exchange connection")
            return exchange
        except Exception as e:
            logger.error(f"Failed to initialize {exchange_name}: {e}")
            raise
    
    def fetch_ohlcv_data(self, symbol: str, timeframe: str = '1d', 
                        limit: int = 999, since: Optional[int] = None) -> List[List]:
        """
        Fetch OHLCV (candlestick) data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for data ('1d', '4h', etc.)
            limit: Number of candles to fetch
            since: Start timestamp in milliseconds
            
        Returns:
            List of OHLCV data: [[timestamp, open, high, low, close, volume], ...]
        """
        try:
            if since is None:
                # Default to get most recent data (last 'limit' candles)
                since = None
            
            logger.info(f"Fetching {timeframe} OHLCV data for {symbol}")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            logger.info(f"Retrieved {len(ohlcv)} candles for {symbol}")
            
            return ohlcv
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            raise RuntimeError(f"Failed to fetch OHLCV data for {symbol}: {e}")
    
    
    def fetch_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        Fetch order book data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of price levels to fetch
            
        Returns:
            Order book data with bids and asks
        """
        try:
            logger.info(f"Fetching order book for {symbol}")
            order_book = self.exchange.fetch_order_book(symbol, limit)
            return order_book
            
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            raise RuntimeError(f"Failed to fetch order book for {symbol}: {e}")
    
    def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch current ticker data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Ticker data with current prices and volume
        """
        try:
            logger.info(f"Fetching ticker for {symbol}")
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise RuntimeError(f"Failed to fetch ticker for {symbol}: {e}")
    
    def fetch_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Fetch recent trades for CVD calculation.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of recent trades to fetch (default: 500)
            
        Returns:
            List of trade data with timestamp, amount, price, side
        """
        try:
            logger.info(f"Fetching {limit} recent trades for {symbol}")
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            logger.info(f"Retrieved {len(trades)} trades for {symbol}")
            
            return trades
            
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            raise RuntimeError(f"Failed to fetch trades for {symbol}: {e}")
    
    
    def get_market_info(self, symbol: str) -> Dict:
        """
        Get market information for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Market information including trading limits and fees
        """
        try:
            markets = self.exchange.load_markets()
            if symbol in markets:
                return markets[symbol]
            else:
                logger.warning(f"Symbol {symbol} not found in markets")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting market info for {symbol}: {e}")
            raise RuntimeError(f"Failed to get market info for {symbol}: {e}")
    
    def fetch_orderbook_stream(self, symbol: str, limit: int = 100) -> Dict:
        """
        Fetch L2 market depth data for real-time order flow analysis.
        Enhanced version of fetch_order_book with additional metadata.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of price levels to fetch
            
        Returns:
            Enhanced order book data with timestamps and metadata
        """
        try:
            logger.debug(f"Fetching L2 market depth for {symbol}")
            order_book = self.exchange.fetch_order_book(symbol, limit)
            
            # Add metadata for enhanced analysis
            if order_book:
                order_book['symbol'] = symbol
                order_book['fetch_timestamp'] = self.exchange.milliseconds()
                order_book['limit'] = limit
                
                # Calculate additional metrics
                if 'bids' in order_book and 'asks' in order_book:
                    bids = order_book['bids']
                    asks = order_book['asks']
                    
                    if bids and asks:
                        order_book['bid_ask_spread'] = float(asks[0][0]) - float(bids[0][0])
                        order_book['mid_price'] = (float(asks[0][0]) + float(bids[0][0])) / 2
                        
                        # Calculate total liquidity
                        total_bid_volume = sum(float(bid[1]) for bid in bids)
                        total_ask_volume = sum(float(ask[1]) for ask in asks)
                        order_book['total_bid_volume'] = total_bid_volume
                        order_book['total_ask_volume'] = total_ask_volume
                        order_book['liquidity_imbalance'] = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            
            return order_book
            
        except Exception as e:
            logger.error(f"Error fetching L2 market depth for {symbol}: {e}")
            raise RuntimeError(f"Failed to fetch L2 market depth for {symbol}: {e}")
    
    def fetch_trades_stream(self, symbol: str, limit: int = 500, since: Optional[int] = None) -> List[Dict]:
        """
        Fetch tick trade data for real-time order flow analysis.
        Enhanced version of fetch_recent_trades with additional metadata.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of recent trades to fetch
            since: Start timestamp in milliseconds
            
        Returns:
            Enhanced trade data with additional classification metadata
        """
        try:
            logger.debug(f"Fetching {limit} tick trades for {symbol}")
            trades = self.exchange.fetch_trades(symbol, since=since, limit=limit)
            
            # Add enhanced metadata for each trade
            enhanced_trades = []
            for i, trade in enumerate(trades):
                enhanced_trade = trade.copy()
                enhanced_trade['sequence'] = i
                enhanced_trade['symbol'] = symbol
                
                # Add price movement classification
                if i > 0:
                    prev_price = float(trades[i-1].get('price', 0))
                    curr_price = float(trade.get('price', 0))
                    
                    if curr_price > prev_price:
                        enhanced_trade['price_movement'] = 'uptick'
                    elif curr_price < prev_price:
                        enhanced_trade['price_movement'] = 'downtick'
                    else:
                        enhanced_trade['price_movement'] = 'no_change'
                else:
                    enhanced_trade['price_movement'] = 'first_trade'
                
                enhanced_trades.append(enhanced_trade)
            
            logger.debug(f"Enhanced {len(enhanced_trades)} trades for {symbol}")
            return enhanced_trades
            
        except Exception as e:
            logger.error(f"Error fetching tick trades for {symbol}: {e}")
            raise RuntimeError(f"Failed to fetch tick trades for {symbol}: {e}")
    
    def export_ohlcv_to_csv(self, symbol: str, timeframe: str = '1d', 
                           limit: int = 999, since: Optional[int] = None,
                           output_dir: str = 'data/csv_exports') -> str:
        """
        Export OHLCV data to CSV file, replacing any existing file.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for data ('1d', '4h', etc.)
            limit: Number of candles to fetch
            since: Start timestamp in milliseconds
            output_dir: Directory to save CSV files (will be created if doesn't exist)
            
        Returns:
            Path to the created CSV file
        """
        try:
            # Fetch OHLCV data
            ohlcv_data = self.fetch_ohlcv_data(symbol, timeframe, limit, since)
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp to avoid conflicts
            safe_symbol = symbol.replace('/', '_')
            filename = f"{safe_symbol}_{timeframe}_{limit}candles.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Write to CSV file (this will replace existing file)
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume'])
                
                # Write data rows
                for candle in ohlcv_data:
                    timestamp = candle[0]
                    # Convert timestamp to readable datetime
                    dt = datetime.fromtimestamp(timestamp / 1000)
                    datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    row = [
                        timestamp,      # Raw timestamp
                        datetime_str,   # Human readable datetime
                        candle[1],      # Open
                        candle[2],      # High
                        candle[3],      # Low
                        candle[4],      # Close
                        candle[5]       # Volume
                    ]
                    writer.writerow(row)
            
            logger.info(f"OHLCV data exported to: {filepath}")
            logger.info(f"Exported {len(ohlcv_data)} candles for {symbol} ({timeframe})")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting OHLCV to CSV for {symbol}: {e}")
            raise RuntimeError(f"Failed to export OHLCV data to CSV for {symbol}: {e}")