"""
Data collector module for fetching cryptocurrency market data from exchanges.
Handles OHLCV data, order books, and ticker information using CCXT library.
"""

import ccxt
import logging
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
import os

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
        Fetch OHLCV (candlestick) data for a symbol and export to fetch_data.csv.
        
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
            
            # Export to CSV
            if ohlcv:
                self._export_ohlcv_to_csv(ohlcv)
            
            return ohlcv
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return []
    
    def _export_ohlcv_to_csv(self, ohlcv_data: List[List]) -> None:
        """
        Export OHLCV data to fetch_data.csv with datetime formatting.
        
        Args:
            ohlcv_data: List of OHLCV data [[timestamp, open, high, low, close, volume], ...]
        """
        try:
            # Create DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime in Asia/Jakarta timezone
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Jakarta')
            
            # Reorder columns to match your format
            df = df[['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Ensure csv_exports directory exists
            os.makedirs('data/csv_exports', exist_ok=True)
            
            # Export to CSV
            csv_path = 'fetch_data.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"OHLCV data exported to {csv_path}")
            
        except Exception as e:
            logger.error(f"Error exporting OHLCV data to CSV: {e}")
    
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
            return {}
    
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
            return {}
    
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
            return {}