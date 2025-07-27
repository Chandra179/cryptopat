"""
Data collector module for fetching cryptocurrency market data from exchanges.
Handles OHLCV data, order books, and ticker information using CCXT library.
"""

import ccxt
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """Collects cryptocurrency market data from exchanges using CCXT."""
    
    def __init__(self, exchange_name: str = 'binance'):
        """
        Initialize the data collector with specified exchange.
        
        Args:
            exchange_name: Name of the exchange to use (default: binance)
        """
        self.exchange_name = exchange_name
        self.exchange = self._initialize_exchange(exchange_name)
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT', 'PENGU/USDT']
        # Remove hardcoded timeframes - they should be passed as parameters
        
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
                        limit: int = 365, since: Optional[int] = None) -> List[List]:
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
                # Default to 1 year ago
                since = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
            
            logger.info(f"Fetching {timeframe} OHLCV data for {symbol}")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            logger.info(f"Retrieved {len(ohlcv)} candles for {symbol}")
            return ohlcv
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return []
    
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
    
    def collect_all_data(self, symbols: Optional[List[str]] = None, timeframes: Optional[List[str]] = None) -> Dict:
        """
        Collect OHLCV data for all target cryptocurrencies.
        
        Args:
            symbols: List of symbols to collect (uses default if None)
            timeframes: List of timeframes to collect (defaults to ['1d', '4h'] if None)
            
        Returns:
            Dictionary containing collected data for each symbol and timeframe
        """
        if symbols is None:
            symbols = self.symbols
        
        if timeframes is None:
            timeframes = ['1d', '4h']
            
        collected_data = {}
        
        for symbol in symbols:
            collected_data[symbol] = {}
            
            for timeframe in timeframes:
                try:
                    # Add delay between requests to respect rate limits
                    time.sleep(self.exchange.rateLimit / 1000)
                    
                    ohlcv_data = self.fetch_ohlcv_data(symbol, timeframe)
                    collected_data[symbol][timeframe] = ohlcv_data
                    
                    logger.info(f"Collected {len(ohlcv_data)} candles for {symbol} {timeframe}")
                    
                except Exception as e:
                    logger.error(f"Failed to collect data for {symbol} {timeframe}: {e}")
                    collected_data[symbol][timeframe] = []
        
        return collected_data
    
    def validate_data(self, ohlcv_data: List[List]) -> Tuple[bool, str]:
        """
        Validate OHLCV data integrity.
        
        Args:
            ohlcv_data: List of OHLCV candles
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not ohlcv_data:
            return False, "No data provided"
        
        try:
            for i, candle in enumerate(ohlcv_data):
                if len(candle) != 6:
                    return False, f"Invalid candle format at index {i}"
                
                timestamp, open_price, high, low, close, volume = candle
                
                # Check for valid price relationships
                if high < max(open_price, close) or low > min(open_price, close):
                    return False, f"Invalid price relationships at index {i}"
                
                # Check for negative values
                if any(val < 0 for val in [open_price, high, low, close, volume]):
                    return False, f"Negative values found at index {i}"
            
            return True, "Data validation passed"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
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