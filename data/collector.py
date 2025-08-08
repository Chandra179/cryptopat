"""
Data collector module for fetching cryptocurrency market data from exchanges.
Handles OHLCV data, order books, and ticker information using CCXT library.
"""

from typing import List, Dict, Optional
from .base_collector import DataCollectorBase
from .ohlcv_fetcher import OHLCVFetcher
from .order_book_fetcher import OrderBookFetcher
from .ticker_fetcher import TickerFetcher
from .trades_fetcher import TradesFetcher


class DataCollector(DataCollectorBase):
    """Main DataCollector class that composes all fetcher functionality."""
    
    def __init__(self, exchange_name: str = 'binance'):
        """
        Initialize the data collector with specified exchange.
        Uses singleton pattern to ensure single instance.
        
        Args:
            exchange_name: Name of the exchange to use (default: binance)
        """
        super().__init__(exchange_name)
        self._ohlcv_fetcher = OHLCVFetcher(self.exchange)
        self._order_book_fetcher = OrderBookFetcher(self.exchange)
        self._ticker_fetcher = TickerFetcher(self.exchange)
        self._trades_fetcher = TradesFetcher(self.exchange)
    
    def fetch_ohlcv_data(self, symbol: str, timeframe: str = '1d', 
                        limit: int = 999, since: Optional[int] = None) -> List[List]:
        """
        Fetch OHLCV (candlestick) data for a symbol.
        structure: https://docs.ccxt.com/#/?id=ohlcv-structure
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for data ('1d', '4h', etc.)
            limit: Number of candles to fetch
            since: Start timestamp in milliseconds
            
        Returns:
            List of OHLCV data: [[timestamp, open, high, low, close, volume], ...]
        """
        return self._ohlcv_fetcher.fetch_ohlcv_data(symbol, timeframe, limit, since)
    
    def fetch_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        Fetch order book data for a symbol.
        structure: https://docs.ccxt.com/#/?id=order-book-structure
        
        Args:
            symbol: Trading pair symbol
            limit: Number of price levels to fetch
            
        Returns:
            Order book data with bids and asks
        """
        return self._order_book_fetcher.fetch_order_book(symbol, limit)
    
    def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch current ticker data for a symbol.
        structure: https://docs.ccxt.com/#/?id=ticker-structure
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Ticker data with current prices and volume
        """
        return self._ticker_fetcher.fetch_ticker(symbol)
    
    def fetch_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Fetch recent trades for CVD calculation.
        structure: https://docs.ccxt.com/#/?id=trade-structure
        
        Args:
            symbol: Trading pair symbol
            limit: Number of recent trades to fetch (default: 500)
            
        Returns:
            List of trade data
        """
        return self._trades_fetcher.fetch_trades(symbol, limit)


