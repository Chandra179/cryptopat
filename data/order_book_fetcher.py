"""
Order book data fetcher module for market depth collection.
"""

import csv
import os
import logging
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)


class OrderBookFetcher:
    """Handles order book data fetching."""
    
    def __init__(self, exchange, output_dir: str = "data/csv_exports"):
        self.exchange = exchange
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
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
        try:
            logger.info(f"Fetching order book for {symbol}")
            order_book = self.exchange.fetch_order_book(symbol, limit)
            order_book['symbol'] = symbol
            
            # Export to CSV
            try:
                csv_path = self.export_order_book_data(order_book, symbol)
                logger.info(f"Order book data exported to: {csv_path}")
            except Exception as e:
                logger.warning(f"Failed to export order book to CSV: {e}")
            
            return order_book
            
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            raise RuntimeError(f"Failed to fetch order book for {symbol}: {e}")
    
    def export_order_book_data(self, data: Dict, symbol: str) -> str:
        """
        Export order book data to CSV following CCXT structure.
        
        Args:
            data: Order book data dictionary
            symbol: Trading pair symbol
            
        Returns:
            Path to saved CSV file
        """
        if not data or not data.get('bids') or not data.get('asks'):
            raise ValueError("No order book data to export")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"orderbook_{symbol.replace('/', '_')}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write metadata
            writer.writerow(['symbol', data.get('symbol', symbol)])
            writer.writerow(['timestamp', data.get('timestamp', '')])
            writer.writerow(['datetime', data.get('datetime', '')])
            writer.writerow(['nonce', data.get('nonce', '')])
            writer.writerow([])  # Empty row separator
            
            # Write bids header
            writer.writerow(['BIDS'])
            writer.writerow(['price', 'amount'])
            for bid in data.get('bids', []):
                if len(bid) >= 2:
                    writer.writerow([float(bid[0]), float(bid[1])])
            
            writer.writerow([])  # Empty row separator
            
            # Write asks header
            writer.writerow(['ASKS'])
            writer.writerow(['price', 'amount'])
            for ask in data.get('asks', []):
                if len(ask) >= 2:
                    writer.writerow([float(ask[0]), float(ask[1])])
        
        return filepath