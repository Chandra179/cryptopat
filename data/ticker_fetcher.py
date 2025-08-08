"""
Ticker data fetcher module for current price and volume information.
"""

import csv
import os
import logging
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)


class TickerFetcher:
    """Handles ticker data fetching."""
    
    def __init__(self, exchange, output_dir: str = "data/csv_exports"):
        self.exchange = exchange
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch current ticker data for a symbol.
        structure: https://docs.ccxt.com/#/?id=ticker-structure
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Ticker data with current prices and volume
        """
        try:
            logger.info(f"Fetching ticker for {symbol}")
            ticker = self.exchange.fetch_ticker(symbol)
            
            # Export to CSV
            try:
                csv_path = self.export_ticker_data(ticker, symbol)
                logger.info(f"Ticker data exported to: {csv_path}")
            except Exception as e:
                logger.warning(f"Failed to export ticker to CSV: {e}")
            
            return ticker
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise RuntimeError(f"Failed to fetch ticker for {symbol}: {e}")
    
    def export_ticker_data(self, data: Dict, symbol: str) -> str:
        """
        Export ticker data to CSV following CCXT structure.
        
        Args:
            data: Ticker data dictionary
            symbol: Trading pair symbol
            
        Returns:
            Path to saved CSV file
        """
        if not data:
            raise ValueError("No ticker data to export")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ticker_{symbol.replace('/', '_')}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header row with CCXT ticker fields
            writer.writerow([
                'symbol', 'timestamp', 'datetime', 'high', 'low', 'bid', 'bidVolume',
                'ask', 'askVolume', 'vwap', 'open', 'close', 'last', 'previousClose',
                'change', 'percentage', 'average', 'baseVolume', 'quoteVolume'
            ])
            
            # Data row
            writer.writerow([
                data.get('symbol', symbol),
                data.get('timestamp', ''),
                data.get('datetime', ''),
                data.get('high', ''),
                data.get('low', ''),
                data.get('bid', ''),
                data.get('bidVolume', ''),
                data.get('ask', ''),
                data.get('askVolume', ''),
                data.get('vwap', ''),
                data.get('open', ''),
                data.get('close', ''),
                data.get('last', ''),
                data.get('previousClose', ''),
                data.get('change', ''),
                data.get('percentage', ''),
                data.get('average', ''),
                data.get('baseVolume', ''),
                data.get('quoteVolume', '')
            ])
        
        return filepath