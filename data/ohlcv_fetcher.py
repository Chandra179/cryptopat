"""
OHLCV data fetcher module for candlestick data collection.
"""

import csv
import os
import logging
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


class OHLCVFetcher:
    """Handles OHLCV candlestick data fetching."""
    
    def __init__(self, exchange, output_dir: str = "data/csv_exports"):
        self.exchange = exchange
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
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
        try:
            if since is None:
                # Default to get most recent data (last 'limit' candles)
                since = None
            
            logger.info(f"Fetching {timeframe} OHLCV data for {symbol}")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            logger.info(f"Retrieved {len(ohlcv)} candles for {symbol}")
            
            # Export to CSV
            try:
                csv_path = self.export_ohlcv_data(ohlcv, symbol, timeframe)
                logger.info(f"OHLCV data exported to: {csv_path}")
            except Exception as e:
                logger.warning(f"Failed to export OHLCV to CSV: {e}")
            
            return ohlcv
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            raise RuntimeError(f"Failed to fetch OHLCV data for {symbol}: {e}")
    
    def export_ohlcv_data(self, data: List[List], symbol: str, timeframe: str = "1d") -> str:
        """
        Export OHLCV data to CSV following CCXT structure.
        Format: [timestamp, open, high, low, close, volume]
        
        Args:
            data: OHLCV data list
            symbol: Trading pair symbol
            timeframe: Timeframe used
            
        Returns:
            Path to saved CSV file
        """
        if not data:
            raise ValueError("No OHLCV data to export")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ohlcv_{symbol.replace('/', '_')}_{timeframe}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            for candle in data:
                if len(candle) >= 6:
                    writer.writerow([
                        int(candle[0]),  # timestamp (milliseconds)
                        float(candle[1]),  # open
                        float(candle[2]),  # high
                        float(candle[3]),  # low
                        float(candle[4]),  # close
                        float(candle[5])   # volume
                    ])
        
        return filepath