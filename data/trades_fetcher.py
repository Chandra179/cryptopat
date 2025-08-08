"""
Trades data fetcher module for recent trade history collection.
"""

import csv
import os
import logging
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)


class TradesFetcher:
    """Handles trades data fetching."""
    
    def __init__(self, exchange, output_dir: str = "data/csv_exports"):
        self.exchange = exchange
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
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
        try:
            logger.info(f"Fetching {limit} recent trades for {symbol}")
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            logger.info(f"Retrieved {len(trades)} trades for {symbol}")
            
            # Export to CSV
            try:
                csv_path = self.export_trades_data(trades, symbol)
                logger.info(f"Trades data exported to: {csv_path}")
            except Exception as e:
                logger.warning(f"Failed to export trades to CSV: {e}")
            
            return trades
            
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            raise RuntimeError(f"Failed to fetch trades for {symbol}: {e}")
    
    def export_trades_data(self, data: List[Dict], symbol: str) -> str:
        """
        Export trades data to CSV following CCXT structure.
        
        Args:
            data: List of trade data dictionaries
            symbol: Trading pair symbol
            
        Returns:
            Path to saved CSV file
        """
        if not data:
            raise ValueError("No trades data to export")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trades_{symbol.replace('/', '_')}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header row with CCXT trade fields
            writer.writerow([
                'id', 'timestamp', 'datetime', 'symbol', 'order', 'type', 'side',
                'takerOrMaker', 'price', 'amount', 'cost', 'fee_cost', 'fee_currency', 'fee_rate'
            ])
            
            # Data rows
            for trade in data:
                fee = trade.get('fee', {}) or {}
                writer.writerow([
                    trade.get('id', ''),
                    trade.get('timestamp', ''),
                    trade.get('datetime', ''),
                    trade.get('symbol', symbol),
                    trade.get('order', ''),
                    trade.get('type', ''),
                    trade.get('side', ''),
                    trade.get('takerOrMaker', ''),
                    trade.get('price', ''),
                    trade.get('amount', ''),
                    trade.get('cost', ''),
                    fee.get('cost', ''),
                    fee.get('currency', ''),
                    fee.get('rate', '')
                ])
        
        return filepath