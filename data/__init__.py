"""
Data package for cryptocurrency market data collection and management.
Provides singleton DataCollector instance for efficient resource usage.
"""

from .collector import DataCollector

# Global singleton instance - initialized on first import
_collector_instance = None

def get_data_collector(exchange_name: str = 'binance') -> DataCollector:
    """
    Get the singleton DataCollector instance.
    
    Args:
        exchange_name: Exchange to use (default: binance)
        
    Returns:
        DataCollector singleton instance
    """
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = DataCollector(exchange_name)
    return _collector_instance

__all__ = ['DataCollector', 'get_data_collector']