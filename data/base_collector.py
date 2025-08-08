"""
Base collector module containing the singleton pattern and exchange initialization.
"""

import ccxt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollectorSingleton:
    """Singleton DataCollector to ensure only one instance per application run."""
    _instance = None
    _initialized = False
    
    def __new__(cls):
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


class DataCollectorBase(DataCollectorSingleton):
    """Base DataCollector class that implements singleton pattern."""
    
    def __init__(self, exchange_name: str = 'binance'):
        """
        Initialize the data collector with specified exchange.
        Uses singleton pattern to ensure single instance.
        
        Args:
            exchange_name: Name of the exchange to use (default: binance)
        """
        super().__init__(exchange_name)