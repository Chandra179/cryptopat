"""
Configuration loader for technical indicators.

This module provides utilities to load and validate indicator configurations
from YAML files, with fallback to default values.
"""

import os
import yaml
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Singleton configuration loader for technical indicators."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        config_path = os.path.join(os.path.dirname(__file__), 'indicators.yaml')
        
        try:
            with open(config_path, 'r') as file:
                self._config = yaml.safe_load(file)
                logger.info(f"Loaded configuration from {config_path}")
        except FileNotFoundError:
            logger.warning(f"Configuration file not found at {config_path}. Using empty config.")
            self._config = {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}. Using empty config.")
            self._config = {}
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}. Using empty config.")
            self._config = {}
    
    def get_indicator_config(self, indicator_name: str, timeframe: str) -> Dict[str, Any]:
        """
        Get configuration for a specific indicator and timeframe.
        
        Args:
            indicator_name: Name of the indicator (e.g., 'bollinger_bands')
            timeframe: Timeframe string (e.g., '1d', '1h')
            
        Returns:
            Dictionary containing merged timeframe-specific and default parameters
        """
        if not self._config:
            logger.warning(f"No configuration available for {indicator_name}")
            return {}
        
        indicator_config = self._config.get(indicator_name, {})
        if not indicator_config:
            logger.warning(f"No configuration found for indicator: {indicator_name}")
            return {}
        
        # Get timeframe-specific parameters
        timeframe_params = indicator_config.get('timeframes', {}).get(timeframe, {})
        
        # Get default parameters
        default_params = indicator_config.get('params', {})
        
        # Merge defaults with timeframe-specific parameters (timeframe overrides defaults)
        merged_config = default_params.copy()
        merged_config.update(timeframe_params)
        
        return merged_config

# Global instance
config_loader = ConfigLoader()


def get_indicator_params(indicator_name: str, timeframe: str) -> Dict[str, Any]:
    """
    Convenience function to get indicator parameters.
    
    Args:
        indicator_name: Name of the indicator
        timeframe: Timeframe string
        
    Returns:
        Configuration dictionary
    """
    return config_loader.get_indicator_config(indicator_name, timeframe)