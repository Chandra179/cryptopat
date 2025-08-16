# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Start the interactive CLI
python cli.py

# Example usage in CLI:
cryptopat> s=BTC/USDT t=1d l=100    # Bitcoin daily, 100 candles
cryptopat> s=ETH/USDT t=1h l=200    # Ethereum hourly, 200 candles
cryptopat> help                     # Show available commands
```

### Environment Setup
```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Architecture Overview

### Core Components

**Data Collection Layer** (`data/`)
- `DataCollectorSingleton`: Singleton pattern for exchange connection management
- `collector.py`: Main data collection orchestrator 
- Individual fetchers: `ohlcv_fetcher.py`, `ticker_fetcher.py`, `order_book_fetcher.py`, `trades_fetcher.py`
- Uses CCXT library for exchange connectivity (default: Binance)

**Technical Analysis Engine** (`techin/`)
- Modular indicator system with 14+ technical indicators
- Each indicator is a standalone class that calculates its own signals
- Indicators include: MACD, RSI, Bollinger Bands, Ichimoku, VWAP, SuperTrend, etc.
- All indicators accept standard parameters: `(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)`

**Analysis Framework** (`summary/`)
- `analyzer.py`: Core analysis engine with weighted signal aggregation
- `formatters.py`: Result formatting and presentation layer  
- `config.py`: Configuration management for indicator parameters
- Implements academic research-based signal classification and clustering algorithms

**Configuration System** (`config/`)
- `indicators.yaml`: Comprehensive indicator configuration with timeframe-specific parameters
- `loader.py`: YAML configuration loader
- Supports different parameter sets for 1d, 1w, 1M timeframes

### Data Flow
1. CLI parses user commands (format: `s=SYMBOL t=TIMEFRAME l=LIMIT`)
2. DataCollector fetches market data from exchange
3. All technical indicators run in parallel using ThreadPoolExecutor
4. AnalysisSummarizer aggregates results using weighted consensus methodology
5. Formatted analysis summary displayed to user
