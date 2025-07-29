# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CryptoPat is a Python-based cryptocurrency pattern recognition system for detecting chart patterns using historical market data. It's a backend-only data analysis tool focused on technical analysis and trend prediction.

**Core Components:**
- Data collection via CCXT library (Binance exchange by default)
- Pattern detection and technical analysis algorithms
- Trend prediction with bearish/bullish confidence scoring
- Support for multiple cryptocurrencies and timeframes

## Development Commands

**Environment Setup:**
```bash
# Activate virtual environment
make act
# OR manually: source venv/bin/activate

# Run setup script (installs dependencies, creates directories)
./setup.sh
```

**Running the Application:**
```bash
# Main application entry point
python main.py

# Direct data collection testing
python -m data.collector

# Test data fetching with CSV export
python test_data_fetch.py
```

**Development Testing:**
```bash
# Example analysis commands from Makefile
make pengu1  # Run PENGU/USDT analysis with SMA, EMA indicators
```

## Architecture

**Project Structure:**
- `main.py` - Application entry point with basic logging setup
- `data/collector.py` - CCXT-based data collection from exchanges
- `data/csv_exports/` - Directory for exported OHLCV data in CSV format
- `test_data_fetch.py` - Testing script for data collection and CSV export
- `trend/` - Directory for technical analysis modules (currently empty)
- `Makefile` - Contains shortcuts for common development tasks

**Key Classes:**
- `DataCollector` (data/collector.py:37) - Singleton class that handles all market data fetching with rate limiting
  - Uses singleton pattern to ensure only one exchange connection per application run
  - `fetch_ohlcv_data()` - Gets candlestick data for specified symbols/timeframes
  - `fetch_order_book()` - Gets bid/ask price levels
  - `fetch_ticker()` - Gets current market prices and volume
  - `get_market_info()` - Gets trading limits and fees
- `get_data_collector()` (data/__init__.py:11) - Factory function that returns the singleton DataCollector instance

**Data Flow:**
1. DataCollector singleton initializes exchange connection with rate limiting (1.2s between requests)
2. All trend analysis modules share the same DataCollector instance to avoid multiple exchange connections
3. Fetches OHLCV, order book, and ticker data from Binance exchange
4. Data exported to CSV files in `data/csv_exports/` for analysis
5. Analysis modules process data for pattern detection using technical indicators

**Supported Data Types:**
- OHLCV candlestick data with configurable timeframes (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 3d, 1w, 1M)
- Order book depth data
- Current ticker information
- Default symbols: BTC/USDT, ETH/USDT, XRP/USDT, SOL/USDT, PENGU/USDT

**Rate Limiting:**
The DataCollector implements 1.2-second delays between API calls to respect exchange limits.