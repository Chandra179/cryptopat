# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CryptoPat is a Python-based cryptocurrency analysis system for detecting chart patterns and order flow analysis using historical price, volume, and order book data. It's a pure data analysis system with no frontend - focuses on pattern detection and technical analysis.

## Key Architecture

### Core Components
- **data/**: Data collection layer using CCXT library for exchange connectivity
  - `collector.py`: Main orchestrator for all data fetching operations
  - `*_fetcher.py`: Specialized fetchers for OHLCV, order book, ticker, and trades data
  - All data gets exported to CSV files in `data/csv_exports/`

- **techin/**: Technical analysis indicators (ATR, ADX, Bollinger Bands, EMA, MACD, RSI, etc.)
- **pattern/**: Chart pattern detection (double top/bottom, head & shoulders, Elliott wave, etc.)  
- **orderflow/**: Order flow analysis strategies (absorption, CVD, footprint, SMC)

### Data Flow
1. CLI (`cli.py`) parses commands in format: `s=BTC/USDT t=1d l=100`
2. `DataCollector` fetches data from Binance via CCXT
3. Order flow strategies analyze the data and output results
4. All data automatically exported to timestamped CSV files

## Common Commands

### Environment Setup
```bash
# Activate virtual environment
make act
# or
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### CLI Usage
```bash
# Run the interactive CLI
python cli.py

# Example commands within CLI:
s=BTC/USDT t=1d l=100    # Fetch BTC daily data, 100 candles
s=PENGU/USDT t=1h l=50   # Fetch PENGU hourly data, 50 candles
```

### Supported Data
- **Symbols**: BTC/USDT, ETH/USDT, XRP/USDT, SOL/USDT, PENGU/USDT
- **Timeframes**: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 3d, 1w, 1M
- **Data Types**: OHLCV candlesticks, order books, ticker data, recent trades

## Key Dependencies
- **ccxt**: Exchange connectivity and market data
- **pandas/numpy**: Data manipulation and analysis
- **matplotlib/mplfinance**: Charting capabilities
- **python-dotenv**: Environment variable management

## Development Notes
- No formal test framework currently in place
- Uses virtual environment (`venv/`) for dependency isolation  
- Exchange data format follows CCXT standards for consistency
- All fetched data automatically exported to CSV for analysis/debugging
