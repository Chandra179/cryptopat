# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CryptoPat is a Python-based cryptocurrency pattern detection and technical analysis system. It fetches market data through CCXT and analyzes it using various technical indicators and order flow strategies.

## Core Architecture

### Data Layer (`data/`)
- **DataCollector**: Singleton pattern for fetching market data from exchanges (default: Binance)
- **Fetchers**: Specialized classes for different data types (OHLCV, order book, ticker, trades)
- Uses CCXT library for exchange connectivity
- All fetchers inherit from `DataCollectorBase`

### Analysis Modules
- **Technical Indicators (`techin/`)**: Traditional TA indicators (MACD, Bollinger Bands, RSI, etc.)
- **Order Flow Analysis (`orderflow/`)**: Advanced strategies (Absorption, CVD, Smart Money Concepts, etc.)

### CLI Interface
- Interactive command-line interface with readline support and command history
- Command format: `s=SYMBOL t=TIMEFRAME l=LIMIT` (e.g., `s=BTC/USDT t=1d l=100`)
- All analysis modules are executed automatically when data is fetched

## Development Commands

### Environment Setup
```bash
# Activate virtual environment (already created)
make act
# or manually:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Start CLI interface
python cli.py
# or
python3 cli.py
```

### CLI Commands
- `s=SYMBOL t=TIMEFRAME l=LIMIT` - Fetch data and run all analyses
- `help` - Show available commands
- `clear` - Clear screen
- `exit` - Exit application

Supported symbols: BTC/USDT, ETH/USDT, XRP/USDT, SOL/USDT, PENGU/USDT
Supported timeframes: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 3d, 1w, 1M

## Key Data Structures

### OHLCV Data
`[timestamp_ms, open, high, low, close, volume]`

### Order Book
```python
{
    'bids': [[price, amount], ...],
    'asks': [[price, amount], ...],
    'symbol': 'BTC/USDT',
    'timestamp': 1499280391811
}
```

### Ticker Data
Current market prices, 24h changes, volume data

## Analysis Module Pattern

All analysis modules follow this pattern:
1. Initialize with `symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades`
2. Implement `calculate()` method for analysis logic
3. Store configuration in `self.rules` dictionary
4. Use consistent logging with module-specific logger

## File Organization

- `cli.py` - Main CLI application entry point
- `data/` - Data collection and exchange API integration
- `techin/` - Technical analysis indicators
- `orderflow/` - Order flow and smart money analysis
- `requirements.txt` - Python dependencies
- `Makefile` - Development shortcuts (activate venv)

## Dependencies

Key libraries:
- `ccxt` - Cryptocurrency exchange connectivity
- `pandas`, `numpy` - Data manipulation
- `scipy` - Mathematical calculations
- `mplfinance`, `matplotlib` - Charting (if needed)
- `python-dotenv` - Environment configuration