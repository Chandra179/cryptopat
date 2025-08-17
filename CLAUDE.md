# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CryptoPat is a comprehensive Python-based cryptocurrency pattern detection and technical analysis system. It fetches real-time market data from exchanges and performs advanced analysis using technical indicators.

## Core Architecture

### Main Components

- **CLI Interface** (`cli.py`): Interactive command-line interface with history and signal handling
- **Data Collection** (`data/`): Modular data fetchers using CCXT library with singleton pattern
- **Technical Indicators** (`techin/`): Individual indicator implementations with YAML configuration files
- **Analysis Pipeline**: Concurrent execution of multiple technical indicators

### Data Flow

1. CLI receives user commands (format: `s=SYMBOL t=TIMEFRAME l=LIMIT`)
2. DataCollector singleton fetches market data concurrently (OHLCV, ticker, order book, trades)
3. All technical indicators run in parallel using ThreadPoolExecutor
4. Results are stored in memory for analysis summary generation

### Data Package Structure

The `data/` package uses composition pattern:
- `DataCollector` (main class) composes specialized fetchers
- `OHLCVFetcher`, `OrderBookFetcher`, `TickerFetcher`, `TradesFetcher`
- Singleton pattern via `get_data_collector()` function in `__init__.py`

### Technical Indicators Architecture

Each indicator in `techin/` follows a consistent pattern:
- Python implementation file (e.g., `bollinger_bands.py`)
- YAML configuration file with timeframe-specific parameters
- Standard constructor: `(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)`
- `calculate()` method for execution

## Development Commands

### Running the Application
```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Start the interactive CLI
python cli.py
```

### Dependencies
```bash
# Install/update dependencies
pip install -r requirements.txt
```

### CLI Usage Examples
```bash
# In the cryptopat CLI:
s=BTC/USDT t=1d l=100    # Bitcoin daily, 100 candles
s=ETH/USDT t=1w l=50     # Ethereum weekly, 50 candles
help                     # Show available commands
clear                    # Clear screen
exit                     # Exit application
```

## Key Technologies

- **CCXT**: Exchange connectivity (Binance primary, 100+ exchanges supported)
- **NumPy/Pandas**: Data processing and calculations
- **PyYAML**: Configuration management
- **asyncio**: Concurrent data fetching
- **ThreadPoolExecutor**: Parallel indicator execution

## Supported Timeframes

- `1d` (daily)
- `1w` (weekly) 
- `1M` (monthly)

## Data Formats

All data follows CCXT standard formats:
- **OHLCV**: `[timestamp, open, high, low, close, volume]`
- **Order Book**: `{bids: [[price, amount]], asks: [[price, amount]], symbol, timestamp}`
- **Ticker**: Price and volume data with standardized fields
- **Trades**: Recent trade history for volume analysis

## Technical Indicators

Current implementation includes:
- Trend: MACD, EMA (20/50), Parabolic SAR, Supertrend
- Volatility: Bollinger Bands, Keltner Channel, Donchian Channel
- Volume: OBV, VWAP, Chaikin Money Flow
- Momentum: RSI
- Support/Resistance: Pivot Points, Ichimoku Cloud
- Alternative: Renko Charts

Each indicator uses YAML configuration for timeframe-specific parameters and has standardized output format with signal detection.

## File Patterns

- Technical indicators: `techin/{indicator_name}.py` + `techin/{indicator_name}.yaml`
- Data fetchers: `data/{type}_fetcher.py`
- CSV exports: `data/csv_exports/{type}_{symbol}_{timestamp}.csv`
- Configuration templates: `data/format/{type}_format.txt`