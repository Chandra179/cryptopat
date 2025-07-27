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
```

## Architecture

**Project Structure:**
- `main.py` - Application entry point with CLI argument parsing
- `data/collector.py` - CCXT-based data collection from exchanges
- `analysis/` - Pattern recognition and technical analysis modules (to be implemented)
- `patterns/` - Pattern detection algorithms (to be implemented)

**Key Classes:**
- `CryptoTrendApp` (main.py:22) - Main application orchestrator
- `DataCollector` (data/collector.py:17) - Handles all market data fetching

**Data Flow:**
1. User specifies prediction timeframe and analysis window via CLI
2. DataCollector fetches OHLCV, order book, and ticker data from exchange
3. Analysis modules process data for pattern detection
4. Output provides bearish/bullish confidence percentages

**Supported Data Types:**
- OHLCV candlestick data (1d, 4h timeframes)
- Order book depth
- Current ticker information
- Default symbols: BTC/USDT, ETH/USDT, XRP/USDT, SOL/USDT, PENGU/USDT

**Rate Limiting:**
The DataCollector implements 1.2-second delays between API calls to respect exchange limits.