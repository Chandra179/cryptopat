# CryptoPat

A Python-based cryptocurrency pattern detection and technical analysis system that fetches market data through CCXT and analyzes it using various technical indicators and order flow strategies.

## Features

### Technical Indicators
- ATR & ADX
- Bollinger Bands
- EMA (9/21 and other periods)
- MACD
- OBV (On-Balance Volume)
- RSI (14-period)
- Supertrend
- VWAP (Volume Weighted Average Price)

### Pattern Detection
- Butterfly Pattern
- Double Bottom/Top
- Elliott Wave
- Flag Patterns
- Head and Shoulders (including Inverse)
- Shark Pattern
- Triangle Patterns
- Wedge Patterns

### Order Flow Analysis
- Absorption Analysis
- CVD (Cumulative Volume Delta)
- Smart Money Concepts
- Footprint
- StopSweep

## Quick Start

### Environment Setup
```bash
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
python cli.py
```

### CLI Commands
Use the format: `s=SYMBOL t=TIMEFRAME l=LIMIT`

Examples:
```
s=BTC/USDT t=1d l=100
s=ETH/USDT t=4h l=50
s=SOL/USDT t=1h l=200
```

#### Supported Timeframes
| Key     | Meaning    |
| ------- | ---------- |
| `'1m'`  | 1 minute   |
| `'3m'`  | 3 minutes  |
| `'5m'`  | 5 minutes  |
| `'15m'` | 15 minutes |
| `'30m'` | 30 minutes |
| `'1h'`  | 1 hour     |
| `'2h'`  | 2 hours    |
| `'4h'`  | 4 hours    |
| `'6h'`  | 6 hours    |
| `'12h'` | 12 hours   |
| `'1d'`  | 1 day      |
| `'3d'`  | 3 days     |
| `'1w'`  | 1 week     |
| `'1M'`  | 1 month    |

## Architecture

### Data Layer (`data/`)
- **DataCollector**: Singleton pattern for fetching market data from exchanges (Binance by default)
- **Fetchers**: Specialized classes for different data types (OHLCV, order book, ticker, trades)
- Uses CCXT library for exchange connectivity

### Analysis Modules
- **Technical Indicators (`techin/`)**: Traditional TA indicators
- **Order Flow Analysis (`orderflow/`)**: Advanced strategies and smart money analysis

## Development
All analysis modules are executed automatically when data is fetched. The system uses a modular architecture where each analysis component follows a consistent pattern with `calculate()` methods and configuration stored in `self.rules` dictionaries.