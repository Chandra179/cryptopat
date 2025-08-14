# CryptoPat

A Python-based cryptocurrency pattern detection and technical analysis system that fetches real-time market data and performs comprehensive analysis using technical indicators and advanced order flow strategies.

## Features

### Technical Analysis Indicators
- **Trend Indicators**: MACD, EMA (20/50), Parabolic SAR, Supertrend
- **Volatility Indicators**: Bollinger Bands, Keltner Channel, Donchian Channel
- **Volume Indicators**: OBV (On-Balance Volume), VWAP, Chaikin Money Flow
- **Momentum Indicators**: RSI (Relative Strength Index)
- **Support/Resistance**: Pivot Points, Ichimoku Cloud
- **Alternative Charts**: Renko Charts

### Advanced Order Flow Analysis
- **Volume Footprint**: Analyze buying/selling pressure at price levels
- **Cumulative Volume Delta (CVD)**: Track net volume flow
- **Absorption Strategy**: Detect large order absorption patterns
- **Smart Money Concepts**: Identify institutional trading patterns
- **Stop Sweep Detection**: Find liquidity raids and stop hunts

### Real-Time Data Integration
- **Exchange Support**: Binance (via CCXT library)
- **Market Data**: OHLCV candlesticks, order books, ticker data, recent trades
- **Multiple Timeframes**: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 3d, 1w, 1M
- **Popular Pairs**: BTC/USDT, ETH/USDT, XRP/USDT, SOL/USDT, PENGU/USDT

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cryptopat
   ```

2. **Setup virtual environment**
   ```bash
   # or manually: source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

**Start the CLI**
```bash
python cli.py
```

**Basic Commands**
```bash
# Fetch BTC/USDT daily data with 100 candles
cryptopat> s=BTC/USDT t=1d l=100

# Fetch ETH hourly data
cryptopat> s=ETH/USDT t=1h l=200

# Get help
cryptopat> help

# Clear screen
cryptopat> clear

# Exit
cryptopat> exit
```

## 📊 Analysis Output

When you run a data fetch command, CryptoPat automatically executes all available analysis modules:

1. **Market Data Summary**
   - Current price and 24h change
   - Volume information
   - Order book depth
   - Recent trades count

2. **Technical Indicators**
   - All 14 technical indicators run automatically
   - Results logged with specific signals and values

3. **Order Flow Analysis**
   - 5 advanced strategies execute in parallel
   - Institutional activity detection
   - Volume flow analysis

## Configuration

### Supported Exchanges
- **Default**: Binance (configurable in DataCollector)
- **Extensible**: CCXT supports 100+ exchanges

### Timeframes
- **Intraday**: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h
- **Daily+**: 1d, 3d, 1w, 1M

## 📝 Command History

The CLI maintains command history with readline support:
- **History File**: `~/.cryptopat_history`
- **Navigation**: Use ↑/↓ arrows
- **Completion**: Tab completion enabled
