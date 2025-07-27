# Crypto Pattern Recognition - Backend Data Analysis

## Overview
A Python-based system for detecting chart patterns in cryptocurrency data using historical price, volume, and order book information. No frontend - pure data analysis and pattern detection.

## Core Objectives
- Collect and analyze crypto market data (price, volume, order book)
- Detect technical analysis using algorithmic approaches
- Output bearish/bullish confidence in percentage

## CCXT Public API Data Available
Core Market Data Types:
- OHLCV Data: Open, High, Low, Close, Volume candlestick data
- Order Book: Bid/ask prices and volumes at different levels
- Ticker Data: Current market prices, 24h volume, price changes
- Trades: Recent trade history with price, volume, timestamp
- Markets: Available trading pairs and exchange information

## Technical analysis

## Phase 1: initialization
- user can choose days to predict, i.e: i want to predict price movement bullish/bearish next 2 days
- user can choose timeframe he want to analyze, i.e: i want to analyze 7 days historic data to see if the next day or n days is bearish or bullish
- this phase only accepting params from the terminal, with default param

### Phase 2: trend analysis
- Simple moving average
- Exponential moving average