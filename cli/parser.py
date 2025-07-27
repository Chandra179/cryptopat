#!/usr/bin/env python3

import argparse
from config.settings import DEFAULT_SYMBOLS, DEFAULT_EXCHANGE, DEFAULT_TIMEFRAME, DEFAULT_METHODS, VALID_TIMEFRAMES, METHOD_REQUIREMENTS

def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='CryptoPat - Cryptocurrency Pattern Recognition and Trend Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python main.py analyze                           # Use defaults (predict 2 days, analyze 7 days)
  python main.py analyze -p 5 -a 14               # Predict 5 days using 14 days analysis
  python main.py analyze --predict-days 3 --analysis-days 30 --symbols BTC/USDT ETH/USDT --methods sma,ema
  python main.py collect --symbol BTC/USDT        # Test data collection for specific symbol
  python main.py --help                           # Show this help message

Method Requirements:
{chr(10).join([f"  {method}: min {req['min_analysis_days']} days, {req['min_timeframe']} timeframe - {req['description']}" for method, req in METHOD_REQUIREMENTS.items()])}
        """
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run trend analysis')
    analyze_parser.add_argument(
        '-p', '--predict-days',
        type=int,
        default=2,
        help='Number of days to predict ahead (default: 2)'
    )
    analyze_parser.add_argument(
        '-a', '--analysis-days',
        type=int,
        default=7,
        help='Number of historical days to analyze (default: 7)'
    )
    analyze_parser.add_argument(
        '-s', '--symbols',
        nargs='+',
        default=DEFAULT_SYMBOLS,
        help=f'Cryptocurrency symbols to analyze (default: {" ".join(DEFAULT_SYMBOLS)})'
    )
    analyze_parser.add_argument(
        '-e', '--exchange',
        type=str,
        default=DEFAULT_EXCHANGE,
        help=f'Exchange to use for data collection (default: {DEFAULT_EXCHANGE})'
    )
    analyze_parser.add_argument(
        '-t', '--timeframe',
        type=str,
        default=DEFAULT_TIMEFRAME,
        choices=VALID_TIMEFRAMES,
        help=f'Timeframe for data analysis (default: {DEFAULT_TIMEFRAME})'
    )
    analyze_parser.add_argument(
        '-m', '--methods',
        type=str,
        default=','.join(DEFAULT_METHODS),
        help=f'Analysis methods (comma-separated): sma, ema, ema_cross, macd, rsi (default: {",".join(DEFAULT_METHODS)})'
    )
    
    # Collect command (for testing data collection)
    collect_parser = subparsers.add_parser('collect', help='Test data collection')
    collect_parser.add_argument(
        '--symbol',
        type=str,
        default=DEFAULT_SYMBOLS[0],
        help=f'Symbol to collect data for (default: {DEFAULT_SYMBOLS[0]})'
    )
    collect_parser.add_argument(
        '--exchange',
        type=str,
        default=DEFAULT_EXCHANGE,
        help=f'Exchange to use (default: {DEFAULT_EXCHANGE})'
    )
    collect_parser.add_argument(
        '--timeframe',
        type=str,
        default=DEFAULT_TIMEFRAME,
        choices=VALID_TIMEFRAMES,
        help=f'Timeframe for data collection (default: {DEFAULT_TIMEFRAME})'
    )
    
    return parser