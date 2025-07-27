#!/usr/bin/env python3

import argparse
import asyncio
import logging
import sys
from data.collector import DataCollector
from app.crypto_trend_app import CryptoTrendApp
from config.settings import DEFAULT_SYMBOLS, VALID_METHODS

logger = logging.getLogger(__name__)

def test_data_collection(args):
    """Test data collection for a specific symbol."""
    try:
        logger.info(f"Testing data collection for {args.symbol} on {args.exchange}")
        collector = DataCollector(args.exchange)
        
        # Test OHLCV data
        print(f"\n📊 Testing OHLCV data collection for {args.symbol}...")
        ohlcv_data = collector.fetch_ohlcv_data(args.symbol, args.timeframe, limit=5)
        if ohlcv_data:
            print(f"✅ Successfully collected {len(ohlcv_data)} OHLCV records")
            print(f"   Latest close price: ${ohlcv_data[-1][4]:,.2f}")
        else:
            print("❌ Failed to collect OHLCV data")
        
        # Test ticker data
        print(f"\n🎯 Testing ticker data for {args.symbol}...")
        ticker = collector.fetch_ticker(args.symbol)
        if ticker:
            print(f"✅ Current price: ${ticker.get('last', 0):,.2f}")
            print(f"   24h volume: {ticker.get('baseVolume', 0):,.2f}")
        else:
            print("❌ Failed to collect ticker data")
        
        print(f"\n✅ Data collection test completed for {args.symbol}")
        
    except Exception as e:
        logger.error(f"Data collection test failed: {e}")
        sys.exit(1)

def handle_analyze_command(args_list):
    """Handle analyze command in interactive mode."""
    # Create a parser for analyze command
    parser = argparse.ArgumentParser(prog='analyze', add_help=False)
    parser.add_argument('-p', '--predict-days', type=int, default=2)
    parser.add_argument('-a', '--analysis-days', type=int, default=7)
    parser.add_argument('-s', '--symbols', nargs='+', 
                       default=DEFAULT_SYMBOLS)
    parser.add_argument('-e', '--exchange', type=str, default='binance')
    parser.add_argument('-t', '--timeframe', type=str, default='1d', 
                       choices=['1d', '4h', '1h'])
    parser.add_argument('-m', '--methods', type=str, default='sma,ema')
    
    try:
        args = parser.parse_args(args_list)
        
        # Validate arguments
        if args.predict_days <= 0:
            print("❌ Predict days must be positive")
            return
        
        if args.analysis_days <= 0:
            print("❌ Analysis days must be positive")
            return
        
        if args.analysis_days < args.predict_days:
            print(f"⚠️  Analysis period ({args.analysis_days} days) is shorter than prediction period ({args.predict_days} days)")
        
        # Parse methods
        methods = [m.strip().lower() for m in args.methods.split(',') if m.strip()]
        methods = [m for m in methods if m in VALID_METHODS]
        
        if not methods:
            print(f"❌ No valid analysis methods specified. Use: {', '.join(VALID_METHODS)}, or combinations")
            return
        
        # Run analysis
        app = CryptoTrendApp(exchange=args.exchange)
        asyncio.run(app.run_analysis(
            predict_days=args.predict_days,
            analysis_days=args.analysis_days,
            symbols=args.symbols,
            methods=methods,
            timeframe=args.timeframe
        ))
        
    except SystemExit:
        print("❌ Invalid arguments for analyze command")
        print("Use: analyze -p DAYS -a DAYS -s SYMBOL1 SYMBOL2 ... -m sma,ema,ema_cross")
    except Exception as e:
        print(f"❌ Error running analysis: {e}")

def handle_collect_command(args_list):
    """Handle collect command in interactive mode."""
    # Create a parser for collect command
    parser = argparse.ArgumentParser(prog='collect', add_help=False)
    parser.add_argument('--symbol', type=str, default='BTC/USDT')
    parser.add_argument('--exchange', type=str, default='binance')
    parser.add_argument('--timeframe', type=str, default='1d', 
                       choices=['1d', '4h', '1h'])
    
    try:
        args = parser.parse_args(args_list)
        test_data_collection(args)
        
    except SystemExit:
        print("❌ Invalid arguments for collect command")
        print("Use: collect --symbol SYMBOL --exchange EXCHANGE")
    except Exception as e:
        print(f"❌ Error testing data collection: {e}")