#!/usr/bin/env python3

import os
import sys
import signal
import readline
from typing import Dict
from data import get_data_collector
from orderflow.absorption import AbsorptionStrategy
from orderflow.cvd import CVDStrategy
from orderflow.footprint import VolumeFootprint
from orderflow.smartmoney import SmartMoneyConcepts
from orderflow.stopsweep import StopSweep
from techin.bollingerbands import BollingerBands
from techin.chaikin import ChaikinMoneyFlow
from techin.donchain import DonchianChannel
from techin.ichimoku import IchimokuCloud
from techin.keltner import KeltnerChannel
from techin.macd import MACD
from techin.obv import OBV
from techin.parabolicsar import ParabolicSAR
from techin.pivotpoint import PivotPoint
from techin.renko import Renko
from techin.supertrend import Supertrend
from techin.vwap import VWAP

class CryptoPatCLI:
    def __init__(self):
        self.data_collector = get_data_collector()
        self.history_file = os.path.expanduser("~/.cryptopat_history")
        self.setup_readline()
        self.setup_signal_handlers()
        
    def setup_readline(self):
        """Setup readline for command history and completion"""
        try:
            readline.read_history_file(self.history_file)
        except FileNotFoundError:
            pass
        
        readline.set_history_length(1000)
        
        # Enable tab completion
        readline.parse_and_bind('tab: complete')
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful exit"""
        def signal_handler(_sig, _frame):
            print("\nExiting CryptoPat CLI...")
            self.save_history()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        
    def save_history(self):
        """Save command history to file"""
        try:
            readline.write_history_file(self.history_file)
        except Exception as e:
            print(f"Warning: Could not save history: {e}")
            
    def parse_command(self, command: str) -> Dict[str, str]:
        """Parse command arguments like s=BTC/USDT t=1d l=100"""
        args = {}
        parts = command.strip().split()
        
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                args[key.lower()] = value
        
        return args
        
    def validate_timeframe(self, timeframe: str) -> bool:
        """Validate if timeframe is supported"""
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '3d', '1w', '1M']
        return timeframe in valid_timeframes
        
    def execute_fetch_command(self, args: Dict[str, str]) -> None:
        """Execute data fetch command with parsed arguments"""
        symbol = args.get('s', 'BTC/USDT')
        timeframe = args.get('t', '1d')
        limit = int(args.get('l', '100'))
        
        if not self.validate_timeframe(timeframe):
            print(f"Error: Invalid timeframe '{timeframe}'. Valid options: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 3d, 1w, 1M")
            return
            
        try:
            print(f"Fetching {symbol} data for timeframe {timeframe} with limit {limit}...")
            
            # Fetch OHLCV data
            ohlcv_data = self.data_collector.fetch_ohlcv_data(symbol, timeframe, limit)
            print(f"✓ Fetched {len(ohlcv_data)} OHLCV records")
            
            # Fetch ticker data
            ticker = self.data_collector.fetch_ticker(symbol)
            if ticker:
                print(f"✓ Current price: {ticker['last']}")
                print(f"✓ 24h change: {ticker['percentage']:.2f}%")
                print(f"✓ Volume: {ticker['baseVolume']:.2f}")
            
            # Fetch order book
            order_book = self.data_collector.fetch_order_book(symbol, limit)
            if order_book:
                print(f"✓ Order book: {len(order_book['bids'])} bids, {len(order_book['asks'])} asks")

            # Fetch trades
            trades = self.data_collector.fetch_trades(symbol, limit)
            if trades:
                print(f"✓ Retrieved {len(trades)} recent trades")
                print(f"✓ Latest trade: {trades[-1].get('price')} @ {trades[-1].get('amount')}")

            bollingerbands = BollingerBands(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)
            bollingerbands.calculate()
            chaikinmoneyflow = ChaikinMoneyFlow(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)
            chaikinmoneyflow.calculate()
            donchain = DonchianChannel(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)
            donchain.calculate()
            ichimoku = IchimokuCloud(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)
            ichimoku.calculate()
            keltner = KeltnerChannel(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)
            keltner.calculate()
            macd = MACD(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)
            macd.calculate()
            absorption = AbsorptionStrategy(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)
            absorption.calculate()
            cvd = CVDStrategy(symbol, timeframe, limit, order_book, ohlcv_data, ticker, trades)
            cvd.calculate()
            footprint = VolumeFootprint(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)
            footprint.calculate()
            smartmoney = SmartMoneyConcepts(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)
            smartmoney.calculate()
            stopsweep = StopSweep(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)
            stopsweep.calculate()
            obv = OBV(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)
            obv.calculate()
            parabolicsar = ParabolicSAR(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)
            parabolicsar.calculate()
            pivotpoint = PivotPoint(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)
            pivotpoint.calculate()
            renko = Renko(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)
            renko.calculate()
            supertrend = Supertrend(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)
            supertrend.calculate()
            vwap = VWAP(symbol, timeframe, limit, order_book, ticker, ohlcv_data, trades)
            vwap.calculate()

                        
        except Exception as e:
            print(f"Error fetching data: {e}")
            
    def show_help(self):
        """Display help information"""
        print("\nCryptoPat CLI Commands:")
        print("  s=SYMBOL t=TIMEFRAME l=LIMIT  - Fetch market data")
        print("  Example: s=BTC/USDT t=1d l=100")
        print("  ")
        print("  clear                         - Clear the screen")
        print("  help                         - Show this help")
        print("  exit                         - Exit the CLI")
        print("")
        print("Supported symbols: BTC/USDT, ETH/USDT, XRP/USDT, SOL/USDT, PENGU/USDT")
        print("Supported timeframes: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 3d, 1w, 1M")
        print("Use Ctrl+C to exit")
        print()
        
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def run(self):
        """Main CLI loop"""
        print("Welcome to CryptoPat CLI")
        print("Type 'help' for available commands or Ctrl+C to exit")
        print()
        
        while True:
            try:
                command = input("cryptopat> ").strip()
                
                if not command:
                    continue
                    
                if command.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                elif command.lower() == 'clear':
                    self.clear_screen()
                elif command.lower() == 'help':
                    self.show_help()
                else:
                    # Parse and execute fetch command
                    args = self.parse_command(command)
                    if args:  # If we have parsed arguments, treat as fetch command
                        self.execute_fetch_command(args)
                    else:
                        print(f"Unknown command: {command}")
                        print("Type 'help' for available commands")
                        
            except EOFError:
                print("\nGoodbye!")
                break
            except KeyboardInterrupt:
                print("\nUse 'exit' or Ctrl+C to quit")
                continue
                
        self.save_history()

if __name__ == "__main__":
    cli = CryptoPatCLI()
    cli.run()