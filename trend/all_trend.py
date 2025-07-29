"""
All trend analysis module that orchestrates EMA 9/21, MACD, and RSI 14 strategies.
Provides comprehensive market analysis by running all trend indicators together.
"""

import sys
from datetime import datetime
from typing import List, Tuple, Dict, Any
from data import get_data_collector
from trend.ema_9_21 import EMA9_21Strategy
from trend.macd import MACDStrategy
from trend.rsi_14 import RSI14Strategy


class AllTrendStrategy:
    """All trend analysis strategy combining EMA, MACD, and RSI indicators."""
    
    def __init__(self):
        self.collector = get_data_collector()
        self.ema_strategy = EMA9_21Strategy()
        self.macd_strategy = MACDStrategy()
        self.rsi_strategy = RSI14Strategy()
    
    def get_ema_signals(self, symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        """Get EMA 9/21 signals with timestamps."""
        try:
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
            if len(ohlcv_data) < 50:
                return []
            
            timestamps = [candle[0] for candle in ohlcv_data]
            closes = [candle[4] for candle in ohlcv_data]
            volumes = [candle[5] for candle in ohlcv_data]
            
            ema9 = self.ema_strategy.calculate_ema(closes, 9)
            ema21 = self.ema_strategy.calculate_ema(closes, 21)
            
            if not ema9 or not ema21:
                return []
            
            signals = self.ema_strategy.detect_crossovers(ema9, ema21, closes, volumes)
            
            # Add timestamps to signals
            timestamped_signals = []
            for signal in signals:
                timestamp_idx = signal['index'] + 20
                if timestamp_idx < len(timestamps):
                    signal['timestamp'] = timestamps[timestamp_idx]
                    signal['strategy'] = 'EMA'
                    timestamped_signals.append(signal)
            
            return timestamped_signals
        except Exception:
            return []
    
    def get_macd_signals(self, symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        """Get MACD signals with timestamps."""
        try:
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
            if len(ohlcv_data) < 50:
                return []
            
            timestamps = [candle[0] for candle in ohlcv_data]
            closes = [candle[4] for candle in ohlcv_data]
            
            macd_line, signal_line, histogram = self.macd_strategy.calculate_macd(closes)
            
            if not macd_line or not signal_line or not histogram:
                return []
            
            signals = self.macd_strategy.detect_crossovers(macd_line, signal_line, histogram, closes)
            
            # Add timestamps to signals
            timestamped_signals = []
            for signal in signals:
                timestamp_idx = signal['index'] + 25 + 9  # MACD offset
                if timestamp_idx < len(timestamps):
                    signal['timestamp'] = timestamps[timestamp_idx]
                    signal['strategy'] = 'MACD'
                    timestamped_signals.append(signal)
            
            return timestamped_signals
        except Exception:
            return []
    
    def get_rsi_signals(self, symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        """Get RSI 14 signals with timestamps."""
        try:
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
            if len(ohlcv_data) < 20:
                return []
            
            timestamps = [candle[0] for candle in ohlcv_data]
            closes = [candle[4] for candle in ohlcv_data]
            
            rsi_values = self.rsi_strategy.calculate_rsi(closes, 14)
            
            if not rsi_values:
                return []
            
            signals = self.rsi_strategy.detect_rsi_signals(rsi_values, closes)
            
            # Add timestamps to signals
            timestamped_signals = []
            for signal in signals:
                timestamp_idx = signal['index'] + 15  # RSI offset
                if timestamp_idx < len(timestamps):
                    signal['timestamp'] = timestamps[timestamp_idx]
                    signal['strategy'] = 'RSI'
                    timestamped_signals.append(signal)
            
            return timestamped_signals
        except Exception:
            return []
    
    def format_ema_output(self, signal: Dict[str, Any], dt: datetime) -> str:
        """Format EMA signal for display."""
        signal_icon = "⬆️" if signal['signal'] == 'BUY' else "⬇️" if signal['signal'] == 'SELL' else "➖"
        confirmed_icon = "✔️" if signal['confirmed'] else "⏳"
        
        return (f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"CLOSE: {signal['close']:.4f} | "
                f"EMA9: {signal['ema9']:.4f} | "
                f"EMA21: {signal['ema21']:.4f} | "
                f"{signal_icon} {signal['signal']} | "
                f"Trend: {signal['trend']} | "
                f"{confirmed_icon} {'Confirmed' if signal['confirmed'] else 'Waiting'}")
    
    def format_macd_output(self, signal: Dict[str, Any], dt: datetime) -> str:
        """Format MACD signal for display."""
        signal_icon = "⬆️" if signal['signal'] == 'BUY' else "⬇️" if signal['signal'] == 'SELL' else "➖"
        momentum_icons = {
            'STRONG': '🔥',
            'CONFIRMING': '🔄',
            'UPTREND': '📈',
            'DOWNTREND': '📉',
            'WEAK': '🧨'
        }
        momentum_icon = momentum_icons.get(signal['momentum'], '➖')
        
        return (f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"MACD: {signal['macd']:.6f} | "
                f"SIGNAL: {signal['signal_line']:.6f} | "
                f"HIST: {signal['histogram']:.6f} | "
                f"{signal_icon} {signal['signal']} | "
                f"{momentum_icon} {signal['momentum']}")
    
    def format_rsi_output(self, signal: Dict[str, Any], dt: datetime) -> str:
        """Format RSI signal for display."""
        # Format condition indicators
        condition_icons = {
            'OVERBOUGHT': "⚠️",
            'OVERSOLD': "🔽",
            'SIDEWAYS': "↔️",
            'BULLISH_MOMENTUM': "📈",
            'BEARISH_MOMENTUM': "📉",
            'NORMAL': "➖"
        }
        condition_icon = condition_icons.get(signal['condition'], "➖")
        
        signal_icon = "⬆️" if signal['signal'] == 'BUY' else "⬇️" if signal['signal'] == 'SELL' else "➖"
        confirmed_icon = "✅" if signal['confirmed'] else "⏳"
        
        return (f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"CLOSE: {signal['close']:.4f} | "
                f"RSI(14): {signal['rsi']:.2f} | "
                f"{condition_icon} {signal['condition']} | "
                f"Signal: {signal['signal']} | "
                f"{confirmed_icon} {'Confirmed' if signal['confirmed'] else 'Waiting'}")
    
    def analyze(self, symbol: str, timeframe: str, limit: int) -> None:
        """
        Perform comprehensive trend analysis using all strategies.
        
        Args:
            symbol: Trading pair (e.g., 'XRP/USDT')
            timeframe: Timeframe (e.g., '4h', '1d', '1h')
            limit: Number of candles to analyze
        """
        print(f"\nAll Trend Analysis for {symbol} ({timeframe}) - TODAY ONLY")
        print("=" * 80)
        
        # Get signals from all strategies
        ema_signals = self.get_ema_signals(symbol, timeframe, limit)
        macd_signals = self.get_macd_signals(symbol, timeframe, limit)
        rsi_signals = self.get_rsi_signals(symbol, timeframe, limit)
        
        # Combine all signals with timestamps
        all_signals = []
        
        today = datetime.now().date()
        
        # Process EMA signals
        for signal in ema_signals:
            dt = datetime.fromtimestamp(signal['timestamp'] / 1000)
            if dt.date() == today:
                all_signals.append((dt, 'EMA', signal))
        
        # Process MACD signals
        for signal in macd_signals:
            dt = datetime.fromtimestamp(signal['timestamp'] / 1000)
            if dt.date() == today:
                all_signals.append((dt, 'MACD', signal))
        
        # Process RSI signals
        for signal in rsi_signals:
            dt = datetime.fromtimestamp(signal['timestamp'] / 1000)
            if dt.date() == today:
                all_signals.append((dt, 'RSI', signal))
        
        # Sort by timestamp
        all_signals.sort(key=lambda x: x[0])
        
        if not all_signals:
            print(f"No signals found for today ({today})")
            
            # Show latest signals from each strategy for reference
            print("\nLatest signals for reference:")
            
            if ema_signals:
                latest_ema = ema_signals[-1]
                dt = datetime.fromtimestamp(latest_ema['timestamp'] / 1000)
                print(f"EMA: {self.format_ema_output(latest_ema, dt)}")
            
            if macd_signals:
                latest_macd = macd_signals[-1]
                dt = datetime.fromtimestamp(latest_macd['timestamp'] / 1000)
                print(f"MACD: {self.format_macd_output(latest_macd, dt)}")
            
            if rsi_signals:
                latest_rsi = rsi_signals[-1]
                dt = datetime.fromtimestamp(latest_rsi['timestamp'] / 1000)
                print(f"RSI: {self.format_rsi_output(latest_rsi, dt)}")
        else:
            # Display all today's signals chronologically
            for dt, strategy, signal in all_signals:
                if strategy == 'EMA':
                    print(self.format_ema_output(signal, dt))
                elif strategy == 'MACD':
                    print(self.format_macd_output(signal, dt))
                elif strategy == 'RSI':
                    print(self.format_rsi_output(signal, dt))
        
        # Provide signal consensus summary
        if all_signals:
            self.print_consensus_summary(all_signals)
    
    def print_consensus_summary(self, signals: List[Tuple[datetime, str, Dict[str, Any]]]) -> None:
        """Print consensus summary of all signals."""
        print("\n" + "=" * 80)
        print("SIGNAL CONSENSUS SUMMARY")
        print("=" * 80)
        
        # Count signals by type
        buy_signals = 0
        sell_signals = 0
        neutral_signals = 0
        
        strategy_signals = {'EMA': [], 'MACD': [], 'RSI': []}
        
        for dt, strategy, signal in signals:
            strategy_signals[strategy].append(signal['signal'])
            
            if signal['signal'] == 'BUY':
                buy_signals += 1
            elif signal['signal'] == 'SELL':
                sell_signals += 1
            else:
                neutral_signals += 1
        
        total_signals = len(signals)
        
        print(f"Total signals today: {total_signals}")
        print(f"BUY signals: {buy_signals} ({buy_signals/total_signals*100:.1f}%)")
        print(f"SELL signals: {sell_signals} ({sell_signals/total_signals*100:.1f}%)")
        print(f"NEUTRAL signals: {neutral_signals} ({neutral_signals/total_signals*100:.1f}%)")
        
        # Latest signal from each strategy
        print(f"\nLatest signals:")
        for strategy, signal_list in strategy_signals.items():
            if signal_list:
                latest = signal_list[-1]
                icon = "🟢" if latest == 'BUY' else "🔴" if latest == 'SELL' else "🟡"
                print(f"{strategy}: {icon} {latest}")
        
        # Overall consensus
        if buy_signals > sell_signals:
            consensus = f"🟢 BULLISH BIAS ({buy_signals}/{total_signals} bullish)"
        elif sell_signals > buy_signals:
            consensus = f"🔴 BEARISH BIAS ({sell_signals}/{total_signals} bearish)"
        else:
            consensus = "🟡 MIXED SIGNALS"
        
        print(f"\nOverall consensus: {consensus}")


def parse_command(command: str) -> Tuple[str, str, int]:
    """
    Parse terminal command: all_trend s=XRP/USDT t=4h l=100
    
    Args:
        command: Command string
        
    Returns:
        Tuple of (symbol, timeframe, limit)
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'all_trend':
        raise ValueError("Invalid command format. Use: all_trend s=SYMBOL t=TIMEFRAME l=LIMIT")
    
    symbol = None
    timeframe = '4h'  # default
    limit = 100  # default
    
    for part in parts[1:]:
        if part.startswith('s='):
            symbol = part[2:]
        elif part.startswith('t='):
            timeframe = part[2:]
        elif part.startswith('l='):
            try:
                limit = int(part[2:])
            except ValueError:
                raise ValueError(f"Invalid limit value: {part[2:]}")
    
    if symbol is None:
        raise ValueError("Symbol (s=) is required")
    
    return symbol, timeframe, limit


def main():
    """Main entry point for all trend analysis."""
    if len(sys.argv) < 2:
        print("Usage: python all_trend.py s=SYMBOL t=TIMEFRAME l=LIMIT")
        print("Example: python all_trend.py s=XRP/USDT t=4h l=100")
        return
    
    try:
        # Parse command line arguments
        command = ' '.join(['all_trend'] + sys.argv[1:])
        symbol, timeframe, limit = parse_command(command)
        
        # Run analysis
        strategy = AllTrendStrategy()
        strategy.analyze(symbol, timeframe, limit)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()