"""
Statistical Pattern Validation System for CryptoPat
Validates effectiveness of technical analysis patterns using historical data.
Uses existing trend analysis modules for comprehensive backtesting and statistical analysis.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

from data import get_data_collector
from trend.all_trend import AllTrendStrategy
from trend.macd import MACDStrategy
from trend.divergence import DivergenceDetector
from trend.supertrend import SupertrendStrategy
from trend.bollinger_bands import BollingerBandsStrategy
from trend.obv import OBVStrategy
from trend.atr_adx import ATR_ADXStrategy
from trend.rsi_14 import RSI14Strategy
from trend.smc import SMCStrategy


class PatternOutcome(Enum):
    SUCCESS = "Success"
    FAIL = "Fail"
    NEUTRAL = "Neutral"


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class PatternSignal:
    """Represents a pattern signal with outcome tracking."""
    timestamp: datetime
    pattern_type: str
    signal_type: SignalType
    entry_price: float
    take_profit: float
    stop_loss: float
    outcome: Optional[PatternOutcome] = None
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    bars_to_outcome: Optional[int] = None
    profit_loss_pct: Optional[float] = None


@dataclass
class ValidationMetrics:
    """Pattern validation metrics."""
    pattern_type: str
    total_signals: int
    win_rate: float
    expectancy: float
    avg_win: float
    avg_loss: float
    signal_delay: float
    frequency: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    
    # Confusion Matrix
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0


class StatPatternValidation:
    """Statistical Pattern Validation Engine."""
    
    def __init__(self, tp_percent: float = 2.0, sl_percent: float = 1.0):
        """
        Initialize validation engine.
        
        Args:
            tp_percent: Take profit percentage (default 2%)
            sl_percent: Stop loss percentage (default 1%)
        """
        self.collector = get_data_collector()
        self.all_trend = AllTrendStrategy()
        self.tp_percent = tp_percent / 100.0
        self.sl_percent = sl_percent / 100.0
        
        # Pattern signal storage
        self.signals: List[PatternSignal] = []
        self.validation_metrics: Dict[str, ValidationMetrics] = {}
        
        # Pattern strategies
        self.strategies = {
            'MACD': MACDStrategy(),
            'Divergence': DivergenceDetector(),
            'Supertrend': SupertrendStrategy(),
            'Bollinger_Bands': BollingerBandsStrategy(),
            'OBV': OBVStrategy(),
            'ATR_ADX': ATR_ADXStrategy(),
            'RSI': RSI14Strategy(),
            'SMC': SMCStrategy()
        }
    
    def extract_signals_from_strategy(self, strategy_name: str, ohlcv_data: pd.DataFrame, 
                                    symbol: str, timeframe: str) -> List[PatternSignal]:
        """
        Extract trading signals from existing strategy modules.
        
        Args:
            strategy_name: Name of the strategy
            ohlcv_data: OHLCV DataFrame
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            List of pattern signals
        """
        signals = []
        
        if strategy_name == 'MACD':
            signals.extend(self._extract_macd_signals(ohlcv_data))
        elif strategy_name == 'Supertrend':
            signals.extend(self._extract_supertrend_signals(ohlcv_data))
        elif strategy_name == 'Bollinger_Bands':
            signals.extend(self._extract_bb_signals(ohlcv_data))
        elif strategy_name == 'RSI':
            signals.extend(self._extract_rsi_signals(ohlcv_data))
        elif strategy_name == 'Divergence':
            signals.extend(self._extract_divergence_signals(ohlcv_data))
        
        return signals
    
    def _extract_macd_signals(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Extract MACD crossover signals."""
        signals = []
        closes = df['close'].tolist()
        
        try:
            macd_line, signal_line, _ = self.strategies['MACD'].calculate_macd(closes)
            
            for i in range(1, len(macd_line)):
                # Bullish crossover
                if macd_line[i-1] <= signal_line[i-1] and macd_line[i] > signal_line[i]:
                    entry_price = closes[i + 12]  # Adjust for MACD calculation offset
                    signals.append(PatternSignal(
                        timestamp=df.iloc[i + 12]['timestamp'],
                        pattern_type='MACD_Bullish_Crossover',
                        signal_type=SignalType.BUY,
                        entry_price=entry_price,
                        take_profit=entry_price * (1 + self.tp_percent),
                        stop_loss=entry_price * (1 - self.sl_percent)
                    ))
                
                # Bearish crossover
                elif macd_line[i-1] >= signal_line[i-1] and macd_line[i] < signal_line[i]:
                    entry_price = closes[i + 12]
                    signals.append(PatternSignal(
                        timestamp=df.iloc[i + 12]['timestamp'],
                        pattern_type='MACD_Bearish_Crossover',
                        signal_type=SignalType.SELL,
                        entry_price=entry_price,
                        take_profit=entry_price * (1 - self.tp_percent),
                        stop_loss=entry_price * (1 + self.sl_percent)
                    ))
        except Exception as e:
            print(f"Error extracting MACD signals: {e}")
        
        return signals
    
    def _extract_supertrend_signals(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Extract Supertrend signals."""
        signals = []
        highs = df['high'].tolist()
        lows = df['low'].tolist()
        closes = df['close'].tolist()
        
        try:
            supertrend_values, trend_direction = self.strategies['Supertrend'].calculate_supertrend(
                highs, lows, closes
            )
            
            for i in range(1, len(trend_direction)):
                # Bullish trend change
                if trend_direction[i-1] == -1 and trend_direction[i] == 1:
                    entry_price = closes[i]
                    signals.append(PatternSignal(
                        timestamp=df.iloc[i]['timestamp'],
                        pattern_type='Supertrend_Bullish',
                        signal_type=SignalType.BUY,
                        entry_price=entry_price,
                        take_profit=entry_price * (1 + self.tp_percent),
                        stop_loss=entry_price * (1 - self.sl_percent)
                    ))
                
                # Bearish trend change
                elif trend_direction[i-1] == 1 and trend_direction[i] == -1:
                    entry_price = closes[i]
                    signals.append(PatternSignal(
                        timestamp=df.iloc[i]['timestamp'],
                        pattern_type='Supertrend_Bearish',
                        signal_type=SignalType.SELL,
                        entry_price=entry_price,
                        take_profit=entry_price * (1 - self.tp_percent),
                        stop_loss=entry_price * (1 + self.sl_percent)
                    ))
        except Exception as e:
            print(f"Error extracting Supertrend signals: {e}")
        
        return signals
    
    def _extract_bb_signals(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Extract Bollinger Bands signals."""
        signals = []
        closes = df['close'].tolist()
        
        try:
            bb_upper, bb_middle, bb_lower = self.strategies['Bollinger_Bands'].calculate_bollinger_bands(closes)
            
            for i in range(1, len(closes)):
                if i < len(bb_upper) and i < len(bb_lower):
                    # Bounce from lower band (bullish)
                    if closes[i-1] <= bb_lower[i-1] and closes[i] > bb_lower[i]:
                        entry_price = closes[i]
                        signals.append(PatternSignal(
                            timestamp=df.iloc[i + 20]['timestamp'],  # Adjust for BB calculation
                            pattern_type='BB_Lower_Bounce',
                            signal_type=SignalType.BUY,
                            entry_price=entry_price,
                            take_profit=entry_price * (1 + self.tp_percent),
                            stop_loss=entry_price * (1 - self.sl_percent)
                        ))
                    
                    # Rejection from upper band (bearish)
                    elif closes[i-1] >= bb_upper[i-1] and closes[i] < bb_upper[i]:
                        entry_price = closes[i]
                        signals.append(PatternSignal(
                            timestamp=df.iloc[i + 20]['timestamp'],
                            pattern_type='BB_Upper_Rejection',
                            signal_type=SignalType.SELL,
                            entry_price=entry_price,
                            take_profit=entry_price * (1 - self.tp_percent),
                            stop_loss=entry_price * (1 + self.sl_percent)
                        ))
        except Exception as e:
            print(f"Error extracting Bollinger Bands signals: {e}")
        
        return signals
    
    def _extract_rsi_signals(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Extract RSI overbought/oversold signals."""
        signals = []
        closes = df['close'].tolist()
        
        try:
            rsi_values = self.strategies['RSI'].calculate_rsi(closes)
            
            for i in range(1, len(rsi_values)):
                # RSI oversold bounce
                if rsi_values[i-1] <= 30 and rsi_values[i] > 30:
                    entry_price = closes[i + 14]  # Adjust for RSI calculation
                    signals.append(PatternSignal(
                        timestamp=df.iloc[i + 14]['timestamp'],
                        pattern_type='RSI_Oversold_Bounce',
                        signal_type=SignalType.BUY,
                        entry_price=entry_price,
                        take_profit=entry_price * (1 + self.tp_percent),
                        stop_loss=entry_price * (1 - self.sl_percent)
                    ))
                
                # RSI overbought rejection
                elif rsi_values[i-1] >= 70 and rsi_values[i] < 70:
                    entry_price = closes[i + 14]
                    signals.append(PatternSignal(
                        timestamp=df.iloc[i + 14]['timestamp'],
                        pattern_type='RSI_Overbought_Rejection',
                        signal_type=SignalType.SELL,
                        entry_price=entry_price,
                        take_profit=entry_price * (1 - self.tp_percent),
                        stop_loss=entry_price * (1 + self.sl_percent)
                    ))
        except Exception as e:
            print(f"Error extracting RSI signals: {e}")
        
        return signals
    
    def _extract_divergence_signals(self, df: pd.DataFrame) -> List[PatternSignal]:
        """Extract divergence signals."""
        signals = []
        
        try:
            # This would require implementing divergence detection logic
            # For now, return empty list as divergence detection is complex
            pass
        except Exception as e:
            print(f"Error extracting divergence signals: {e}")
        
        return signals
    
    def validate_signals(self, signals: List[PatternSignal], ohlcv_data: pd.DataFrame) -> List[PatternSignal]:
        """
        Validate signals against future price movements.
        
        Args:
            signals: List of signals to validate
            ohlcv_data: OHLCV data for validation
            
        Returns:
            List of validated signals with outcomes
        """
        validated_signals = []
        
        for signal in signals:
            # Find signal index in OHLCV data
            signal_idx = None
            for i, row in ohlcv_data.iterrows():
                if row['timestamp'] >= signal.timestamp:
                    signal_idx = i
                    break
            
            if signal_idx is None or signal_idx >= len(ohlcv_data) - 10:
                continue
            
            # Check future price movements
            outcome, exit_price, exit_timestamp, bars_to_outcome = self._check_signal_outcome(
                signal, ohlcv_data.iloc[signal_idx:], signal_idx
            )
            
            # Update signal with outcome
            signal.outcome = outcome
            signal.exit_price = exit_price
            signal.exit_timestamp = exit_timestamp
            signal.bars_to_outcome = bars_to_outcome
            
            if exit_price:
                if signal.signal_type == SignalType.BUY:
                    signal.profit_loss_pct = (exit_price - signal.entry_price) / signal.entry_price * 100
                else:
                    signal.profit_loss_pct = (signal.entry_price - exit_price) / signal.entry_price * 100
            
            validated_signals.append(signal)
        
        return validated_signals
    
    def _check_signal_outcome(self, signal: PatternSignal, future_data: pd.DataFrame, 
                            start_idx: int) -> Tuple[PatternOutcome, Optional[float], Optional[datetime], Optional[int]]:
        """
        Check if signal hit TP or SL within future data.
        
        Args:
            signal: Pattern signal to check
            future_data: Future OHLCV data
            start_idx: Starting index in the data
            
        Returns:
            Tuple of (outcome, exit_price, exit_timestamp, bars_to_outcome)
        """
        max_bars = min(50, len(future_data))  # Check up to 50 bars
        
        for i in range(1, max_bars):
            row = future_data.iloc[i]
            high, low, close = row['high'], row['low'], row['close']
            
            if signal.signal_type == SignalType.BUY:
                # Check TP first
                if high >= signal.take_profit:
                    return PatternOutcome.SUCCESS, signal.take_profit, row['timestamp'], i
                # Check SL
                elif low <= signal.stop_loss:
                    return PatternOutcome.FAIL, signal.stop_loss, row['timestamp'], i
            
            else:  # SELL signal
                # Check TP first
                if low <= signal.take_profit:
                    return PatternOutcome.SUCCESS, signal.take_profit, row['timestamp'], i
                # Check SL
                elif high >= signal.stop_loss:
                    return PatternOutcome.FAIL, signal.stop_loss, row['timestamp'], i
        
        # No TP/SL hit within timeframe
        last_row = future_data.iloc[max_bars-1]
        return PatternOutcome.NEUTRAL, last_row['close'], last_row['timestamp'], max_bars-1
    
    def calculate_metrics(self, signals: List[PatternSignal], total_candles: int) -> Dict[str, ValidationMetrics]:
        """
        Calculate validation metrics for each pattern type.
        
        Args:
            signals: List of validated signals
            total_candles: Total number of candles analyzed
            
        Returns:
            Dictionary of pattern metrics
        """
        pattern_groups = {}
        
        # Group signals by pattern type
        for signal in signals:
            if signal.pattern_type not in pattern_groups:
                pattern_groups[signal.pattern_type] = []
            pattern_groups[signal.pattern_type].append(signal)
        
        metrics = {}
        
        for pattern_type, pattern_signals in pattern_groups.items():
            wins = [s for s in pattern_signals if s.outcome == PatternOutcome.SUCCESS]
            losses = [s for s in pattern_signals if s.outcome == PatternOutcome.FAIL]
            neutrals = [s for s in pattern_signals if s.outcome == PatternOutcome.NEUTRAL]
            
            total_signals = len(pattern_signals)
            win_count = len(wins)
            loss_count = len(losses)
            
            # Basic metrics
            win_rate = (win_count / total_signals * 100) if total_signals > 0 else 0
            
            avg_win = np.mean([s.profit_loss_pct for s in wins]) if wins else 0
            avg_loss = np.mean([abs(s.profit_loss_pct) for s in losses]) if losses else 0
            
            expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
            
            avg_signal_delay = np.mean([s.bars_to_outcome for s in pattern_signals 
                                     if s.bars_to_outcome is not None])
            
            frequency = (total_signals / total_candles * 100) if total_candles > 0 else 0
            
            # Advanced Analytics
            sharpe_ratio = self._calculate_sharpe_ratio(pattern_signals)
            max_drawdown = self._calculate_max_drawdown(pattern_signals)
            
            # Confusion Matrix (simplified)
            tp = win_count  # True positives
            fp = loss_count  # False positives
            fn = 0  # False negatives (missed opportunities - hard to calculate)
            tn = 0  # True negatives (correctly avoided bad trades - hard to calculate)
            
            metrics[pattern_type] = ValidationMetrics(
                pattern_type=pattern_type,
                total_signals=total_signals,
                win_rate=win_rate,
                expectancy=expectancy,
                avg_win=avg_win,
                avg_loss=avg_loss,
                signal_delay=avg_signal_delay,
                frequency=frequency,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                true_positives=tp,
                false_positives=fp,
                true_negatives=tn,
                false_negatives=fn
            )
        
        return metrics
    
    def _calculate_sharpe_ratio(self, signals: List[PatternSignal], risk_free_rate: float = 0.02) -> Optional[float]:
        """
        Calculate Sharpe ratio for a set of signals.
        
        Args:
            signals: List of pattern signals
            risk_free_rate: Risk-free rate (default 2% annually)
            
        Returns:
            Sharpe ratio or None if insufficient data
        """
        returns = [s.profit_loss_pct/100 for s in signals if s.profit_loss_pct is not None]
        
        if len(returns) < 2:
            return None
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return None
        
        # Annualize the risk-free rate to match signal frequency
        risk_free_rate_adjusted = risk_free_rate / 252  # Daily adjustment
        
        sharpe = (avg_return - risk_free_rate_adjusted) / std_return
        return sharpe
    
    def _calculate_max_drawdown(self, signals: List[PatternSignal]) -> Optional[float]:
        """
        Calculate maximum drawdown for a set of signals.
        
        Args:
            signals: List of pattern signals
            
        Returns:
            Maximum drawdown percentage or None if insufficient data
        """
        returns = [s.profit_loss_pct/100 for s in signals if s.profit_loss_pct is not None]
        
        if len(returns) < 2:
            return None
        
        # Calculate cumulative returns
        cumulative = np.cumprod([1 + r for r in returns])
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        max_drawdown = np.min(drawdown) * 100  # Convert to percentage
        return abs(max_drawdown)
    
    def generate_outcome_histogram(self, pattern_type: str = None) -> Dict[str, List[float]]:
        """
        Generate histogram data for outcome distribution.
        
        Args:
            pattern_type: Specific pattern type or None for all patterns
            
        Returns:
            Dictionary with histogram data
        """
        if pattern_type:
            signals = [s for s in self.signals if s.pattern_type == pattern_type]
        else:
            signals = self.signals
        
        histogram_data = {}
        
        # Group by pattern type
        pattern_groups = {}
        for signal in signals:
            if signal.pattern_type not in pattern_groups:
                pattern_groups[signal.pattern_type] = []
            if signal.profit_loss_pct is not None:
                pattern_groups[signal.pattern_type].append(signal.profit_loss_pct)
        
        for pattern, returns in pattern_groups.items():
            if returns:
                histogram_data[pattern] = returns
        
        return histogram_data
    
    def run_validation(self, symbol: str, timeframe: str, limit: int = 1000) -> Dict[str, ValidationMetrics]:
        """
        Run complete pattern validation analysis.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            limit: Number of candles to analyze
            
        Returns:
            Dictionary of validation metrics per pattern
        """
        print(f"🔍 Running Pattern Validation for {symbol} on {timeframe}")
        print("="*60)
        
        # Fetch OHLCV data
        ohlcv_raw = self.collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if not ohlcv_raw:
            raise RuntimeError(f"No OHLCV data available for {symbol} on {timeframe} timeframe")
        
        # Convert to DataFrame
        ohlcv_data = pd.DataFrame(ohlcv_raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        ohlcv_data['timestamp'] = pd.to_datetime(ohlcv_data['timestamp'], unit='ms')
        
        print(f"📊 Analyzing {len(ohlcv_data)} candles from {ohlcv_data.iloc[0]['timestamp']} to {ohlcv_data.iloc[-1]['timestamp']}")
        
        # Extract signals from all strategies
        all_signals = []
        
        for strategy_name in self.strategies.keys():
            print(f"🔄 Extracting {strategy_name} signals...")
            strategy_signals = self.extract_signals_from_strategy(strategy_name, ohlcv_data, symbol, timeframe)
            all_signals.extend(strategy_signals)
            print(f"   ✅ Found {len(strategy_signals)} {strategy_name} signals")
        
        print(f"\n📈 Total signals extracted: {len(all_signals)}")
        
        # Validate signals
        print("🔍 Validating signals against future price movements...")
        validated_signals = self.validate_signals(all_signals, ohlcv_data)
        
        # Calculate metrics
        metrics = self.calculate_metrics(validated_signals, len(ohlcv_data))
        
        # Store results
        self.signals = validated_signals
        self.validation_metrics = metrics
        
        return metrics
    
    def print_results(self) -> None:
        """Print validation results in a formatted way."""
        if not self.validation_metrics:
            print("❌ No validation results available. Run validation first.")
            return
        
        print("\n" + "="*80)
        print("📊 PATTERN VALIDATION RESULTS")
        print("="*80)
        
        for pattern_type, metrics in self.validation_metrics.items():
            print(f"\n🎯 {pattern_type}")
            print("-" * 50)
            print(f"Total Signals: {metrics.total_signals}")
            print(f"Win Rate: {metrics.win_rate:.2f}%")
            print(f"Expectancy: {metrics.expectancy:.4f}%")
            print(f"Avg Win: {metrics.avg_win:.2f}%")
            print(f"Avg Loss: {metrics.avg_loss:.2f}%")
            print(f"Signal Delay: {metrics.signal_delay:.1f} bars")
            print(f"Frequency: {metrics.frequency:.3f}% (per 100 candles)")
            
            # Advanced Analytics
            if metrics.sharpe_ratio is not None:
                print(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
            if metrics.max_drawdown is not None:
                print(f"Max Drawdown: {metrics.max_drawdown:.2f}%")
            
            # Confusion Matrix
            print(f"\n📈 Classification Matrix:")
            print(f"True Positives: {metrics.true_positives}")
            print(f"False Positives: {metrics.false_positives}")
            
            # Outcome Distribution Sample
            histogram_data = self.generate_outcome_histogram(pattern_type)
            if pattern_type in histogram_data and histogram_data[pattern_type]:
                returns = histogram_data[pattern_type]
                print(f"\n📊 Outcome Distribution:")
                print(f"Best Return: {max(returns):.2f}%")
                print(f"Worst Return: {min(returns):.2f}%")
                print(f"Median Return: {np.median(returns):.2f}%")
                print(f"Std Deviation: {np.std(returns):.2f}%")
            
        print("\n" + "="*80)
        print("✅ Pattern Validation Complete")
        print("="*80)
    
    def export_results(self, filename: str = None) -> str:
        """
        Export validation results to JSON file.
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pattern_validation_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        export_data = {
            'validation_settings': {
                'take_profit_percent': self.tp_percent * 100,
                'stop_loss_percent': self.sl_percent * 100,
                'timestamp': datetime.now().isoformat()
            },
            'metrics': {},
            'signals': []
        }
        
        # Export metrics
        for pattern_type, metrics in self.validation_metrics.items():
            export_data['metrics'][pattern_type] = {
                'total_signals': metrics.total_signals,
                'win_rate': metrics.win_rate,
                'expectancy': metrics.expectancy,
                'avg_win': metrics.avg_win,
                'avg_loss': metrics.avg_loss,
                'signal_delay': metrics.signal_delay,
                'frequency': metrics.frequency,
                'true_positives': metrics.true_positives,
                'false_positives': metrics.false_positives
            }
        
        # Export signals
        for signal in self.signals:
            export_data['signals'].append({
                'timestamp': signal.timestamp.isoformat(),
                'pattern_type': signal.pattern_type,
                'signal_type': signal.signal_type.value,
                'entry_price': signal.entry_price,
                'outcome': signal.outcome.value if signal.outcome else None,
                'profit_loss_pct': signal.profit_loss_pct,
                'bars_to_outcome': signal.bars_to_outcome
            })
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📄 Results exported to: {filename}")
        return filename


def parse_command(command: str) -> Tuple[str, str, int, float, float]:
    """
    Parse terminal command: stat_validation s=BTC/USDT t=4h l=1000 tp=2.0 sl=1.0
    
    Args:
        command: Command string
        
    Returns:
        Tuple of (symbol, timeframe, limit, tp_percent, sl_percent)
    """
    parts = command.strip().split()
    
    if len(parts) < 2 or parts[0] != 'stat_validation':
        raise ValueError("Invalid command format. Use: stat_validation s=SYMBOL t=TIMEFRAME l=LIMIT tp=TP_PERCENT sl=SL_PERCENT")
    
    symbol = None
    timeframe = '4h'  # default
    limit = 1000  # default
    tp_percent = 2.0  # default
    sl_percent = 1.0  # default
    
    for part in parts[1:]:
        if part.startswith('s='):
            symbol = part[2:]
        elif part.startswith('t='):
            timeframe = part[2:]
        elif part.startswith('l='):
            limit = int(part[2:])
        elif part.startswith('tp='):
            tp_percent = float(part[3:])
        elif part.startswith('sl='):
            sl_percent = float(part[3:])
    
    if symbol is None:
        raise ValueError("Symbol (s=) is required")
    
    return symbol, timeframe, limit, tp_percent, sl_percent


def main():
    """Main entry point for pattern validation."""
    if len(sys.argv) < 2:
        print("Usage: python stat_pattern_validation.py s=SYMBOL t=TIMEFRAME l=LIMIT tp=TP_PERCENT sl=SL_PERCENT")
        print("Example: python stat_pattern_validation.py s=BTC/USDT t=4h l=1000 tp=2.0 sl=1.0")
        return
    
    try:
        # Parse command line arguments
        command = ' '.join(['stat_validation'] + sys.argv[1:])
        symbol, timeframe, limit, tp_percent, sl_percent = parse_command(command)
        
        # Run validation
        validator = StatPatternValidation(tp_percent=tp_percent, sl_percent=sl_percent)
        metrics = validator.run_validation(symbol, timeframe, limit)
        
        # Print results
        validator.print_results()
        
        # Export results
        validator.export_results()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()