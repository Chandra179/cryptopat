#!/usr/bin/env python3
"""
Comprehensive Analysis Coordinator
Runs all analysis modules and formats output using the CLI formatter
"""

import sys
import traceback
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent))

# Import all analysis modules
from cli_formatter import format_analysis_output
from data import get_data_collector

# Import all analysis modules with correct class names
available_techin = {}
available_patterns = {}
available_orderflow = {}

# Technical indicators
try:
    from techin.ema_9_21 import EMA9_21Strategy
    available_techin['ema_9_21'] = EMA9_21Strategy
except ImportError as e:
    print(f"Warning: Could not import EMA9_21Strategy: {e}")

try:
    from techin.macd import MACDStrategy
    available_techin['macd'] = MACDStrategy
except ImportError as e:
    print(f"Warning: Could not import MACDStrategy: {e}")

try:
    from techin.supertrend import SupertrendStrategy
    available_techin['supertrend'] = SupertrendStrategy
except ImportError as e:
    print(f"Warning: Could not import SupertrendStrategy: {e}")

try:
    from techin.bollinger_bands import BollingerBandsStrategy
    available_techin['bollinger_bands'] = BollingerBandsStrategy
except ImportError as e:
    print(f"Warning: Could not import BollingerBandsStrategy: {e}")

try:
    from techin.rsi_14 import RSI14Strategy
    available_techin['rsi_14'] = RSI14Strategy
except ImportError as e:
    print(f"Warning: Could not import RSI14Strategy: {e}")

try:
    from techin.atr_adx import ATR_ADXStrategy
    available_techin['atr_adx'] = ATR_ADXStrategy
except ImportError as e:
    print(f"Warning: Could not import ATR_ADXStrategy: {e}")

try:
    from techin.obv import OBVStrategy
    available_techin['obv'] = OBVStrategy
except ImportError as e:
    print(f"Warning: Could not import OBVStrategy: {e}")

try:
    from techin.vwap import VWAPStrategy
    available_techin['vwap'] = VWAPStrategy
except ImportError as e:
    print(f"Warning: Could not import VWAPStrategy: {e}")

# Chart patterns
try:
    from pattern.flag import FlagStrategy
    available_patterns['flag'] = FlagStrategy
except ImportError as e:
    print(f"Warning: Could not import FlagStrategy: {e}")

try:
    from pattern.triangle import TriangleStrategy
    available_patterns['triangle'] = TriangleStrategy
except ImportError as e:
    print(f"Warning: Could not import TriangleStrategy: {e}")

try:
    from pattern.head_and_shoulders import HeadAndShouldersStrategy
    available_patterns['head_and_shoulders'] = HeadAndShouldersStrategy
except ImportError as e:
    print(f"Warning: Could not import HeadAndShouldersStrategy: {e}")

try:
    from pattern.inverse_head_and_shoulders import InverseHeadAndShouldersStrategy
    available_patterns['inverse_head_and_shoulders'] = InverseHeadAndShouldersStrategy
except ImportError as e:
    print(f"Warning: Could not import InverseHeadAndShouldersStrategy: {e}")

try:
    from pattern.double_top import DoubleTopStrategy
    available_patterns['double_top'] = DoubleTopStrategy
except ImportError as e:
    print(f"Warning: Could not import DoubleTopStrategy: {e}")

try:
    from pattern.double_bottom import DoubleBottomStrategy
    available_patterns['double_bottom'] = DoubleBottomStrategy
except ImportError as e:
    print(f"Warning: Could not import DoubleBottomStrategy: {e}")

try:
    from pattern.wedge import WedgeStrategy
    available_patterns['wedge'] = WedgeStrategy
except ImportError as e:
    print(f"Warning: Could not import WedgeStrategy: {e}")

try:
    from pattern.shark_pattern import SharkStrategy
    available_patterns['shark_pattern'] = SharkStrategy
except ImportError as e:
    print(f"Warning: Could not import SharkStrategy: {e}")

try:
    from pattern.butterfly_pattern import ButterflyStrategy
    available_patterns['butterfly_pattern'] = ButterflyStrategy
except ImportError as e:
    print(f"Warning: Could not import ButterflyStrategy: {e}")

try:
    from pattern.elliott_wave import ElliottWaveStrategy
    available_patterns['elliott_wave'] = ElliottWaveStrategy
except ImportError as e:
    print(f"Warning: Could not import ElliottWaveStrategy: {e}")

# Order flow analysis
try:
    from orderflow.smc import SMCStrategy
    available_orderflow['smc'] = SMCStrategy
except ImportError as e:
    print(f"Warning: Could not import SMCStrategy: {e}")

try:
    from orderflow.cvd import CVDAnalyzer
    available_orderflow['cvd'] = CVDAnalyzer
except ImportError as e:
    print(f"Warning: Could not import CVDAnalyzer: {e}")

try:
    from orderflow.imbalance import OrderFlowImbalanceDetector
    available_orderflow['imbalance'] = OrderFlowImbalanceDetector
except ImportError as e:
    print(f"Warning: Could not import OrderFlowImbalanceDetector: {e}")

try:
    from orderflow.absorption import AbsorptionStrategy
    available_orderflow['absorption'] = AbsorptionStrategy
except ImportError as e:
    print(f"Warning: Could not import AbsorptionStrategy: {e}")

try:
    from orderflow.stop_sweep import StopSweepStrategy
    available_orderflow['stop_sweep'] = StopSweepStrategy
except ImportError as e:
    print(f"Warning: Could not import StopSweepStrategy: {e}")

try:
    from orderflow.footprint import FootprintStrategy
    available_orderflow['footprint'] = FootprintStrategy
except ImportError as e:
    print(f"Warning: Could not import FootprintStrategy: {e}")

try:
    from orderflow.orderbook_heatmap import OrderBookHeatmapStrategy
    available_orderflow['orderbook_heatmap'] = OrderBookHeatmapStrategy
except ImportError as e:
    print(f"Warning: Could not import OrderBookHeatmapStrategy: {e}")


class ComprehensiveAnalyzer:
    """Comprehensive market analysis coordinator"""
    
    def __init__(self):
        """Initialize all analysis strategies"""
        self.techin_strategies = {}
        self.pattern_strategies = {}
        self.orderflow_strategies = {}
        self.collector = get_data_collector()
        
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize all available strategies"""
        
        # Initialize technical indicators
        for name, strategy_class in available_techin.items():
            try:
                self.techin_strategies[name] = strategy_class()
            except Exception as e:
                print(f"Warning: Could not initialize {name}: {e}")
        
        # Initialize chart patterns
        for name, strategy_class in available_patterns.items():
            try:
                self.pattern_strategies[name] = strategy_class()
            except Exception as e:
                print(f"Warning: Could not initialize {name}: {e}")
        
        # Initialize order flow strategies
        for name, strategy_class in available_orderflow.items():
            try:
                self.orderflow_strategies[name] = strategy_class()
            except Exception as e:
                print(f"Warning: Could not initialize {name}: {e}")
    
    def _run_analysis_category(self, strategies: Dict[str, Any], symbol: str, timeframe: str, candles: int, ohlcv_data: List[List]) -> Dict[str, Any]:
        """Run analysis for a category of strategies with shared OHLCV data"""
        results = {}
        
        for name, strategy in strategies.items():
            try:
                print(f"  Running {name}...")
                result = strategy.analyze(symbol, timeframe, candles, ohlcv_data=ohlcv_data)
                results[name] = result
                
                # Basic validation
                if not isinstance(result, dict):
                    print(f"    Warning: {name} returned non-dict result")
                    results[name] = {'success': False, 'error': 'Invalid result type'}
                elif result.get('success') == False:  # Only show error if explicitly marked as failed
                    print(f"    Warning: {name} analysis failed: {result.get('error', 'Unknown error')}")
                elif 'error' in result:
                    print(f"    Warning: {name} analysis failed: {result['error']}")
                else:
                    print(f"    ✅ {name} completed successfully")
                
            except Exception as e:
                print(f"    Error in {name}: {str(e)}")
                results[name] = {
                    'success': False,
                    'error': f"Exception: {str(e)}",
                    'symbol': symbol,
                    'timeframe': timeframe
                }
                # Print traceback for debugging
                traceback.print_exc()
        
        return results
    
    def analyze_comprehensive(self, symbol: str, timeframe: str, candles: int) -> str:
        """
        Run comprehensive analysis and return formatted CLI output
        
        Args:
            symbol: Trading symbol (e.g. 'BTC/USDT')
            timeframe: Analysis timeframe (e.g. '1h')
            candles: Number of candles to analyze
            
        Returns:
            Formatted CLI output string
        """
        print(f"🚀 Starting comprehensive analysis for {symbol} {timeframe} ({candles} candles)")
        print("=" * 80)
        
        # Fetch OHLCV data once for all analysis methods
        print(f"\n📊 Fetching OHLCV data for {symbol} {timeframe}...")
        try:
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, candles)
            print(f"✅ Successfully fetched {len(ohlcv_data)} candles")
        except Exception as e:
            error_msg = f"Failed to fetch OHLCV data: {str(e)}"
            print(f"❌ {error_msg}")
            return f"""
===============================================================================
                        CRYPTOPAT ANALYSIS REPORT                            
===============================================================================
Symbol: {symbol} | Timeframe: {timeframe} | Candles: {candles}
Status: ERROR - {error_msg}
===============================================================================

Unable to fetch market data. Please check:
- Internet connection
- Symbol format (e.g., 'BTC/USDT')
- Exchange availability
===============================================================================
"""
        
        # Run technical indicators
        print("\n📈 Running technical indicators...")
        techin_results = self._run_analysis_category(self.techin_strategies, symbol, timeframe, candles, ohlcv_data)
        
        # Run chart patterns
        print("\n📊 Running chart pattern analysis...")
        pattern_results = self._run_analysis_category(self.pattern_strategies, symbol, timeframe, candles, ohlcv_data)
        
        # Run order flow analysis
        print("\n💰 Running order flow analysis...")
        orderflow_results = self._run_analysis_category(self.orderflow_strategies, symbol, timeframe, candles, ohlcv_data)
        
        print("\n📋 Formatting comprehensive report...")
        
        # Generate formatted output
        try:
            formatted_output = format_analysis_output(
                techin_results=techin_results,
                pattern_results=pattern_results,
                orderflow_results=orderflow_results,
                symbol=symbol,
                timeframe=timeframe,
                candles=candles
            )
            
            print("✅ Analysis complete!")
            return formatted_output
            
        except Exception as e:
            error_msg = f"Error formatting output: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            
            # Return basic error report
            return f"""
===============================================================================
                        CRYPTOPAT ANALYSIS REPORT                            
===============================================================================
Symbol: {symbol} | Timeframe: {timeframe} | Candles: {candles}
Status: ERROR - {error_msg}
===============================================================================

Technical Indicators: {len([r for r in techin_results.values() if r.get('success', False)])} successful
Chart Patterns: {len([r for r in pattern_results.values() if r.get('success', False)])} successful  
Order Flow: {len([r for r in orderflow_results.values() if r.get('success', False)])} successful

Please check logs for detailed error information.
===============================================================================
"""
    
    def get_analysis_summary(self, symbol: str, timeframe: str, candles: int) -> Dict[str, Any]:
        """
        Get raw analysis results without formatting
        
        Returns:
            Dictionary with raw results from all categories
        """
        print(f"Running analysis summary for {symbol} {timeframe} ({candles} candles)")
        
        # Fetch OHLCV data once for all analysis methods
        try:
            ohlcv_data = self.collector.fetch_ohlcv_data(symbol, timeframe, candles)
            print(f"Fetched {len(ohlcv_data)} candles")
        except Exception as e:
            return {
                'error': f"Failed to fetch OHLCV data: {str(e)}",
                'symbol': symbol,
                'timeframe': timeframe,
                'candles': candles
            }
        
        techin_results = self._run_analysis_category(self.techin_strategies, symbol, timeframe, candles, ohlcv_data)
        pattern_results = self._run_analysis_category(self.pattern_strategies, symbol, timeframe, candles, ohlcv_data)
        orderflow_results = self._run_analysis_category(self.orderflow_strategies, symbol, timeframe, candles, ohlcv_data)
        
        return {
            'techin_results': techin_results,
            'pattern_results': pattern_results,
            'orderflow_results': orderflow_results,
            'symbol': symbol,
            'timeframe': timeframe,
            'candles': candles,
            'total_successful': (
                len([r for r in techin_results.values() if r.get('success', False)]) +
                len([r for r in pattern_results.values() if r.get('success', False)]) +
                len([r for r in orderflow_results.values() if r.get('success', False)])
            )
        }


def main():
    """Main function for command line usage"""
    if len(sys.argv) != 4:
        print("Usage: python comprehensive_analyzer.py <symbol> <timeframe> <candles>")
        print("Example: python comprehensive_analyzer.py BTC/USDT 1h 100")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    timeframe = sys.argv[2]
    candles = int(sys.argv[3])
    
    analyzer = ComprehensiveAnalyzer()
    result = analyzer.analyze_comprehensive(symbol, timeframe, candles)
    
    print("\n" + result)


if __name__ == "__main__":
    main()