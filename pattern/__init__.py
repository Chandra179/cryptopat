"""
Pattern Recognition Module for CryptoPat.

This module contains chart pattern detection algorithms for cryptocurrency analysis.
Currently supports:
- Double Bottom pattern detection
- All patterns consolidated analysis

Future patterns to be added:
- Double Top
- Head and Shoulders
- Triangle patterns
- Flag and Pennant patterns
- Wedge patterns
"""

from .double_bottom import DoubleBottomStrategy
# from .all_patterns import analyze_all_patterns, AllPatternsAnalyzer

__all__ = [
    'DoubleBottomStrategy',
    # 'analyze_all_patterns', 
    # 'AllPatternsAnalyzer'
]