"""
VWAP (Volume Weighted Average Price) Analysis
Calculates VWAP and provides trading signals based on price position relative to VWAP
"""

import pandas as pd
from typing import List, Optional
from data import get_data_collector


def calculate_vwap(df: pd.DataFrame, anchor_index: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate VWAP (Volume Weighted Average Price)
    
    Args:
        df: DataFrame with OHLCV data
        anchor_index: Optional index to start VWAP calculation from (for anchored VWAP)
    
    Returns:
        DataFrame with VWAP values added
    """
    df = df.copy()
    
    # Calculate typical price (H+L+C)/3
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate price * volume
    df['pv'] = df['typical_price'] * df['volume']
    
    # Set starting point for calculation
    start_index = anchor_index if anchor_index is not None else 0
    
    # Initialize VWAP column
    df['vwap'] = 0.0
    
    # Calculate cumulative VWAP from anchor point
    cumulative_pv = 0
    cumulative_volume = 0
    
    for i in range(start_index, len(df)):
        cumulative_pv += df.iloc[i]['pv']
        cumulative_volume += df.iloc[i]['volume']
        
        if cumulative_volume > 0:
            df.iloc[i, df.columns.get_loc('vwap')] = cumulative_pv / cumulative_volume
    
    return df


def analyze_vwap_signals(df: pd.DataFrame) -> List[dict]:
    """
    Analyze VWAP signals and generate trading recommendations
    
    Args:
        df: DataFrame with OHLCV data and VWAP
    
    Returns:
        List of signal dictionaries
    """
    signals = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        close_price = row['close']
        vwap_value = row['vwap']
        timestamp = row['timestamp']
        
        # Determine signal based on price position relative to VWAP
        if close_price > vwap_value:
            signal = "BUY"
            bias = "🟢 Price Above VWAP"
        elif close_price < vwap_value:
            signal = "SELL" 
            bias = "🔻 Bearish Bias"
        else:
            signal = "NEUTRAL"
            bias = "⚖️ At VWAP"
        
        signals.append({
            'timestamp': timestamp,
            'close': close_price,  
            'vwap': vwap_value,
            'signal': signal,
            'bias': bias
        })
    
    return signals


def format_vwap_output(signals: List[dict]) -> str:
    """
    Format VWAP analysis output for terminal display
    
    Args:
        signals: List of signal dictionaries
    
    Returns:
        Formatted string for terminal output
    """
    output_lines = []
    
    for signal in signals:
        timestamp = signal['timestamp']
        close = signal['close']
        vwap = signal['vwap']
        signal_type = signal['signal']
        bias = signal['bias']
        
        # Determine trend emoji based on signal
        if signal_type == "BUY":
            trend_emoji = "📈"
        elif signal_type == "SELL":
            trend_emoji = "📉"
        else:
            trend_emoji = "➖"
        
        line = f"[{timestamp}] Price: {close:.4f} | VWAP: {vwap:.4f} | Signal: {signal_type} | {trend_emoji} {bias}"
        output_lines.append(line)
    
    return '\n'.join(output_lines)


def analyze(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 100, anchor: Optional[str] = None) -> dict:
    """
    Run complete VWAP analysis
    
    Args:
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        timeframe: Timeframe for analysis (e.g., "1h", "4h", "1d")
        limit: Number of candles to analyze
        anchor: Optional anchor timestamp for anchored VWAP (format: "2025-07-29T04:00:00")
    
    Returns:
        Dictionary with analysis results
    """
    try:
        collector = get_data_collector()
        
        # Fetch OHLCV data
        ohlcv_data = collector.fetch_ohlcv_data(symbol, timeframe, limit)
        
        if not ohlcv_data:
            return f"Error: No data received for {symbol} {timeframe}"
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to readable format
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Jakarta')
        
        # Find anchor index if anchor timestamp is provided
        anchor_index = None
        if anchor:
            try:
                anchor_dt = pd.to_datetime(anchor)
                # Find closest timestamp to anchor
                time_diffs = abs(df['timestamp'] - anchor_dt)
                anchor_index = time_diffs.idxmin()
            except Exception as e:
                return f"Error parsing anchor timestamp: {e}"
        
        # Calculate VWAP
        df_with_vwap = calculate_vwap(df, anchor_index)
        
        # Analyze signals
        signals = analyze_vwap_signals(df_with_vwap)
        
        # Get latest signal
        latest_signal = signals[-1] if signals else None
        
        # Calculate overall bias
        buy_signals = sum(1 for s in signals if s['signal'] == 'BUY')
        sell_signals = sum(1 for s in signals if s['signal'] == 'SELL')
        total_signals = len(signals)
        
        if buy_signals > sell_signals:
            overall_bias = "Bullish"
            bias_confidence = (buy_signals / total_signals) * 100
        elif sell_signals > buy_signals:
            overall_bias = "Bearish"
            bias_confidence = (sell_signals / total_signals) * 100
        else:
            overall_bias = "Neutral"
            bias_confidence = 50.0
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'latest_signal': latest_signal,
            'overall_bias': overall_bias,
            'bias_confidence': round(bias_confidence, 2),
            'signals_analyzed': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'formatted_output': format_vwap_output(signals),
            'anchor_info': f" (Anchored from {anchor})" if anchor else ""
        }
        
    except Exception as e:
        return {
            'error': f"Error in VWAP analysis: {str(e)}",
            'symbol': symbol,
            'timeframe': timeframe
        }


