from typing import List, Dict
import pandas as pd
import logging
from summary import add_indicator_result, IndicatorResult
from config import get_indicator_params

logger = logging.getLogger(__name__)


class ChaikinMoneyFlow:
    
    def __init__(self, 
                 symbol: str,
                 timeframe: str,
                 limit: int,
                 ob: dict,
                 ticker: dict,            
                 ohlcv: List[List],       
                 trades: List[Dict]):    
        
        self.param = get_indicator_params('chaikin_money_flow', timeframe)
        self.ob = ob
        self.ohlcv = ohlcv
        self.trades = trades
        self.ticker = ticker
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
    
    def calculate(self):
        """
        Calculate Chaikin Money Flow (CMF) developed by Marc Chaikin.
        
        The Chaikin Money Flow indicator measures the flow of money into and out of a security
        over a specified period, combining price and volume to assess buying/selling pressure.
        
        Formula (Source: Marc Chaikin, TradingView, Fidelity, Investopedia):
        1. Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        2. Money Flow Volume = Money Flow Multiplier × Volume
        3. CMF = Sum(Money Flow Volume, N) / Sum(Volume, N)
        
        Where:
        - N = Period (typically 20 or 21 days)
        - High, Low, Close = Price data for each period
        - Volume = Trading volume for each period
        
        Interpretation:
        - CMF > 0: Buying pressure dominates (bullish)
        - CMF < 0: Selling pressure dominates (bearish)
        - CMF near 0: Balanced buying/selling pressure
        - CMF > +0.25: Strong buying pressure
        - CMF < -0.25: Strong selling pressure
        
        References:
        - Developed by Marc Chaikin in the 1980s
        - TradingView: https://www.tradingview.com/support/solutions/43000501970-chaikin-money-flow-cmf/
        - Fidelity: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/chaikin-money-flow
        - Investopedia: https://www.investopedia.com/terms/c/chaikinmoneyflow.asp
        - StockCharts: https://school.stockcharts.com/doku.php?id=technical_indicators:chaikin_money_flow_cmf
        
        Key Features:
        - Combines price and volume analysis
        - Leading indicator for price reversals
        - Effective for identifying accumulation/distribution phases
        - Works best in trending markets
        """
        if not self.ohlcv or len(self.ohlcv) < self.param["period"]:
            result = {
                "error": f"Insufficient data: need at least {self.param['period']} candles, got {len(self.ohlcv) if self.ohlcv else 0}"
            }
            return result
            
        df = pd.DataFrame(self.ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        period = self.param["period"]
        
        # Calculate Money Flow Multiplier
        # Avoid division by zero when high == low
        hl_diff = df['high'] - df['low']
        hl_diff = hl_diff.replace(0, 0.0001)  # Small value to avoid division by zero
        
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / hl_diff
        
        # Calculate Money Flow Volume
        money_flow_volume = money_flow_multiplier * df['volume']
        
        # Calculate CMF using rolling sums
        mfv_sum = money_flow_volume.rolling(window=period).sum()
        volume_sum = df['volume'].rolling(window=period).sum()
        
        # Avoid division by zero
        volume_sum = volume_sum.replace(0, 1)  # Replace zero volume with 1 to avoid division by zero
        cmf = mfv_sum / volume_sum
        
        # Current values
        current_cmf = float(cmf.iloc[-1]) if not pd.isna(cmf.iloc[-1]) else 0.0
        current_price = float(df['close'].iloc[-1])
        current_volume = float(df['volume'].iloc[-1])
        current_mfm = float(money_flow_multiplier.iloc[-1])
        current_mfv = float(money_flow_volume.iloc[-1])
        
        # Volume analysis
        volume_ma = df['volume'].rolling(window=self.param["volume_ma_period"]).mean()
        high_volume = current_volume > (volume_ma.iloc[-1] * self.param["high_volume_multiplier"])
        
        # Signal generation
        signal = "neutral"
        strength = "normal"
        
        if current_cmf >= self.param["strong_bullish"]:
            signal = "strong_bullish"
            strength = "strong"
        elif current_cmf >= self.param["bullish_threshold"]:
            signal = "bullish"
            strength = "weak" if current_cmf < 0.15 else "normal"
        elif current_cmf <= self.param["strong_bearish"]:
            signal = "strong_bearish"
            strength = "strong"
        elif current_cmf <= self.param["bearish_threshold"]:
            signal = "bearish"
            strength = "weak" if current_cmf > -0.15 else "normal"
        
        # Divergence detection (simplified)
        if len(cmf) >= self.param["divergence_period"] + 1:
            price_trend = df['close'].iloc[-1] - df['close'].iloc[-self.param["divergence_period"]]
            cmf_trend = cmf.iloc[-1] - cmf.iloc[-self.param["divergence_period"]]
            
            # Bullish divergence: price falling, CMF rising
            bullish_divergence = (price_trend < -self.param["price_trend_strength"] and 
                                 cmf_trend > self.param["cmf_trend_strength"])
            
            # Bearish divergence: price rising, CMF falling
            bearish_divergence = (price_trend > self.param["price_trend_strength"] and 
                                 cmf_trend < -self.param["cmf_trend_strength"])
        else:
            bullish_divergence = False
            bearish_divergence = False
        
        # Money flow direction
        if current_mfm > 0.5:
            flow_direction = "accumulation"
        elif current_mfm < -0.5:
            flow_direction = "distribution" 
        else:
            flow_direction = "neutral"
        
        result = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": current_price,
            "cmf": current_cmf,
            "money_flow_multiplier": current_mfm,
            "money_flow_volume": current_mfv,
            "signal": signal,
            "strength": strength,
            "flow_direction": flow_direction,
            "high_volume": high_volume,
            "bullish_divergence": bullish_divergence,
            "bearish_divergence": bearish_divergence,
            "parameters": {
                "period": period,
                "bullish_threshold": self.param["bullish_threshold"],
                "bearish_threshold": self.param["bearish_threshold"],
                "strong_bullish": self.param["strong_bullish"],
                "strong_bearish": self.param["strong_bearish"]
            }
        }
        
        # Add result to analysis summary
        indicator_result = IndicatorResult(
            name="Chaikin Money Flow",
            signal=result["signal"],
            value=result["cmf"],
            strength=result["strength"],
            metadata={
                "flow_direction": result["flow_direction"],
                "money_flow_multiplier": result["money_flow_multiplier"],
                "money_flow_volume": result["money_flow_volume"],
                "high_volume": result["high_volume"],
                "bullish_divergence": result["bullish_divergence"],
                "bearish_divergence": result["bearish_divergence"],
                "parameters": result["parameters"]
            }
        )
        add_indicator_result(indicator_result)
        
        return result
