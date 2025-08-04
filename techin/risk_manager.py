#!/usr/bin/env python3
"""
Advanced Risk Management Calculator
Industry-standard position sizing, risk/reward optimization, and portfolio management
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics
import math


@dataclass
class TradeSetup:
    """Complete trade setup with all parameters"""
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    signal: str  # 'BUY' or 'SELL'
    confidence: int
    atr_value: float
    timeframe: str


@dataclass
class RiskMetrics:
    """Risk metrics for a trade or portfolio"""
    risk_amount: float
    risk_percentage: float
    position_size: float
    units: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float


@dataclass
class PortfolioPosition:
    """Individual position in portfolio"""
    symbol: str
    units: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    risk_amount: float


class RiskManager:
    """Advanced risk management system with industry-standard methods"""
    
    def __init__(self, account_balance: float = 10000):
        self.account_balance = account_balance
        self.max_risk_per_trade = 2.0  # 2% max risk per trade
        self.max_portfolio_risk = 10.0  # 10% max total portfolio risk
        self.max_positions = 5  # Maximum concurrent positions
        self.min_risk_reward = 1.5  # Minimum R/R ratio
        self.position_correlation_limit = 0.7  # Max correlation between positions
        
        # Risk scaling based on confidence
        self.confidence_multipliers = {
            90: 1.0,    # Full position size
            80: 0.8,    # 80% of calculated size
            70: 0.6,    # 60% of calculated size
            60: 0.4,    # 40% of calculated size
            50: 0.2,    # 20% of calculated size
        }
    
    def calculate_position_size(self, trade_setup: TradeSetup, 
                               base_risk_percent: float = None) -> Dict[str, Any]:
        """
        Calculate optimal position size using multiple methods
        
        Methods:
        1. Fixed percentage risk
        2. Volatility-adjusted sizing (ATR-based)
        3. Confidence-adjusted sizing
        4. Kelly Criterion (if historical data available)
        """
        
        if base_risk_percent is None:
            base_risk_percent = self.max_risk_per_trade
        
        entry_price = trade_setup.entry_price
        stop_loss = trade_setup.stop_loss
        confidence = trade_setup.confidence
        atr_value = trade_setup.atr_value
        
        # Method 1: Fixed percentage risk
        risk_amount = self.account_balance * (base_risk_percent / 100)
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return self._empty_position_result()
        
        base_units = risk_amount / risk_per_unit
        base_position_value = base_units * entry_price
        base_position_percent = (base_position_value / self.account_balance) * 100
        
        # Method 2: Volatility adjustment (ATR-based)
        if atr_value > 0:
            volatility_percent = (atr_value / entry_price) * 100
            
            # Reduce position size in high volatility
            if volatility_percent > 5.0:  # High volatility
                volatility_multiplier = 0.5
            elif volatility_percent > 3.0:  # Medium volatility
                volatility_multiplier = 0.75
            elif volatility_percent < 1.0:  # Low volatility
                volatility_multiplier = 1.25
            else:  # Normal volatility
                volatility_multiplier = 1.0
        else:
            volatility_multiplier = 1.0
            volatility_percent = 2.0  # Default assumption
        
        # Method 3: Confidence adjustment
        confidence_multiplier = self._get_confidence_multiplier(confidence)
        
        # Method 4: Risk/Reward adjustment
        rr_ratio = self._calculate_rr_ratio(trade_setup)
        if rr_ratio < self.min_risk_reward:
            rr_multiplier = 0.5  # Reduce size for poor R/R
        elif rr_ratio > 3.0:
            rr_multiplier = 1.2  # Increase size for excellent R/R
        else:
            rr_multiplier = 1.0
        
        # Combine all multipliers
        final_multiplier = volatility_multiplier * confidence_multiplier * rr_multiplier
        final_units = base_units * final_multiplier
        final_position_value = final_units * entry_price
        final_position_percent = (final_position_value / self.account_balance) * 100
        final_risk_amount = final_units * risk_per_unit
        
        # Apply maximum position size limits
        max_position_percent = 20.0  # Max 20% of account per position
        if final_position_percent > max_position_percent:
            size_reduction = max_position_percent / final_position_percent
            final_units *= size_reduction
            final_position_value *= size_reduction
            final_position_percent = max_position_percent
            final_risk_amount *= size_reduction
        
        # Calculate expected returns
        tp1_return = abs(trade_setup.take_profit_1 - entry_price) * final_units
        tp2_return = abs(trade_setup.take_profit_2 - entry_price) * final_units
        max_loss = final_risk_amount
        
        # Expected value calculation (assuming 50% hit TP1, 25% hit TP2, 25% hit SL)
        expected_value = (tp1_return * 0.5) + (tp2_return * 0.25) - (max_loss * 0.25)
        
        return {
            'base_calculation': {
                'units': round(base_units, 8),
                'position_value': round(base_position_value, 2),
                'position_percent': round(base_position_percent, 2),
                'risk_amount': round(risk_amount, 2)
            },
            'adjustments': {
                'volatility_multiplier': round(volatility_multiplier, 2),
                'confidence_multiplier': round(confidence_multiplier, 2),
                'rr_multiplier': round(rr_multiplier, 2),
                'final_multiplier': round(final_multiplier, 2)
            },
            'final_position': {
                'units': round(final_units, 8),
                'position_value': round(final_position_value, 2),
                'position_percent': round(final_position_percent, 2),
                'risk_amount': round(final_risk_amount, 2),
                'risk_percent': round((final_risk_amount / self.account_balance) * 100, 2)
            },
            'expected_returns': {
                'tp1_return': round(tp1_return, 2),
                'tp2_return': round(tp2_return, 2),
                'max_loss': round(max_loss, 2),
                'expected_value': round(expected_value, 2),
                'rr_ratio': round(rr_ratio, 2)
            },
            'risk_metrics': {
                'volatility_percent': round(volatility_percent, 2),
                'risk_per_unit': round(risk_per_unit, 4),
                'max_drawdown_percent': round((final_risk_amount / self.account_balance) * 100, 2)
            }
        }
    
    def _get_confidence_multiplier(self, confidence: int) -> float:
        """Get position size multiplier based on confidence level"""
        for conf_level in sorted(self.confidence_multipliers.keys(), reverse=True):
            if confidence >= conf_level:
                return self.confidence_multipliers[conf_level]
        return 0.1  # Very low confidence
    
    def _calculate_rr_ratio(self, trade_setup: TradeSetup) -> float:
        """Calculate risk/reward ratio for the trade"""
        risk = abs(trade_setup.entry_price - trade_setup.stop_loss)
        reward = abs(trade_setup.take_profit_1 - trade_setup.entry_price)
        return reward / risk if risk > 0 else 0
    
    def _empty_position_result(self) -> Dict[str, Any]:
        """Return empty result when position cannot be calculated"""
        return {
            'base_calculation': {'units': 0, 'position_value': 0, 'position_percent': 0, 'risk_amount': 0},
            'adjustments': {'volatility_multiplier': 0, 'confidence_multiplier': 0, 'rr_multiplier': 0, 'final_multiplier': 0},
            'final_position': {'units': 0, 'position_value': 0, 'position_percent': 0, 'risk_amount': 0, 'risk_percent': 0},
            'expected_returns': {'tp1_return': 0, 'tp2_return': 0, 'max_loss': 0, 'expected_value': 0, 'rr_ratio': 0},
            'risk_metrics': {'volatility_percent': 0, 'risk_per_unit': 0, 'max_drawdown_percent': 0}
        }
    
    def calculate_portfolio_risk(self, positions: List[PortfolioPosition]) -> Dict[str, Any]:
        """Calculate overall portfolio risk metrics"""
        
        if not positions:
            return {
                'total_exposure': 0,
                'total_risk': 0,
                'portfolio_beta': 1.0,
                'diversification_ratio': 0,
                'max_drawdown': 0,
                'var_95': 0,  # Value at Risk 95%
                'position_count': 0,
                'risk_distribution': {}
            }
        
        total_exposure = sum(abs(pos.units * pos.current_price) for pos in positions)
        total_risk = sum(pos.risk_amount for pos in positions)
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
        
        # Calculate exposure by symbol
        exposure_by_symbol = {}
        risk_by_symbol = {}
        
        for pos in positions:
            symbol = pos.symbol
            exposure = abs(pos.units * pos.current_price)
            
            if symbol not in exposure_by_symbol:
                exposure_by_symbol[symbol] = 0
                risk_by_symbol[symbol] = 0
            
            exposure_by_symbol[symbol] += exposure
            risk_by_symbol[symbol] += pos.risk_amount
        
        # Calculate diversification metrics
        position_count = len(positions)
        avg_position_size = total_exposure / position_count if position_count > 0 else 0
        
        # Concentration risk (Herfindahl index)
        concentration_index = sum((exp / total_exposure) ** 2 for exp in exposure_by_symbol.values())
        diversification_ratio = (1 - concentration_index) if concentration_index < 1 else 0
        
        # Portfolio beta (simplified - assumes equal correlation to market)
        portfolio_beta = 1.0  # Would need market data for accurate calculation
        
        # Value at Risk (simplified Monte Carlo estimation)
        position_risks = [pos.risk_amount for pos in positions]
        if position_risks:
            var_95 = statistics.quantiles(position_risks, n=20)[18] if len(position_risks) > 1 else position_risks[0]
        else:
            var_95 = 0
        
        # Maximum drawdown estimate
        max_individual_risk = max(pos.risk_amount for pos in positions) if positions else 0
        portfolio_max_drawdown = min(total_risk, max_individual_risk * 2)  # Simplified calculation
        
        return {
            'total_exposure': round(total_exposure, 2),
            'total_exposure_percent': round((total_exposure / self.account_balance) * 100, 2),
            'total_risk': round(total_risk, 2),
            'total_risk_percent': round((total_risk / self.account_balance) * 100, 2),
            'unrealized_pnl': round(total_unrealized_pnl, 2),
            'portfolio_beta': round(portfolio_beta, 2),
            'diversification_ratio': round(diversification_ratio, 2),
            'concentration_index': round(concentration_index, 3),
            'max_drawdown': round(portfolio_max_drawdown, 2),
            'max_drawdown_percent': round((portfolio_max_drawdown / self.account_balance) * 100, 2),
            'var_95': round(var_95, 2),
            'position_count': position_count,
            'avg_position_size': round(avg_position_size, 2),
            'risk_distribution': {
                symbol: {
                    'exposure': round(exp, 2),
                    'exposure_percent': round((exp / total_exposure) * 100, 2),
                    'risk': round(risk_by_symbol[symbol], 2),
                    'risk_percent': round((risk_by_symbol[symbol] / total_risk) * 100, 2)
                }
                for symbol, exp in exposure_by_symbol.items()
            }
        }
    
    def optimize_position_sizes(self, trade_setups: List[TradeSetup],
                               max_total_risk: float = None) -> Dict[str, Any]:
        """
        Optimize position sizes across multiple potential trades
        Uses portfolio optimization techniques
        """
        
        if not trade_setups:
            return {'optimized_trades': [], 'total_risk': 0, 'expected_return': 0}
        
        if max_total_risk is None:
            max_total_risk = self.max_portfolio_risk
        
        # Calculate individual position sizes
        individual_positions = []
        total_individual_risk = 0
        
        for setup in trade_setups:
            pos_calc = self.calculate_position_size(setup)
            individual_positions.append({
                'setup': setup,
                'calculation': pos_calc,
                'risk': pos_calc['final_position']['risk_amount'],
                'expected_return': pos_calc['expected_returns']['expected_value'],
                'rr_ratio': pos_calc['expected_returns']['rr_ratio']
            })
            total_individual_risk += pos_calc['final_position']['risk_amount']
        
        # If total risk exceeds limit, scale down positions
        if total_individual_risk > (self.account_balance * max_total_risk / 100):
            scale_factor = (self.account_balance * max_total_risk / 100) / total_individual_risk
            
            for pos in individual_positions:
                # Scale down the position
                calc = pos['calculation']
                calc['final_position']['units'] *= scale_factor
                calc['final_position']['position_value'] *= scale_factor
                calc['final_position']['risk_amount'] *= scale_factor
                calc['expected_returns']['expected_value'] *= scale_factor
                pos['risk'] *= scale_factor
                pos['expected_return'] *= scale_factor
        
        # Sort by risk-adjusted return (Sharpe-like ratio)
        individual_positions.sort(
            key=lambda x: x['expected_return'] / max(x['risk'], 1),
            reverse=True
        )
        
        # Select top positions within risk limits
        selected_positions = []
        cumulative_risk = 0
        max_risk_budget = self.account_balance * max_total_risk / 100
        
        for pos in individual_positions:
            if (cumulative_risk + pos['risk'] <= max_risk_budget and 
                len(selected_positions) < self.max_positions and
                pos['rr_ratio'] >= self.min_risk_reward):
                
                selected_positions.append(pos)
                cumulative_risk += pos['risk']
        
        # Calculate portfolio metrics
        total_expected_return = sum(pos['expected_return'] for pos in selected_positions)
        total_risk = sum(pos['risk'] for pos in selected_positions)
        portfolio_sharpe = total_expected_return / max(total_risk, 1)
        
        return {
            'optimized_trades': [
                {
                    'symbol': pos['setup'].symbol,
                    'signal': pos['setup'].signal,
                    'confidence': pos['setup'].confidence,
                    'position_details': pos['calculation']['final_position'],
                    'expected_return': round(pos['expected_return'], 2),
                    'risk_amount': round(pos['risk'], 2),
                    'rr_ratio': round(pos['rr_ratio'], 2)
                }
                for pos in selected_positions
            ],
            'portfolio_metrics': {
                'total_positions': len(selected_positions),
                'total_risk': round(total_risk, 2),
                'total_risk_percent': round((total_risk / self.account_balance) * 100, 2),
                'total_expected_return': round(total_expected_return, 2),
                'portfolio_sharpe': round(portfolio_sharpe, 2),
                'risk_utilization': round((total_risk / max_risk_budget) * 100, 2)
            },
            'rejected_trades': len(trade_setups) - len(selected_positions),
            'available_risk_budget': round(max_risk_budget - total_risk, 2)
        }
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Kelly % = (Win Rate * Avg Win - Loss Rate * Avg Loss) / Avg Win
        """
        
        if avg_win <= 0:
            return 0
        
        loss_rate = 1 - win_rate
        kelly_percent = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        
        # Cap Kelly at 25% for safety (original Kelly can be very aggressive)
        return min(max(kelly_percent, 0), 0.25)
    
    def backtest_risk_metrics(self, historical_trades: List[Dict]) -> Dict[str, Any]:
        """
        Calculate risk metrics from historical trade data
        
        Expected format for historical_trades:
        [{'entry_price': float, 'exit_price': float, 'units': float, 'result': 'win'/'loss'}]
        """
        
        if not historical_trades:
            return self._empty_backtest_result()
        
        # Calculate basic metrics
        total_trades = len(historical_trades)
        winning_trades = [t for t in historical_trades if t.get('result') == 'win']
        losing_trades = [t for t in historical_trades if t.get('result') == 'loss']
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Calculate P&L
        pnl_list = []
        for trade in historical_trades:
            entry = trade['entry_price']
            exit_price = trade['exit_price']
            units = trade['units']
            pnl = (exit_price - entry) * units
            pnl_list.append(pnl)
        
        total_pnl = sum(pnl_list)
        avg_pnl = statistics.mean(pnl_list) if pnl_list else 0
        
        # Win/Loss metrics
        winning_amounts = [pnl for pnl in pnl_list if pnl > 0]
        losing_amounts = [abs(pnl) for pnl in pnl_list if pnl < 0]
        
        avg_win = statistics.mean(winning_amounts) if winning_amounts else 0
        avg_loss = statistics.mean(losing_amounts) if losing_amounts else 0
        largest_win = max(winning_amounts) if winning_amounts else 0
        largest_loss = max(losing_amounts) if losing_amounts else 0
        
        # Risk metrics
        profit_factor = sum(winning_amounts) / sum(losing_amounts) if losing_amounts else float('inf')
        
        # Drawdown calculation
        running_balance = self.account_balance
        peak_balance = running_balance
        max_drawdown = 0
        drawdown_periods = []
        
        for pnl in pnl_list:
            running_balance += pnl
            if running_balance > peak_balance:
                peak_balance = running_balance
            else:
                current_drawdown = (peak_balance - running_balance) / peak_balance
                max_drawdown = max(max_drawdown, current_drawdown)
        
        # Sharpe ratio (simplified)
        if pnl_list:
            returns_std = statistics.stdev(pnl_list) if len(pnl_list) > 1 else 0
            sharpe_ratio = avg_pnl / returns_std if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Kelly Criterion
        kelly_percent = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
        
        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': round(win_rate * 100, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'largest_win': round(largest_win, 2),
            'largest_loss': round(largest_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown_percent': round(max_drawdown * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'kelly_percent': round(kelly_percent * 100, 2),
            'expectancy': round(avg_pnl, 2),
            'risk_of_ruin': self._calculate_risk_of_ruin(win_rate, avg_win, avg_loss)
        }
    
    def _calculate_risk_of_ruin(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate risk of ruin percentage"""
        if avg_win <= 0 or avg_loss <= 0:
            return 100.0
        
        # Simplified risk of ruin calculation
        a = avg_loss / avg_win
        p = win_rate
        q = 1 - win_rate
        
        if p == q:  # Equal probability
            return 100.0
        
        try:
            if a == 1:
                risk_of_ruin = q / p
            else:
                risk_of_ruin = ((q/p) * a) ** (self.account_balance / avg_loss)
            
            return min(100.0, risk_of_ruin * 100)
        except:
            return 50.0  # Default estimate
    
    def _empty_backtest_result(self) -> Dict[str, Any]:
        """Return empty backtest result"""
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0,
            'total_pnl': 0, 'avg_win': 0, 'avg_loss': 0, 'largest_win': 0, 'largest_loss': 0,
            'profit_factor': 0, 'max_drawdown_percent': 0, 'sharpe_ratio': 0, 'kelly_percent': 0,
            'expectancy': 0, 'risk_of_ruin': 100
        }
    
    def generate_risk_report(self, trade_setup: TradeSetup, 
                           portfolio_positions: List[PortfolioPosition] = None) -> str:
        """Generate comprehensive risk management report"""
        
        # Calculate position sizing
        position_calc = self.calculate_position_size(trade_setup)
        
        # Calculate portfolio risk if positions provided
        portfolio_risk = {}
        if portfolio_positions:
            portfolio_risk = self.calculate_portfolio_risk(portfolio_positions)
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          RISK MANAGEMENT REPORT                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Symbol: {trade_setup.symbol:<10} │ Signal: {trade_setup.signal:<4} │ Confidence: {trade_setup.confidence}%       ║
║ Entry: ${trade_setup.entry_price:<10.2f} │ Stop: ${trade_setup.stop_loss:<10.2f} │ ATR: ${trade_setup.atr_value:<10.2f}   ║
║ TP1: ${trade_setup.take_profit_1:<12.2f} │ TP2: ${trade_setup.take_profit_2:<12.2f}                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                           POSITION SIZING                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Recommended Units: {position_calc['final_position']['units']:<15.8f}                      ║
║ Position Value: ${position_calc['final_position']['position_value']:<12.2f} ({position_calc['final_position']['position_percent']:<5.1f}% of account)    ║
║ Risk Amount: ${position_calc['final_position']['risk_amount']:<14.2f} ({position_calc['final_position']['risk_percent']:<5.1f}% of account)      ║
║ R/R Ratio: 1:{position_calc['expected_returns']['rr_ratio']:<13.1f}                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                         EXPECTED RETURNS                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ TP1 Profit: ${position_calc['expected_returns']['tp1_return']:<12.2f} │ TP2 Profit: ${position_calc['expected_returns']['tp2_return']:<12.2f}     ║
║ Max Loss: ${position_calc['expected_returns']['max_loss']:<14.2f} │ Expected Value: ${position_calc['expected_returns']['expected_value']:<10.2f}   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                          RISK ADJUSTMENTS                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Volatility Adj: {position_calc['adjustments']['volatility_multiplier']:<6.2f} │ Confidence Adj: {position_calc['adjustments']['confidence_multiplier']:<6.2f}        ║
║ R/R Adj: {position_calc['adjustments']['rr_multiplier']:<12.2f} │ Final Multiplier: {position_calc['adjustments']['final_multiplier']:<6.2f}        ║
║ Market Volatility: {position_calc['risk_metrics']['volatility_percent']:<6.1f}% │ Max Drawdown: {position_calc['risk_metrics']['max_drawdown_percent']:<6.1f}%          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        
        if portfolio_risk:
            report += f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PORTFOLIO RISK SUMMARY                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Total Positions: {portfolio_risk['position_count']:<8} │ Total Exposure: ${portfolio_risk['total_exposure']:<12.2f}      ║
║ Portfolio Risk: ${portfolio_risk['total_risk']:<10.2f} ({portfolio_risk['total_risk_percent']:<5.1f}% of account)           ║
║ Diversification: {portfolio_risk['diversification_ratio']:<6.1%} │ Max Drawdown: {portfolio_risk['max_drawdown_percent']:<6.1f}%          ║
║ Value at Risk (95%): ${portfolio_risk['var_95']:<10.2f}                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        
        return report


if __name__ == "__main__":
    # Example usage
    risk_manager = RiskManager(account_balance=10000)
    
    # Create sample trade setup
    trade_setup = TradeSetup(
        symbol="BTC/USDT",
        entry_price=50000,
        stop_loss=48000,
        take_profit_1=52000,
        take_profit_2=54000,
        signal="BUY",
        confidence=75,
        atr_value=800,
        timeframe="1h"
    )
    
    # Calculate position size
    position_calc = risk_manager.calculate_position_size(trade_setup)
    
    # Generate risk report
    report = risk_manager.generate_risk_report(trade_setup)
    print(report)
    
    print("\nPosition Calculation Details:")
    print(f"Recommended Units: {position_calc['final_position']['units']}")
    print(f"Risk Amount: ${position_calc['final_position']['risk_amount']}")
    print(f"Expected Return: ${position_calc['expected_returns']['expected_value']}")