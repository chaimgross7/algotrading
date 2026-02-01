"""Backtesting engine."""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch
import logging

logger = logging.getLogger("algotrade.backtesting")


@dataclass
class Trade:
    """A single trade."""
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp = None
    exit_price: float = None
    direction: int = 1  # 1=long, -1=short
    size: float = 1.0
    
    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return self.direction * self.size * (self.exit_price - self.entry_price)
    
    @property
    def return_pct(self) -> float:
        if self.exit_price is None:
            return 0.0
        return self.direction * (self.exit_price / self.entry_price - 1)


@dataclass
class BacktestResult:
    """Results from a backtest."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    
    def summary(self) -> str:
        return f"""
Backtest Results
----------------
Return:      {self.total_return:>8.2%}
Sharpe:      {self.sharpe_ratio:>8.2f}
Max DD:      {self.max_drawdown:>8.2%}
Win Rate:    {self.win_rate:>8.2%}
Trades:      {self.num_trades:>8}
""".strip()


class Backtester:
    """Simple vectorized backtester."""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.001,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
    
    def run(
        self,
        prices: pd.Series,
        signals: pd.Series,
        position_size: float = 1.0,
    ) -> BacktestResult:
        """
        Run backtest.
        
        Args:
            prices: Close prices indexed by date
            signals: Signal series: 1=long, -1=short, 0=flat
            position_size: Fraction of capital to use (0-1)
        """
        # Align data
        common = prices.index.intersection(signals.index)
        prices = prices.loc[common]
        signals = signals.loc[common]
        
        # Calculate returns
        returns = prices.pct_change().fillna(0)
        
        # Position from signals (shifted to avoid lookahead)
        position = signals.shift(1).fillna(0)
        
        # Strategy returns with costs
        costs = position.diff().abs() * (self.commission_pct + self.slippage_pct)
        strategy_returns = position * returns * position_size - costs
        
        # Equity curve
        equity = self.initial_capital * (1 + strategy_returns).cumprod()
        
        # Metrics
        total_return = equity.iloc[-1] / self.initial_capital - 1
        
        daily_rets = strategy_returns
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
        
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Extract trades
        trades = self._extract_trades(prices, signals)
        winning = sum(1 for t in trades if t.pnl > 0)
        win_rate = winning / len(trades) if trades else 0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=abs(max_drawdown),
            win_rate=win_rate,
            num_trades=len(trades),
            trades=trades,
            equity_curve=equity,
        )
    
    def run_from_model(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        prices: np.ndarray,
        dates: pd.DatetimeIndex,
        device: torch.device = None,
        threshold: float = 0.4,
    ) -> BacktestResult:
        """Run backtest using model predictions."""
        device = device or torch.device("cpu")
        model.eval()
        
        with torch.no_grad():
            X = X.to(device)
            preds = model(X)
            direction_probs = torch.softmax(preds["direction"], dim=-1)
            # Get predicted class: 0=down, 1=flat, 2=up
            pred_class = direction_probs.argmax(dim=-1).cpu().numpy()
            max_prob = direction_probs.max(dim=-1).values.cpu().numpy()
        
        # Convert to discrete signals: only trade if confident enough
        discrete = np.zeros(len(pred_class))
        confident_up = (pred_class == 2) & (max_prob > threshold)
        confident_down = (pred_class == 0) & (max_prob > threshold)
        discrete[confident_up] = 1
        discrete[confident_down] = -1
        
        print(f"Signals: {(discrete == 1).sum()} long, {(discrete == -1).sum()} short, {(discrete == 0).sum()} flat")
        
        price_series = pd.Series(prices, index=dates)
        signal_series = pd.Series(discrete, index=dates)
        
        return self.run(price_series, signal_series)
    
    def _extract_trades(self, prices: pd.Series, signals: pd.Series) -> List[Trade]:
        """Extract individual trades from signals."""
        trades = []
        current_trade = None
        
        for i, (date, signal) in enumerate(signals.items()):
            price = prices.loc[date]
            
            if current_trade is None and signal != 0:
                # Open trade
                current_trade = Trade(
                    entry_date=date,
                    entry_price=price,
                    direction=int(signal),
                )
            elif current_trade is not None:
                if signal == 0 or signal != current_trade.direction:
                    # Close trade
                    current_trade.exit_date = date
                    current_trade.exit_price = price
                    trades.append(current_trade)
                    
                    if signal != 0:
                        # Open new trade
                        current_trade = Trade(
                            entry_date=date,
                            entry_price=price,
                            direction=int(signal),
                        )
                    else:
                        current_trade = None
        
        return trades


def calculate_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.04) -> Dict[str, float]:
    """Calculate performance metrics from equity curve."""
    returns = equity_curve.pct_change().dropna()
    
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    ann_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
    
    vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / vol if vol > 0 else 0
    
    # Sortino (downside deviation)
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (ann_return - risk_free_rate) / downside if downside > 0 else 0
    
    # Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = abs(drawdown.min())
    
    # Calmar
    calmar = ann_return / max_dd if max_dd > 0 else 0
    
    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "volatility": vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
    }
