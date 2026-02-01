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
    # Benchmark (buy-and-hold) fields
    benchmark_equity: pd.Series = field(default_factory=pd.Series)
    benchmark_return: float = 0.0
    benchmark_sharpe: float = 0.0
    benchmark_max_drawdown: float = 0.0
    # Relative performance fields
    alpha: float = 0.0  # Strategy return - benchmark return
    information_ratio: float = 0.0  # Excess return / tracking error
    
    def summary(self) -> str:
        summary_str = f"""
Backtest Results
----------------
Strategy:
  Return:      {self.total_return:>8.2%}
  Sharpe:      {self.sharpe_ratio:>8.2f}
  Max DD:      {self.max_drawdown:>8.2%}
  Win Rate:    {self.win_rate:>8.2%}
  Trades:      {self.num_trades:>8}

Benchmark (Buy & Hold):
  Return:      {self.benchmark_return:>8.2%}
  Sharpe:      {self.benchmark_sharpe:>8.2f}
  Max DD:      {self.benchmark_max_drawdown:>8.2%}

Relative Performance:
  Alpha:       {self.alpha:>8.2%}
  Info Ratio:  {self.information_ratio:>8.2f}
""".strip()
        return summary_str


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
        
        # --- Buy-and-hold benchmark ---
        benchmark_returns = returns  # Fully invested, no trading
        benchmark_equity = self.initial_capital * (1 + benchmark_returns).cumprod()
        benchmark_return = benchmark_equity.iloc[-1] / self.initial_capital - 1
        benchmark_sharpe = benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252) if benchmark_returns.std() > 0 else 0
        
        benchmark_rolling_max = benchmark_equity.cummax()
        benchmark_drawdown = (benchmark_equity - benchmark_rolling_max) / benchmark_rolling_max
        benchmark_max_drawdown = abs(benchmark_drawdown.min())
        
        # --- Relative performance metrics ---
        alpha = total_return - benchmark_return
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252) if len(excess_returns) > 1 else 0
        information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=abs(max_drawdown),
            win_rate=win_rate,
            num_trades=len(trades),
            trades=trades,
            equity_curve=equity,
            # Benchmark fields
            benchmark_equity=benchmark_equity,
            benchmark_return=benchmark_return,
            benchmark_sharpe=benchmark_sharpe,
            benchmark_max_drawdown=benchmark_max_drawdown,
            # Relative performance
            alpha=alpha,
            information_ratio=information_ratio,
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
        
        logger.info(f"Signals: {(discrete == 1).sum()} long, {(discrete == -1).sum()} short, {(discrete == 0).sum()} flat")
        
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
    
    def _tune_parameter(
        self,
        run_fn,
        param_values: List[float],
        metric: str,
        param_name: str,
    ) -> tuple:
        """Generic parameter tuning by grid search.
        
        Args:
            run_fn: Callable that takes a parameter value and returns BacktestResult
            param_values: List of parameter values to try
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'win_rate')
            param_name: Name of parameter for logging
            
        Returns:
            Tuple of (best_value, best_result, all_results)
        """
        best_value = param_values[0]
        best_result = None
        best_score = float('-inf')
        all_results = {}
        
        logger.info(f"Tuning {param_name} on {len(param_values)} values, optimizing {metric}...")
        
        for value in param_values:
            result = run_fn(value)
            score = getattr(result, metric, 0.0)
            all_results[value] = {
                'result': result,
                'score': score,
                'sharpe': result.sharpe_ratio,
                'return': result.total_return,
                'trades': result.num_trades,
            }
            
            logger.info(f"  {param_name}={value}: {metric}={score:.4f}, trades={result.num_trades}")
            
            if score > best_score:
                best_score = score
                best_value = value
                best_result = result
        
        logger.info(f"Best {param_name}: {best_value} with {metric}={best_score:.4f}")
        
        return best_value, best_result, all_results

    def tune_threshold(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        prices: np.ndarray,
        dates: pd.DatetimeIndex,
        thresholds: List[float] = None,
        metric: str = "sharpe_ratio",
        device: torch.device = None,
    ) -> tuple:
        """Find optimal confidence threshold by grid search."""
        if thresholds is None:
            thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        
        def run_with_threshold(thresh):
            return self.run_from_model(model, X, prices, dates, device=device, threshold=thresh)
        
        return self._tune_parameter(run_with_threshold, thresholds, metric, "threshold")
    
    def run_from_magnitude(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        prices: np.ndarray,
        dates: pd.DatetimeIndex,
        device: torch.device = None,
        long_threshold: float = 0.002,
        short_threshold: float = -0.002,
        label_stats: dict = None,
    ) -> BacktestResult:
        """Run backtest using magnitude (return) predictions instead of direction classification.
        
        Args:
            model: Trained model
            X: Input features tensor
            prices: Price array
            dates: DatetimeIndex for prices
            device: Torch device
            long_threshold: Go long if predicted return > this (e.g., 0.002 = 0.2%)
            short_threshold: Go short if predicted return < this (e.g., -0.002 = -0.2%)
            label_stats: Dict with 'magnitude_mean' and 'magnitude_std' for denormalization
            
        Returns:
            BacktestResult
        """
        device = device or torch.device("cpu")
        model.eval()
        
        with torch.no_grad():
            X = X.to(device)
            preds = model(X)
            # Use magnitude prediction (predicted return)
            pred_returns = preds["magnitude"].squeeze(-1).cpu().numpy()
        
        # Denormalize if label_stats provided
        if label_stats is not None:
            pred_returns = pred_returns * label_stats["magnitude_std"] + label_stats["magnitude_mean"]
        
        # Convert to discrete signals based on predicted return
        discrete = np.zeros(len(pred_returns))
        discrete[pred_returns > long_threshold] = 1   # Long
        discrete[pred_returns < short_threshold] = -1  # Short
        
        logger.info(f"Magnitude signals: {(discrete == 1).sum()} long, {(discrete == -1).sum()} short, {(discrete == 0).sum()} flat")
        logger.info(f"Predicted returns - mean: {pred_returns.mean():.4f}, std: {pred_returns.std():.4f}, min: {pred_returns.min():.4f}, max: {pred_returns.max():.4f}")
        
        price_series = pd.Series(prices, index=dates)
        signal_series = pd.Series(discrete, index=dates)
        
        return self.run(price_series, signal_series)
    
    def tune_magnitude_threshold(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        prices: np.ndarray,
        dates: pd.DatetimeIndex,
        thresholds: List[float] = None,
        metric: str = "sharpe_ratio",
        device: torch.device = None,
        label_stats: dict = None,
    ) -> tuple:
        """Find optimal magnitude thresholds by grid search."""
        if thresholds is None:
            thresholds = [0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.01]
        
        def run_with_threshold(thresh):
            return self.run_from_magnitude(
                model, X, prices, dates,
                device=device,
                long_threshold=thresh,
                short_threshold=-thresh,
                label_stats=label_stats,
            )
        
        return self._tune_parameter(run_with_threshold, thresholds, metric, "magnitude_threshold")


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
