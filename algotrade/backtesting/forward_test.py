"""Forward testing logger for out-of-sample performance tracking.

Logs predictions, trades, and session metrics to both JSON and CSV formats
for later analysis and model calibration.
"""

import json
import csv
import atexit
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("algotrade.forward_test")


@dataclass
class PredictionLog:
    """A single prediction record."""
    timestamp: str
    symbol: str
    price: float
    up_prob: float
    down_prob: float
    flat_prob: float
    predicted_class: int  # 0=down, 1=flat, 2=up
    confidence: float
    signal: str  # "BUY", "SELL", "HOLD"
    magnitude_pred: Optional[float] = None
    volatility_pred: Optional[float] = None


@dataclass
class TradeLog:
    """A single trade record."""
    timestamp: str
    symbol: str
    action: str  # "BUY", "SELL", "CLOSE"
    quantity: float
    price: float
    position_after: float
    equity_after: float
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class SessionSummary:
    """Summary of a forward test session."""
    session_id: str
    start_time: str
    end_time: str
    symbol: str
    initial_capital: float
    final_equity: float
    strategy_return: float
    benchmark_return: float
    alpha: float
    num_predictions: int
    num_trades: int
    # Prediction accuracy (if we can measure)
    correct_predictions: int = 0
    accuracy: float = 0.0


class ForwardTestLogger:
    """Logger for out-of-sample forward testing.
    
    Tracks predictions, trades, and session metrics with dual output
    to JSON (structured) and CSV (tabular) formats.
    
    Usage:
        logger = ForwardTestLogger(symbol="SPY", initial_capital=100000)
        
        # During trading loop:
        logger.log_prediction(
            timestamp=datetime.now(),
            symbol="SPY",
            price=450.50,
            probabilities={"up": 0.65, "down": 0.15, "flat": 0.20},
            signal="BUY",
            magnitude_pred=0.0023
        )
        
        logger.log_trade(
            timestamp=datetime.now(),
            symbol="SPY",
            action="BUY",
            quantity=100,
            price=450.50,
            position_after=100,
            equity_after=100500
        )
        
        # On exit (automatic via atexit, or manual):
        logger.save_session(final_equity=105000, benchmark_price=460.00)
    """
    
    def __init__(
        self,
        symbol: str,
        initial_capital: float,
        log_dir: str = "logs/forward_test",
        auto_save: bool = True,
    ):
        """Initialize the forward test logger.
        
        Args:
            symbol: Trading symbol
            initial_capital: Starting capital
            log_dir: Directory to save logs
            auto_save: If True, register atexit handler to save on exit
        """
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        self.start_price: Optional[float] = None
        
        self.predictions: List[PredictionLog] = []
        self.trades: List[TradeLog] = []
        
        self._saved = False
        
        if auto_save:
            atexit.register(self._atexit_save)
        
        logger.info(f"Forward test session started: {self.session_id}")
    
    def set_benchmark_start(self, price: float) -> None:
        """Set the starting price for benchmark comparison."""
        self.start_price = price
        logger.info(f"Benchmark start price: ${price:.2f}")
    
    def log_prediction(
        self,
        timestamp: datetime,
        symbol: str,
        price: float,
        probabilities: Dict[str, float],
        signal: str,
        magnitude_pred: Optional[float] = None,
        volatility_pred: Optional[float] = None,
    ) -> None:
        """Log a model prediction.
        
        Args:
            timestamp: Prediction timestamp
            symbol: Trading symbol
            price: Current price
            probabilities: Dict with keys 'up', 'down', 'flat'
            signal: Generated signal ("BUY", "SELL", "HOLD")
            magnitude_pred: Predicted return magnitude (optional)
            volatility_pred: Predicted volatility (optional)
        """
        up_prob = probabilities.get("up", 0.0)
        down_prob = probabilities.get("down", 0.0)
        flat_prob = probabilities.get("flat", 0.0)
        
        # Determine predicted class and confidence
        probs = [down_prob, flat_prob, up_prob]
        predicted_class = probs.index(max(probs))
        confidence = max(probs)
        
        pred = PredictionLog(
            timestamp=timestamp.isoformat(),
            symbol=symbol,
            price=price,
            up_prob=up_prob,
            down_prob=down_prob,
            flat_prob=flat_prob,
            predicted_class=predicted_class,
            confidence=confidence,
            signal=signal,
            magnitude_pred=magnitude_pred,
            volatility_pred=volatility_pred,
        )
        self.predictions.append(pred)
        
        # Set benchmark start on first prediction
        if self.start_price is None:
            self.set_benchmark_start(price)
    
    def log_trade(
        self,
        timestamp: datetime,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        position_after: float,
        equity_after: float,
        pnl: float = 0.0,
        pnl_pct: float = 0.0,
    ) -> None:
        """Log an executed trade.
        
        Args:
            timestamp: Trade timestamp
            symbol: Trading symbol
            action: "BUY", "SELL", or "CLOSE"
            quantity: Number of shares/contracts
            price: Execution price
            position_after: Position size after trade
            equity_after: Total equity after trade
            pnl: Realized P&L from this trade
            pnl_pct: Realized P&L percentage
        """
        trade = TradeLog(
            timestamp=timestamp.isoformat(),
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            position_after=position_after,
            equity_after=equity_after,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )
        self.trades.append(trade)
        logger.debug(f"Trade logged: {action} {quantity} {symbol} @ ${price:.2f}")
    
    def get_current_benchmark_return(self, current_price: float) -> float:
        """Calculate current benchmark (buy-and-hold) return."""
        if self.start_price is None or self.start_price == 0:
            return 0.0
        return (current_price / self.start_price) - 1
    
    def get_current_alpha(self, current_price: float, current_equity: float) -> float:
        """Calculate current alpha (strategy return - benchmark return)."""
        strategy_return = (current_equity / self.initial_capital) - 1
        benchmark_return = self.get_current_benchmark_return(current_price)
        return strategy_return - benchmark_return
    
    def save_session(
        self,
        final_equity: float,
        benchmark_price: Optional[float] = None,
    ) -> Dict[str, Path]:
        """Save session data to JSON and CSV files.
        
        Args:
            final_equity: Final portfolio equity
            benchmark_price: Final benchmark price (for return calculation)
            
        Returns:
            Dict with paths to saved files
        """
        if self._saved:
            logger.warning("Session already saved, skipping duplicate save")
            return {}
        
        self._saved = True
        end_time = datetime.now()
        
        # Calculate returns
        strategy_return = (final_equity / self.initial_capital) - 1
        
        if benchmark_price is not None and self.start_price is not None:
            benchmark_return = (benchmark_price / self.start_price) - 1
        else:
            benchmark_return = 0.0
        
        alpha = strategy_return - benchmark_return
        
        # Create summary
        summary = SessionSummary(
            session_id=self.session_id,
            start_time=self.start_time.isoformat(),
            end_time=end_time.isoformat(),
            symbol=self.symbol,
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            strategy_return=strategy_return,
            benchmark_return=benchmark_return,
            alpha=alpha,
            num_predictions=len(self.predictions),
            num_trades=len(self.trades),
        )
        
        saved_files = {}
        
        # --- Save JSON (full structured data) ---
        json_path = self.log_dir / f"session_{self.session_id}.json"
        json_data = {
            "summary": asdict(summary),
            "predictions": [asdict(p) for p in self.predictions],
            "trades": [asdict(t) for t in self.trades],
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        saved_files["json"] = json_path
        logger.info(f"Saved session JSON: {json_path}")
        
        # --- Save predictions CSV ---
        if self.predictions:
            pred_csv_path = self.log_dir / f"predictions_{self.session_id}.csv"
            with open(pred_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(self.predictions[0]).keys()))
                writer.writeheader()
                for pred in self.predictions:
                    writer.writerow(asdict(pred))
            saved_files["predictions_csv"] = pred_csv_path
            logger.info(f"Saved predictions CSV: {pred_csv_path}")
        
        # --- Save trades CSV ---
        if self.trades:
            trades_csv_path = self.log_dir / f"trades_{self.session_id}.csv"
            with open(trades_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(self.trades[0]).keys()))
                writer.writeheader()
                for trade in self.trades:
                    writer.writerow(asdict(trade))
            saved_files["trades_csv"] = trades_csv_path
            logger.info(f"Saved trades CSV: {trades_csv_path}")
        
        # --- Save summary CSV (append to aggregate file) ---
        summary_csv_path = self.log_dir / "sessions_summary.csv"
        file_exists = summary_csv_path.exists()
        with open(summary_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(summary).keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(asdict(summary))
        saved_files["summary_csv"] = summary_csv_path
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Forward Test Session: {self.session_id}")
        print(f"{'='*50}")
        print(f"Duration:          {self.start_time:%Y-%m-%d %H:%M} → {end_time:%H:%M:%S}")
        print(f"Symbol:            {self.symbol}")
        print(f"Predictions:       {len(self.predictions)}")
        print(f"Trades:            {len(self.trades)}")
        print(f"\nPerformance:")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Final Equity:    ${final_equity:,.2f}")
        print(f"  Strategy Return: {strategy_return:+.2%}")
        print(f"  Benchmark Return:{benchmark_return:+.2%}")
        print(f"  Alpha:           {alpha:+.2%}")
        print(f"\nLogs saved to: {self.log_dir}")
        print(f"{'='*50}\n")
        
        return saved_files
    
    def _atexit_save(self) -> None:
        """Atexit handler to save session if not already saved."""
        if not self._saved and (self.predictions or self.trades):
            # Try to get final equity from last trade, or use initial
            if self.trades:
                final_equity = self.trades[-1].equity_after
            else:
                final_equity = self.initial_capital
            
            # Try to get final price from last prediction
            if self.predictions:
                benchmark_price = self.predictions[-1].price
            else:
                benchmark_price = self.start_price
            
            logger.warning("Auto-saving session on exit")
            self.save_session(final_equity, benchmark_price)


def load_session(session_path: str) -> Dict[str, Any]:
    """Load a saved session from JSON file.
    
    Args:
        session_path: Path to session JSON file
        
    Returns:
        Dict with summary, predictions, and trades
    """
    with open(session_path, "r") as f:
        return json.load(f)


def aggregate_sessions(log_dir: str = "logs/forward_test") -> Dict[str, Any]:
    """Aggregate metrics across all saved sessions.
    
    Args:
        log_dir: Directory containing session files
        
    Returns:
        Dict with aggregated statistics
    """
    log_path = Path(log_dir)
    sessions = []
    
    for json_file in log_path.glob("session_*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            sessions.append(data["summary"])
    
    if not sessions:
        return {"error": "No sessions found"}
    
    import statistics
    
    returns = [s["strategy_return"] for s in sessions]
    alphas = [s["alpha"] for s in sessions]
    
    return {
        "num_sessions": len(sessions),
        "total_predictions": sum(s["num_predictions"] for s in sessions),
        "total_trades": sum(s["num_trades"] for s in sessions),
        "avg_return": statistics.mean(returns),
        "std_return": statistics.stdev(returns) if len(returns) > 1 else 0,
        "avg_alpha": statistics.mean(alphas),
        "std_alpha": statistics.stdev(alphas) if len(alphas) > 1 else 0,
        "best_session": max(sessions, key=lambda s: s["strategy_return"]),
        "worst_session": min(sessions, key=lambda s: s["strategy_return"]),
    }
