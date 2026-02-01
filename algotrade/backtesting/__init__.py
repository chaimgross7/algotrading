"""Backtesting module."""

from algotrade.backtesting.backtest import (
    Backtester,
    BacktestResult,
    Trade,
    calculate_metrics,
)
from algotrade.backtesting.forward_test import (
    ForwardTestLogger,
    PredictionLog,
    TradeLog,
    SessionSummary,
    load_session,
    aggregate_sessions,
)

__all__ = [
    "Backtester",
    "BacktestResult",
    "Trade",
    "calculate_metrics",
    "ForwardTestLogger",
    "PredictionLog",
    "TradeLog",
    "SessionSummary",
    "load_session",
    "aggregate_sessions",
]
