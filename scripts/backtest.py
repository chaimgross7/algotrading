#!/usr/bin/env python3
"""Backtest a trained model."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from algotrade.utils import setup_logging, get_device, load_config
from algotrade.data import fetch_data, DataCache
from algotrade.features import compute_features, create_labels, prepare_sequences, normalize_features
from algotrade.models import LSTMModel, TransformerModel
from algotrade.backtesting import Backtester, calculate_metrics


def main():
    parser = argparse.ArgumentParser(description="Backtest AlgoTrade model")
    parser.add_argument("--model", "-m", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", "-c", default="config/experiments/mvp_spy_lstm.yaml")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--capital", type=float, default=100000)
    args = parser.parse_args()
    
    setup_logging()
    device = get_device("auto")
    
    # Load config
    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    
    print(f"Model: {args.model}")
    print(f"Symbol: {args.symbol}")
    
    # Fetch data
    cache = DataCache()
    interval = data_cfg.get("interval", "1d")
    df = fetch_data(args.symbol, start=args.start, end=args.end, interval=interval, cache=cache)
    print(f"Data: {len(df)} rows ({interval})")
    
    # Prepare features
    features = compute_features(df, cfg.get("features", {}))
    features, _ = normalize_features(features)
    labels = create_labels(df)
    
    lookback = model_cfg.get("lookback", 60)
    X, y, dates = prepare_sequences(features, labels, lookback=lookback)
    
    # Load model
    input_dim = X.shape[-1]
    model_type = model_cfg.get("type", "lstm")
    
    if model_type == "lstm":
        model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 2),
        )
    else:
        transformer_cfg = model_cfg.get("transformer", {})
        model = TransformerModel(
            input_dim=input_dim,
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_heads=transformer_cfg.get("num_heads", 8),
            num_layers=transformer_cfg.get("num_layers", 4),
            ff_dim=transformer_cfg.get("ff_dim", 512),
            dropout=model_cfg.get("dropout", 0.2),
        )
    
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"Loaded model: {model.num_parameters():,} parameters")
    
    # Get prices aligned with sequences - dates from prepare_sequences is already aligned
    prices = df.loc[dates, "close"].values
    
    # Run backtest
    backtester = Backtester(initial_capital=args.capital)
    result = backtester.run_from_model(
        model=model,
        X=torch.tensor(X, dtype=torch.float32),
        prices=prices,
        dates=dates,
        device=device,
    )
    
    print("\n" + result.summary())
    
    # Additional metrics
    if len(result.equity_curve) > 0:
        metrics = calculate_metrics(result.equity_curve)
        print(f"\nAnnualized Return: {metrics['annualized_return']:.2%}")
        print(f"Volatility: {metrics['volatility']:.2%}")
        print(f"Sortino: {metrics['sortino_ratio']:.2f}")
        print(f"Calmar: {metrics['calmar_ratio']:.2f}")


if __name__ == "__main__":
    main()
