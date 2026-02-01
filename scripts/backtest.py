#!/usr/bin/env python3
"""Backtest a trained model."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from algotrade.utils import setup_logging, get_device, load_config
from algotrade.data import fetch_data, DataCache
from algotrade.data.symbols import get_symbols
from algotrade.features import compute_features, create_labels, prepare_sequences, normalize_features
from algotrade.models import load_model
from algotrade.backtesting import Backtester, calculate_metrics


def main():
    parser = argparse.ArgumentParser(description="Backtest AlgoTrade model")
    parser.add_argument("--model", "-m", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", "-c", default="config/experiments/mvp_spy_lstm.yaml")
    parser.add_argument("--data-config", "-d", default="config/data/nasdaq100.yaml",
                        help="Path to data config file (default: NASDAQ 100)")
    parser.add_argument("--symbol", default=None, help="Symbol to backtest (overrides data config)")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold (auto-tune if not set)")
    parser.add_argument("--tune-threshold", action="store_true", help="Grid search for optimal threshold")
    parser.add_argument("--mode", choices=["direction", "magnitude"], default="direction",
                        help="Signal mode: direction classification or magnitude regression")
    parser.add_argument("--long-threshold", type=float, default=0.002,
                        help="Long threshold for magnitude mode (e.g., 0.002 = 0.2%%)")
    parser.add_argument("--short-threshold", type=float, default=-0.002,
                        help="Short threshold for magnitude mode")
    args = parser.parse_args()
    
    setup_logging()
    device = get_device("auto")
    
    # Load config
    cfg = load_config(args.config, data_config=args.data_config)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    
    # Resolve symbol
    if args.symbol:
        symbol = args.symbol
    elif "symbols_preset" in data_cfg:
        symbols = get_symbols(data_cfg["symbols_preset"])
        symbol = symbols[0]
        print(f"Using first symbol from preset '{data_cfg['symbols_preset']}': {symbol}")
    else:
        symbols = data_cfg.get("symbols", ["SPY"])
        symbol = symbols[0]
    
    print(f"Model: {args.model}")
    print(f"Symbol: {symbol}")
    
    # Fetch data
    cache = DataCache()
    interval = data_cfg.get("interval", "1d")
    df = fetch_data(symbol, start=args.start, end=args.end, interval=interval, cache=cache)
    print(f"Data: {len(df)} rows ({interval})")
    
    # Prepare features
    features = compute_features(df, cfg.get("features", {}))
    features, _ = normalize_features(features)
    labels = create_labels(df)
    
    lookback = model_cfg.get("lookback", 60)
    X, y, dates = prepare_sequences(features, labels, lookback=lookback)
    
    # Load model using factory
    input_dim = X.shape[-1]
    model = load_model(args.model, input_dim, model_cfg, device)
    print(f"Loaded model: {model.num_parameters():,} parameters")
    
    # Load label stats for denormalization
    label_stats_path = Path(args.model).parent / "label_stats.json"
    if label_stats_path.exists():
        with open(label_stats_path) as f:
            label_stats = json.load(f)
        print(f"Label stats: mean={label_stats['magnitude_mean']:.6f}, std={label_stats['magnitude_std']:.6f}")
    else:
        label_stats = None
        print("No label_stats.json found - using raw predictions")
    
    # Get prices aligned with sequences
    prices = df.loc[dates, "close"].values
    
    # Run backtest
    backtester = Backtester(initial_capital=args.capital)
    
    if args.mode == "magnitude":
        print(f"\nUsing magnitude mode (long > {args.long_threshold:.3%}, short < {args.short_threshold:.3%})")
        
        if args.tune_threshold:
            print("Tuning magnitude threshold...")
            best_thresh, result, all_results = backtester.tune_magnitude_threshold(
                model=model,
                X=torch.tensor(X, dtype=torch.float32),
                prices=prices,
                dates=dates,
                device=device,
                metric="sharpe_ratio",
                label_stats=label_stats,
            )
            print(f"\nOptimal threshold: ±{best_thresh:.4f}")
        else:
            result = backtester.run_from_magnitude(
                model=model,
                X=torch.tensor(X, dtype=torch.float32),
                prices=prices,
                dates=dates,
                device=device,
                long_threshold=args.long_threshold,
                short_threshold=args.short_threshold,
                label_stats=label_stats,
            )
    else:
        if args.tune_threshold:
            print("\nTuning confidence threshold...")
            best_thresh, result, all_results = backtester.tune_threshold(
                model=model,
                X=torch.tensor(X, dtype=torch.float32),
                prices=prices,
                dates=dates,
                device=device,
                metric="sharpe_ratio",
            )
            print(f"\nOptimal threshold: {best_thresh:.2f}")
        else:
            threshold = args.threshold if args.threshold is not None else 0.4
            result = backtester.run_from_model(
                model=model,
                X=torch.tensor(X, dtype=torch.float32),
                prices=prices,
                dates=dates,
                device=device,
                threshold=threshold,
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
