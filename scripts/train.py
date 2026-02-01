#!/usr/bin/env python3
"""Train a prediction model."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from algotrade.utils import setup_logging, get_device, load_config
from algotrade.data import fetch_data, DataCache
from algotrade.data.symbols import get_symbols
from algotrade.features import compute_features, create_labels, prepare_sequences, normalize_features
from algotrade.models import create_model
from algotrade.training import Trainer, TrainerConfig


def load_symbol_data(
    symbols: List[str],
    data_cfg: Dict,
    features_cfg: Dict,
    lookback: int,
) -> Tuple[List[np.ndarray], List[Dict], List[str]]:
    """Load and process data for all symbols.
    
    Returns:
        Tuple of (X_list, y_list, skipped_symbols)
    """
    cache = DataCache()
    interval = data_cfg.get("interval", "1d")
    
    all_X, all_y = [], []
    skipped_symbols = []
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        try:
            df = fetch_data(
                symbols=symbol,
                start=data_cfg.get("start_date", "2020-01-01"),
                end=data_cfg.get("end_date"),
                interval=interval,
                cache=cache,
            )
            if df is None or len(df) == 0:
                print(f"  WARNING: No data for {symbol}, skipping...")
                skipped_symbols.append(symbol)
                continue
            print(f"  Raw data: {len(df)} rows")
            
            # Compute features and labels
            features = compute_features(df, features_cfg)
            features, _ = normalize_features(features)
            labels = create_labels(df, threshold=0.002)
            
            # Prepare sequences
            X, y, dates = prepare_sequences(features, labels, lookback=lookback)
            print(f"  Sequences: {X.shape[0]}")
            
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"  ERROR: Failed to process {symbol}: {e}")
            skipped_symbols.append(symbol)
            continue
    
    return all_X, all_y, skipped_symbols


def prepare_datasets(
    all_X: List[np.ndarray],
    all_y: List[Dict],
    normalize_labels: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict, Dict]:
    """Combine symbol data and split into train/val sets.
    
    Returns:
        Tuple of (X_train, X_val, y_train, y_val, label_stats)
    """
    # Combine all symbols
    X = np.concatenate(all_X, axis=0)
    y = {
        "direction": np.concatenate([d["direction"] for d in all_y], axis=0),
        "magnitude": np.concatenate([d["magnitude"] for d in all_y], axis=0),
        "volatility": np.concatenate([d["volatility"] for d in all_y], axis=0),
    }
    
    # Normalize magnitude labels if requested
    label_stats = {"magnitude_mean": 0.0, "magnitude_std": 1.0}
    if normalize_labels:
        label_stats["magnitude_mean"] = float(y["magnitude"].mean())
        label_stats["magnitude_std"] = float(y["magnitude"].std())
        y["magnitude"] = (y["magnitude"] - label_stats["magnitude_mean"]) / (label_stats["magnitude_std"] + 1e-8)
        print(f"Normalized magnitude labels: mean={label_stats['magnitude_mean']:.6f}, std={label_stats['magnitude_std']:.6f}")
    
    # Shuffle before split
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = {k: v[indices] for k, v in y.items()}
    
    # Train/val split
    split = int(len(X) * 0.8)
    X_train = torch.tensor(X[:split])
    X_val = torch.tensor(X[split:])
    y_train = {k: torch.tensor(v[:split]) for k, v in y.items()}
    y_val = {k: torch.tensor(v[split:]) for k, v in y.items()}
    
    return X_train, X_val, y_train, y_val, label_stats


def main():
    parser = argparse.ArgumentParser(description="Train AlgoTrade model")
    parser.add_argument("--config", "-c", default="config/experiments/mvp_spy_lstm.yaml")
    parser.add_argument("--data-config", "-d", default="config/data/nasdaq100.yaml",
                        help="Path to data config file (default: NASDAQ 100)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate data without training")
    args = parser.parse_args()
    
    setup_logging()
    device = get_device(args.device)
    
    # Load config
    cfg = load_config(args.config, data_config=args.data_config)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    
    print(f"Config: {args.config}")
    print(f"Data config: {args.data_config}")
    print(f"Device: {device}")
    
    # Resolve symbols
    if "symbols_preset" in data_cfg:
        symbols = get_symbols(data_cfg["symbols_preset"])
        print(f"Using symbol preset '{data_cfg['symbols_preset']}': {len(symbols)} symbols")
    else:
        symbols = data_cfg.get("symbols", ["SPY"])
    
    lookback = model_cfg.get("lookback", 60)
    normalize_labels = train_cfg.get("normalize_labels", False)
    
    # Load data
    all_X, all_y, skipped = load_symbol_data(
        symbols, data_cfg, cfg.get("features", {}), lookback
    )
    
    if not all_X:
        raise ValueError("No valid symbols loaded. Check your symbol list.")
    
    if skipped:
        print(f"\nWARNING: Skipped {len(skipped)} symbols: {skipped}")
    
    loaded = len(symbols) - len(skipped)
    print(f"\nTotal data: {sum(x.shape[0] for x in all_X)} sequences from {loaded}/{len(symbols)} symbols")
    print(f"Features: {all_X[0].shape[-1]}")
    
    if args.dry_run:
        print("Dry run complete - data is valid")
        return
    
    # Prepare datasets
    X_train, X_val, y_train, y_val, label_stats = prepare_datasets(
        all_X, all_y, normalize_labels
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Create model using factory
    input_dim = all_X[0].shape[-1]
    model = create_model(input_dim, model_cfg)
    print(f"Model: {model_cfg.get('type', 'lstm')}, {model.num_parameters():,} parameters")
    
    # Create trainer config
    trainer_cfg = TrainerConfig.from_config(train_cfg)
    
    # Set class weights if direction is enabled
    if trainer_cfg.direction_weight > 0:
        trainer_cfg.class_weights = [4.0, 1.0, 4.0]
        print(f"Class weights: {trainer_cfg.class_weights}")
    else:
        trainer_cfg.label_smoothing = 0.0
        print("Direction head disabled (weight=0), focusing on magnitude regression")
    
    print(f"Loss weights - direction: {trainer_cfg.direction_weight}, "
          f"magnitude: {trainer_cfg.magnitude_weight}, volatility: {trainer_cfg.volatility_weight}")
    
    trainer = Trainer(model=model, device=device, config=trainer_cfg)
    
    print(f"Using uncertainty loss: {trainer_cfg.use_uncertainty_loss}, "
          f"warmup: {trainer_cfg.warmup_epochs} epochs, label_smoothing: {trainer_cfg.label_smoothing}")
    
    # Train
    epochs = args.epochs or train_cfg.get("epochs", 100)
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=train_cfg.get("batch_size", 32),
        patience=train_cfg.get("patience", 20),
    )
    
    print(f"\nTraining complete!")
    print(f"Best val loss: {min(history['val_loss']):.4f}")
    if trainer_cfg.direction_weight > 0:
        print(f"Best val acc: {max(history['val_acc']):.2%}")
    
    # Save label stats for inference
    if normalize_labels:
        stats_path = Path(trainer_cfg.checkpoint_dir) / "label_stats.json"
        with open(stats_path, "w") as f:
            json.dump(label_stats, f)
        print(f"Saved label stats to {stats_path}")


if __name__ == "__main__":
    main()
