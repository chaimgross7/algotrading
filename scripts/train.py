#!/usr/bin/env python3
"""Train a prediction model."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from algotrade.utils import setup_logging, get_device, load_config
from algotrade.data import fetch_data, DataCache
from algotrade.features import compute_features, create_labels, prepare_sequences, normalize_features
from algotrade.models import LSTMModel, TransformerModel
from algotrade.training import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train AlgoTrade model")
    parser.add_argument("--config", "-c", default="config/experiments/mvp_spy_lstm.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate data without training")
    args = parser.parse_args()
    
    setup_logging()
    device = get_device(args.device)
    
    # Load config
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    
    # Fetch data
    cache = DataCache()
    symbols = data_cfg.get("symbols", ["SPY"])
    interval = data_cfg.get("interval", "1d")
    df = fetch_data(
        symbols=symbols[0] if len(symbols) == 1 else symbols,
        start=data_cfg.get("start_date", "2020-01-01"),
        end=data_cfg.get("end_date"),
        interval=interval,
        cache=cache,
    )
    print(f"Data: {len(df)} rows")
    
    # Compute features and labels
    features = compute_features(df, cfg.get("features", {}))
    features, _ = normalize_features(features)
    labels = create_labels(df, threshold=0.002)  # 0.2% threshold for up/down
    
    # Prepare sequences
    lookback = model_cfg.get("lookback", 60)
    X, y, dates = prepare_sequences(features, labels, lookback=lookback)
    print(f"Sequences: {X.shape}")
    
    if args.dry_run:
        print("Dry run complete - data is valid")
        return
    
    # Train/val split
    split = int(len(X) * 0.8)
    X_train, X_val = torch.tensor(X[:split]), torch.tensor(X[split:])
    y_train = {k: torch.tensor(v[:split]) for k, v in y.items()}
    y_val = {k: torch.tensor(v[split:]) for k, v in y.items()}
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Create model
    input_dim = X.shape[-1]
    model_type = model_cfg.get("type", "lstm")
    
    if model_type == "lstm":
        model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.2),
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
    
    print(f"Model: {model_type}, {model.num_parameters():,} parameters")
    
    # Compute class weights based on distribution (inverse frequency)
    # down=16%, flat=67%, up=17% -> weights ~[4.0, 1.0, 4.0]
    class_weights = [4.0, 1.0, 4.0]  # Penalize errors on up/down more
    print(f"Class weights: {class_weights}")
    
    # Train
    trainer = Trainer(
        model=model,
        device=device,
        lr=train_cfg.get("learning_rate", 1e-3),
        checkpoint_dir=train_cfg.get("checkpoint_dir", "checkpoints"),
        class_weights=class_weights,
    )
    
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
    print(f"Best val acc: {max(history['val_acc']):.2%}")


if __name__ == "__main__":
    main()
