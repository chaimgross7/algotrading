#!/usr/bin/env python3
"""Paper trading with a trained model."""

import argparse
import sys
import time
import signal
from pathlib import Path
from datetime import datetime, timedelta

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from algotrade.utils import setup_logging, get_device, load_config
from algotrade.data import fetch_data, DataCache
from algotrade.features import compute_features, normalize_features
from algotrade.models import LSTMModel, TransformerModel
from algotrade.execution import SimulatedBroker, RiskManager


class GracefulExit:
    def __init__(self):
        self.exit = False
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'exit', True))


def main():
    parser = argparse.ArgumentParser(description="Paper trade with AlgoTrade model")
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--config", "-c", default="config/experiments/mvp_spy_lstm.yaml")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--interval", type=int, default=60, help="Seconds between updates")
    parser.add_argument("--max-iters", type=int, default=None)
    args = parser.parse_args()
    
    setup_logging()
    device = get_device("auto")
    handler = GracefulExit()
    
    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    lookback = model_cfg.get("lookback", 60)
    
    # Initialize
    cache = DataCache()
    broker = SimulatedBroker(initial_capital=args.capital)
    risk = RiskManager()
    
    # Load initial data
    start = (datetime.now() - timedelta(days=lookback * 3)).strftime("%Y-%m-%d")
    df = fetch_data(args.symbol, start=start, cache=cache)
    features = compute_features(df)
    features, norm_params = normalize_features(features)
    input_dim = len(features.columns)
    
    # Load model
    if model_cfg.get("type", "lstm") == "lstm":
        model = LSTMModel(input_dim=input_dim, hidden_dim=model_cfg.get("hidden_dim", 128))
    else:
        model = TransformerModel(input_dim=input_dim, hidden_dim=model_cfg.get("hidden_dim", 128))
    
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"=== Paper Trading: {args.symbol} ===")
    print(f"Capital: ${args.capital:,.0f}")
    print(f"Model: {model.num_parameters():,} params")
    print("Press Ctrl+C to stop\n")
    
    iteration = 0
    while not handler.exit:
        if args.max_iters and iteration >= args.max_iters:
            break
        
        iteration += 1
        
        # Refresh data
        df = fetch_data(args.symbol, start=start, cache=None)  # No cache for fresh data
        features = compute_features(df)
        features, _ = normalize_features(features)
        
        if len(features) < lookback:
            print("Waiting for more data...")
            time.sleep(args.interval)
            continue
        
        # Get prediction
        seq = features.iloc[-lookback:].values
        X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = model(X)
            probs = torch.softmax(preds["direction"], dim=-1)[0]
            up_prob, down_prob = probs[2].item(), probs[0].item()
        
        price = df["close"].iloc[-1]
        broker.update_price(args.symbol, price)
        
        # Trading logic
        signal = "HOLD"
        if up_prob > 0.6:
            signal = "BUY"
        elif down_prob > 0.6:
            signal = "SELL"
        
        pos = broker.positions.get(args.symbol)
        
        if signal == "BUY" and (not pos or pos.quantity <= 0):
            qty = (broker.equity * 0.5) / price
            allowed, reason = risk.check_order(broker, args.symbol, qty, True)
            if allowed:
                if pos and pos.quantity < 0:
                    broker.close_position(args.symbol)
                broker.buy(args.symbol, qty)
        
        elif signal == "SELL" and (not pos or pos.quantity >= 0):
            if pos and pos.quantity > 0:
                broker.close_position(args.symbol)
        
        # Status
        pnl_pct = (broker.equity / args.capital - 1) * 100
        print(f"[{datetime.now():%H:%M:%S}] {args.symbol}: ${price:.2f} | "
              f"Signal: {signal} | Equity: ${broker.equity:,.0f} ({pnl_pct:+.1f}%)")
        
        if not handler.exit:
            time.sleep(args.interval)
    
    print(f"\n=== Session Ended ===")
    print(f"Final Equity: ${broker.equity:,.2f}")
    print(f"Total Return: {(broker.equity / args.capital - 1):.2%}")
    print(f"Trades: {len(broker.trades)}")


if __name__ == "__main__":
    main()
