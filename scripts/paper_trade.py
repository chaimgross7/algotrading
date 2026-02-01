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
from algotrade.data.symbols import get_symbols
from algotrade.features import compute_features, normalize_features
from algotrade.models import load_model
from algotrade.execution import SimulatedBroker, RiskManager
from algotrade.backtesting import ForwardTestLogger


class GracefulExit:
    def __init__(self):
        self.exit = False
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'exit', True))


def main():
    parser = argparse.ArgumentParser(description="Paper trade with AlgoTrade model")
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--config", "-c", default="config/experiments/mvp_spy_lstm.yaml")
    parser.add_argument("--data-config", "-d", default="config/data/nasdaq100.yaml",
                        help="Path to data config file (default: NASDAQ 100)")
    parser.add_argument("--symbol", default=None, help="Symbol to trade (overrides data config)")
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--interval", type=int, default=60, help="Seconds between updates")
    parser.add_argument("--max-iters", type=int, default=None)
    args = parser.parse_args()
    
    setup_logging()
    device = get_device("auto")
    handler = GracefulExit()
    
    cfg = load_config(args.config, data_config=args.data_config)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    lookback = model_cfg.get("lookback", 60)
    
    # Resolve symbol: CLI arg > data config
    if args.symbol:
        symbol = args.symbol
    elif "symbols_preset" in data_cfg:
        symbols = get_symbols(data_cfg["symbols_preset"])
        symbol = symbols[0]  # Use first symbol for paper trading
    else:
        symbols = data_cfg.get("symbols", ["SPY"])
        symbol = symbols[0]
    
    # Initialize
    cache = DataCache()
    broker = SimulatedBroker(initial_capital=args.capital)
    risk = RiskManager()
    
    # Initialize forward test logger
    fwd_logger = ForwardTestLogger(
        symbol=symbol,
        initial_capital=args.capital,
        auto_save=True,
    )
    
    # Load initial data
    start = (datetime.now() - timedelta(days=lookback * 3)).strftime("%Y-%m-%d")
    df = fetch_data(symbol, start=start, cache=cache)
    features = compute_features(df)
    features, norm_params = normalize_features(features)
    input_dim = len(features.columns)
    
    # Load model using factory
    model = load_model(args.model, input_dim, model_cfg, device)
    
    print(f"=== Paper Trading: {symbol} ===")
    print(f"Capital: ${args.capital:,.0f}")
    print(f"Model: {model.num_parameters():,} params")
    print("Press Ctrl+C to stop\n")
    
    iteration = 0
    while not handler.exit:
        if args.max_iters and iteration >= args.max_iters:
            break
        
        iteration += 1
        
        # Refresh data
        df = fetch_data(symbol, start=start, cache=None)  # No cache for fresh data
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
        broker.update_price(symbol, price)
        
        # Log prediction with all probabilities
        fwd_logger.log_prediction(
            timestamp=datetime.now(),
            symbol=symbol,
            price=price,
            probabilities={"up": up_prob, "down": down_prob, "flat": probs[1].item()},
            signal="BUY" if up_prob > 0.6 else ("SELL" if down_prob > 0.6 else "HOLD"),
            magnitude_pred=preds.get("magnitude", [None])[0].item() if "magnitude" in preds else None,
        )
        
        # Trading logic
        signal = "HOLD"
        if up_prob > 0.6:
            signal = "BUY"
        elif down_prob > 0.6:
            signal = "SELL"
        
        pos = broker.positions.get(symbol)
        prev_equity = broker.equity
        
        if signal == "BUY" and (not pos or pos.quantity <= 0):
            qty = (broker.equity * 0.5) / price
            allowed, reason = risk.check_order(broker, symbol, qty, True)
            if allowed:
                if pos and pos.quantity < 0:
                    broker.close_position(symbol)
                broker.buy(symbol, qty)
                # Log the trade
                current_pos = broker.positions.get(symbol)
                fwd_logger.log_trade(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    action="BUY",
                    quantity=qty,
                    price=price,
                    position_after=current_pos.quantity if current_pos else 0,
                    equity_after=broker.equity,
                    pnl=broker.equity - prev_equity,
                )
        
        elif signal == "SELL" and (not pos or pos.quantity >= 0):
            if pos and pos.quantity > 0:
                pnl = broker.close_position(symbol)
                # Log the close trade
                fwd_logger.log_trade(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    action="CLOSE",
                    quantity=pos.quantity,
                    price=price,
                    position_after=0,
                    equity_after=broker.equity,
                    pnl=pnl if pnl else 0,
                )
        
        # Status with benchmark comparison
        pnl_pct = (broker.equity / args.capital - 1) * 100
        benchmark_ret = fwd_logger.get_current_benchmark_return(price) * 100
        alpha = fwd_logger.get_current_alpha(price, broker.equity) * 100
        print(f"[{datetime.now():%H:%M:%S}] {symbol}: ${price:.2f} | "
              f"Signal: {signal} | Equity: ${broker.equity:,.0f} ({pnl_pct:+.1f}%) | "
              f"Benchmark: {benchmark_ret:+.1f}% | Alpha: {alpha:+.1f}%")
        
        if not handler.exit:
            time.sleep(args.interval)
    
    # Save forward test session with final metrics
    final_price = df["close"].iloc[-1] if len(df) > 0 else fwd_logger.start_price
    fwd_logger.save_session(
        final_equity=broker.equity,
        benchmark_price=final_price,
    )
    
    print(f"\n=== Session Ended ===")
    print(f"Final Equity: ${broker.equity:,.2f}")
    print(f"Total Return: {(broker.equity / args.capital - 1):.2%}")
    print(f"Trades: {len(broker.trades)}")


if __name__ == "__main__":
    main()
