# AlgoTrade

Neural network-based algorithmic trading system for US stocks. Predicts price direction, magnitude, and volatility using LSTM and Transformer models trained on technical indicators.

---

## Project Overview

**Goal**: Build a multi-task prediction system that forecasts:
1. **Direction** - Up/Flat/Down classification (3 classes)
2. **Magnitude** - Expected return value (regression)
3. **Volatility** - Future price volatility (regression)

**Current Phase**: MVP with hourly SPY prediction using Transformer model.

**Tech Stack**: Python 3.12+, PyTorch 2.0+, yfinance, pandas-ta, YAML configs, Parquet caching.

---

## Quick Start

```bash
# Install with uv
uv sync

# Or pip
pip install -e .

# Train a model
python scripts/train.py --config config/experiments/mvp_spy_lstm.yaml

# Backtest trained model
python scripts/backtest.py --model checkpoints/best.pt --symbol SPY

# Paper trade (simulated live)
python scripts/paper_trade.py --model checkpoints/best.pt --symbol SPY
```

---

## Directory Structure

```
algotrade/
├── data/
│   └── loader.py           # DataCache + fetch_data()
├── features/
│   └── pipeline.py         # compute_features(), create_labels(), prepare_sequences()
├── models/
│   ├── base_model.py       # BaseModel ABC with prediction heads
│   ├── lstm.py             # LSTMModel
│   ├── transformer.py      # TransformerModel with CLS token
│   └── losses.py           # MultiTaskLoss, UncertaintyWeightedLoss
├── training/
│   └── trainer.py          # Trainer class with train/evaluate/save/load
├── backtesting/
│   └── backtest.py         # Backtester, BacktestResult, calculate_metrics()
├── execution/
│   └── broker.py           # SimulatedBroker, RiskManager
└── utils/
    └── __init__.py         # setup_logging(), get_device(), load_config()

scripts/
├── train.py                # CLI: train a model
├── backtest.py             # CLI: backtest a trained model
└── paper_trade.py          # CLI: paper trade loop

config/
├── default.yaml            # Base configuration
└── experiments/
    └── mvp_spy_lstm.yaml   # MVP experiment config
```

---

## Module Details

### `algotrade/data/loader.py`

Data fetching and caching layer.

**Classes**:
- `DataCache`: Parquet-based cache in `data/cache/`. Stores data as `{symbol}_{interval}.parquet`.

**Functions**:
- `fetch_data(symbols, start, end, interval, cache)`: Fetches OHLCV from Yahoo Finance. Supports single symbol or list. Returns DataFrame with columns: `open, high, low, close, volume`.

**Usage**:
```python
from algotrade.data import fetch_data, DataCache
cache = DataCache()
df = fetch_data("SPY", start="2020-01-01", cache=cache)
```

---

### `algotrade/features/pipeline.py`

Feature engineering and sequence preparation.

**Functions**:
- `compute_features(df, config)`: Computes technical indicators from OHLCV:
  - Price: `log_return`, `volatility_5`, `volatility_20`
  - Moving averages: `sma_5`, `sma_10`, `sma_20`, `sma_50` + ratios
  - Oscillators: `rsi_14`, `macd`, `macd_signal`, `macd_hist`
  - Bands: `bb_upper`, `bb_mid`, `bb_lower`, `bb_pct`
  - Volatility: `atr_14`
  - Volume: `volume_sma_20`, `volume_ratio`
  - Range: `high_low_pct`, `close_open_pct`

- `create_labels(df, threshold=0.001, horizon=1)`: Creates multi-task labels:
  - `direction`: 0=down, 1=flat, 2=up (based on threshold)
  - `magnitude`: future return value
  - `volatility`: rolling 20-day volatility

- `prepare_sequences(features, labels, lookback=60)`: Creates sliding window sequences for time-series models. Returns `(X, y_dict, dates)` where X is `(N, lookback, features)`.

- `normalize_features(df, method="zscore")`: Z-score or min-max normalization. Returns `(normalized_df, params)` for inverse transform.

---

### `algotrade/models/base_model.py`

Abstract base class for all prediction models.

**Class `BaseModel(ABC, nn.Module)`**:
- All models inherit from this
- Defines `_init_heads(backbone_dim)`: Creates 3 prediction heads:
  - `direction_head`: Linear → ReLU → Dropout → Linear(3) for classification
  - `magnitude_head`: Linear → ReLU → Dropout → Linear(1) for regression
  - `volatility_head`: Linear → ReLU → Dropout → Linear(1) for regression
- `_predict_from_backbone(h)`: Generates predictions dict from backbone output
- `num_parameters(trainable_only=True)`: Returns parameter count

---

### `algotrade/models/lstm.py`

LSTM-based time-series model.

**Class `LSTMModel(BaseModel)`**:
```python
LSTMModel(
    input_dim: int,        # Number of input features
    hidden_dim: int = 128, # LSTM hidden size
    num_layers: int = 2,   # Number of LSTM layers
    dropout: float = 0.2,  # Dropout rate
    bidirectional: bool = False,
)
```

**Architecture**:
1. Multi-layer LSTM with batch_first=True
2. LayerNorm on final hidden state
3. Dropout
4. 3 prediction heads (direction/magnitude/volatility)

**Forward pass**: Takes `(batch, seq_len, features)` → outputs `{"direction": (B,3), "magnitude": (B,1), "volatility": (B,1)}`

---

### `algotrade/models/transformer.py`

Transformer encoder with CLS token for sequence classification.

**Class `PositionalEncoding(nn.Module)`**: Sinusoidal positional encoding with max_len=1000.

**Class `TransformerModel(BaseModel)`**:
```python
TransformerModel(
    input_dim: int,
    hidden_dim: int = 128, # d_model
    num_heads: int = 8,    # Attention heads
    num_layers: int = 4,   # Encoder layers
    ff_dim: int = 512,     # Feedforward dimension
    dropout: float = 0.2,
)
```

**Architecture**:
1. Linear projection to hidden_dim
2. Prepend learnable CLS token
3. Add positional encoding
4. TransformerEncoder (GELU, pre-norm)
5. Extract CLS token output
6. 3 prediction heads

---

### `algotrade/models/losses.py`

Multi-task loss functions for joint training.

**Class `MultiTaskLoss(nn.Module)`**:
```python
MultiTaskLoss(
    direction_weight: float = 1.0,   # CrossEntropy weight
    magnitude_weight: float = 1.0,   # MSE weight
    volatility_weight: float = 0.5,  # MSE weight
    class_weights: List[float] = None,  # Per-class weights for imbalanced data
)
```
- Direction: CrossEntropyLoss (with optional class weights)
- Magnitude/Volatility: MSELoss
- Returns weighted sum, optionally with per-component breakdown
- **Class weights**: Use `[4.0, 1.0, 4.0]` to penalize errors on minority classes (up/down)

**Class `UncertaintyWeightedLoss(nn.Module)`**:
- Learns task weights via homoscedastic uncertainty (Kendall et al., 2018)
- Maintains learnable `log_vars` for each task
- Automatically balances tasks during training
- `get_weights()`: Returns current learned weights

---

### `algotrade/training/trainer.py`

Supervised training loop with early stopping and checkpointing.

**Class `Trainer`**:
```python
Trainer(
    model: nn.Module,
    device: torch.device = None,    # Auto-detects CUDA/MPS/CPU
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    class_weights: list = None,     # Per-class weights [down, flat, up]
)
```

**Methods**:
- `train(X_train, y_train, X_val, y_val, epochs, batch_size, patience)`:
  - Uses AdamW optimizer
  - ReduceLROnPlateau scheduler (factor=0.5, patience=10)
  - Gradient clipping (max_norm=1.0)
  - Early stopping based on validation loss
  - Saves best model to `checkpoints/best.pt`
  - Returns history dict: `{train_loss, val_loss, val_acc}`

- `save(filename)` / `load(filename)`: Checkpoint management

**Training targets dict structure**:
```python
y = {
    "direction": torch.Tensor,  # (N,) int64, values 0/1/2
    "magnitude": torch.Tensor,  # (N,) float32
    "volatility": torch.Tensor, # (N,) float32
}
```

---

### `algotrade/backtesting/backtest.py`

Vectorized backtesting engine.

**Dataclasses**:
- `Trade`: entry_date, entry_price, exit_date, exit_price, direction, size, pnl, return_pct
- `BacktestResult`: total_return, sharpe_ratio, max_drawdown, win_rate, num_trades, trades, equity_curve

**Class `Backtester`**:
```python
Backtester(
    initial_capital: float = 100000.0,
    commission_pct: float = 0.001,  # 0.1%
    slippage_pct: float = 0.001,    # 0.1%
)
```

**Methods**:
- `run(prices, signals, position_size)`: Vectorized backtest from signal series. Signals: 1=long, -1=short, 0=flat. Positions shift by 1 to avoid lookahead.
- `run_from_model(model, X, prices, dates, device, threshold=0.4)`: Runs backtest using model predictions. Uses argmax prediction with confidence threshold. Only trades when `max_prob > threshold`.

**Function `calculate_metrics(equity_curve)`**: Computes:
- total_return, annualized_return
- volatility (annualized)
- sharpe_ratio, sortino_ratio (risk-free=4%)
- max_drawdown, calmar_ratio

---

### `algotrade/execution/broker.py`

Paper trading simulation.

**Dataclasses**:
- `Position`: symbol, quantity, avg_price, current_price, value, pnl
- `Trade`: symbol, side, quantity, price, timestamp, commission

**Class `SimulatedBroker`**:
```python
SimulatedBroker(
    initial_capital: float = 100000.0,
    commission_per_share: float = 0.01,
    slippage_pct: float = 0.001,
)
```

**Methods**:
- `update_price(symbol, price)`: Update market price
- `buy(symbol, quantity)` / `sell(symbol, quantity)`: Execute orders with slippage
- `close_position(symbol)`: Close entire position
- Properties: `equity`, `unrealized_pnl`

**Class `RiskManager`**:
```python
RiskManager(
    max_position_pct: float = 0.25,  # Max 25% of equity in one position
    max_drawdown: float = 0.2,       # 20% drawdown triggers kill switch
    max_daily_loss: float = 0.05,    # Not implemented yet
)
```

- `check_order(broker, symbol, quantity, is_buy)`: Returns `(allowed: bool, reason: str)`
- Tracks high water mark and triggers kill switch on max drawdown

---

### `algotrade/utils/__init__.py`

Utility functions.

- `setup_logging(level="INFO", log_file=None)`: Configure logging with timestamp format
- `get_device(device="auto")`: Auto-detect best device (CUDA → MPS → CPU)
- `load_config(path)` / `save_config(config, path)`: YAML config I/O

---

## Scripts

### `scripts/train.py`

Train a prediction model.

```bash
python scripts/train.py --config config/experiments/mvp_spy_lstm.yaml
python scripts/train.py --config config/experiments/mvp_spy_lstm.yaml --epochs 50
python scripts/train.py --dry-run  # Validate data pipeline only
```

**Flow**:
1. Load config → fetch data → compute features → normalize → create labels
2. Prepare sequences with lookback window
3. Split 80/20 train/val
4. Create LSTM or Transformer model based on config
5. Train with early stopping, save best to `checkpoints/best.pt`

---

### `scripts/backtest.py`

Backtest a trained model.

```bash
python scripts/backtest.py --model checkpoints/best.pt --symbol SPY --start 2023-01-01
```

**Output**: Return, Sharpe, Max Drawdown, Win Rate, number of trades.

---

### `scripts/paper_trade.py`

Paper trading loop with simulated execution.

```bash
python scripts/paper_trade.py --model checkpoints/best.pt --symbol SPY --interval 60
```

**Features**:
- Refreshes data every `--interval` seconds
- Generates BUY/SELL/HOLD signals based on direction probability (>0.6 threshold)
- Uses RiskManager for position limits
- Graceful shutdown with Ctrl+C

---

## Configuration

Config files are YAML with these sections:

```yaml
experiment:
  name: "mvp_spy_lstm"
  seed: 42

data:
  symbols: ["SPY"]
  start_date: "2024-03-01"
  interval: "1h"            # "1d", "1h", "5m", "1m"
  granularity: "hourly"
  cache_dir: "data/cache"

features:
  technical: [sma_20, rsi_14, macd, bbands_upper, atr_14]
  normalize: true
  normalize_method: "zscore"

model:
  type: "transformer"       # or "lstm"
  hidden_dim: 256
  dropout: 0.2
  lookback: 60              # Sequence length
  
  # LSTM-specific
  lstm:
    num_layers: 2
    bidirectional: false
  
  # Transformer-specific
  transformer:
    num_heads: 8
    num_layers: 6
    ff_dim: 1024

training:
  epochs: 100
  batch_size: 128
  learning_rate: 0.001
  patience: 20
  checkpoint_dir: "checkpoints"

labels:
  direction_threshold: 0.001  # ±0.1% to trigger up/down
  horizon: 1                  # Predict next N bars
```

---

## Model Output Format

All models return a dictionary:

```python
{
    "direction": torch.Tensor,   # (batch, 3) logits for [down, flat, up]
    "magnitude": torch.Tensor,   # (batch, 1) predicted return
    "volatility": torch.Tensor,  # (batch, 1) predicted volatility
}
```

For inference:
```python
probs = torch.softmax(preds["direction"], dim=-1)  # Probabilities
pred_class = probs.argmax(dim=-1)  # 0=down, 1=flat, 2=up
```

---

## Data Pipeline

```
Raw OHLCV (yfinance)
    ↓
compute_features() → 20-24 technical indicators
    ↓
normalize_features() → z-score normalization
    ↓
create_labels() → direction/magnitude/volatility targets
    ↓
prepare_sequences() → (N, lookback, features) tensors
    ↓
Model training with MultiTaskLoss
```

---

## Checkpoints

Saved to `checkpoints/` with structure:
```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "best_loss": float,
}
```

Load with:
```python
checkpoint = torch.load("checkpoints/best.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
```

---

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Format
black algotrade/ scripts/

# Lint
ruff check algotrade/ scripts/

# Test
pytest tests/
```

---

## Experiment Results

### MVP SPY Transformer (Hourly)

| Metric | Value |
|--------|-------|
| Model | Transformer (4.8M params) |
| Data | SPY 1h bars, 2024-03-01 to present |
| Features | 24 technical indicators |
| Threshold | 0.2% for direction labels |
| Class weights | [4.0, 1.0, 4.0] |
| Best val accuracy | 69% |
| Backtest win rate | 60% |
| Backtest return | -29% |

**Observations**:
- Model has bullish bias (never predicts DOWN)
- High win rate but losing trades larger than winners
- Class imbalance: 67% flat, 17% up, 16% down

**Next steps**:
- Add stop-loss/take-profit logic
- Try longer horizons (daily/weekly)
- Add sentiment features
- Use ensemble models

---

## Warnings

⚠️ **yfinance is unofficial** - Data may have gaps or errors. Not suitable for production trading. Hourly data limited to 730 days.

⚠️ **Paper trading only** - No real broker integration. Use for research and development.

⚠️ **No financial advice** - This is an educational project. Past performance does not guarantee future results.

⚠️ **Market prediction is hard** - Random walk hypothesis applies. Most ML approaches fail to beat buy-and-hold.
