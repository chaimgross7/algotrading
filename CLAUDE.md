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

## Recent Updates (Feb 2026)

### Code Cleanup & Simplification
- **Model Factory**: New `factory.py` with `create_model()` and `load_model()` eliminates duplication across scripts
- **TrainerConfig**: Dataclass groups 15+ training parameters into a single config object with `from_config()` factory method
- **Refactored Features**: `compute_features()` split into focused helpers: `_compute_price_features()`, `_compute_technical_features()`, `_compute_temporal_features()`, `_compute_regime_features()`
- **Unified Tuning**: Generic `_tune_parameter()` method in Backtester replaces duplicated threshold tuning logic
- **Helper Utilities**: `_ensure_2d()` tensor helper in losses.py, consistent logger usage, error handling in data loader
- **Simplified Scripts**: `train.py` refactored with `load_symbol_data()` and `prepare_datasets()` helpers

### Previous Updates
- **Multi-Symbol Training**: Load and combine data from 100+ symbols (NASDAQ 100) for more robust training
- **Symbol Presets**: New `symbols.py` module with curated symbol lists (`nasdaq100`, `nasdaq100_top25`, `spy`, `major_etfs`)
- **Magnitude-Based Trading**: Trade on predicted return magnitude instead of direction classification
- **Benchmark Comparison**: Backtest results now include buy-and-hold benchmark, alpha, and information ratio
- **AdaptivePlateauScheduler**: LR scheduler with auto-restart to escape local minima after multiple reductions
- **Forward Testing Logger**: Track out-of-sample predictions and trades with JSON/CSV logging
- **Separated Data Configs**: `--data-config` flag to decouple symbol data from experiment configs
- **Label Normalization**: Optional z-score normalization of magnitude/volatility labels
- **Rate Limiting**: 0.3s delay between yfinance API calls for multi-symbol fetching

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
│   ├── loader.py           # DataCache + fetch_data() with rate limiting and error handling
│   └── symbols.py          # NASDAQ 100 symbol lists and presets
├── features/
│   └── pipeline.py         # compute_features(), create_labels(), prepare_sequences()
├── models/
│   ├── base_model.py       # BaseModel ABC with prediction heads
│   ├── factory.py          # create_model(), load_model() factory functions
│   ├── lstm.py             # LSTMModel
│   ├── transformer.py      # TransformerModel with CLS token
│   └── losses.py           # MultiTaskLoss, UncertaintyWeightedLoss, _ensure_2d()
├── training/
│   └── trainer.py          # Trainer, TrainerConfig, AdaptivePlateauScheduler
├── backtesting/
│   ├── backtest.py         # Backtester, BacktestResult, benchmark metrics
│   └── forward_test.py     # ForwardTestLogger for out-of-sample tracking
├── execution/
│   └── broker.py           # SimulatedBroker, RiskManager
└── utils/
    └── __init__.py         # setup_logging(), get_device(), load_config(), deep_merge()

scripts/
├── train.py                # CLI: train a model (uses model factory and TrainerConfig)
├── backtest.py             # CLI: backtest with magnitude mode and threshold tuning
└── paper_trade.py          # CLI: paper trade with forward test logging

config/
├── default.yaml            # Base configuration
├── data/                   # Separate data configurations
│   ├── nasdaq100.yaml      # NASDAQ 100 symbols (10+ years daily)
│   └── spy.yaml            # SPY-only configuration
└── experiments/
    ├── mvp_spy_lstm.yaml          # MVP experiment
    ├── daily_multi_symbol.yaml    # Multi-symbol magnitude training
    ├── nasdaq100.yaml             # NASDAQ 100 training config
    ├── spy_magnitude_regression.yaml  # Magnitude-only regression
    └── transformer_large.yaml     # Large transformer config
```

---

## Module Details

### `algotrade/data/loader.py`

Data fetching and caching layer.

**Classes**:
- `DataCache`: Parquet-based cache in `data/cache/`. Stores data as `{symbol}_{interval}.parquet`.

**Functions**:
- `fetch_data(symbols, start, end, interval, cache, delay=0.3)`: Fetches OHLCV from Yahoo Finance. Supports single symbol or list. Returns DataFrame with columns: `open, high, low, close, volume`. When fetching multiple symbols, adds 0.3s delay between API calls to avoid rate limiting.

**Usage**:
```python
from algotrade.data import fetch_data, DataCache
cache = DataCache()

# Single symbol
df = fetch_data("SPY", start="2020-01-01", cache=cache)

# Multiple symbols (with rate limiting)
from algotrade.data.symbols import get_symbols
symbols = get_symbols("nasdaq100_top25")
df = fetch_data(symbols, start="2015-01-01", cache=cache)  # 0.3s delay between calls
```

---

### `algotrade/data/symbols.py`

Stock symbol lists for training data. Manually maintained for reproducibility.

**Constants**:
- `NASDAQ_100`: Full NASDAQ 100 index components (100+ symbols, cleaned of delisted stocks)
- `NASDAQ_100_TOP10`: Top 10 by market cap for quick testing
- `NASDAQ_100_TOP25`: Top 25 for intermediate experiments
- `SPY`: Single-symbol list `["SPY"]`
- `MAJOR_ETFS`: `["SPY", "QQQ", "IWM", "DIA"]`

**Functions**:
- `get_symbols(name)`: Returns symbol list by preset name. Available: `"nasdaq100"`, `"nasdaq100_top10"`, `"nasdaq100_top25"`, `"spy"`, `"major_etfs"`

**Usage**:
```python
from algotrade.data.symbols import get_symbols, NASDAQ_100

symbols = get_symbols("nasdaq100_top25")  # ['AAPL', 'MSFT', 'NVDA', ...]
all_symbols = NASDAQ_100  # Full 100+ symbols
```

---

### `algotrade/features/pipeline.py`

Feature engineering and sequence preparation. Refactored into modular helper functions.

**Internal helper functions** (called by `compute_features`):
- `_compute_price_features()`: Returns, volatility, moving averages, price range
- `_compute_technical_features()`: RSI, MACD, Bollinger Bands, ATR, volume
- `_compute_temporal_features()`: Cyclical hour/day/month encoding
- `_compute_regime_features()`: Trend, volatility regime, momentum indicators

**Public functions**:
- `compute_features(df, config)`: Computes all technical indicators from OHLCV by calling the helper functions:
  - Price: `log_return`, `volatility_5`, `volatility_20`
  - Moving averages: `sma_5`, `sma_10`, `sma_20`, `sma_50` + ratios
  - Oscillators: `rsi_14`, `macd`, `macd_signal`, `macd_hist`
  - Bands: `bb_upper`, `bb_mid`, `bb_lower`, `bb_pct`
  - Volatility: `atr_14`
  - Volume: `volume_sma_20`, `volume_ratio`
  - Range: `high_low_pct`, `close_open_pct`
  - Temporal: `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos` (cyclical encoding)
  - Regime: `ma_cross_50_200`, `ma_distance_50_200`, `vol_regime`, `momentum_20/60`, `mean_reversion_20`, `di_diff`

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

LSTM-based time-series model with attention pooling.

**Class `AttentionPooling(nn.Module)`**: Learnable attention-weighted pooling over sequence dimension. Uses 2-layer network to compute attention scores.

**Class `LSTMModel(BaseModel)`**:
```python
LSTMModel(
    input_dim: int,        # Number of input features
    hidden_dim: int = 128, # LSTM hidden size
    num_layers: int = 2,   # Number of LSTM layers
    dropout: float = 0.2,  # Dropout rate
    bidirectional: bool = False,
    use_attention_pooling: bool = True,  # NEW: use attention over all timesteps
)
```

**Architecture**:
1. Multi-layer LSTM with batch_first=True
2. **Attention pooling** over all timesteps (or fallback to final hidden state if disabled)
3. LayerNorm on pooled representation
4. Dropout
5. 3 prediction heads (direction/magnitude/volatility)

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

### `algotrade/models/factory.py`

Model factory for creating and loading models. Eliminates duplication across scripts.

**Functions**:
- `create_model(input_dim, model_cfg)`: Creates model based on config dict
- `load_model(checkpoint_path, input_dim, model_cfg, device)`: Loads model from checkpoint

**Usage**:
```python
from algotrade.models import create_model, load_model

# Create from config
model = create_model(input_dim=45, model_cfg={"type": "lstm", "hidden_dim": 128})

# Load from checkpoint
model = load_model("checkpoints/best.pt", input_dim=45, model_cfg=cfg["model"], device=device)
```

---

### `algotrade/models/losses.py`

Multi-task loss functions for joint training.

**Helper function `_ensure_2d(tensor)`**: Ensures tensor has 2 dimensions `(batch, 1)` for MSE loss compatibility.

**Class `MultiTaskLoss(nn.Module)`**:
```python
MultiTaskLoss(
    direction_weight: float = 1.0,   # CrossEntropy weight
    magnitude_weight: float = 1.0,   # MSE weight
    volatility_weight: float = 0.5,  # MSE weight
    class_weights: List[float] = None,  # Per-class weights for imbalanced data
    label_smoothing: float = 0.1,    # Reduces overconfidence, improves generalization
)
```
- Direction: CrossEntropyLoss (with optional class weights and label smoothing)
- Magnitude/Volatility: MSELoss
- Returns weighted sum, optionally with per-component breakdown
- **Class weights**: Use `[4.0, 1.0, 4.0]` to penalize errors on minority classes (up/down)

**Class `UncertaintyWeightedLoss(nn.Module)`**:
- Learns task weights via homoscedastic uncertainty (Kendall et al., 2018)
- Maintains learnable `log_vars` for each task
- Automatically balances tasks during training
- Supports `label_smoothing` parameter
- `get_weights()`: Returns current learned weights

---

### `algotrade/training/trainer.py`

Supervised training loop with early stopping, checkpointing, and adaptive LR scheduling.

**Dataclass `TrainerConfig`**:
Groups all training configuration into a single object.
```python
TrainerConfig(
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    class_weights: Optional[List[float]] = None,
    use_uncertainty_loss: bool = False,
    warmup_epochs: int = 5,
    label_smoothing: float = 0.1,
    direction_weight: float = 1.0,
    magnitude_weight: float = 0.5,
    volatility_weight: float = 0.3,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.1,
    scheduler_max_reductions: int = 4,
    scheduler_restart_lr_factor: float = 0.5,
)
```

**Factory method `TrainerConfig.from_config(train_cfg)`**: Creates TrainerConfig from config dict (YAML).

**Class `AdaptivePlateauScheduler`**:
LR scheduler that reduces on plateau and restarts after multiple reductions to escape local minima.
```python
AdaptivePlateauScheduler(
    optimizer,
    initial_lr: float,
    factor: float = 0.1,         # LR reduction factor
    patience: int = 10,          # Epochs to wait before reducing
    max_reductions: int = 4,     # Trigger restart after N reductions
    restart_lr_factor: float = 0.5,  # Restart LR = initial_lr * restart_lr_factor
    min_lr: float = 1e-7,
)
```
- Wraps `ReduceLROnPlateau` with restart logic
- After 4 consecutive reductions, restarts to 50% of initial LR
- Helps escape local minima in long training runs

**Class `Trainer`**:
```python
Trainer(
    model: nn.Module,
    device: torch.device = None,    # Auto-detects CUDA/MPS/CPU
    config: TrainerConfig = None,   # Preferred: use TrainerConfig
    **kwargs,                       # Legacy: individual params for backward compatibility
)
```

**Helper function `compute_class_weights(labels, num_classes)`**: Computes inverse-frequency class weights for imbalanced labels.

**Methods**:
- `train(X_train, y_train, X_val, y_val, epochs, batch_size, patience)`:
  - Uses AdamW optimizer
  - **LR Warmup** for first N epochs (linear ramp-up)
  - **AdaptivePlateauScheduler** with restart logic (or ReduceLROnPlateau fallback)
  - Gradient clipping (max_norm=1.0)
  - Early stopping based on validation loss
  - **Label normalization**: Optionally z-score normalize magnitude/volatility to prevent mean collapse
  - Logs learned task weights when using UncertaintyWeightedLoss
  - Saves best model to `checkpoints/best.pt` and `label_stats.json`
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

**Magnitude-Only Training**: Set `direction_weight: 0` in config to disable direction loss and train a pure regression model.

---

### `algotrade/backtesting/backtest.py`

Vectorized backtesting engine with benchmark comparison.

**Dataclasses**:
- `Trade`: entry_date, entry_price, exit_date, exit_price, direction, size, pnl, return_pct
- `BacktestResult`: Enhanced with benchmark metrics:
  - Strategy: `total_return`, `sharpe_ratio`, `max_drawdown`, `win_rate`, `num_trades`, `trades`, `equity_curve`
  - Benchmark (buy-and-hold): `benchmark_return`, `benchmark_sharpe`, `benchmark_max_drawdown`, `benchmark_equity`
  - Comparison: `alpha` (strategy - benchmark return), `information_ratio` (excess return / tracking error)

**Class `Backtester`**:
```python
Backtester(
    initial_capital: float = 100000.0,
    commission_pct: float = 0.001,  # 0.1%
    slippage_pct: float = 0.001,    # 0.1%
)
```

**Methods**:
- `run(prices, signals, position_size)`: Vectorized backtest from signal series. Signals: 1=long, -1=short, 0=flat. Positions shift by 1 to avoid lookahead. Returns `BacktestResult` with benchmark comparison.
- `run_from_model(model, X, prices, dates, device, threshold=0.4)`: Runs backtest using direction predictions with confidence threshold.
- `run_from_magnitude(model, X, prices, dates, device, long_threshold, short_threshold)`: **NEW** - Backtest using magnitude predictions. Goes long when `magnitude > long_threshold`, short when `magnitude < short_threshold`.
- `tune_threshold(model, X, prices, dates, thresholds, metric, device)`: Grid search for optimal confidence threshold.
- `tune_magnitude_threshold(model, X, prices, dates, long_thresholds, short_thresholds, metric, device)`: **NEW** - Grid search for optimal magnitude thresholds. Returns best long/short thresholds.

**Function `calculate_metrics(equity_curve)`**: Computes:
- total_return, annualized_return
- volatility (annualized)
- sharpe_ratio, sortino_ratio (risk-free=4%)
- max_drawdown, calmar_ratio

**Output Format**:
```
Strategy:
  Return:       12.50%
  Sharpe:        1.25
  Max DD:        8.50%
  Win Rate:     62.50%
  Trades:          48

Benchmark (Buy & Hold):
  Return:        8.20%
  Sharpe:        0.95
  Max DD:       12.30%

Comparison:
  Alpha:         4.30%
  Info Ratio:    0.85
```

---

### `algotrade/backtesting/forward_test.py`

Forward testing logger for out-of-sample performance tracking.

**Dataclasses**:
- `PredictionLog`: timestamp, symbol, price, probabilities (up/down/flat), signal, magnitude_pred, volatility_pred
- `TradeLog`: timestamp, symbol, action, quantity, price, position_after, equity_after, pnl
- `SessionSummary`: session_id, start/end time, returns, alpha, accuracy metrics

**Class `ForwardTestLogger`**:
```python
ForwardTestLogger(
    symbol: str,
    initial_capital: float,
    log_dir: str = "logs/forward_test",
    auto_save: bool = True,  # Register atexit handler
)
```

**Methods**:
- `set_benchmark_start(price)`: Set starting price for benchmark comparison
- `log_prediction(timestamp, symbol, price, probabilities, signal, magnitude_pred, volatility_pred)`: Log a model prediction
- `log_trade(timestamp, symbol, action, quantity, price, position_after, equity_after, pnl)`: Log a trade execution
- `save_session(final_equity, benchmark_price)`: Save session to JSON and CSV files

**Output Files** (in `logs/forward_test/`):
- `{session_id}_predictions.csv`: All predictions with probabilities
- `{session_id}_trades.csv`: All trades with P&L
- `{session_id}_summary.json`: Session summary with strategy vs benchmark

**Usage**:
```python
from algotrade.backtesting.forward_test import ForwardTestLogger

logger = ForwardTestLogger(symbol="SPY", initial_capital=100000)
logger.log_prediction(timestamp=now, symbol="SPY", price=450.50,
                      probabilities={"up": 0.65, "down": 0.15, "flat": 0.20},
                      signal="BUY", magnitude_pred=0.0023)
logger.log_trade(timestamp=now, symbol="SPY", action="BUY",
                 quantity=100, price=450.50, position_after=100, equity_after=100500)
# On exit (automatic via atexit):
logger.save_session(final_equity=105000, benchmark_price=460.00)
```

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
- `deep_merge(base, override)`: **NEW** - Recursively merge two dicts. Override values take precedence, nested dicts are merged.
- `load_config(path, data_config=None)`: **UPDATED** - Load YAML config with optional data config merging. If `data_config` is provided, its `data` section is merged into the result.
- `save_config(config, path)`: Save config to YAML file

**Usage**:
```python
from algotrade.utils import load_config, deep_merge

# Load experiment config with separate data config
config = load_config(
    "config/experiments/daily_multi_symbol.yaml",
    data_config="config/data/nasdaq100.yaml"
)

# Manual deep merge
base = {"model": {"hidden_dim": 64, "dropout": 0.2}}
override = {"model": {"hidden_dim": 128}}  # Only override hidden_dim
merged = deep_merge(base, override)  # {"model": {"hidden_dim": 128, "dropout": 0.2}}
```

---

## Scripts

### `scripts/train.py`

Train a prediction model.

```bash
# Basic training
python scripts/train.py --config config/experiments/mvp_spy_lstm.yaml

# With separate data config (for multi-symbol)
python scripts/train.py --config config/experiments/daily_multi_symbol.yaml \
                         --data-config config/data/nasdaq100.yaml

# Dry run to validate pipeline
python scripts/train.py --config config/experiments/nasdaq100.yaml --dry-run

# Override epochs
python scripts/train.py --config config/experiments/mvp_spy_lstm.yaml --epochs 50
```

**New Flags**:
- `--data-config`: Path to separate data config (merged into experiment config)
- `--dry-run`: Validate data pipeline without training

**Flow**:
1. Load config (with optional data config merge)
2. Fetch data (single or multi-symbol with rate limiting)
3. Compute features → normalize → create labels
4. Prepare sequences with lookback window
5. Split train/val/test (default 70/15/15 or 80/20)
6. Create LSTM or Transformer model based on config
7. Train with early stopping and AdaptivePlateauScheduler
8. Save best model to `checkpoints/best.pt` and `label_stats.json`

---

### `scripts/backtest.py`

Backtest a trained model.

```bash
# Basic backtest with direction predictions
python scripts/backtest.py --model checkpoints/best.pt --symbol SPY --start 2023-01-01

# Tune confidence threshold
python scripts/backtest.py --model checkpoints/best.pt --tune-threshold

# Magnitude mode (trade on predicted returns)
python scripts/backtest.py --model checkpoints/best.pt --mode magnitude \
                            --long-threshold 0.002 --short-threshold -0.002

# Tune magnitude thresholds
python scripts/backtest.py --model checkpoints/best.pt --mode magnitude --tune-threshold
```

**New Flags**:
- `--mode`: `direction` (default) or `magnitude`
- `--tune-threshold`: Grid search for optimal threshold
- `--long-threshold`: Magnitude threshold for long signals (default 0.002)
- `--short-threshold`: Magnitude threshold for short signals (default -0.002)
- `--threshold`: Confidence threshold for direction mode (default 0.4)

**Output**: Strategy vs Benchmark comparison with alpha and information ratio.

---

### `scripts/paper_trade.py`

Paper trading loop with simulated execution and forward test logging.

```bash
python scripts/paper_trade.py --model checkpoints/best.pt --symbol SPY --interval 60
```

**Features**:
- Refreshes data every `--interval` seconds
- Generates BUY/SELL/HOLD signals based on direction probability (>0.6 threshold)
- Uses RiskManager for position limits
- **Forward Test Logging**: Tracks all predictions and trades to `logs/forward_test/`
- Shows real-time alpha vs buy-and-hold benchmark
- Graceful shutdown with Ctrl+C (auto-saves session)

**Output Files**:
- `logs/forward_test/{session_id}_predictions.csv`
- `logs/forward_test/{session_id}_trades.csv`
- `logs/forward_test/{session_id}_summary.json`

---

## Configuration

Config files are YAML with these sections:

```yaml
experiment:
  name: "mvp_spy_lstm"
  seed: 42

data:
  symbols: ["SPY"]           # Explicit symbol list
  symbols_preset: "nasdaq100" # OR use preset from symbols.py
  start_date: "2024-03-01"
  interval: "1h"             # "1d", "1h", "5m", "1m"
  granularity: "hourly"
  cache_dir: "data/cache"
  train_ratio: 0.7           # Train/val/test split
  val_ratio: 0.15
  test_ratio: 0.15

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
    use_attention_pooling: true
  
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
  warmup_epochs: 5
  
  # Task loss weights (set direction: 0 for magnitude-only)
  loss_weights:
    direction: 1.0
    magnitude: 1.0
    volatility: 0.5
  
  # Label normalization for regression tasks
  normalize_labels: true
  
  # AdaptivePlateauScheduler config
  scheduler:
    type: "adaptive_plateau"  # or "reduce_on_plateau"
    patience: 10
    factor: 0.1
    max_reductions: 4         # Restart after 4 reductions
    restart_lr_factor: 0.5    # Restart to 50% of initial LR
  
  early_stopping:
    enabled: true
    patience: 20
    min_delta: 0.0001

labels:
  direction_threshold: 0.001  # ±0.1% to trigger up/down
  horizon: 1                  # Predict next N bars
```

### Separate Data Configs

Data configuration can be separated from experiment configs using `--data-config`:

```yaml
# config/data/nasdaq100.yaml
data:
  provider: "yfinance"
  symbols_preset: "nasdaq100"  # Uses symbols.py list
  start_date: "2015-01-01"
  interval: "1d"
  cache_dir: "data/cache"
```

```bash
python scripts/train.py --config config/experiments/daily_multi_symbol.yaml \
                         --data-config config/data/nasdaq100.yaml
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

**Completed improvements (Feb 2026)**:
- ✅ LSTM attention pooling over all timesteps
- ✅ Confidence threshold tuning via grid search
- ✅ UncertaintyWeightedLoss wiring (auto-learned task weights)
- ✅ Class balancing helper function
- ✅ LR warmup scheduler
- ✅ Label smoothing (0.1 default)
- ✅ Extended features: temporal (hour/dow/month), regime indicators (MA cross, vol regime, momentum)
- ✅ Multi-symbol training (NASDAQ 100 with 10+ years daily data)
- ✅ Magnitude-based trading mode
- ✅ Benchmark comparison with alpha/information ratio
- ✅ AdaptivePlateauScheduler with restart logic
- ✅ Forward test logging for out-of-sample tracking
- ✅ Separated data configs for flexibility

**Next steps**:
- Add stop-loss/take-profit logic
- Try longer horizons (daily/weekly)
- Add sentiment features
- Use ensemble models
- Implement walk-forward validation

---

## Warnings

⚠️ **yfinance is unofficial** - Data may have gaps or errors. Not suitable for production trading. Hourly data limited to 730 days.

⚠️ **Paper trading only** - No real broker integration. Use for research and development.

⚠️ **No financial advice** - This is an educational project. Past performance does not guarantee future results.

⚠️ **Market prediction is hard** - Random walk hypothesis applies. Most ML approaches fail to beat buy-and-hold.
