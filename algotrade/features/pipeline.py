"""Feature engineering pipeline."""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False


def _compute_price_features(
    features: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    open_: pd.Series,
    config: Dict,
) -> None:
    """Compute price-based features (returns, volatility, moving averages)."""
    features["log_return"] = np.log(close / close.shift(1))
    features["volatility_20"] = features["log_return"].rolling(20).std()
    features["volatility_5"] = features["log_return"].rolling(5).std()
    
    # Moving averages
    for period in config.get("sma_periods", [5, 10, 20, 50]):
        features[f"sma_{period}"] = close.rolling(period).mean()
        features[f"sma_{period}_ratio"] = close / features[f"sma_{period}"]
    
    # Price range
    features["high_low_pct"] = (high - low) / close
    features["close_open_pct"] = (close - open_) / open_


def _compute_technical_features(
    features: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> None:
    """Compute technical indicators (RSI, MACD, BB, ATR, volume)."""
    if HAS_PANDAS_TA:
        features["rsi_14"] = ta.rsi(close, length=14)
        
        macd = ta.macd(close)
        if macd is not None:
            features["macd"] = macd.iloc[:, 0]
            features["macd_signal"] = macd.iloc[:, 1]
            features["macd_hist"] = macd.iloc[:, 2]
        
        bb = ta.bbands(close, length=20)
        if bb is not None:
            features["bb_upper"] = bb.iloc[:, 0]
            features["bb_mid"] = bb.iloc[:, 1]
            features["bb_lower"] = bb.iloc[:, 2]
            features["bb_pct"] = (close - features["bb_lower"]) / (features["bb_upper"] - features["bb_lower"])
        
        features["atr_14"] = ta.atr(high, low, close, length=14)
    else:
        # Simple RSI without pandas_ta
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        features["rsi_14"] = 100 - (100 / (1 + gain / loss))
    
    # Volume features
    features["volume_sma_20"] = volume.rolling(20).mean()
    features["volume_ratio"] = volume / features["volume_sma_20"]


def _compute_temporal_features(features: pd.DataFrame, index: pd.Index) -> None:
    """Compute cyclical temporal features (hour, day of week, month)."""
    if not isinstance(index, pd.DatetimeIndex):
        return
    
    # Hour of day (for intraday data) - cyclical encoding
    hour = index.hour
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    
    # Day of week - cyclical encoding
    dow = index.dayofweek
    features["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    
    # Month - cyclical encoding (for seasonality)
    month = index.month
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)


def _compute_regime_features(
    features: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
) -> None:
    """Compute market regime indicators (trend, volatility regime, momentum)."""
    # MA crossover signals (trend regime)
    sma_50 = features.get("sma_50", close.rolling(50).mean())
    sma_200 = close.rolling(200).mean()
    features["ma_cross_50_200"] = (sma_50 > sma_200).astype(float)
    features["ma_distance_50_200"] = (sma_50 - sma_200) / close
    
    # Volatility regime (high/low vol)
    vol_20 = features.get("volatility_20", features["log_return"].rolling(20).std())
    vol_60 = features["log_return"].rolling(60).std()
    features["vol_regime"] = vol_20 / vol_60
    
    # Momentum regime
    features["momentum_20"] = close.pct_change(20)
    features["momentum_60"] = close.pct_change(60)
    
    # Mean reversion indicator (distance from 20-day mean)
    features["mean_reversion_20"] = (close - close.rolling(20).mean()) / close.rolling(20).std()
    
    # Trend strength (ADX-like using directional movement)
    if "atr_14" in features.columns:
        atr = features["atr_14"]
    else:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
    
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    features["di_diff"] = plus_di - minus_di


def compute_features(df: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
    """Compute all features from OHLCV data."""
    config = config or {}
    features = pd.DataFrame(index=df.index)
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]
    
    _compute_price_features(features, close, high, low, open_, config)
    _compute_technical_features(features, close, high, low, volume)
    _compute_temporal_features(features, df.index)
    _compute_regime_features(features, close, high, low)
    
    return features


def create_labels(df: pd.DataFrame, threshold: float = 0.001, horizon: int = 1) -> pd.DataFrame:
    """Create labels for multi-task prediction."""
    close = df["close"]
    future_ret = close.pct_change(horizon).shift(-horizon)
    
    labels = pd.DataFrame(index=df.index)
    labels["magnitude"] = future_ret
    labels["direction"] = 1  # flat
    labels.loc[future_ret > threshold, "direction"] = 2  # up
    labels.loc[future_ret < -threshold, "direction"] = 0  # down
    labels["volatility"] = close.pct_change().rolling(20).std().shift(-horizon)
    
    return labels


def prepare_sequences(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    lookback: int = 60,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], pd.DatetimeIndex]:
    """Create sequences for time-series models."""
    # Align and drop NaN
    data = pd.concat([features, labels], axis=1).dropna()
    feat_cols = features.columns.tolist()
    
    X, y_dir, y_mag, y_vol, dates = [], [], [], [], []
    
    for i in range(lookback, len(data)):
        X.append(data[feat_cols].iloc[i-lookback:i].values)
        y_dir.append(data["direction"].iloc[i])
        y_mag.append(data["magnitude"].iloc[i])
        y_vol.append(data["volatility"].iloc[i])
        dates.append(data.index[i])
    
    return (
        np.array(X, dtype=np.float32),
        {
            "direction": np.array(y_dir, dtype=np.int64),
            "magnitude": np.array(y_mag, dtype=np.float32),
            "volatility": np.array(y_vol, dtype=np.float32),
        },
        pd.DatetimeIndex(dates),
    )


def normalize_features(df: pd.DataFrame, method: str = "zscore") -> Tuple[pd.DataFrame, Dict]:
    """Normalize features. Returns normalized df and params for inverse transform."""
    params = {}
    result = df.copy()
    
    for col in df.columns:
        if method == "zscore":
            mean, std = df[col].mean(), df[col].std()
            params[col] = {"mean": mean, "std": std}
            result[col] = (df[col] - mean) / (std + 1e-8)
        elif method == "minmax":
            min_val, max_val = df[col].min(), df[col].max()
            params[col] = {"min": min_val, "max": max_val}
            result[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)
    
    return result, params
