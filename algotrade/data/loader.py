"""Data loading and caching."""

from pathlib import Path
from typing import Union, List
from datetime import datetime
import time
import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger("algotrade.data")


class DataCache:
    """Simple parquet-based cache for OHLCV data."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _path(self, symbol: str, interval: str) -> Path:
        return self.cache_dir / f"{symbol}_{interval}.parquet"
    
    def load(self, symbol: str, interval: str = "1d") -> pd.DataFrame | None:
        path = self._path(symbol, interval)
        if path.exists():
            return pd.read_parquet(path)
        return None
    
    def save(self, df: pd.DataFrame, symbol: str, interval: str = "1d"):
        df.to_parquet(self._path(symbol, interval))


def fetch_data(
    symbols: Union[str, List[str]],
    start: str = "2020-01-01",
    end: str = None,
    interval: str = "1d",
    cache: DataCache = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance.
    
    Args:
        symbols: Single symbol or list (e.g., "SPY" or ["SPY", "QQQ"])
        start: Start date (YYYY-MM-DD)
        end: End date (defaults to today)
        interval: "1d", "1h", "5m", "1m"
        cache: Optional DataCache instance
    
    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    
    end = end or datetime.now().strftime("%Y-%m-%d")
    all_data = []
    
    for symbol in symbols:
        # Try cache first
        if cache:
            cached = cache.load(symbol, interval)
            if cached is not None:
                # Filter to requested date range
                mask = (cached.index >= start) & (cached.index <= end)
                filtered = cached[mask]
                if len(filtered) > 0:
                    logger.info(f"Loaded {symbol} from cache ({len(filtered)} rows)")
                    all_data.append(filtered)
                    continue
        
        # Fetch from yfinance with error handling
        logger.info(f"Fetching {symbol} from Yahoo Finance...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval)
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            continue
        
        if df.empty:
            logger.warning(f"No data for {symbol}")
            continue
        
        # Normalize column names
        df.columns = df.columns.str.lower()
        df = df[["open", "high", "low", "close", "volume"]]
        df["symbol"] = symbol
        
        # Cache it
        if cache:
            cache.save(df.drop(columns=["symbol"]), symbol, interval)
        
        all_data.append(df)
        logger.info(f"Fetched {symbol}: {len(df)} rows")
        
        # Rate limiting to avoid yfinance API throttling
        if len(symbols) > 1:
            time.sleep(0.3)
    
    if not all_data:
        raise ValueError(f"No data fetched for symbols: {symbols}")
    
    if len(all_data) == 1:
        return all_data[0].drop(columns=["symbol"], errors="ignore")
    
    return pd.concat(all_data)
