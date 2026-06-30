"""Market data collection with robust offline fallback."""

from __future__ import annotations

import os
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.config import (
        DATA_SOURCE_PATH,
        RAW_MARKET_DATA_PATH,
        START_DATE,
        SYNTHETIC_DATA_PATH,
        TICKERS,
        ensure_directories,
    )
    from src.synthetic_data import save_synthetic_market_data
except ImportError:  # pragma: no cover
    from config import DATA_SOURCE_PATH, RAW_MARKET_DATA_PATH, START_DATE, SYNTHETIC_DATA_PATH, TICKERS, ensure_directories
    from synthetic_data import save_synthetic_market_data


def _flatten_yfinance_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance's wide OHLCV output into a modeling-ready frame."""
    if data.empty:
        raise ValueError("yfinance returned an empty DataFrame")

    close_field = "Adj Close" if "Adj Close" in data.columns.get_level_values(0) else "Close"
    closes = data[close_field].dropna(how="all")
    closes = closes.ffill().dropna()
    if closes.empty or len(closes) < 300:
        raise ValueError("Downloaded market data is too sparse for modeling")

    available = [ticker for ticker in TICKERS if ticker in closes.columns]
    closes = closes[available]
    returns = closes.pct_change().fillna(0)
    weights = np.repeat(1 / len(available), len(available))
    portfolio_return = returns.to_numpy() @ weights

    if "Volume" in data.columns.get_level_values(0):
        volumes = data["Volume"][available].fillna(0).sum(axis=1).reindex(closes.index).ffill()
    else:
        volumes = pd.Series(1.0, index=closes.index)

    df = pd.DataFrame(
        {
            "date": closes.index,
            "market_index": closes["SPY"] if "SPY" in closes else closes.iloc[:, 0],
            "portfolio_value": 100 * np.cumprod(1 + portfolio_return),
            "market_return": returns["SPY"] if "SPY" in returns else returns.mean(axis=1),
            "portfolio_return": portfolio_return,
            "volume": volumes,
            "volume_change": volumes.pct_change().replace([np.inf, -np.inf], 0).fillna(0),
            "yield_change": returns["TLT"] * -0.35 if "TLT" in returns else returns.mean(axis=1) * -0.15,
            "volatility_proxy": returns.mean(axis=1).rolling(20).std().fillna(0) * 1000 + 15,
            "market_correlation": returns.rolling(40).corr(returns["SPY"] if "SPY" in returns else returns.mean(axis=1)).mean(axis=1).fillna(0.5),
        }
    )
    for ticker in available:
        df[f"{ticker}_close"] = closes[ticker].to_numpy()
    return df.reset_index(drop=True)


def try_yfinance_download() -> pd.DataFrame | None:
    """Attempt to download real market data from yfinance."""
    try:
        import yfinance as yf

        print("Attempting yfinance download...")
        data = yf.download(
            tickers=TICKERS,
            start=START_DATE,
            end=date.today().isoformat(),
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        df = _flatten_yfinance_frame(data)
        df.to_csv(RAW_MARKET_DATA_PATH, index=False)
        DATA_SOURCE_PATH.write_text("yfinance", encoding="utf-8")
        print(f"Real market data saved to {RAW_MARKET_DATA_PATH}")
        return df
    except Exception as exc:  # noqa: BLE001 - clear fallback path is intentional
        print(f"yfinance download failed: {exc}")
        return None


def try_kaggle_download() -> pd.DataFrame | None:
    """Try Kaggle only when credentials are discoverable."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env_credentials = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    if not (has_env_credentials or kaggle_json.exists()):
        print("Kaggle credentials not found; skipping Kaggle fallback.")
        return None

    try:
        print("Kaggle credentials detected, attempting dataset search...")
        subprocess.run(["kaggle", "datasets", "list", "-s", "sp500 historical stock market", "--csv"], check=True, capture_output=True, text=True)
        print("Kaggle CLI is available, but automatic dataset selection is intentionally conservative.")
    except Exception as exc:  # noqa: BLE001
        print(f"Kaggle fallback skipped: {exc}")
    return None


def collect_market_data() -> pd.DataFrame:
    """Collect real data when possible, otherwise generate synthetic data."""
    ensure_directories()
    for collector in (try_yfinance_download, try_kaggle_download):
        df = collector()
        if df is not None and not df.empty:
            return df
    print("Using synthetic data fallback.")
    return save_synthetic_market_data(SYNTHETIC_DATA_PATH)


if __name__ == "__main__":
    collect_market_data()
