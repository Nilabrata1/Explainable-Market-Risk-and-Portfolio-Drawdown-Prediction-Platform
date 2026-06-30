"""Feature engineering for portfolio drawdown-risk classification."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from src.config import (
        DRAWDOWN_THRESHOLD,
        PROCESSED_DATA_PATH,
        RAW_MARKET_DATA_PATH,
        SYNTHETIC_DATA_PATH,
        TARGET_HORIZON_DAYS,
        ensure_directories,
    )
except ImportError:  # pragma: no cover
    from config import DRAWDOWN_THRESHOLD, PROCESSED_DATA_PATH, RAW_MARKET_DATA_PATH, SYNTHETIC_DATA_PATH, TARGET_HORIZON_DAYS, ensure_directories


FEATURE_COLUMNS = [
    "daily_return",
    "portfolio_return",
    "5_day_return",
    "10_day_return",
    "20_day_return",
    "rolling_volatility_5",
    "rolling_volatility_10",
    "rolling_volatility_20",
    "moving_average_5",
    "moving_average_20",
    "moving_average_ratio",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "max_drawdown_20",
    "volume_change",
    "volatility_proxy",
    "yield_change",
    "market_correlation",
    "beta_proxy",
]


def _max_drawdown(series: pd.Series) -> float:
    """Return max drawdown for a rolling price/value window."""
    running_max = series.cummax()
    drawdown = series / running_max - 1
    return float(drawdown.min())


def create_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe features and the future drawdown target."""
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "portfolio_value" not in df:
        if "portfolio_return" in df:
            df["portfolio_value"] = 100 * (1 + df["portfolio_return"].fillna(0)).cumprod()
        else:
            raise ValueError("Input data must include portfolio_value or portfolio_return")

    if "market_return" not in df:
        df["market_return"] = df["portfolio_value"].pct_change().fillna(0)
    if "portfolio_return" not in df:
        df["portfolio_return"] = df["portfolio_value"].pct_change().fillna(0)
    if "volume" not in df:
        df["volume"] = 1.0

    df["daily_return"] = df["portfolio_value"].pct_change().fillna(0)
    for window in (5, 10, 20):
        df[f"{window}_day_return"] = df["portfolio_value"].pct_change(window)
        df[f"rolling_volatility_{window}"] = df["daily_return"].rolling(window).std()
        df[f"momentum_{window}"] = df["portfolio_value"] / df["portfolio_value"].shift(window) - 1

    df["moving_average_5"] = df["portfolio_value"].rolling(5).mean()
    df["moving_average_20"] = df["portfolio_value"].rolling(20).mean()
    df["moving_average_ratio"] = df["moving_average_5"] / df["moving_average_20"] - 1
    df["max_drawdown_20"] = df["portfolio_value"].rolling(20).apply(_max_drawdown, raw=False)
    df["volume_change"] = df.get("volume_change", df["volume"].pct_change()).replace([np.inf, -np.inf], np.nan)
    df["volatility_proxy"] = df.get("volatility_proxy", df["daily_return"].rolling(20).std() * 1000 + 15)
    df["yield_change"] = df.get("yield_change", -0.15 * df["market_return"])
    df["market_correlation"] = df.get("market_correlation", df["daily_return"].rolling(40).corr(df["market_return"]))
    rolling_cov = df["daily_return"].rolling(60).cov(df["market_return"])
    rolling_var = df["market_return"].rolling(60).var()
    df["beta_proxy"] = rolling_cov / rolling_var.replace(0, np.nan)

    df["future_5d_return"] = df["portfolio_value"].shift(-TARGET_HORIZON_DAYS) / df["portfolio_value"] - 1
    df["high_drawdown_risk"] = (df["future_5d_return"] <= DRAWDOWN_THRESHOLD).astype(int)

    keep_cols = ["date", *FEATURE_COLUMNS, "future_5d_return", "high_drawdown_risk"]
    output = df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return output


def engineer_features(input_path=None, output_path=PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Read raw data, create features, and save the processed dataset."""
    ensure_directories()
    input_path = input_path or (RAW_MARKET_DATA_PATH if RAW_MARKET_DATA_PATH.exists() else SYNTHETIC_DATA_PATH)
    raw_df = pd.read_csv(input_path)
    features = create_features(raw_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"Features saved to {output_path} with shape {features.shape}")
    return features


if __name__ == "__main__":
    engineer_features()

