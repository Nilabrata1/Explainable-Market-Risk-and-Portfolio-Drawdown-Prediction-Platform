"""Synthetic financial time-series generation for offline operation."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from src.config import DATA_SOURCE_PATH, RANDOM_STATE, SYNTHETIC_DATA_PATH, ensure_directories
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DATA_SOURCE_PATH, RANDOM_STATE, SYNTHETIC_DATA_PATH, ensure_directories


def generate_synthetic_market_data(n_rows: int = 3_500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate realistic synthetic market and portfolio data.

    The generator combines market regimes, correlated asset returns, volatility
    clustering, volume spikes, yield changes, and occasional drawdown shocks.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_rows)

    regime = np.zeros(n_rows, dtype=int)
    for i in range(1, n_rows):
        if rng.random() < 0.018:
            regime[i] = rng.choice([0, 1, 2], p=[0.55, 0.30, 0.15])
        else:
            regime[i] = regime[i - 1]

    drift = np.choose(regime, [0.00045, -0.00025, -0.0014])
    vol = np.choose(regime, [0.008, 0.015, 0.032])
    market_noise = rng.normal(drift, vol)

    shock_days = rng.choice(np.arange(30, n_rows - 30), size=max(12, n_rows // 180), replace=False)
    for day in shock_days:
        length = int(rng.integers(3, 11))
        shock = rng.uniform(-0.055, -0.025)
        market_noise[day : day + length] += shock * np.exp(-np.arange(length) / 4)
        regime[day : day + length] = 2

    volatility_proxy = 16 + pd.Series(market_noise).rolling(20, min_periods=1).std().to_numpy() * 900
    volatility_proxy += np.where(regime == 2, 20, np.where(regime == 1, 7, 0))
    volatility_proxy += rng.normal(0, 1.8, n_rows)
    volatility_proxy = np.clip(volatility_proxy, 8, 85)

    yield_change = rng.normal(0.00002, 0.0025, n_rows) - 0.10 * market_noise
    rate_level = 0.025 + np.cumsum(yield_change) / 8

    asset_betas = np.array([1.00, 1.18, 0.92, 1.28, 1.35, 1.20, 1.10, 0.75, -0.15, -0.35])
    idiosyncratic_vol = np.array([0.003, 0.006, 0.004, 0.008, 0.011, 0.010, 0.009, 0.012, 0.007, 0.006])
    asset_returns = {}
    tickers = ["SPY", "QQQ", "DIA", "IWM", "AAPL", "MSFT", "JPM", "XOM", "GLD", "TLT"]
    for ticker, beta, eps_vol in zip(tickers, asset_betas, idiosyncratic_vol):
        eps = rng.normal(0, eps_vol, n_rows)
        asset_returns[ticker] = beta * market_noise + eps + rng.normal(0, 0.001, n_rows)

    weights = np.array([0.18, 0.14, 0.10, 0.08, 0.12, 0.12, 0.08, 0.06, 0.06, 0.06])
    returns_matrix = np.column_stack([asset_returns[t] for t in tickers])
    portfolio_return = returns_matrix @ weights
    portfolio_return += np.where(volatility_proxy > 35, -0.0018, 0.0002)

    market_index = 100 * np.cumprod(1 + market_noise)
    portfolio_value = 100 * np.cumprod(1 + portfolio_return)
    base_volume = 90_000_000 * (1 + np.abs(market_noise) * 18 + np.maximum(volatility_proxy - 20, 0) / 45)
    volume = base_volume * rng.lognormal(mean=0, sigma=0.18, size=n_rows)

    price_cols = {}
    for ticker in tickers:
        price_cols[f"{ticker}_close"] = 100 * np.cumprod(1 + asset_returns[ticker])

    returns_df = pd.DataFrame(asset_returns)
    rolling_corr = (
        returns_df.rolling(40, min_periods=10)
        .corr(returns_df["SPY"])
        .groupby(level=0)
        .mean(numeric_only=True)["SPY"]
        .fillna(0.6)
        .to_numpy()
    )

    df = pd.DataFrame(
        {
            "date": dates,
            "market_index": market_index,
            "portfolio_value": portfolio_value,
            "market_return": market_noise,
            "portfolio_return": portfolio_return,
            "volume": volume,
            "volume_change": pd.Series(volume).pct_change().fillna(0).to_numpy(),
            "yield_change": yield_change,
            "rate_level": rate_level,
            "volatility_proxy": volatility_proxy,
            "market_correlation": rolling_corr,
            "regime": regime,
            **price_cols,
        }
    )
    return df


def save_synthetic_market_data(path=SYNTHETIC_DATA_PATH, n_rows: int = 3_500) -> pd.DataFrame:
    """Generate and save synthetic market data."""
    ensure_directories()
    df = generate_synthetic_market_data(n_rows=n_rows)
    df.to_csv(path, index=False)
    DATA_SOURCE_PATH.write_text("synthetic", encoding="utf-8")
    print(f"Synthetic market data saved to {path}")
    return df


if __name__ == "__main__":
    save_synthetic_market_data()

