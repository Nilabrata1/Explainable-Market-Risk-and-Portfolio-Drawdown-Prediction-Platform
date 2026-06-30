"""Single prediction Streamlit page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.feature_engineering import FEATURE_COLUMNS  # noqa: E402
from src.predict import predict_single  # noqa: E402

st.set_page_config(page_title="Single Prediction", layout="wide")
st.title("Single Portfolio Risk Score")

defaults = {
    "daily_return": 0.001,
    "portfolio_return": 0.001,
    "5_day_return": 0.003,
    "10_day_return": 0.006,
    "20_day_return": 0.012,
    "rolling_volatility_5": 0.012,
    "rolling_volatility_10": 0.014,
    "rolling_volatility_20": 0.016,
    "moving_average_5": 120.0,
    "moving_average_20": 118.0,
    "moving_average_ratio": 0.01,
    "momentum_5": 0.003,
    "momentum_10": 0.006,
    "momentum_20": 0.012,
    "max_drawdown_20": -0.025,
    "volume_change": 0.05,
    "volatility_proxy": 24.0,
    "yield_change": 0.0002,
    "market_correlation": 0.65,
    "beta_proxy": 1.05,
}

values = {}
left, right = st.columns(2)
for idx, feature in enumerate(FEATURE_COLUMNS):
    container = left if idx % 2 == 0 else right
    values[feature] = container.number_input(feature, value=float(defaults.get(feature, 0.0)), format="%.6f")

if st.button("Score Risk", type="primary"):
    try:
        result = predict_single(values)
        st.metric("Drawdown Probability", f"{result['drawdown_probability']:.2%}")
        st.metric("Risk Level", result["risk_level"])
        st.write(result["recommended_action"])
        st.subheader("Top Risk Drivers")
        st.dataframe(result["top_risk_drivers"], use_container_width=True)
    except FileNotFoundError as exc:
        st.error(str(exc))

