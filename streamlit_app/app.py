"""Streamlit dashboard landing page."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import DATA_SOURCE_PATH, METRICS_PATH, MODEL_COMPARISON_PATH, MODEL_METADATA_PATH  # noqa: E402
from src.utils import load_json  # noqa: E402
from streamlit_app.bootstrap import artifacts_ready, build_demo_artifacts, missing_artifacts  # noqa: E402

st.set_page_config(page_title="Portfolio Drawdown Risk", layout="wide")

st.title("Explainable Market Risk and Portfolio Drawdown Prediction Platform")
st.caption("Machine learning system for 5-day high drawdown-risk classification")

if not artifacts_ready():
    st.warning("Model artifacts are not available in this deployment yet.")
    with st.expander("Missing artifacts", expanded=False):
        st.code("\n".join(missing_artifacts()))
    if st.button("Build demo model artifacts", type="primary"):
        with st.spinner("Building demo artifacts. This can take a few minutes on Streamlit Cloud."):
            build_demo_artifacts()
        st.success("Artifacts built. Refreshing dashboard...")
        st.rerun()

source = DATA_SOURCE_PATH.read_text(encoding="utf-8").strip() if DATA_SOURCE_PATH.exists() else "not generated yet"
metadata = load_json(MODEL_METADATA_PATH, default={})
metrics = load_json(METRICS_PATH, default={})

col1, col2, col3 = st.columns(3)
col1.metric("Dataset Source", source)
col2.metric("Best Model", metadata.get("model_name", "Not trained"))
col3.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}" if metrics else "N/A")

st.subheader("Business Problem")
st.write(
    "Portfolio managers and risk teams need early warnings when market conditions imply elevated probability "
    "of a material short-horizon drawdown. This platform classifies whether a portfolio is likely to lose more "
    "than 3% over the next 5 trading days."
)

st.subheader("ML Formulation")
st.write(
    "The target is `high_drawdown_risk`: 1 when future 5-day portfolio return is less than or equal to -3%, "
    "otherwise 0. Features use only present and historical price, volatility, momentum, macro-like, volume, "
    "correlation, and beta signals."
)

if MODEL_COMPARISON_PATH.exists():
    st.subheader("Model Comparison")
    st.dataframe(pd.read_csv(MODEL_COMPARISON_PATH), use_container_width=True)
else:
    st.info("Run `python run_project.py` to generate model artifacts.")
