"""Model performance Streamlit page."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import METRICS_PATH, MODEL_COMPARISON_PATH  # noqa: E402
from src.utils import load_json  # noqa: E402

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("Model Performance")

if MODEL_COMPARISON_PATH.exists():
    comparison = pd.read_csv(MODEL_COMPARISON_PATH)
    st.dataframe(comparison, use_container_width=True)
    metric_cols = [col for col in ["accuracy", "precision", "recall", "f1", "roc_auc"] if col in comparison]
    melted = comparison.melt(id_vars="model_name", value_vars=metric_cols, var_name="metric", value_name="value")
    st.plotly_chart(px.bar(melted, x="model_name", y="value", color="metric", barmode="group"), use_container_width=True)
else:
    st.info("Run `python run_project.py` to generate model comparison artifacts.")

metrics = load_json(METRICS_PATH, default={})
if metrics:
    st.subheader("Best Model Metrics")
    st.json(metrics)

