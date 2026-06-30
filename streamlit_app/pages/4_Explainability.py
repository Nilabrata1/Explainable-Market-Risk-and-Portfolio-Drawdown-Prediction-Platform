"""Explainability Streamlit page."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import SHAP_IMPORTANCE_PATH, SHAP_SUMMARY_PATH  # noqa: E402

st.set_page_config(page_title="Explainability", layout="wide")
st.title("Model Explainability")

if SHAP_SUMMARY_PATH.exists():
    st.image(str(SHAP_SUMMARY_PATH), caption="SHAP summary or fallback feature-importance plot")
else:
    st.info("Run `python run_project.py` to generate explainability artifacts.")

if SHAP_IMPORTANCE_PATH.exists():
    importance = pd.read_csv(SHAP_IMPORTANCE_PATH)
    st.dataframe(importance, use_container_width=True)
    st.plotly_chart(px.bar(importance.head(15), x="importance", y="feature", orientation="h"), use_container_width=True)

