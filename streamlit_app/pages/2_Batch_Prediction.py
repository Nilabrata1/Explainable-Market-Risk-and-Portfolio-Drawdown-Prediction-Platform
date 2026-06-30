"""Batch prediction Streamlit page."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.predict import predict_batch  # noqa: E402

st.set_page_config(page_title="Batch Prediction", layout="wide")
st.title("Batch Portfolio Risk Scoring")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)
        predictions = predict_batch(tmp_path)
        tmp_path.unlink(missing_ok=True)
        st.dataframe(predictions, use_container_width=True)
        st.download_button("Download Predictions", predictions.to_csv(index=False), "portfolio_risk_predictions.csv", "text/csv")
    except FileNotFoundError as exc:
        st.error(str(exc))

