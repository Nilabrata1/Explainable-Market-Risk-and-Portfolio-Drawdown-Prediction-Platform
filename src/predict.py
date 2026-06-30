"""Prediction interface for portfolio drawdown risk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

try:
    from src.config import BEST_MODEL_PATH, MODEL_METADATA_PATH
    from src.explain import top_risk_drivers
    from src.utils import load_json
except ImportError:  # pragma: no cover
    from config import BEST_MODEL_PATH, MODEL_METADATA_PATH
    from explain import top_risk_drivers
    from utils import load_json


RECOMMENDATIONS = {
    "Low": "Maintain current exposure and monitor.",
    "Medium": "Review risk concentration and rebalance if needed.",
    "High": "Reduce high-beta exposure, increase defensive allocation, and investigate volatility drivers.",
}


def probability_to_risk_level(probability: float) -> str:
    """Map drawdown-risk probability to a business risk band."""
    if probability < 0.35:
        return "Low"
    if probability < 0.65:
        return "Medium"
    return "High"


def _load_model_and_metadata():
    if not BEST_MODEL_PATH.exists() or not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError("Model is not trained yet. Please run: python run_project.py")
    return joblib.load(BEST_MODEL_PATH), load_json(MODEL_METADATA_PATH)


def predict_single(feature_values: dict[str, Any]) -> dict[str, Any]:
    """Predict risk for one feature dictionary."""
    model, metadata = _load_model_and_metadata()
    feature_columns = metadata["feature_columns"]
    x = pd.DataFrame([{feature: feature_values.get(feature, 0) for feature in feature_columns}])
    probability = float(model.predict_proba(x)[0, 1])
    prediction = int(probability >= 0.5)
    risk_level = probability_to_risk_level(probability)
    return {
        "prediction": prediction,
        "drawdown_probability": probability,
        "risk_level": risk_level,
        "top_risk_drivers": top_risk_drivers(x.iloc[0].to_dict(), top_n=5),
        "recommended_action": RECOMMENDATIONS[risk_level],
        "model_name": metadata.get("model_name", "unknown"),
    }


def predict_batch(csv_path: str | Path) -> pd.DataFrame:
    """Predict risk for every row in a CSV file."""
    model, metadata = _load_model_and_metadata()
    feature_columns = metadata["feature_columns"]
    df = pd.read_csv(csv_path)
    x = df.reindex(columns=feature_columns, fill_value=0)
    probabilities = model.predict_proba(x)[:, 1]
    result = df.copy()
    result["prediction"] = (probabilities >= 0.5).astype(int)
    result["drawdown_probability"] = probabilities
    result["risk_level"] = [probability_to_risk_level(float(prob)) for prob in probabilities]
    result["recommended_action"] = result["risk_level"].map(RECOMMENDATIONS)
    result["model_name"] = metadata.get("model_name", "unknown")
    return result

