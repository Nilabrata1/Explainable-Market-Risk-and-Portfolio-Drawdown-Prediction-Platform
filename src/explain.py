"""Explainability utilities using SHAP with robust fallbacks."""

from __future__ import annotations

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from src.config import BEST_MODEL_PATH, MODEL_METADATA_PATH, PROCESSED_DATA_PATH, SHAP_IMPORTANCE_PATH, SHAP_SUMMARY_PATH, ensure_directories
    from src.utils import load_json
except ImportError:  # pragma: no cover
    from config import BEST_MODEL_PATH, MODEL_METADATA_PATH, PROCESSED_DATA_PATH, SHAP_IMPORTANCE_PATH, SHAP_SUMMARY_PATH, ensure_directories
    from utils import load_json


def _extract_importance(model, x: pd.DataFrame) -> pd.DataFrame:
    """Use native feature importances when SHAP is unavailable."""
    estimator = model.named_steps.get("model", model)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        importances = np.abs(x.corrwith(pd.Series(model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.predict(x)))).fillna(0).to_numpy()
    table = pd.DataFrame({"feature": x.columns, "importance": np.asarray(importances, dtype=float)})
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def create_explainability_artifacts(sample_size: int = 600) -> pd.DataFrame:
    """Create SHAP summary artifacts or fallback feature-importance artifacts."""
    ensure_directories()
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError("Model artifact not found. Run python run_project.py first.")

    model = joblib.load(BEST_MODEL_PATH)
    metadata = load_json(MODEL_METADATA_PATH)
    feature_columns = metadata.get("feature_columns", [])
    df = pd.read_csv(PROCESSED_DATA_PATH)
    x = df[feature_columns].tail(sample_size)

    try:
        import shap

        transformed = model.named_steps["imputer"].transform(x) if hasattr(model, "named_steps") else x
        estimator = model.named_steps.get("model", model)
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        if getattr(shap_values, "ndim", 2) == 3:
            shap_values = shap_values[:, :, -1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = pd.DataFrame({"feature": feature_columns, "importance": mean_abs}).sort_values("importance", ascending=False)
        shap.summary_plot(shap_values, pd.DataFrame(transformed, columns=feature_columns), show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(SHAP_SUMMARY_PATH, dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"SHAP failed; falling back to model feature importances: {exc}")
        importance = _extract_importance(model, x)
        plt.figure(figsize=(9, 6))
        top = importance.head(15).iloc[::-1]
        plt.barh(top["feature"], top["importance"], color="#2f6f73")
        plt.title("Fallback Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(SHAP_SUMMARY_PATH, dpi=180, bbox_inches="tight")
        plt.close()

    importance.to_csv(SHAP_IMPORTANCE_PATH, index=False)
    print(f"Explainability artifacts saved to {SHAP_IMPORTANCE_PATH} and {SHAP_SUMMARY_PATH}")
    return importance


def top_risk_drivers(feature_values: dict, top_n: int = 5) -> list[dict]:
    """Return the top risk drivers for one prediction using saved importances."""
    metadata = load_json(MODEL_METADATA_PATH)
    feature_columns = metadata.get("feature_columns", [])
    if SHAP_IMPORTANCE_PATH.exists():
        importance = pd.read_csv(SHAP_IMPORTANCE_PATH).set_index("feature")["importance"].to_dict()
    elif BEST_MODEL_PATH.exists():
        model = joblib.load(BEST_MODEL_PATH)
        dummy = pd.DataFrame([feature_values], columns=feature_columns).fillna(0)
        importance = _extract_importance(model, dummy).set_index("feature")["importance"].to_dict()
    else:
        importance = {feature: 1.0 for feature in feature_columns}

    rows = []
    for feature in feature_columns:
        value = float(feature_values.get(feature, 0) or 0)
        weight = float(importance.get(feature, 0))
        rows.append({"feature": feature, "value": value, "importance": weight, "driver_score": abs(value) * weight})
    return sorted(rows, key=lambda item: item["driver_score"], reverse=True)[:top_n]


if __name__ == "__main__":
    create_explainability_artifacts()

