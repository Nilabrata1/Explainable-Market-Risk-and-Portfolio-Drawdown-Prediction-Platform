"""Cloud-friendly helpers for the Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import BEST_MODEL_PATH, METRICS_PATH, MODEL_COMPARISON_PATH, MODEL_METADATA_PATH, SHAP_IMPORTANCE_PATH, SHAP_SUMMARY_PATH  # noqa: E402


REQUIRED_ARTIFACTS = [
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    SHAP_IMPORTANCE_PATH,
    SHAP_SUMMARY_PATH,
]


def artifacts_ready() -> bool:
    """Return True when all Streamlit dashboard artifacts are available."""
    return all(path.exists() for path in REQUIRED_ARTIFACTS)


def missing_artifacts() -> list[str]:
    """List missing dashboard artifact paths relative to the project root."""
    missing = []
    for path in REQUIRED_ARTIFACTS:
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
    return missing


def build_demo_artifacts() -> None:
    """Run the project pipeline to create model and report artifacts."""
    from src.data_collection import collect_market_data
    from src.explain import create_explainability_artifacts
    from src.feature_engineering import engineer_features
    from src.train import train_models

    collect_market_data()
    engineer_features()
    train_models()
    create_explainability_artifacts()

