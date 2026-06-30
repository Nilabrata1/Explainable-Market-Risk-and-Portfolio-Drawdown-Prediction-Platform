"""Run the full data, training, and explainability pipeline."""

from __future__ import annotations

from src.config import DATA_SOURCE_PATH, MODEL_METADATA_PATH, ensure_directories
from src.data_collection import collect_market_data
from src.explain import create_explainability_artifacts
from src.feature_engineering import engineer_features
from src.train import train_models
from src.utils import load_json


def main() -> None:
    """Execute the complete project pipeline."""
    ensure_directories()
    print("Starting portfolio drawdown-risk ML pipeline...")
    collect_market_data()
    engineer_features()
    train_models()
    create_explainability_artifacts()

    metadata = load_json(MODEL_METADATA_PATH)
    source = DATA_SOURCE_PATH.read_text(encoding="utf-8").strip() if DATA_SOURCE_PATH.exists() else "unknown"
    print("\nPipeline completed successfully.")
    print(f"Dataset source used: {source}")
    print(f"Best model: {metadata.get('model_name', 'unknown')}")
    print(f"Metrics: {metadata.get('metrics', {})}")
    print("\nNext commands:")
    print("  uvicorn api.main:app --reload --port 8000")
    print("  streamlit run streamlit_app/app.py")
    print("  mlflow ui --backend-store-uri ./mlruns --port 5000")


if __name__ == "__main__":
    main()

