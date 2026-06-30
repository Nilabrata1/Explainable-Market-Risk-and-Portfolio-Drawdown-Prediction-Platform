"""Train and compare portfolio drawdown-risk models."""

from __future__ import annotations

import os
import tempfile
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None

try:
    from src.config import (
        BEST_MODEL_PATH,
        METRICS_PATH,
        MLRUNS_DIR,
        MODEL_COMPARISON_PATH,
        MODEL_METADATA_PATH,
        PREPROCESSOR_PATH,
        PROCESSED_DATA_PATH,
        RANDOM_STATE,
        ensure_directories,
    )
    from src.evaluate import classification_metrics, selection_score
    from src.feature_engineering import FEATURE_COLUMNS
    from src.utils import save_json
except ImportError:  # pragma: no cover
    from config import BEST_MODEL_PATH, METRICS_PATH, MLRUNS_DIR, MODEL_COMPARISON_PATH, MODEL_METADATA_PATH, PREPROCESSOR_PATH, PROCESSED_DATA_PATH, RANDOM_STATE, ensure_directories
    from evaluate import classification_metrics, selection_score
    from feature_engineering import FEATURE_COLUMNS
    from utils import save_json


def _optional_model_factories() -> dict[str, Any]:
    """Return model factories, skipping optional libraries when unavailable."""
    factories: dict[str, Any] = {
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    }
    try:
        from xgboost import XGBClassifier

        factories["XGBoost"] = lambda: XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.045,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"XGBoost unavailable and will be skipped: {exc}")
    try:
        from lightgbm import LGBMClassifier

        factories["LightGBM"] = lambda: LGBMClassifier(
            n_estimators=240,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"LightGBM unavailable and will be skipped: {exc}")
    try:
        from catboost import CatBoostClassifier

        factories["CatBoost"] = lambda: CatBoostClassifier(
            iterations=220,
            depth=5,
            learning_rate=0.045,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"CatBoost unavailable and will be skipped: {exc}")
    return factories


def _predict_probability(model: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities with a decision fallback."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    scores = model.decision_function(x_test)
    return 1 / (1 + np.exp(-scores))


def train_models(data_path=PROCESSED_DATA_PATH) -> tuple[Pipeline, dict]:
    """Train candidate models with a time-based split and persist the winner."""
    ensure_directories()
    df = pd.read_csv(data_path).sort_values("date").reset_index(drop=True)
    x = df[FEATURE_COLUMNS]
    y = df["high_drawdown_risk"].astype(int)

    split_idx = int(len(df) * 0.8)
    x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("The train/test split must contain both classes. Generate more data or adjust the threshold.")

    if mlflow:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(f"file:{MLRUNS_DIR.resolve()}")
        mlflow.set_experiment("portfolio-drawdown-risk")

    rows: list[dict[str, Any]] = []
    best_model: Pipeline | None = None
    best_metadata: dict[str, Any] = {}
    best_score = -np.inf

    for model_name, factory in _optional_model_factories().items():
        estimator = factory()
        pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])

        run_context = mlflow.start_run(run_name=model_name) if mlflow else None
        if run_context:
            run_context.__enter__()
        try:
            pipeline.fit(x_train, y_train)
            y_prob = _predict_probability(pipeline, x_test)
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = classification_metrics(y_test, y_pred, y_prob)
            score = selection_score(metrics)

            if mlflow:
                mlflow.log_params({"model_name": model_name, "split": "time_based_80_20", "features": len(FEATURE_COLUMNS)})
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float) and not np.isnan(v)})
                try:
                    mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(f"MLflow native model logging failed; logging Joblib artifact instead: {exc}")
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        artifact_path = os.path.join(tmp_dir, f"{model_name}_pipeline.pkl")
                        joblib.dump(pipeline, artifact_path)
                        mlflow.log_artifact(artifact_path, artifact_path="model")

            row = {"model_name": model_name, **metrics, "selection_score": score}
            rows.append(row)
            print(f"{model_name}: ROC-AUC={metrics.get('roc_auc'):.4f}, F1={metrics.get('f1'):.4f}")

            if score > best_score:
                best_score = score
                best_model = pipeline
                best_metadata = {
                    "model_name": model_name,
                    "feature_columns": FEATURE_COLUMNS,
                    "metrics": metrics,
                    "train_rows": int(len(x_train)),
                    "test_rows": int(len(x_test)),
                    "target": "high_drawdown_risk",
                    "selection_metric": "roc_auc_if_available_else_f1",
                }
        finally:
            if run_context:
                run_context.__exit__(None, None, None)

    if best_model is None:
        raise RuntimeError("No model could be trained. Install scikit-learn and at least one supported estimator.")

    comparison = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(best_model.named_steps["imputer"], PREPROCESSOR_PATH)
    save_json(best_metadata, MODEL_METADATA_PATH)
    save_json(best_metadata["metrics"], METRICS_PATH)
    print(f"Best model: {best_metadata['model_name']} saved to {BEST_MODEL_PATH}")
    return best_model, best_metadata


if __name__ == "__main__":
    train_models()
