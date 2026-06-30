"""FastAPI backend for portfolio drawdown-risk predictions."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from api.schemas import RiskPredictionRequest, RiskPredictionResponse  # noqa: E402
from src.config import METRICS_PATH, MODEL_COMPARISON_PATH, SHAP_IMPORTANCE_PATH  # noqa: E402
from src.predict import predict_batch, predict_single  # noqa: E402
from src.utils import load_json  # noqa: E402

app = FastAPI(title="Explainable Market Risk API", version="1.0.0")


@app.get("/")
def root() -> dict:
    return {"message": "Explainable Market Risk and Portfolio Drawdown Prediction API"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict-risk", response_model=RiskPredictionResponse)
def predict_risk(payload: RiskPredictionRequest):
    try:
        return predict_single(payload.features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename or "batch.csv").suffix or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        predictions = predict_batch(tmp_path)
        tmp_path.unlink(missing_ok=True)
        return JSONResponse(predictions.to_dict(orient="records"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/model-performance")
def model_performance():
    if MODEL_COMPARISON_PATH.exists():
        return pd.read_csv(MODEL_COMPARISON_PATH).to_dict(orient="records")
    if METRICS_PATH.exists():
        return load_json(METRICS_PATH)
    raise HTTPException(status_code=404, detail="Model performance artifacts not found. Run python run_project.py")


@app.get("/feature-importance")
def feature_importance():
    if not SHAP_IMPORTANCE_PATH.exists():
        raise HTTPException(status_code=404, detail="Feature importance not found. Run python run_project.py")
    return pd.read_csv(SHAP_IMPORTANCE_PATH).to_dict(orient="records")

