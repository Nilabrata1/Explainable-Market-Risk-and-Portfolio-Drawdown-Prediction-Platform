# Explainable Market Risk and Portfolio Drawdown Prediction Platform

An end-to-end machine learning platform that predicts whether a stock portfolio is likely to enter a high drawdown-risk zone over the next 5 trading days.

## Business Problem

Risk managers, portfolio analysts, and investment teams need an explainable early-warning system for short-horizon drawdown risk. This project classifies whether a portfolio is likely to experience a future 5-day return less than or equal to -3%.

## ML Problem Formulation

- Task: supervised binary classification
- Target: `high_drawdown_risk`
- Positive class: `future_5d_return <= -0.03`
- Negative class: `future_5d_return > -0.03`
- Split: time-based 80% train, 20% test
- Selection metric: best ROC-AUC, with F1 fallback

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Random Forest, XGBoost, LightGBM, CatBoost, SHAP, FastAPI, Streamlit, Docker, MLflow, Joblib, Matplotlib, Plotly, yfinance, Pytest.

## Dataset Strategy

The data collection script follows a resilient fallback order:

1. Download OHLCV data with `yfinance` for SPY, QQQ, DIA, IWM, AAPL, MSFT, JPM, XOM, GLD, and TLT from 2012-01-01 to the latest available date.
2. Try Kaggle only when credentials are available in the environment or `~/.kaggle/kaggle.json`.
3. Generate a realistic synthetic financial time-series dataset with market regimes, correlated assets, volatility clustering, yield changes, volume changes, and drawdown events.

The project is designed to work without internet access.

## Folder Structure

```text
portfolio-risk-mlops-platform/
  data/
    raw/
    processed/
  notebooks/
  src/
  api/
  streamlit_app/
  models/
  reports/
  tests/
  mlruns/
  Dockerfile
  docker-compose.yml
  requirements.txt
  requirements-full.txt
  run_project.py
```

## How to Run Locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-full.txt
python run_project.py
pytest
```

## FastAPI

```bash
uvicorn api.main:app --reload --port 8000
```

Endpoints:

- `GET /`
- `GET /health`
- `POST /predict-risk`
- `POST /batch-predict`
- `GET /model-performance`
- `GET /feature-importance`

Example request:

```json
{
  "features": {
    "daily_return": 0.001,
    "portfolio_return": 0.001,
    "5_day_return": 0.004,
    "10_day_return": 0.008,
    "20_day_return": 0.012,
    "rolling_volatility_5": 0.012,
    "rolling_volatility_10": 0.014,
    "rolling_volatility_20": 0.016,
    "moving_average_5": 120,
    "moving_average_20": 118,
    "moving_average_ratio": 0.01,
    "momentum_5": 0.004,
    "momentum_10": 0.008,
    "momentum_20": 0.012,
    "max_drawdown_20": -0.02,
    "volume_change": 0.05,
    "volatility_proxy": 24,
    "yield_change": 0.0002,
    "market_correlation": 0.65,
    "beta_proxy": 1.05
  }
}
```

## Streamlit Dashboard

```bash
streamlit run streamlit_app/app.py
```

The dashboard includes the business problem, dataset source, model comparison, single prediction, batch prediction, model performance, and explainability views.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud and choose **Create app**.
3. Select the GitHub repository, branch, and set the main file path to:

```text
streamlit_app/app.py
```

4. Deploy the app.

This repository keeps `requirements.txt` lean for Streamlit Cloud and `requirements-full.txt` for the full local/API/MLflow stack. It also includes `runtime.txt`, a root `.streamlit/config.toml`, and lightweight model/report artifacts for an immediately usable dashboard. If artifacts are missing in a fresh deployment, the home page includes a **Build demo model artifacts** button.

## MLflow Tracking

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Each model run logs parameters, metrics, and the model artifact when MLflow is installed.

## Docker

```bash
docker compose up --build
```

Services:

- FastAPI: http://localhost:8000
- Streamlit: http://localhost:8501
- MLflow: http://localhost:5000

## Model Training Details

The pipeline compares:

- RandomForestClassifier
- XGBClassifier
- LGBMClassifier
- CatBoostClassifier

Metrics saved to `reports/metrics.json` and `reports/model_comparison.csv`:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

The best model is saved to `models/best_model.pkl`, and metadata is saved to `models/model_metadata.json`.

## SHAP Explainability

`src/explain.py` generates:

- `reports/figures/shap_summary.png`
- `reports/shap_feature_importance.csv`

If SHAP fails for the selected estimator, the project falls back to native model feature importances.

## Resume Bullets

Explainable Market Risk and Portfolio Drawdown Prediction Platform | Python, XGBoost, LightGBM, CatBoost, SHAP, FastAPI, MLflow

- Built an end-to-end portfolio drawdown risk prediction platform using Random Forest, XGBoost, LightGBM, and CatBoost to classify low and high-risk market regimes from price, volatility, momentum, and macro-style features.
- Engineered financial time-series features including rolling volatility, momentum, moving-average ratios, beta proxy, correlation, and future drawdown labels while preventing data leakage through time-based train-test splitting.
- Integrated MLflow for experiment tracking and SHAP for model explainability, then deployed the best model using FastAPI, Streamlit, and Docker for real-time portfolio risk scoring and dashboard-based analysis.

## Limitations

- The target is based on a fixed -3% 5-day drawdown threshold and may need calibration for different portfolio mandates.
- Synthetic data is useful for demonstration but is not a substitute for validated production market data.
- The platform does not execute trades or portfolio optimization decisions.

## Future Improvements

- Add portfolio holdings ingestion and position-level factor attribution.
- Add walk-forward validation and probability calibration.
- Add drift monitoring and scheduled retraining.
- Add authenticated API access and persistent model registry promotion.
