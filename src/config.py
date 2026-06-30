"""Central project configuration."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

RAW_MARKET_DATA_PATH = RAW_DATA_DIR / "market_data.csv"
SYNTHETIC_DATA_PATH = RAW_DATA_DIR / "synthetic_market_data.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "features.csv"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
SHAP_SUMMARY_PATH = FIGURES_DIR / "shap_summary.png"
SHAP_IMPORTANCE_PATH = REPORTS_DIR / "shap_feature_importance.csv"
DATA_SOURCE_PATH = RAW_DATA_DIR / "data_source.txt"

TICKERS = ["SPY", "QQQ", "DIA", "IWM", "AAPL", "MSFT", "JPM", "XOM", "GLD", "TLT"]
START_DATE = "2012-01-01"
TARGET_HORIZON_DAYS = 5
DRAWDOWN_THRESHOLD = -0.03
RANDOM_STATE = 42


def ensure_directories() -> None:
    """Create all runtime directories used by the project."""
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        MLRUNS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

