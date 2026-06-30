"""Feature engineering tests."""

from src.feature_engineering import create_features
from src.synthetic_data import generate_synthetic_market_data


def test_feature_engineering_returns_target_column():
    raw = generate_synthetic_market_data(n_rows=400)
    features = create_features(raw)
    assert "high_drawdown_risk" in features.columns
    assert "future_5d_return" in features.columns
    assert features["high_drawdown_risk"].isin([0, 1]).all()
    assert len(features) > 250

