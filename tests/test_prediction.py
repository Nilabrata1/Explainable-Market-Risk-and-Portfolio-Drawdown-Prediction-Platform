"""Prediction helper tests."""

from src.predict import probability_to_risk_level


def test_probability_to_risk_level():
    assert probability_to_risk_level(0.10) == "Low"
    assert probability_to_risk_level(0.50) == "Medium"
    assert probability_to_risk_level(0.80) == "High"

