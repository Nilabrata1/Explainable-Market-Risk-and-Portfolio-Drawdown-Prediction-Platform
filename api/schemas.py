"""Pydantic schemas for the FastAPI service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RiskPredictionRequest(BaseModel):
    """Flexible feature payload for a single prediction."""

    features: dict[str, float] = Field(..., description="Dictionary of feature name to numeric value.")


class RiskPredictionResponse(BaseModel):
    """Prediction response returned by the API."""

    prediction: int
    drawdown_probability: float
    risk_level: str
    top_risk_drivers: list[dict[str, Any]]
    recommended_action: str
    model_name: str

