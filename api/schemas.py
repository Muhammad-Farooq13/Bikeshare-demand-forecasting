"""Pydantic request/response schemas for the demand forecasting API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DemandRequest(BaseModel):
    """Features required to predict bike rental demand for one hour."""

    temp: float = Field(..., ge=0, le=1, description="Normalized temperature (0-1)")
    atemp: float = Field(..., ge=0, le=1, description="Normalized 'feels like' temperature (0-1)")
    hum: float = Field(..., ge=0, le=1, description="Normalized humidity (0-1)")
    windspeed: float = Field(..., ge=0, le=1, description="Normalized windspeed (0-1)")
    lag_1: float = Field(..., ge=0, description="Actual rental count 1 hour ago")
    lag_24: float = Field(..., ge=0, description="Actual rental count 24 hours ago (same hour, previous day)")
    lag_168: float = Field(..., ge=0, description="Actual rental count 168 hours ago (same hour, previous week)")
    roll_mean_24: float = Field(..., ge=0, description="Rolling mean rental count over the prior 24 hours")
    roll_mean_168: float = Field(..., ge=0, description="Rolling mean rental count over the prior 168 hours")
    hour_sin: float = Field(..., description="Sine-encoded hour of day")
    hour_cos: float = Field(..., description="Cosine-encoded hour of day")
    month_sin: float = Field(..., description="Sine-encoded month")
    month_cos: float = Field(..., description="Cosine-encoded month")
    weekday_sin: float = Field(..., description="Sine-encoded day of week")
    weekday_cos: float = Field(..., description="Cosine-encoded day of week")
    season: Literal[1, 2, 3, 4] = Field(..., description="1=winter, 2=spring, 3=summer, 4=fall")
    weathersit: Literal[1, 2, 3, 4] = Field(..., description="1=clear ... 4=heavy rain/snow")
    holiday: int = Field(..., ge=0, le=1)
    workingday: int = Field(..., ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "temp": 0.5, "atemp": 0.48, "hum": 0.6, "windspeed": 0.2,
                "lag_1": 120, "lag_24": 140, "lag_168": 135,
                "roll_mean_24": 95.0, "roll_mean_168": 110.0,
                "hour_sin": 0.965, "hour_cos": -0.258,
                "month_sin": 0.5, "month_cos": 0.866,
                "weekday_sin": 0.781, "weekday_cos": 0.623,
                "season": 2, "weathersit": 1, "holiday": 0, "workingday": 1,
            }
        }
    }


class DemandResponse(BaseModel):
    """Predicted rental demand for the requested hour."""

    predicted_count: int = Field(..., description="Predicted number of bike rentals for the hour")


class HealthResponse(BaseModel):
    """Service health check response."""

    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
