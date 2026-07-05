"""FastAPI service exposing the hourly demand forecasting model.

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import DemandRequest, DemandResponse, HealthResponse
from src.logger import get_logger
from src.predict import DemandModel, get_model

logger = get_logger(__name__)

_model_load_error: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the model cache at startup."""
    global _model_load_error
    try:
        get_model()
    except FileNotFoundError as exc:
        _model_load_error = str(exc)
        logger.warning("Model not available at startup: %s", exc)
    yield


app = FastAPI(
    title="Bike-Share Demand Forecasting API",
    description="Hourly rental demand prediction for bike-share fleet rebalancing.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness/readiness probe endpoint."""
    return HealthResponse(status="ok", model_loaded=_model_load_error is None)


@app.post("/predict", response_model=DemandResponse, tags=["inference"])
def predict(request: DemandRequest) -> DemandResponse:
    """Predict bike rental demand for one hour given weather/calendar/lag features."""
    try:
        model: DemandModel = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result = model.predict_one(request.model_dump())
    return DemandResponse(**result)


@app.get("/", tags=["ops"])
def root() -> dict:
    """Basic service info."""
    return {"service": "bikeshare-demand-forecasting-api", "status": "running", "docs": "/docs"}
