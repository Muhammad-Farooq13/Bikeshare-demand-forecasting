"""Inference utilities for the demand forecasting model."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

import joblib
import pandas as pd

from src.config import load_config, resolve_path
from src.features import get_feature_columns
from src.logger import get_logger

logger = get_logger(__name__)


class DemandModel:
    """Wraps the fitted preprocessor and regressor for inference."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or load_config()
        model_cfg = self.config["model"]

        model_path = resolve_path(model_cfg["artifact_path"])
        preproc_path = resolve_path(model_cfg["preprocessor_path"])

        if not model_path.exists() or not preproc_path.exists():
            raise FileNotFoundError(
                "Model artifacts not found. Run `python -m src.train` (or `make train`) first."
            )

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preproc_path)
        self.feature_columns = get_feature_columns(self.config["features"])
        logger.info("Loaded model artifacts from %s", model_path)

    def predict_one(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict bike rental demand for a single hour given its features.

        Args:
            features: Dict of raw feature values keyed by feature name,
                including lag/rolling features (the caller is responsible
                for supplying recent actual demand history for those).

        Returns:
            Dict with the predicted rental count.
        """
        missing = [c for c in self.feature_columns if c not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        row = pd.DataFrame([{col: features[col] for col in self.feature_columns}])
        X = self.preprocessor.transform(row)
        prediction = float(self.model.predict(X)[0])
        prediction = max(0.0, prediction)  # demand can't be negative

        return {"predicted_count": round(prediction)}


@lru_cache(maxsize=1)
def get_model() -> DemandModel:
    """Return a process-wide cached ``DemandModel`` instance."""
    return DemandModel()
