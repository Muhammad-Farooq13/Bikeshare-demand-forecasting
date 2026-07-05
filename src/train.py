"""Model training pipeline for hourly bike-share demand forecasting.

Critically, this uses **time-based** splitting throughout — never a random
train/test split or random K-fold — because a random split on a time series
leaks future information into the training set (e.g. training on 3pm data
from the same day you're "predicting" 2pm for). Cross-validation uses
scikit-learn's ``TimeSeriesSplit`` (expanding-window CV), and the final
held-out test set is the most recent N days, exactly as a real deployment
would only ever have past data available to predict the future.

Run directly:
    python -m src.train
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.config import load_config, resolve_path
from src.features import build_features, build_preprocessor, split_X_y
from src.logger import get_logger

logger = get_logger(__name__)


def time_based_split(df: pd.DataFrame, test_size_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a chronologically sorted dataframe into train/test by date cutoff.

    Args:
        df: Feature-engineered dataframe, already sorted chronologically,
            with a ``dteday`` column.
        test_size_days: Number of most-recent days to hold out as the test set.

    Returns:
        (train_df, test_df) tuple.
    """
    cutoff = df["dteday"].max() - pd.Timedelta(days=test_size_days)
    train_df = df[df["dteday"] <= cutoff].reset_index(drop=True)
    test_df = df[df["dteday"] > cutoff].reset_index(drop=True)
    return train_df, test_df


def seasonal_naive_baseline(test_df: pd.DataFrame) -> np.ndarray:
    """Predict demand using the "same hour, one week ago" naive baseline.

    This is the honest baseline any real forecasting model must beat — if a
    trained model can't outperform "assume this week looks like last week",
    it isn't adding value.
    """
    return test_df["lag_168"].values


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard forecasting error metrics."""
    y_pred_clipped = np.clip(y_pred, 0, None)  # demand can't be negative
    return {
        "mae": float(mean_absolute_error(y_true, y_pred_clipped)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_clipped))),
        "mape": float(mean_absolute_percentage_error(np.maximum(y_true, 1), y_pred_clipped)),
    }


def load_processed_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load (or build, if missing) the feature-engineered dataset."""
    data_cfg = config["data"]
    processed_path = resolve_path(data_cfg["processed_path"])

    if processed_path.exists():
        df = pd.read_csv(processed_path, parse_dates=["dteday"])
        logger.info("Loaded processed features from %s", processed_path)
        return df

    raw_path = resolve_path(data_cfg["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError("Raw data not found — run `python -m src.ingest` first.")

    raw_df = pd.read_csv(raw_path)
    df = build_features(raw_df, config["features"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    return df


def train_and_evaluate(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train the demand forecasting model end-to-end and return metrics.

    Args:
        config: Full configuration dictionary.

    Returns:
        Dictionary of evaluation metrics, including the naive-baseline
        comparison, cross-validation scores, and held-out test metrics.
    """
    df = load_processed_data(config)
    feature_cfg = config["features"]
    split_cfg = config["split"]
    model_cfg = config["model"]

    train_df, test_df = time_based_split(df, split_cfg["test_size_days"])
    logger.info(
        "Time-based split — train: %d rows (%s to %s), test: %d rows (%s to %s)",
        len(train_df), train_df["dteday"].min().date(), train_df["dteday"].max().date(),
        len(test_df), test_df["dteday"].min().date(), test_df["dteday"].max().date(),
    )

    X_train, y_train = split_X_y(train_df, feature_cfg)
    X_test, y_test = split_X_y(test_df, feature_cfg)

    preprocessor = build_preprocessor(feature_cfg)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Expanding-window time-series cross-validation — NOT random K-fold.
    tscv = TimeSeriesSplit(n_splits=split_cfg["cv_n_splits"])
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_depth=8, random_state=model_cfg["random_seed"]
    )
    cv_scores = cross_val_score(model, X_train_t, y_train, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_mae_mean, cv_mae_std = float(-cv_scores.mean()), float(cv_scores.std())
    logger.info("Time-series CV (%d expanding-window splits) MAE: %.2f ± %.2f", split_cfg["cv_n_splits"], cv_mae_mean, cv_mae_std)

    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)
    gbm_metrics = compute_regression_metrics(y_test.values, y_pred)

    # Baseline 1: Ridge regression (linear) on the same features
    ridge = Ridge(alpha=1.0, random_state=model_cfg["random_seed"])
    ridge.fit(X_train_t, y_train)
    ridge_pred = ridge.predict(X_test_t)
    ridge_metrics = compute_regression_metrics(y_test.values, ridge_pred)

    # Baseline 2: seasonal naive ("same hour last week")
    naive_pred = seasonal_naive_baseline(test_df)
    naive_metrics = compute_regression_metrics(y_test.values, naive_pred)

    metrics = {
        "model": {**gbm_metrics, "cv_mae_mean": cv_mae_mean, "cv_mae_std": cv_mae_std},
        "ridge_baseline": ridge_metrics,
        "seasonal_naive_baseline": naive_metrics,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_date_range": [str(train_df["dteday"].min().date()), str(train_df["dteday"].max().date())],
        "test_date_range": [str(test_df["dteday"].min().date()), str(test_df["dteday"].max().date())],
        "improvement_over_naive_pct": round(
            100 * (naive_metrics["mae"] - gbm_metrics["mae"]) / naive_metrics["mae"], 2
        ),
    }

    logger.info(
        "Test MAE — Model: %.2f | Ridge: %.2f | Seasonal-naive: %.2f (model improves on naive by %.1f%%)",
        gbm_metrics["mae"], ridge_metrics["mae"], naive_metrics["mae"], metrics["improvement_over_naive_pct"],
    )

    _persist_artifacts(model, preprocessor, metrics, config)
    return metrics


def _persist_artifacts(model, preprocessor, metrics: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Save the fitted model, preprocessor, and metrics to disk."""
    import json

    model_cfg = config["model"]
    model_path = resolve_path(model_cfg["artifact_path"])
    preproc_path = resolve_path(model_cfg["preprocessor_path"])
    metrics_path = resolve_path(model_cfg["metrics_path"])

    for path in (model_path, preproc_path, metrics_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preproc_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved preprocessor to %s", preproc_path)
    logger.info("Saved metrics to %s", metrics_path)


def main() -> None:
    config = load_config()
    train_and_evaluate(config)


if __name__ == "__main__":
    main()
