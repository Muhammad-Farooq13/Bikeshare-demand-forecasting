"""Time-series feature engineering for hourly demand forecasting.

Builds lag features, rolling means, and cyclical (sin/cos) encodings of
hour/month/weekday — the standard feature set for tree-based demand
forecasting models, as opposed to naively feeding raw integer hour/month
values (which implies hour 23 and hour 0 are far apart, when they're
actually adjacent).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_config, resolve_path
from src.logger import get_logger

logger = get_logger(__name__)


def _cyclical_encode(series: pd.Series, period: int, name: str | None = None) -> pd.DataFrame:
    """Encode a periodic integer feature (hour, month, weekday) as sin/cos pairs.

    Args:
        series: The integer series to encode (e.g. hour of day).
        period: The period of the cycle (24 for hour, 12 for month, 7 for weekday).
        name: Output column name prefix. Defaults to the series' own name if
            not given (useful when the raw column name differs from the
            desired feature name, e.g. raw "hr" -> feature "hour").
    """
    label = name or series.name
    radians = 2 * np.pi * series / period
    return pd.DataFrame({f"{label}_sin": np.sin(radians), f"{label}_cos": np.cos(radians)})


def build_features(df: pd.DataFrame, feature_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Engineer lag, rolling, and cyclical features from the raw hourly dataset.

    Args:
        df: Raw hourly dataframe (must be sorted chronologically already, or
            will be sorted here by ``dteday`` + ``hr``).
        feature_cfg: The "features" section of the config.

    Returns:
        DataFrame with all engineered feature columns plus the target, with
        rows containing NaN lag/rolling values (the earliest rows, before
        enough history exists) dropped.
    """
    df = df.copy()
    df["dteday"] = pd.to_datetime(df["dteday"])
    df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)

    target = feature_cfg["target"]

    for lag in feature_cfg["lag_hours"]:
        df[f"lag_{lag}"] = df[target].shift(lag)

    for window in feature_cfg["rolling_windows"]:
        df[f"roll_mean_{window}"] = df[target].shift(1).rolling(window=window, min_periods=window).mean()

    df = pd.concat([df, _cyclical_encode(df["hr"], 24, name="hour")], axis=1)
    df = pd.concat([df, _cyclical_encode(df["mnth"], 12, name="month")], axis=1)
    df = pd.concat([df, _cyclical_encode(df["weekday"], 7, name="weekday")], axis=1)

    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info("Dropped %d rows with insufficient lag/rolling history", n_before - len(df))

    return df


def build_preprocessor(feature_cfg: Dict[str, Any]) -> ColumnTransformer:
    """Build the feature preprocessing pipeline (scaling + one-hot encoding).

    Args:
        feature_cfg: The "features" section of the config.

    Returns:
        A ``ColumnTransformer`` ready to be fit on the training features.
    """
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, feature_cfg["numeric"]),
            ("categorical", categorical_pipeline, feature_cfg["categorical"]),
            ("binary", "passthrough", feature_cfg["binary"]),
        ]
    )
    return preprocessor


def get_output_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Get human-readable feature names after transformation (for SHAP/explainability)."""
    return list(preprocessor.get_feature_names_out())


def get_feature_columns(feature_cfg: Dict[str, Any]) -> List[str]:
    """Flatten the feature config into a single ordered list of input columns."""
    return feature_cfg["numeric"] + feature_cfg["categorical"] + feature_cfg["binary"]


def split_X_y(df: pd.DataFrame, feature_cfg: Dict[str, Any]):
    """Split a feature-engineered dataframe into X (features) and y (target)."""
    columns = get_feature_columns(feature_cfg)
    X = df[columns].copy()
    y = df[feature_cfg["target"]].copy()
    return X, y


def main() -> None:
    config = load_config()
    data_cfg = config["data"]
    raw_path = resolve_path(data_cfg["raw_path"])

    if not raw_path.exists():
        raise FileNotFoundError("Raw data not found — run `python -m src.ingest` first.")

    df = pd.read_csv(raw_path)
    features_df = build_features(df, config["features"])

    out_path = resolve_path(data_cfg["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(out_path, index=False)
    logger.info("Saved %d feature-engineered rows to %s", len(features_df), out_path)


if __name__ == "__main__":
    main()
