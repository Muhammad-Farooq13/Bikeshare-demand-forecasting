"""Unit tests for time-series feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config, resolve_path
from src.features import build_features, build_preprocessor, split_X_y

RAW_DATA_PATH = resolve_path("data/hour_raw.csv")

pytestmark = pytest.mark.skipif(
    not RAW_DATA_PATH.exists(), reason="Raw data not found — run `make ingest` first."
)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def raw_df():
    return pd.read_csv(RAW_DATA_PATH)


@pytest.fixture(scope="module")
def features_df(raw_df, config):
    return build_features(raw_df, config["features"])


def test_build_features_no_nulls(features_df):
    assert features_df.isnull().sum().sum() == 0


def test_build_features_drops_early_rows(raw_df, features_df):
    # Rows requiring 168h (1 week) of lag history must be dropped
    assert len(features_df) < len(raw_df)
    assert len(features_df) >= len(raw_df) - 168 - 10


def test_lag_features_correctness(features_df):
    # lag_1 at row i should equal cnt at row i-1 (post-sort, pre-dropna alignment
    # is validated indirectly: lag_1 should never exceed plausible bounds and
    # should correlate strongly with the target)
    correlation = features_df["cnt"].corr(features_df["lag_1"])
    assert correlation > 0.7


def test_cyclical_encoding_bounds(features_df):
    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos", "weekday_sin", "weekday_cos"]:
        assert features_df[col].between(-1.0, 1.0).all()


def test_cyclical_encoding_continuity():
    """Hour 23 and hour 0 should be close in cyclical space, unlike raw integers."""
    from src.features import _cyclical_encode

    hours = pd.Series([0, 23], name="hr")
    encoded = _cyclical_encode(hours, 24, name="hour")
    dist = np.sqrt((encoded.iloc[0] - encoded.iloc[1]).pow(2).sum())
    assert dist < 0.3  # much closer than |23 - 0| = 23 in raw integer space


def test_split_X_y_shapes(features_df, config):
    X, y = split_X_y(features_df, config["features"])
    assert len(X) == len(y) == len(features_df)
    assert "cnt" not in X.columns


def test_build_preprocessor_fit_transform(features_df, config):
    X, _ = split_X_y(features_df, config["features"])
    preprocessor = build_preprocessor(config["features"])
    X_t = preprocessor.fit_transform(X)
    assert X_t.shape[0] == len(X)
    assert X_t.shape[1] > len(config["features"]["numeric"])  # one-hot expands categoricals
