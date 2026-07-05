"""Tests for the training pipeline, with special attention to time-series
validation correctness — the most important thing to get right (and easiest
to get subtly wrong) in a forecasting project.
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from src.config import load_config, resolve_path
from src.features import build_features
from src.train import compute_regression_metrics, time_based_split, train_and_evaluate

RAW_DATA_PATH = resolve_path("data/hour_raw.csv")

pytestmark = pytest.mark.skipif(
    not RAW_DATA_PATH.exists(), reason="Raw data not found — run `make ingest` first."
)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def features_df(config):
    raw_df = pd.read_csv(RAW_DATA_PATH)
    return build_features(raw_df, config["features"])


def test_time_based_split_no_overlap(features_df):
    train_df, test_df = time_based_split(features_df, test_size_days=120)
    assert train_df["dteday"].max() < test_df["dteday"].min()


def test_time_based_split_test_is_most_recent(features_df):
    train_df, test_df = time_based_split(features_df, test_size_days=120)
    assert test_df["dteday"].max() == features_df["dteday"].max()


def test_time_based_split_no_row_lost_or_duplicated(features_df):
    train_df, test_df = time_based_split(features_df, test_size_days=120)
    assert len(train_df) + len(test_df) == len(features_df)


def test_compute_regression_metrics_perfect_prediction():
    import numpy as np

    y = np.array([10.0, 20.0, 30.0])
    metrics = compute_regression_metrics(y, y)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)


def test_compute_regression_metrics_clips_negative_predictions():
    import numpy as np

    y_true = np.array([5.0, 10.0])
    y_pred = np.array([-3.0, 10.0])  # negative demand prediction should be clipped to 0
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(2.5)  # |5-0| + |10-10| / 2


@pytest.fixture(scope="module")
def small_config(tmp_path_factory):
    config = load_config()
    config = copy.deepcopy(config)
    tmp_dir = tmp_path_factory.mktemp("train_test")
    config["data"]["processed_path"] = str(tmp_dir / "features.csv")
    config["model"]["artifact_path"] = str(tmp_dir / "model.joblib")
    config["model"]["preprocessor_path"] = str(tmp_dir / "preprocessor.joblib")
    config["model"]["metrics_path"] = str(tmp_dir / "metrics.json")
    config["split"]["cv_n_splits"] = 3
    return config


def test_train_and_evaluate_produces_metrics(small_config, monkeypatch):
    from src import config as config_module

    monkeypatch.setattr(config_module, "resolve_path", lambda p: config_module.Path(p))

    metrics = train_and_evaluate(small_config)

    assert metrics["model"]["mae"] >= 0
    assert metrics["n_train"] > 0
    assert metrics["n_test"] > 0
    assert "seasonal_naive_baseline" in metrics


def test_model_beats_seasonal_naive_baseline(small_config, monkeypatch):
    """The trained model should outperform the 'same hour last week' baseline —
    otherwise it isn't adding value over a trivial heuristic."""
    from src import config as config_module

    monkeypatch.setattr(config_module, "resolve_path", lambda p: config_module.Path(p))

    metrics = train_and_evaluate(small_config)
    assert metrics["model"]["mae"] < metrics["seasonal_naive_baseline"]["mae"]
