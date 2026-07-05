"""Model explainability using SHAP.

Generates a SHAP summary plot showing which features drive hourly demand
predictions — useful for an operations team deciding which signals
(weather? time of day? day of week?) actually matter for rebalancing
decisions.

Run directly:
    python -m src.explain
"""

from __future__ import annotations

from typing import Any, Dict

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.config import load_config, resolve_path
from src.features import get_output_feature_names, split_X_y
from src.logger import get_logger

logger = get_logger(__name__)


def run_explainability(config: Dict[str, Any]) -> str:
    """Compute SHAP values for a sample of test-period hours and save a summary plot."""
    model_cfg = config["model"]
    explain_cfg = config["explainability"]
    feature_cfg = config["features"]
    data_cfg = config["data"]

    model_path = resolve_path(model_cfg["artifact_path"])
    preproc_path = resolve_path(model_cfg["preprocessor_path"])
    processed_path = resolve_path(data_cfg["processed_path"])

    for path in (model_path, preproc_path, processed_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}. Run `make train` first.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preproc_path)
    df = pd.read_csv(processed_path, parse_dates=["dteday"])

    sample_size = min(explain_cfg["shap_sample_size"], len(df))
    sample = df.sample(n=sample_size, random_state=42)
    X_sample, _ = split_X_y(sample, feature_cfg)
    X_sample_t = preprocessor.transform(X_sample)
    feature_names = get_output_feature_names(preprocessor)

    logger.info("Computing SHAP values for %d sampled hours", sample_size)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample_t)

    out_path = resolve_path(explain_cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X_sample_t, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Saved SHAP summary plot to %s", out_path)
    return str(out_path)


def main() -> None:
    config = load_config()
    run_explainability(config)


if __name__ == "__main__":
    main()
