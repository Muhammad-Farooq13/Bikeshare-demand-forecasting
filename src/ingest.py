"""Corpus/dataset ingestion.

Downloads the real UCI Bike Sharing hourly dataset (see data/README.md for
provenance and license). Run directly:

    python -m src.ingest
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import requests

from src.config import load_config, resolve_path
from src.logger import get_logger

logger = get_logger(__name__)


def download_raw_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Download the raw hourly dataset if not already present locally.

    Args:
        config: Full configuration dictionary.

    Returns:
        The raw dataset as a DataFrame.
    """
    data_cfg = config["data"]
    raw_path = resolve_path(data_cfg["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        logger.info("Raw data already present at %s", raw_path)
        return pd.read_csv(raw_path)

    url = data_cfg["source_url"]
    logger.info("Downloading dataset from %s", url)
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    raw_path.write_bytes(response.content)
    df = pd.read_csv(raw_path)
    logger.info("Downloaded %d rows to %s", len(df), raw_path)
    return df


def main() -> None:
    config = load_config()
    download_raw_data(config)


if __name__ == "__main__":
    main()
