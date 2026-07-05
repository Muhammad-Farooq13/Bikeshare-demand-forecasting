"""Configuration loading utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load the YAML configuration file."""
    env_path = os.getenv("CONFIG_PATH")
    resolved = Path(env_path) if env_path else Path(config_path)

    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found at: {resolved}")

    with open(resolved, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT / relative_path
