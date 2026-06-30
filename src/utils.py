"""Small utility helpers shared across the project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Write a dictionary as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load JSON if present, otherwise return a default dictionary."""
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    """Return a DataFrame when a CSV exists, otherwise None."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def write_text(path: Path, text: str) -> None:
    """Write text to a path, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

