"""Shared helpers for filesystem paths, reproducibility, and reporting."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_PATH = DATA_DIR / "raw" / "concrete_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def ensure_directories(extra_paths: Iterable[Path] | None = None) -> None:
    """Create the standard project directories plus any additional ones."""

    base_paths = [
        DATA_DIR / "raw",
        DATA_DIR / "interim",
        DATA_DIR / "processed",
        DATA_DIR / "external",
        MODELS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MLRUNS_DIR,
    ]
    for path in (extra_paths or []):
        base_paths.append(path)

    for path in base_paths:
        path.mkdir(parents=True, exist_ok=True)


def set_all_seeds(seed: int = 42) -> None:
    """Set python and NumPy seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)


def describe_split(name: str, size: int, total: int) -> str:
    pct = 100 * size / max(total, 1)
    return f"{name}: {size} ({pct:.1f}%)"
