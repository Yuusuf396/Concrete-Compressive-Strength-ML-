"""Data ingestion, preprocessing, and splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import utils

COLUMN_NAMES = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
    "strength",
]
FEATURE_COLUMNS = COLUMN_NAMES[:-1]
CLASS_COLUMN = "strength_class"
REG_COLUMN = "strength"


@dataclass
class SplitBundle:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def download_raw_dataset(force: bool = False) -> Path:
    """Download the concrete dataset from the UCI repository."""

    utils.ensure_directories()
    if utils.RAW_DATA_PATH.exists() and not force:
        return utils.RAW_DATA_PATH

    from ucimlrepo import fetch_ucirepo  # lazy import to avoid dependency during docs builds

    dataset = fetch_ucirepo(id=165)
    features = dataset.data.features
    targets = dataset.data.targets
    df = pd.concat([features, targets], axis=1)
    df.columns = COLUMN_NAMES
    df.to_csv(utils.RAW_DATA_PATH, index=False)
    return utils.RAW_DATA_PATH


def load_raw_dataframe() -> pd.DataFrame:
    if not utils.RAW_DATA_PATH.exists():
        download_raw_dataset()
    df = pd.read_csv(utils.RAW_DATA_PATH)
    df.columns = COLUMN_NAMES
    return df


def add_targets(df: pd.DataFrame, threshold: float | str = 32.0) -> Tuple[pd.DataFrame, float]:
    enriched = df.copy()
    if threshold == "median":
        cutoff = enriched[REG_COLUMN].median()
    elif isinstance(threshold, (int, float)):
        cutoff = float(threshold)
    else:
        raise ValueError("threshold must be 'median' or a numeric value")
    enriched[CLASS_COLUMN] = (enriched[REG_COLUMN] >= cutoff).astype(int)
    return enriched, cutoff


def save_processed_copy(df: pd.DataFrame) -> Path:
    utils.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = utils.PROCESSED_DATA_DIR / "concrete_with_targets.csv"
    df.to_csv(out_path, index=False)
    summary_path = utils.PROCESSED_DATA_DIR / "summary_statistics.csv"
    df.describe(include="all").to_csv(summary_path)
    return out_path


def _split_indices(df: pd.DataFrame, split: Tuple[float, float, float], random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_frac, val_frac, test_frac = split
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Train/val/test fractions must sum to 1.0")
    indices = df.index.to_numpy()
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_frac,
        random_state=random_state,
        shuffle=True,
    )
    val_relative = val_frac / (train_frac + val_frac)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative,
        random_state=random_state,
        shuffle=True,
    )
    return train_idx, val_idx, test_idx


def build_splits(
    df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_state: int = 42,
) -> Dict[str, SplitBundle]:
    X = df[FEATURE_COLUMNS].copy()
    y_class = df[CLASS_COLUMN]
    y_reg = df[REG_COLUMN]
    train_idx, val_idx, test_idx = _split_indices(df, split, random_state)

    def bundle(y: pd.Series) -> SplitBundle:
        return SplitBundle(
            X_train=X.iloc[train_idx],
            X_val=X.iloc[val_idx],
            X_test=X.iloc[test_idx],
            y_train=y.iloc[train_idx],
            y_val=y.iloc[val_idx],
            y_test=y.iloc[test_idx],
        )

    return {
        "classification": bundle(y_class),
        "regression": bundle(y_reg),
    }


def main() -> None:
    utils.ensure_directories()
    df = load_raw_dataframe()
    enriched, cutoff = add_targets(df)
    save_path = save_processed_copy(enriched)
    print("Processed dataset saved to", save_path)
    print(f"Classification threshold: {cutoff:.2f} MPa")


if __name__ == "__main__":
    main()
