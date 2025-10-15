"""Utility helpers for the BookTrends recommendation pipeline."""
from __future__ import annotations

from pathlib import Path
import json
import random
from typing import Iterable

import numpy as np
import pandas as pd

SEED = 42
DATA = Path("data")
RAW = DATA / "raw"
PROC = DATA / "processed"
FIGS = Path("figs") / "eda"
MODELS = Path("models")
METRICS = Path("metrics")


def ensure_dirs() -> None:
    """Create all persistent directories required by the pipeline."""
    for path in [RAW, PROC, FIGS, MODELS, METRICS]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path | str) -> None:
    """Persist a dictionary as indented JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def set_seed(seed: int = SEED) -> None:
    """Seed Python and NumPy RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def read_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    """Convenience wrapper around :func:`pandas.read_csv` with UTF-8 default."""
    return pd.read_csv(path, encoding=kwargs.pop("encoding", "utf-8"), **kwargs)


def train_val_test_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into train/val/test partitions preserving index order."""
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size must be < 1")

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_total = len(df)
    n_train = int(train_size * n_total)
    n_val = int(val_size * n_total)
    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val :].reset_index(drop=True)
    return train, val, test


def describe_sparse_matrix(user_ids: Iterable[int], item_ids: Iterable[int]) -> pd.DataFrame:
    """Summarise interaction sparsity statistics."""
    user_series = pd.Series(list(user_ids))
    item_series = pd.Series(list(item_ids))
    return pd.DataFrame(
        {
            "n_users": [user_series.nunique()],
            "n_items": [item_series.nunique()],
            "n_interactions": [len(user_series)],
        }
    )
