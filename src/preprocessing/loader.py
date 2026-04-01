"""Data-loading functions used across experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(path: str | Path, drop_first_row: bool = True) -> pd.DataFrame:
    """Load CSV/Excel data and optionally drop first metadata row."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix in {".xls", ".xlsx"}:
        data = pd.read_excel(file_path)
    else:
        separator = " " if "HeartEW.csv" in file_path.name else ","
        data = pd.read_csv(file_path, sep=separator, header=None)

    if drop_first_row and len(data) > 0:
        data = data.drop(index=0).reset_index(drop=True)

    return data
