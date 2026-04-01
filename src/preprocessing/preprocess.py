"""Preprocessing utilities for tabular datasets."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def preprocess_dataframe(
    df: pd.DataFrame,
    drop_constant_features: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Encode target labels, scale features, and optionally drop constants."""
    x_features = df.iloc[:, :-1].copy()
    y_target = df.iloc[:, -1].copy()

    x_features = x_features.fillna(0)

    if y_target.dtype == "object":
        y_target = pd.Series(LabelEncoder().fit_transform(y_target))

    x_features = pd.DataFrame(MinMaxScaler().fit_transform(x_features))

    if drop_constant_features:
        nunique = x_features.nunique()
        x_features = x_features.loc[:, nunique > 1].copy()
        x_features.columns = range(x_features.shape[1])

    return x_features, y_target
