"""Distance utilities for binary feature-selection vectors."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def hamming_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> int:
    """Return Hamming distance between two binary vectors."""
    return int(np.sum(vector_a != vector_b))


def nearest_reference_indices(
    candidates: Iterable[np.ndarray],
    references: Iterable[np.ndarray],
) -> list[int]:
    """Return nearest reference index for each candidate using Hamming distance."""
    reference_list = list(references)
    nearest: list[int] = []
    for candidate in candidates:
        distances = [hamming_distance(candidate, ref) for ref in reference_list]
        nearest.append(int(np.argmin(distances)))
    return nearest
