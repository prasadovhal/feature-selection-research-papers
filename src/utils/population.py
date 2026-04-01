"""Population helper functions for metaheuristic algorithms."""

from __future__ import annotations

import numpy as np


def initialize_population(population_size: int, bit_size: int) -> np.ndarray:
    """Create a random binary population."""
    return np.random.randint(0, 2, size=(population_size, bit_size))


def ensure_non_empty_population(population: np.ndarray) -> np.ndarray:
    """Ensure every individual selects at least one feature."""
    fixed = population.copy()
    for idx in range(len(fixed)):
        if np.sum(fixed[idx]) == 0:
            fixed[idx] = np.random.randint(0, 2, size=fixed.shape[1])
    return fixed


def select_columns_by_mask(matrix, mask: np.ndarray):
    """Select columns from a DataFrame-like matrix using a binary mask."""
    return matrix.iloc[:, mask == 1].copy()
