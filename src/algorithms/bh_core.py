"""Shared black-hole algorithm primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.evaluation.fitness import score_with_model
from src.utils.distance import nearest_reference_indices
from src.utils.population import ensure_non_empty_population, select_columns_by_mask


@dataclass
class BlackHoleConfig:
    """Runtime configuration for black-hole selection."""

    population_size: int = 20
    max_iterations: int = 30
    num_black_holes: int = 1
    runs: int = 3


def separate_black_holes(population: np.ndarray, x_train, x_test, y_train, y_test, num_black_holes: int):
    """Split population into top black holes and remaining stars."""
    fitness_scores = []
    for individual in population:
        x_tr = select_columns_by_mask(x_train, individual)
        x_te = select_columns_by_mask(x_test, individual)
        fitness_scores.append(score_with_model(x_tr, x_te, y_train, y_test, model_name="svm"))

    order = np.argsort(fitness_scores)[::-1]
    black_hole_idx = order[:num_black_holes]
    stars_idx = order[num_black_holes:]
    return population[black_hole_idx], np.array(fitness_scores)[black_hole_idx], population[stars_idx], np.array(fitness_scores)[stars_idx]


def run_black_hole_selection(population: np.ndarray, x_train, x_test, y_train, y_test, config: BlackHoleConfig) -> np.ndarray:
    """Run the black-hole update loop."""
    bit_size = population.shape[1]
    current = ensure_non_empty_population(population)
    for _ in range(config.max_iterations):
        black_holes, bh_fitness, stars, star_fitness = separate_black_holes(
            current, x_train, x_test, y_train, y_test, config.num_black_holes
        )
        if len(stars) == 0:
            break
        event_horizon = float(bh_fitness[0] / max(np.sum(star_fitness), 1e-9))
        nearest_idx = nearest_reference_indices(stars, black_holes)
        for i, star in enumerate(stars):
            b_idx = nearest_idx[i]
            if abs(float(bh_fitness[b_idx] - star_fitness[i])) <= event_horizon:
                stars[i] = np.random.randint(0, 2, size=bit_size)
            else:
                diff_idx = np.where(star != black_holes[b_idx])[0]
                if len(diff_idx) > 0:
                    mutate_n = max(1, int(0.25 * len(diff_idx)))
                    chosen = np.random.choice(diff_idx, size=mutate_n, replace=False)
                    stars[i][chosen] = 1 - stars[i][chosen]
        current = ensure_non_empty_population(np.vstack([stars, black_holes]))
    return current
