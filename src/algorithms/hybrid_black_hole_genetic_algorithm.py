"""
Improving black-hole algorithm performance by coupling with genetic algorithm for feature selection  
  Authors: Hrushikesh Bhosale, Prasad Ovhal, Aamod Sane, Jayaraman K Valadi

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.algorithms.bh_core import BlackHoleConfig, run_black_hole_selection, separate_black_holes
from src.evaluation.fitness import pure_accuracy, score_with_model
from src.preprocessing.loader import load_dataset
from src.preprocessing.preprocess import preprocess_dataframe
from src.utils.population import ensure_non_empty_population, initialize_population, select_columns_by_mask


@dataclass
class HybridConfig(BlackHoleConfig):
    """Configuration for hybrid BH + GA."""

    crossover_probability: float = 0.7
    mutation_probability: float = 0.01
    bh_probability: float = 0.4


def ga_step(population: np.ndarray, x_train, x_test, y_train, y_test, mutation_probability: float) -> np.ndarray:
    """Run one GA generation over the binary population."""
    scores = np.array(
        [score_with_model(select_columns_by_mask(x_train, p), select_columns_by_mask(x_test, p), y_train, y_test, model_name="svm") for p in population]
    )
    sorted_idx = np.argsort(scores)[::-1]
    parents = population[sorted_idx[: len(population) // 2]]
    children = []
    bit_size = population.shape[1]
    while len(children) < len(population):
        p1, p2 = parents[np.random.randint(len(parents))], parents[np.random.randint(len(parents))]
        split = np.random.randint(1, bit_size)
        child = np.concatenate([p1[:split], p2[split:]])
        flip_mask = np.random.random(bit_size) < mutation_probability
        child[flip_mask] = 1 - child[flip_mask]
        children.append(child)
    return ensure_non_empty_population(np.array(children))


def main() -> None:
    """Run coupled BH + GA over benchmark datasets."""
    datasets = ["biodeg.csv", "BreastEW.csv", "IonosphereEW.csv", "WineEW.csv"]
    cfg = HybridConfig(population_size=12, max_iterations=8, runs=3)
    all_rows: list[dict] = []
    for dataset_name in datasets:
        path = Path("datasets") / dataset_name
        if not path.exists():
            continue
        data = load_dataset(path)
        x_data, y_data = preprocess_dataframe(data)
        for run in range(cfg.runs):
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42 + run)
            population = initialize_population(cfg.population_size, x_data.shape[1])
            for _ in range(cfg.max_iterations):
                if np.random.random() < cfg.bh_probability:
                    population = run_black_hole_selection(population, x_train, x_test, y_train, y_test, cfg)
                else:
                    population_bh = run_black_hole_selection(population, x_train, x_test, y_train, y_test, cfg)
                    population_ga = ga_step(population, x_train, x_test, y_train, y_test, cfg.mutation_probability)
                    merged = np.vstack([population_bh, population_ga])
                    scores = np.array(
                        [
                            score_with_model(
                                select_columns_by_mask(x_train, mask),
                                select_columns_by_mask(x_test, mask),
                                y_train,
                                y_test,
                                model_name="svm",
                            )
                            for mask in merged
                        ]
                    )
                    best_idx = np.argsort(scores)[::-1][: cfg.population_size]
                    population = merged[best_idx]
            black_holes, bh_fitness, _, _ = separate_black_holes(population, x_train, x_test, y_train, y_test, 1)
            best_mask = black_holes[0]
            all_rows.append(
                {
                    "dataset": dataset_name,
                    "run": run,
                    "subset": np.where(best_mask == 1)[0].tolist(),
                    "subset_size": int(np.sum(best_mask)),
                    "fitness": float(bh_fitness[0]),
                    "accuracy": pure_accuracy(
                        select_columns_by_mask(x_train, best_mask),
                        select_columns_by_mask(x_test, best_mask),
                        y_train,
                        y_test,
                        model_name="svm",
                    ),
                }
            )
    if all_rows:
        pd.DataFrame(all_rows).to_csv("coupled_algo_GA_BH_clean.csv", index=False)


if __name__ == "__main__":
    main()
