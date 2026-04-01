"""
Improved filter-ranking incorporated binary black-hole algorithm for feature selection  
Authors: Prasad Ovhal, Shubham Kulkarni, Jayaraman K Valadi
  
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.evaluation.fitness import pure_accuracy, score_with_model
from src.preprocessing.loader import load_dataset
from src.preprocessing.preprocess import preprocess_dataframe
from src.utils.population import ensure_non_empty_population, initialize_population, select_columns_by_mask


@dataclass
class RankedBHConfig:
    """Configuration for filter-ranking BH algorithm."""

    population_size: int = 20
    max_iterations: int = 30
    runs: int = 3
    rank_weight: float = 0.05
    corr_weight: float = 0.01


def evaluate_ranked_fitness(mask: np.ndarray, x_train, x_test, y_train, y_test, feature_importance: np.ndarray, corr_matrix, config: RankedBHConfig) -> float:
    """Evaluate penalized score using model fitness, ranking, and correlation penalty."""
    x_tr = select_columns_by_mask(x_train, mask)
    x_te = select_columns_by_mask(x_test, mask)
    base = score_with_model(x_tr, x_te, y_train, y_test, model_name="rf")
    columns = x_tr.columns.to_numpy(dtype=int)
    rank_bonus = float(feature_importance[columns].sum()) if len(columns) else 0.0
    corr_penalty = float(np.triu(corr_matrix.loc[columns, columns]).sum() - np.trace(corr_matrix.loc[columns, columns])) if len(columns) else 0.0
    return base + config.rank_weight * rank_bonus - config.corr_weight * corr_penalty


def iterate_black_hole(population: np.ndarray, x_train, x_test, y_train, y_test, feature_importance: np.ndarray, corr_matrix, config: RankedBHConfig) -> np.ndarray:
    """Run ranked BH optimization loop."""
    bit_size = population.shape[1]
    current = ensure_non_empty_population(population)
    for _ in range(config.max_iterations):
        scores = np.array(
            [evaluate_ranked_fitness(mask, x_train, x_test, y_train, y_test, feature_importance, corr_matrix, config) for mask in current]
        )
        sorted_idx = np.argsort(scores)[::-1]
        black_hole = current[sorted_idx[0]]
        stars = current[sorted_idx[1:]]
        star_scores = scores[sorted_idx[1:]]
        if len(stars) == 0:
            break
        event_horizon = float(scores[sorted_idx[0]] / max(np.sum(star_scores), 1e-9))
        for star_idx in range(len(stars)):
            if abs(float(scores[sorted_idx[0]] - star_scores[star_idx])) <= event_horizon:
                stars[star_idx] = np.random.randint(0, 2, size=bit_size)
            else:
                diff = np.where(stars[star_idx] != black_hole)[0]
                if len(diff):
                    flip_size = max(1, int(0.25 * len(diff)))
                    chosen = np.random.choice(diff, size=flip_size, replace=False)
                    stars[star_idx][chosen] = 1 - stars[star_idx][chosen]
        current = ensure_non_empty_population(np.vstack([stars, black_hole]))
    return current


def main() -> None:
    """Run ranked BH experiment over configured benchmark datasets."""
    datasets = ["biodeg.csv", "BreastEW.csv", "IonosphereEW.csv", "WineEW.csv"]
    config = RankedBHConfig()
    rows: list[dict] = []
    for dataset_name in datasets:
        dataset_path = Path("datasets") / dataset_name
        if not dataset_path.exists():
            continue
        data = load_dataset(dataset_path)
        x_data, y_data = preprocess_dataframe(data, drop_constant_features=True)
        for run in range(config.runs):
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=21 + run)
            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(x_train, y_train)
            feature_importance = rf.feature_importances_
            corr_matrix = x_train.corr()
            population = initialize_population(config.population_size, x_data.shape[1])
            final_population = iterate_black_hole(population, x_train, x_test, y_train, y_test, feature_importance, corr_matrix, config)
            final_scores = np.array(
                [evaluate_ranked_fitness(mask, x_train, x_test, y_train, y_test, feature_importance, corr_matrix, config) for mask in final_population]
            )
            best_mask = final_population[int(np.argmax(final_scores))]
            x_tr_best = select_columns_by_mask(x_train, best_mask)
            x_te_best = select_columns_by_mask(x_test, best_mask)
            rows.append(
                {
                    "dataset": dataset_name,
                    "run": run,
                    "subset": np.where(best_mask == 1)[0].tolist(),
                    "subset_size": int(np.sum(best_mask)),
                    "fitness": float(np.max(final_scores)),
                    "accuracy": pure_accuracy(x_tr_best, x_te_best, y_train, y_test, model_name="rf"),
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv("filter_ranking_BH_FS_clean.csv", index=False)


if __name__ == "__main__":
    main()
