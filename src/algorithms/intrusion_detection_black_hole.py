"""
Intrusion detection with black-hole feature selection  
  Authors: Shubham Kulkarni, Prasad Ovhal, Jayaraman K Valadi
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.algorithms.improved_filter_ranking_black_hole import iterate_black_hole
from src.algorithms.improved_filter_ranking_black_hole import RankedBHConfig
from src.evaluation.fitness import score_with_model
from src.preprocessing.preprocess import preprocess_dataframe
from src.utils.population import initialize_population, select_columns_by_mask


@dataclass
class FireflyBHConfig(RankedBHConfig):
    """Configuration for firefly-paper BH setup."""

    runs: int = 3
    rank_weight: float = 0.02
    corr_weight: float = 0.0


def load_kdd_pair(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load NSL-KDD train/test pair and align categorical encoders through concat flow."""
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    combined = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    x_all, y_all = preprocess_dataframe(combined)
    train_rows = len(train_df)
    train_xy = pd.concat([x_all.iloc[:train_rows, :], y_all.iloc[:train_rows]], axis=1)
    test_xy = pd.concat([x_all.iloc[train_rows:, :], y_all.iloc[train_rows:]], axis=1)
    return train_xy, test_xy


def main() -> None:
    """Run BH feature selection for the KDD train/test pair."""
    train_path = Path("datasets") / "KDDTrain_firefly_paper_2.csv"
    test_path = Path("datasets") / "KDDTest_firefly_paper_2.csv"
    if not (train_path.exists() and test_path.exists()):
        return

    train_xy, test_xy = load_kdd_pair(train_path, test_path)
    x_train = train_xy.iloc[:, :-1].copy()
    y_train = train_xy.iloc[:, -1].copy()
    x_test = test_xy.iloc[:, :-1].copy()
    y_test = test_xy.iloc[:, -1].copy()

    cfg = FireflyBHConfig(population_size=30, max_iterations=20)
    rows: list[dict] = []
    for run in range(cfg.runs):
        population = initialize_population(cfg.population_size, x_train.shape[1])
        feature_importance = np.ones(x_train.shape[1], dtype=float)
        corr_matrix = x_train.corr()
        final_population = iterate_black_hole(population, x_train, x_test, y_train, y_test, feature_importance, corr_matrix, cfg)
        best_mask = final_population[0]
        x_tr_best = select_columns_by_mask(x_train, best_mask)
        x_te_best = select_columns_by_mask(x_test, best_mask)
        rows.append(
            {
                "run": run,
                "subset": np.where(best_mask == 1)[0].tolist(),
                "subset_size": int(np.sum(best_mask)),
                "f1_micro": score_with_model(
                    x_tr_best,
                    x_te_best,
                    y_train,
                    y_test,
                    model_name="rf",
                    use_f1=True,
                ),
            }
        )

    pd.DataFrame(rows).to_csv("BH_firefly_kdd_clean.csv", index=False)


if __name__ == "__main__":
    main()
