"""
Twin and multiple black-holes algorithm for feature selection  
  Authors: Prasad T Ovhal, Jayaraman K Valadi, Aamod Sane
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.algorithms.bh_core import BlackHoleConfig, run_black_hole_selection, separate_black_holes
from src.evaluation.fitness import pure_accuracy
from src.preprocessing.loader import load_dataset
from src.preprocessing.preprocess import preprocess_dataframe
from src.utils.population import initialize_population, select_columns_by_mask


def run_dataset(dataset_path: Path, config: BlackHoleConfig) -> list[dict]:
    """Run black-hole feature selection repeatedly on one dataset."""
    records: list[dict] = []
    data = load_dataset(dataset_path)
    x_data, y_data = preprocess_dataframe(data)
    for run in range(config.runs):
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42 + run)
        pop = initialize_population(config.population_size, x_data.shape[1])
        final_pop = run_black_hole_selection(pop, x_train, x_test, y_train, y_test, config)
        black_holes, bh_fitness, _, _ = separate_black_holes(
            final_pop, x_train, x_test, y_train, y_test, config.num_black_holes
        )
        best = black_holes[0]
        x_tr_best = select_columns_by_mask(x_train, best)
        x_te_best = select_columns_by_mask(x_test, best)
        records.append(
            {
                "dataset": str(dataset_path),
                "run": run,
                "subset": np.where(best == 1)[0].tolist(),
                "subset_size": int(np.sum(best)),
                "fitness": float(bh_fitness[0]),
                "accuracy": pure_accuracy(x_tr_best, x_te_best, y_train, y_test, model_name="svm"),
            }
        )
    return records


def main() -> None:
    """Execute the twin/multiple black-hole experiment."""
    datasets = [
        "biodeg.csv",
        "BreastEW.csv",
        "Cardiotocography.xls",
        "colon.csv",
        "derm.csv",
        "HeartEW.csv",
        "IonosphereEW.csv",
        "leukemia.csv",
        "spambase.csv",
        "steel-plates-fault_csv.csv",
        "WaveformEW.csv",
        "WineEW.csv",
    ]
    config = BlackHoleConfig()
    all_rows: list[dict] = []
    for file_name in datasets:
        dataset_path = Path("datasets") / file_name
        if dataset_path.exists():
            all_rows.extend(run_dataset(dataset_path, config))
    if all_rows:
        pd.DataFrame(all_rows).to_csv("BBH_results_clean.csv", index=False)


if __name__ == "__main__":
    main()
