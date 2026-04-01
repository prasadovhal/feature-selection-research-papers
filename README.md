# Feature Selection Research Papers

A curated repository for feature-selection research implementations focused on optimization-based methods for high-dimensional machine learning.

## Implemented Methods

- Improved filter-ranking incorporated binary black-hole algorithm for feature selection  
  Authors: Prasad Ovhal, Shubham Kulkarni, Jayaraman K Valadi
- Twin and multiple black-holes algorithm for feature selection  
  Authors: Prasad T Ovhal, Jayaraman K Valadi, Aamod Sane
- Intrusion detection with black-hole feature selection  
  Authors: Shubham Kulkarni, Prasad Ovhal, Jayaraman K Valadi
- Improving black-hole algorithm performance by coupling with genetic algorithm for feature selection  
  Authors: Hrushikesh Bhosale, Prasad Ovhal, Aamod Sane, Jayaraman K Valadi

## Datasets

- UCI benchmark datasets
- Microarray datasets (Colon, Leukemia, etc.)
- Network intrusion datasets (KDD, NSL-KDD)

## Repository Structure

```text
feature-selection-research-papers/
├── .gitignore
├── pyproject.toml
├── README.md
├── datasets/
└── src/
    ├── __init__.py
    ├── algorithms/
    │   ├── __init__.py
    │   ├── improved_filter_ranking_black_hole.py
    │   ├── intrusion_detection_black_hole.py
    │   ├── bh_core.py
    │   ├── hybrid_black_hole_genetic_algorithm.py
    │   └── twin_multiple_black_holes.py
    ├── evaluation/
    │   ├── __init__.py
    │   └── fitness.py
    ├── experiments/
    │   └── run_experiment.py
    ├── preprocessing/
    │   ├── __init__.py
    │   ├── loader.py
    │   └── preprocess.py
    └── utils/
        ├── __init__.py
        ├── distance.py
        └── population.py
```

## Usage

```bash
poetry install
poetry run python src/experiments/run_experiment.py
```

## Quick Start

```bash
poetry env use 3.10
poetry install
poetry run python src/experiments/run_experiment.py
```

## Run Algorithms

```bash
poetry run python src/algorithms/twin_multiple_black_holes.py
poetry run python src/algorithms/improved_filter_ranking_black_hole.py
poetry run python src/algorithms/intrusion_detection_black_hole.py
poetry run python src/algorithms/hybrid_black_hole_genetic_algorithm.py
```

## Dataset Setup

- Put all input files in `datasets/`.
- Benchmark scripts expect names such as `biodeg.csv`, `BreastEW.csv`, `IonosphereEW.csv`, and `WineEW.csv`.
- Intrusion-detection script expects:
  - `datasets/KDDTrain_firefly_paper_2.csv`
  - `datasets/KDDTest_firefly_paper_2.csv`

## Outputs

- Scripts write result CSV files in the project root, for example:
  - `BBH_results_clean.csv`
  - `filter_ranking_BH_FS_clean.csv`
  - `BH_firefly_kdd_clean.csv`
  - `coupled_algo_GA_BH_clean.csv`

## Reproducibility

- Most scripts use fixed `random_state` values in train/test split.
- For stricter reproducibility, set global seeds before runs (`numpy`, model seeds, and CV seeds where applicable).

## Citation

- If you use this repository in research, cite the corresponding papers listed in **Implemented Methods**.

## License

- Add your project license here (for example, MIT, Apache-2.0, or GPL-3.0).

## Notes

- Main algorithm implementations are in `src/algorithms`.
- Shared reusable components are separated into `src/utils`, `src/preprocessing`, and `src/evaluation`.
