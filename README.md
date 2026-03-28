# Feature Selection Research

A curated repository of feature selection research papers, implementations, and optimization-driven approaches for high-dimensional machine learning.

## 🔬 Implemented Methods
- Black Hole Algorithm (BH)
- Genetic Algorithm (GA)
- Hybrid BH + GA
- Firefly-based Feature Selection

## 📊 Datasets
- UCI Benchmark datasets
- Microarray datasets (Colon, Leukemia, etc.)
- Network intrusion datasets (KDD, NSL-KDD)

## 📁 Structure
- `src/algorithms/` → Core FS algorithms
- `src/preprocessing/` → Data handling
- `src/evaluation/` → Fitness + metrics
- `experiments/` → Reproducible runs
- `results/` → Outputs

## Folder Structure

feature-selection-research/
│
├── pyproject.toml
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── src/
│   ├── __init__.py
│   │
│   ├── algorithms/
│   │   ├── bh.py
│   │   ├── ga.py
│   │   ├── hybrid_bh_ga.py
│   │   ├── firefly_bh.py
│   │
│   ├── evaluation/
│   │   ├── fitness.py
│   │   ├── metrics.py
│   │
│   ├── preprocessing/
│   │   ├── loader.py
│   │   ├── preprocess.py
│   │
│   ├── experiments/
│   │   ├── run_experiment.py
│   │   ├── config.yaml
│   │
│   └── utils/
│       ├── population.py
│       ├── distance.py
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── results_visualization.ipynb
│
├── results/
│   ├── csv/
│   ├── plots/
│
├── papers/
│   ├── pdfs/
│   ├── summaries.md
│   ├── citations.bib
│
└── tests/
    ├── test_algorithms.py

## Research papers

1. Improved filter ranking incorporated binary black hole algorithm for feature selection
    Authors - Prasad Ovhal, Shubham Kulkarni, Jayaraman K Valadi
2. Twin and multiple black holes algorithm for feature selection
    Authors - Prasad T Ovhal, Jayaraman K Valadi, Aamod Sane
3. Intrusion Detection with Black Hole Feature Selection
    Authors - Shubham Kulkarni, Prasad Ovhal, Jayaraman K Valadi
4. Improving Black Hole Algorithm Performance by Coupling with Genetic Algorithm for Feature Selection
    Authors - Hrushikesh Bhosale, Prasad Ovhal, Aamod Sane, Jayaraman K Valadi

## 🚀 Usage
```bash
poetry install
poetry run python src/experiments/run_experiment.py
