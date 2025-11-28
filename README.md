# Concrete Compressive Strength – Midpoint Repo

Predicting concrete compressive strength is framed as a **dual task**:

- **Regression** – estimate the target strength in MPa.
- **Classification** – detect high-strength mixes (>= 32 MPa, matching the midpoint report).

This repo packages the midpoint deliverable with a clean Cookiecutter-style layout, reproducible scripts, and MLflow tracking. Classical baselines and neural-network baselines are separate entry points, while shared utilities live under `src/`.

## Repository layout

```
project/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md           # download instructions (no raw CSV tracked)
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── mlruns/                 # local MLflow tracking dir (gitignored except .gitkeep)
├── models/                 # lightweight serialized artifacts
├── notebooks/              # optional exploratory notebooks
├── reports/                # derived tables/figures (auto-generated)
└── src/
	├── data.py             # loading, cleaning, splitting
	├── features.py         # engineered features + scaling
	├── evaluate.py         # metrics, plots, confusion/residuals
	├── train_baselines.py  # classical ML for both tasks
	├── train_nn.py         # MLP-based baselines for both tasks
	└── utils.py            # shared helpers (paths, seeds, dirs)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All dependencies are pinned for deterministic builds. MLflow uses a file-based backend under `mlruns/`.

## Data

Raw datasets are **not** committed. To download the UCI Concrete CSV and emit processed artifacts, run:

```bash
python -m src.data
```

This command fetches the file via `ucimlrepo`, writes `data/raw/concrete_data.csv`, adds the binary `strength_class` label, and saves enriched outputs under `data/processed/`.

## Training workflows

### 1. Classical baselines

```bash
python src/train_baselines.py
# or: python -m src.train_baselines
```

Models: Logistic Regression, GaussianNB, LinearRegression, DecisionTreeRegressor. Outputs:

- MLflow runs under experiment `ConcreteStrength_Baselines`.
- Saved models in `models/`.
- Classification/regression metric tables in `reports/tables/`.
- Confusion matrix & residual plots in `reports/figures/`.

### 2. Neural-network baselines

```bash
python src/train_nn.py
# or: python -m src.train_nn
```

Models: feed-forward MLPClassifier and MLPRegressor (scikit-learn). Metrics/plots/logs follow the same convention under experiment `ConcreteStrength_NeuralNets`.

### 3. Inspect runs with MLflow UI

```bash
mlflow ui --backend-store-uri "file:$(pwd)/mlruns"
```

## Reproducibility checklist

- **Seeds:** `src.utils.set_all_seeds()` enforces deterministic NumPy/Python RNG states. Classical baselines use seed `42`; neural nets use seed `21`. Update one constant to propagate everywhere.
- **Deterministic splits:** `src.data.build_splits()` performs a fixed 70/15/15 split (`random_state=42`), ensuring both baseline scripts read identical folds.
- **Data availability:** `python -m src.data` is the single command to download/refresh `data/raw/concrete_data.csv` (kept out of git) and regenerate processed parquet + summary stats.
- **Experiment logging:** Both `train_baselines.py` and `train_nn.py` start MLflow runs, record parameters/metrics, save confusion + residual plots, and log serialized `.joblib` models. Launch `mlflow ui` (above) to compare runs.
- **Artifacts:** Generated figures/tables/models stay in `reports/` and `models/`, which are gitignored by default to keep the repo lightweight.

## Evaluation helpers

`src.evaluate` centralizes metric computations and plotting so both training scripts remain lean. Confusion matrices and residual plots are written once per script, making it easy to drop them into midpoint reports.

## Deliverables linkage (midpoint narrative)

1. **Data prep (`src.data`)** – fetch + enrich raw records, freeze splits (70/15/15).
2. **Feature engineering (`src.features`)** – build ratios (e.g., water/cement), scale numeric fields.
3. **Classical baselines (`src.train_baselines`)** – log metrics/tables/plots to reports + MLflow.
4. **Neural nets (`src.train_nn`)** – compare non-linear capacity vs. baselines.
5. **Artifacts (`reports/`, `models/`)** – ready to embed into the midpoint PDF.

## Next steps

- Expand notebooks in `notebooks/` for richer EDA.
- Add automated tests (PyTest) to guard data transforms and splits.
- Explore hyper-parameter sweeps (Optuna) using the same layout.

For any questions, open an issue or ping the maintainers of **Concrete-Strength-G1**.
