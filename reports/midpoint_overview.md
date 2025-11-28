# Midpoint pipeline walkthrough

1. **Data ingestion (`src/data.py`)**
   - `download_raw_dataset()` pulls the UCI concrete dataset (optional when `data/raw/concrete_data.csv` already exists).
   - `add_targets()` adds the binary `strength_class` label using the median MPa as the cutoff (configurable).
   - `build_splits()` freezes a 70/15/15 train/val/test partition that is reused by every downstream script.

2. **Feature engineering (`src/features.py`)**
   - `make_feature_frame()` augments the eight numeric inputs with domain-driven ratios (water/cement, binder totals, aggregate ratios).
   - `inject_features()` syncs the engineered matrix with each split bundle, keeping the same row indices for both regression and classification targets.
   - `scale_features()` standardizes every column via `StandardScaler` so linear models and MLPs train stably.

3. **Classical baselines (`src.train_baselines.py`)**
   - Trains LogisticRegression + RandomForestClassifier for classification; LinearRegression + DecisionTreeRegressor for regression.
   - Logs metrics to MLflow experiment `ConcreteStrength_Baselines`, saves `.joblib` models into `models/`, and writes summary tables to `reports/tables/`.
   - Uses `src.evaluate` to create confusion matrices and residual plots for the best-performing models (by validation F1 and validation RMSE respectively).

4. **Neural-network baselines (`src.train_nn.py`)**
   - Runs scikit-learn `MLPClassifier`/`MLPRegressor` with early stopping, reusing the same data splits and feature scaler.
   - Logs to experiment `ConcreteStrength_NeuralNets` and saves artifacts under `models/` + `reports/figures/`.

5. **Evaluation helpers (`src.evaluate.py`)**
   - Central location for metric calculations and plotting utilities so both training scripts stay concise and consistent.
   - `save_metrics_table()` ensures CSV tables are emitted for quick import into midpoint reports.

6. **Artifacts**
   - `models/` keeps lightweight serialized estimators (ignored except for `.gitkeep`).
   - `reports/figures/` + `reports/tables/` store auto-generated plots/tables that feed into the midpoint presentation or PDF.
   - `mlruns/` captures the full experiment history for instructor review (via `mlflow ui`).

This modular split replaces the previous monolithic `midpoint_pipeline.py`, making it easier to upgrade individual stages (e.g., swap baselines, add feature engineering, or plug in a new NN architecture) without rewriting the entire workflow.
