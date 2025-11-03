# Concrete Compressive Strength Prediction
**Group Name:** Concrete-Strength-G1  

## Project Overview
Machine learning project to predict concrete compressive strength using both regression (exact MPa) and classification (high vs. low strength) approaches. This repository contains our midpoint deliverables including baseline models, evaluation metrics, and visualizations.
Concrete is one of the most widely used construction materials, and its compressive strength is a critical factor for ensuring safety and durability of structures. Accurate prediction of compressive strength helps engineers design cost-effective mixtures, minimize material waste, and improve construction safety.  

This project uses **machine learning** to predict compressive strength both as:
- A continuous value (**regression**)  
- A binary outcome (**classification: high vs. low strength**)  

Comparing classical ML models and neural networks will highlight their effectiveness and limitations.

---

## 2. Dataset Description
- **Name & Source:** [Concrete Strength Dataset – Kaggle (by Hamza Khurshed)](https://www.kaggle.com/datasets/hamzakhurshed/concrete-strength-dataset)  
- **License:** CC0: Public Domain  
- **Size:** 1,030 rows × 9 columns  

**Features (all numeric):**  
- Cement (kg/m³)  
- Blast Furnace Slag (kg/m³)  
- Fly Ash (kg/m³)  
- Water (kg/m³)  
- Superplasticizer (kg/m³)  
- Coarse Aggregate (kg/m³)  
- Fine Aggregate (kg/m³)  
- Age (days)  

**Target Variable:** Strength (continuous, in MPa)  
- **Missingness:** 0%  
- **Sensitive Attributes:** None (only engineering material measurements)  
- **Derived Label:** Binary `Strength_Class` (High/Low using median split)  

This dataset is widely used in ML benchmarks and is suitable because it supports both continuous regression and discretized classification tasks.  

📂 **GitHub Repo Link:** [Concrete-Compressive-Strength-ML](https://github.com/Yuusuf396/Concrete-Compressive-Strength-ML-.git)

---

## 3. Tasks

### Regression Task
# Concrete Compressive Strength — Midpoint Report

**Group:** Concrete-Strength-G1

## Summary

This repository contains the midpoint deliverables for predicting concrete compressive strength using classical machine learning approaches. We address both regression (predicting strength in MPa) and classification (high vs. low strength using a 32 MPa threshold). The project includes data processing utilities, exploratory analysis, baseline model training with MLflow tracking, evaluation artifacts (plots and tables), and a compiled PDF report.

## Project status (midpoint)

- Data: `concrete_data.csv` (1,030 samples, 8 features)
- Baselines implemented and tracked with MLflow: Logistic Regression, GaussianNB (classification); Linear Regression, Decision Tree Regressor (regression)
- Deliverables generated: 4 plots, 2 metric tables, and a PDF report (`midpoint_Concrete-Strength-G1.pdf`).

## Repository layout

```
.
├── concrete_data.csv
├── requirements.txt
├── midpoint_Concrete-Strength-G1.pdf
├── plots/        # Generated figures (plot1..plot4, table images)
├── reports/      # CSV metric tables and artifact paths
└── src/          # Source code (data, EDA, training, pipeline)
```

Key scripts:

- `src/midpoint_pipeline.py` — single-run pipeline that trains baselines, logs to MLflow, produces required plots/tables and assembles the PDF report.
- `src/eda_plots.py` — exploratory visualizations used by the pipeline.
- `src/train_nn.py` — starter code for neural network experiments (future work).

## Quick start

1. Create a virtual environment and activate it:

```powershell
python -m venv .venv
.venv\\Scripts\\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the midpoint pipeline (from repo root):

```powershell
python src/midpoint_pipeline.py
```

Outputs are written to `plots/`, `reports/`, and the root PDF file.

## Midpoint results (high-level)

- Classification: Logistic Regression (with StandardScaler) achieved validation F1 ≈ 0.82 and test F1 = 0.90, indicating strong discriminative power for the chosen features.
- Regression: DecisionTreeRegressor outperforms LinearRegression (test RMSE 7.86 MPa vs 9.81 MPa), indicating important non-linear effects in the data.

## Key insights

- Cement and supplementary cementitious materials (slag, fly ash) show the strongest positive associations with compressive strength; water has a negative effect. Linear model coefficients (approx.) — Cement: +13.20, Slag: +10.33, Age: +7.28, Water: -2.27 (MPa per unit).
- The superior performance of tree-based models suggests non-linear interactions and time-dependent (age) effects; a neural network and targeted feature engineering (e.g., water/cement ratio, age interactions) are natural next steps.

## Reproducibility and tracking

- Experiments are logged under MLflow experiment `ConcreteStrength_Baselines`.
- Random seeds are fixed in scripts for reproducibility.

## Next steps

1. Implement and tune an MLP to capture non-linear interactions and age dependence.
2. Add domain-driven feature engineering (e.g., water/cement ratio, interaction terms).
3. Add CI checks and a small set of unit tests for data-loading and pipeline sanity.

## Data source

Concrete Strength Dataset (Kaggle) — Hamza Khurshed (CC0 Public Domain). See `concrete_data.csv` in the repo.

## Contact

For questions or collaboration, please open an issue or pull request on the repository.

---
_Prepared as a midpoint deliverable._
