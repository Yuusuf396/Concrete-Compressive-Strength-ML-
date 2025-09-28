# Concrete-Compressive-Strength-ML-
Group Name: Concrete Strength-G1
1. Problem and Motivation
Concrete is one of the most widely used construction materials, and its compressive strength is a critical factor for ensuring safety and durability of structures. Accurate prediction of compressive strength helps engineers design cost-effective mixtures, minimize material waste, and improve construction safety. This project uses machine learning to predict compressive strength both as a continuous value (regression) and as a binary outcome (classification: high vs. low strength). Comparing classical ML models and neural networks will highlight their effectiveness and limitations.
2. Dataset Description
Name & Source: Concrete Strength Dataset – Kaggle (by Hamza Khurshed).
Link: Concrete Strength Dataset
License: CC0: Public Domain.
Size: 1,030 rows × 9 columns.

Features (all numeric):
Cement (kg/m³)
Blast Furnace Slag (kg/m³)
Fly Ash (kg/m³)
Water (kg/m³)
Superplasticizer (kg/m³)
Coarse Aggregate (kg/m³)
Fine Aggregate (kg/m³)
Age (days)

Target Variable: Strength (continuous, in MPa).
Missingness: 0%
Sensitive Attributes: None (dataset contains only engineering material measurements).
Derived Label: Binary Strength_Class (High/Low using median split).
This dataset is widely used in machine learning benchmarks and is suitable because it supports both a continuous regression task and a discretized classification task (via median split).

GitHub Repo Link : https://github.com/Yuusuf396/Concrete-Compressive-Strength-ML-.git
3. Tasks

Regression Task:
Predict exact compressive strength using Linear Regression, Multiple Linear Regression, Polynomial Regression. 
Classification Task:
Predict high vs low strength using Decision Trees and Bayesian Learning for classification

4. Planned Metrics
Classification: Accuracy and F1 Score (with confusion matrix at midpoint/final).
Regression: MAE and RMSE (with residual analysis).
Validation will include train/test splits and K-Fold cross validation to avoid overfitting. 



5. Baseline Plan
Classification Baselines: Logistic Regression, Decision Tree.
Regression Baselines: Linear Regression, Decision Tree Regressor.
        These will serve as interpretable baselines before introducing a neural network.
6. Reproducibility Plan
Use Python with MLflow to log experiments, metrics, and artifacts.
Pin dependencies in requirements.txt.
Fix random seeds for reproducibility.
Repository will follow course-recommended structure (src/, models/, mlruns/, etc.).







