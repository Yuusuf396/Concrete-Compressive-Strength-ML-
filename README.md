# CS-4120 Project Proposal  
**Group Name:** Concrete Strength-G1  

---

## 1. Problem and Motivation
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
- Predict exact compressive strength using:  
  - Linear Regression  
  - Multiple Linear Regression  
  - Polynomial Regression  

### Classification Task
- Predict high vs low strength using:  
  - Decision Trees  
  - Bayesian Learning  

---

## 4. Planned Metrics
- **Classification:** Accuracy and F1 Score (with confusion matrix at midpoint/final)  
- **Regression:** MAE and RMSE (with residual analysis)  

Validation will include **train/test splits** and **K-Fold cross-validation** to avoid overfitting.  

---

## 5. Baseline Plan
- **Classification Baselines:** Logistic Regression, Decision Tree  
- **Regression Baselines:** Linear Regression, Decision Tree Regressor  

These will serve as interpretable baselines before introducing a neural network.  

---

## 6. Reproducibility Plan
- Use **Python** with **MLflow** to log experiments, metrics, and artifacts  
- Pin dependencies in `requirements.txt`  
- Fix random seeds for reproducibility  
- Repository follows course-recommended structure (`src/`, `models/`, `mlruns/`, etc.)  

---

## 7. Tables

### Table 1 – Dataset Snapshot

| Rows | Columns | Targets | % Missing (Top 5) | Class Distribution |
|------|---------|---------|--------------------|-------------------|
| 1030 | 9       | High vs Low (classification); Continuous MPa (regression) | 0% (all features) | ≈50% High / 50% Low |

---

### Table 2 – Planned Models and Metrics

| Task          | Baseline Models                           | Metrics          |
|---------------|-------------------------------------------|------------------|
| Classification| Logistic Regression, Decision Tree        | Accuracy, F1     |
| Regression    | Linear Regression, Decision Tree Regressor| MAE, RMSE        |

---
