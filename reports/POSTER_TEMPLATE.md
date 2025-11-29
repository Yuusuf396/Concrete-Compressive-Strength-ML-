# POSTER TEMPLATE (Landscape Format)

# Predicting Concrete Compressive Strength: Classical ML vs Neural Networks

---

## [LEFT COLUMN - 1/3 width]

### 🎯 MOTIVATION & DATASET

**Problem**: Predicting concrete compressive strength is critical for construction safety and cost optimization.

**Dataset**: UCI Concrete Compressive Strength

- 1,030 concrete samples
- 8 input features: cement, water, age, aggregates, additives
- Target: Compressive strength (MPa)

**Tasks**:

1. **Classification**: High vs Low strength (threshold: 32 MPa)
2. **Regression**: Exact strength prediction

---

### 🔬 METHODOLOGY

**Preprocessing**:

- Feature engineering (water/cement ratio, binder totals)
- StandardScaler normalization
- 70/15/15 train/val/test split

**Models Tested**:

_Classical ML_:

- Logistic Regression
- Gaussian Naive Bayes
- Linear Regression
- Decision Tree Regressor

_Neural Networks_:

- MLPClassifier (64→32 neurons)
- MLPRegressor (128→64 neurons)
- Adam optimizer, early stopping

---

## [MIDDLE COLUMN - 1/3 width]

### 📊 VISUAL 1: CONFUSION MATRIX

**[Insert: plot3_classification_confusion_matrix.png]**

_Best Classification Model: Logistic Regression_

- 89.0% Test Accuracy
- 90.4% F1-Score

---

### 📈 VISUAL 2: FEATURE IMPORTANCE

**[Insert: plot5_feature_importance.png]**

_Most Critical Features_:

1. **Age** (curing time) - Most important
2. **Cement** content
3. **Water/Cement ratio** (engineered feature)

**Insight**: Domain knowledge improves predictions!

---

## [RIGHT COLUMN - 1/3 width]

### 📋 RESULTS TABLE

| Task               | Best Classical      | Neural Network | Winner     |
| ------------------ | ------------------- | -------------- | ---------- |
| **Classification** |                     |                |            |
| Model              | Logistic Regression | MLP (64→32)    | Classical  |
| Test Accuracy      | **89.0%**           | 85.2%          | +3.8%      |
| Test F1-Score      | **90.4%**           | 86.7%          | +3.7%      |
| **Regression**     |                     |                |            |
| Model              | Decision Tree       | MLP (128→64)   | Neural Net |
| Test RMSE (MPa)    | 6.66                | **5.36**       | -1.3 MPa   |
| Test MAE (MPa)     | 4.76                | **4.03**       | -0.73 MPa  |

**Color coding**: Green = Winner

---

### 🔑 KEY FINDINGS

✅ **Neural networks excel at regression** (capture non-linear relationships)

✅ **Classical models competitive for classification** (simpler, interpretable)

✅ **Feature engineering matters**: Engineered features ranked in top 3

✅ **Early stopping prevents overfitting**: Both NNs converged by epoch 100

---

### ⚠️ LIMITATIONS & RISKS

- Small dataset (1,030 samples)
- Safety-critical application requires validation
- Models supplement, don't replace, engineering judgment
- No uncertainty quantification

---

### 🏁 CONCLUSIONS

1. **No universal winner**: Task-dependent model selection
2. **Neural networks**: Better for complex non-linear regression
3. **Classical models**: Sufficient for linear classification
4. **Domain expertise**: Critical for feature engineering and risk mitigation

**Future Work**: Ensemble methods, Bayesian uncertainty, larger datasets

---

### 👥 TEAM & CONTACT

**Group Name**: [Your Group Name]  
**Members**: [List team members]  
**Course**: Machine Learning  
**Semester**: [Your semester]

---

## LAYOUT INSTRUCTIONS:

1. Use landscape orientation (11" × 8.5")
2. Three equal columns
3. Large, readable fonts (title: 48pt, headings: 24pt, body: 16pt)
4. High contrast for readability from distance
5. Minimal text, maximum visual impact
6. Insert actual images from reports/figures/
