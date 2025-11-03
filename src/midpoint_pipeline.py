"""
Midpoint pipeline to produce required deliverables:
- Train/val/test split (70/15/15, random_state=42)
- Train classification baselines: LogisticRegression, GaussianNB
- Train regression baselines: LinearRegression, DecisionTreeRegressor
- Log runs/metrics/models to MLflow
- Produce exactly 4 plots and exactly 2 tables, save to `plots/` and `reports/`
- Assemble a PDF `midpoint_Concrete-Strength-G1.pdf` in project root

Run from project root:
& "D:/UPEI/Machine Learning/Project/Concrete-Compressive-Strength-ML-/.venv/Scripts/python.exe" "./src/midpoint_pipeline.py"
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Config
GROUP_NAME = "Concrete-Strength-G1"  # used for PDF filename
PDF_FILENAME = f"midpoint_{GROUP_NAME}.pdf"
EXPERIMENT_NAME = "ConcreteStrength_Baselines"
RANDOM_STATE = 42
SPLIT = (0.70, 0.15, 0.15)  # train/val/test

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT_DIR, 'concrete_data.csv')
PLOTS_DIR = os.path.join(ROOT_DIR, 'plots')
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print(f"Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df.columns = [
    'cement', 'slag', 'fly_ash', 'water', 'superplasticizer',
    'coarse_aggregate', 'fine_aggregate', 'age', 'strength'
]

# Classification target: High-strength >= 32 MPa (as in proposal)
HIGH_STRENGTH_THRESHOLD = 32  # MPa
df['strength_class'] = (df['strength'] >= HIGH_STRENGTH_THRESHOLD).astype(int)

# Features and targets
X = df.drop(['strength', 'strength_class'], axis=1)
y_reg = df['strength']
y_clf = df['strength_class']

# Create pipelines with scaling for linear models
logreg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])

linreg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Train/val/test split helper
def split_train_val_test(X, y, train_frac, val_frac, test_frac, random_state=42):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_frac, random_state=random_state)
    # Now split train_val into train and val
    val_relative = val_frac / (train_frac + val_frac)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_relative, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Apply split for classification and regression (same split for features for reproducibility)
X_train_c, X_val_c, X_test_c, y_train_c, y_val_c, y_test_c = split_train_val_test(X, y_clf, *SPLIT, random_state=RANDOM_STATE)
X_train_r, X_val_r, X_test_r, y_train_r, y_val_r, y_test_r = split_train_val_test(X, y_reg, *SPLIT, random_state=RANDOM_STATE)

# MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)

# Keep metric records for tables
classification_results = []  # list of dicts: model, val_acc, val_f1, test_acc, test_f1
regression_results = []      # list of dicts: model, val_mae, val_rmse, test_mae, test_rmse

# ---------- Classification baselines ----------
classification_models = [
    ("LogisticRegression", logreg_pipeline, {}),
    ("GaussianNB", GaussianNB, {}),
]

for name, model, params in classification_models:
    with mlflow.start_run(run_name=f"clf_{name}"):
        mlflow.log_param("model", name)
        if name == "LogisticRegression":
            mlflow.log_param("scaling", "StandardScaler")
        for k, v in params.items():
            mlflow.log_param(k, v)
        if isinstance(model, Pipeline):
            model.fit(X_train_c, y_train_c)
        else:
            model = model(**params)
            model.fit(X_train_c, y_train_c)
        val_preds = model.predict(X_val_c)
        test_preds = model.predict(X_test_c)
        val_acc = accuracy_score(y_val_c, val_preds)
        val_f1 = f1_score(y_val_c, val_preds)
        test_acc = accuracy_score(y_test_c, test_preds)
        test_f1 = f1_score(y_test_c, test_preds)
        mlflow.log_metric("val_accuracy", float(val_acc))
        mlflow.log_metric("val_f1", float(val_f1))
        mlflow.log_metric("test_accuracy", float(test_acc))
        mlflow.log_metric("test_f1", float(test_f1))
        mlflow.sklearn.log_model(model, "model")

        classification_results.append({
            "model": name,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
        })

# ---------- Regression baselines ----------
regression_models = [
    ("LinearRegression", linreg_pipeline, {}),
    ("DecisionTreeRegressor", DecisionTreeRegressor, {"max_depth":5, "random_state":RANDOM_STATE}),
]

for name, model, params in regression_models:
    with mlflow.start_run(run_name=f"reg_{name}"):
        mlflow.log_param("model", name)
        if name == "LinearRegression":
            mlflow.log_param("scaling", "StandardScaler")
        for k, v in params.items():
            mlflow.log_param(k, v)
        if isinstance(model, Pipeline):
            model.fit(X_train_r, y_train_r)
            # If it's the LinearRegression, analyze coefficients
            if name == "LinearRegression":
                coefficients = model.named_steps['regressor'].coef_
                feature_importances = pd.DataFrame({
                    'feature': X.columns,
                    'coefficient': coefficients
                })
                print("\nLinear Regression Coefficients (impact on strength in MPa per unit):")
                print(feature_importances.sort_values(by='coefficient', ascending=False))
                mlflow.log_param("feature_coefficients", feature_importances.to_dict())
        else:
            model = model(**params)
            model.fit(X_train_r, y_train_r)
        val_preds = model.predict(X_val_r)
        test_preds = model.predict(X_test_r)
        val_mae = mean_absolute_error(y_val_r, val_preds)
        val_rmse = np.sqrt(mean_squared_error(y_val_r, val_preds))
        test_mae = mean_absolute_error(y_test_r, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test_r, test_preds))
        mlflow.log_metric("val_mae", float(val_mae))
        mlflow.log_metric("val_rmse", float(val_rmse))
        mlflow.log_metric("test_mae", float(test_mae))
        mlflow.log_metric("test_rmse", float(test_rmse))
        mlflow.sklearn.log_model(model, "model")

        regression_results.append({
            "model": name,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
        })

# ---------- Select best models for required plots ----------
# For classification: pick model with highest val_f1
clf_df = pd.DataFrame(classification_results)
best_clf = clf_df.loc[clf_df['val_f1'].idxmax()]['model']
# For regression: pick model with lowest val_rmse
reg_df = pd.DataFrame(regression_results)
best_reg = reg_df.loc[reg_df['val_rmse'].idxmin()]['model']

print(f"Best classification model by val F1: {best_clf}")
print(f"Best regression model by val RMSE: {best_reg}")

# Retrain best models on combined train+val (optional) or use saved predictions from earlier runs. Here we'll retrain on train+val for final test eval.
# Combine train+val
from sklearn.base import clone

# Helper to get model class or pipeline by name
model_lookup = {
    'LogisticRegression': (logreg_pipeline, {}),
    'GaussianNB': (GaussianNB, {}),
    'LinearRegression': (linreg_pipeline, {}),
    'DecisionTreeRegressor': (DecisionTreeRegressor, {"max_depth":5, "random_state":RANDOM_STATE}),
}

# Final train for best clf
model, params = model_lookup[best_clf]
if isinstance(model, Pipeline):
    final_clf = model
else:
    final_clf = model(**params)
X_trainval_c = pd.concat([X_train_c, X_val_c])
y_trainval_c = pd.concat([y_train_c, y_val_c])
final_clf.fit(X_trainval_c, y_trainval_c)
final_clf_test_preds = final_clf.predict(X_test_c)

# Final train for best reg
model, params = model_lookup[best_reg]
if isinstance(model, Pipeline):
    final_reg = model
else:
    final_reg = model(**params)
X_trainval_r = pd.concat([X_train_r, X_val_r])
y_trainval_r = pd.concat([y_train_r, y_val_r])
final_reg.fit(X_trainval_r, y_trainval_r)
final_reg_test_preds = final_reg.predict(X_test_r)

# ---------- Required Plots ----------
# Plot 1: Target distribution (bar plot of class counts)
plt.figure(figsize=(5,4))
sns.countplot(x='strength_class', data=df, palette='pastel')
plt.title('Target Distribution (0=Low, 1=High)')
plt.xlabel('Strength Class')
plt.ylabel('Count')
plt.tight_layout()
plot1_path = os.path.join(PLOTS_DIR, 'plot1_target_distribution.png')
plt.savefig(plot1_path)
plt.close()

# Plot 2: Correlation heatmap
plt.figure(figsize=(8,6))
corr = df.drop('strength_class', axis=1).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Features')
plt.tight_layout()
plot2_path = os.path.join(PLOTS_DIR, 'plot2_correlation_heatmap.png')
plt.savefig(plot2_path)
plt.close()

# Plot 3: Confusion matrix for best classification baseline on test set
plt.figure(figsize=(6,5))
ConfusionMatrixDisplay.from_estimator(final_clf, X_test_c, y_test_c, cmap='Blues')
plt.title(f'Confusion Matrix - {best_clf} (test)')
plt.tight_layout()
plot3_path = os.path.join(PLOTS_DIR, 'plot3_confusion_matrix.png')
plt.savefig(plot3_path)
plt.close()

# Plot 4: Residuals vs Predicted for best regression baseline on test set
plt.figure(figsize=(6,5))
sns.scatterplot(x=final_reg_test_preds, y=y_test_r - final_reg_test_preds, color='teal', alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Strength (MPa)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title(f'Residuals vs Predicted - {best_reg} (test)')
# trend line
z = np.polyfit(final_reg_test_preds, (y_test_r - final_reg_test_preds), 1)
p = np.poly1d(z)
plt.plot(final_reg_test_preds, p(final_reg_test_preds), 'r--', alpha=0.8)
plt.tight_layout()
plot4_path = os.path.join(PLOTS_DIR, 'plot4_residuals_vs_predicted.png')
plt.savefig(plot4_path)
plt.close()

# ---------- Required Tables ----------
# Table 1 – Classification metrics for all baselines: Accuracy and F1 on validation and test.
clf_table = pd.DataFrame(classification_results)
clf_table = clf_table[["model", "val_accuracy", "val_f1", "test_accuracy", "test_f1"]]
clf_table_path = os.path.join(REPORTS_DIR, 'table1_classification_metrics.csv')
clf_table.to_csv(clf_table_path, index=False)

# Table 2 – Regression metrics for all baselines: MAE and RMSE on validation and test.
reg_table = pd.DataFrame(regression_results)
reg_table = reg_table[["model", "val_mae", "val_rmse", "test_mae", "test_rmse"]]
reg_table_path = os.path.join(REPORTS_DIR, 'table2_regression_metrics.csv')
reg_table.to_csv(reg_table_path, index=False)

# Also save tables as images for PDF embedding
def dataframe_to_image(df, outpath, title=None, fontsize=10):
    fig, ax = plt.subplots(figsize=(8, max(1, 0.5 + 0.25 * len(df))))
    ax.axis('off')
    if title:
        ax.text(0.5, 1.0, title, transform=ax.transAxes, ha='center', va='bottom', fontsize=12)
    table = ax.table(cellText=df.round(4).values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.2)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)

table1_img = os.path.join(PLOTS_DIR, 'table1_classification_metrics.png')
dataframe_to_image(clf_table, table1_img, title='Table 1: Classification metrics (val & test)')

table2_img = os.path.join(PLOTS_DIR, 'table2_regression_metrics.png')
dataframe_to_image(reg_table, table2_img, title='Table 2: Regression metrics (val & test)')

# ---------- Assemble PDF (2-5 pages, exactly 4 plots and 2 tables) ----------
pdf_path = os.path.join(ROOT_DIR, PDF_FILENAME)
with PdfPages(pdf_path) as pdf:
    # Page 1: Title, dataset description, split and methods (text)
    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4
    ax.axis('off')
    title = f"Midpoint Report (Week 8) - {GROUP_NAME}"
    ax.text(0.5, 0.94, title, ha='center', va='center', fontsize=16, weight='bold')
    # Short dataset and split info
    text = (
        "Dataset: Concrete Compressive Strength (1,030 samples, 8 features)\n\n"
        "Split: 70% train / 15% val / 15% test (random_state=42)\n\n"
        "Baselines: Classification - LogisticRegression, GaussianNB; Regression - LinearRegression, DecisionTreeRegressor\n\n"
        "MLflow experiment: " + EXPERIMENT_NAME + "\n"
    )
    ax.text(0.06, 0.72, text, ha='left', va='top', fontsize=10)
    ax.text(0.06, 0.4, "Required Figures:\n1) Target distribution\n2) Correlation heatmap\n3) Confusion matrix (best clf)\n4) Residuals vs Predicted (best reg)", ha='left', va='top', fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)

    # Page 2: Plots 1 & 2
    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
    img1 = plt.imread(plot1_path)
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[0].set_title('Plot 1: Target distribution')
    img2 = plt.imread(plot2_path)
    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title('Plot 2: Correlation heatmap')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Page 3: Plots 3 & 4
    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
    img3 = plt.imread(plot3_path)
    axes[0].imshow(img3)
    axes[0].axis('off')
    axes[0].set_title('Plot 3: Confusion matrix (best classification)')
    img4 = plt.imread(plot4_path)
    axes[1].imshow(img4)
    axes[1].axis('off')
    axes[1].set_title('Plot 4: Residuals vs Predicted (best regression)')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Page 4: Tables and a short discussion (0.5-1 page)
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis('off')
    ax.text(0.5, 0.95, 'Tables: Metrics for baselines', ha='center', va='center', fontsize=14, weight='bold')
    # place table images
    img_t1 = plt.imread(table1_img)
    img_t2 = plt.imread(table2_img)
    ax.imshow(img_t1, extent=(0.06, 0.94, 0.52, 0.92))
    ax.imshow(img_t2, extent=(0.06, 0.94, 0.06, 0.46))
    # Discussion text
    discussion = (
        "Results and discussion:\n\n"
        "The classification baselines show comparable performance; best classifier by validation F1 is: " + str(best_clf) + "\n\n"
        "The regression baselines report RMSE and MAE; best regression by validation RMSE is: " + str(best_reg) + "\n\n"
        "Failure modes: some large residuals remain (outliers), linear models underfit non-linear patterns. Next steps: train an MLP with early stopping and compare to baselines.\n\n"
        "Neural network plan: MLP with 2-3 dense layers, ReLU activations, dropout, Adam optimizer."
    )
    ax.text(0.06, 0.02, discussion, ha='left', va='bottom', fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF assembled at: {pdf_path}")
print("Plots and reports saved in:")
print(PLOTS_DIR)
print(REPORTS_DIR)

# Save artifact paths to reports directory for instructor convenience
with open(os.path.join(REPORTS_DIR, 'artifact_paths.txt'), 'w') as f:
    f.write(f"pdf: {pdf_path}\n")
    f.write(f"plots dir: {PLOTS_DIR}\n")
    f.write(f"reports dir: {REPORTS_DIR}\n")

print("Midpoint pipeline finished.")
