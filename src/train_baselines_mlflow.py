import os
import tempfile
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Config
EXPERIMENT_NAME = "ConcreteStrength_Baselines"
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "concrete_data.csv")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
PLOTS_DIR = os.path.abspath(PLOTS_DIR)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load data
print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df.columns = [
    'cement', 'slag', 'fly_ash', 'water', 'superplasticizer',
    'coarse_aggregate', 'fine_aggregate', 'age', 'strength'
]

# Create classification target using median split (consistent with earlier scripts)
median_strength = df['strength'].median()
df['strength_class'] = (df['strength'] > median_strength).astype(int)

# Prepare features
X = df.drop(['strength', 'strength_class'], axis=1)

# Set MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)

# Utility to save and log a confusion matrix
def save_and_log_confusion_matrix(y_true, y_pred, run_dir, name="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    out_path = os.path.join(run_dir, name)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    mlflow.log_artifact(out_path, artifact_path="plots")

# Utility to save and log residuals plot
def save_and_log_residuals(y_true, y_pred, run_dir, name="residuals.png"):
    fig, ax = plt.subplots(figsize=(6,5))
    sns = __import__('seaborn')
    sns.scatterplot(x=y_pred, y=y_true - y_pred, color='teal', alpha=0.6, ax=ax)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Predicted Strength')
    ax.set_ylabel('Residuals (Actual - Predicted)')
    ax.set_title('Residuals vs Predicted')
    out_path = os.path.join(run_dir, name)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    mlflow.log_artifact(out_path, artifact_path="plots")

# -------------------------
# Classification baselines
# -------------------------
X_clf = X.copy()
y_clf = df['strength_class']
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

classification_models = [
    ("logreg_clf", LogisticRegression, {"max_iter":500, "random_state":42}),
    ("rf_clf", RandomForestClassifier, {"n_estimators":100, "random_state":42}),
]

for run_name, ModelClass, params in classification_models:
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", ModelClass.__name__)
        for k, v in params.items():
            mlflow.log_param(k, v)

        model = ModelClass(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)

        mlflow.log_metric("test_accuracy", float(acc))
        mlflow.log_metric("test_f1", float(f1))
        mlflow.log_metric("test_precision", float(prec))
        mlflow.log_metric("test_recall", float(rec))

        # Save classification report as an artifact
        report_text = (
            f"accuracy: {acc:.4f}\n"
            f"f1: {f1:.4f}\n"
            f"precision: {prec:.4f}\n"
            f"recall: {rec:.4f}\n"
        )
        run_dir = tempfile.mkdtemp()
        report_path = os.path.join(run_dir, "classification_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        mlflow.log_artifact(report_path, artifact_path="reports")

        # Save confusion matrix and log as artifact
        try:
            save_and_log_confusion_matrix(y_test, preds, run_dir, name="confusion_matrix.png")
        except Exception as e:
            print("Warning: failed to save confusion matrix:", e)

        # Log model
        mlflow.sklearn.log_model(model, "model")

# -------------------------
# Regression baselines
# -------------------------
X_reg = X.copy()
y_reg = df['strength']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

regression_models = [
    ("linreg", LinearRegression, {}),
    ("tree_reg", DecisionTreeRegressor, {"max_depth":5, "random_state":42}),
]

for run_name, ModelClass, params in regression_models:
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", ModelClass.__name__)
        for k, v in params.items():
            mlflow.log_param(k, v)

        model = ModelClass(**params)
        model.fit(X_train_r, y_train_r)
        preds = model.predict(X_test_r)
        mae = mean_absolute_error(y_test_r, preds)
        rmse = np.sqrt(mean_squared_error(y_test_r, preds))
        r2 = r2_score(y_test_r, preds)

        mlflow.log_metric("test_mae", float(mae))
        mlflow.log_metric("test_rmse", float(rmse))
        mlflow.log_metric("test_r2", float(r2))

        # Save regression report
        run_dir = tempfile.mkdtemp()
        report_path = os.path.join(run_dir, "regression_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}\n")
        mlflow.log_artifact(report_path, artifact_path="reports")

        # Save residuals plot and log
        try:
            save_and_log_residuals(y_test_r, preds, run_dir, name="residuals.png")
        except Exception as e:
            print("Warning: failed to save residuals plot:", e)

        # Log model
        mlflow.sklearn.log_model(model, "model")

print("All baseline runs complete. Check the MLflow UI with: mlflow ui")
